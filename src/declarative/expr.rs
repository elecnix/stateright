//! Expressions and effects: compilation from JSON and evaluation.

use super::value::{Hashed, Show, Sym, Symbols, Value};
use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

/// A compiled expression. Names are resolved at compile time: a state field
/// to its slot in the state, a local name to its slot in the current frame.
#[derive(Debug)]
pub(crate) enum Expr {
    Lit(Value),
    Field(usize),
    Local(usize),
    List(Vec<Expr>),
    Rec(Vec<(Sym, Expr)>),
    Get(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    /// `get` chains on a state field or a local, read by reference so the
    /// lists and records along the way are not copied.
    Path(Root, Vec<Expr>, Hints),
    Len(Box<Expr>),
    If(Vec<Expr>),
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Not(Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    Cmp(Ordering, bool, Box<Expr>, Box<Expr>),
    Arith(Arith, Vec<Expr>),
    In(Box<Expr>, Box<Expr>),
    IndexOf(Box<Expr>, Box<Expr>),
    /// Membership in a literal list, read without copying the list.
    InLit(Box<Expr>, Vec<Value>),
    Range(Box<Expr>, Box<Expr>),
    Slice(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    Concat(Vec<Expr>),
    SetAt(Box<Expr>, Box<Expr>, Box<Expr>),
    With(Box<Expr>, Box<Expr>, Box<Expr>),
    SetOp(SetOp, Box<Expr>, Box<Expr>),
    Subset(Box<Expr>, Box<Expr>),
    Iter {
        op: IterOp,
        list: Box<Expr>,
        slot: usize,
        /// Whether the body can read the index, in the slot after `slot`.
        indexed: bool,
        body: Box<Expr>,
    },
    Let(Vec<(usize, Expr)>, Box<Expr>),
    Call(usize, Vec<Expr>),
}

/// What compilation and earlier evaluations learned about a [`Expr::Path`].
#[derive(Debug, Default)]
pub(crate) struct Hints {
    /// Whether every key is a literal or a local, so the path can be read
    /// in place. See [`simple`].
    simple: bool,
    /// For each key, the position where a record last had it. Records built
    /// by the same expression have the same fields, so the position is
    /// usually right and saves the search. Threads may race to store it,
    /// and a stale position only costs the search.
    found: Vec<AtomicU32>,
}

impl Hints {
    fn of(keys: &[Expr]) -> Hints {
        Hints {
            simple: keys
                .iter()
                .all(|k| matches!(k, Expr::Lit(_) | Expr::Local(_))),
            found: keys.iter().map(|_| AtomicU32::new(0)).collect(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Root {
    Field(usize),
    Local(usize),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Arith {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum SetOp {
    Union,
    Intersect,
    Difference,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum IterOp {
    Map,
    Filter,
    Some,
    All,
    Count,
    Find,
    FindLast,
    FindIndex,
    FindLastIndex,
    SortBy,
}

/// A compiled effect: one step of an action's update.
#[derive(Debug)]
pub(crate) enum Effect {
    Set {
        field: usize,
        path: Vec<Expr>,
        value: Expr,
    },
    Append {
        field: usize,
        path: Vec<Expr>,
        value: Expr,
    },
    Clear {
        field: usize,
        path: Vec<Expr>,
    },
    Remove {
        field: usize,
        path: Vec<Expr>,
        at: Expr,
    },
    If(Expr, Vec<Effect>, Vec<Effect>),
    Let(usize, Expr),
    Require(Expr),
    Violate(u32),
}

/// A compiled `defs` entry.
#[derive(Debug)]
pub(crate) struct Def {
    pub(crate) name: String,
    pub(crate) arity: usize,
    pub(crate) frame: usize,
    pub(crate) body: Expr,
}

/// The values an iteration or an action parameter ranges over. A `range`
/// is iterated without building its list.
pub(crate) enum Domain {
    List(Arc<Hashed<Value>>),
    Range(i64, usize),
}

impl Domain {
    pub(crate) fn len(&self) -> usize {
        match self {
            Domain::List(l) => l.len(),
            Domain::Range(_, n) => *n,
        }
    }

    pub(crate) fn get(&self, i: usize) -> Value {
        match self {
            Domain::List(l) => l[i].clone(),
            Domain::Range(a, _) => Value::Int(a + i as i64),
        }
    }
}

/// An evaluation error: a type error, an index out of range, an overflow.
/// These are errors in the model, not in the system it describes. The
/// message is boxed so a result fits in two machine words.
#[derive(Debug)]
#[allow(clippy::box_collection)] // a thin pointer, where `String` would take three words
pub(crate) struct EvalError(Box<String>);

impl From<String> for EvalError {
    fn from(message: String) -> Self {
        EvalError(Box::new(message))
    }
}

impl From<&str> for EvalError {
    fn from(message: &str) -> Self {
        EvalError(Box::new(message.to_string()))
    }
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<EvalError> for String {
    fn from(error: EvalError) -> Self {
        *error.0
    }
}

pub(crate) type EvalResult<T> = Result<T, EvalError>;

/// The names visible while compiling one frame: state fields, then the
/// locals bound so far.
pub(crate) struct Scope<'a> {
    pub(crate) fields: &'a [String],
    pub(crate) defs: &'a [String],
    pub(crate) transitions: &'a [String],
    pub(crate) symbols: &'a Symbols,
    locals: Vec<(String, usize)>,
    next: usize,
    max: usize,
    /// Every def a frame calls, to reject recursion.
    pub(crate) calls: Vec<usize>,
}

impl<'a> Scope<'a> {
    pub(crate) fn new(
        fields: &'a [String],
        defs: &'a [String],
        transitions: &'a [String],
        symbols: &'a Symbols,
    ) -> Self {
        Scope {
            fields,
            defs,
            transitions,
            symbols,
            locals: Vec::new(),
            next: 0,
            max: 0,
            calls: Vec::new(),
        }
    }

    /// Binds a name to a fresh slot and returns the slot.
    pub(crate) fn bind(&mut self, name: &str) -> usize {
        let slot = self.next;
        self.locals.push((name.to_string(), slot));
        self.next += 1;
        self.max = self.max.max(self.next);
        slot
    }

    /// Reserves a slot no name refers to.
    fn reserve(&mut self) -> usize {
        let slot = self.next;
        self.next += 1;
        self.max = self.max.max(self.next);
        slot
    }

    fn mark(&self) -> (usize, usize) {
        (self.locals.len(), self.next)
    }

    fn release(&mut self, (locals, next): (usize, usize)) {
        self.locals.truncate(locals);
        self.next = next;
    }

    /// The frame size: the most slots in use at once.
    pub(crate) fn frame(&self) -> usize {
        self.max
    }

    fn resolve(&self, name: &str) -> Option<Expr> {
        if let Some((_, slot)) = self.locals.iter().rev().find(|(n, _)| n == name) {
            return Some(Expr::Local(*slot));
        }
        self.fields.iter().position(|f| f == name).map(Expr::Field)
    }
}

type Json = serde_json::Value;

fn args<'j>(op: &str, json: &'j Json, at: &str) -> Result<&'j [Json], String> {
    match json {
        Json::Array(items) => Ok(items),
        _ => Err(format!("{at}: \"{op}\" takes an array of arguments")),
    }
}

fn arity<'j>(op: &str, json: &'j Json, n: usize, at: &str) -> Result<&'j [Json], String> {
    let a = args(op, json, at)?;
    if a.len() != n {
        return Err(format!(
            "{at}: \"{op}\" takes {n} arguments, found {}",
            a.len()
        ));
    }
    Ok(a)
}

fn boxed(scope: &mut Scope, json: &Json, at: &str) -> Result<Box<Expr>, String> {
    Ok(Box::new(compile(scope, json, at)?))
}

fn all(scope: &mut Scope, items: &[Json], at: &str) -> Result<Vec<Expr>, String> {
    items
        .iter()
        .enumerate()
        .map(|(i, j)| compile(scope, j, &format!("{at}[{i}]")))
        .collect()
}

fn binder(json: &Json, at: &str) -> Result<(String, Option<String>), String> {
    match json {
        Json::String(x) => Ok((x.clone(), None)),
        Json::Array(names) if names.len() == 2 => match (&names[0], &names[1]) {
            (Json::String(x), Json::String(i)) => Ok((x.clone(), Some(i.clone()))),
            _ => Err(format!("{at}: a binder is a name or [element, index]")),
        },
        _ => Err(format!("{at}: a binder is a name or [element, index]")),
    }
}

/// Compiles an expression.
pub(crate) fn compile(scope: &mut Scope, json: &Json, at: &str) -> Result<Expr, String> {
    let fields = match json {
        Json::Object(fields) => fields,
        Json::Array(items) => {
            let items = all(scope, items, at)?;
            return Ok(match constant(&items) {
                Some(values) => Expr::Lit(Value::list(values)),
                None => Expr::List(items),
            });
        }
        _ => return Ok(Expr::Lit(scope.symbols.value_of(json)?)),
    };
    if fields.len() != 1 {
        return Err(format!(
            "{at}: an operation is an object with one key; use {{\"obj\": ...}} for a record"
        ));
    }
    let (op, a) = fields.iter().next().unwrap();
    let at = &format!("{at}.{op}");
    let e = match op.as_str() {
        "var" => {
            let name = a
                .as_str()
                .ok_or_else(|| format!("{at}: \"var\" takes a name"))?;
            scope
                .resolve(name)
                .ok_or_else(|| format!("{at}: unknown name {name:?}"))?
        }
        "lit" => Expr::Lit(scope.symbols.value_of(a)?),
        "obj" => {
            let Json::Object(rec) = a else {
                return Err(format!("{at}: \"obj\" takes an object"));
            };
            let mut out = Vec::new();
            for (k, v) in rec {
                let sym = scope.symbols.id(k).unwrap();
                out.push((sym, compile(scope, v, &format!("{at}.{k}"))?));
            }
            out.sort_by_key(|(k, _)| *k);
            if out.iter().all(|(_, e)| matches!(e, Expr::Lit(_))) {
                let fields = out
                    .into_iter()
                    .map(|(k, e)| match e {
                        Expr::Lit(v) => (k, v),
                        _ => unreachable!(),
                    })
                    .collect();
                Expr::Lit(Value::rec(fields))
            } else {
                Expr::Rec(out)
            }
        }
        "get" => {
            let a = args(op, a, at)?;
            if a.len() != 2 && a.len() != 3 {
                return Err(format!(
                    "{at}: \"get\" takes [value, key] or [value, key, default]"
                ));
            }
            let x = compile(scope, &a[0], at)?;
            let k = compile(scope, &a[1], at)?;
            match (x, a.get(2)) {
                (Expr::Field(f), None) => path(Root::Field(f), vec![k]),
                (Expr::Local(l), None) => path(Root::Local(l), vec![k]),
                (Expr::Path(root, mut keys, _), None) => {
                    keys.push(k);
                    path(root, keys)
                }
                (x, default) => Expr::Get(
                    Box::new(x),
                    Box::new(k),
                    default.map(|d| boxed(scope, d, at)).transpose()?,
                ),
            }
        }
        "len" => Expr::Len(boxed(scope, a, at)?),
        "if" => {
            let a = args(op, a, at)?;
            if a.len() < 3 || a.len() % 2 == 0 {
                return Err(format!(
                    "{at}: \"if\" takes [condition, then, else] or more condition-value pairs before else"
                ));
            }
            Expr::If(all(scope, a, at)?)
        }
        "and" => Expr::And(all(scope, args(op, a, at)?, at)?),
        "or" => Expr::Or(all(scope, args(op, a, at)?, at)?),
        "not" => Expr::Not(boxed(scope, a, at)?),
        "==" | "!=" | "<" | "<=" | ">" | ">=" | "in" | "index_of" | "range" | "subset"
        | "union" | "intersect" | "difference" => {
            let a = arity(op, a, 2, at)?;
            let (l, r) = (boxed(scope, &a[0], at)?, boxed(scope, &a[1], at)?);
            match op.as_str() {
                "==" => Expr::Eq(l, r),
                "!=" => Expr::Ne(l, r),
                "<" => Expr::Cmp(Ordering::Less, false, l, r),
                "<=" => Expr::Cmp(Ordering::Greater, true, l, r),
                ">" => Expr::Cmp(Ordering::Greater, false, l, r),
                ">=" => Expr::Cmp(Ordering::Less, true, l, r),
                "index_of" => Expr::IndexOf(l, r),
                "in" => match *r {
                    Expr::Lit(Value::List(items)) => Expr::InLit(l, items.to_vec()),
                    r => Expr::In(l, Box::new(r)),
                },
                "range" => Expr::Range(l, r),
                "subset" => Expr::Subset(l, r),
                "union" => Expr::SetOp(SetOp::Union, l, r),
                "intersect" => Expr::SetOp(SetOp::Intersect, l, r),
                _ => Expr::SetOp(SetOp::Difference, l, r),
            }
        }
        "+" | "*" | "min" | "max" | "-" | "/" | "%" => {
            let items = args(op, a, at)?;
            let (kind, binary) = match op.as_str() {
                "+" => (Arith::Add, false),
                "*" => (Arith::Mul, false),
                "min" => (Arith::Min, false),
                "max" => (Arith::Max, false),
                "-" => (Arith::Sub, true),
                "/" => (Arith::Div, true),
                _ => (Arith::Mod, true),
            };
            if (binary && items.len() != 2) || items.is_empty() {
                return Err(format!("{at}: wrong number of arguments"));
            }
            Expr::Arith(kind, all(scope, items, at)?)
        }
        "slice" => {
            let a = args(op, a, at)?;
            if a.len() != 2 && a.len() != 3 {
                return Err(format!(
                    "{at}: \"slice\" takes [list, from] or [list, from, to]"
                ));
            }
            Expr::Slice(
                boxed(scope, &a[0], at)?,
                boxed(scope, &a[1], at)?,
                a.get(2).map(|d| boxed(scope, d, at)).transpose()?,
            )
        }
        "concat" => Expr::Concat(all(scope, args(op, a, at)?, at)?),
        "set_at" | "with" => {
            let a = arity(op, a, 3, at)?;
            let (x, k, v) = (
                boxed(scope, &a[0], at)?,
                boxed(scope, &a[1], at)?,
                boxed(scope, &a[2], at)?,
            );
            if op == "set_at" {
                Expr::SetAt(x, k, v)
            } else {
                Expr::With(x, k, v)
            }
        }
        "map" | "filter" | "some" | "all" | "count" | "find" | "find_last" | "find_index"
        | "find_last_index" | "sort_by" => {
            let a = arity(op, a, 3, at)?;
            let iter = match op.as_str() {
                "map" => IterOp::Map,
                "filter" => IterOp::Filter,
                "some" => IterOp::Some,
                "all" => IterOp::All,
                "count" => IterOp::Count,
                "find" => IterOp::Find,
                "find_last" => IterOp::FindLast,
                "find_index" => IterOp::FindIndex,
                "find_last_index" => IterOp::FindLastIndex,
                _ => IterOp::SortBy,
            };
            let list = boxed(scope, &a[0], at)?;
            let (x, i) = binder(&a[1], at)?;
            let mark = scope.mark();
            let slot = scope.bind(&x);
            let indexed = i.is_some();
            match i {
                Some(i) => {
                    scope.bind(&i);
                }
                None => {
                    scope.reserve();
                }
            }
            let body = boxed(scope, &a[2], at)?;
            scope.release(mark);
            Expr::Iter {
                op: iter,
                list,
                slot,
                indexed,
                body,
            }
        }
        "let" => {
            let a = arity(op, a, 2, at)?;
            let Json::Array(bindings) = &a[0] else {
                return Err(format!("{at}: \"let\" takes [[[name, value], ...], body]"));
            };
            let mark = scope.mark();
            let mut out = Vec::new();
            for (i, b) in bindings.iter().enumerate() {
                let (name, value) = binding(b, &format!("{at}[{i}]"))?;
                let value = compile(scope, value, &format!("{at}.{name}"))?;
                out.push((scope.bind(name), value));
            }
            let body = boxed(scope, &a[1], at)?;
            scope.release(mark);
            Expr::Let(out, body)
        }
        "call" => {
            let a = args(op, a, at)?;
            let name = a
                .first()
                .and_then(Json::as_str)
                .ok_or_else(|| format!("{at}: \"call\" takes [name, arguments...]"))?;
            let def = scope
                .defs
                .iter()
                .position(|d| d == name)
                .ok_or_else(|| format!("{at}: unknown def {name:?}"))?;
            scope.calls.push(def);
            Expr::Call(def, all(scope, &a[1..], at)?)
        }
        _ => return Err(format!("{at}: unknown operation {op:?}")),
    };
    Ok(e)
}

fn path(root: Root, keys: Vec<Expr>) -> Expr {
    let hints = Hints::of(&keys);
    Expr::Path(root, keys, hints)
}

/// Reads one key of a list or a record by reference. `found` is the
/// position where a record last had the key at this place in the model.
#[inline(always)]
fn step<'v>(
    symbols: &Symbols,
    cur: &'v Value,
    key: &Value,
    found: &AtomicU32,
) -> EvalResult<&'v Value> {
    match (cur, key) {
        (Value::List(l), Value::Int(i)) => {
            let n = l.len() as i64;
            let i = if *i < 0 { n + i } else { *i };
            if (0..n).contains(&i) {
                return Ok(&l[i as usize]);
            }
        }
        (Value::Rec(r), Value::Str(s)) => {
            match r.get(found.load(AtomicOrdering::Relaxed) as usize) {
                Some((k, v)) if k == s => return Ok(v),
                _ => {
                    if let Ok(i) = r.binary_search_by_key(s, |(k, _)| *k) {
                        found.store(i as u32, AtomicOrdering::Relaxed);
                        return Ok(&r[i].1);
                    }
                }
            }
        }
        _ => {}
    }
    Err(no_key(symbols, cur, key))
}

#[cold]
#[inline(never)]
fn no_key(symbols: &Symbols, cur: &Value, key: &Value) -> EvalError {
    EvalError::from(format!(
        "get: no {} in {}",
        Show(symbols, key),
        Show(symbols, cur)
    ))
}

/// Whether two values are equal. Scalars of the same type compare here,
/// without a call.
#[inline(always)]
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Str(a), Value::Str(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        _ => a == b,
    }
}

/// Whether an expression reads a value already stored: a literal, a state
/// field, a local, or a path whose keys are literals or locals.
#[inline]
fn simple(e: &Expr) -> bool {
    match e {
        Expr::Lit(_) | Expr::Field(_) | Expr::Local(_) => true,
        Expr::Path(_, _, hints) => hints.simple,
        _ => false,
    }
}

/// The values of a list of expressions, when every one is a literal.
fn constant(items: &[Expr]) -> Option<Vec<Value>> {
    items
        .iter()
        .map(|e| match e {
            Expr::Lit(v) => Some(v.clone()),
            _ => None,
        })
        .collect()
}

/// Reads a `[name, value]` pair.
pub(crate) fn binding<'j>(json: &'j Json, at: &str) -> Result<(&'j str, &'j Json), String> {
    match json {
        Json::Array(pair) if pair.len() == 2 => match &pair[0] {
            Json::String(name) => Ok((name, &pair[1])),
            _ => Err(format!("{at}: a binding is [name, value]")),
        },
        _ => Err(format!("{at}: a binding is [name, value]")),
    }
}

fn target(scope: &mut Scope, json: &Json, at: &str) -> Result<(usize, Vec<Expr>), String> {
    let (field, rest) = match json {
        Json::String(f) => (f.as_str(), &[][..]),
        Json::Array(items) if !items.is_empty() => match &items[0] {
            Json::String(f) => (f.as_str(), &items[1..]),
            _ => return Err(format!("{at}: a path starts with a state field name")),
        },
        _ => return Err(format!("{at}: a path is a field name or [field, key, ...]")),
    };
    let slot = scope
        .fields
        .iter()
        .position(|f| f == field)
        .ok_or_else(|| format!("{at}: unknown state field {field:?}"))?;
    Ok((slot, all(scope, rest, at)?))
}

fn get<'j>(
    fields: &'j serde_json::Map<String, Json>,
    key: &str,
    at: &str,
) -> Result<&'j Json, String> {
    fields
        .get(key)
        .ok_or_else(|| format!("{at}: missing \"{key}\""))
}

/// Compiles a list of effects. Names an effect binds with `let` stay visible
/// to the effects after it, up to the end of the enclosing list.
pub(crate) fn compile_effects(
    scope: &mut Scope,
    json: &Json,
    at: &str,
) -> Result<Vec<Effect>, String> {
    let Json::Array(items) = json else {
        return Err(format!("{at}: effects are an array"));
    };
    let mark = scope.mark();
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let at = &format!("{at}[{i}]");
        let Json::Object(fields) = item else {
            return Err(format!("{at}: an effect is an object"));
        };
        let effect = if let Some(p) = fields.get("set") {
            let (field, path) = target(scope, p, at)?;
            let value = compile(scope, get(fields, "to", at)?, at)?;
            Effect::Set { field, path, value }
        } else if let Some(p) = fields.get("append") {
            let (field, path) = target(scope, p, at)?;
            let value = compile(scope, get(fields, "value", at)?, at)?;
            Effect::Append { field, path, value }
        } else if let Some(p) = fields.get("clear") {
            let (field, path) = target(scope, p, at)?;
            Effect::Clear { field, path }
        } else if let Some(p) = fields.get("remove") {
            let (field, path) = target(scope, p, at)?;
            let at_index = compile(scope, get(fields, "at", at)?, at)?;
            Effect::Remove {
                field,
                path,
                at: at_index,
            }
        } else if let Some(c) = fields.get("if") {
            let cond = compile(scope, c, at)?;
            let then = compile_effects(scope, get(fields, "then", at)?, &format!("{at}.then"))?;
            let otherwise = match fields.get("else") {
                Some(e) => compile_effects(scope, e, &format!("{at}.else"))?,
                None => Vec::new(),
            };
            Effect::If(cond, then, otherwise)
        } else if let Some(n) = fields.get("let") {
            let name = n
                .as_str()
                .ok_or_else(|| format!("{at}: \"let\" takes a name"))?;
            let value = compile(scope, get(fields, "be", at)?, at)?;
            Effect::Let(scope.bind(name), value)
        } else if let Some(c) = fields.get("require") {
            Effect::Require(compile(scope, c, at)?)
        } else if let Some(n) = fields.get("violate") {
            let name = n
                .as_str()
                .ok_or_else(|| format!("{at}: \"violate\" takes an invariant name"))?;
            let bit = scope
                .transitions
                .iter()
                .position(|t| t == name)
                .ok_or_else(|| {
                    format!("{at}: {name:?} is not declared as a transition invariant")
                })?;
            Effect::Violate(bit as u32)
        } else {
            return Err(format!("{at}: unknown effect"));
        };
        out.push(effect);
    }
    scope.release(mark);
    Ok(out)
}

/// Calls `f` on each direct subexpression of `e`.
fn each_child<'e>(e: &'e Expr, f: &mut impl FnMut(&'e Expr)) {
    match e {
        Expr::Lit(_) | Expr::Field(_) | Expr::Local(_) => {}
        Expr::List(items)
        | Expr::Path(_, items, _)
        | Expr::If(items)
        | Expr::And(items)
        | Expr::Or(items)
        | Expr::Arith(_, items)
        | Expr::Concat(items)
        | Expr::Call(_, items) => items.iter().for_each(f),
        Expr::Rec(fields) => fields.iter().for_each(|(_, v)| f(v)),
        Expr::Len(x) | Expr::Not(x) | Expr::InLit(x, _) => f(x),
        Expr::Eq(a, b)
        | Expr::Ne(a, b)
        | Expr::Cmp(_, _, a, b)
        | Expr::In(a, b)
        | Expr::IndexOf(a, b)
        | Expr::Range(a, b)
        | Expr::SetOp(_, a, b)
        | Expr::Subset(a, b) => {
            f(a);
            f(b);
        }
        Expr::Get(a, b, c) | Expr::Slice(a, b, c) => {
            f(a);
            f(b);
            if let Some(c) = c {
                f(c);
            }
        }
        Expr::SetAt(a, b, c) | Expr::With(a, b, c) => {
            f(a);
            f(b);
            f(c);
        }
        Expr::Iter { list, body, .. } => {
            f(list);
            f(body);
        }
        Expr::Let(bindings, body) => {
            bindings.iter().for_each(|(_, v)| f(v));
            f(body);
        }
    }
}

/// The state fields a piece of a model reads and the ones it changes, as
/// one flag per field. A piece that reads only some fields gives the same
/// result in every state that agrees on them.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Access {
    pub(crate) reads: Vec<bool>,
    pub(crate) writes: Vec<bool>,
}

impl Access {
    pub(crate) fn none(fields: usize) -> Access {
        Access {
            reads: vec![false; fields],
            writes: vec![false; fields],
        }
    }

    pub(crate) fn merge(&mut self, other: &Access) {
        for (mine, theirs) in self.reads.iter_mut().zip(&other.reads) {
            *mine |= theirs;
        }
        for (mine, theirs) in self.writes.iter_mut().zip(&other.writes) {
            *mine |= theirs;
        }
    }

    /// Marks the fields `e` reads. `def_reads` gives the fields each def
    /// reads, the defs it calls included.
    pub(crate) fn expr(&mut self, e: &Expr, def_reads: &[Vec<bool>]) {
        match e {
            Expr::Field(f) | Expr::Path(Root::Field(f), ..) => self.reads[*f] = true,
            Expr::Call(def, _) => {
                for (mine, theirs) in self.reads.iter_mut().zip(&def_reads[*def]) {
                    *mine |= theirs;
                }
            }
            _ => {}
        }
        each_child(e, &mut |child| self.expr(child, def_reads));
    }

    /// Marks the fields `effects` read and change. A field an effect may
    /// change counts as read too: part of it can stay as it was, and so can
    /// all of it when the effect is in a branch not taken.
    pub(crate) fn effects(&mut self, effects: &[Effect], def_reads: &[Vec<bool>]) {
        for effect in effects {
            match effect {
                Effect::Set { field, path, value } | Effect::Append { field, path, value } => {
                    self.change(*field, path, def_reads);
                    self.expr(value, def_reads);
                }
                Effect::Clear { field, path } => self.change(*field, path, def_reads),
                Effect::Remove { field, path, at } => {
                    self.change(*field, path, def_reads);
                    self.expr(at, def_reads);
                }
                Effect::If(cond, then, otherwise) => {
                    self.expr(cond, def_reads);
                    self.effects(then, def_reads);
                    self.effects(otherwise, def_reads);
                }
                Effect::Let(_, value) => self.expr(value, def_reads),
                Effect::Require(cond) => self.expr(cond, def_reads),
                Effect::Violate(_) => {}
            }
        }
    }

    fn change(&mut self, field: usize, path: &[Expr], def_reads: &[Vec<bool>]) {
        self.reads[field] = true;
        self.writes[field] = true;
        path.iter().for_each(|k| self.expr(k, def_reads));
    }
}

/// The fields each def reads, the defs it calls included. Defs do not
/// recurse, so repeating the pass once per def follows every chain of calls.
pub(crate) fn def_reads(defs: &[Def], fields: usize) -> Vec<Vec<bool>> {
    let mut reads = vec![vec![false; fields]; defs.len()];
    for _ in 0..defs.len() {
        for (i, def) in defs.iter().enumerate() {
            let mut access = Access::none(fields);
            access.expr(&def.body, &reads);
            reads[i] = access.reads;
        }
    }
    reads
}

/// Evaluation context: the state the expressions read, the defs, and a stack
/// of local slots.
pub(crate) struct Ctx<'a> {
    pub(crate) symbols: &'a Symbols,
    pub(crate) defs: &'a [Def],
    pub(crate) stack: Vec<Value>,
    /// Values of zero-argument defs for the one state this context reads.
    /// Empty when the context evaluates against a state that effects change.
    pub(crate) memo: Vec<Option<Value>>,
    /// The keys of the path the current effect changes.
    keys: Vec<Value>,
}

/// The allocations of a [`Ctx`], kept between evaluations so that each one
/// does not allocate its own.
#[derive(Default)]
pub(crate) struct Buffers {
    stack: Vec<Value>,
    memo: Vec<Option<Value>>,
    keys: Vec<Value>,
}

#[cold]
#[inline(never)]
fn type_error<T>(ctx: &Ctx, what: &str, v: &Value) -> EvalResult<T> {
    Err(EvalError::from(format!(
        "{what} expects a different type, found {} {}",
        v.type_name(),
        Show(ctx.symbols, v)
    )))
}

#[cold]
#[inline(never)]
fn unordered(symbols: &Symbols, a: &Value, b: &Value) -> EvalError {
    EvalError::from(format!(
        "cannot order {} and {}",
        Show(symbols, a),
        Show(symbols, b)
    ))
}

fn index(ctx: &Ctx, len: usize, i: &Value, what: &str) -> EvalResult<Option<usize>> {
    let Some(i) = i.as_int() else {
        return type_error(ctx, what, i);
    };
    let i = if i < 0 { len as i64 + i } else { i };
    Ok(if i >= 0 && (i as usize) < len {
        Some(i as usize)
    } else {
        None
    })
}

fn floor_div(a: i64, b: i64) -> EvalResult<(i64, i64)> {
    if b == 0 {
        return Err(EvalError::from("division by zero"));
    }
    let q = a.checked_div(b).ok_or("integer overflow")?;
    let r = a - q * b;
    if r != 0 && ((r < 0) != (b < 0)) {
        Ok((q - 1, r + b))
    } else {
        Ok((q, r))
    }
}

fn as_set(v: &Value) -> Vec<Value> {
    let mut items = v.as_list().map(|l| l.to_vec()).unwrap_or_default();
    items.sort();
    items.dedup();
    items
}

impl<'a> Ctx<'a> {
    /// A context with `frame` local slots, and with one memo slot per def
    /// when `memoize` is set. Set it only to evaluate against one state
    /// that does not change.
    pub(crate) fn new(
        symbols: &'a Symbols,
        defs: &'a [Def],
        frame: usize,
        memoize: bool,
        buffers: Buffers,
    ) -> Self {
        let Buffers {
            mut stack,
            mut memo,
            mut keys,
        } = buffers;
        stack.clear();
        stack.resize(frame, Value::Null);
        memo.clear();
        if memoize {
            memo.resize(defs.len().max(1), None);
        }
        keys.clear();
        Ctx {
            symbols,
            defs,
            stack,
            memo,
            keys,
        }
    }

    /// Gives the allocations back for the next context.
    pub(crate) fn into_buffers(self) -> Buffers {
        Buffers {
            stack: self.stack,
            memo: self.memo,
            keys: self.keys,
        }
    }

    /// Evaluates an expression. A value already stored is copied here, and
    /// every operation goes through [`Ctx::compute`].
    #[inline(always)]
    pub(crate) fn eval(&mut self, e: &Expr, state: &[Value], base: usize) -> EvalResult<Value> {
        if simple(e) {
            Ok(self.peek(e, state, base)?.clone())
        } else {
            self.compute(e, state, base)
        }
    }

    #[inline(never)]
    fn compute(&mut self, e: &Expr, state: &[Value], base: usize) -> EvalResult<Value> {
        Ok(match e {
            Expr::Lit(v) => v.clone(),
            Expr::Field(i) => state[*i].clone(),
            Expr::Local(i) => self.stack[base + i].clone(),
            Expr::List(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.eval(item, state, base)?);
                }
                Value::list(out)
            }
            Expr::Rec(fields) => {
                let mut out = Vec::with_capacity(fields.len());
                for (k, v) in fields {
                    out.push((*k, self.eval(v, state, base)?));
                }
                Value::rec(out)
            }
            Expr::Get(x, k, default) => {
                let x = self.eval(x, state, base)?;
                let k = self.eval(k, state, base)?;
                let found = match (&x, &k) {
                    (Value::List(l), _) => index(self, l.len(), &k, "get")?.map(|i| l[i].clone()),
                    (Value::Rec(r), Value::Str(s)) => r
                        .binary_search_by_key(s, |(k, _)| *k)
                        .ok()
                        .map(|i| r[i].1.clone()),
                    _ => return type_error(self, "get", &x),
                };
                match (found, default) {
                    (Some(v), _) => v,
                    (None, Some(d)) => self.eval(d, state, base)?,
                    (None, None) => {
                        return Err(EvalError::from(format!(
                            "get: no {} in {}",
                            Show(self.symbols, &k),
                            Show(self.symbols, &x)
                        )));
                    }
                }
            }
            Expr::Path(root, keys, hints) => {
                if hints.simple {
                    self.peek(e, state, base)?.clone()
                } else {
                    let mut evaluated = Vec::with_capacity(keys.len());
                    for k in keys {
                        evaluated.push(self.eval(k, state, base)?);
                    }
                    let mut cur = match root {
                        Root::Field(f) => &state[*f],
                        Root::Local(l) => &self.stack[base + l],
                    };
                    for (key, found) in evaluated.iter().zip(&hints.found) {
                        cur = step(self.symbols, cur, key, found)?;
                    }
                    cur.clone()
                }
            }
            Expr::Len(x) => {
                let held;
                let v = if simple(x) {
                    self.peek(x, state, base)?
                } else {
                    held = self.eval(x, state, base)?;
                    &held
                };
                match v {
                    Value::List(l) => Value::Int(l.len() as i64),
                    Value::Rec(r) => Value::Int(r.len() as i64),
                    v => return type_error(self, "len", v),
                }
            }
            Expr::If(items) => {
                let mut i = 0;
                while i + 1 < items.len() {
                    if self.test(&items[i], state, base, "if")? {
                        return self.eval(&items[i + 1], state, base);
                    }
                    i += 2;
                }
                self.eval(&items[items.len() - 1], state, base)?
            }
            Expr::And(_)
            | Expr::Or(_)
            | Expr::Not(_)
            | Expr::Eq(..)
            | Expr::Ne(..)
            | Expr::Cmp(..)
            | Expr::InLit(..) => Value::Bool(self.truth(e, state, base, "a condition")?),
            Expr::Arith(op, items) => Value::Int(self.arith(*op, items, state, base)?),
            Expr::IndexOf(l, x) => self.pair(x, l, state, base, |ctx, x, l| match l {
                Value::List(l) => Ok(match l.iter().position(|v| same(v, x)) {
                    Some(i) => Value::Int(i as i64),
                    None => Value::Null,
                }),
                v => type_error(ctx, "index_of", v),
            })?,
            Expr::In(..) => Value::Bool(self.truth(e, state, base, "in")?),
            Expr::Range(a, b) => {
                let (a, b) = (
                    self.int(a, state, base, "range")?,
                    self.int(b, state, base, "range")?,
                );
                Value::list((a..b).map(Value::Int).collect())
            }
            Expr::Slice(l, from, to) => {
                let list = self.eval(l, state, base)?;
                let Value::List(items) = &list else {
                    return type_error(self, "slice", &list);
                };
                let n = items.len() as i64;
                let clamp = |i: i64| (if i < 0 { n + i } else { i }).clamp(0, n) as usize;
                let from = clamp(self.int(from, state, base, "slice")?);
                let to = match to {
                    Some(t) => clamp(self.int(t, state, base, "slice")?),
                    None => n as usize,
                };
                Value::list(if from < to {
                    items[from..to].to_vec()
                } else {
                    Vec::new()
                })
            }
            Expr::Concat(items) => {
                let mut out = Vec::new();
                for item in items {
                    match self.eval(item, state, base)? {
                        Value::List(l) => out.extend(l.iter().cloned()),
                        v => return type_error(self, "concat", &v),
                    }
                }
                Value::list(out)
            }
            Expr::SetAt(l, i, v) => {
                let list = self.eval(l, state, base)?;
                let i = self.eval(i, state, base)?;
                let v = self.eval(v, state, base)?;
                let Value::List(mut items) = list else {
                    return type_error(self, "set_at", &list);
                };
                let Some(i) = index(self, items.len(), &i, "set_at")? else {
                    return Err(EvalError::from(format!(
                        "set_at: index {} out of range",
                        Show(self.symbols, &i)
                    )));
                };
                Hashed::make_mut(&mut items)[i] = v;
                Value::List(items)
            }
            Expr::With(r, k, v) => {
                let rec = self.eval(r, state, base)?;
                let k = self.eval(k, state, base)?;
                let v = self.eval(v, state, base)?;
                let (Value::Rec(mut fields), Value::Str(k)) = (rec.clone(), &k) else {
                    return type_error(self, "with", &rec);
                };
                let fields_mut = Hashed::make_mut(&mut fields);
                match fields_mut.binary_search_by_key(k, |(k, _)| *k) {
                    Ok(i) => fields_mut[i].1 = v,
                    Err(i) => fields_mut.insert(i, (*k, v)),
                }
                Value::Rec(fields)
            }
            Expr::SetOp(op, a, b) => {
                let (a, b) = (self.eval(a, state, base)?, self.eval(b, state, base)?);
                for v in [&a, &b] {
                    if v.as_list().is_none() {
                        return type_error(self, "a set operation", v);
                    }
                }
                let (a, b) = (as_set(&a), as_set(&b));
                let out = match op {
                    SetOp::Union => {
                        let mut out = a;
                        out.extend(b);
                        out.sort();
                        out.dedup();
                        out
                    }
                    SetOp::Intersect => a
                        .into_iter()
                        .filter(|x| b.binary_search(x).is_ok())
                        .collect(),
                    SetOp::Difference => a
                        .into_iter()
                        .filter(|x| b.binary_search(x).is_err())
                        .collect(),
                };
                Value::list(out)
            }
            Expr::Subset(a, b) => {
                let (a, b) = (self.eval(a, state, base)?, self.eval(b, state, base)?);
                match (&a, &b) {
                    (Value::List(a), Value::List(b)) => {
                        Value::Bool(a.iter().all(|x| b.contains(x)))
                    }
                    _ => return type_error(self, "subset", &a),
                }
            }
            Expr::Iter {
                op,
                list,
                slot,
                indexed,
                body,
            } => self.iter(*op, list, *slot, *indexed, body, state, base)?,
            Expr::Let(bindings, body) => {
                for (slot, value) in bindings {
                    let v = self.eval(value, state, base)?;
                    self.stack[base + slot] = v;
                }
                self.eval(body, state, base)?
            }
            Expr::Call(def, args) if args.is_empty() && !self.memo.is_empty() => {
                if let Some(v) = &self.memo[*def] {
                    return Ok(v.clone());
                }
                let v = self.call(*def, args, state, base, |ctx, body, base| {
                    ctx.eval(body, state, base)
                })?;
                self.memo[*def] = Some(v.clone());
                v
            }
            Expr::Call(def, args) => self.call(*def, args, state, base, |ctx, body, base| {
                ctx.eval(body, state, base)
            })?,
        })
    }

    fn arith(
        &mut self,
        op: Arith,
        items: &[Expr],
        state: &[Value],
        base: usize,
    ) -> EvalResult<i64> {
        let mut acc = self.int(&items[0], state, base, "arithmetic")?;
        for item in &items[1..] {
            let x = self.int(item, state, base, "arithmetic")?;
            acc = match op {
                Arith::Add => acc.checked_add(x).ok_or("integer overflow")?,
                Arith::Sub => acc.checked_sub(x).ok_or("integer overflow")?,
                Arith::Mul => acc.checked_mul(x).ok_or("integer overflow")?,
                Arith::Div => floor_div(acc, x)?.0,
                Arith::Mod => floor_div(acc, x)?.1,
                Arith::Min => acc.min(x),
                Arith::Max => acc.max(x),
            };
        }
        Ok(acc)
    }

    /// Calls a def: evaluates the arguments into a new frame, then hands
    /// the def's body and the frame's base to `evaluate`.
    fn call<R>(
        &mut self,
        def: usize,
        args: &[Expr],
        state: &[Value],
        base: usize,
        evaluate: impl FnOnce(&mut Self, &Expr, usize) -> EvalResult<R>,
    ) -> EvalResult<R> {
        let def_ref = &self.defs[def];
        if args.len() != def_ref.arity {
            return Err(EvalError::from(format!(
                "call {}: takes {} arguments, found {}",
                def_ref.name,
                def_ref.arity,
                args.len()
            )));
        }
        let frame = def_ref.frame;
        let new_base = self.stack.len();
        for a in args {
            match self.eval(a, state, base) {
                Ok(v) => self.stack.push(v),
                Err(e) => {
                    self.stack.truncate(new_base);
                    return Err(e);
                }
            }
        }
        self.stack.resize(new_base + frame, Value::Null);
        let defs = self.defs;
        let result = evaluate(self, &defs[def].body, new_base);
        self.stack.truncate(new_base);
        result
    }

    /// Evaluates the list an iteration or a parameter ranges over.
    pub(crate) fn domain(&mut self, e: &Expr, state: &[Value], base: usize) -> EvalResult<Domain> {
        Ok(match e {
            Expr::Range(a, b) => {
                let a = self.int(a, state, base, "range")?;
                let b = self.int(b, state, base, "range")?;
                Domain::Range(a, (b - a).max(0) as usize)
            }
            _ => match self.eval(e, state, base)? {
                Value::List(items) => Domain::List(items),
                other => return type_error(self, "an iteration", &other),
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn iter(
        &mut self,
        op: IterOp,
        list: &Expr,
        slot: usize,
        indexed: bool,
        body: &Expr,
        state: &[Value],
        base: usize,
    ) -> EvalResult<Value> {
        let items = self.domain(list, state, base)?;
        let n = items.len();
        let item = |i: usize| items.get(i);
        let backwards = matches!(op, IterOp::FindLast | IterOp::FindLastIndex);
        // A list result takes one allocation of the largest size it can
        // have, where growing it item by item would take several.
        let mut mapped = Vec::new();
        let mut keyed = Vec::new();
        match op {
            IterOp::Map | IterOp::Filter => mapped.reserve_exact(n),
            IterOp::SortBy => keyed.reserve_exact(n),
            _ => {}
        }
        let mut count = 0;
        for step in 0..n {
            let i = if backwards { n - 1 - step } else { step };
            self.stack[base + slot] = item(i);
            if indexed {
                self.stack[base + slot + 1] = Value::Int(i as i64);
            }
            match op {
                IterOp::Map => mapped.push(self.eval(body, state, base)?),
                IterOp::SortBy => keyed.push((self.eval(body, state, base)?, i)),
                _ => {
                    let hit = self.test(body, state, base, "an iteration")?;
                    match op {
                        IterOp::Filter if hit => mapped.push(item(i)),
                        IterOp::Count if hit => count += 1,
                        IterOp::Some if hit => return Ok(Value::Bool(true)),
                        IterOp::All if !hit => return Ok(Value::Bool(false)),
                        IterOp::Find | IterOp::FindLast if hit => return Ok(item(i)),
                        IterOp::FindIndex | IterOp::FindLastIndex if hit => {
                            return Ok(Value::Int(i as i64));
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(match op {
            // A filter that keeps everything, or a list already in order,
            // returns the list it was given rather than a copy.
            IterOp::Filter if mapped.len() == n => match &items {
                Domain::List(l) => Value::List(Arc::clone(l)),
                Domain::Range(..) => Value::list(mapped),
            },
            IterOp::Map | IterOp::Filter => Value::list(mapped),
            IterOp::SortBy if keyed.windows(2).all(|w| w[0].0 <= w[1].0) => match &items {
                Domain::List(l) => Value::List(Arc::clone(l)),
                Domain::Range(..) => Value::list((0..n).map(item).collect()),
            },
            IterOp::SortBy => {
                keyed.sort_by(|a, b| a.0.cmp(&b.0));
                Value::list(keyed.into_iter().map(|(_, i)| item(i)).collect())
            }
            IterOp::Count => Value::Int(count),
            IterOp::Some => Value::Bool(false),
            IterOp::All => Value::Bool(true),
            _ => Value::Null,
        })
    }

    /// Evaluates a condition. Boolean operations, comparisons and literal
    /// membership answer here directly, reading simple operands in place.
    /// Evaluates a condition. The commonest one, a comparison of two
    /// stored values for equality, answers here without a call. Every
    /// other condition goes through [`Ctx::truth`].
    #[inline(always)]
    pub(crate) fn test(
        &mut self,
        e: &Expr,
        state: &[Value],
        base: usize,
        what: &str,
    ) -> EvalResult<bool> {
        match e {
            Expr::Eq(l, r) if simple(l) && simple(r) => Ok(same(
                self.peek(l, state, base)?,
                self.peek(r, state, base)?,
            )),
            Expr::Ne(l, r) if simple(l) && simple(r) => Ok(!same(
                self.peek(l, state, base)?,
                self.peek(r, state, base)?,
            )),
            Expr::Cmp(ord, negate, l, r) if simple(l) && simple(r) => {
                match (self.peek(l, state, base)?, self.peek(r, state, base)?) {
                    (Value::Int(a), Value::Int(b)) => Ok((a.cmp(b) == *ord) != *negate),
                    _ => self.truth(e, state, base, what),
                }
            }
            _ => self.truth(e, state, base, what),
        }
    }

    pub(crate) fn truth(
        &mut self,
        e: &Expr,
        state: &[Value],
        base: usize,
        what: &str,
    ) -> EvalResult<bool> {
        match e {
            Expr::And(items) => {
                for item in items {
                    if !self.test(item, state, base, "and")? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Expr::Or(items) => {
                for item in items {
                    if self.test(item, state, base, "or")? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Expr::Not(x) => Ok(!self.test(x, state, base, "not")?),
            Expr::Eq(l, r) => self.pair(l, r, state, base, |_, a, b| Ok(same(a, b))),
            Expr::Ne(l, r) => self.pair(l, r, state, base, |_, a, b| Ok(!same(a, b))),
            Expr::In(x, l) => self.pair(x, l, state, base, |ctx, x, l| match l {
                Value::List(l) => Ok(l.iter().any(|v| same(v, x))),
                v => type_error(ctx, "in", v),
            }),
            Expr::Cmp(ord, negate, l, r) => {
                let symbols = self.symbols;
                self.pair(l, r, state, base, |_, a, b| {
                    let order = match (a, b) {
                        (Value::Int(a), Value::Int(b)) => a.cmp(b),
                        _ if std::mem::discriminant(a) != std::mem::discriminant(b) => {
                            return Err(unordered(symbols, a, b));
                        }
                        _ => a.cmp(b),
                    };
                    Ok((order == *ord) != *negate)
                })
            }
            Expr::InLit(x, items) => {
                if simple(x) {
                    Ok(items.contains(self.peek(x, state, base)?))
                } else {
                    let x = self.eval(x, state, base)?;
                    Ok(items.contains(&x))
                }
            }
            Expr::Lit(Value::Bool(b)) => Ok(*b),
            // A condition inside an `if`, a `let` or a def answers as a
            // condition all the way down, without a value in between.
            Expr::If(items) => {
                let mut i = 0;
                while i + 1 < items.len() {
                    if self.test(&items[i], state, base, "if")? {
                        return self.test(&items[i + 1], state, base, what);
                    }
                    i += 2;
                }
                self.test(&items[items.len() - 1], state, base, what)
            }
            Expr::Let(bindings, body) => {
                for (slot, value) in bindings {
                    let v = self.eval(value, state, base)?;
                    self.stack[base + slot] = v;
                }
                self.test(body, state, base, what)
            }
            Expr::Call(def, args) if !args.is_empty() || self.memo.is_empty() => {
                self.call(*def, args, state, base, |ctx, body, base| {
                    ctx.test(body, state, base, what)
                })
            }
            _ => {
                let v = self.eval(e, state, base)?;
                match v.as_bool() {
                    Some(b) => Ok(b),
                    None => type_error(self, what, &v),
                }
            }
        }
    }

    /// Applies `f` to two operands, reading simple ones in place.
    #[inline(always)]
    fn pair<R>(
        &mut self,
        l: &Expr,
        r: &Expr,
        state: &[Value],
        base: usize,
        f: impl FnOnce(&Self, &Value, &Value) -> EvalResult<R>,
    ) -> EvalResult<R> {
        if simple(l) && simple(r) {
            return f(
                self,
                self.peek(l, state, base)?,
                self.peek(r, state, base)?,
            );
        }
        let lo = if simple(l) {
            None
        } else {
            Some(self.compute(l, state, base)?)
        };
        let ro = if simple(r) {
            None
        } else {
            Some(self.compute(r, state, base)?)
        };
        let lv = match &lo {
            Some(v) => v,
            None => self.peek(l, state, base)?,
        };
        let rv = match &ro {
            Some(v) => v,
            None => self.peek(r, state, base)?,
        };
        f(self, lv, rv)
    }

    /// Reads a [`simple`] expression in place.
    #[inline(always)]
    fn peek<'s>(&'s self, e: &'s Expr, state: &'s [Value], base: usize) -> EvalResult<&'s Value> {
        match e {
            Expr::Lit(v) => Ok(v),
            Expr::Field(f) => Ok(&state[*f]),
            Expr::Local(l) => Ok(&self.stack[base + l]),
            Expr::Path(root, keys, hints) => self.follow(*root, keys, hints, state, base),
            _ => unreachable!("peek on an expression that is not simple"),
        }
    }

    /// Reads a path whose keys are literals or locals.
    fn follow<'s>(
        &'s self,
        root: Root,
        keys: &'s [Expr],
        hints: &'s Hints,
        state: &'s [Value],
        base: usize,
    ) -> EvalResult<&'s Value> {
        let mut cur = match root {
            Root::Field(f) => &state[f],
            Root::Local(l) => &self.stack[base + l],
        };
        for (k, found) in keys.iter().zip(&hints.found) {
            let key = match k {
                Expr::Lit(v) => v,
                Expr::Local(l) => &self.stack[base + l],
                _ => unreachable!("a simple path with a computed key"),
            };
            cur = step(self.symbols, cur, key, found)?;
        }
        Ok(cur)
    }

    /// Evaluates an integer. Stored values, lengths and arithmetic answer
    /// here, without a value in between.
    fn int(&mut self, e: &Expr, state: &[Value], base: usize, what: &str) -> EvalResult<i64> {
        let held;
        let v = match e {
            Expr::Arith(op, items) => return self.arith(*op, items, state, base),
            Expr::Len(x) if simple(x) => {
                return match self.peek(x, state, base)? {
                    Value::List(l) => Ok(l.len() as i64),
                    Value::Rec(r) => Ok(r.len() as i64),
                    v => type_error(self, "len", v),
                };
            }
            _ if simple(e) => self.peek(e, state, base)?,
            _ => {
                held = self.compute(e, state, base)?;
                &held
            }
        };
        match v.as_int() {
            Some(i) => Ok(i),
            None => type_error(self, what, v),
        }
    }

    /// Applies effects to `state` in order. Returns `false` when a `require`
    /// fails, which disables the action. Sets bits in `flags` for each
    /// transition invariant an effect reports violated.
    pub(crate) fn apply(
        &mut self,
        effects: &[Effect],
        state: &mut [Value],
        flags: &mut u64,
        base: usize,
    ) -> EvalResult<bool> {
        for effect in effects {
            match effect {
                Effect::Set { field, path, value } => {
                    let keys = self.keys(path, state, base)?;
                    let v = self.eval(value, state, base)?;
                    self.update(&mut state[*field], &keys, |_, slot| {
                        *slot = v;
                        Ok(())
                    })?;
                    self.keys = keys;
                }
                Effect::Append { field, path, value } => {
                    let keys = self.keys(path, state, base)?;
                    let v = self.eval(value, state, base)?;
                    self.update(&mut state[*field], &keys, |ctx, slot| match slot {
                        Value::List(l) => {
                            Hashed::make_mut(l).push(v);
                            Ok(())
                        }
                        other => type_error(ctx, "append", other),
                    })?;
                    self.keys = keys;
                }
                Effect::Clear { field, path } => {
                    let keys = self.keys(path, state, base)?;
                    self.update(&mut state[*field], &keys, |_, slot| {
                        *slot = match slot {
                            Value::List(_) => Value::list(Vec::new()),
                            _ => Value::Null,
                        };
                        Ok(())
                    })?;
                    self.keys = keys;
                }
                Effect::Remove { field, path, at } => {
                    let keys = self.keys(path, state, base)?;
                    let at = self.eval(at, state, base)?;
                    self.update(&mut state[*field], &keys, |ctx, slot| match slot {
                        Value::List(l) => match index(ctx, l.len(), &at, "remove")? {
                            Some(i) => {
                                Hashed::make_mut(l).remove(i);
                                Ok(())
                            }
                            None => Err(EvalError::from(format!(
                                "remove: index {} out of range",
                                Show(ctx.symbols, &at)
                            ))),
                        },
                        other => type_error(ctx, "remove", other),
                    })?;
                    self.keys = keys;
                }
                Effect::If(cond, then, otherwise) => {
                    let branch = if self.test(cond, state, base, "if")? {
                        then
                    } else {
                        otherwise
                    };
                    if !self.apply(branch, state, flags, base)? {
                        return Ok(false);
                    }
                }
                Effect::Let(slot, value) => {
                    let v = self.eval(value, state, base)?;
                    self.stack[base + slot] = v;
                }
                Effect::Require(cond) => {
                    if !self.truth(cond, state, base, "require")? {
                        return Ok(false);
                    }
                }
                Effect::Violate(bit) => *flags |= 1 << bit,
            }
        }
        Ok(true)
    }

    /// Evaluates the keys of an effect's path into the context's buffer,
    /// which the caller gives back once the effect is applied.
    fn keys(&mut self, path: &[Expr], state: &[Value], base: usize) -> EvalResult<Vec<Value>> {
        let mut keys = std::mem::take(&mut self.keys);
        keys.clear();
        for k in path {
            keys.push(self.eval(k, state, base)?);
        }
        Ok(keys)
    }

    /// Calls `f` on the value at `keys` below `root`, copying shared lists
    /// and records on the way down.
    fn update(
        &self,
        root: &mut Value,
        keys: &[Value],
        f: impl FnOnce(&Self, &mut Value) -> EvalResult<()>,
    ) -> EvalResult<()> {
        let mut slot = root;
        for k in keys {
            slot = match (slot, k) {
                (Value::List(l), _) => {
                    let Some(i) = index(self, l.len(), k, "a path")? else {
                        return Err(EvalError::from(format!(
                            "path index {} out of range",
                            Show(self.symbols, k)
                        )));
                    };
                    &mut Hashed::make_mut(l)[i]
                }
                (Value::Rec(r), Value::Str(s)) => {
                    let r = Hashed::make_mut(r);
                    let i = match r.binary_search_by_key(s, |(k, _)| *k) {
                        Ok(i) => i,
                        Err(i) => {
                            r.insert(i, (*s, Value::Null));
                            i
                        }
                    };
                    &mut r[i].1
                }
                (other, _) => return type_error(self, "a path", other),
            };
        }
        f(self, slot)
    }
}
