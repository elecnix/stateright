//! A [`Model`] compiled from a declarative JSON document.

use super::expr::{Ctx, Def, Effect, Expr, Scope, binding, compile, compile_effects};
use super::value::{Symbols, Value, collect_strings};
use crate::{Fingerprint, Model, Property, fingerprint};
use dashmap::DashSet;
use nohash_hasher::NoHashHasher;
use std::fmt;
use std::hash::BuildHasherDefault;
use std::sync::{Arc, Mutex};

/// The schema version this crate reads, in the document's `"stateright"` key.
pub const SCHEMA_VERSION: i64 = 1;

/// The most invariants one model may declare.
pub const MAX_INVARIANTS: usize = 64;

/// A state of a declarative model: its field values in declaration order, and
/// a bit for each transition invariant the step into this state violated.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct JsonState {
    pub(crate) fields: Vec<Value>,
    pub(crate) violated: u64,
}

impl fmt::Debug for JsonState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.fields)
    }
}

/// An enabled action: which action, the values of its parameters, and the
/// state it leads to, computed once when the action is listed.
#[derive(Clone, PartialEq)]
pub struct JsonAction {
    pub(crate) action: usize,
    pub(crate) args: Vec<Value>,
    next: JsonState,
}

impl fmt::Debug for JsonAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:?}", self.action, self.args)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Always,
    Sometimes,
    Transition(u32),
}

#[derive(Debug)]
pub(crate) struct Invariant {
    pub(crate) name: &'static str,
    pub(crate) kind: Kind,
    expr: Option<Expr>,
    frame: usize,
}

#[derive(Debug)]
pub(crate) struct Action {
    pub(crate) name: String,
    pub(crate) params: Vec<(String, Expr)>,
    lets: Vec<(usize, Expr)>,
    guard: Option<Expr>,
    effects: Vec<Effect>,
    /// False when the action keeps a normalized state normalized, so the
    /// model's `normalize` effects can be skipped after it.
    normalize: bool,
    frame: usize,
}

/// A transition system read from JSON. See the crate README for the schema.
pub struct JsonModel {
    pub(crate) name: String,
    pub(crate) symbols: Symbols,
    pub(crate) fields: Vec<String>,
    init: Vec<JsonState>,
    defs: Vec<Def>,
    pub(crate) actions: Vec<Action>,
    normalize: Vec<Effect>,
    normalize_frame: usize,
    /// The largest frame of any action.
    action_frame: usize,
    pub(crate) invariants: Vec<Invariant>,
    properties: Vec<Property<JsonModel>>,
    pub(crate) max_depth: Option<usize>,
    /// Distinct states reached by a step that violated a transition
    /// invariant. They are terminal and are not states of the system.
    /// The set holds fingerprints, which is how the checker tells states
    /// apart, so the two counts subtract exactly.
    pub(crate) flagged: DashSet<Fingerprint, BuildHasherDefault<NoHashHasher<u64>>>,
    /// The first evaluation error. An error stops no thread by itself, so
    /// [`check`](super::check) reports it once the search ends.
    pub(crate) error: Mutex<Option<String>>,
}

impl fmt::Debug for JsonModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JsonModel({})", self.name)
    }
}

type Json = serde_json::Value;

fn obj<'j>(json: &'j Json, at: &str) -> Result<&'j serde_json::Map<String, Json>, String> {
    json.as_object()
        .ok_or_else(|| format!("{at}: expected an object"))
}

fn list<'j>(json: Option<&'j Json>, at: &str) -> Result<&'j [Json], String> {
    match json {
        None => Ok(&[]),
        Some(Json::Array(items)) => Ok(items),
        Some(_) => Err(format!("{at}: expected an array")),
    }
}

impl JsonModel {
    /// Parses and compiles a model from JSON text.
    pub fn from_json_str(text: &str) -> Result<JsonModel, String> {
        let json: Json = serde_json::from_str(text).map_err(|e| format!("invalid JSON: {e}"))?;
        JsonModel::from_json(&json)
    }

    /// Compiles a model from a parsed JSON document.
    pub fn from_json(doc: &Json) -> Result<JsonModel, String> {
        let top = obj(doc, "model")?;
        for key in top.keys() {
            if !matches!(
                key.as_str(),
                "stateright"
                    | "name"
                    | "description"
                    | "init"
                    | "defs"
                    | "actions"
                    | "normalize"
                    | "invariants"
                    | "max_depth"
            ) {
                return Err(format!("model: unknown key {key:?}"));
            }
        }
        match top.get("stateright").and_then(Json::as_i64) {
            Some(SCHEMA_VERSION) => {}
            other => {
                return Err(format!(
                    "model: \"stateright\" must be {SCHEMA_VERSION}, the schema version; found {other:?}"
                ));
            }
        }
        let name = top
            .get("name")
            .and_then(Json::as_str)
            .unwrap_or("model")
            .to_string();
        let mut strings = Vec::new();
        collect_strings(doc, &mut strings);
        let symbols = Symbols::from_strings(strings);

        // State fields come from the first initial state.
        let init_json = list(top.get("init"), "init")?;
        let first = init_json
            .first()
            .ok_or("init: a model needs at least one initial state")?;
        let fields: Vec<String> = obj(first, "init[0]")?.keys().cloned().collect();
        let mut init = Vec::new();
        for (i, s) in init_json.iter().enumerate() {
            let at = format!("init[{i}]");
            let s = obj(s, &at)?;
            if s.len() != fields.len() || !fields.iter().all(|f| s.contains_key(f)) {
                return Err(format!(
                    "{at}: every initial state has the fields of init[0]"
                ));
            }
            let values = fields
                .iter()
                .map(|f| symbols.value_of(&s[f]))
                .collect::<Result<_, _>>()?;
            init.push(JsonState {
                fields: values,
                violated: 0,
            });
        }

        // Invariants are declared before actions compile, because an effect
        // names the transition invariant it violates.
        let inv_json = list(top.get("invariants"), "invariants")?;
        if inv_json.len() > MAX_INVARIANTS {
            return Err(format!(
                "invariants: at most {MAX_INVARIANTS} invariants, found {}",
                inv_json.len()
            ));
        }
        let mut transitions = Vec::new();
        for (i, inv) in inv_json.iter().enumerate() {
            let inv = obj(inv, &format!("invariants[{i}]"))?;
            if inv.get("transition").and_then(Json::as_bool) == Some(true) {
                let name = inv.get("name").and_then(Json::as_str).unwrap_or_default();
                transitions.push(name.to_string());
            }
        }

        // Defs: each compiles in its own frame, its parameters first.
        let defs_json = match top.get("defs") {
            None => serde_json::Map::new(),
            Some(d) => obj(d, "defs")?.clone(),
        };
        let def_names: Vec<String> = defs_json.keys().cloned().collect();
        let mut defs = Vec::new();
        let mut calls = Vec::new();
        for (name, def) in &defs_json {
            let at = format!("defs.{name}");
            let d = obj(def, &at)?;
            let params = list(d.get("params"), &at)?;
            let mut scope = Scope::new(&fields, &def_names, &transitions, &symbols);
            for p in params {
                let p = p
                    .as_str()
                    .ok_or_else(|| format!("{at}.params: parameter names are strings"))?;
                scope.bind(p);
            }
            let body = d
                .get("body")
                .ok_or_else(|| format!("{at}: missing \"body\""))?;
            let body = compile(&mut scope, body, &format!("{at}.body"))?;
            calls.push(std::mem::take(&mut scope.calls));
            defs.push(Def {
                name: name.clone(),
                arity: params.len(),
                frame: scope.frame(),
                body,
            });
        }
        reject_recursion(&def_names, &calls)?;

        let mut actions = Vec::new();
        for (i, a) in list(top.get("actions"), "actions")?.iter().enumerate() {
            actions.push(compile_action(
                a,
                &format!("actions[{i}]"),
                &fields,
                &def_names,
                &transitions,
                &symbols,
            )?);
        }

        let mut scope = Scope::new(&fields, &def_names, &transitions, &symbols);
        let normalize = match top.get("normalize") {
            None => Vec::new(),
            Some(n) => compile_effects(&mut scope, n, "normalize")?,
        };
        let normalize_frame = scope.frame();

        let mut invariants = Vec::new();
        let mut names = std::collections::HashSet::new();
        for (i, inv) in inv_json.iter().enumerate() {
            let at = format!("invariants[{i}]");
            let inv = obj(inv, &at)?;
            let name = inv
                .get("name")
                .and_then(Json::as_str)
                .ok_or_else(|| format!("{at}: missing \"name\""))?;
            if !names.insert(name.to_string()) {
                return Err(format!("{at}: duplicate invariant name {name:?}"));
            }
            let mut scope = Scope::new(&fields, &def_names, &transitions, &symbols);
            let (kind, expr) = if let Some(e) = inv.get("always") {
                (
                    Kind::Always,
                    Some(compile(&mut scope, e, &format!("{at}.always"))?),
                )
            } else if let Some(e) = inv.get("sometimes") {
                (
                    Kind::Sometimes,
                    Some(compile(&mut scope, e, &format!("{at}.sometimes"))?),
                )
            } else if inv.get("transition").and_then(Json::as_bool) == Some(true) {
                let bit = transitions.iter().position(|t| t == name).unwrap();
                (Kind::Transition(bit as u32), None)
            } else {
                return Err(format!(
                    "{at}: an invariant has \"always\", \"sometimes\" or \"transition\": true"
                ));
            };
            invariants.push(Invariant {
                name: Box::leak(name.to_string().into_boxed_str()),
                kind,
                expr,
                frame: scope.frame(),
            });
        }

        let max_depth = match top.get("max_depth") {
            None | Some(Json::Null) => None,
            Some(d) => Some(
                d.as_u64()
                    .ok_or("max_depth: expected a non-negative integer")? as usize,
            ),
        };

        let action_frame = actions.iter().map(|a| a.frame).max().unwrap_or(0);
        let mut model = JsonModel {
            name,
            symbols,
            fields,
            init,
            defs,
            actions,
            normalize,
            normalize_frame,
            action_frame,
            invariants,
            properties: Vec::new(),
            max_depth,
            flagged: DashSet::default(),
            error: Mutex::new(None),
        };
        model.properties = model
            .invariants
            .iter()
            .enumerate()
            .map(|(i, inv)| {
                let condition = CONDITIONS[i];
                match inv.kind {
                    Kind::Sometimes => Property::sometimes(inv.name, condition),
                    _ => Property::always(inv.name, condition),
                }
            })
            .collect();
        if model.properties.is_empty() {
            // The checker stops once every property has a discovery, and a
            // model without properties would stop at once. This one never
            // has a discovery, so the checker visits every state.
            model
                .properties
                .push(Property::sometimes("(visit every state)", |_, _| false));
        }
        Ok(model)
    }

    /// The model's name, from its `"name"` key.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Converts a state to a JSON object.
    pub fn state_to_json(&self, state: &JsonState) -> Json {
        Json::Object(
            self.fields
                .iter()
                .zip(&state.fields)
                .map(|(f, v)| (f.clone(), self.symbols.json_of(v)))
                .collect(),
        )
    }

    /// Converts an action to a JSON object: its name and its parameters.
    pub fn action_to_json(&self, action: &JsonAction) -> Json {
        let a = &self.actions[action.action];
        let args: serde_json::Map<String, Json> = a
            .params
            .iter()
            .zip(&action.args)
            .map(|((name, _), v)| (name.clone(), self.symbols.json_of(v)))
            .collect();
        serde_json::json!({ "name": a.name, "args": args })
    }

    /// The names of the transition invariants the step into `state`
    /// violated.
    pub fn violations(&self, state: &JsonState) -> Vec<&'static str> {
        self.invariants
            .iter()
            .filter(|inv| matches!(inv.kind, Kind::Transition(b) if state.violated & (1 << b) != 0))
            .map(|inv| inv.name)
            .collect()
    }

    /// The number of distinct states the checker generated that a step
    /// violating a transition invariant led to. Subtract it from the
    /// checker's unique state count to count the system's states.
    pub fn flagged_state_count(&self) -> usize {
        self.flagged.len()
    }

    fn fail(&self, error: String) {
        let mut slot = self.error.lock().unwrap();
        if slot.is_none() {
            *slot = Some(error);
        }
    }

    /// The first evaluation error the search met, if any.
    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }

    fn ctx(&self, frame: usize) -> Ctx<'_> {
        Ctx {
            symbols: &self.symbols,
            defs: &self.defs,
            stack: vec![Value::Null; frame],
            memo: Vec::new(),
        }
    }

    fn holds(&self, i: usize, state: &JsonState) -> bool {
        if let Kind::Transition(bit) = self.invariants[i].kind {
            return state.violated & (1 << bit) == 0;
        }
        // The checker asks for each invariant of a state in turn. The first
        // question evaluates them all with one context, so zero-argument
        // defs they share run once, and the rest read the cached answers.
        let key = self as *const JsonModel as usize;
        CHECKED.with(|cell| {
            let mut cached = cell.borrow_mut();
            if let Some((model, fields, results)) = cached.as_ref()
                && *model == key
                && same_object(fields, &state.fields)
            {
                return results[i];
            }
            let results = self.evaluate_invariants(state);
            let answer = results[i];
            *cached = Some((key, state.fields.clone(), results));
            answer
        })
    }

    fn evaluate_invariants(&self, state: &JsonState) -> Vec<bool> {
        let frame = self
            .invariants
            .iter()
            .map(|inv| inv.frame)
            .max()
            .unwrap_or(0);
        let mut ctx = self.ctx(frame);
        ctx.memo = vec![None; self.defs.len().max(1)];
        self.invariants
            .iter()
            .map(|inv| match (&inv.kind, &inv.expr) {
                (Kind::Transition(bit), _) => state.violated & (1 << bit) == 0,
                (_, Some(expr)) => match ctx.truth(expr, &state.fields, 0, inv.name) {
                    Ok(b) => b,
                    Err(e) => {
                        self.fail(format!("invariant {:?}: {e}", inv.name));
                        true
                    }
                },
                (_, None) => true,
            })
            .collect()
    }

    fn enabled(&self, state: &JsonState, out: &mut Vec<JsonAction>) -> Result<(), String> {
        let mut ctx = self.ctx(self.action_frame);
        let mut nctx = self.ctx(self.normalize_frame);
        let mut args = Vec::new();
        for (ai, action) in self.actions.iter().enumerate() {
            self.each_binding(ai, state, &mut ctx, &mut nctx, &mut args, out)
                .map_err(|e| format!("action {:?}: {e}", action.name))?;
        }
        Ok(())
    }

    fn each_binding(
        &self,
        ai: usize,
        state: &JsonState,
        ctx: &mut Ctx,
        nctx: &mut Ctx,
        args: &mut Vec<Value>,
        out: &mut Vec<JsonAction>,
    ) -> Result<(), String> {
        let action = &self.actions[ai];
        let depth = args.len();
        if depth < action.params.len() {
            let domain = ctx.domain(&action.params[depth].1, &state.fields, 0)?;
            for i in 0..domain.len() {
                // Parameters take the first slots of the action's frame.
                let v = domain.get(i);
                ctx.stack[depth] = v.clone();
                args.push(v);
                self.each_binding(ai, state, ctx, nctx, args, out)?;
                args.pop();
            }
            return Ok(());
        }
        for (slot, e) in &action.lets {
            let v = ctx.eval(e, &state.fields, 0)?;
            ctx.stack[*slot] = v;
        }
        if let Some(guard) = &action.guard
            && !ctx.truth(guard, &state.fields, 0, "guard")?
        {
            return Ok(());
        }
        let mut next = state.fields.clone();
        let mut violated = 0;
        if !ctx.apply(&action.effects, &mut next, &mut violated, 0)? {
            return Ok(());
        }
        if action.normalize && !self.normalize.is_empty() {
            let mut ignored = 0;
            if !nctx
                .apply(&self.normalize, &mut next, &mut ignored, 0)
                .map_err(|e| format!("normalize: {e}"))?
            {
                return Ok(());
            }
        }
        if violated != 0 {
            // The step breaks a transition invariant. The system still moves
            // to `next`, and a terminal copy marked with the violation lets
            // the checker report the step.
            out.push(JsonAction {
                action: ai,
                args: args.clone(),
                next: JsonState {
                    fields: next.clone(),
                    violated: 0,
                },
            });
        }
        out.push(JsonAction {
            action: ai,
            args: args.clone(),
            next: JsonState {
                fields: next,
                violated,
            },
        });
        Ok(())
    }
}

fn compile_action(
    json: &Json,
    at: &str,
    fields: &[String],
    defs: &[String],
    transitions: &[String],
    symbols: &Symbols,
) -> Result<Action, String> {
    let a = obj(json, at)?;
    for key in a.keys() {
        if !matches!(
            key.as_str(),
            "name" | "for" | "let" | "guard" | "effects" | "normalize" | "description"
        ) {
            return Err(format!("{at}: unknown key {key:?}"));
        }
    }
    let name = a
        .get("name")
        .and_then(Json::as_str)
        .ok_or_else(|| format!("{at}: missing \"name\""))?
        .to_string();
    let at = &format!("{at}({name})");
    let mut scope = Scope::new(fields, defs, transitions, symbols);
    let mut params = Vec::new();
    for (i, p) in list(a.get("for"), &format!("{at}.for"))?.iter().enumerate() {
        let (pname, domain) = binding(p, &format!("{at}.for[{i}]"))?;
        let domain = compile(&mut scope, domain, &format!("{at}.for.{pname}"))?;
        let slot = scope.bind(pname);
        debug_assert_eq!(slot, i);
        params.push((pname.to_string(), domain));
    }
    let mut lets = Vec::new();
    for (i, l) in list(a.get("let"), &format!("{at}.let"))?.iter().enumerate() {
        let (lname, value) = binding(l, &format!("{at}.let[{i}]"))?;
        let value = compile(&mut scope, value, &format!("{at}.let.{lname}"))?;
        lets.push((scope.bind(lname), value));
    }
    let guard = a
        .get("guard")
        .map(|g| compile(&mut scope, g, &format!("{at}.guard")))
        .transpose()?;
    let effects = match a.get("effects") {
        None => Vec::new(),
        Some(e) => compile_effects(&mut scope, e, &format!("{at}.effects"))?,
    };
    let normalize = match a.get("normalize") {
        None => true,
        Some(Json::Bool(b)) => *b,
        Some(_) => return Err(format!("{at}.normalize: expected true or false")),
    };
    Ok(Action {
        name,
        params,
        lets,
        guard,
        effects,
        normalize,
        frame: scope.frame(),
    })
}

fn reject_recursion(names: &[String], calls: &[Vec<usize>]) -> Result<(), String> {
    // 0 = unvisited, 1 = on the current path, 2 = done.
    fn visit(
        i: usize,
        calls: &[Vec<usize>],
        mark: &mut [u8],
        names: &[String],
    ) -> Result<(), String> {
        match mark[i] {
            1 => {
                return Err(format!(
                    "defs: {:?} calls itself, directly or through other defs",
                    names[i]
                ));
            }
            2 => return Ok(()),
            _ => {}
        }
        mark[i] = 1;
        for &c in &calls[i] {
            visit(c, calls, mark, names)?;
        }
        mark[i] = 2;
        Ok(())
    }
    let mut mark = vec![0u8; names.len()];
    for i in 0..names.len() {
        visit(i, calls, &mut mark, names)?;
    }
    Ok(())
}

impl Model for JsonModel {
    type State = JsonState;
    type Action = JsonAction;

    fn init_states(&self) -> Vec<JsonState> {
        self.init.clone()
    }

    fn actions(&self, state: &JsonState, actions: &mut Vec<JsonAction>) {
        if state.violated != 0 {
            return;
        }
        if let Err(e) = self.enabled(state, actions) {
            actions.clear();
            self.fail(e);
        }
    }

    fn next_state(&self, _state: &JsonState, action: JsonAction) -> Option<JsonState> {
        if action.next.violated != 0 {
            self.flagged.insert(fingerprint(&action.next));
        }
        Some(action.next)
    }

    fn format_action(&self, action: &JsonAction) -> String {
        self.action_to_json(action).to_string()
    }

    fn properties(&self) -> Vec<Property<Self>> {
        self.properties.clone()
    }
}

thread_local! {
    /// The invariant answers for the state this thread checked last.
    static CHECKED: std::cell::RefCell<Option<Checked>> = const { std::cell::RefCell::new(None) };
}

/// A model's address, a state's fields, and one answer per invariant.
type Checked = (usize, Vec<Value>, Vec<bool>);

/// Whether two states are the same object: every list and record is the
/// same allocation and every scalar is equal. The cache keeps its copy of
/// the fields alive, so an address it compares can't be reused by another
/// state while the entry exists.
fn same_object(a: &[Value], b: &[Value]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| match (x, y) {
            (Value::List(x), Value::List(y)) => Arc::ptr_eq(x, y),
            (Value::Rec(x), Value::Rec(y)) => Arc::ptr_eq(x, y),
            (Value::List(_) | Value::Rec(_), _) | (_, Value::List(_) | Value::Rec(_)) => false,
            (x, y) => x == y,
        })
}

/// One condition function per invariant slot. [`Property`] takes a plain
/// function pointer, so each slot is a separate instance of `holds::<I>`.
const CONDITIONS: [fn(&JsonModel, &JsonState) -> bool; MAX_INVARIANTS] = {
    fn holds<const I: usize>(m: &JsonModel, s: &JsonState) -> bool {
        m.holds(I, s)
    }
    macro_rules! table {
        ($($i:literal)*) => { [$(holds::<$i>),*] };
    }
    table!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
           32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63)
};
