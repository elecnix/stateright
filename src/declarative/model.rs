//! A [`Model`] compiled from a declarative JSON document.

use super::expr::{
    Access, Buffers, Ctx, Def, Effect, Expr, Scope, binding, compile, compile_effects, def_reads,
};
use super::value::{Symbols, Value, collect_strings};
use crate::{Fingerprint, Model, Property, fingerprint};
use dashmap::DashSet;
use nohash_hasher::NoHashHasher;
use std::fmt;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
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
    pub(crate) args: Arc<[Value]>,
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
    /// The state fields that decide what the action does: every field its
    /// parameters, guard and effects read or change, and the fields the
    /// model's `normalize` effects read or change when they run after it.
    reads: Vec<usize>,
    /// The state fields the action can change, a subset of `reads`.
    writes: Vec<usize>,
}

/// A transition system read from JSON. See the crate README for the schema.
pub struct JsonModel {
    /// A number no other model in this process has, so a thread can tell
    /// whose results it has cached.
    id: u64,
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
    /// The largest frame of any invariant.
    invariant_frame: usize,
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

        let def_reads = def_reads(&defs, fields.len());

        let mut scope = Scope::new(&fields, &def_names, &transitions, &symbols);
        let normalize = match top.get("normalize") {
            None => Vec::new(),
            Some(n) => compile_effects(&mut scope, n, "normalize")?,
        };
        let normalize_frame = scope.frame();
        let mut normalize_access = Access::none(fields.len());
        normalize_access.effects(&normalize, &def_reads);

        let mut actions = Vec::new();
        for (i, a) in list(top.get("actions"), "actions")?.iter().enumerate() {
            actions.push(compile_action(
                a,
                &format!("actions[{i}]"),
                &fields,
                &def_names,
                &transitions,
                &symbols,
                &def_reads,
                &normalize_access,
            )?);
        }

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
        let invariant_frame = invariants.iter().map(|i| i.frame).max().unwrap_or(0);
        let mut model = JsonModel {
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            name,
            symbols,
            fields,
            init,
            defs,
            actions,
            normalize,
            normalize_frame,
            action_frame,
            invariant_frame,
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
            .zip(action.args.iter())
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

    fn holds(&self, i: usize, state: &JsonState) -> bool {
        if let Kind::Transition(bit) = self.invariants[i].kind {
            return state.violated & (1 << bit) == 0;
        }
        // The checker asks for each invariant of a state in turn. The first
        // question evaluates them all with one context, so zero-argument
        // defs they share run once, and the rest read the cached answers.
        CHECKED.with(|cell| {
            let mut checked = cell.borrow_mut();
            if checked.model != self.id || !same_object(&checked.fields, &state.fields) {
                checked.model = self.id;
                checked.fields.clone_from(&state.fields);
                self.evaluate_invariants(state, &mut checked);
            }
            checked.results[i]
        })
    }

    fn evaluate_invariants(&self, state: &JsonState, checked: &mut Checked) {
        let buffers = std::mem::take(&mut checked.buffers);
        let mut ctx = Ctx::new(
            &self.symbols,
            &self.defs,
            self.invariant_frame,
            true,
            buffers,
        );
        checked.results.clear();
        for inv in &self.invariants {
            checked.results.push(match (&inv.kind, &inv.expr) {
                (Kind::Transition(bit), _) => state.violated & (1 << bit) == 0,
                (_, Some(expr)) => match ctx.truth(expr, &state.fields, 0, inv.name) {
                    Ok(b) => b,
                    Err(e) => {
                        self.fail(format!("invariant {:?}: {e}", inv.name));
                        true
                    }
                },
                (_, None) => true,
            });
        }
        checked.buffers = ctx.into_buffers();
    }

    fn enabled(&self, state: &JsonState, out: &mut Vec<JsonAction>) -> Result<(), String> {
        MEMO.with(|cell| {
            let mut memo = cell.borrow_mut();
            memo.attach(self);
            let context =
                |frame, buffers| Ctx::new(&self.symbols, &self.defs, frame, false, buffers);
            let mut run = Run {
                ctx: context(self.action_frame, std::mem::take(&mut memo.buffers.0)),
                nctx: context(self.normalize_frame, std::mem::take(&mut memo.buffers.1)),
                no_args: Arc::clone(&memo.no_args),
                args: Vec::new(),
            };
            let result = self.enabled_with(state, &mut run, &mut memo, out);
            memo.buffers = (run.ctx.into_buffers(), run.nctx.into_buffers());
            result
        })
    }

    fn enabled_with(
        &self,
        state: &JsonState,
        run: &mut Run,
        memo: &mut Memo,
        out: &mut Vec<JsonAction>,
    ) -> Result<(), String> {
        for (ai, action) in self.actions.iter().enumerate() {
            let hash = match memo.lookup(ai, &action.reads, &state.fields) {
                Lookup::Hit(steps) => {
                    for step in steps {
                        let mut fields = state.fields.clone();
                        for (f, v) in action.writes.iter().zip(&step.written) {
                            fields[*f] = v.clone();
                        }
                        out.push(JsonAction {
                            action: ai,
                            args: Arc::clone(&step.args),
                            next: JsonState {
                                fields,
                                violated: step.violated,
                            },
                        });
                    }
                    continue;
                }
                Lookup::Miss(hash) => Some(hash),
                Lookup::Paused => None,
            };
            let start = out.len();
            self.each_binding(ai, state, run, out)
                .map_err(|e| format!("action {:?}: {e}", action.name))?;
            if let Some(hash) = hash {
                let steps = out[start..]
                    .iter()
                    .map(|step| Step {
                        args: Arc::clone(&step.args),
                        violated: step.next.violated,
                        written: pick(&step.next.fields, &action.writes),
                    })
                    .collect();
                memo.remember(hash, ai, pick(&state.fields, &action.reads), steps);
            }
        }
        Ok(())
    }

    fn each_binding(
        &self,
        ai: usize,
        state: &JsonState,
        run: &mut Run,
        out: &mut Vec<JsonAction>,
    ) -> Result<(), String> {
        let action = &self.actions[ai];
        let depth = run.args.len();
        if depth < action.params.len() {
            let domain = run.ctx.domain(&action.params[depth].1, &state.fields, 0)?;
            for i in 0..domain.len() {
                // Parameters take the first slots of the action's frame.
                let v = domain.get(i);
                run.ctx.stack[depth] = v.clone();
                run.args.push(v);
                self.each_binding(ai, state, run, out)?;
                run.args.pop();
            }
            return Ok(());
        }
        for (slot, e) in &action.lets {
            let v = run.ctx.eval(e, &state.fields, 0)?;
            run.ctx.stack[*slot] = v;
        }
        if let Some(guard) = &action.guard
            && !run.ctx.truth(guard, &state.fields, 0, "guard")?
        {
            return Ok(());
        }
        let mut next = state.fields.clone();
        let mut violated = 0;
        if !run
            .ctx
            .apply(&action.effects, &mut next, &mut violated, 0)?
        {
            return Ok(());
        }
        if action.normalize && !self.normalize.is_empty() {
            let mut ignored = 0;
            if !run
                .nctx
                .apply(&self.normalize, &mut next, &mut ignored, 0)
                .map_err(|e| format!("normalize: {e}"))?
            {
                return Ok(());
            }
        }
        let args: Arc<[Value]> = if run.args.is_empty() {
            Arc::clone(&run.no_args)
        } else {
            Arc::from(&run.args[..])
        };
        if violated != 0 {
            // The step breaks a transition invariant. The system still moves
            // to `next`, and a terminal copy marked with the violation lets
            // the checker report the step.
            out.push(JsonAction {
                action: ai,
                args: Arc::clone(&args),
                next: JsonState {
                    fields: next.clone(),
                    violated: 0,
                },
            });
        }
        out.push(JsonAction {
            action: ai,
            args,
            next: JsonState {
                fields: next,
                violated,
            },
        });
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn compile_action(
    json: &Json,
    at: &str,
    fields: &[String],
    defs: &[String],
    transitions: &[String],
    symbols: &Symbols,
    def_reads: &[Vec<bool>],
    normalize_access: &Access,
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
    let mut access = Access::none(fields.len());
    for (_, domain) in &params {
        access.expr(domain, def_reads);
    }
    for (_, value) in &lets {
        access.expr(value, def_reads);
    }
    if let Some(guard) = &guard {
        access.expr(guard, def_reads);
    }
    access.effects(&effects, def_reads);
    if normalize {
        access.merge(normalize_access);
    }
    Ok(Action {
        name,
        params,
        lets,
        guard,
        effects,
        normalize,
        frame: scope.frame(),
        reads: set(&access.reads),
        writes: set(&access.writes),
    })
}

/// The indices of the flags that are set.
fn set(flags: &[bool]) -> Vec<usize> {
    (0..flags.len()).filter(|f| flags[*f]).collect()
}

/// The values of the fields `which` lists.
fn pick(fields: &[Value], which: &[usize]) -> Vec<Value> {
    which.iter().map(|f| fields[*f].clone()).collect()
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

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

/// The slots of a thread's memo. A power of two.
const MEMO_SLOTS: usize = 1 << 16;
/// How many lookups of one action a thread counts before it judges whether
/// the memo helps that action.
const MEMO_WINDOW: u32 = 1024;
/// The fewest hits in a window for the memo to stay on for an action.
const MEMO_MIN_HITS: u32 = MEMO_WINDOW / 8;
/// How many lookups an action skips the memo for after its first poor
/// window. Each poor window in a row doubles the pause, up to
/// `MEMO_LONGEST_PAUSE`.
const MEMO_PAUSE: u32 = 8 * MEMO_WINDOW;
const MEMO_LONGEST_PAUSE: u32 = 1 << 24;

/// What the evaluation of one state's actions works with.
struct Run<'m> {
    /// The context of the actions and the context of `normalize`.
    ctx: Ctx<'m>,
    nctx: Ctx<'m>,
    /// The arguments of an action without parameters.
    no_args: Arc<[Value]>,
    /// The arguments bound so far.
    args: Vec<Value>,
}

/// One step an action offers: its arguments, the invariant bits the step
/// violates, and the new values of the fields the action can change.
struct Step {
    args: Arc<[Value]>,
    violated: u64,
    written: Vec<Value>,
}

/// What an action did in the states that have `key` in the fields it reads.
struct Remembered {
    hash: u64,
    action: usize,
    key: Vec<Value>,
    steps: Vec<Step>,
}

#[derive(Clone, Default)]
struct Usage {
    lookups: u32,
    hits: u32,
    /// Lookups left to skip.
    pause: u32,
    /// The length of the next pause, or zero after a good window.
    next_pause: u32,
    #[cfg(test)]
    total_hits: u64,
}

enum Lookup<'m> {
    /// The steps remembered for these field values.
    Hit(&'m [Step]),
    /// Nothing remembered. Carries the hash to remember the steps under.
    Miss(u64),
    /// The memo is off for this action for now, because it rarely hit.
    Paused,
}

/// A thread's memo of what each action does. An action reads some of the
/// state's fields, and it does the same in every state that agrees on them.
/// Many states do when the action reads few fields: the search reaches them
/// by changing the others. The memo remembers the steps the action offered
/// for the field values it has seen.
///
/// The memo is a fixed table, where a new entry replaces the one in its
/// slot, so its size does not grow with the search. Each thread has its
/// own, so a lookup takes no lock. An action that rarely hits, because it
/// reads most of the state, stops using the memo for a while.
struct Memo {
    model: u64,
    slots: Vec<Option<Remembered>>,
    usage: Vec<Usage>,
    /// The arguments of an action without parameters.
    no_args: Arc<[Value]>,
    /// The allocations of the action context and the `normalize` context.
    buffers: (Buffers, Buffers),
}

impl Memo {
    fn new() -> Memo {
        Memo {
            model: 0,
            slots: Vec::new(),
            usage: Vec::new(),
            no_args: Arc::from(Vec::new()),
            buffers: Default::default(),
        }
    }

    /// Empties the memo when the thread turns to another model.
    fn attach(&mut self, model: &JsonModel) {
        if self.model != model.id {
            self.model = model.id;
            self.slots = Vec::new();
            self.usage = vec![Usage::default(); model.actions.len()];
        }
    }

    /// Looks up what action `ai` did for the values `fields` has in the
    /// fields the action reads.
    fn lookup(&mut self, ai: usize, reads: &[usize], fields: &[Value]) -> Lookup<'_> {
        let usage = &mut self.usage[ai];
        if usage.pause > 0 {
            usage.pause -= 1;
            return Lookup::Paused;
        }
        if usage.lookups == MEMO_WINDOW {
            let poor = usage.hits < MEMO_MIN_HITS;
            usage.lookups = 0;
            usage.hits = 0;
            if poor {
                let pause = usage.next_pause.max(MEMO_PAUSE);
                usage.next_pause = (pause * 2).min(MEMO_LONGEST_PAUSE);
                // This lookup is the first of the pause.
                usage.pause = pause - 1;
                return Lookup::Paused;
            }
            usage.next_pause = 0;
        }
        usage.lookups += 1;
        let mut hasher = crate::stable::hasher();
        hasher.write_usize(ai);
        for f in reads {
            fields[*f].hash(&mut hasher);
        }
        let hash = hasher.finish();
        match self.slots.get(hash as usize & (MEMO_SLOTS - 1)) {
            Some(Some(entry))
                if entry.hash == hash
                    && entry.action == ai
                    && entry.key.iter().eq(reads.iter().map(|f| &fields[*f])) =>
            {
                usage.hits += 1;
                #[cfg(test)]
                {
                    usage.total_hits += 1;
                }
                Lookup::Hit(&entry.steps)
            }
            _ => Lookup::Miss(hash),
        }
    }

    /// Remembers the steps action `ai` offered for the field values in
    /// `key`. The table is allocated on the first call.
    fn remember(&mut self, hash: u64, ai: usize, key: Vec<Value>, steps: Vec<Step>) {
        if self.slots.is_empty() {
            self.slots.resize_with(MEMO_SLOTS, || None);
        }
        self.slots[hash as usize & (MEMO_SLOTS - 1)] = Some(Remembered {
            hash,
            action: ai,
            key,
            steps,
        });
    }
}

/// How many times this thread's memo answered for `model`, per action.
#[cfg(test)]
pub(crate) fn memo_hits(model: &JsonModel) -> Vec<u64> {
    MEMO.with(|cell| {
        let memo = cell.borrow();
        assert_eq!(memo.model, model.id, "the memo is for another model");
        memo.usage.iter().map(|u| u.total_hits).collect()
    })
}

thread_local! {
    /// What each action did in the states this thread expanded.
    static MEMO: std::cell::RefCell<Memo> = std::cell::RefCell::new(Memo::new());

    /// The invariant answers for the state this thread checked last.
    static CHECKED: std::cell::RefCell<Checked> = std::cell::RefCell::new(Checked::default());
}

/// A model's id, a state's fields, and one answer per invariant.
#[derive(Default)]
struct Checked {
    model: u64,
    fields: Vec<Value>,
    results: Vec<bool>,
    buffers: Buffers,
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The steps `state` offers, as `(action, next state)` in JSON.
    fn steps(model: &JsonModel, state: &JsonState) -> Vec<(Json, Json)> {
        let mut actions = Vec::new();
        model.actions(state, &mut actions);
        actions
            .into_iter()
            .map(|action| {
                let shown = model.action_to_json(&action);
                let next = model.next_state(state, action).unwrap();
                (shown["args"].clone(), model.state_to_json(&next))
            })
            .collect()
    }

    fn names(model: &JsonModel, fields: &[usize]) -> Vec<String> {
        fields.iter().map(|f| model.fields[*f].clone()).collect()
    }

    fn bump(by: i64) -> JsonModel {
        JsonModel::from_json(&json!({
            "stateright": 1,
            "init": [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}],
            "actions": [{
                "name": "bump", "for": [["x", [1, 2]]],
                "guard": {">": [{"var": "a"}, 0]},
                "effects": [{"set": "a", "to": {"+": [{"var": "a"}, {"*": [{"var": "x"}, by]}]}}]
            }]
        }))
        .unwrap()
    }

    #[test]
    fn an_action_reads_what_its_parts_and_the_defs_they_call_read() {
        let model = JsonModel::from_json(&json!({
            "stateright": 1,
            "init": [{"w": [], "x": 0, "y": 0, "z": 0, "other": 0, "tidy": 0}],
            "defs": {
                "inner": {"params": [], "body": {"var": "x"}},
                "outer": {"params": [], "body": {"==": [{"call": ["inner"]}, 0]}}
            },
            "actions": [
                {"name": "a", "guard": {"call": ["outer"]}, "normalize": false, "effects": [
                    {"set": "y", "to": 1},
                    {"if": {">": [{"var": "z"}, 0]}, "then": [{"append": "w", "value": 1}]}
                ]},
                {"name": "b", "effects": [{"set": "y", "to": 1}]}
            ],
            "normalize": [{"set": "tidy", "to": {"var": "other"}}]
        }))
        .unwrap();
        let a = &model.actions[0];
        assert_eq!(names(&model, &a.reads), ["w", "x", "y", "z"]);
        assert_eq!(names(&model, &a.writes), ["w", "y"]);
        // The `normalize` effects run after `b`, so it reads and changes
        // what they do.
        let b = &model.actions[1];
        assert_eq!(names(&model, &b.reads), ["other", "tidy", "y"]);
        assert_eq!(names(&model, &b.writes), ["tidy", "y"]);
    }

    #[test]
    fn an_action_repeats_its_steps_in_states_that_agree_on_what_it_reads() {
        let model = bump(1);
        let init = model.init_states();
        let first = steps(&model, &init[0]);
        assert_eq!(
            first,
            [
                (json!({"x": 1}), json!({"a": 2, "b": 10})),
                (json!({"x": 2}), json!({"a": 3, "b": 10})),
            ]
        );
        assert_eq!(memo_hits(&model), [0]);
        // The second state has the same `a`, the one field the action
        // reads. Its steps come from the memo, and keep its own `b`.
        let second = steps(&model, &init[1]);
        assert_eq!(memo_hits(&model), [1]);
        assert_eq!(
            second,
            [
                (json!({"x": 1}), json!({"a": 2, "b": 20})),
                (json!({"x": 2}), json!({"a": 3, "b": 20})),
            ]
        );
        // The third state has another `a`, so the action runs again.
        let third = steps(&model, &init[2]);
        assert_eq!(memo_hits(&model), [1]);
        assert_eq!(third[1], (json!({"x": 2}), json!({"a": 4, "b": 10})));
    }

    #[test]
    fn a_field_changed_in_a_branch_not_taken_keeps_each_states_value() {
        let model = JsonModel::from_json(&json!({
            "stateright": 1,
            "init": [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 9, "b": 30}],
            "actions": [{"name": "reset", "effects": [
                {"if": {">": [{"var": "a"}, 5]}, "then": [{"set": "b", "to": 0}]}
            ]}]
        }))
        .unwrap();
        let init = model.init_states();
        let next = |i: usize| steps(&model, &init[i])[0].1.clone();
        assert_eq!(next(0), json!({"a": 1, "b": 10}));
        assert_eq!(next(1), json!({"a": 1, "b": 20}));
        assert_eq!(next(2), json!({"a": 9, "b": 0}));
    }

    #[test]
    fn a_step_that_violates_a_transition_invariant_repeats_with_its_mark() {
        let model = JsonModel::from_json(&json!({
            "stateright": 1,
            "init": [{"a": 1, "b": 10}, {"a": 1, "b": 20}],
            "actions": [{"name": "break", "effects": [
                {"set": "a", "to": 2}, {"violate": "never"}
            ]}],
            "invariants": [{"name": "never", "transition": true}]
        }))
        .unwrap();
        let init = model.init_states();
        for (state, b) in init.iter().zip([10, 20]) {
            let mut actions = Vec::new();
            model.actions(state, &mut actions);
            // The system's step, then its terminal copy with the mark.
            let marks: Vec<u64> = actions.iter().map(|a| a.next.violated).collect();
            assert_eq!(marks, [0, 1]);
            for action in actions {
                let next = model.next_state(state, action).unwrap();
                assert_eq!(model.state_to_json(&next), json!({"a": 2, "b": b}));
            }
        }
        assert_eq!(memo_hits(&model), [1]);
    }

    #[test]
    fn two_models_on_one_thread_do_not_share_remembered_steps() {
        let (one, ten) = (bump(1), bump(10));
        let a = |model: &JsonModel| steps(model, &model.init_states()[0])[0].1["a"].clone();
        assert_eq!(a(&one), 2);
        assert_eq!(a(&ten), 11);
        assert_eq!(a(&one), 2);
    }

    #[test]
    fn the_memo_pauses_for_an_action_that_never_repeats() {
        let model = bump(1);
        let mut memo = Memo::new();
        memo.attach(&model);
        let mut paused = 0;
        let mut lookups = 0;
        for a in 0..(MEMO_WINDOW as i64 + 1 + MEMO_PAUSE as i64) {
            let fields = [Value::Int(a), Value::Int(0)];
            match memo.lookup(0, &model.actions[0].reads, &fields) {
                Lookup::Paused => paused += 1,
                Lookup::Miss(hash) => {
                    lookups += 1;
                    memo.remember(hash, 0, vec![fields[0].clone()], Vec::new());
                }
                Lookup::Hit(_) => panic!("every state differs in the field the action reads"),
            }
        }
        // One window of misses, then the pause, then the memo is tried
        // again.
        assert_eq!(lookups, MEMO_WINDOW + 1);
        assert_eq!(paused, MEMO_PAUSE);
    }

    #[test]
    fn the_memo_stays_on_for_an_action_that_repeats() {
        let model = bump(1);
        let mut memo = Memo::new();
        memo.attach(&model);
        let fields = [Value::Int(1), Value::Int(0)];
        for round in 0..4 * MEMO_WINDOW {
            match memo.lookup(0, &model.actions[0].reads, &fields) {
                Lookup::Paused => panic!("paused in round {round}"),
                Lookup::Miss(hash) => {
                    assert_eq!(round, 0);
                    memo.remember(hash, 0, vec![fields[0].clone()], Vec::new());
                }
                Lookup::Hit(steps) => assert!(steps.is_empty()),
            }
        }
    }
}
