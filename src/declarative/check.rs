//! Runs the breadth-first checker on a [`JsonModel`] and reports in JSON.

use super::model::{JsonModel, Kind};
use crate::{Checker, Model};
use serde_json::{Value as Json, json};
use std::time::Instant;

/// Options for [`check`].
#[derive(Clone, Debug)]
pub struct CheckOptions {
    /// Checker threads. Traces are shortest paths when this is 1. With more
    /// threads each trace is the shortest one found before the run stopped.
    pub threads: usize,
    /// Overrides the model's `max_depth`.
    pub max_depth: Option<usize>,
}

impl Default for CheckOptions {
    fn default() -> Self {
        CheckOptions {
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            max_depth: None,
        }
    }
}

/// The outcome of a check.
#[derive(Clone, Debug)]
pub struct CheckReport {
    /// Whether every `always` and `transition` invariant held.
    pub ok: bool,
    /// The report as JSON. The README documents its fields.
    pub json: Json,
}

/// Checks every state reachable within the model's bound, breadth first.
/// Fails when an expression meets a type error, an index out of range or an
/// overflow, because the result of such a search means nothing.
pub fn check(model: JsonModel, options: &CheckOptions) -> Result<CheckReport, String> {
    let max_depth = options.max_depth.or(model.max_depth);
    let start = Instant::now();
    let mut builder = model.checker().threads(options.threads.max(1));
    if let Some(d) = max_depth {
        // The checker counts the initial state as depth 1.
        builder = builder.target_max_depth(d + 1);
    }
    let checker = builder.spawn_bfs().join();
    let seconds = start.elapsed().as_secs_f64();
    let model = checker.model();
    if let Some(e) = model.error() {
        return Err(e);
    }

    let mut ok = true;
    let mut invariants = Vec::new();
    for inv in &model.invariants {
        let path = checker.discovery(inv.name);
        let found = path.is_some();
        let (kind, status) = match inv.kind {
            Kind::Always => ("always", if found { "violated" } else { "holds" }),
            Kind::Transition(_) => ("transition", if found { "violated" } else { "holds" }),
            Kind::Sometimes => ("sometimes", if found { "reached" } else { "unreached" }),
        };
        if status == "violated" {
            ok = false;
        }
        let mut entry = json!({ "name": inv.name, "kind": kind, "status": status });
        if let Some(path) = path {
            let mut steps = Vec::new();
            let mut taken = None;
            for (state, action) in path.into_vec() {
                let mut step = serde_json::Map::new();
                if let Some(a) = taken.take() {
                    step.insert("action".into(), model.action_to_json(&a));
                }
                step.insert("state".into(), model.state_to_json(&state));
                let violated = model.violations(&state);
                if !violated.is_empty() {
                    step.insert("violates".into(), json!(violated));
                }
                steps.push(Json::Object(step));
                taken = action;
            }
            entry["trace"] = Json::Array(steps);
        }
        invariants.push(entry);
    }
    let flagged = model.flagged_state_count();
    let json = json!({
        "model": model.name(),
        "result": if ok { "ok" } else { "violated" },
        "states": checker.unique_state_count() - flagged,
        "max_depth": max_depth,
        "depth_reached": checker.max_depth().saturating_sub(1),
        "threads": options.threads.max(1),
        "seconds": (seconds * 1000.0).round() / 1000.0,
        "complete": checker.discoveries().len() < checker.model().properties().len(),
        "invariants": invariants,
    });
    Ok(CheckReport { ok, json })
}
