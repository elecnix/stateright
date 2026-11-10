//! Models written as JSON documents instead of Rust code.
//!
//! A [`JsonModel`] compiles a declarative description of a transition system:
//! initial states as JSON objects, actions as guarded updates, and invariants
//! as predicates. It implements [`Model`](crate::Model), so every checker in
//! this crate runs it. The `stateright` binary checks such a file from the
//! command line. The schema is documented in the crate README, under
//! "Declarative JSON models".

mod check;
mod expr;
mod model;
mod value;

pub use check::{CheckOptions, CheckReport, check};
pub use model::{JsonAction, JsonModel, JsonState, MAX_INVARIANTS, SCHEMA_VERSION};
pub use value::{Hashed, Sym, Value};

#[cfg(test)]
mod tests;
