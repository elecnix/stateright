//! Runtime values of a declarative model.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// An interned string. Symbols are numbered in lexical order of their text, so
/// comparing two symbols compares their strings.
pub type Sym = u32;

/// A value in a declarative model state.
///
/// Lists and records share their contents through [`Arc`], so copying a state
/// copies pointers, and an update copies only the path it changes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Value {
    /// JSON `null`.
    #[default]
    Null,
    /// JSON `true` or `false`.
    Bool(bool),
    /// A JSON integer. Models have no floating point numbers.
    Int(i64),
    /// A JSON string, interned.
    Str(Sym),
    /// A JSON array.
    List(Arc<Vec<Value>>),
    /// A JSON object. Fields are sorted by symbol, which is lexical order.
    Rec(Arc<Vec<(Sym, Value)>>),
}

impl Value {
    pub(crate) fn list(items: Vec<Value>) -> Value {
        Value::List(Arc::new(items))
    }

    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Int(_) => "integer",
            Value::Str(_) => "string",
            Value::List(_) => "array",
            Value::Rec(_) => "object",
        }
    }

    /// A value is true when it is `true`. Other values are type errors in
    /// conditions, so this only answers for booleans.
    pub(crate) fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub(crate) fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub(crate) fn as_list(&self) -> Option<&Arc<Vec<Value>>> {
        match self {
            Value::List(l) => Some(l),
            _ => None,
        }
    }
}

/// The table of strings a model uses. Every string appears in the model file,
/// because expressions cannot build new strings, so the table is complete
/// once the model is compiled.
#[derive(Clone, Debug, Default)]
pub struct Symbols {
    names: Vec<String>,
    ids: BTreeMap<String, Sym>,
}

impl Symbols {
    /// Builds the table from every string in the model, numbered in lexical
    /// order.
    pub(crate) fn from_strings(strings: impl IntoIterator<Item = String>) -> Symbols {
        let set: std::collections::BTreeSet<String> = strings.into_iter().collect();
        let names: Vec<String> = set.into_iter().collect();
        let ids = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i as Sym))
            .collect();
        Symbols { names, ids }
    }

    pub(crate) fn id(&self, name: &str) -> Option<Sym> {
        self.ids.get(name).copied()
    }

    pub(crate) fn name(&self, sym: Sym) -> &str {
        &self.names[sym as usize]
    }

    /// Converts a JSON value to a model value. Fails on a floating point
    /// number or a string missing from the table.
    pub(crate) fn value_of(&self, json: &serde_json::Value) -> Result<Value, String> {
        Ok(match json {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(*b),
            serde_json::Value::Number(n) => match n.as_i64() {
                Some(i) => Value::Int(i),
                None => return Err(format!("{n} is not a 64-bit integer")),
            },
            serde_json::Value::String(s) => Value::Str(
                self.id(s)
                    .ok_or_else(|| format!("string {s:?} is not interned"))?,
            ),
            serde_json::Value::Array(items) => Value::list(
                items
                    .iter()
                    .map(|v| self.value_of(v))
                    .collect::<Result<_, _>>()?,
            ),
            serde_json::Value::Object(fields) => {
                let mut rec = Vec::with_capacity(fields.len());
                for (k, v) in fields {
                    let k = self
                        .id(k)
                        .ok_or_else(|| format!("key {k:?} is not interned"))?;
                    rec.push((k, self.value_of(v)?));
                }
                rec.sort_by_key(|(k, _)| *k);
                Value::Rec(Arc::new(rec))
            }
        })
    }

    /// Converts a model value back to JSON.
    pub(crate) fn json_of(&self, value: &Value) -> serde_json::Value {
        match value {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(*b),
            Value::Int(i) => serde_json::Value::from(*i),
            Value::Str(s) => serde_json::Value::String(self.name(*s).to_string()),
            Value::List(items) => {
                serde_json::Value::Array(items.iter().map(|v| self.json_of(v)).collect())
            }
            Value::Rec(fields) => serde_json::Value::Object(
                fields
                    .iter()
                    .map(|(k, v)| (self.name(*k).to_string(), self.json_of(v)))
                    .collect(),
            ),
        }
    }
}

/// Collects every string in a JSON document: object keys and string values.
pub(crate) fn collect_strings(json: &serde_json::Value, out: &mut Vec<String>) {
    match json {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(items) => {
            for v in items {
                collect_strings(v, out);
            }
        }
        serde_json::Value::Object(fields) => {
            for (k, v) in fields {
                out.push(k.clone());
                collect_strings(v, out);
            }
        }
        _ => {}
    }
}

/// Formats a value as compact JSON, for error messages.
pub(crate) struct Show<'a>(pub &'a Symbols, pub &'a Value);

impl fmt::Display for Show<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.json_of(self.1))
    }
}
