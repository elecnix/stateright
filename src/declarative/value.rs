//! Runtime values of a declarative model.

use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// An interned string. Symbols are numbered in lexical order of their text, so
/// comparing two symbols compares their strings.
pub type Sym = u32;

/// A value in a declarative model state.
///
/// Lists and records share their contents through [`Arc`], so copying a state
/// copies pointers, and an update copies only the path it changes.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
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
    List(Arc<Hashed<Value>>),
    /// A JSON object. Fields are sorted by symbol, which is lexical order.
    Rec(Arc<Hashed<(Sym, Value)>>),
}

/// The items of a list or a record, with their hash remembered.
///
/// A successor state shares most of its lists and records with the state it
/// came from. Each one computes its hash once, so fingerprinting a state
/// hashes only what the step changed, and two values with different
/// remembered hashes compare unequal without a walk.
pub struct Hashed<T> {
    /// The hash of `items`, or zero when it has not been computed.
    hash: AtomicU64,
    items: Vec<T>,
}

impl<T> Hashed<T> {
    /// Wraps the items of a new list or record.
    pub fn new(items: Vec<T>) -> Self {
        Hashed {
            hash: AtomicU64::new(0),
            items,
        }
    }

    /// The items, for a change in place. Copies them first when another
    /// value shares them, and forgets the hash either way.
    pub fn make_mut(this: &mut Arc<Self>) -> &mut Vec<T>
    where
        T: Clone,
    {
        let unique = Arc::make_mut(this);
        *unique.hash.get_mut() = 0;
        &mut unique.items
    }
}

impl<T: Hash> Hashed<T> {
    /// The hash of the items: computed on the first call, read afterwards.
    /// Threads may race to compute it, and they all store the same value.
    fn hash_value(&self) -> u64 {
        let known = self.hash.load(Ordering::Relaxed);
        if known != 0 {
            return known;
        }
        let mut hasher = crate::stable::hasher();
        hasher.write_usize(self.items.len());
        for item in &self.items {
            item.hash(&mut hasher);
        }
        // Zero means "not computed", so a hash of zero is stored as one.
        let hash = hasher.finish().max(1);
        self.hash.store(hash, Ordering::Relaxed);
        hash
    }
}

impl<T> std::ops::Deref for Hashed<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.items
    }
}

impl<T: Clone> Clone for Hashed<T> {
    fn clone(&self) -> Self {
        Hashed::new(self.items.clone())
    }
}

impl<T: fmt::Debug> fmt::Debug for Hashed<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.items.fmt(f)
    }
}

impl<T: PartialEq> PartialEq for Hashed<T> {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (
            self.hash.load(Ordering::Relaxed),
            other.hash.load(Ordering::Relaxed),
        );
        (a == 0 || b == 0 || a == b) && self.items == other.items
    }
}

impl<T: Eq> Eq for Hashed<T> {}

impl<T: PartialOrd> PartialOrd for Hashed<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.items.partial_cmp(&other.items)
    }
}

impl<T: Ord> Ord for Hashed<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.items.cmp(&other.items)
    }
}

/// Hashes each value with one write: a tag in the high word and the payload
/// in the low word. The checker fingerprints every successor state, so this
/// is on the hot path. A list or a record writes the hash its items remember.
/// It agrees with `Eq`: equal values write equal words.
impl Hash for Value {
    fn hash<H: Hasher>(&self, h: &mut H) {
        let word = |tag: u128, payload: u64| (tag << 64) | payload as u128;
        match self {
            Value::Null => h.write_u128(word(0, 0)),
            Value::Bool(b) => h.write_u128(word(1, *b as u64)),
            Value::Int(i) => h.write_u128(word(2, *i as u64)),
            Value::Str(s) => h.write_u128(word(3, *s as u64)),
            Value::List(items) => h.write_u128(word(4, items.hash_value())),
            Value::Rec(fields) => h.write_u128(word(5, fields.hash_value())),
        }
    }
}

impl Value {
    pub(crate) fn list(items: Vec<Value>) -> Value {
        Value::List(Arc::new(Hashed::new(items)))
    }

    pub(crate) fn rec(fields: Vec<(Sym, Value)>) -> Value {
        Value::Rec(Arc::new(Hashed::new(fields)))
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

    pub(crate) fn as_list(&self) -> Option<&Arc<Hashed<Value>>> {
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
                Value::rec(rec)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_of(value: &Value) -> u64 {
        let mut hasher = crate::stable::hasher();
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn ints(items: &[i64]) -> Value {
        Value::list(items.iter().map(|i| Value::Int(*i)).collect())
    }

    #[test]
    fn a_change_in_place_forgets_the_remembered_hash() {
        let mut list = Value::list(vec![Value::Int(1), ints(&[2, 3])]);
        let before = hash_of(&list);
        let Value::List(items) = &mut list else {
            unreachable!()
        };
        // The change is two levels down, so both levels must forget.
        let Value::List(inner) = &mut Hashed::make_mut(items)[1] else {
            unreachable!()
        };
        Hashed::make_mut(inner)[0] = Value::Int(7);
        let rebuilt = Value::list(vec![Value::Int(1), ints(&[7, 3])]);
        assert_ne!(hash_of(&list), before);
        assert_eq!(hash_of(&list), hash_of(&rebuilt));
        assert_eq!(list, rebuilt);
    }

    #[test]
    fn a_shared_value_keeps_its_hash_when_a_copy_changes() {
        let original = ints(&[1, 2]);
        let before = hash_of(&original);
        let mut copy = original.clone();
        let Value::List(items) = &mut copy else {
            unreachable!()
        };
        Hashed::make_mut(items).push(Value::Int(3));
        assert_eq!(hash_of(&original), before);
        assert_eq!(original, ints(&[1, 2]));
        assert_eq!(hash_of(&copy), hash_of(&ints(&[1, 2, 3])));
    }

    #[test]
    fn equality_does_not_depend_on_which_hashes_are_remembered() {
        let (a, b, c) = (ints(&[1, 2]), ints(&[1, 2]), ints(&[1, 3]));
        // Neither side hashed, then one side, then both.
        assert_eq!(a, b);
        assert_ne!(a, c);
        hash_of(&a);
        assert_eq!(a, b);
        assert_ne!(a, c);
        hash_of(&b);
        hash_of(&c);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a < c);
    }

    #[test]
    fn a_record_and_a_list_with_the_same_items_hash_apart() {
        let list = Value::list(vec![]);
        let record = Value::rec(vec![]);
        assert_ne!(hash_of(&list), hash_of(&record));
    }
}
