[![chat](https://img.shields.io/discord/781357978652901386)](https://discord.gg/JbxGSVP4A6)
[![crates.io](https://img.shields.io/crates/v/stateright.svg)](https://crates.io/crates/stateright)
[![docs.rs](https://docs.rs/stateright/badge.svg)](https://docs.rs/stateright)
[![LICENSE](https://img.shields.io/crates/l/stateright.svg)](https://github.com/stateright/stateright/blob/master/LICENSE)

Correctly implementing distributed algorithms such as the
[Paxos](https://en.wikipedia.org/wiki/Paxos_%28computer_science%29) and
[Raft](https://en.wikipedia.org/wiki/Raft_%28computer_science%29) consensus
protocols is notoriously difficult due to inherent nondetermism such as message
reordering by network devices. Stateright is a
[Rust](https://www.rust-lang.org/) actor library that aims to solve this
problem by providing an embedded [model
checker](https://en.wikipedia.org/wiki/Model_checking), a UI for exploring
system behavior ([demo](http://demo.stateright.rs:3000/)), and a lightweight
actor runtime. It also features a linearizability tester that can be run within
the model checker for more exhaustive test coverage than similar solutions such
as [Jepsen](https://jepsen.io/).

![Stateright Explorer screenshot](https://raw.githubusercontent.com/stateright/stateright/master/explorer.png)

## About this fork

This is elecnix's fork of [stateright/stateright](https://github.com/stateright/stateright).
It keeps the library API of the upstream 0.31.0 release and adds a
command-line checker for declarative JSON models. Releases carry prebuilt
binaries, so a CI job can check a model without a Rust toolchain.

### Code review key

Pull requests on this repository get an automated review from
[Cite](https://github.com/elecnix/cite), run by `.github/workflows/review.yml`.
That workflow reads three repository secrets: `MODEL_API_KEY`,
`MODEL_BASE_URL` and `MODEL_ID`. The key is dedicated to this repository. It
can send chat-completion requests to one staging model gateway and do nothing
else: it can't read or change gateway configuration, other keys, or any
production system. The workflow runs on `pull_request_target`, never checks
out pull request code, and skips pull requests from other forks, so a fork
can't spend the key.

Anyone who can push to this repository can edit the workflows that read
these secrets, and through them use the key. Today only the repository owner
has push access.


## Declarative JSON models

`stateright check MODEL.json` checks a transition system written as JSON,
with the same breadth-first engine as `spawn_bfs`. It needs no Rust
toolchain: the [releases](https://github.com/elecnix/stateright/releases)
carry prebuilt binaries for macOS arm64 and Linux x86_64. A Rust program can
load the same file with `stateright::declarative::JsonModel`.

```
stateright check [--threads N] [--max-depth N] [--output FILE] MODEL.json
```

The exit status is 0 when every `always` and `transition` invariant holds,
1 when one is violated, and 2 for a usage error or an error in the model.
With `--threads 1`, every trace is a shortest path, and two runs report the
same trace. With more threads, each trace is the shortest one found before
the search stopped.

[`py/stateright_model.py`](py/stateright_model.py) builds these documents
from Python. [`py/examples/required_checks.py`](py/examples/required_checks.py)
is a complete model: a pull request whose required checks run, pass, fail
and get re-run while the head and base move.

### The document

```json
{
  "stateright": 1,
  "name": "counter",
  "init": [{"n": 0}],
  "defs": {"next": {"params": ["x"], "body": {"+": [{"var": "x"}, 1]}}},
  "actions": [
    {"name": "inc", "for": [["by", [1, 2]]],
     "guard": {"<": [{"var": "n"}, 5]},
     "effects": [{"set": "n", "to": {"+": [{"var": "n"}, {"var": "by"}]}}]}
  ],
  "normalize": [],
  "invariants": [{"name": "small", "always": {"<=": [{"var": "n"}, 6]}}],
  "max_depth": 10
}
```

| Key | Meaning |
|---|---|
| `stateright` | Schema version. This crate reads `1`. |
| `name`, `description` | Text for the report. |
| `init` | Initial states. Each is a JSON object, and every one has the same fields. Those fields are the state; actions change their values and never add or remove a field. |
| `defs` | Named functions: `params` and an expression `body`. A def can call another def, and recursion is rejected. |
| `actions` | Guarded updates, described below. |
| `normalize` | Effects run after every action, before the checker compares states. Use it to put states in a canonical form, such as sorting a list whose order means nothing. |
| `invariants` | Up to 64 properties, described below. |
| `max_depth` | Optional. Explore at most this many actions from an initial state. Without it, the model's own guards must keep the state space finite. |

Values are JSON values: `null`, booleans, 64-bit integers, strings, arrays
and objects. Floating point numbers are rejected. A set is an array; the set
operations sort their result and drop duplicates, so two equal sets compare
equal.

### Actions

| Key | Meaning |
|---|---|
| `name` | Shown in traces. |
| `for` | Parameters: `[["name", list], ...]`. The action runs once per combination, in order. A later parameter's list can use the earlier ones, as in `[["i", {"range": [0, {"len": {"var": "xs"}}]}]]`. |
| `let` | Names bound in order, before the guard: `[["name", expression], ...]`. |
| `guard` | A boolean expression. The action is enabled when it is true. |
| `effects` | Updates, applied in order. Each effect sees the state as the effects before it left it. |

Effects:

| Effect | Meaning |
|---|---|
| `{"set": PATH, "to": E}` | Sets the value at PATH. |
| `{"append": PATH, "value": E}` | Appends to the array at PATH. |
| `{"clear": PATH}` | Empties the array at PATH, or sets any other value to `null`. |
| `{"remove": PATH, "at": E}` | Removes one element of the array at PATH. |
| `{"if": E, "then": [...], "else": [...]}` | Runs one branch. `else` is optional. |
| `{"let": "name", "be": E}` | Binds a name for the effects after it. |
| `{"require": E}` | Disables the action when E is false, discarding the effects so far. |
| `{"violate": "name"}` | Reports that this step breaks the transition invariant `name`. |

A PATH is a field name, or an array that starts with a field name and
continues with keys and indexes: `["runs", {"var": "i"}, "status"]`.

### Expressions

An expression is a literal, an array of expressions (which builds an
array), or an object with one key that names an operation. Strings are
literals. `{"var": "x"}` reads a local name (a parameter, a `let`, a def
parameter or an iteration variable) or else a state field. `{"lit": V}`
is V as it is, and `{"obj": {"k": E}}` builds an object.

| Operations | Meaning |
|---|---|
| `and`, `or`, `not` | Booleans, short-circuiting. |
| `if` | `[c, a, b]`, or `[c1, a1, c2, a2, ..., else]`. |
| `==`, `!=` | Deep equality of any two values. |
| `<`, `<=`, `>`, `>=` | Ordering of two values of one type. |
| `+`, `*`, `min`, `max` | Integers, two or more arguments. |
| `-`, `/`, `%` | Integers, two arguments. Division rounds down, and `%` takes the sign of the divisor. Overflow is an error. |
| `get` | `[x, key]` reads an object key or an array index (negative counts from the end). `[x, key, default]` returns the default when the key is missing. |
| `len` | Length of an array or object. |
| `range` | `[a, b]` is the integers from a up to b, excluding b. |
| `slice` | `[xs, from]` or `[xs, from, to]`, with Python's rules for negative and out-of-range bounds. |
| `concat` | Joins arrays. |
| `set_at`, `with` | `[xs, i, v]` and `[obj, key, v]` return a copy with one element replaced. |
| `in`, `index_of` | `[x, xs]` membership; `[xs, x]` is the first index of x, or `null`. |
| `union`, `intersect`, `difference`, `subset` | Sets. |
| `map`, `filter`, `some`, `all`, `count`, `find`, `find_last`, `find_index`, `find_last_index`, `sort_by` | `[xs, binder, body]`. The binder is a name for the element, or `["x", "i"]` for the element and its index. The `find` operations return `null` when nothing matches. `sort_by` is stable. |
| `let` | `[[["a", E1], ["b", E2]], body]`, bound in order. |
| `call` | `["def name", arguments...]`. |

A type error, a missing key or an overflow is an error in the model. The
check stops with exit status 2 and names the action or invariant.

### Invariants

| Form | Meaning |
|---|---|
| `{"name": N, "always": E}` | E holds in every reachable state. The report gives a shortest trace to a state where it fails. |
| `{"name": N, "sometimes": E}` | Some reachable state satisfies E. The report gives a trace to one. |
| `{"name": N, "transition": true}` | A property of steps. An action reports a violation with `{"violate": N}`, and the report gives a trace that ends with that step. The step still happens: the search continues from its target. |

### The report

```json
{
  "model": "counter",
  "result": "ok",
  "states": 7,
  "max_depth": 10,
  "depth_reached": 3,
  "threads": 1,
  "seconds": 0.001,
  "complete": true,
  "invariants": [
    {"name": "small", "kind": "always", "status": "holds"}
  ]
}
```

`result` is `ok` or `violated`. `states` counts distinct states. `complete`
is false when the search stopped early because every invariant already had
a result. A trace is a list of steps. The first step has only the initial
`state`. Each later step has the `action` taken (`{"name", "args"}`) and the
`state` it led to. A step that breaks a transition invariant lists it under
`violates`.

### Conformance and speed

CI runs [`py/conformance.py`](py/conformance.py). It checks the
required-checks example through the binary and compares the result with a
plain Python breadth-first search of the same system. The distinct state
counts match (4,756, 15,965, 3,752,047 and 7,585,107 states), and for each
seeded bug the counterexample matches step for step.

The JSON route is slower than a model written in Rust. On the 3,752,047-state
configuration with 8 threads, it took 1.5 to 2.6 s, against 0.6 s for the
same model as a Rust `Model`. A breadth-first search in Python took 11.5 s on
one core.

## Getting Started

1. **Please see the book, "[Building Distributed Systems With
   Stateright](https://www.stateright.rs)."**
2. A [video
   introduction](https://youtube.com/playlist?list=PLUhyBsVvEJjaF1VpNhLRfIA4E7CFPirmz)
   is also available.
3. Stateright also has detailed [API docs](https://docs.rs/stateright/).
4. Consider also joining the [Stateright Discord
   server](https://discord.gg/JbxGSVP4A6) for Q&A or other feedback.

## Examples

Stateright includes a variety of
[examples](https://github.com/stateright/stateright/tree/master/examples), such
as a [Single Decree Paxos
cluster](https://github.com/stateright/stateright/blob/master/examples/paxos.rs)
and an [abstract two phase commit
model](https://github.com/stateright/stateright/blob/master/examples/2pc.rs).

Passing a `check` CLI argument causes each example to validate itself using
Stateright's model checker:

```sh
# Two phase commit with 3 resource managers.
cargo run --release --example 2pc check 3
# Paxos cluster with 3 clients.
cargo run --release --example paxos check 3
# Single-copy (unreplicated) register with 3 clients.
cargo run --release --example single-copy-register check 3
# Linearizable distributed register (ABD algorithm) with 2 clients
# assuming ordered channels between actors.
cargo run --release --example linearizable-register check 2 ordered
```

Passing an `explore` CLI argument causes each example to spin up the Stateright
Explorer web UI locally on port 3000, allowing you to browse system behaviors:

```sh
cargo run --release --example 2pc explore
cargo run --release --example paxos explore
cargo run --release --example single-copy-register explore
cargo run --release --example linearizable-register explore
```

Passing a `spawn` CLI argument to the examples leveraging the actor
functionality will cause each to spawn actors using the included runtime,
transmitting JSON messages over UDP:

```sh
cargo run --release --example paxos spawn
cargo run --release --example single-copy-register spawn
cargo run --release --example linearizable-register spawn
```

The `bench.sh` script runs all the examples with various settings for
benchmarking the performance impact of changes to the library.

```sh
./bench.sh
```

# Features

Stateright contains a general purpose model checker offering:

- Invariant checks via "always" properties.
- Nontriviality checks via "sometimes" properties.
- Liveness checks via "eventually" properties (experimental/incomplete).
- A web browser UI for interactively exploring state space.
- [Linearizability](https://en.wikipedia.org/wiki/Linearizability)
  and [sequential consistency](https://en.wikipedia.org/wiki/Sequential_consistency)
  testers.
- Support for symmetry reduction to reduce state spaces.

Stateright's actor system features include:

- An actor runtime that can execute actors outside the model checker in the
  "real world."
- A model for lossy/lossless duplicating/non-duplicating networks with the
  ability to capture actor message
  [history](https://lamport.azurewebsites.net/tla/auxiliary/auxiliary.html) to
  check an actor system against an expected consistency model.
- Pluggable network semantics for model checking, allowing you to choose
  between fewer assumptions (e.g. "lossy unordered duplicating") or more
  assumptions (speeding up model checking; e.g. "lossless ordered").
- An optional network adapter that provides a lossless non-duplicating ordered
  virtual channel for messages between a pair of actors.

In contrast with other actor libraries, Stateright enables you to [formally
verify](https://en.wikipedia.org/wiki/Formal_verification) the correctness of
your implementation, and in contrast with model checkers such as TLC for
[TLA+](https://lamport.azurewebsites.net/tla/tla.html), systems implemented
using Stateright can also be run on a real network without being reimplemented
in a different language.

## Contribution

Contributions are welcome! Please [fork the
library](https://github.com/stateright/stateright/fork), push changes to your
fork, and send a [pull
request](https://help.github.com/articles/creating-a-pull-request-from-a-fork/).
All contributions are shared under an MIT license unless explicitly stated
otherwise in the pull request.

## License

Stateright is copyright 2018 Jonathan Nadal and other
[contributors](https://github.com/stateright/stateright/graphs/contributors).
It is made available under the MIT License.

To avoid the need for a Javascript package manager, the Stateright repository
includes code for the following Javascript dependency used by Stateright
Explorer:

- [KnockoutJS](https://knockoutjs.com/) is copyright 2010 Steven Sanderson, the
  Knockout.js team, and other contributors. It is made available under the MIT
  License.
