"""Helpers for writing declarative Stateright models from Python.

A model is a JSON document that `stateright check` reads. These helpers build
the expression and effect objects, so a model reads like code:

    from stateright_model import *
    m = Model("counter", init=[{"n": 0}])
    m.action("inc", guard=lt(var("n"), 3), effects=[set_("n", add(var("n"), 1))])
    m.always("small", le(var("n"), 3))
    m.write("counter.json")

Every helper returns plain dicts and lists; nothing here evaluates a model.
The schema is documented in the README, under "Declarative JSON models".
"""

import json

SCHEMA_VERSION = 1


def var(name):
    return {"var": name}


def lit(value):
    return {"lit": value}


def obj(**fields):
    return {"obj": fields}


def rec(fields):
    """A record from a dict, for keys that are not Python identifiers."""
    return {"obj": dict(fields)}


def get(x, key, *default):
    return {"get": [x, key, *default]}


def length(x):
    return {"len": x}


def if_(*branches):
    return {"if": list(branches)}


def and_(*xs):
    return {"and": list(xs)}


def or_(*xs):
    return {"or": list(xs)}


def not_(x):
    return {"not": x}


def _binary(op):
    return lambda a, b: {op: [a, b]}


eq, ne, lt, le, gt, ge = (_binary(op) for op in ("==", "!=", "<", "<=", ">", ">="))
in_, range_, subset = _binary("in"), _binary("range"), _binary("subset")
union, intersect, difference = _binary("union"), _binary("intersect"), _binary("difference")
sub, div, mod = _binary("-"), _binary("/"), _binary("%")


def add(*xs):
    return {"+": list(xs)}


def mul(*xs):
    return {"*": list(xs)}


def min_(*xs):
    return {"min": list(xs)}


def max_(*xs):
    return {"max": list(xs)}


def slice_(x, start, *end):
    return {"slice": [x, start, *end]}


def concat(*xs):
    return {"concat": list(xs)}


def set_at(x, i, v):
    return {"set_at": [x, i, v]}


def with_(r, k, v):
    return {"with": [r, k, v]}


def _iter(op):
    def build(xs, binder, body):
        return {op: [xs, binder, body]}
    return build


map_, filter_, some, all_, count = (_iter(op) for op in ("map", "filter", "some", "all", "count"))
find, find_last, find_index, find_last_index, sort_by = (
    _iter(op) for op in ("find", "find_last", "find_index", "find_last_index", "sort_by"))


def let(bindings, body):
    """`bindings` is a list of (name, value) pairs, bound in order."""
    return {"let": [[[n, v] for n, v in bindings], body]}


def call(name, *args):
    return {"call": [name, *args]}


# Effects


def set_(path, value):
    return {"set": path, "to": value}


def append(path, value):
    return {"append": path, "value": value}


def clear(path):
    return {"clear": path}


def remove(path, at):
    return {"remove": path, "at": at}


def when(cond, then, otherwise=()):
    e = {"if": cond, "then": list(then)}
    if otherwise:
        e["else"] = list(otherwise)
    return e


def bind(name, value):
    return {"let": name, "be": value}


def require(cond):
    return {"require": cond}


def violate(name):
    return {"violate": name}


class Model:
    """A model under construction. `to_json` returns the document."""

    def __init__(self, name, init, max_depth=None, description=None):
        self.doc = {"stateright": SCHEMA_VERSION, "name": name}
        if description:
            self.doc["description"] = description
        self.doc["init"] = list(init)
        self.doc["defs"] = {}
        self.doc["actions"] = []
        self.doc["normalize"] = []
        self.doc["invariants"] = []
        if max_depth is not None:
            self.doc["max_depth"] = max_depth

    def define(self, name, params, body):
        self.doc["defs"][name] = {"params": list(params), "body": body}

    def action(self, name, params=(), lets=(), guard=None, effects=(), normalize=True):
        """`normalize=False` skips the model's normalize effects after this
        action. Use it only when the action keeps a normalized state
        normalized."""
        a = {"name": name}
        if params:
            a["for"] = [[n, d] for n, d in params]
        if lets:
            a["let"] = [[n, v] for n, v in lets]
        if guard is not None:
            a["guard"] = guard
        a["effects"] = list(effects)
        if not normalize:
            a["normalize"] = False
        self.doc["actions"].append(a)

    def normalize(self, *effects):
        self.doc["normalize"].extend(effects)

    def always(self, name, expr):
        self.doc["invariants"].append({"name": name, "always": expr})

    def sometimes(self, name, expr):
        self.doc["invariants"].append({"name": name, "sometimes": expr})

    def transition(self, name):
        self.doc["invariants"].append({"name": name, "transition": True})

    def to_json(self):
        doc = {k: v for k, v in self.doc.items() if v not in ({}, [])}
        doc.setdefault("actions", [])
        return doc

    def write(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)
            f.write("\n")
