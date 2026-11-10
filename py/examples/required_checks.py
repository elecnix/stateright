"""A pull request with required status checks, as a declarative model.

One pull request has C required checks. Each check keeps a list of runs, and
each run records the head commit and the base it tested, and a status:
queued, running, passed, failed or superseded.

Events: a push moves the head and supersedes every queued or running run. A
retarget moves the base and leaves runs alone. A check queues a run on the
current head and base, a runner starts the oldest queued run, the running run
passes or fails, and a re-run queues a copy of a finished run, stale or not.

The invariant: the merge button is green only when every check has a passed
run on the current head and base. The `green` rule decides the button:

  correct     the newest run on the current head and base must have passed
  stale       any passed run counts, whatever it tested
  baseblind   the newest run on the current head must have passed, any base
  latest      the newest run must have passed, whatever it tested

The last three are seeded bugs; the checker finds a counterexample for each.

    python3 required_checks.py emit C H B R RERUN GREEN > model.json
    python3 required_checks.py reference C H B R RERUN GREEN

`reference` runs a plain Python breadth-first search over the same system,
written directly rather than through the model, for cross-checking counts.
C checks, H pushes, B retargets, at most R runs per check, RERUN 0 or 1.
"""

import json
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stateright_model import (  # noqa: E402
    Model, add, and_, append, eq, find_index, find_last, get, if_, in_, length, let, lt,
    map_, ne, obj, or_, range_, set_, some, all_, var, with_)

GREEN_RULES = ("correct", "stale", "baseblind", "latest")


def model(c, h, b, r, rerun, green):
    m = Model(
        f"required-checks C={c} H={h} B={b} R={r} rerun={rerun} green={green}",
        init=[{"head": 0, "base": 0, "runs": [[] for _ in range(c)]}],
        description="A pull request with required checks; see py/examples/required_checks.py.")
    runs_of = lambda ci: get(var("runs"), ci)  # noqa: E731
    checks = range_(0, length(var("runs")))

    m.action("push", guard=lt(var("head"), h), effects=[
        set_("head", add(var("head"), 1)),
        set_("runs", map_(var("runs"), "g", map_(var("g"), "run", if_(
            in_(get(var("run"), "st"), ["queued", "running"]),
            with_(var("run"), "st", "superseded"),
            var("run"))))),
    ])
    m.action("retarget", guard=lt(var("base"), b), effects=[
        set_("base", add(var("base"), 1)),
    ])
    m.action("queue", params=[("check", checks)],
             guard=lt(length(runs_of(var("check"))), r),
             effects=[append(["runs", var("check")],
                             obj(h=var("head"), b=var("base"), st="queued"))])
    m.action("start", params=[("check", checks)],
             lets=[("i", find_index(runs_of(var("check")), "run",
                                    eq(get(var("run"), "st"), "queued")))],
             guard=ne(var("i"), None),
             effects=[set_(["runs", var("check"), var("i"), "st"], "running")])
    m.action("conclude", params=[("check", checks), ("ok", [True, False])],
             lets=[("i", find_index(runs_of(var("check")), "run",
                                    eq(get(var("run"), "st"), "running")))],
             guard=ne(var("i"), None),
             effects=[set_(["runs", var("check"), var("i"), "st"],
                           if_(var("ok"), "passed", "failed"))])
    if rerun:
        m.action("rerun",
                 params=[("check", checks), ("i", range_(0, length(runs_of(var("check")))))],
                 lets=[("old", get(runs_of(var("check")), var("i")))],
                 guard=and_(lt(length(runs_of(var("check"))), r),
                            in_(get(var("old"), "st"), ["passed", "failed", "superseded"])),
                 effects=[append(["runs", var("check")],
                                 obj(h=get(var("old"), "h"), b=get(var("old"), "b"),
                                     st="queued"))])

    passed = lambda run: eq(get(run, "st"), "passed")  # noqa: E731
    on_head = lambda run: eq(get(run, "h"), var("head"))  # noqa: E731
    on_base = lambda run: eq(get(run, "b"), var("base"))  # noqa: E731
    if green == "stale":
        check_green = some(var("g"), "run", passed(var("run")))
    elif green == "latest":
        check_green = and_(ne(length(var("g")), 0), passed(get(var("g"), -1)))
    else:
        current = (on_head(var("run")) if green == "baseblind"
                   else and_(on_head(var("run")), on_base(var("run"))))
        check_green = let([("x", find_last(var("g"), "run", current))],
                          and_(ne(var("x"), None), passed(var("x"))))
    m.always("green only on a passed run for the current head and base", or_(
        {"not": all_(var("runs"), "g", check_green)},
        all_(var("runs"), "g", some(var("g"), "run", and_(
            on_head(var("run")), on_base(var("run")), passed(var("run"))))),
    ))
    return m


# The reference search: the same system written as plain Python.

Q, RUN, PASS, FAIL, SUP = range(5)


def successors(s, c, h, b, r, rerun):
    head, base, runs = s
    if head < h:
        yield ("push",), (head + 1, base, tuple(
            tuple((x, y, SUP if st in (Q, RUN) else st) for x, y, st in g) for g in runs))
    if base < b:
        yield ("retarget",), (head, base + 1, runs)
    for ci, g in enumerate(runs):
        put = lambda ng, ci=ci: runs[:ci] + (ng,) + runs[ci + 1:]  # noqa: E731
        if len(g) < r:
            yield ("queue", ci), (head, base, put(g + ((head, base, Q),)))
        for i, (x, y, st) in enumerate(g):
            if st == Q:
                yield ("start", ci), (head, base, put(g[:i] + ((x, y, RUN),) + g[i + 1:]))
                break
        for i, (x, y, st) in enumerate(g):
            if st == RUN:
                for ok in (True, False):
                    yield ("conclude", ci, ok), (head, base, put(
                        g[:i] + ((x, y, PASS if ok else FAIL),) + g[i + 1:]))
                break
        if rerun and len(g) < r:
            for i, (x, y, st) in enumerate(g):
                if st in (PASS, FAIL, SUP):
                    yield ("rerun", ci, i), (head, base, put(g + ((x, y, Q),)))


def holds(s, green):
    head, base, runs = s

    def check_green(g):
        if green == "stale":
            return any(st == PASS for _, _, st in g)
        if green == "latest":
            return bool(g) and g[-1][2] == PASS
        cur = [st for x, y, st in g if x == head and (green == "baseblind" or y == base)]
        return bool(cur) and cur[-1] == PASS

    if not all(check_green(g) for g in runs):
        return True
    return all(any(x == head and y == base and st == PASS for x, y, st in g) for g in runs)


def reference(c, h, b, r, rerun, green):
    """Returns the distinct state count and the first violating trace, or None."""
    init = (0, 0, tuple(() for _ in range(c)))
    parent = {init: None}
    q = deque([init])
    while q:
        s = q.popleft()
        if not holds(s, green):
            trace = []
            while parent[s] is not None:
                s, act = parent[s]
                trace.append(act)
            return len(parent), list(reversed(trace))
        for act, t in successors(s, c, h, b, r, rerun):
            if t not in parent:
                parent[t] = (s, act)
                q.append(t)
    return len(parent), None


def main(argv):
    if len(argv) != 7 or argv[0] not in ("emit", "reference") or argv[6] not in GREEN_RULES:
        sys.exit(__doc__)
    c, h, b, r, rerun = (int(x) for x in argv[1:6])
    if argv[0] == "emit":
        json.dump(model(c, h, b, r, rerun, argv[6]).to_json(), sys.stdout, indent=1)
        print()
    else:
        n, trace = reference(c, h, b, r, rerun, argv[6])
        print(json.dumps({"states": n, "trace": trace}))


if __name__ == "__main__":
    main(sys.argv[1:])
