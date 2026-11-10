"""Conformance: `stateright check` against a plain Python search.

Runs the required-checks example (py/examples/required_checks.py) through the
binary and compares each result with the Python reference search in the same
file: the distinct state count, and for a seeded bug the counterexample, step
by step. The largest configurations compare against counts the reference
search measured once, because it takes about a minute on them.

    python3 py/conformance.py target/release/stateright [--large]
"""

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "examples"))
import required_checks as rc  # noqa: E402

# (C, H, B, R, RERUN, GREEN, states from the reference search or None to run it)
CASES = [
    (1, 3, 0, 4, 0, "correct", None),
    (1, 3, 0, 4, 0, "stale", None),
    (1, 2, 1, 3, 1, "baseblind", None),
    (1, 2, 0, 3, 1, "latest", None),
    (2, 1, 1, 2, 1, "correct", None),
    (2, 2, 1, 3, 1, "correct", 3_752_047),
]
LARGE = [
    (3, 2, 1, 2, 1, "correct", 7_585_107),
]

ACTION_ARGS = {"push": [], "retarget": [], "queue": ["check"], "start": ["check"],
               "conclude": ["check", "ok"], "rerun": ["check", "i"]}


def as_reference(action):
    """Converts a report action to the reference search's label."""
    return [action["name"], *(action["args"][a] for a in ACTION_ARGS[action["name"]])]


def run(binary, case, tmp):
    c, h, b, r, rerun, green, known = case
    path = os.path.join(tmp, "model.json")
    with open(path, "w") as f:
        json.dump(rc.model(c, h, b, r, rerun, green).to_json(), f)
    # One thread: breadth-first search then returns a shortest trace, and
    # the reference search breaks ties between shortest traces the same way.
    t0 = time.perf_counter()
    out = subprocess.run([binary, "check", "--threads", "1", path],
                         capture_output=True, text=True)
    seconds = time.perf_counter() - t0
    if out.returncode not in (0, 1):
        raise SystemExit(f"{case}: exit {out.returncode}\n{out.stderr}")
    report = json.loads(out.stdout)
    inv = report["invariants"][0]
    trace = None
    if inv["status"] == "violated":
        trace = [as_reference(step["action"]) for step in inv["trace"][1:]]
    if known is not None and green == "correct":
        expected = (known, None)
    else:
        n, ref_trace = rc.reference(c, h, b, r, rerun, green)
        expected = (n if ref_trace is None else None,
                    None if ref_trace is None else [list(a) for a in ref_trace])
    ok = True
    if expected[1] is None:
        ok = trace is None and report["states"] == expected[0]
    else:
        ok = trace == expected[1]
    label = f"C={c} H={h} B={b} R={r} rerun={rerun} green={green}"
    detail = (f"{report['states']} states" if trace is None
              else f"{len(trace)}-step counterexample")
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {detail} in {seconds:.2f} s")
    if not ok:
        print(f"     expected {expected}, got states={report['states']} trace={trace}")
    return ok


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    binary = os.path.abspath(argv[0])
    cases = CASES + (LARGE if "--large" in argv else [])
    with tempfile.TemporaryDirectory() as tmp:
        results = [run(binary, case, tmp) for case in cases]
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
