use super::{CheckOptions, JsonModel, check};
use serde_json::json;

fn run(model: serde_json::Value, threads: usize) -> serde_json::Value {
    let model = JsonModel::from_json(&model).unwrap();
    check(
        model,
        &CheckOptions {
            threads,
            max_depth: None,
        },
    )
    .unwrap()
    .json
}

fn counter(limit: i64, invariant: serde_json::Value) -> serde_json::Value {
    json!({
        "stateright": 1,
        "name": "counter",
        "init": [{"n": 0, "log": []}],
        "actions": [
            {"name": "inc", "for": [["by", [1, 2]]],
             "guard": {"<": [{"var": "n"}, limit]},
             "effects": [
                 {"set": "n", "to": {"min": [{"+": [{"var": "n"}, {"var": "by"}]}, limit]}},
                 {"append": "log", "value": {"var": "by"}},
                 {"if": {">": [{"len": {"var": "log"}}, 2]},
                  "then": [{"remove": "log", "at": 0}]}
             ]}
        ],
        "invariants": [invariant]
    })
}

#[test]
fn counts_states_and_reports_ok() {
    // n in 0..=3, log holds the last two steps (each 1 or 2).
    let report = run(
        counter(
            3,
            json!({"name": "bounded", "always": {"<=": [{"var": "n"}, 3]}}),
        ),
        1,
    );
    assert_eq!(report["result"], "ok");
    assert_eq!(report["invariants"][0]["status"], "holds");
    assert_eq!(report["complete"], true);
    let n = report["states"].as_u64().unwrap();
    assert!(n > 4, "{report}");
}

#[test]
fn reports_the_shortest_counterexample() {
    let report = run(
        counter(
            5,
            json!({"name": "below 3", "always": {"<": [{"var": "n"}, 3]}}),
        ),
        1,
    );
    assert_eq!(report["result"], "violated");
    let trace = report["invariants"][0]["trace"].as_array().unwrap();
    // 0 -> 2 -> 3 is a shortest path to n >= 3; n = 4 needs two steps too,
    // and breadth-first order with by = 1 listed first reaches 3 first.
    assert_eq!(trace.len(), 3);
    assert_eq!(trace[0], json!({"state": {"n": 0, "log": []}}));
    assert_eq!(trace[2]["state"]["n"], 3);
    assert_eq!(
        trace[1]["action"],
        json!({"name": "inc", "args": {"by": 1}})
    );
}

#[test]
fn sometimes_reports_reached() {
    let report = run(
        counter(
            3,
            json!({"name": "reaches 3", "sometimes": {"==": [{"var": "n"}, 3]}}),
        ),
        2,
    );
    assert_eq!(report["result"], "ok");
    assert_eq!(report["invariants"][0]["status"], "reached");
}

fn transitions(max_depth: Option<u64>) -> serde_json::Value {
    // A clock that ticks forever. A tick from 2 breaks "no tick from 2", and
    // the system still reaches 3 and beyond.
    let mut m = json!({
        "stateright": 1,
        "init": [{"t": 0}],
        "actions": [{"name": "tick", "effects": [
            {"if": {"==": [{"var": "t"}, 2]}, "then": [{"violate": "no tick from 2"}]},
            {"set": "t", "to": {"+": [{"var": "t"}, 1]}}
        ]}],
        "invariants": [
            {"name": "no tick from 2", "transition": true},
            {"name": "reaches 4", "sometimes": {"==": [{"var": "t"}, 4]}}
        ]
    });
    if let Some(d) = max_depth {
        m["max_depth"] = json!(d);
    }
    m
}

#[test]
fn transition_invariants_report_the_step_and_keep_exploring() {
    let report = run(transitions(Some(5)), 1);
    assert_eq!(report["result"], "violated");
    let inv = &report["invariants"][0];
    assert_eq!(inv["kind"], "transition");
    assert_eq!(inv["status"], "violated");
    let trace = inv["trace"].as_array().unwrap();
    assert_eq!(trace.len(), 4);
    assert_eq!(trace[3]["state"]["t"], 3);
    assert_eq!(trace[3]["violates"], json!(["no tick from 2"]));
    // The violating step's target is still a state of the system.
    assert_eq!(report["invariants"][1]["status"], "reached");
}

#[test]
fn max_depth_bounds_the_search_and_counts_exactly() {
    // t = 0..=5 within five ticks; the marker state is not counted.
    let report = run(transitions(Some(5)), 4);
    assert_eq!(report["states"], 6);
    assert_eq!(report["depth_reached"], 5);
}

#[test]
fn expressions_cover_lists_records_and_sets() {
    let m = serde_json::from_str::<serde_json::Value>(
        r#"{
        "stateright": 1,
        "defs": {
            "double": {"params": ["x"], "body": {"*": [{"var": "x"}, 2]}}
        },
        "init": [{"xs": [3, 1, 2], "r": {"a": 1}, "s": ["b", "a"], "ok": false}],
        "actions": [{"name": "eval", "guard": {"not": {"var": "ok"}}, "effects": [
            {"let": "sorted", "be": {"sort_by": [{"var": "xs"}, "x", {"var": "x"}]}},
            {"require": {"==": [{"var": "sorted"}, [1, 2, 3]]}},
            {"require": {"==": [{"map": [{"var": "xs"}, ["x", "i"], {"+": [{"var": "x"}, {"var": "i"}]}]}, [3, 2, 4]]}},
            {"require": {"==": [{"filter": [{"range": [0, 6]}, "i", {"==": [{"%": [{"var": "i"}, 2]}, 0]}]}, [0, 2, 4]]}},
            {"require": {"==": [{"find_last_index": [{"var": "xs"}, "x", {"<": [{"var": "x"}, 3]}]}, 2]}},
            {"require": {"==": [{"find": [{"var": "xs"}, "x", {">": [{"var": "x"}, 5]}]}, null]}},
            {"require": {"==": [{"index_of": [{"var": "xs"}, 2]}, 2]}},
            {"require": {"==": [{"count": [{"var": "xs"}, "x", {">": [{"var": "x"}, 1]}]}, 2]}},
            {"require": {"==": [{"get": [{"var": "xs"}, -1]}, 2]}},
            {"require": {"==": [{"get": [{"var": "xs"}, 9, "none"]}, "none"]}},
            {"require": {"==": [{"slice": [{"var": "xs"}, 1]}, [1, 2]]}},
            {"require": {"==": [{"concat": [{"var": "xs"}, [4]]}, [3, 1, 2, 4]]}},
            {"require": {"==": [{"set_at": [{"var": "xs"}, 0, 9]}, [9, 1, 2]]}},
            {"require": {"==": [{"get": [{"with": [{"var": "r"}, "b", 2]}, "b"]}, 2]}},
            {"require": {"==": [{"obj": {"a": {"get": [{"var": "r"}, "a"]}}}, {"lit": {"a": 1}}]}},
            {"require": {"==": [{"union": [{"var": "s"}, ["c"]]}, ["a", "b", "c"]]}},
            {"require": {"==": [{"intersect": [{"var": "s"}, ["a", "z"]]}, ["a"]]}},
            {"require": {"==": [{"difference": [{"var": "s"}, ["a"]]}, ["b"]]}},
            {"require": {"subset": [["a"], {"var": "s"}]}},
            {"require": {"in": ["a", {"var": "s"}]}},
            {"require": {"==": [{"call": ["double", 21]}, 42]}},
            {"require": {"==": [{"/": [-7, 2]}, -4]}},
            {"require": {"==": [{"%": [-7, 2]}, 1]}},
            {"require": {"==": [{"if": [false, 1, {"==": [1, 2]}, 2, 3]}, 3]}},
            {"require": {"==": [{"let": [[["a", 1], ["b", {"+": [{"var": "a"}, 1]}]], {"var": "b"}]}, 2]}},
            {"set": ["r", "a"], "to": 5},
            {"clear": "xs"},
            {"set": "ok", "to": true}
        ]}],
        "invariants": [{"name": "evaluated", "sometimes": {"and": [
            {"var": "ok"}, {"==": [{"var": "xs"}, []]}, {"==": [{"get": [{"var": "r"}, "a"]}, 5]}
        ]}}]
    }"#,
    )
    .unwrap();
    let report = run(m, 1);
    assert_eq!(report["invariants"][0]["status"], "reached", "{report}");
}

#[test]
fn rejects_malformed_models_with_a_location() {
    let err = |m: serde_json::Value| JsonModel::from_json(&m).unwrap_err();
    let base = || json!({"stateright": 1, "init": [{"n": 0}], "actions": []});
    let mut m = base();
    m["stateright"] = json!(2);
    assert!(err(m).contains("schema version"));
    let mut m = base();
    m["actions"] = json!([{"name": "a", "guard": {"var": "missing"}}]);
    let e = err(m);
    assert!(
        e.contains("actions[0](a).guard") && e.contains("unknown name"),
        "{e}"
    );
    let mut m = base();
    m["actions"] = json!([{"name": "a", "effects": [{"violate": "nope"}]}]);
    assert!(err(m).contains("not declared as a transition invariant"));
    let mut m = base();
    m["defs"] = json!({
        "f": {"params": [], "body": {"call": ["g"]}},
        "g": {"params": [], "body": {"call": ["f"]}}
    });
    assert!(err(m).contains("calls itself"));
    let mut m = base();
    m["init"] = json!([{"n": 0}, {"m": 0}]);
    assert!(err(m).contains("init[1]"));
    let mut m = base();
    m["init"] = json!([{"n": 1.5}]);
    assert!(err(m).contains("64-bit integer"));
}

#[test]
fn type_errors_fail_the_check() {
    let m = json!({
        "stateright": 1,
        "init": [{"n": 0}],
        "actions": [{"name": "a", "guard": {"+": [{"var": "n"}, "x"]}}],
        "invariants": [{"name": "i", "always": true}]
    });
    let e = check(JsonModel::from_json(&m).unwrap(), &CheckOptions::default()).unwrap_err();
    assert!(
        e.contains("action \"a\"") && e.contains("expects a different type"),
        "{e}"
    );
}

#[test]
fn a_path_reads_records_that_keep_a_field_at_different_positions() {
    // "k" is the second field of the first record, the first of the second
    // and the third of the third, so no remembered position fits them all.
    let m = json!({
        "stateright": 1,
        "init": [{"xs": [{"a": 1, "k": 5}, {"k": 6}, {"b": 0, "c": 0, "k": 7}], "ok": false}],
        "actions": [{"name": "eval", "guard": {"not": {"var": "ok"}}, "effects": [
            {"require": {"==": [
                {"map": [{"range": [0, 3]}, "i", {"get": [{"get": [{"var": "xs"}, {"var": "i"}]}, "k"]}]},
                [5, 6, 7]]}},
            {"require": {"==": [
                {"map": [[2, 1, 0, 2], "i", {"get": [{"get": [{"var": "xs"}, {"var": "i"}]}, "k"]}]},
                [7, 6, 5, 7]]}},
            {"require": {"==": [{"map": [{"var": "xs"}, "x", {"get": [{"var": "x"}, "k"]}]}, [5, 6, 7]]}},
            {"set": "ok", "to": true}
        ]}],
        "invariants": [{"name": "evaluated", "sometimes": {"var": "ok"}}]
    });
    let report = run(m, 1);
    assert_eq!(report["invariants"][0]["status"], "reached", "{report}");
}

#[test]
fn conditions_answer_through_if_let_and_defs() {
    let m = json!({
        "stateright": 1,
        "defs": {
            "small": {"params": ["x"], "body": {"<": [{"var": "x"}, 3]}},
            "none": {"params": [], "body": {"==": [{"len": {"var": "xs"}}, 0]}}
        },
        "init": [{"n": 0, "xs": [1, 2, 3]}],
        "actions": [{"name": "inc",
            "guard": {"and": [
                {"if": [{"==": [{"var": "n"}, 0]}, true, {"call": ["small", {"var": "n"}]}]},
                {"let": [[["m", {"+": [{"var": "n"}, 1]}]], {"<=": [{"var": "m"}, 3]}]},
                {"not": {"call": ["none"]}},
                {"<": [{"-": [{"len": {"var": "xs"}}, 1]}, {"len": {"var": "xs"}}]},
                {"in": [{"+": [{"var": "n"}, 1]}, {"var": "xs"}]},
                {"some": [{"var": "xs"}, "x", {">": [{"var": "x"}, {"var": "n"}]}]}
            ]},
            "effects": [{"set": "n", "to": {"+": [{"var": "n"}, 1]}}]}],
        "invariants": [{"name": "small", "always": {"<=": [{"var": "n"}, 3]}}]
    });
    let report = run(m, 1);
    assert_eq!(report["result"], "ok", "{report}");
    assert_eq!(report["states"], 4);
}

#[test]
fn a_condition_that_is_not_a_boolean_fails_wherever_it_is_nested() {
    for guard in [
        json!({"if": [true, 1, false]}),
        json!({"let": [[["m", 1]], {"var": "m"}]}),
        json!({"call": ["one"]}),
        json!({"and": [true, {"if": [false, true, "x"]}]}),
    ] {
        let m = json!({
            "stateright": 1,
            "defs": {"one": {"params": [], "body": 1}},
            "init": [{"n": 0}],
            "actions": [{"name": "a", "guard": guard}],
            "invariants": [{"name": "i", "always": true}]
        });
        let e = check(JsonModel::from_json(&m).unwrap(), &CheckOptions::default()).unwrap_err();
        assert!(e.contains("expects a different type"), "{guard}: {e}");
    }
}

#[test]
fn comparisons_order_values_of_one_type_and_reject_mixed_ones() {
    let m = json!({
        "stateright": 1,
        "init": [{"n": 1, "s": "b", "ok": false}],
        "actions": [{"name": "eval", "guard": {"not": {"var": "ok"}}, "effects": [
            {"require": {"<": [{"var": "n"}, 2]}},
            {"require": {">=": [{"var": "n"}, 1]}},
            {"require": {"<": ["a", {"var": "s"}]}},
            {"require": {"<=": [[1, 2], [1, 3]]}},
            {"set": "ok", "to": {">": [{"var": "s"}, {"var": "n"}]}}
        ]}],
        "invariants": [{"name": "i", "always": true}]
    });
    let e = check(JsonModel::from_json(&m).unwrap(), &CheckOptions::default()).unwrap_err();
    assert!(e.contains("cannot order"), "{e}");
}
