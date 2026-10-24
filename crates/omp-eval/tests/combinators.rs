//! λ-RLM combinators + SERA rubrics (prelude_combinators.py) against a scripted host:
//! types checked at call time, termination, budget bounds, rubric-before-spawn order.

use std::sync::Mutex;

use omp_eval::js::{default_runtime, JsKernel};
use omp_eval::{CellStatus, ExecOptions, HostBridge, PythonKernel};
use serde_json::{json, Value};
mod common;

#[derive(Default)]
struct Host {
    calls: Mutex<Vec<(String, Value)>>,
    /// `__lam_plan__` behaviour: None = honest planner, Some(v) = canned reply, Err = refusal.
    plan: Mutex<Option<Result<Value, String>>>,
}

impl Host {
    fn ops(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|(o, _)| o.clone())
            .collect()
    }
    fn count(&self, op: &str) -> usize {
        self.ops().iter().filter(|o| *o == op).count()
    }
}

impl HostBridge for Host {
    fn call(&self, op: &str, args: Value) -> Result<Value, String> {
        self.calls
            .lock()
            .unwrap()
            .push((op.to_string(), args.clone()));
        match op {
            "__llm_query__" => {
                let p = args["prompt"].as_str().unwrap_or("");
                // Rubric scorer prompts are answered by prompt content; leaves echo their length.
                let text = if p.contains("scores a delegated subtask")
                    || p.contains("You score a delegated")
                {
                    // score candidate by its output text: "good" -> 1.0, otherwise 0.2
                    if p.contains("<output>\ngood") {
                        "{\"success\": true, \"scores\": [1.0, 1.0]}"
                    } else {
                        "{\"success\": true, \"scores\": [0.2, 0.2]}"
                    }
                    .to_string()
                } else if p.contains("independent attempt 2") {
                    "good".to_string()
                } else if p.contains("Select the task type") {
                    "aggregate".to_string()
                } else {
                    format!("leaf:{}", p.len())
                };
                Ok(json!({"text": text}))
            }
            "__lam_plan__" => match self.plan.lock().unwrap().clone() {
                Some(r) => r,
                None => {
                    let n = args["n"].as_u64().unwrap();
                    let w = args["window"].as_u64().unwrap();
                    let tau = args.get("tau").and_then(Value::as_u64).unwrap_or(w).min(w);
                    let extra = args["extra_calls"].as_u64().unwrap_or(0);
                    if n <= w {
                        return Ok(
                            json!({"n": n, "tau": tau, "k": 1, "depth": 0, "max_leaves": 1, "max_calls": 1, "direct": true}),
                        );
                    }
                    let k = args
                        .get("fanout")
                        .and_then(Value::as_u64)
                        .unwrap_or_else(|| n.div_ceil(tau))
                        .clamp(2, 64);
                    let (mut s, mut d) = (n, 0u64);
                    while s > tau {
                        s = s.div_ceil(k);
                        d += 1;
                    }
                    let leaves = k.pow(d as u32).min(n);
                    Ok(
                        json!({"n": n, "tau": tau, "k": k, "depth": d, "max_leaves": leaves, "max_calls": leaves + extra, "direct": false}),
                    )
                }
            },
            "__rubric_freeze__" => Ok(json!({"id": "rb-test", "criteria": [], "frozen_ms": 1})),
            "__rubric_score__" => {
                // Stand-in for the host op: score by output text and echo the verified outcome.
                let out = args["output"].as_str().unwrap_or("");
                let score = if out == "good" { 1.0 } else { 0.2 };
                Ok(
                    json!({"score": score, "verified": args.get("verified").cloned().unwrap_or(Value::Null), "gate": true}),
                )
            }
            "__rubric_rank__" => Ok(json!({"loss": 0.0})),
            other => Err(format!("unknown op {other}")),
        }
    }
    fn notify(&self, _op: &str, _args: Value) {}
}

fn kernel(tag: &str) -> PythonKernel {
    let dir = std::env::temp_dir().join(format!("omp-eval-comb-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    PythonKernel::start("python3", &dir).expect("kernel starts")
}

fn run(k: &PythonKernel, host: &Host, code: &str) -> String {
    let r = k
        .execute_with(
            code,
            ExecOptions {
                bridge: Some(host),
                ..Default::default()
            },
        )
        .expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{code}\n{}", r.output);
    r.output
}

#[test]
fn types_are_checked_at_call_time() {
    let (k, h) = (kernel("types"), Host::default());
    let out = run(
        &k,
        &h,
        r#"
def caught(f, *a):
    try:
        f(*a)
    except (LamTypeError, LamRangeError) as e:
        return type(e).__name__ + ": " + str(e)
    return "NO ERROR"
print(caught(lam_split, 5, 2))
print(caught(lam_split, "abc", "2"))
print(caught(lam_split, "abc", 0))
print(caught(lam_peek, "abc", 2, 1))
print(caught(lam_peek, "abc", 0, 9))
print(caught(lam_map, 3, [1]))
print(caught(lam_map, lambda x: x if x else "s", [1, 0]))
print(caught(lam_filter, lambda x: 1, [1]))
print(caught(lam_reduce, lambda a, b: str(a), [1, 2]))
print(caught(lam_reduce, lambda a, b: a, []))
print(caught(lam_concat, ["a", 2]))
print(caught(lam_cross, [1], "x"))
print(caught(lam_M, ""))
print(caught(lam_rlm, "text", ""))
print(caught(lam_rlm, "text", "q", "bogus"))
print(lam_split("abcdefg", 3), lam_peek("abcdef", 1, 3), lam_concat(lam_split("hello", 2)))
print(lam_cross([1, 2], ["a"]), lam_reduce(lambda a, b: a + b, [1, 2, 3]), lam_filter(lambda x: x > 1, [1, 2, 3]))
"#,
    );
    let lines: Vec<&str> = out.lines().collect();
    assert!(
        lines[0].starts_with("LamTypeError: split: s must be str")
            && lines[0].contains("signature str x int -> [str]"),
        "{out}"
    );
    assert!(
        lines[1].starts_with("LamTypeError: split: k must be int"),
        "{out}"
    );
    assert!(
        lines[2].starts_with("LamRangeError: split: k must be >= 1"),
        "{out}"
    );
    assert!(lines[3].starts_with("LamRangeError: peek"), "{out}");
    assert!(lines[4].starts_with("LamRangeError: peek"), "{out}");
    assert!(
        lines[5].starts_with("LamTypeError: map: f must be callable"),
        "{out}"
    );
    assert!(lines[6].contains("must be homogeneous"), "{out}");
    assert!(lines[7].contains("pred must return bool"), "{out}");
    assert!(lines[8].contains("op changed the type"), "{out}");
    assert!(
        lines[9].starts_with("LamRangeError: reduce: empty list"),
        "{out}"
    );
    assert!(lines[10].contains("elements must be str"), "{out}");
    assert!(lines[11].starts_with("LamTypeError: cross"), "{out}");
    assert!(
        lines[12].starts_with("LamTypeError: M: prompt must be a non-empty str"),
        "{out}"
    );
    assert!(lines[13].contains("query must be a non-empty str"), "{out}");
    assert!(lines[14].contains("task must be 'auto' or one of"), "{out}");
    assert_eq!(
        lines[15],
        "['abc', 'de', 'fg'] bc ['he', 'llo']".replace("['he', 'llo']", "hello"),
        "{out}"
    );
    assert_eq!(lines[16], "[(1, 'a'), (2, 'a')] 6 [2, 3]", "{out}");
    assert_eq!(
        h.count("__llm_query__"),
        0,
        "type errors must fire before any model call"
    );
}

#[test]
fn rlm_terminates_within_planned_bounds() {
    let (k, h) = (kernel("term"), Host::default());
    let out = run(
        &k,
        &h,
        r#"
P = "word " * 2000   # 10_000 chars
r = lam_rlm(P, "count the words", "classify", window=1000)
print(r.plan["depth"], r.plan["k"], r.calls, r.plan["max_calls"], r.task, r.count("leaf:"))
"#,
    );
    assert_eq!(out.trim(), "1 10 10 10 classify 10", "{out}");
    assert_eq!(h.count("__llm_query__"), 10);

    // A deeper plan: explicit fan-out forces depth > 1, still bounded.
    let out = run(
        &k,
        &h,
        r#"
r = lam_rlm("x" * 5000, "summarise it", "classify", window=10, fanout=3)
print(r.plan["depth"], r.calls <= r.plan["max_calls"])
"#,
    );
    assert!(out.starts_with("6 True"), "{out}");

    // A host plan that does not cover the input (depth 0 for a big input) is caught, not looped.
    *h.plan.lock().unwrap() = Some(Ok(
        json!({"n": 5000, "tau": 10, "k": 2, "depth": 0, "max_leaves": 1, "max_calls": 99, "direct": false}),
    ));
    let before = h.count("__llm_query__");
    let out = run(
        &k,
        &h,
        "try:\n    lam_rlm('x' * 5000, 'q', 'classify', window=10)\nexcept LamTerminationError as e:\n    print('stopped:', e)",
    );
    assert!(
        out.contains("stopped: lam: depth bound 0 exceeded"),
        "{out}"
    );
    assert_eq!(
        h.count("__llm_query__"),
        before,
        "no model call on a rejected plan"
    );
}

#[test]
fn budget_is_enforced_before_and_during_a_run() {
    let (k, h) = (kernel("budget"), Host::default());
    // Planner refusal (shared-ledger budget too small) surfaces with zero model calls.
    *h.plan.lock().unwrap() = Some(Err(
        "lam: plan needs up to 10 model calls but only 3 remain".into(),
    ));
    let out = run(
        &k,
        &h,
        "try:\n    lam_rlm('x' * 5000, 'q', 'classify', window=500)\nexcept Exception as e:\n    print(type(e).__name__, e)",
    );
    assert!(out.contains("only 3 remain"), "{out}");
    assert_eq!(h.count("__llm_query__"), 0);

    // A plan whose declared bound is smaller than what Φ needs trips LamBudgetError mid-run.
    *h.plan.lock().unwrap() = Some(Ok(
        json!({"n": 5000, "tau": 500, "k": 10, "depth": 1, "max_leaves": 10, "max_calls": 4, "direct": false}),
    ));
    let out = run(
        &k,
        &h,
        "try:\n    lam_rlm('x' * 5000, 'q', 'classify', window=500)\nexcept LamBudgetError as e:\n    print('bound:', e)",
    );
    assert!(out.contains("bound: M: planned bound of"), "{out}");
    assert_eq!(h.count("__llm_query__"), 4);
}

#[test]
fn auto_task_detection_is_one_budgeted_model_call() {
    let (k, h) = (kernel("auto"), Host::default());
    let out = run(&k, &h, "r = lam_rlm('item\\n' * 400, 'list the items', window=500)\nprint(r.task, r.calls, r.plan['max_calls'])");
    // 2000 chars / 500 => 4 leaves + 1 detection call; the declared bound (6) also reserves a synthesis call.
    assert_eq!(out.trim(), "aggregate 5 6", "{out}");
}

#[test]
fn rubric_is_frozen_before_the_child_and_default_path_is_untouched() {
    let (k, h) = (kernel("rubric"), Host::default());
    run(&k, &h, "a = llm_query('plain')");
    {
        let calls = h.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].1,
            json!({"prompt": "plain", "depth": 1}),
            "no-rubric path must be byte-identical"
        );
    }
    h.calls.lock().unwrap().clear();
    let out = run(
        &k,
        &h,
        r#"
r = llm_query("solve", rubric=[{"criterion": "correct", "weight": 2}, {"criterion": "short", "weight": 1}],
              n=3, verify=lambda t: t == "good")
print(str(r), r.score, [c["verified"] for c in r.candidates])
print(rubric_rank([{"score": c["score"], "verified": c["verified"]} for c in r.candidates])["loss"])
"#,
    );
    assert!(out.starts_with("good 1.0 [0.0, 1.0, 0.0]"), "{out}");
    let ops = h.ops();
    let freeze = ops.iter().position(|o| o == "__rubric_freeze__").unwrap();
    let first_child = ops.iter().position(|o| o == "__llm_query__").unwrap();
    assert!(
        freeze < first_child,
        "rubric must be committed before the first child call: {ops:?}"
    );
    assert_eq!(h.count("__rubric_score__"), 3);
    assert_eq!(h.count("__rubric_freeze__"), 1);
    let bad = k
        .execute_with(
            "llm_query('x', n=2)",
            ExecOptions {
                bridge: Some(&h),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(bad.status, CellStatus::Error);
}

#[test]
fn js_combinators_types_termination_and_rubric_order() {
    let dir = std::env::temp_dir().join(format!("omp-eval-comb-js-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    if !common::require_bun("js_combinators_types_termination_and_rubric_order") {
        return;
    }
    let k = JsKernel::start(default_runtime(), &dir).expect("JS kernel starts");
    let h = Host::default();
    let exec = |code: &str| {
        let r = k
            .execute_with(
                code,
                ExecOptions {
                    bridge: Some(&h),
                    ..Default::default()
                },
            )
            .expect("cell");
        assert_eq!(r.status, CellStatus::Ok, "{code}\n{}", r.output);
        r.output
    };
    let out = exec(
        r#"
const caught = async f => { try { await f(); return "NO ERROR"; } catch (e) { return e.name + ": " + e.message; } };
const a = await caught(() => lam_split(5, 2));
const b = await caught(() => lam_split("abc", 0));
const c = await caught(() => lam_filter(x => 1, [1]));
const d = await caught(() => lam_peek("abc", 0, 9));
JSON.stringify([a, b, c, d, lam_split("abcdefg", 3), lam_concat(lam_split("hello", 2))])
"#,
    );
    assert!(out.contains("LamTypeError: split: s must be str"), "{out}");
    assert!(
        out.contains("LamRangeError: split: k must be >= 1"),
        "{out}"
    );
    assert!(out.contains("pred must return bool"), "{out}");
    assert!(out.contains("LamRangeError: peek"), "{out}");
    assert!(
        out.contains("[\\\"abc\\\",\\\"de\\\",\\\"fg\\\"]") || out.contains("abc"),
        "{out}"
    );
    assert_eq!(h.count("__llm_query__"), 0);
    let out = exec(
        r#"
const r = await lam_rlm("word ".repeat(2000), "count", { task: "classify", window: 1000 });
`${r.plan.depth} ${r.plan.k} ${r.calls} ${r.plan.max_calls} ${r.task}`
"#,
    );
    assert!(out.contains("1 10 10 10 classify"), "{out}");
    h.calls.lock().unwrap().clear();
    let out = exec(
        r#"
const r = await llm_query("solve", { rubric: [{ criterion: "correct", weight: 2 }], n: 3, verify: t => t === "good" });
`${String(r)} ${r.score} ${JSON.stringify(r.candidates.map(c => c.verified))}`
"#,
    );
    assert!(out.contains("good 1 [0,1,0]"), "{out}");
    let ops = h.ops();
    assert!(
        ops.iter().position(|o| o == "__rubric_freeze__").unwrap()
            < ops.iter().position(|o| o == "__llm_query__").unwrap(),
        "{ops:?}"
    );
}
