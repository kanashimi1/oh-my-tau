//! Integration tests for the in-cell host bridge: `host` frames round-trip
//! through [`HostBridge`], idle-timeout pauses while the host works, cancel
//! interrupts a running cell, and the preludes install their helpers.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use omp_eval::js::{default_runtime, JsKernel};
use omp_eval::{CancelToken, CellStatus, ExecOptions, HostBridge, PythonKernel};
use serde_json::{json, Value};
mod common;

/// Records every call/notify; `echo` answers its args, `fail` errors,
/// `slow` sleeps before answering.
#[derive(Default)]
struct Recorder {
    calls: Mutex<Vec<(String, Value)>>,
    notes: Mutex<Vec<(String, Value)>>,
    slow: Duration,
}

impl HostBridge for Recorder {
    fn call(&self, op: &str, args: Value) -> Result<Value, String> {
        self.calls
            .lock()
            .unwrap()
            .push((op.to_string(), args.clone()));
        match op {
            "echo" => Ok(json!({ "echo": args })),
            "fail" => Err("nope from host".to_string()),
            "slow" => {
                std::thread::sleep(self.slow);
                Ok(json!("late"))
            }
            "write" => Ok(json!("Delivered.")),
            op if op.starts_with("__") => {
                Ok(json!({"status": "running", "cancelled": true, "spent": 5, "ids": ["wp#1"]}))
            }
            other => Err(format!("unknown op {other}")),
        }
    }
    fn notify(&self, op: &str, args: Value) {
        self.notes.lock().unwrap().push((op.to_string(), args));
    }
}

fn py(tag: &str) -> PythonKernel {
    let dir = std::env::temp_dir().join(format!("omp-eval-bridge-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("cwd");
    PythonKernel::start("python3", &dir).expect("kernel starts")
}

fn js(tag: &str) -> JsKernel {
    let dir = std::env::temp_dir().join(format!("omp-eval-bridge-js-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("cwd");
    JsKernel::start(default_runtime(), &dir).expect("JS kernel starts (bun on PATH)")
}

fn run<'a>(bridge: &'a dyn HostBridge) -> ExecOptions<'a> {
    ExecOptions {
        bridge: Some(bridge),
        ..Default::default()
    }
}

#[test]
fn py_host_call_round_trip_and_error() {
    let k = py("call");
    let b = Recorder::default();
    let r = k
        .execute_with("__omp_host_call('echo', {'x': 1})", run(&b))
        .expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{}", r.output);
    assert!(r.output.contains("'echo': {'x': 1}"), "{}", r.output);
    assert_eq!(b.calls.lock().unwrap()[0].0, "echo");

    let r = k
        .execute_with(
            "try:\n    __omp_host_call('fail', {})\nexcept OmpHostError as e:\n    print('caught', e)",
            run(&b),
        )
        .expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{}", r.output);
    assert!(r.output.contains("caught nope from host"), "{}", r.output);
}

#[test]
fn py_without_bridge_raises_in_cell() {
    let k = py("nobridge");
    let r = k
        .execute("__omp_host_call('echo', {})", None)
        .expect("cell");
    assert_eq!(r.status, CellStatus::Error);
    assert!(r.output.contains("host bridge unavailable"), "{}", r.output);
}

#[test]
fn py_notify_is_delivered() {
    let k = py("notify");
    let b = Recorder::default();
    let r = k
        .execute_with("__omp_host_notify('phase', {'title': 'x'})\n1", run(&b))
        .expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{}", r.output);
    assert_eq!(
        b.notes.lock().unwrap().as_slice(),
        &[("phase".to_string(), json!({"title": "x"}))]
    );
}

#[test]
fn py_idle_timeout_pauses_during_host_call_but_fires_on_busy_loop() {
    let k = py("idle");
    let b = Recorder {
        slow: Duration::from_millis(2500),
        ..Default::default()
    };
    let opts = ExecOptions {
        timeout: Some(Duration::from_secs(1)),
        idle_timeout: true,
        bridge: Some(&b),
        cancel: None,
    };
    let r = k
        .execute_with("__omp_host_call('slow', {})", opts)
        .expect("cell");
    assert_eq!(
        r.status,
        CellStatus::Ok,
        "host time must not count: {}",
        r.output
    );

    let opts = ExecOptions {
        timeout: Some(Duration::from_secs(1)),
        idle_timeout: true,
        bridge: Some(&b),
        cancel: None,
    };
    let r = k.execute_with("while True:\n    pass", opts).expect("cell");
    assert!(r.cancelled, "busy cell must time out: {:?}", r.status);
}

#[test]
fn py_cancel_interrupts_and_kernel_survives() {
    let k = Arc::new(py("cancel"));
    let token = CancelToken::new();
    let t2 = token.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(400));
        t2.cancel();
    });
    let started = Instant::now();
    let r = k
        .execute_with(
            "import time\ntime.sleep(20)",
            ExecOptions {
                cancel: Some(token),
                ..Default::default()
            },
        )
        .expect("cell");
    assert!(r.cancelled, "status {:?}", r.status);
    assert!(started.elapsed() < Duration::from_secs(10));
    let r2 = k.execute("1 + 1", None).expect("next cell");
    assert_eq!(r2.output.trim(), "2", "kernel survives an interrupt");
}

#[test]
fn py_preludes_install_helpers() {
    let k = py("prelude");
    let r = k
        .execute(
            "all(callable(f) for f in (read, write, log, phase, env, output, wait, completion, judge, agent)) and tool is not None",
            None,
        )
        .expect("cell");
    assert_eq!(r.output.trim(), "True", "{}", r.output);
}

/// Agent helpers are classes; a bare `__omp_host_call` in a method body is
/// name-mangled to `_Class__omp_host_call` and raised NameError.
#[test]
fn py_agent_helper_methods_reach_the_host() {
    let k = py("methods");
    let b = Recorder::default();
    let code = "h = AgentHandle('agt-1', 'task')\n\
print(h.status, h.cancel(), h.send('hi'))\n\
p = WorkPool('wp', 'task', 32)\n\
print(p.push('x'), p.status()['status'], p.peek()['status'], p.close()['status'])\n\
print(budget.spent(), budget.hard)";
    let r = k.execute_with(code, run(&b)).expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{}", r.output);
    assert!(r.output.contains("running True Delivered."), "{}", r.output);
    assert!(
        r.output.contains("['wp#1'] running running running"),
        "{}",
        r.output
    );
    assert!(r.output.contains("5 False"), "{}", r.output);
    let ops: Vec<String> = b
        .calls
        .lock()
        .unwrap()
        .iter()
        .map(|(op, _)| op.clone())
        .collect();
    assert_eq!(
        ops.iter().filter(|op| *op == "__workpool__").count(),
        4,
        "{ops:?}"
    );
}

#[test]
fn py_failed_cell_rolls_back_new_and_rebound_names() {
    // Default (no OMP_EVAL_TXN): a failed cell's new name is deleted, a
    // rebound name is restored to its previous object, untouched earlier
    // bindings survive.
    let k = py("txn");
    let r = k.execute("keep = 'original'\nkeep", None).expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{}", r.output);
    let r = k
        .execute(
            "keep = 'clobbered'\nnew_name = 1\nraise ValueError('x')",
            None,
        )
        .expect("cell");
    assert_eq!(r.status, CellStatus::Error);
    assert!(
        r.output.contains("[omp-txn] rolled back: keep, new_name"),
        "{}",
        r.output
    );
    // New name is gone ...
    let r = k.execute("'new_name' in globals()", None).expect("cell");
    assert_eq!(r.output.trim(), "False", "{}", r.output);
    // ... rebound name is restored to the earlier object ...
    let r = k.execute("keep", None).expect("cell");
    assert_eq!(r.output.trim(), "'original'", "{}", r.output);
    // ... and untouched earlier bindings survive.
    let r = k.execute("'keep' in globals()", None).expect("cell");
    assert_eq!(r.output.trim(), "True", "{}", r.output);
}

#[test]
fn py_omp_eval_txn_zero_keeps_ts_behavior() {
    // Opt-out: OMP_EVAL_TXN=0 keeps the TS semantics (a failed cell keeps
    // every binding it made).
    let dir = std::env::temp_dir().join(format!("omp-eval-bridge-txn0-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("cwd");
    let k = PythonKernel::start_with_env("python3", &dir, &[("OMP_EVAL_TXN".into(), "0".into())])
        .expect("kernel starts");
    let r = k
        .execute("a = 1\nraise ValueError('x')", None)
        .expect("cell");
    assert_eq!(r.status, CellStatus::Error);
    assert!(!r.output.contains("[omp-txn]"), "{}", r.output);
    let r = k.execute("a", None).expect("cell");
    assert_eq!(r.output.trim(), "1", "{}", r.output);
}

#[test]
fn js_host_call_round_trip_and_cancel() {
    if !common::require_bun("js_host_call_round_trip_and_cancel") {
        return;
    }
    let k = js("call");
    let b = Recorder::default();
    let r = k
        .execute_with("await __ompHostCall('echo', { x: 1 })", run(&b))
        .expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{}", r.output);
    assert!(r.output.contains("echo"), "{}", r.output);
    let r = k
        .execute_with(
            "let m = null\ntry { await __ompHostCall('fail', {}) } catch (e) { m = e.message }\nm",
            run(&b),
        )
        .expect("cell");
    assert!(r.output.contains("nope from host"), "{}", r.output);

    let token = CancelToken::new();
    let t2 = token.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(400));
        t2.cancel();
    });
    let r = k
        .execute_with(
            "await new Promise(r => setTimeout(r, 20000))",
            ExecOptions {
                cancel: Some(token),
                ..Default::default()
            },
        )
        .expect("cell");
    assert!(r.cancelled);
    assert_eq!(r.status, CellStatus::KernelKilled);
}

#[test]
fn js_failed_cell_rolls_back_new_global_names() {
    if !common::require_bun("js_failed_cell_rolls_back_new_global_names") {
        return;
    }
    let k = js("txn");
    let r = k
        .execute("globalThis.keep = 'original'\nkeep", None)
        .expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{}", r.output);
    let r = k
        .execute(
            "globalThis.keep = 'clobbered'\nglobalThis.newName = 1\nthrow new Error('x')",
            None,
        )
        .expect("cell");
    assert_eq!(r.status, CellStatus::Error);
    assert!(
        r.output.contains("[omp-txn] rolled back: keep, newName"),
        "{}",
        r.output
    );
    let r = k.execute("'newName' in globalThis", None).expect("cell");
    assert_eq!(r.output.trim(), "false", "{}", r.output);
    let r = k.execute("keep", None).expect("cell");
    assert_eq!(r.output.trim(), "\"original\"", "{}", r.output);
}

#[test]
fn js_preludes_install_helpers() {
    if !common::require_bun("js_preludes_install_helpers") {
        return;
    }
    let k = js("prelude");
    let r = k
        .execute(
            "[read, write, log, phase, env, output, wait, completion, judge, agent].every(f => typeof f === 'function') && typeof tool === 'function'",
            None,
        )
        .expect("cell");
    assert_eq!(r.output.trim(), "true", "{}", r.output);
}
