//! Integration tests: concurrent dispatch of independent read-only host
//! frames (JS `Promise.all` over bridge calls) — wave parallelism proof,
//! serial barriers for mutating ops, reply correlation under reversed
//! completion, idle-deadline and cancel behavior, output order, and batch-cap
//! stress. All bun-gated like the rest of the JS kernel tests.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use omp_eval::js::{default_runtime, JsKernel};
use omp_eval::{CancelToken, CellStatus, ExecOptions, HostBridge};
use serde_json::{json, Value};
mod common;

fn kernel(tag: &str) -> JsKernel {
    let dir = std::env::temp_dir().join(format!("omp-eval-js-par-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("cwd");
    JsKernel::start(default_runtime(), &dir).expect("kernel starts; bun must be on PATH")
}

/// Probe bridge: counts concurrent reads/writes, optionally waits for a full
/// read wave (a barrier that only a parallel dispatch can satisfy), supports
/// per-path delays via `d<N>` paths, and flags any read/write overlap.
struct ProbeBridge {
    read_delay: Duration,
    write_delay: Duration,
    /// Barrier: reads block until this many reads are in flight (`0` = off).
    expect_read_wave: usize,
    reads_in_flight: AtomicUsize,
    max_reads_in_flight: AtomicUsize,
    wave_hits: AtomicUsize,
    overlap_violation: AtomicBool,
    /// Completion log `(op, path)` in finish order (correlation checks).
    finished: Mutex<Vec<(String, String)>>,
}

impl ProbeBridge {
    fn new(read_delay: Duration) -> Self {
        Self {
            read_delay,
            write_delay: read_delay,
            expect_read_wave: 0,
            reads_in_flight: AtomicUsize::new(0),
            max_reads_in_flight: AtomicUsize::new(0),
            wave_hits: AtomicUsize::new(0),
            overlap_violation: AtomicBool::new(false),
            finished: Mutex::new(Vec::new()),
        }
    }

    fn with_wave(mut self, wave: usize) -> Self {
        self.expect_read_wave = wave;
        self
    }

    fn reads_in_flight(&self) -> usize {
        self.reads_in_flight.load(Ordering::SeqCst)
    }
}

impl HostBridge for ProbeBridge {
    fn call(&self, op: &str, args: Value) -> Result<Value, String> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_owned();
        if op == "write" {
            // A mutating op must never overlap a read: check before, during
            // and after its work window.
            for _ in 0..3 {
                if self.reads_in_flight() > 0 {
                    self.overlap_violation.store(true, Ordering::SeqCst);
                }
                std::thread::sleep(self.write_delay / 3);
            }
            if self.reads_in_flight() > 0 {
                self.overlap_violation.store(true, Ordering::SeqCst);
            }
            self.finished
                .lock()
                .expect("finished log")
                .push((op.to_owned(), path.clone()));
            return Ok(json!({ "wrote": path }));
        }
        // Read-only call.
        let now = self.reads_in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_reads_in_flight.fetch_max(now, Ordering::SeqCst);
        if self.expect_read_wave > 0 {
            let deadline = Instant::now() + Duration::from_millis(1500);
            while self.reads_in_flight() < self.expect_read_wave && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(2));
            }
            if self.reads_in_flight() >= self.expect_read_wave {
                self.wave_hits.fetch_add(1, Ordering::SeqCst);
            }
        }
        // `d<N>` paths encode their own delay in milliseconds.
        let delay = path
            .strip_prefix('d')
            .and_then(|ms| ms.parse::<u64>().ok())
            .map_or(self.read_delay, Duration::from_millis);
        std::thread::sleep(delay);
        self.reads_in_flight.fetch_sub(1, Ordering::SeqCst);
        self.finished
            .lock()
            .expect("finished log")
            .push((op.to_owned(), path.clone()));
        Ok(json!({ "read": path }))
    }

    fn notify(&self, _op: &str, _args: Value) {}
}

/// `Promise.all` over four `read` calls: all four must be in flight together
/// (the barrier is satisfied), results must come back in call order, and the
/// wall time must beat the serial sum.
#[test]
fn promise_all_read_wave_runs_concurrently() {
    if !common::require_bun("promise_all_read_wave_runs_concurrently") {
        return;
    }
    let k = kernel("wave");
    let bridge = ProbeBridge::new(Duration::from_millis(60)).with_wave(4);
    let code = r#"
const rs = await Promise.all(
  ["p1", "p2", "p3", "p4"].map(p => globalThis.__ompHostCall("read", { path: p }))
)
rs.map(r => r.read).join(",")
"#;
    let started = Instant::now();
    let r = k
        .execute_with(
            code,
            ExecOptions {
                bridge: Some(&bridge),
                ..ExecOptions::default()
            },
        )
        .expect("cell runs");
    let elapsed = started.elapsed();
    assert_eq!(r.status, CellStatus::Ok, "output: {:?}", r.output);
    assert_eq!(r.output.trim(), "\"p1,p2,p3,p4\"", "order: {:?}", r.output);
    assert_eq!(
        bridge.wave_hits.load(Ordering::SeqCst),
        4,
        "all four reads must overlap (parallel dispatch)"
    );
    assert!(
        elapsed < Duration::from_millis(2500),
        "batch must beat the serial sum (>= 4x barrier 1.5s): {elapsed:?}"
    );
    k.shutdown();
}

/// A mutating call inside the same `Promise.all` acts as a barrier: it never
/// overlaps a read, and every reply keeps its own frame's payload.
#[test]
fn mutating_op_is_serialized_between_reads() {
    if !common::require_bun("mutating_op_is_serialized_between_reads") {
        return;
    }
    let k = kernel("barrier");
    let bridge = ProbeBridge::new(Duration::from_millis(40));
    let code = r#"
const rs = await Promise.all([
  globalThis.__ompHostCall("read", { path: "ra" }),
  globalThis.__ompHostCall("write", { path: "wb" }),
  globalThis.__ompHostCall("read", { path: "rb" }),
])
rs.map(r => r.read ?? r.wrote).join(",")
"#;
    let r = k
        .execute_with(
            code,
            ExecOptions {
                bridge: Some(&bridge),
                ..ExecOptions::default()
            },
        )
        .expect("cell runs");
    assert_eq!(r.status, CellStatus::Ok, "output: {:?}", r.output);
    assert_eq!(r.output.trim(), "\"ra,wb,rb\"", "order: {:?}", r.output);
    assert!(
        !bridge.overlap_violation.load(Ordering::SeqCst),
        "write overlapped a read"
    );
    k.shutdown();
}

/// Arbitrary completion order: the slowest call starts first, but every
/// result keeps its own frame's payload (reply correlation by id).
#[test]
fn reversed_completion_keeps_correlation() {
    if !common::require_bun("reversed_completion_keeps_correlation") {
        return;
    }
    let k = kernel("correlation");
    let bridge = ProbeBridge::new(Duration::from_millis(0));
    let code = r#"
const rs = await Promise.all([
  globalThis.__ompHostCall("read", { path: "d150" }),
  globalThis.__ompHostCall("read", { path: "d80" }),
  globalThis.__ompHostCall("read", { path: "d10" }),
])
rs.map(r => r.read).join("|")
"#;
    let r = k
        .execute_with(
            code,
            ExecOptions {
                bridge: Some(&bridge),
                ..ExecOptions::default()
            },
        )
        .expect("cell runs");
    assert_eq!(r.status, CellStatus::Ok, "output: {:?}", r.output);
    assert_eq!(
        r.output.trim(),
        "\"d150|d80|d10\"",
        "call order preserved: {:?}",
        r.output
    );
    let finished = bridge.finished.lock().expect("finished log").clone();
    let fast_first = finished
        .iter()
        .take(2)
        .all(|(_, path)| path == "d10" || path == "d80");
    assert!(fast_first, "slowest call must finish last: {finished:?}");
    k.shutdown();
}

/// A `Promise.all` batch counts as one host-call episode for the idle
/// deadline: three 200 ms reads finish well inside a 500 ms idle window
/// (serial serving would exceed it).
#[test]
fn batch_fits_idle_deadline() {
    if !common::require_bun("batch_fits_idle_deadline") {
        return;
    }
    let k = kernel("deadline");
    let bridge = ProbeBridge::new(Duration::from_millis(200));
    let code = r#"
const rs = await Promise.all([
  globalThis.__ompHostCall("read", { path: "a" }),
  globalThis.__ompHostCall("read", { path: "b" }),
  globalThis.__ompHostCall("read", { path: "c" }),
])
rs.length
"#;
    let started = Instant::now();
    let r = k
        .execute_with(
            code,
            ExecOptions {
                timeout: Some(Duration::from_millis(500)),
                idle_timeout: true,
                bridge: Some(&bridge),
                ..ExecOptions::default()
            },
        )
        .expect("cell runs");
    let elapsed = started.elapsed();
    assert_eq!(r.status, CellStatus::Ok, "output: {:?}", r.output);
    assert!(!r.timed_out, "must not trip the idle deadline");
    assert_eq!(r.output.trim(), "3");
    assert!(
        elapsed < Duration::from_millis(450),
        "parallel batch wall time: {elapsed:?}"
    );
    k.shutdown();
}

/// Cancel firing while a batch is in flight: the cell still terminates
/// promptly with the omp JS semantics (worker killed, cancelled = true) —
/// batching must not hang abort.
#[test]
fn cancel_during_batch_reports_promptly() {
    if !common::require_bun("cancel_during_batch_reports_promptly") {
        return;
    }
    let k = kernel("cancel");
    let bridge = ProbeBridge::new(Duration::from_millis(200));
    let cancel = CancelToken::new();
    let tok = cancel.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(80));
        tok.cancel();
    });
    let code = r#"
const rs = await Promise.all([
  globalThis.__ompHostCall("read", { path: "a" }),
  globalThis.__ompHostCall("read", { path: "b" }),
  globalThis.__ompHostCall("read", { path: "c" }),
  globalThis.__ompHostCall("read", { path: "e" }),
])
rs.length
"#;
    let started = Instant::now();
    let r = k
        .execute_with(
            code,
            ExecOptions {
                bridge: Some(&bridge),
                cancel: Some(cancel),
                ..ExecOptions::default()
            },
        )
        .expect("cell returns");
    let elapsed = started.elapsed();
    assert!(r.cancelled, "cancelled must be reported");
    assert_eq!(r.status, CellStatus::KernelKilled);
    assert!(
        elapsed < Duration::from_millis(2500),
        "cancel must not hang on a batch: {elapsed:?}"
    );
    k.shutdown();
}

/// Output frames drained around a wave keep their arrival order, and a
/// 40-call wave (batch cap 32) plus a sequential tail all round-trip.
#[test]
fn output_order_and_wave_cap_stress() {
    if !common::require_bun("output_order_and_wave_cap_stress") {
        return;
    }
    let k = kernel("stress");
    let bridge = ProbeBridge::new(Duration::from_millis(10)).with_wave(8);
    let mut paths = Vec::new();
    for i in 0..40 {
        paths.push(format!("w{i:02}"));
    }
    let seq_paths: Vec<String> = (0..10).map(|i| format!("s{i:02}")).collect();
    let code = format!(
        r#"print("before")
const wave = await Promise.all(
  {paths:?}.map(p => globalThis.__ompHostCall("read", {{ path: p }}))
)
print("after")
const seq = []
for (const p of {seq_paths:?}) {{
  seq.push(await globalThis.__ompHostCall("read", {{ path: p }}))
}}
wave.map(r => r.read).join(",") + ";" + seq.map(r => r.read).join(",")
"#
    );
    let r = k
        .execute_with(
            &code,
            ExecOptions {
                bridge: Some(&bridge),
                ..ExecOptions::default()
            },
        )
        .expect("cell runs");
    assert_eq!(r.status, CellStatus::Ok, "output: {:?}", r.output);
    assert!(
        r.output.starts_with("before\n"),
        "drained stdout first: {:?}",
        r.output
    );
    assert!(
        r.output.contains("after\n"),
        "stdout after the wave: {:?}",
        r.output
    );
    let expected_wave: String = paths.join(",");
    let expected_seq: String = seq_paths.join(",");
    let expected = format!("\"{expected_wave};{expected_seq}\"");
    assert!(
        r.output.contains(&expected),
        "all 50 replies correlated in call order: {:?}",
        r.output
    );
    assert!(
        bridge.max_reads_in_flight.load(Ordering::SeqCst) >= 8,
        "stress wave must overlap"
    );
    assert!(
        !bridge.overlap_violation.load(Ordering::SeqCst),
        "no write overlap expected"
    );
    k.shutdown();
}
