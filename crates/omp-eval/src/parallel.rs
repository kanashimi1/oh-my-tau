//! Concurrent dispatch of independent read-only host calls for the JS eval
//! kernel.
//!
//! A cell that evaluates several host calls at once —
//! `Promise.all([read("artifact://a"), read("agent://b"), tool.grep(…)])` —
//! emits all `host` frames back-to-back and blocks on their replies, but the
//! kernel read loop used to serve them strictly one at a time: each
//! [`HostBridge::call`] serialized behind the previous reply. For read-only
//! ops the calls are independent, so the batch is dispatched concurrently.
//!
//! Soundness: [`HostBridge: Send + Sync`](HostBridge) makes concurrent
//! `call(&self, …)` invocations a trait-contract obligation for every
//! implementor — no implementor or `omp-cli` change is needed. The dispatch
//! runs on `std::thread::scope` threads that are all joined before any reply
//! is written, so nothing leaks.
//!
//! Guarantees:
//! - reply correlation: each reply carries the frame's own `id`; the runner
//!   routes it to the awaiting promise, so arbitrary completion order is fine;
//! - ordering: replies are written to the runner's stdin in frame order under
//!   the process lock (serialized writes); stdout/stderr/display frames
//!   drained between host frames keep their arrival order in the cell output;
//! - serial barriers: mutating (`wait: true`, non-read-only) and notify
//!   (`wait: false`) frames flush the pending batch first and are then served
//!   exactly like the sequential loop serves them — a write never overlaps a
//!   read;
//! - deadline/cancel: a batch is one host-call episode (idle-timeout re-arm
//!   after it completes, matching per-call re-arm semantics); cancel and
//!   deadline polls resume at the next read-loop pass, bounded by the slowest
//!   call in the batch (never worse than the serial sum).

use std::io::Write;
use std::sync::atomic::Ordering;
use std::sync::mpsc::TryRecvError;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::Value;

use crate::{CancelToken, Frame, HostBridge, KernelProcess};

/// Host ops that may join a concurrent batch. Exact-match allowlist,
/// deliberately conservative: everything not listed (write/bash/env, agent
/// and LLM ops, unknown future ops) stays sequential. `read` covers the
/// scheme-path reads (`artifact://`, `agent://` handles, `output(id…)`);
/// the bare tool names and their `tool.`-prefixed forms cover the bridge's
/// tool-fallback dispatch and the prelude `tool.<name>` proxy.
pub const PARALLEL_SAFE_HOST_OPS: &[&str] = &[
    "read",
    "glob",
    "grep",
    "tool.read",
    "tool.glob",
    "tool.grep",
];

/// Is this host op safe to dispatch concurrently with sibling read-only
/// calls? Anything not in [`PARALLEL_SAFE_HOST_OPS`] is sequential.
#[must_use]
pub fn is_parallel_safe_host_op(op: &str) -> bool {
    PARALLEL_SAFE_HOST_OPS.contains(&op)
}

/// One pending `wait: true` host frame: its reply id (`<cellId>:<n>`), the
/// op name, and the op args.
pub(crate) struct HostCall {
    pub(crate) host_id: String,
    pub(crate) op: String,
    pub(crate) args: Value,
}

/// Quiet window: after the last frame seen, wait this long for a sibling
/// before the batch is considered complete (`Promise.all` emits its frames
/// back-to-back, so the window is what lets a wave form).
const DRAIN_QUIET: Duration = Duration::from_millis(2);
/// Hard cap on one drain sweep, so a pathological frame stream cannot stall
/// the read loop (a blocked runner cannot produce frames anyway).
const DRAIN_BUDGET: Duration = Duration::from_millis(20);
/// Poll granularity while waiting out the quiet window.
const DRAIN_POLL: Duration = Duration::from_micros(200);
/// Upper bound on one concurrent batch (and therefore on transient scoped
/// threads); frames beyond the cap are served by the following pass.
const MAX_BATCH: usize = 32;
/// Cap in-flight blocking host calls. Large reads may each retain a full
/// file plus serialized reply, so unbounded Promise.all waves otherwise
/// multiply peak RSS (and spawn unbounded threads).
const MAX_CONCURRENT: usize = 4;

/// Dispatch the batch's calls concurrently on scoped threads and return the
/// results in batch order. A lone call stays inline (no thread). All threads
/// are joined before returning: no leaked work, and a panic inside
/// `HostBridge::call` propagates exactly like the sequential call would.
pub(crate) fn dispatch(bridge: &dyn HostBridge, batch: &[HostCall]) -> Vec<Result<Value, String>> {
    if batch.len() <= 1 {
        return batch
            .iter()
            .map(|call| bridge.call(&call.op, call.args.clone()))
            .collect();
    }
    let slots: Mutex<Vec<Option<Result<Value, String>>>> =
        Mutex::new((0..batch.len()).map(|_| None).collect());
    for (start, chunk) in batch.chunks(MAX_CONCURRENT).enumerate() {
        let offset = start * MAX_CONCURRENT;
        std::thread::scope(|scope| {
            for (index, call) in chunk.iter().enumerate() {
                let slots = &slots;
                scope.spawn(move || {
                    let result = bridge.call(&call.op, call.args.clone());
                    slots.lock()[offset + index] = Some(result);
                });
            }
        });
    }
    slots
        .into_inner()
        .into_iter()
        .map(|slot| slot.unwrap_or_else(|| Err("internal: batch slot unset after join".to_owned())))
        .collect()
}

/// Write one `host_reply` frame to the runner's stdin (serialized under the
/// process lock). A dead runner here just means the cell ends via the
/// disconnected branch of the kernel loop, as in the sequential path.
pub(crate) fn write_host_reply(
    process: &Mutex<Option<KernelProcess>>,
    host_id: &str,
    reply: &Value,
) {
    let mut guard = process.lock();
    if let Some(p) = guard.as_mut() {
        let frame = serde_json::json!({
            "type": "host_reply",
            "id": host_id,
            "ok": reply["ok"],
            "error": reply.get("error").cloned().unwrap_or(Value::Null),
            "result": reply.get("result").cloned().unwrap_or(Value::Null),
        });
        let _ = p.stdin.write_all(format!("{frame}\n").as_bytes());
        let _ = p.stdin.flush();
    }
}

/// Dispatch a collected batch concurrently and answer its frames in frame
/// order under one process-lock acquisition (serialized stdin writes, one
/// flush).
fn flush_batch(
    process: &Mutex<Option<KernelProcess>>,
    bridge: &dyn HostBridge,
    batch: &[HostCall],
    cancel: Option<&CancelToken>,
) -> bool {
    if batch.is_empty() {
        return true;
    }
    // Cancel boundary: the sequential loop observed a fired token between
    // calls, so a batch must not complete a wave (and resume the cell)
    // after the user aborted mid-batch. Aborting leaves the runner blocked
    // on the missing replies; the kernel loop's next poll detects the token
    // and force-kills the worker (omp JS cancel semantics).
    if cancel.is_some_and(CancelToken::is_cancelled) {
        return false;
    }
    let replies: Vec<Value> = dispatch(bridge, batch)
        .into_iter()
        .map(|result| match result {
            Ok(result) => serde_json::json!({ "ok": true, "result": result }),
            Err(error) => serde_json::json!({ "ok": false, "error": error }),
        })
        .collect();
    // Post-join boundary: the token may have fired while the batch ran.
    // Dropping the replies here leaves the runner blocked on them, so the
    // kernel loop's poll still kills the worker mid-cell — the same
    // observable behavior as the sequential path aborting between calls.
    if cancel.is_some_and(CancelToken::is_cancelled) {
        return false;
    }
    let mut guard = process.lock();
    if let Some(p) = guard.as_mut() {
        for (call, reply) in batch.iter().zip(replies.iter()) {
            let frame = serde_json::json!({
                "type": "host_reply",
                "id": call.host_id,
                "ok": reply["ok"],
                "error": reply.get("error").cloned().unwrap_or(Value::Null),
                "result": reply.get("result").cloned().unwrap_or(Value::Null),
            });
            let _ = p.stdin.write_all(format!("{frame}\n").as_bytes());
        }
        let _ = p.stdin.flush();
    }
    true
}

/// Serve one barrier frame sequentially (the pending batch was already
/// flushed and answered): `wait: true` blocks on the bridge and writes the
/// single reply; `wait: false` is fire-and-forget.
fn serve_barrier(
    process: &Mutex<Option<KernelProcess>>,
    bridge: &dyn HostBridge,
    host_id: &str,
    op: &str,
    args: Value,
    wait: bool,
) {
    if wait {
        let reply = match bridge.call(op, args) {
            Ok(result) => serde_json::json!({ "ok": true, "result": result }),
            Err(error) => serde_json::json!({ "ok": false, "error": error }),
        };
        write_host_reply(process, host_id, &reply);
    } else {
        bridge.notify(op, args);
    }
}

/// Serve a read-only `wait: true` host frame together with every queued
/// sibling: collect the frames the runner already emitted (plus a short
/// quiet window so a `Promise.all` wave forms), dispatch the batch
/// concurrently, answer in frame order, and keep drained output frames in
/// arrival order.
///
/// Barrier frames keep serial semantics: a mutating call or a notify first
/// flushes the pending batch, then is served inline exactly like the
/// sequential loop serves it. `Done`/`Error`/`Result` frames are returned to
/// the caller for its regular arms after the pending batch is answered.
///
/// `probe_only` skips the quiet window (adaptive hint: after a lone call the
/// next serve pays no added latency; after a wave it is `false` so further
/// waves batch fully). Returns the deferred terminal frame (if one was
/// consumed) and the number of host calls served.
// The serve mirrors the kernel loop's whole per-frame context (process,
// bridge, cell id, first frame, output collectors, drain hint, cancel).
#[allow(clippy::too_many_arguments)]
pub(crate) fn serve_read_only_batch(
    process: &Mutex<Option<KernelProcess>>,
    bridge: &dyn HostBridge,
    cell_id: &str,
    first: (String, String, Value),
    output: &mut String,
    displays: &mut Vec<Value>,
    probe_only: bool,
    cancel: Option<&CancelToken>,
) -> (Option<Frame>, usize) {
    let host_prefix = format!("{cell_id}:");
    let (host_id, op, args) = first;
    let mut batch = vec![HostCall { host_id, op, args }];
    let mut served = 0usize;
    let mut deferred: Option<Frame> = None;
    let mut quiet = Instant::now() + DRAIN_QUIET;
    let budget = Instant::now() + DRAIN_BUDGET;
    'drain: loop {
        if batch.len() >= MAX_BATCH || Instant::now() >= budget {
            break 'drain;
        }
        let next = {
            let guard = process.lock();
            let Some(p) = guard.as_ref() else {
                // Kernel taken (killed): drop the batch — replies would go
                // unread; the kernel loop reports the disconnect, matching
                // the sequential path where these frames are never served.
                return (None, 0);
            };
            if p.dead.load(Ordering::SeqCst) {
                return (None, 0);
            }
            p.rx.try_recv()
        };
        match next {
            Ok(frame) => {
                quiet = Instant::now() + DRAIN_QUIET;
                if frame.id() != cell_id && !frame.id().starts_with(&host_prefix) {
                    // Stale foreign frame: drop, exactly like `recv`.
                    continue;
                }
                match frame {
                    Frame::Host {
                        id,
                        op,
                        args,
                        wait: true,
                    } if is_parallel_safe_host_op(&op) => {
                        batch.push(HostCall {
                            host_id: id,
                            op,
                            args,
                        });
                    }
                    Frame::Stdout { data, .. } | Frame::Stderr { data, .. } => {
                        output.push_str(&data);
                    }
                    Frame::Display { bundle, .. } => displays.push(bundle),
                    // Barrier frames (mutating call / notify): answer what
                    // came before, then serve exactly like the sequential
                    // loop would.
                    Frame::Host { id, op, args, wait } => {
                        if !flush_batch(process, bridge, &batch, cancel) {
                            return (None, 0);
                        }
                        served += batch.len();
                        batch.clear();
                        serve_barrier(process, bridge, &id, &op, args, wait);
                        served += 1;
                    }
                    terminal => {
                        if !flush_batch(process, bridge, &batch, cancel) {
                            return (None, 0);
                        }
                        served += batch.len();
                        batch.clear();
                        deferred = Some(terminal);
                        break 'drain;
                    }
                }
            }
            Err(TryRecvError::Disconnected) => {
                // Runner died mid-batch: abort without dispatching; the
                // kernel loop reports KernelKilled (sequential behavior).
                return (None, 0);
            }
            Err(TryRecvError::Empty) => {
                if probe_only || Instant::now() >= quiet {
                    break 'drain;
                }
                std::thread::sleep(DRAIN_POLL);
            }
        }
    }
    if !flush_batch(process, bridge, &batch, cancel) {
        return (None, 0);
    }
    served += batch.len();
    (deferred, served)
}

#[cfg(test)]
mod tests {
    use super::{dispatch, is_parallel_safe_host_op, HostCall};
    use crate::HostBridge;
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Bridge that tracks the concurrent in-flight high-water mark.
    struct OverlapBridge {
        in_flight: Arc<AtomicUsize>,
        max_in_flight: Arc<AtomicUsize>,
    }

    impl HostBridge for OverlapBridge {
        fn call(&self, op: &str, args: Value) -> Result<Value, String> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(40));
            let result = Ok(json!({ "op": op, "path": args["path"] }));
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            result
        }

        fn notify(&self, _op: &str, _args: Value) {}
    }

    fn batch_of(paths: &[&str]) -> Vec<HostCall> {
        paths
            .iter()
            .enumerate()
            .map(|(index, path)| HostCall {
                host_id: format!("c1:{index}"),
                op: "read".to_owned(),
                args: json!({ "path": path }),
            })
            .collect()
    }

    #[test]
    fn classification_is_exact_match() {
        for op in [
            "read",
            "glob",
            "grep",
            "tool.read",
            "tool.glob",
            "tool.grep",
        ] {
            assert!(is_parallel_safe_host_op(op), "{op} must be parallel-safe");
        }
        for op in [
            "write",
            "edit",
            "bash",
            "env",
            "output",
            "tool.list",
            "tool.describe",
            "tool.write",
            "tool.bash",
            "__agent__",
            "__llm_query__",
            "__budget__",
            "log",
            "phase",
            "tool_define",
            "unknown_op",
            "",
        ] {
            assert!(!is_parallel_safe_host_op(op), "{op} must stay sequential");
        }
    }

    #[test]
    fn lone_call_stays_inline() {
        let bridge = OverlapBridge {
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_in_flight: Arc::new(AtomicUsize::new(0)),
        };
        let batch = batch_of(&["a"]);
        let results = dispatch(&bridge, &batch);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
        assert_eq!(
            bridge.max_in_flight.load(Ordering::SeqCst),
            1,
            "a lone call must not spawn a thread"
        );
    }

    #[test]
    fn concurrent_dispatch_overlaps_and_preserves_order() {
        let bridge = OverlapBridge {
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_in_flight: Arc::new(AtomicUsize::new(0)),
        };
        let batch = batch_of(&["a", "b", "c", "d"]);
        let started = Instant::now();
        let results = dispatch(&bridge, &batch);
        let elapsed = started.elapsed();
        assert_eq!(results.len(), 4);
        assert!(results.iter().all(Result::is_ok));
        // Reply correlation: result i carries call i's own payload.
        for (call, result) in batch.iter().zip(results.iter()) {
            let value = result
                .as_ref()
                .unwrap_or_else(|error| panic!("ok: {error}"));
            assert_eq!(value["path"], call.args["path"], "correlation");
        }
        // Overlap proof: a sequential loop peaks at 1 and takes >= 160 ms.
        assert!(
            bridge.max_in_flight.load(Ordering::SeqCst) >= 2,
            "calls must overlap"
        );
        assert!(
            elapsed < Duration::from_millis(150),
            "batch wall time: {elapsed:?}"
        );
    }

    #[test]
    fn concurrent_dispatch_bounds_in_flight_calls() {
        let bridge = OverlapBridge {
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_in_flight: Arc::new(AtomicUsize::new(0)),
        };
        let batch = batch_of(&["a", "b", "c", "d", "e", "f", "g", "h"]);
        let results = dispatch(&bridge, &batch);
        assert_eq!(results.len(), 8);
        assert!(results.iter().all(Result::is_ok));
        let peak = bridge.max_in_flight.load(Ordering::SeqCst);
        assert!((2..=super::MAX_CONCURRENT).contains(&peak), "peak={peak}");
    }
}
