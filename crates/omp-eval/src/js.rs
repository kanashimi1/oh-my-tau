//! Persistent JavaScript eval kernel: one long-lived `bun` subprocess per
//! kernel, speaking the same NDJSON frame protocol as the Python kernel
//! (see the crate root and `runner.js`).
//!
//! Port of omp's JS eval kernel (`pi-coding-agent/src/eval/js`), reduced to
//! the spike scope: persistent Bun process, top-level await, state across
//! cells, prelude helpers (`display`/`print`/`log`/`read`/`write`/`env`),
//! trailing-expression results, and timeouts.
//!
//! Timeout semantics differ from Python deliberately, matching omp: there is
//! no safe in-process interrupt for synchronous JS, so the host **kills** the
//! worker when the deadline expires (omp force-kills the worker too). The
//! cell is reported [`CellStatus::Interrupted`] with `timed_out` set, the
//! process is dropped, and the next `execute` transparently recreates the
//! kernel (state is gone; [`CellResult::js_timeout_annotation`] says so).

use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::{
    CancelToken, CellResult, CellStatus, ExecOptions, Frame, KernelError, KernelProcess,
    RecvOutcome,
};

const RUNNER_JS: &str = include_str!("runner.js");

/// Startup preludes, loaded silently at runner startup in this exact order
/// (omp prelude chain: I/O + tools, agents, transactional cells). Injected
/// into the runner script as `globalThis.__ompPreludeSources__`.
const PRELUDE_SOURCES: &[(&str, &str)] = &[
    ("prelude.js", include_str!("prelude.js")),
    ("prelude_agents.js", include_str!("prelude_agents.js")),
    (
        "prelude_combinators.js",
        include_str!("prelude_combinators.js"),
    ),
    ("prelude_txn.js", include_str!("prelude_txn.js")),
];

/// The `-e` script the Bun runner is spawned with: the prelude sources
/// injected first, then the runner proper (which loads them at startup).
fn runner_script() -> String {
    let sources = serde_json::json!(PRELUDE_SOURCES
        .iter()
        .map(|(name, source)| serde_json::json!({ "name": name, "source": source }))
        .collect::<Vec<_>>());
    format!("globalThis.__ompPreludeSources__ = {sources};\n{RUNNER_JS}")
}

/// Default JS runtime resolution (omp requires Bun).
#[must_use]
pub fn default_runtime() -> &'static str {
    "bun"
}

/// A persistent JavaScript kernel: one Bun runner subprocess per kernel.
/// Cells execute sequentially and share the global scope, so top-level
/// bindings persist across `execute` calls like omp's retained worker.
pub struct JsKernel {
    process: Mutex<Option<KernelProcess>>,
    next_id: AtomicU64,
    runtime: String,
    cwd: std::path::PathBuf,
}

impl JsKernel {
    /// Spawn the Bun runner (the prelude installs itself inside the runner;
    /// unlike Python there are no startup cells).
    ///
    /// # Errors
    /// Fails when the runtime cannot be spawned.
    pub fn start(runtime: &str, cwd: &std::path::Path) -> io::Result<Self> {
        Ok(Self {
            process: Mutex::new(Some(KernelProcess::spawn(
                runtime,
                &["-e", &runner_script()],
                cwd,
                &[],
            )?)),
            next_id: AtomicU64::new(1),
            runtime: runtime.to_owned(),
            cwd: cwd.to_path_buf(),
        })
    }

    /// Execute one cell in the persistent global scope. `timeout == None`
    /// waits indefinitely. On timeout the worker is killed (see module docs)
    /// and the next call recreates it.
    ///
    /// # Errors
    /// Fails on broken kernel I/O.
    ///
    /// # Panics
    /// Panics if the kernel mutex is poisoned (another thread panicked while
    /// holding it).
    pub fn execute(&self, code: &str, timeout: Option<Duration>) -> io::Result<CellResult> {
        self.execute_with(
            code,
            ExecOptions {
                timeout,
                ..ExecOptions::default()
            },
        )
    }

    /// Execute one cell with the full option set: a host bridge serving the
    /// cell's `host` frames, a cooperative cancel token, and optional
    /// idle-timeout semantics (the deadline counts only time spent outside
    /// `HostBridge::call`, mirroring omp's `IdleTimeout`).
    ///
    /// Host frame handling: frames of the form `{"type":"host","id":"<cellId>:
    /// <n>","op","args","wait"}` are answered on the runner's stdin with
    /// `{"type":"host_reply","id":…,"ok":true,"result":…}` (or `ok:false,
    /// "error":…`). `wait:true` ops block on `HostBridge::call` and raise in
    /// the cell on `Err`; `wait:false` ops are `HostBridge::notify`
    /// fire-and-forget. Each served `wait:true` call re-arms the deadline
    /// when `idle_timeout` is set. Without a bridge, `wait:true` calls get
    /// `ok:false, "host bridge unavailable"`.
    ///
    /// Cancel follows omp's JS semantics: the worker is force-killed (state
    /// is lost), the cell reports `cancelled: true` with
    /// [`CellStatus::KernelKilled`], and the next call recreates the kernel.
    ///
    /// # Errors
    /// Fails on broken kernel I/O.
    ///
    /// # Panics
    /// Panics if the kernel mutex is poisoned.
    pub fn execute_with(&self, code: &str, opts: ExecOptions<'_>) -> io::Result<CellResult> {
        // JS workers are routinely killed on timeout; recreate transparently
        // (omp's session registry does the same on worker death).
        self.ensure_running()?;
        let seq = self.next_id.fetch_add(1, Ordering::SeqCst);
        let id;
        {
            let mut guard = self.process.lock().expect("kernel lock");
            let p = guard
                .as_mut()
                .ok_or_else(|| io::Error::other("JS kernel is not running"))?;
            id = p.cell_id(seq);
            let mut req = serde_json::json!({
                "id": id,
                "code": code,
                "storeHistory": true,
                "idleTimeout": opts.idle_timeout,
            });
            if let Some(t) = opts.timeout {
                req["timeoutMs"] = serde_json::json!(t.as_millis() as u64);
            }
            p.stdin.write_all(format!("{req}\n").as_bytes())?;
            p.stdin.flush()?;
        }

        let mut output = String::new();
        let mut displays = Vec::new();
        let mut error: Option<KernelError> = None;
        let mut status = CellStatus::Ok;
        let mut deadline = opts.timeout.map(|t| Instant::now() + t);

        loop {
            match recv(&self.process, deadline, &id, opts.cancel.as_ref()) {
                Ok(frame) => match frame {
                    Frame::Stdout { data, .. } | Frame::Stderr { data, .. } => {
                        output.push_str(&data)
                    }
                    Frame::Result { bundle, .. } => {
                        if let Some(serde_json::Value::String(text)) = bundle.get("text/plain") {
                            output.push_str(text);
                            if !text.ends_with('\n') {
                                output.push('\n');
                            }
                        }
                    }
                    Frame::Display { bundle, .. } => displays.push(bundle),
                    Frame::Host {
                        id: host_id,
                        op,
                        args,
                        wait,
                    } => {
                        if wait {
                            let reply = match opts.bridge {
                                Some(bridge) => bridge.call(&op, args).map(
                                    |result| serde_json::json!({ "ok": true, "result": result }),
                                ),
                                None => Err("host bridge unavailable".to_owned()),
                            };
                            let reply = reply.unwrap_or_else(
                                |error| serde_json::json!({ "ok": false, "error": error }),
                            );
                            let mut guard = self.process.lock().expect("kernel lock");
                            if let Some(p) = guard.as_mut() {
                                let frame = serde_json::json!({
                                    "type": "host_reply",
                                    "id": host_id,
                                    "ok": reply["ok"],
                                    "error": reply.get("error").cloned().unwrap_or(Value::Null),
                                    "result": reply.get("result").cloned().unwrap_or(Value::Null),
                                });
                                // A dead runner here just means the cell will
                                // end via the disconnected branch below.
                                let _ = p.stdin.write_all(format!("{frame}\n").as_bytes());
                                let _ = p.stdin.flush();
                            }
                        } else if let Some(bridge) = opts.bridge {
                            bridge.notify(&op, args);
                        }
                        // Idle timeout: time inside a host call does not count,
                        // so each served call resets a fresh full window.
                        if opts.idle_timeout {
                            deadline = opts.timeout.map(|t| Instant::now() + t);
                        }
                    }
                    Frame::Error {
                        ename,
                        evalue,
                        traceback,
                        ..
                    } => {
                        status = CellStatus::Error;
                        // omp executor pushes `error.stack` as the cell output.
                        output.push_str(&traceback.join("\n"));
                        output.push('\n');
                        error = Some(KernelError {
                            name: ename,
                            value: evalue,
                            traceback,
                        });
                    }
                    Frame::Done {
                        status: done_status,
                        execution_count,
                        cancelled,
                        ..
                    } => {
                        if done_status == "error" && status == CellStatus::Ok {
                            status = CellStatus::Error;
                        }
                        return Ok(CellResult {
                            output,
                            status,
                            cancelled: cancelled.unwrap_or(false),
                            execution_count: execution_count.unwrap_or_default(),
                            error,
                            displays,
                            timed_out: false,
                        });
                    }
                    Frame::Started { .. } => {}
                },
                Err(RecvOutcome::TimedOut | RecvOutcome::Cancelled) => {
                    if opts.cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
                        // Turn cancel (omp JS semantics): force-kill the
                        // worker — the cell is reported cancelled with the
                        // kernel-killed status, and the next call recreates
                        // it. `timed_out` stays false: this was not a
                        // deadline expiry.
                        if let Some(mut p) = self.process.lock().expect("kernel lock").take() {
                            p.kill_group();
                            drop(p);
                        }
                        return Ok(CellResult {
                            cancelled: true,
                            ..CellResult::partial(output, displays, CellStatus::KernelKilled, false)
                        });
                    }
                    // omp JS semantics: force-kill the worker (state is lost)
                    // and report the cell as timed out. `Child::drop` does not
                    // kill, so do it explicitly; otherwise a sync-spinning
                    // cell would leak a live subprocess.
                    if let Some(mut p) = self.process.lock().expect("kernel lock").take() {
                        p.kill_group();
                        drop(p);
                    }
                    return Ok(CellResult::partial(
                        output,
                        displays,
                        CellStatus::Interrupted,
                        true,
                    ));
                }
                Err(RecvOutcome::Disconnected) => {
                    // Worker died or wedged mid-cell: make sure it is reaped,
                    // then report like omp so the caller recreates the kernel.
                    if let Some(mut p) = self.process.lock().expect("kernel lock").take() {
                        p.kill_group();
                        drop(p);
                    }
                    return Ok(CellResult::partial(
                        output,
                        displays,
                        CellStatus::KernelKilled,
                        false,
                    ));
                }
            }
        }
    }

    fn ensure_running(&self) -> io::Result<()> {
        let mut guard = self.process.lock().expect("kernel lock");
        let alive = guard
            .as_mut()
            .is_some_and(|p| p.child.try_wait().expect("try_wait").is_none());
        if !alive {
            // try_wait() already reaped the dead runner above, so no kill is
            // needed (and none is safe: the pid could have been recycled).
            if let Some(p) = guard.take() {
                drop(p);
            }
            *guard = Some(KernelProcess::spawn(
                &self.runtime,
                &["-e", &runner_script()],
                &self.cwd,
                &[],
            )?);
        }
        Ok(())
    }

    /// Graceful shutdown: protocol `exit`, wait for the runner to drain.
    ///
    /// # Panics
    /// Panics if the kernel mutex is poisoned.
    pub fn shutdown(&self) {
        let mut guard = self.process.lock().expect("kernel lock");
        if let Some(mut p) = guard.take() {
            p.settle_exit();
        }
    }

    /// True while the runner subprocess is alive.
    ///
    /// # Panics
    /// Panics if the kernel mutex is poisoned.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        let mut guard = self.process.lock().expect("kernel lock");
        match guard.as_mut() {
            Some(p) => p.child.try_wait().expect("try_wait").is_none(),
            None => false,
        }
    }

    /// Runner process id while the kernel is running, else `None`
    /// (introspection/test aid).
    ///
    /// # Panics
    /// Panics if the kernel mutex is poisoned.
    #[must_use]
    pub fn pid(&self) -> Option<u32> {
        let guard = self.process.lock().expect("kernel lock");
        guard.as_ref().map(|p| p.child.id())
    }

    /// Runtime command this kernel was started with.
    #[must_use]
    pub fn runtime(&self) -> &str {
        &self.runtime
    }
}

/// Read the next frame for request `id`; skips stale foreign frames. Frames
/// whose id is `id` or starts with `id + ":"` belong to this cell — the
/// latter are its host bridge requests (`<cellId>:<n>`), which the caller
/// answers; both are returned. When `cancel` is set the receive is polled at
/// ~50 ms granularity so a fired token unblocks promptly; a spurious poll
/// timeout just loops again (unless the deadline itself has passed, which is
/// indistinguishable here and checked by the caller against `cancel`).
fn recv(
    process: &Mutex<Option<KernelProcess>>,
    deadline: Option<Instant>,
    id: &str,
    cancel: Option<&CancelToken>,
) -> Result<Frame, RecvOutcome> {
    let host_prefix = format!("{id}:");
    let poll = Duration::from_millis(50);
    let wait = |d: Option<Instant>| {
        let base = d.map_or(Duration::from_hours(1), |d| {
            d.saturating_duration_since(Instant::now())
        });
        if cancel.is_some() {
            base.min(poll)
        } else {
            base
        }
    };
    loop {
        let outcome = {
            let mut guard = process.lock().expect("kernel lock");
            let Some(p) = guard.as_mut() else {
                return Err(RecvOutcome::Disconnected);
            };
            if p.dead.load(Ordering::SeqCst) {
                return Err(RecvOutcome::Disconnected);
            }
            p.rx.recv_timeout(wait(deadline))
        };
        match outcome {
            Ok(frame) => {
                if frame.id() == id || frame.id().starts_with(&host_prefix) {
                    return Ok(frame);
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if cancel.is_some_and(CancelToken::is_cancelled) {
                    return Err(RecvOutcome::Cancelled);
                }
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    return Err(RecvOutcome::TimedOut);
                }
                // Spurious poll wake-up (cancel granularity): keep waiting.
            }
            Err(RecvTimeoutError::Disconnected) => return Err(RecvOutcome::Disconnected),
        }
    }
}

impl Drop for JsKernel {
    fn drop(&mut self) {
        self.shutdown();
    }
}
