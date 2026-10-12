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

use crate::{CellResult, CellStatus, Frame, KernelError, KernelProcess, RecvOutcome};

const RUNNER_JS: &str = include_str!("runner.js");

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
                &["-e", RUNNER_JS],
                cwd,
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
        // JS workers are routinely killed on timeout; recreate transparently
        // (omp's session registry does the same on worker death).
        self.ensure_running()?;
        let id = self.next_id.fetch_add(1, Ordering::SeqCst).to_string();
        {
            let mut guard = self.process.lock().expect("kernel lock");
            let p = guard
                .as_mut()
                .ok_or_else(|| io::Error::other("JS kernel is not running"))?;
            let req = serde_json::json!({ "id": id, "code": code, "storeHistory": true });
            p.stdin.write_all(format!("{req}\n").as_bytes())?;
            p.stdin.flush()?;
        }

        let mut output = String::new();
        let mut displays = Vec::new();
        let mut error: Option<KernelError> = None;
        let mut status = CellStatus::Ok;
        let deadline = timeout.map(|t| Instant::now() + t);

        loop {
            match recv(&self.process, deadline, &id) {
                Ok(frame) => match frame {
                    Frame::Stdout { data, .. } | Frame::Stderr { data, .. } => output.push_str(&data),
                    Frame::Result { bundle, .. } => {
                        if let Some(serde_json::Value::String(text)) = bundle.get("text/plain") {
                            output.push_str(text);
                            if !text.ends_with('\n') {
                                output.push('\n');
                            }
                        }
                    }
                    Frame::Display { bundle, .. } => displays.push(bundle),
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
                Err(RecvOutcome::TimedOut) => {
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
            *guard = Some(KernelProcess::spawn(&self.runtime, &["-e", RUNNER_JS], &self.cwd)?);
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

/// Read the next frame for request `id`; skips stale foreign frames.
fn recv(
    process: &Mutex<Option<KernelProcess>>,
    deadline: Option<Instant>,
    id: &str,
) -> Result<Frame, RecvOutcome> {
    let wait = |d: Option<Instant>| {
        d.map_or(Duration::from_hours(1), |d| {
            d.saturating_duration_since(Instant::now())
        })
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
                if frame.id() == id {
                    return Ok(frame);
                }
            }
            Err(RecvTimeoutError::Timeout) => return Err(RecvOutcome::TimedOut),
            Err(RecvTimeoutError::Disconnected) => return Err(RecvOutcome::Disconnected),
        }
    }
}

impl Drop for JsKernel {
    fn drop(&mut self) {
        self.shutdown();
    }
}
