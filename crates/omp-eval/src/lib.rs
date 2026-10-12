//! Persistent Python eval kernel for omp-rs.
//!
//! Port of omp's `eval/py` subprocess kernel (TS `pi-coding-agent/src/eval/py`),
//! minimally: the host spawns `python3 -u` with an embedded self-contained
//! runner script, speaks the same NDJSON line protocol, and keeps one
//! long-lived interpreter process per kernel so state persists across cells.
//!
//! Timeout: the workspace forbids `unsafe`, so the host cannot `kill(2)` the
//! interpreter. Instead the runner arms a SIGALRM per cell (same observable
//! behavior as omp's host-side SIGINT — the running cell gets
//! `KeyboardInterrupt("Execution interrupted")` and the kernel survives) and
//! a watchdog thread force-exits a wedged runner after omp's escalation
//! window ([`INTERRUPT_ESCALATION`]); the host then reports
//! [`CellStatus::KernelKilled`] and [`PythonKernel::restart`] recreates it.
//!
//! Result text matches omp's eval executor: cell output is the arrival-order
//! concatenation of stdout/stderr chunks plus the trailing-expression repr,
//! and errors append the runner's traceback lines (runner frames filtered,
//! caret-style for cell syntax errors).

#![allow(clippy::missing_errors_doc)] // single io::Error failure mode; per-call docs add noise

use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::Value;

/// Grace window the host grants the runner after a cell timeout before
/// declaring the kernel wedged (omp: `INTERRUPT_ESCALATION_MS = 5_000`).
pub const INTERRUPT_ESCALATION: Duration = Duration::from_secs(5);
/// Startup budget for the init/prelude cells (omp: `STARTUP_TIMEOUT_MS`).
pub const STARTUP_TIMEOUT: Duration = Duration::from_secs(10);

const RUNNER_SCRIPT: &str = include_str!("runner.py");

/// Terminal status of one executed cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellStatus {
    /// Cell ran to completion.
    Ok,
    /// Cell raised; see [`CellResult::error`].
    Error,
    /// Timeout fired; the cell was interrupted and the kernel survives.
    Interrupted,
    /// Timeout fired; the runner ignored the interrupt and was killed.
    /// State is lost — recreate with [`PythonKernel::restart`].
    KernelKilled,
}

/// Structured error frame from the runner (omp `KernelExecuteResult.error`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelError {
    pub name: String,
    pub value: String,
    /// Pre-formatted traceback lines, runner frames already filtered.
    pub traceback: Vec<String>,
}

/// Result of one cell execution.
#[derive(Debug, Clone)]
pub struct CellResult {
    /// Arrival-order stdout + stderr chunks, the trailing-expression repr,
    /// and (on error) the traceback text — the omp eval executor's `output`.
    pub output: String,
    pub status: CellStatus,
    pub cancelled: bool,
    pub execution_count: u64,
    pub error: Option<KernelError>,
    /// Rich display bundles captured from prelude `display()` calls.
    pub displays: Vec<Value>,
    pub timed_out: bool,
}

impl CellResult {
    /// omp `formatSessionTimeoutAnnotation`: timeout rounded up to seconds.
    #[must_use]
    pub fn timeout_annotation(timeout: Option<Duration>) -> String {
        match timeout {
            None => "Command timed out".to_owned(),
            Some(d) => {
                let secs = d.as_millis().div_ceil(1000).max(1);
                format!("Command timed out after {secs} seconds")
            }
        }
    }

    /// omp `formatSessionKernelTimeoutAnnotation` for this result's status.
    #[must_use]
    pub fn kernel_timeout_annotation(&self) -> Option<String> {
        match self.status {
            CellStatus::KernelKilled => Some(
                "eval cell timed out and the kernel was unresponsive to interrupt; the kernel has been killed \
                 and will be recreated on the next call."
                    .to_owned(),
            ),
            CellStatus::Interrupted => Some(
                "eval cell timed out; kernel interrupted but remains running. Reset the kernel via { reset: true } \
                 if state appears corrupted."
                    .to_owned(),
            ),
            _ => None,
        }
    }

    /// True when the cell raised.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.status == CellStatus::Error
    }

    fn partial(output: String, displays: Vec<Value>, status: CellStatus, timed_out: bool) -> Self {
        Self {
            output,
            status,
            cancelled: timed_out,
            execution_count: 0,
            error: None,
            displays,
            timed_out,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
enum Frame {
    Started {
        id: String,
    },
    Stdout {
        id: String,
        data: String,
    },
    Stderr {
        id: String,
        data: String,
    },
    Display {
        id: String,
        bundle: Value,
    },
    Result {
        id: String,
        bundle: Value,
    },
    Error {
        id: String,
        ename: String,
        evalue: String,
        traceback: Vec<String>,
    },
    Done {
        id: String,
        status: String,
        execution_count: Option<u64>,
        cancelled: Option<bool>,
    },
}

impl Frame {
    fn id(&self) -> &str {
        match self {
            Frame::Started { id }
            | Frame::Stdout { id, .. }
            | Frame::Stderr { id, .. }
            | Frame::Display { id, .. }
            | Frame::Result { id, .. }
            | Frame::Error { id, .. }
            | Frame::Done { id, .. } => id,
        }
    }
}

struct KernelProcess {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Frame>,
    dead: Arc<AtomicBool>,
}

impl KernelProcess {
    fn spawn(interpreter: &str, cwd: &std::path::Path) -> io::Result<Self> {
        let mut child = Command::new(interpreter)
            .args(["-u", "-c", RUNNER_SCRIPT])
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdout = child.stdout.take().expect("piped stdout");
        let stdin = child.stdin.take().expect("piped stdin");
        let (tx, rx) = mpsc::channel();
        let dead = Arc::new(AtomicBool::new(false));
        let thread_dead = Arc::clone(&dead);
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap_or(0) > 0 {
                match serde_json::from_str::<Frame>(&line) {
                    Ok(frame) => {
                        if tx.send(frame).is_err() {
                            break;
                        }
                    }
                    Err(_) => continue, // invalid line: omp host logs and skips
                }
                line.clear();
            }
            thread_dead.store(true, Ordering::SeqCst);
        });
        Ok(Self {
            child,
            stdin,
            rx,
            dead,
        })
    }

    fn settle_exit(&mut self) {
        let _ = self.stdin.write_all(b"{\"type\": \"exit\"}\n");
        let _ = self.stdin.flush();
        // Child::wait closes stdin on return, unblocking the runner's main loop
        let _ = self.child.wait();
    }
}

/// A persistent Python kernel: one runner subprocess per kernel; cells execute
/// sequentially and share the interpreter namespace, so bindings persist
/// across `execute` calls like omp's retained session kernel.
pub struct PythonKernel {
    process: Mutex<Option<KernelProcess>>,
    next_id: AtomicU64,
    interpreter: String,
    cwd: std::path::PathBuf,
}

impl PythonKernel {
    /// Spawn the runner and run the init + prelude cells (omp
    /// `PythonKernel::start`: init script then prelude, within
    /// [`STARTUP_TIMEOUT`]).
    ///
    /// # Errors
    /// Fails when the interpreter cannot be spawned or a startup cell fails
    /// or exceeds the timeout.
    pub fn start(interpreter: &str, cwd: &std::path::Path) -> io::Result<Self> {
        let kernel = Self {
            process: Mutex::new(Some(KernelProcess::spawn(interpreter, cwd)?)),
            next_id: AtomicU64::new(1),
            interpreter: interpreter.to_owned(),
            cwd: cwd.to_path_buf(),
        };
        kernel.run_startup(&kernel.init_script())?;
        kernel.run_startup(PRELUDE)?;
        Ok(kernel)
    }

    fn run_startup(&self, source: &str) -> io::Result<()> {
        let result = self.execute_opts(source, Some(STARTUP_TIMEOUT), true)?;
        if result.is_error() {
            return Err(io::Error::other(format!(
                "Python kernel startup failed: {}",
                result
                    .error
                    .map_or_else(|| "unknown error".to_owned(), |e| e.value)
            )));
        }
        Ok(())
    }

    /// Restart the kernel after [`CellStatus::KernelKilled`] (omp replaces the
    /// session kernel and re-runs init + prelude; state is not preserved).
    ///
    /// # Errors
    /// See [`PythonKernel::start`].
    ///
    /// # Panics
    /// Panics if the kernel mutex is poisoned (another thread panicked while
    /// holding it).
    pub fn restart(&self) -> io::Result<()> {
        {
            let mut guard = self.process.lock().expect("kernel lock");
            if let Some(p) = guard.as_mut() {
                p.settle_exit();
            }
            *guard = Some(KernelProcess::spawn(&self.interpreter, &self.cwd)?);
        }
        self.run_startup(&self.init_script())?;
        self.run_startup(PRELUDE)
    }

    fn init_script(&self) -> String {
        let cwd = self.cwd.display().to_string().replace('\\', "\\\\");
        format!(
            "import os, sys\n\
             os.chdir(\"{cwd}\")\n\
             if \"{cwd}\" not in sys.path:\n    sys.path.insert(0, \"{cwd}\")"
        )
    }

    /// Execute one cell in the persistent namespace. `timeout == None` waits
    /// indefinitely. On timeout the runner interrupts the cell (see crate
    /// docs); if the runner wedged and died, [`CellStatus::KernelKilled`] is
    /// returned after [`INTERRUPT_ESCALATION`].
    ///
    /// # Errors
    /// Fails on broken kernel I/O.
    pub fn execute(&self, code: &str, timeout: Option<Duration>) -> io::Result<CellResult> {
        self.execute_opts(code, timeout, false)
    }

    // Mirrors the omp host cell-execution flow step by step; splitting would obscure that parity.
    #[allow(clippy::too_many_lines)]
    fn execute_opts(
        &self,
        code: &str,
        timeout: Option<Duration>,
        startup: bool,
    ) -> io::Result<CellResult> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst).to_string();
        {
            let mut guard = self.process.lock().expect("kernel lock");
            let p = guard
                .as_mut()
                .ok_or_else(|| io::Error::other("Python kernel is not running"))?;
            // omp host sends storeHistory:false for silent startup/control cells
            let mut req = serde_json::json!({ "id": id, "code": code, "storeHistory": !startup });
            if let Some(t) = timeout {
                // runner-enforced SIGALRM + watchdog, see crate docs
                req.as_object_mut().expect("fresh object").insert(
                    "timeoutMs".into(),
                    Value::from(u64::try_from(t.as_millis()).unwrap_or(u64::MAX)),
                );
            }
            p.stdin.write_all(format!("{req}\n").as_bytes())?;
            p.stdin.flush()?;
        }

        let mut output = String::new();
        let mut displays = Vec::new();
        let mut error: Option<KernelError> = None;
        let mut status = CellStatus::Ok;
        let execution_count;
        let mut deadline = timeout.map(|t| Instant::now() + t);
        let mut in_escalation = false;

        loop {
            match self.recv(deadline, &id) {
                Ok(frame) => match frame {
                    Frame::Stdout { data, .. } | Frame::Stderr { data, .. } => {
                        output.push_str(&data);
                    }
                    Frame::Result { bundle, .. } => {
                        if let Some(Value::String(text)) = bundle.get("text/plain") {
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
                        // omp kernel-base: traceback lines joined by \n + trailing \n
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
                        execution_count: ec,
                        cancelled: c,
                        ..
                    } => {
                        execution_count = ec.unwrap_or_default();
                        let cancelled = c.unwrap_or(false) || in_escalation;
                        if in_escalation && error.is_none() && done_status == "error" {
                            // watchdog settled the request without a traceback
                            status = CellStatus::KernelKilled;
                        } else if done_status == "error" && status == CellStatus::Ok {
                            status = CellStatus::Error;
                        }
                        // cancelled (timeout interrupt) outranks the error status:
                        // omp reports these cells as timed out, not failed
                        if cancelled && status == CellStatus::Error {
                            status = CellStatus::Interrupted;
                        }
                        return Ok(CellResult {
                            output,
                            status,
                            cancelled,
                            execution_count,
                            error,
                            displays,
                            timed_out: in_escalation,
                        });
                    }
                    Frame::Started { .. } => {}
                },
                Err(RecvOutcome::TimedOut) => {
                    // The runner emits its own error + done(cancelled) frames on
                    // SIGALRM; grant it omp's escalation window to settle. Frames
                    // that arrive there are processed by the main loop via the
                    // extended deadline below. If nothing settles, the runner
                    // wedged (C extension holding the GIL) and hard-exited:
                    // report KernelKilled, like omp's kill + recreate path.
                    let escalated = Instant::now() + INTERRUPT_ESCALATION;
                    deadline = Some(escalated);
                    in_escalation = true;
                }
                Err(RecvOutcome::Disconnected) if in_escalation => {
                    // runner hard-exited (watchdog or OS): kernel must be recreated
                    return Ok(CellResult::partial(
                        output,
                        displays,
                        CellStatus::KernelKilled,
                        true,
                    ));
                }
                Err(RecvOutcome::Disconnected) => {
                    // runner died mid-cell (watchdog hard-exit or crash): report
                    // KernelKilled like omp, so the caller can recreate the kernel
                    return Ok(CellResult::partial(
                        output,
                        displays,
                        CellStatus::KernelKilled,
                        true,
                    ));
                }
            }
        }
    }

    /// Read the next frame for request `id`; skips stale foreign frames.
    fn recv(&self, deadline: Option<Instant>, id: &str) -> Result<Frame, RecvOutcome> {
        let wait = |d: Option<Instant>| {
            d.map_or(Duration::from_hours(1), |d| {
                d.saturating_duration_since(Instant::now())
            })
        };
        loop {
            let outcome = {
                let mut guard = self.process.lock().expect("kernel lock");
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

    /// Graceful shutdown: protocol `exit`, wait, escalate on refusal.
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

    /// Interpreter path this kernel was started with.
    #[must_use]
    pub fn interpreter(&self) -> &str {
        &self.interpreter
    }
}

enum RecvOutcome {
    TimedOut,
    Disconnected,
}

impl Drop for PythonKernel {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Prelude executed at startup (mirrors omp's `__omp_prelude_loaded__` guard);
/// the runner script already installs `display`, so this only stamps the flag.
const PRELUDE: &str =
    "if \"__omp_prelude_loaded__\" not in globals():\n    __omp_prelude_loaded__ = True";

/// Default interpreter resolution (omp falls back to the platform python3).
#[must_use]
pub fn default_interpreter() -> &'static str {
    "python3"
}

/// Per-cwd retained-kernel registry (minimal stand-in for omp's session
/// registry keyed by sessionId/cwd/interpreter).
#[derive(Default)]
pub struct KernelRegistry {
    kernels: Mutex<std::collections::HashMap<String, Arc<PythonKernel>>>,
}

impl KernelRegistry {
    /// Get or create the retained kernel for `cwd`.
    ///
    /// # Errors
    /// See [`PythonKernel::start`].
    ///
    /// # Panics
    /// Panics if the registry mutex is poisoned.
    pub fn get_or_start(
        &self,
        cwd: &std::path::Path,
        interpreter: Option<&str>,
    ) -> io::Result<Arc<PythonKernel>> {
        let key = cwd.to_string_lossy().into_owned();
        let mut kernels = self.kernels.lock().expect("registry lock");
        if let Some(k) = kernels.get(&key) {
            if k.is_alive() {
                return Ok(Arc::clone(k));
            }
        }
        let kernel = Arc::new(PythonKernel::start(
            interpreter.unwrap_or(default_interpreter()),
            cwd,
        )?);
        kernels.insert(key, Arc::clone(&kernel));
        Ok(kernel)
    }
}
