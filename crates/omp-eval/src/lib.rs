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
//!
//! Orphan protection: every runner is spawned in its own process group
//! (`process_group(0)`), and the runners run an internal parent watchdog
//! (ppid polling in Python, a Bun Worker in JS) that hard-exits when the
//! host process dies — even mid-cell — so a `SIGKILL`ed host cannot leak
//! spinning kernels. The host escalates to `kill(-pgid, SIGKILL)` (whole
//! group, so user-spawned subprocesses of a cell die too) when a runner
//! refuses a graceful shutdown, and on the JS timeout path.
//!
//! Host bridge: cell code (via the preludes) can call back into the host
//! through [`HostBridge`] — `host` NDJSON frames answered by a `host_reply`
//! line on the runner's stdin, the NDJSON stand-in for omp's loopback HTTP
//! bridges. Time spent inside a bridge call can be excluded from the cell
//! timeout (`ExecOptions::idle_timeout`, TS `idle-timeout.ts`); a
//! [`CancelToken`] mirrors the TS turn-cancel (SIGINT into the runner,
//! reusing the escalation path).

#![allow(clippy::missing_errors_doc)] // single io::Error failure mode; per-call docs add noise

use std::io::{self, BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
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

pub mod js;

/// Bridge between a running cell and the host process: the cell-side preludes
/// emit `host` frames (`{"type":"host","id":"<cellId>:<n>","op","args","wait"}`)
/// on the protocol channel, the kernel forwards them here, and (for
/// `wait: true`) writes one `host_reply` frame back to the runner's stdin.
/// This is the NDJSON stand-in for omp's loopback HTTP bridges
/// (`eval/py/tool-bridge.ts` et al.); op names and `args`/`result` JSON shapes
/// are verbatim TS bridge payloads.
pub trait HostBridge: Send + Sync {
    /// Blocking request from the running cell. `Err(message)` is raised in
    /// the cell as `OmpHostError` (Python) / a rejected `Error` (JS).
    fn call(&self, op: &str, args: Value) -> Result<Value, String>;
    /// Fire-and-forget status from the cell (log/phase/progress).
    fn notify(&self, op: &str, args: Value);
}

/// Cooperative cancel for a running cell (TS turn-cancel). Polled by the
/// kernel at ~50 ms granularity; firing it SIGINTs the Python runner (the
/// existing escalation path decides whether the kernel survives).
#[derive(Clone, Default, Debug)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Options for [`PythonKernel::execute_with`] (and the JS twin).
#[derive(Default)]
pub struct ExecOptions<'a> {
    /// Per-cell deadline; `None` waits indefinitely.
    pub timeout: Option<Duration>,
    /// True: the timeout counts only time spent outside `HostBridge::call`
    /// (omp `IdleTimeout` semantics — each bridge call resets a fresh window,
    /// and the runner pauses its own alarm around host calls).
    pub idle_timeout: bool,
    /// Host bridge serving this cell's `host` frames; `None` replies
    /// `ok: false, error: "host bridge unavailable"` to `wait: true` calls.
    pub bridge: Option<&'a dyn HostBridge>,
    /// Optional cooperative cancel, polled while the cell runs.
    pub cancel: Option<CancelToken>,
}

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

    /// omp `formatJsTimeoutAnnotation` for a force-killed JS worker: timeout
    /// rounded up to seconds plus the state-reset notice (JS has no safe
    /// in-process interrupt, so a timed-out worker is killed and recreated).
    #[must_use]
    pub fn js_timeout_annotation(timeout: Option<Duration>) -> String {
        const RESET_NOTE: &str = "The JS worker was force-killed and its VM state was reset; variables from earlier cells are gone.";
        match timeout {
            None => format!("Command timed out. {RESET_NOTE}"),
            Some(d) => {
                let secs = d.as_millis().div_ceil(1000).max(1);
                format!("Command timed out after {secs} seconds. {RESET_NOTE}")
            }
        }
    }

    pub(crate) fn partial(
        output: String,
        displays: Vec<Value>,
        status: CellStatus,
        timed_out: bool,
    ) -> Self {
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
pub(crate) enum Frame {
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
    /// Cell-side host bridge request (`wait` selects call vs notify).
    Host {
        id: String,
        op: String,
        args: Value,
        wait: bool,
    },
    Done {
        id: String,
        status: String,
        execution_count: Option<u64>,
        cancelled: Option<bool>,
    },
}

impl Frame {
    pub(crate) fn id(&self) -> &str {
        match self {
            Frame::Started { id }
            | Frame::Stdout { id, .. }
            | Frame::Stderr { id, .. }
            | Frame::Display { id, .. }
            | Frame::Result { id, .. }
            | Frame::Error { id, .. }
            | Frame::Host { id, .. }
            | Frame::Done { id, .. } => id,
        }
    }
}

/// Upper bound for one NDJSON protocol line from a runner. Longer lines are
/// consumed to their newline and dropped, so a runaway writer can neither
/// exhaust host memory nor desynchronise the frames that follow.
pub(crate) const MAX_FRAME_LINE: usize = 64 * 1024 * 1024;

/// Outcome of [`read_frame_line`].
#[derive(Debug, PartialEq, Eq)]
enum LineRead {
    /// A complete (or EOF-terminated) line of at most the cap is in the buffer.
    Line,
    /// The line exceeded the cap (total length given); the buffer is empty.
    Oversize(usize),
    Eof,
}

/// Read one `\n`-terminated line into `buf` (cleared first), storing at most
/// `cap` bytes; an over-long line is consumed through its newline and
/// reported as [`LineRead::Oversize`].
fn read_frame_line(
    reader: &mut impl BufRead,
    buf: &mut Vec<u8>,
    cap: usize,
) -> io::Result<LineRead> {
    buf.clear();
    let mut total = 0usize;
    loop {
        let available = match reader.fill_buf() {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if available.is_empty() {
            return Ok(match total {
                0 => LineRead::Eof,
                t if t > cap => LineRead::Oversize(t),
                _ => LineRead::Line,
            });
        }
        let (len, complete) = match available.iter().position(|&b| b == b'\n') {
            Some(i) => (i + 1, true),
            None => (available.len(), false),
        };
        if total + len <= cap {
            buf.extend_from_slice(&available[..len]);
        } else {
            buf.clear();
        }
        total += len;
        reader.consume(len);
        if complete {
            return Ok(if total > cap {
                LineRead::Oversize(total)
            } else {
                LineRead::Line
            });
        }
    }
}

/// Forward every protocol frame on `reader` to `tx` until EOF, a read error
/// or a dropped receiver. Blank, non-JSON, non-frame and oversized lines are
/// logged and skipped (omp host behaviour); the line buffer is reset per line
/// so one bad line never affects the next.
fn pump_frames(mut reader: impl BufRead, cap: usize, tx: &mpsc::Sender<Frame>) {
    let mut line = Vec::new();
    loop {
        match read_frame_line(&mut reader, &mut line, cap) {
            Ok(LineRead::Eof) | Err(_) => return,
            Ok(LineRead::Oversize(len)) => {
                tracing::warn!(len, cap, "eval kernel: dropping oversized protocol line");
            }
            Ok(LineRead::Line) => {
                if line.iter().all(u8::is_ascii_whitespace) {
                    continue;
                }
                match serde_json::from_slice::<Frame>(&line) {
                    Ok(frame) => {
                        if tx.send(frame).is_err() {
                            return;
                        }
                    }
                    Err(err) => tracing::warn!(
                        len = line.len(),
                        %err,
                        "eval kernel: skipping non-protocol line"
                    ),
                }
            }
        }
    }
}

/// 128-bit random hex nonce prefixed to the cell ids of one runner process:
/// host frames are matched by cell-id prefix, so ids must not be guessable
/// even by code that reaches the frame channel.
fn cell_id_nonce() -> String {
    use std::io::Read as _;
    let mut bytes = [0u8; 16];
    let urandom = std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut bytes));
    if urandom.is_err() {
        // RandomState keys come from the OS RNG; hash fresh instances.
        use std::hash::{BuildHasher as _, Hasher as _};
        for half in bytes.chunks_mut(8) {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u32(std::process::id());
            h.write_u128(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos()),
            );
            half.copy_from_slice(&h.finish().to_le_bytes());
        }
    }
    bytes.iter().fold(String::with_capacity(32), |mut hex, b| {
        use std::fmt::Write as _;
        let _ = write!(hex, "{b:02x}");
        hex
    })
}

pub(crate) struct KernelProcess {
    pub(crate) child: Child,
    pub(crate) stdin: ChildStdin,
    pub(crate) rx: Receiver<Frame>,
    pub(crate) dead: Arc<AtomicBool>,
    /// Per-process random prefix of every cell id (see [`cell_id_nonce`]).
    id_nonce: String,
}

impl KernelProcess {
    /// Spawn `program args…` speaking the NDJSON kernel protocol on stdio.
    pub(crate) fn spawn(
        program: &str,
        args: &[&str],
        cwd: &std::path::Path,
        env: &[(String, String)],
    ) -> io::Result<Self> {
        // Own session (TS `shouldDetachKernel`: Bun `detached` = setsid): the
        // runner leads its own process group, so the host can kill it plus
        // anything it spawned (kill(-pgid)) without touching its own group,
        // and user code has no controlling terminal to grab (`tcsetpgrp`,
        // raw-mode resets, SIGTTIN). `unsafe` is forbidden (no `pre_exec`):
        // util-linux `setsid` execs the runner in place (it is not a group
        // leader yet, so no fork; pid stays the runner's). Fallback: own group.
        let setsid = ["/usr/bin/setsid", "/bin/setsid"]
            .into_iter()
            .find(|p| std::path::Path::new(p).is_file());
        let mut command = match setsid {
            Some(setsid) => {
                let mut c = Command::new(setsid);
                c.arg("--wait").arg(program);
                c
            }
            None => {
                let mut c = Command::new(program);
                #[cfg(unix)]
                c.process_group(0);
                c
            }
        };
        let mut child = command
            .args(args)
            .current_dir(cwd)
            .envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
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
            pump_frames(BufReader::new(stdout), MAX_FRAME_LINE, &tx);
            thread_dead.store(true, Ordering::SeqCst);
        });
        Ok(Self {
            child,
            stdin,
            rx,
            dead,
            id_nonce: cell_id_nonce(),
        })
    }

    /// Wire id of the `seq`-th cell: `<nonce>-<seq>`.
    pub(crate) fn cell_id(&self, seq: u64) -> String {
        format!("{}-{seq}", self.id_nonce)
    }

    pub(crate) fn settle_exit(&mut self) {
        let _ = self.stdin.write_all(b"{\"type\": \"exit\"}\n");
        let _ = self.stdin.flush();
        // Grant a short grace for the runner's main loop to see the exit
        // request, then escalate to killing the whole process group.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            if self.child.try_wait().expect("try_wait").is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        self.kill_group();
        // Child::wait closes stdin on return, unblocking the runner's main loop
        let _ = self.child.wait();
    }

    /// SIGKILL the runner's whole process group (`process_group(0)` makes the
    /// pgid equal the runner's pid), so cells that spawned subprocesses do not
    /// leak them.
    pub(crate) fn kill_group(&mut self) {
        #[cfg(unix)]
        {
            // Negative pid targets the runner's own process group; nix::kill is
            // a safe wrapper (the workspace forbids `unsafe`, and none is needed).
            if let Ok(pid) = i32::try_from(self.child.id()) {
                let _ = nix::sys::signal::kill(
                    nix::unistd::Pid::from_raw(-pid),
                    nix::sys::signal::Signal::SIGKILL,
                );
            }
        }
        #[cfg(not(unix))]
        {
            // No process groups on this platform: kill the runner itself.
            let _ = self.child.kill();
        }
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
    env: Vec<(String, String)>,
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
        Self::start_with_env(interpreter, cwd, &[])
    }

    /// [`PythonKernel::start`] with extra process environment for the runner
    /// (the eval tool passes `PI_EVAL_LOCAL_ROOTS` / `PI_ARTIFACTS_DIR` /
    /// `PI_SESSION_FILE` so the preludes resolve internal URLs and spill
    /// artifacts next to the session).
    ///
    /// # Errors
    /// See [`PythonKernel::start`].
    pub fn start_with_env(
        interpreter: &str,
        cwd: &std::path::Path,
        env: &[(String, String)],
    ) -> io::Result<Self> {
        let kernel = Self {
            process: Mutex::new(Some(KernelProcess::spawn(
                interpreter,
                &["-u", "-c", RUNNER_SCRIPT],
                cwd,
                env,
            )?)),
            next_id: AtomicU64::new(1),
            interpreter: interpreter.to_owned(),
            cwd: cwd.to_path_buf(),
            env: env.to_vec(),
        };
        kernel.run_startup(&kernel.init_script())?;
        kernel.run_preloads()?;
        Ok(kernel)
    }

    /// Silent startup cells in TS `preludes.ts` order: io helpers, agent
    /// helpers, transactional cells. `prelude.py` self-guards with the TS
    /// `__omp_prelude_loaded__` flag, so no stamp cell precedes it. Loaded via
    /// `include_str!` so the runner needs no filesystem access to its source.
    fn run_preloads(&self) -> io::Result<()> {
        for prelude in [
            PRELUDE_PY,
            PRELUDE_AGENTS_PY,
            PRELUDE_COMBINATORS_PY,
            PRELUDE_TXN_PY,
        ] {
            self.run_startup(prelude)?;
        }
        Ok(())
    }

    fn run_startup(&self, source: &str) -> io::Result<()> {
        let result = self.execute_opts(
            source,
            &ExecOptions {
                timeout: Some(STARTUP_TIMEOUT),
                ..ExecOptions::default()
            },
            true,
        )?;
        // omp's executeWithBudget fails the start on any non-clean cell —
        // including a cancelled/timed-out one (load can push the init or
        // prelude past STARTUP_TIMEOUT). Accepting a wedged kernel here made
        // the next cell report KernelKilled and silently drop user state.
        if result.cancelled || result.timed_out || result.status == CellStatus::KernelKilled {
            return Err(io::Error::other(format!(
                "Python kernel startup cell did not complete cleanly: {result:?}"
            )));
        }
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
            *guard = Some(KernelProcess::spawn(
                &self.interpreter,
                &["-u", "-c", RUNNER_SCRIPT],
                &self.cwd,
                &self.env,
            )?);
        }
        self.run_startup(&self.init_script())?;
        self.run_preloads()
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
        self.execute_with(
            code,
            ExecOptions {
                timeout,
                ..ExecOptions::default()
            },
        )
    }

    /// Execute one cell with the full option surface: host bridge, cancel
    /// token and idle-style timeout (TS `executeWithBudget`).
    ///
    /// # Errors
    /// Fails on broken kernel I/O.
    pub fn execute_with(&self, code: &str, opts: ExecOptions<'_>) -> io::Result<CellResult> {
        self.execute_opts(code, &opts, false)
    }

    // Mirrors the omp host cell-execution flow step by step; splitting would obscure that parity.
    #[allow(clippy::too_many_lines)]
    fn execute_opts(
        &self,
        code: &str,
        opts: &ExecOptions<'_>,
        startup: bool,
    ) -> io::Result<CellResult> {
        let seq = self.next_id.fetch_add(1, Ordering::SeqCst);
        let id;
        {
            let mut guard = self.process.lock().expect("kernel lock");
            let p = guard
                .as_mut()
                .ok_or_else(|| io::Error::other("Python kernel is not running"))?;
            id = p.cell_id(seq);
            // omp host sends storeHistory:false for silent startup/control cells
            let mut req = serde_json::json!({ "id": id, "code": code, "storeHistory": !startup });
            if let Some(t) = opts.timeout {
                // runner-enforced SIGALRM + watchdog, see crate docs
                req.as_object_mut().expect("fresh object").insert(
                    "timeoutMs".into(),
                    Value::from(u64::try_from(t.as_millis()).unwrap_or(u64::MAX)),
                );
                if opts.idle_timeout {
                    // runner pauses/re-arms its alarm around host calls (TS
                    // idle-timeout.ts); the host below resets a fresh window
                    // after every HostBridge::call as well.
                    req.as_object_mut()
                        .expect("fresh object")
                        .insert("idleTimeout".into(), Value::from(true));
                }
            }
            p.stdin.write_all(format!("{req}\n").as_bytes())?;
            p.stdin.flush()?;
        }

        let mut output = String::new();
        let mut displays = Vec::new();
        let mut error: Option<KernelError> = None;
        let mut status = CellStatus::Ok;
        let execution_count;
        let mut deadline = opts.timeout.map(|t| Instant::now() + t);
        let mut in_escalation = false;
        let mut timed_out = false;
        let mut cancel = opts.cancel.clone();

        loop {
            match self.recv(deadline, &id, cancel.as_ref()) {
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
                    Frame::Host {
                        id: hid,
                        op,
                        args,
                        wait,
                    } => {
                        // Cell-side bridge request (TS py/js tool bridges): the
                        // payload shapes are the TS `args`/`result` verbatim.
                        if wait {
                            let reply = match opts.bridge {
                                Some(bridge) => match bridge.call(&op, args) {
                                    Ok(result) => serde_json::json!({
                                        "type": "host_reply",
                                        "id": hid,
                                        "ok": true,
                                        "result": result,
                                    }),
                                    Err(error) => serde_json::json!({
                                        "type": "host_reply",
                                        "id": hid,
                                        "ok": false,
                                        "error": error,
                                    }),
                                },
                                None => serde_json::json!({
                                    "type": "host_reply",
                                    "id": hid,
                                    "ok": false,
                                    "error": "host bridge unavailable",
                                }),
                            };
                            let mut guard = self.process.lock().expect("kernel lock");
                            if let Some(p) = guard.as_mut() {
                                let _ = p.stdin.write_all(format!("{reply}\n").as_bytes());
                                let _ = p.stdin.flush();
                            }
                        } else if let Some(bridge) = opts.bridge {
                            bridge.notify(&op, args);
                        }
                        if opts.idle_timeout {
                            // Time inside a bridge call does not count (TS
                            // IdleTimeout.resume starts a fresh full window).
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
                        // cancelled (timeout interrupt / turn cancel) outranks the
                        // error status: omp reports these cells as interrupted,
                        // not failed
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
                            timed_out,
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
                    timed_out = true;
                }
                Err(RecvOutcome::Cancelled) => {
                    // Turn cancel (TS): omp's host-side SIGINT into the running
                    // cell. The interrupt (like the timeout's) lands as
                    // KeyboardInterrupt("Execution interrupted"); reuse the
                    // escalation window to let the runner settle.
                    if let Some(p) = self.process.lock().expect("kernel lock").as_mut() {
                        #[cfg(unix)]
                        if let Ok(pid) = i32::try_from(p.child.id()) {
                            let _ = nix::sys::signal::kill(
                                nix::unistd::Pid::from_raw(-pid),
                                nix::sys::signal::Signal::SIGINT,
                            );
                        }
                        #[cfg(not(unix))]
                        {
                            let _ = p.child.kill();
                        }
                    }
                    // One shot: stop polling the token while the interrupt lands.
                    cancel = None;
                    deadline = Some(Instant::now() + INTERRUPT_ESCALATION);
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

    /// Read the next frame for request `id`; skips stale foreign frames. Host
    /// frames carry `<cellId>:<n>` ids and are matched by cell prefix. When a
    /// cancel token is present it caps the wait at ~50 ms so cancellation is
    /// observed promptly even without a deadline.
    fn recv(
        &self,
        deadline: Option<Instant>,
        id: &str,
        cancel: Option<&CancelToken>,
    ) -> Result<Frame, RecvOutcome> {
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
                let w = match (deadline, cancel) {
                    (Some(d), Some(_)) => wait(Some(d)).min(Duration::from_millis(50)),
                    (d, None) => wait(d),
                    (None, Some(_)) => Duration::from_millis(50),
                };
                p.rx.recv_timeout(w)
            };
            match outcome {
                Ok(frame) => {
                    let fid = frame.id();
                    // host bridge frames are numbered "<cellId>:<n>" within the cell
                    if fid == id
                        || matches!(frame, Frame::Host { .. })
                            && fid
                                .strip_prefix(id)
                                .is_some_and(|rest| rest.starts_with(':'))
                    {
                        return Ok(frame);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if let Some(token) = cancel {
                        if token.is_cancelled() {
                            return Err(RecvOutcome::Cancelled);
                        }
                        // 50 ms poll granularity: only a real deadline expiry is
                        // a cell timeout.
                        if deadline.is_none_or(|d| Instant::now() < d) {
                            continue;
                        }
                    }
                    return Err(RecvOutcome::TimedOut);
                }
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

    /// Interpreter path this kernel was started with.
    #[must_use]
    pub fn interpreter(&self) -> &str {
        &self.interpreter
    }
}

pub(crate) enum RecvOutcome {
    TimedOut,
    Disconnected,
    /// The cell's [`CancelToken`] fired while waiting on frames.
    Cancelled,
}

impl Drop for PythonKernel {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Startup cells loaded in TS `preludes.ts` order (silent, `storeHistory: false`).
const PRELUDE_PY: &str = include_str!("prelude.py");
const PRELUDE_AGENTS_PY: &str = include_str!("prelude_agents.py");
const PRELUDE_COMBINATORS_PY: &str = include_str!("prelude_combinators.py");
const PRELUDE_TXN_PY: &str = include_str!("prelude_txn.py");

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

#[cfg(test)]
mod reader_tests {
    use super::{cell_id_nonce, pump_frames, read_frame_line, Frame, LineRead};
    use std::io::Cursor;
    use std::sync::mpsc;

    fn frames(input: &[u8], cap: usize) -> Vec<String> {
        let (tx, rx) = mpsc::channel();
        pump_frames(Cursor::new(input.to_vec()), cap, &tx);
        drop(tx);
        rx.iter().map(|f: Frame| f.id().to_owned()).collect()
    }

    #[test]
    fn junk_lines_never_swallow_later_frames() {
        let input = b"not json\n\xff\xfe\n{\"type\":\"bogus\",\"id\":\"x\"}\n\n   \n{\"type\":\"started\",\"id\":\"a\"}\n{\"truncated\": \n{\"type\":\"done\",\"id\":\"b\",\"status\":\"ok\"}\n";
        assert_eq!(frames(input, 1024), ["a", "b"]);
    }

    #[test]
    fn oversized_lines_are_dropped_and_the_stream_resyncs() {
        let big = format!(
            "{{\"type\":\"stdout\",\"id\":\"big\",\"data\":\"{}\"}}\n",
            "x".repeat(500)
        );
        let input = format!("{big}{{\"type\":\"started\",\"id\":\"a\"}}\n{big}{{\"type\":\"started\",\"id\":\"b\"}}");
        assert_eq!(frames(input.as_bytes(), 64), ["a", "b"]);

        let mut reader = Cursor::new(big.into_bytes());
        let mut buf = Vec::new();
        assert!(
            matches!(read_frame_line(&mut reader, &mut buf, 64), Ok(LineRead::Oversize(n)) if n > 500)
        );
        assert!(buf.is_empty());
        assert_eq!(
            read_frame_line(&mut reader, &mut buf, 64).unwrap(),
            LineRead::Eof
        );
    }

    #[test]
    fn nonces_are_random_hex() {
        let (a, b) = (cell_id_nonce(), cell_id_nonce());
        assert_eq!(a.len(), 32);
        assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }
}
