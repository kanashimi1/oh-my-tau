//! Controllable child runs: [`TaskRunner::start`] launches a child in the
//! background and hands back a [`RunControl`] (kill / steer) plus a
//! [`RunJoin`] future that resolves to the settled [`RunOutput`].
//!
//! - Kill: the child runs in its own process group (`process_group(0)`), so a
//!   cancel SIGKILLs the agent and every tool subprocess it spawned, then
//!   settles the run [`RunStatus::Cancelled`]. A cancel that lands while the
//!   run still waits for a concurrency permit settles it without spawning.
//!   Killability is a phase transition: `cancel()` CAS-claims it, and exactly
//!   one of {`cancel()`, the run's own completion} wins — a run that finished
//!   first settles [`RunStatus::Completed`] and a late `cancel()` returns
//!   `false`.
//! - Process groups: because each child is its own group, a terminal Ctrl-C
//!   or SIGHUP aimed at the parent's group does *not* reach the children
//!   (intentional: background agents outlive interactive interrupts; steer or
//!   `cancel()` them instead). The group is SIGKILLed on every exit path —
//!   cancel, timeout, spawn failure, the run's `RunJoin`/guard dropping
//!   (parent runtime shutdown) — and after a *normal* exit too, so surviving
//!   group members (background jobs, servers) are reaped and cannot hold the
//!   stdout/stderr pipes open; pipe joins are additionally time-bounded.
//! - Steer ([`ChildMode::Rpc`] only): the child is `<binary> --mode rpc`; the
//!   prompt goes in as an RPC `prompt` command and stdin stays open while the
//!   agent runs, so [`RunControl::steer`] writes a `steer` command the child
//!   injects at its next turn boundary (TS IRC delivery to a running agent).
//!   The receipt is a delivery receipt: it resolves once the drive loop
//!   actually wrote the frame, and reports failure when the run is already
//!   finishing. Once the prompt settles the run asks for the final assistant
//!   text and the session's token usage under a hard Finishing deadline (a
//!   child that never answers or ignores stdin EOF is killed and settles
//!   failed), then closes stdin so the child exits.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use super::{
    collect_output, is_transient_spawn_error, spawn_failure, RunOutput, RunStatus, TaskRunner,
    TaskSpec,
};
use crate::schema::SchemaValidation;

/// How long pipe readers may linger after a kill (a grandchild that left the
/// process group could otherwise hold the pipe open forever).
const KILL_DRAIN: Duration = Duration::from_secs(2);
/// Bound for pipe readers after a *normal* exit: surviving group members are
/// killed first, but a grandchild that escaped the group could still hold a
/// pipe; never block the settle on it.
const PIPES_DRAIN: Duration = Duration::from_secs(5);
/// Grace for an RPC child to exit after its stdin closed / stdout ended.
const RPC_EXIT_GRACE: Duration = Duration::from_secs(10);
/// Hard deadline for the Finishing phase: once the prompt settled, the
/// final-answer/session-stats requests must complete within this window even
/// if the child keeps stdout open (a missing `get_session_stats` reply or a
/// child ignoring stdin EOF must not hang the run or hold its permit).
const RPC_FINISHING_DEADLINE: Duration = Duration::from_secs(15);

/// Test knob for [`RPC_FINISHING_DEADLINE`] (`OMP_TASK_TEST_FINISHING_DEADLINE_MS`);
/// unset in production.
fn rpc_finishing_deadline() -> Duration {
    std::env::var("OMP_TASK_TEST_FINISHING_DEADLINE_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(RPC_FINISHING_DEADLINE)
}

/// RPC command ids the run uses for its own requests.
const PROMPT_ID: &str = "omp-task-prompt";
const FINAL_TEXT_ID: &str = "omp-task-final-text";
const STATS_ID: &str = "omp-task-stats";

/// How [`TaskRunner::start`] drives the child process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildMode {
    /// `<binary> -p <prompt>`: one-shot, stdout is the final answer, no input
    /// channel (the `task` tool's mode).
    Print,
    /// `<binary> --mode rpc`: JSON-lines RPC; stdin stays open while the
    /// agent runs so it accepts steering messages.
    Rpc,
}

/// Lifecycle phase of a started run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunPhase {
    /// Waiting for a concurrency permit; nothing spawned yet.
    Queued,
    /// Child spawned and working on the prompt.
    Running,
    /// Prompt settled; collecting the final answer, no longer steerable.
    Finishing,
    /// The run resolved (completed, failed or cancelled).
    Settled,
}

impl RunPhase {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Queued,
            1 => Self::Running,
            2 => Self::Finishing,
            _ => Self::Settled,
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            Self::Queued => 0,
            Self::Running => 1,
            Self::Finishing => 2,
            Self::Settled => 3,
        }
    }
}

/// Why a steering message could not be delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SteerError {
    /// [`ChildMode::Print`] children have no input channel.
    #[error("the child runs in print mode and has no input channel")]
    Unsupported,
    /// The prompt already settled (or the run is gone).
    #[error("the child is not running")]
    NotRunning,
}

enum Control {
    Steer(String, oneshot::Sender<Result<(), SteerError>>),
    Kill,
}

/// A steer receipt the drive loop resolves once the frame was actually
/// written (or failed explicitly).
type SteerAck = oneshot::Sender<Result<(), SteerError>>;

struct Shared {
    phase: AtomicU8,
    output_tokens: AtomicU64,
    /// Killability transition: starts `true`; exactly one of `cancel()` and
    /// the run's own completion takes it (the loser observes `false`).
    killable: AtomicBool,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            phase: AtomicU8::default(),
            output_tokens: AtomicU64::default(),
            killable: AtomicBool::new(true),
        }
    }
}

impl Shared {
    fn set_phase(&self, phase: RunPhase) {
        self.phase.store(phase.as_u8(), Ordering::SeqCst);
    }

    /// Claim the killable transition; `false` when already claimed (a cancel
    /// was requested or the run completed).
    fn take_killable(&self) -> bool {
        self.killable.swap(false, Ordering::SeqCst)
    }
}

/// Kill/steer handle of one started run (cheap to clone).
#[derive(Clone)]
pub struct RunControl {
    id: String,
    mode: ChildMode,
    tx: mpsc::UnboundedSender<Control>,
    shared: Arc<Shared>,
}

impl std::fmt::Debug for RunControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunControl")
            .field("id", &self.id)
            .field("mode", &self.mode)
            .field("phase", &self.phase())
            .finish_non_exhaustive()
    }
}

impl RunControl {
    /// The run's agent id (`agent://<id>`).
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The child driving mode.
    #[must_use]
    pub fn mode(&self) -> ChildMode {
        self.mode
    }

    /// Current lifecycle phase.
    #[must_use]
    pub fn phase(&self) -> RunPhase {
        RunPhase::from_u8(self.shared.phase.load(Ordering::SeqCst))
    }

    /// Queue a steering message for the running agent. Messages sent while
    /// the run is still queued are delivered right after the prompt.
    ///
    /// The returned future is a *delivery* receipt: it resolves once the
    /// drive loop actually wrote the steer frame (or failed it explicitly,
    /// e.g. the prompt settled before the frame went out). A send failure
    /// (the run ended before the control was queued) resolves to
    /// [`SteerError::NotRunning`].
    ///
    /// # Errors
    /// [`SteerError::Unsupported`] for print-mode children,
    /// [`SteerError::NotRunning`] once the prompt settled, the run was
    /// cancelled, or the run ended.
    pub async fn steer(&self, message: impl Into<String>) -> Result<(), SteerError> {
        if self.mode == ChildMode::Print {
            return Err(SteerError::Unsupported);
        }
        if !self.shared.killable.load(Ordering::SeqCst)
            || !matches!(self.phase(), RunPhase::Queued | RunPhase::Running)
        {
            return Err(SteerError::NotRunning);
        }
        let (ack, receipt) = oneshot::channel();
        self.tx
            .send(Control::Steer(message.into(), ack))
            .map_err(|_| SteerError::NotRunning)?;
        receipt.await.unwrap_or(Err(SteerError::NotRunning))
    }

    /// Kill the run: SIGKILL the child's process group (or drop the queued
    /// spawn) and settle it [`RunStatus::Cancelled`]. Killability is a phase
    /// transition — returns `true` only if this call won it; a run that
    /// already finished settles [`RunStatus::Completed`] and a late
    /// `cancel()` returns `false`.
    #[must_use = "false means nothing was cancelled"]
    pub fn cancel(&self) -> bool {
        if !self.shared.take_killable() {
            return false;
        }
        // The drive loop (or the queued-phase select in `run_controlled`)
        // still holds `rx` whenever the killable transition was unclaimed, so
        // the send cannot fail; ignore the result regardless — the run owns
        // the outcome and observes the transition via `take_killable`.
        let _ = self.tx.send(Control::Kill);
        true
    }

    /// Output tokens the child session reported (RPC mode, after the prompt
    /// settled); `0` when unknown.
    #[must_use]
    pub fn output_tokens(&self) -> u64 {
        self.shared.output_tokens.load(Ordering::SeqCst)
    }
}

/// Future resolving to the settled [`RunOutput`] of a started run. The run
/// progresses in its own task whether or not this is polled; dropping it
/// does not stop the child (use [`RunControl::cancel`]).
pub struct RunJoin {
    inner: JoinHandle<RunOutput>,
    id: String,
    agent: String,
}

impl Future for RunJoin {
    type Output = RunOutput;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<RunOutput> {
        match Pin::new(&mut self.inner).poll(cx) {
            Poll::Ready(Ok(out)) => Poll::Ready(out),
            Poll::Ready(Err(e)) => Poll::Ready(RunOutput {
                id: self.id.clone(),
                agent: self.agent.clone(),
                status: RunStatus::Cancelled,
                exit_code: None,
                output: String::new(),
                stderr: format!("task run aborted: {e}"),
                schema: SchemaValidation::NoSchema,
                duration: Duration::ZERO,
                truncated: false,
            }),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl TaskRunner {
    /// Start one child in the background under the runner's concurrency
    /// limit and return its control handle plus a join future.
    ///
    /// Must be called inside a tokio runtime.
    #[must_use]
    pub fn start(&self, spec: TaskSpec, mode: ChildMode) -> (RunControl, RunJoin) {
        let id = spec.id.clone().unwrap_or_else(|| self.alloc_id());
        let (tx, rx) = mpsc::unbounded_channel();
        let shared = Arc::new(Shared::default());
        let control = RunControl {
            id: id.clone(),
            mode,
            tx,
            shared: shared.clone(),
        };
        let agent = spec.agent.clone();
        let inner = tokio::spawn(run_controlled(
            self.clone(),
            id.clone(),
            spec,
            mode,
            rx,
            shared,
        ));
        (control, RunJoin { inner, id, agent })
    }
}

async fn run_controlled(
    runner: TaskRunner,
    id: String,
    spec: TaskSpec,
    mode: ChildMode,
    mut rx: mpsc::UnboundedReceiver<Control>,
    shared: Arc<Shared>,
) -> RunOutput {
    let start = Instant::now();
    let mut steers: Vec<(String, SteerAck)> = Vec::new();
    let mut controls_open = true;
    let acquire = runner.semaphore.clone().acquire_owned();
    tokio::pin!(acquire);
    let permit = loop {
        tokio::select! {
            permit = &mut acquire => {
                break permit.expect("task runner semaphore is never closed by the runner itself");
            }
            ctl = rx.recv(), if controls_open => match ctl {
                Some(Control::Kill) => {
                    shared.set_phase(RunPhase::Settled);
                    return cancelled_output(id, spec.agent, None, String::new(), start);
                }
                Some(Control::Steer(message, ack)) => steers.push((message, ack)),
                None => controls_open = false,
            },
        }
    };
    shared.set_phase(RunPhase::Running);
    let out = match spawn_child(&runner.binary, &spec, mode).await {
        Err(e) => spawn_failure(id, spec, &runner.binary, &e, start),
        Ok(child) => {
            // Kill the whole process group when the run ends for any reason —
            // cancel/timeout paths call `kill_tree` themselves; the guard
            // covers normal exits (reaping surviving group members) and the
            // run task being dropped (parent runtime shutdown).
            let _group = GroupGuard::new(&child);
            match mode {
                ChildMode::Print => {
                    drive_print(child, &mut rx, controls_open, id, spec, start).await
                }
                ChildMode::Rpc => {
                    drive_rpc(
                        child,
                        &mut rx,
                        controls_open,
                        steers,
                        id,
                        spec,
                        start,
                        &shared,
                    )
                    .await
                }
            }
        }
    };
    drop(permit);
    let out = if !shared.take_killable() && out.status != RunStatus::Cancelled {
        // `cancel()` won the killable transition after the drive loop already
        // produced its output: the run settles Cancelled, consistently with
        // `cancel()` having returned `true`.
        cancelled_output(out.id, out.agent, out.exit_code, out.stderr, start)
    } else {
        out
    };
    shared.set_phase(RunPhase::Settled);
    out
}

/// SIGKILLs the child's process group when the run's control state drops —
/// the drop-path counterpart of `kill_tree` (`kill_on_drop` only signals the
/// leader pid, not `-pgid`).
struct GroupGuard(Option<i32>);

impl GroupGuard {
    fn new(child: &Child) -> Self {
        Self(child.id().and_then(|pid| i32::try_from(pid).ok()))
    }

    fn kill(&mut self) {
        if let Some(pgid) = self.0.take() {
            kill_group(pgid);
        }
    }
}

impl Drop for GroupGuard {
    fn drop(&mut self) {
        self.kill();
    }
}

/// SIGKILL every process in the group led by `pgid`.
fn kill_group(pgid: i32) {
    #[cfg(unix)]
    let _ = nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(-pgid),
        nix::sys::signal::Signal::SIGKILL,
    );
    #[cfg(not(unix))]
    let _ = pgid;
}

/// After a normal child exit, SIGKILL any surviving member of the child's
/// process group (pgid == child pid under `process_group(0)`): background
/// jobs and servers must not outlive the run or hold its pipes open. Takes
/// the pgid captured *before* `child.wait()`: tokio's `Child::id` returns
/// `None` once the child is reaped.
fn reap_group(pgid: Option<i32>) {
    if let Some(pgid) = pgid {
        kill_group(pgid);
    }
}

/// The child's process group id (valid only before the child is reaped).
fn child_pgid(child: &Child) -> Option<i32> {
    child.id().and_then(|pid| i32::try_from(pid).ok())
}

/// Spawn the child (own process group, killed on drop), retrying transient
/// fork/exec failures like [`TaskRunner::spawn`].
async fn spawn_child(binary: &Path, spec: &TaskSpec, mode: ChildMode) -> std::io::Result<Child> {
    let mut attempt = 0;
    loop {
        let mut command = Command::new(binary);
        match mode {
            ChildMode::Print => {
                command.arg("-p").arg(&spec.prompt);
            }
            ChildMode::Rpc => {
                command.arg("--mode").arg("rpc");
            }
        }
        command.current_dir(&spec.cwd);
        super::apply_child_policy(&mut command, spec);
        if let Some(model) = spec.model.as_deref().filter(|m| !m.is_empty()) {
            command.arg("--model").arg(model);
        }
        if let Some(tools) = spec.tools.as_ref().filter(|t| !t.is_empty()) {
            command.arg("--tools").arg(tools.join(","));
        }
        command
            .stdin(if mode == ChildMode::Rpc {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        command.process_group(0);
        match command.spawn() {
            Err(e) if attempt < 3 && is_transient_spawn_error(&e) => {
                attempt += 1;
                tokio::time::sleep(Duration::from_millis(50 * attempt)).await;
            }
            result => return result,
        }
    }
}

/// SIGKILL the child's whole process group (pgid == child pid under
/// `process_group(0)`), then the child itself as a fallback.
fn kill_tree(child: &mut Child) {
    if let Some(pid) = child.id().and_then(|pid| i32::try_from(pid).ok()) {
        kill_group(pid);
    }
    let _ = child.start_kill();
}

/// Sleep until `at` when given; the `None` case only exists so the branch can
/// sit in `select!` with a precondition (a disabled branch's future is never
/// awaited, so the `unwrap` inside is unreachable then).
async fn sleep_until_opt(at: Option<Instant>) {
    if let Some(at) = at {
        tokio::time::sleep_until(at.into()).await;
    }
}

/// Resolve every control still queued after the drive loop ended: pending
/// steers fail explicitly with [`SteerError::NotRunning`] (the run ended
/// before delivery), so callers never hold a stale delivery receipt.
fn drain_controls(rx: &mut mpsc::UnboundedReceiver<Control>) {
    while let Ok(ctl) = rx.try_recv() {
        if let Control::Steer(_, ack) = ctl {
            let _ = ack.send(Err(SteerError::NotRunning));
        }
    }
}

fn read_all<R: AsyncRead + Unpin + Send + 'static>(reader: Option<R>) -> JoinHandle<Vec<u8>> {
    tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(mut reader) = reader {
            let _ = reader.read_to_end(&mut buf).await;
        }
        buf
    })
}

/// Join a pipe reader; time-bounded on every path (after a kill only
/// [`KILL_DRAIN`] — a grandchild that left the process group could otherwise
/// hold the pipe open forever — and [`PIPES_DRAIN`] after a normal exit, so a
/// group escapee can never block the settle).
async fn join_text(reader: JoinHandle<Vec<u8>>, killed: bool) -> String {
    let bound = if killed { KILL_DRAIN } else { PIPES_DRAIN };
    let bytes = match tokio::time::timeout(bound, reader).await {
        Ok(Ok(bytes)) => bytes,
        _ => Vec::new(),
    };
    String::from_utf8_lossy(&bytes).into_owned()
}

fn cancelled_output(
    id: String,
    agent: String,
    exit_code: Option<i32>,
    stderr: String,
    start: Instant,
) -> RunOutput {
    let stderr = if stderr.trim().is_empty() {
        "run cancelled".to_string()
    } else {
        format!("run cancelled\n{stderr}")
    };
    RunOutput {
        id,
        agent,
        status: RunStatus::Cancelled,
        exit_code,
        output: String::new(),
        stderr,
        schema: SchemaValidation::NoSchema,
        duration: start.elapsed(),
        truncated: false,
    }
}

async fn drive_print(
    mut child: Child,
    rx: &mut mpsc::UnboundedReceiver<Control>,
    mut controls_open: bool,
    id: String,
    spec: TaskSpec,
    start: Instant,
) -> RunOutput {
    let stdout = read_all(child.stdout.take());
    let stderr = read_all(child.stderr.take());
    let pgid = child_pgid(&child);
    let waited = loop {
        tokio::select! {
            status = child.wait() => break Some(status),
            ctl = rx.recv(), if controls_open => match ctl {
                Some(Control::Kill) => break None,
                Some(Control::Steer(_, ack)) => {
                    let _ = ack.send(Err(SteerError::Unsupported));
                }
                None => controls_open = false,
            },
        }
    };
    drain_controls(rx);
    let killed = waited.is_none();
    let status = match waited {
        Some(status) => status,
        None => {
            kill_tree(&mut child);
            child.wait().await
        }
    };
    if !killed {
        // Normal exit: reap surviving group members (background jobs,
        // servers) before joining the pipes, so they cannot hold them open.
        reap_group(pgid);
    }
    let (stdout, stderr) = tokio::join!(join_text(stdout, killed), join_text(stderr, killed));
    let exit_code = status
        .as_ref()
        .ok()
        .and_then(std::process::ExitStatus::code);
    if killed {
        return cancelled_output(id, spec.agent, exit_code, stderr, start);
    }
    match status {
        Ok(status) => collect_output(id, spec, status.success(), exit_code, stdout, stderr, start),
        Err(e) => collect_output(
            id,
            spec,
            false,
            None,
            stdout,
            format!("{stderr}\nwait failed: {e}"),
            start,
        ),
    }
}

/// Write one JSON-lines command; a write failure closes the channel.
async fn send(stdin: &mut Option<ChildStdin>, frame: &Value) {
    let Some(pipe) = stdin.as_mut() else {
        return;
    };
    let mut line = frame.to_string();
    line.push('\n');
    if pipe.write_all(line.as_bytes()).await.is_err() || pipe.flush().await.is_err() {
        *stdin = None;
    }
}

fn steer_frame(message: &str) -> Value {
    json!({"type": "steer", "message": message})
}

#[allow(clippy::too_many_arguments)] // one run's full context; split only to keep select! borrows simple
async fn drive_rpc(
    mut child: Child,
    rx: &mut mpsc::UnboundedReceiver<Control>,
    mut controls_open: bool,
    steers: Vec<(String, SteerAck)>,
    id: String,
    spec: TaskSpec,
    start: Instant,
    shared: &Shared,
) -> RunOutput {
    let stderr = read_all(child.stderr.take());
    let pgid = child_pgid(&child);
    let mut stdin = child.stdin.take();
    let Some(stdout) = child.stdout.take() else {
        kill_tree(&mut child);
        let _ = child.wait().await;
        let stderr = join_text(stderr, true).await;
        return collect_output(
            id,
            spec,
            false,
            None,
            String::new(),
            format!("{stderr}\nchild stdout unavailable"),
            start,
        );
    };
    let mut lines = BufReader::new(stdout).lines();
    let mut rpc = RpcRun::default();
    let mut killed = false;
    let mut finishing_at: Option<Instant> = None;
    send(
        &mut stdin,
        &json!({"id": PROMPT_ID, "type": "prompt", "message": spec.prompt}),
    )
    .await;
    for (message, ack) in steers {
        send(&mut stdin, &steer_frame(&message)).await;
        let _ = ack.send(Ok(()));
    }
    loop {
        let finishing_deadline = finishing_at.map(|at| at + rpc_finishing_deadline());
        tokio::select! {
            line = lines.next_line() => match line {
                Ok(Some(line)) => {
                    for frame in rpc.on_line(&line) {
                        send(&mut stdin, &frame).await;
                    }
                    if rpc.finishing {
                        shared.set_phase(RunPhase::Finishing);
                        finishing_at.get_or_insert_with(Instant::now);
                    }
                    if rpc.close_stdin {
                        stdin = None;
                    }
                }
                Ok(None) | Err(_) => break,
            },
            _ = sleep_until_opt(finishing_deadline), if finishing_deadline.is_some() => {
                // Finishing-phase deadline: the child never answered the
                // final-text/stats requests (or kept stdout open). Kill it and
                // settle failed — never hang the run or hold its permit.
                kill_tree(&mut child);
                let _ = child.wait().await;
                let stderr = join_text(stderr, true).await;
                let error = format!(
                    "timed out in the Finishing phase: no final answer/session stats within {:?}",
                    rpc_finishing_deadline()
                );
                let stderr = if stderr.trim().is_empty() {
                    error
                } else {
                    format!("{error}\n{stderr}")
                };
                drain_controls(rx);
                return collect_output(id, spec, false, None, String::new(), stderr, start);
            }
            ctl = rx.recv(), if controls_open => match ctl {
                Some(Control::Kill) => {
                    killed = true;
                    break;
                }
                Some(Control::Steer(message, ack)) => {
                    if rpc.finishing {
                        // The prompt settled before the steer went out: fail
                        // the delivery explicitly instead of acknowledging it.
                        let _ = ack.send(Err(SteerError::NotRunning));
                    } else {
                        send(&mut stdin, &steer_frame(&message)).await;
                        let _ = ack.send(Ok(()));
                    }
                }
                None => controls_open = false,
            },
        }
    }
    drain_controls(rx);
    drop(stdin);
    if killed {
        kill_tree(&mut child);
    }
    let waited = tokio::time::timeout(RPC_EXIT_GRACE, child.wait()).await;
    let status = match waited {
        Ok(status) => status,
        Err(_) => {
            // The child ignored stdin EOF / its own settled session: kill the
            // whole group rather than waiting forever.
            kill_tree(&mut child);
            child.wait().await
        }
    };
    if !killed {
        // Normal exit: reap surviving group members (background jobs,
        // servers) before joining the pipes, so they cannot hold them open.
        reap_group(pgid);
    }
    let stderr = join_text(stderr, killed).await;
    let exit_code = status
        .as_ref()
        .ok()
        .and_then(std::process::ExitStatus::code);
    if killed {
        return cancelled_output(id, spec.agent, exit_code, stderr, start);
    }
    if let Some(tokens) = rpc.output_tokens {
        shared.output_tokens.store(tokens, Ordering::SeqCst);
    }
    let (ok, text, error) = rpc.outcome();
    let stderr = match error {
        Some(error) if stderr.trim().is_empty() => format!("error: {error}"),
        Some(error) => format!("error: {error}\n{stderr}"),
        None => stderr,
    };
    let exited_ok = status.as_ref().is_ok_and(std::process::ExitStatus::success);
    collect_output(id, spec, ok && exited_ok, exit_code, text, stderr, start)
}

/// Bookkeeping for one RPC-mode prompt (pure, so the frame protocol is
/// unit-testable without a child).
#[derive(Default, Debug)]
struct RpcRun {
    /// `prompt_result` status (`completed` / `error` / `aborted`) + error.
    result: Option<(String, Option<String>)>,
    /// The `prompt` command itself was rejected.
    prompt_error: Option<String>,
    /// `get_last_assistant_text` answer.
    final_text: Option<String>,
    /// Last streamed assistant `message_end` (fallback when the final-text
    /// response overflowed the transport).
    fallback_text: Option<String>,
    fallback_error: Option<String>,
    /// `get_session_stats` `tokens.output`.
    output_tokens: Option<u64>,
    /// Prompt settled: final-answer requests sent, steering closed.
    finishing: bool,
    /// Everything collected: close stdin so the child exits.
    close_stdin: bool,
}

fn error_text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) => Some(s.clone()),
        Value::Object(map) => map
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    }
}

impl RpcRun {
    /// Consume one stdout line; returns the commands to write back.
    fn on_line(&mut self, line: &str) -> Vec<Value> {
        let Ok(frame) = serde_json::from_str::<Value>(line) else {
            return Vec::new();
        };
        match frame.get("type").and_then(Value::as_str) {
            Some("prompt_result") => {
                if self.result.is_none() {
                    let status = frame
                        .get("status")
                        .and_then(Value::as_str)
                        .unwrap_or("completed")
                        .to_string();
                    self.result = Some((status, error_text(frame.get("error"))));
                }
                if frame.get("sessionSettled") == Some(&Value::Bool(true)) {
                    self.finish()
                } else {
                    Vec::new()
                }
            }
            Some("session_settled") if self.result.is_some() => self.finish(),
            Some("response") => {
                let success = frame.get("success") == Some(&Value::Bool(true));
                match frame.get("command").and_then(Value::as_str) {
                    Some("prompt") if !success => {
                        self.prompt_error = Some(
                            error_text(frame.get("error"))
                                .unwrap_or_else(|| "prompt rejected".to_string()),
                        );
                        self.finishing = true;
                        self.close_stdin = true;
                        Vec::new()
                    }
                    // A slash command consumed the prompt without a run: no
                    // prompt_result follows, collect right away.
                    Some("prompt")
                        if frame.pointer("/data/agentInvoked") == Some(&Value::Bool(false)) =>
                    {
                        self.result
                            .get_or_insert_with(|| ("completed".to_string(), None));
                        self.finish()
                    }
                    Some("get_last_assistant_text") => {
                        if success {
                            self.final_text = frame
                                .pointer("/data/text")
                                .and_then(Value::as_str)
                                .map(str::to_string);
                        }
                        Vec::new()
                    }
                    Some("get_session_stats") => {
                        if success {
                            self.output_tokens =
                                frame.pointer("/data/tokens/output").and_then(Value::as_u64);
                        }
                        self.close_stdin = true;
                        Vec::new()
                    }
                    _ => Vec::new(),
                }
            }
            // No UI behind this transport: refuse dialogs instead of hanging
            // the child (print mode has no dialog either).
            Some("extension_ui_request") => frame.get("id").map_or_else(Vec::new, |id| {
                vec![json!({"type": "extension_ui_response", "id": id, "cancelled": true})]
            }),
            Some("message_end") => {
                let message = frame.get("message");
                if message.and_then(|m| m.get("role")).and_then(Value::as_str) == Some("assistant")
                {
                    let text = message
                        .and_then(|m| m.get("content"))
                        .and_then(Value::as_array)
                        .map(|blocks| {
                            blocks
                                .iter()
                                .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                                .filter_map(|b| b.get("text").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_default();
                    self.fallback_text = Some(text);
                    self.fallback_error = message
                        .and_then(|m| m.get("errorMessage"))
                        .and_then(Value::as_str)
                        .map(str::to_string);
                }
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    fn finish(&mut self) -> Vec<Value> {
        if self.finishing {
            return Vec::new();
        }
        self.finishing = true;
        vec![
            json!({"id": FINAL_TEXT_ID, "type": "get_last_assistant_text"}),
            json!({"id": STATS_ID, "type": "get_session_stats"}),
        ]
    }

    /// `(success, final text, error)` once the child is done.
    fn outcome(&self) -> (bool, String, Option<String>) {
        if let Some(error) = &self.prompt_error {
            return (false, String::new(), Some(error.clone()));
        }
        let Some((status, error)) = &self.result else {
            return (
                false,
                String::new(),
                Some("child exited before the prompt settled".to_string()),
            );
        };
        let text = self
            .final_text
            .clone()
            .or_else(|| self.fallback_text.clone())
            .unwrap_or_default();
        match status.as_str() {
            "completed" => (true, text, None),
            "aborted" => (
                false,
                text,
                Some(error.clone().unwrap_or_else(|| "aborted".to_string())),
            ),
            _ => (
                false,
                text,
                Some(
                    error
                        .clone()
                        .or_else(|| self.fallback_error.clone())
                        .unwrap_or_else(|| "prompt failed".to_string()),
                ),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SchemaMode;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    fn fake_binary(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("fake-omp-rs");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "#!/bin/sh\n{body}").unwrap();
        drop(f);
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    fn spec(prompt: &str, cwd: &Path) -> TaskSpec {
        TaskSpec {
            id: Some("w1".to_string()),
            agent: "task".to_string(),
            prompt: prompt.to_string(),
            cwd: cwd.to_path_buf(),
            output_schema: None,
            schema_mode: SchemaMode::Permissive,
            model: None,
            tools: None,
            depth: 0,
            extra_args: Vec::new(),
        }
    }

    #[tokio::test]
    async fn print_mode_start_completes_like_spawn() {
        let dir = tempfile::tempdir().unwrap();
        let bin = fake_binary(dir.path(), r#"echo "$2""#);
        let runner = TaskRunner::new(&bin, 1);
        let (control, join) = runner.start(spec("hello", dir.path()), ChildMode::Print);
        assert_eq!(control.steer("x").await, Err(SteerError::Unsupported));
        let out = join.await;
        assert_eq!(out.status, RunStatus::Completed);
        assert_eq!(out.output.trim(), "hello");
        assert_eq!(control.phase(), RunPhase::Settled);
        assert!(!control.cancel(), "cancel after settle is a no-op");
    }

    #[tokio::test]
    async fn children_never_auto_title() {
        let dir = tempfile::tempdir().unwrap();
        let bin = fake_binary(dir.path(), r#"echo "title-off=$OMP_RS_NO_TITLE""#);
        let runner = TaskRunner::new(&bin, 1);
        let (_control, join) = runner.start(spec("hello", dir.path()), ChildMode::Print);
        assert_eq!(join.await.output.trim(), "title-off=1");
        assert_eq!(
            runner.spawn(spec("hello", dir.path())).await.output.trim(),
            "title-off=1"
        );
    }

    #[tokio::test]
    async fn cancel_kills_running_child_and_its_group() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("survived");
        // A grandchild in the same process group must die with the child.
        let bin = fake_binary(
            dir.path(),
            &format!("(sleep 1; touch {}) &\nsleep 30", marker.display()),
        );
        let runner = TaskRunner::new(&bin, 1);
        let (control, join) = runner.start(spec("x", dir.path()), ChildMode::Print);
        while control.phase() == RunPhase::Queued {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        let started = Instant::now();
        assert!(control.cancel());
        assert!(!control.cancel(), "second cancel reports nothing new");
        let out = join.await;
        assert_eq!(out.status, RunStatus::Cancelled);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(runner.available_permits(), 1, "permit released");
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert!(!marker.exists(), "grandchild survived the group kill");
    }

    #[tokio::test]
    async fn cancel_while_queued_never_spawns() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("spawned");
        let bin = fake_binary(
            dir.path(),
            &format!("echo $$ >> {}; sleep 1", log.display()),
        );
        let runner = TaskRunner::new(&bin, 1);
        let (_first, first_join) = runner.start(spec("a", dir.path()), ChildMode::Print);
        let (second, second_join) = runner.start(
            TaskSpec {
                id: Some("w2".to_string()),
                ..spec("b", dir.path())
            },
            ChildMode::Print,
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(second.phase(), RunPhase::Queued);
        assert!(second.cancel());
        let out = second_join.await;
        assert_eq!(out.status, RunStatus::Cancelled);
        assert_eq!(first_join.await.status, RunStatus::Completed);
        let spawned = std::fs::read_to_string(&log).unwrap();
        assert_eq!(spawned.lines().count(), 1);
    }

    /// Fake `--mode rpc` child: reads the prompt, waits for one steer, then
    /// reports a settled prompt whose final text echoes the steer, and answers
    /// the final-text / stats requests.
    const FAKE_RPC: &str = r#"
[ "$1" = "--mode" ] && [ "$2" = "rpc" ] || { echo "bad args: $*" >&2; exit 2; }
read -r prompt
echo '{"type":"ready"}'
echo '{"type":"response","command":"prompt","success":true,"id":"omp-task-prompt"}'
read -r steer
answer=unsteered
case "$steer" in *'"steer"'*) case "$steer" in *redirect*) answer=steered;; esac;; esac
echo '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"partial"}]}}'
echo '{"type":"prompt_result","agentInvoked":true,"status":"completed","sessionSettled":true}'
read -r q1
read -r q2
echo "{\"type\":\"response\",\"command\":\"get_last_assistant_text\",\"success\":true,\"data\":{\"text\":\"$answer\"}}"
echo '{"type":"response","command":"get_session_stats","success":true,"data":{"tokens":{"output":42}}}'
read -r eof || exit 0
"#;

    #[tokio::test]
    async fn rpc_mode_delivers_steer_and_collects_final_text() {
        let dir = tempfile::tempdir().unwrap();
        let bin = fake_binary(dir.path(), FAKE_RPC);
        let runner = TaskRunner::new(&bin, 1);
        let (control, join) = runner.start(spec("do it", dir.path()), ChildMode::Rpc);
        while control.phase() == RunPhase::Queued {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        control.steer("redirect please").await.unwrap();
        let out = join.await;
        assert_eq!(out.status, RunStatus::Completed, "stderr: {}", out.stderr);
        assert_eq!(out.output, "steered");
        assert_eq!(control.output_tokens(), 42);
        assert_eq!(control.steer("late").await, Err(SteerError::NotRunning));
    }

    #[test]
    fn rpc_error_result_fails_with_message() {
        let mut rpc = RpcRun::default();
        assert!(rpc.on_line(r#"{"type":"prompt_result","status":"error","error":{"message":"boom"},"sessionSettled":false}"#).is_empty());
        assert!(!rpc.finishing, "unsettled session keeps steering open");
        let requests = rpc.on_line(r#"{"type":"session_settled"}"#);
        assert_eq!(requests.len(), 2);
        assert!(
            rpc.on_line(r#"{"type":"session_settled"}"#).is_empty(),
            "finish is once"
        );
        let (ok, _, error) = rpc.outcome();
        assert!(!ok);
        assert_eq!(error.as_deref(), Some("boom"));
    }

    #[test]
    fn rpc_ui_requests_are_refused_and_rejected_prompt_closes() {
        let mut rpc = RpcRun::default();
        let reply = rpc.on_line(r#"{"type":"extension_ui_request","id":"ui-1","method":"ask"}"#);
        assert_eq!(
            reply,
            vec![json!({"type":"extension_ui_response","id":"ui-1","cancelled":true})]
        );
        rpc.on_line(r#"{"type":"response","command":"prompt","success":false,"error":"no model"}"#);
        assert!(rpc.close_stdin);
        assert_eq!(
            rpc.outcome(),
            (false, String::new(), Some("no model".to_string()))
        );
        let empty = RpcRun::default();
        assert!(!empty.outcome().0, "no prompt_result = failure");
    }
}
