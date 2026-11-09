//! Agent-family host ops for eval cells: the host half of the `agent()`,
//! `completion()`, `judge()`, `judge_batch()`, `workpool()`, `budget`, `wait()`,
//! `status()` and `cancel()` prelude helpers (TS `eval/agent-bridge.ts`,
//! `eval/completion-bridge.ts`, `eval/judgment-bridge.ts`,
//! `eval/judgment-batch-bridge.ts`, `eval/workpool-bridge.ts`,
//! `eval/budget-bridge.ts`, `eval/handle-bridge.ts`; payload shapes verbatim
//! from `crates/omp-eval/src/prelude_agents.py`).
//!
//! omp-rs subagents are `omp-rs --mode rpc` subprocesses driven by
//! [`omp_task`]'s controllable runner ([`omp_task::TaskRunner::start`], the
//! same machinery the `task` tool uses): agent handles are process-local
//! registry entries whose runs proceed in the background; `cancel()` kills the
//! child's process group, and `write agent://<id>` steers a running child
//! through its RPC stdin. A subagent exits once its prompt settles, so unlike
//! TS keep-alive agents a finished one cannot be revived by a message, and
//! `workpool()` runs every pushed item as a fresh agent (TS
//! `eval.workpool.freshAgents` behavior). Judgments reuse [`omp_providers::judge`]'s resolved chain;
//! completions run one-shot against the session's live provider target
//! ([`crate::agent::Target`], installed by [`set_runtime_hooks`]) or a resolved
//! `@smol`/`@slow` role.

use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::{Duration, Instant};

use omp_providers::{AgentDir, ResolvedModel};
use omp_providers::{JudgeRequest, JudgmentCache};
use pa_agent::abort::AbortSignal;
use pa_ai::types::{SimpleStreamOptions, SimpleToolChoice};
use pa_types::ai::{
    AssistantContentBlock, Context, Message, Model, StopReason, Tool, UserContent, UserMessage,
};
use serde_json::{json, Map, Value};
use tokio::sync::{Notify, Semaphore};
use tokio_util::sync::CancellationToken;

use super::eval_bridge::OpCtx;
use super::Env;

#[path = "eval_agents/completion.rs"]
mod completion;
#[path = "eval_agents/judge.rs"]
mod judge;
#[path = "eval_agents/judge_batch.rs"]
mod judge_batch;
#[cfg(test)]
#[path = "eval_agents/test_support.rs"]
mod test_support;
#[path = "eval_agents/workpool.rs"]
mod workpool;

use completion::{execute_completion, resolve_query_candidates, run_completion};
use judge::run_judgment_op;
use judge_batch::{run_judge_batch_op, JudgmentBatchRun};
use workpool::{budget_snapshot, run_cancel, run_status, run_wait, run_workpool};

/// TS `EVAL_HANDLE_CONCURRENCY` (`completion-bridge.ts`): process-wide ceiling
/// on eval model requests executing at once, shared by `completion()` handles,
/// `judge()`, and `judge_batch()` items.
const EVAL_HANDLE_CONCURRENCY: usize = 32;

/// Shared eval model-request slots (TS `evalRequestSlots`).
static REQUEST_SLOTS: LazyLock<Semaphore> =
    LazyLock::new(|| Semaphore::new(EVAL_HANDLE_CONCURRENCY));

/// Synthetic tool the model is forced to call when a `schema` is supplied
/// (TS `STRUCTURED_TOOL_NAME`).
const STRUCTURED_TOOL_NAME: &str = "respond";

/// Process-wide ceiling on eval subagent children running at once; also the
/// worker limit `workpool()` reports. Default `min(available_parallelism, 8)`
/// (TS `task.maxConcurrency` protocol default is overridden per deployment by
/// that setting, which the `task` tool resolves; the eval launcher keeps the
/// process-wide auto default).
fn subagent_concurrency() -> usize {
    omp_task::TaskRunner::default_max_concurrency()
}

/// Error text of an *agent* handle cancelled through `cancel()`. TS agent
/// jobs go through `manager.cancel(job.id)` with no reason, so the snapshot
/// carries the killed run's own abort text ("run cancelled…"), never the
/// completion-only "Cancelled by eval handle" (TS `ToolAbortError`).
const AGENT_CANCELLED_TEXT: &str = "run cancelled";

/// Poll cadence while `wait` blocks on handle settlement (the registry is a
/// snapshot store like `jobs::JobRegistry`, so `wait` observes it on a tick).
const WAIT_TICK: Duration = Duration::from_millis(100);

// ---------------------------------------------------------------------------
// Runtime hooks
// ---------------------------------------------------------------------------

/// Runtime-only session data the eval bridges need but the tool `Env` does not
/// carry: the live provider target (`set_model` swaps it, so reads must go
/// through the lock each time).
pub struct EvalRuntimeHooks {
    /// The session's current model/provider target (TS `ToolSession` model
    /// registry + active model for the `default` completion tier).
    pub target: crate::agent::Target,
}

static HOOKS: OnceLock<RwLock<Option<EvalRuntimeHooks>>> = OnceLock::new();

/// Install the process-global eval runtime hooks. Call once at session
/// startup, right after [`crate::agent::build_agent`] returns its target.
pub fn set_runtime_hooks(target: crate::agent::Target) {
    let slot = HOOKS.get_or_init(|| RwLock::new(None));
    *slot.write() = Some(EvalRuntimeHooks { target });
}

fn hooks() -> Result<parking_lot::RwLockReadGuard<'static, Option<EvalRuntimeHooks>>, String> {
    HOOKS
        .get()
        .ok_or_else(|| "eval runtime is not initialized (set_runtime_hooks not called)".to_string())
        .map(|slot| slot.read())
}

/// The live provider target (TS `session.getActiveModel`-backed `default` tier).
fn provider_target() -> Result<crate::adapter::ProviderTarget, String> {
    let guard = hooks()?;
    let hooks = guard.as_ref().ok_or_else(|| {
        "eval runtime is not initialized (set_runtime_hooks not called)".to_string()
    })?;
    let target = hooks.target.read().clone();
    target.ok_or_else(|| "completion() has no provider target set.".to_string())
}

// ---------------------------------------------------------------------------
// Handle registry (agents + completions)
// ---------------------------------------------------------------------------

/// Terminal or current state of one handle (TS `EvalHandleState`).
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum HandleStatus {
    #[default]
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl HandleStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Mutable settle state behind one handle (job and completion entries share
/// the snapshot shape the preludes consume).
#[derive(Default)]
struct HandleInner {
    status: HandleStatus,
    settled: bool,
    text: Option<String>,
    data: Option<Value>,
    error: Option<String>,
}

struct Handle {
    kind: &'static str,
    id: String,
    inner: Mutex<HandleInner>,
    notify: Notify,
    /// Settled exactly once the run/completion future backing this handle
    /// fully resolved (independently of `settled`, which `cancel()` flips
    /// immediately): the abort path of `wait()` joins these before erroring,
    /// like TS `Promise.allSettled(handle.job.promise)`.
    run_finished: std::sync::atomic::AtomicBool,
    run_done: Notify,
    /// Completion handles are abortable through the pa-ai cancellation token.
    cancel: CancellationToken,
    /// Agent handles: kill/steer surface of the subagent run (set at launch).
    run: Mutex<Option<Arc<dyn RunCtl>>>,
}

impl Handle {
    fn new(kind: &'static str, id: String) -> Arc<Self> {
        Arc::new(Self {
            kind,
            id,
            inner: Mutex::new(HandleInner {
                status: HandleStatus::Running,
                settled: false,
                text: None,
                data: None,
                error: None,
            }),
            notify: Notify::new(),
            run_finished: std::sync::atomic::AtomicBool::new(false),
            run_done: Notify::new(),
            cancel: CancellationToken::new(),
            run: Mutex::new(None),
        })
    }

    /// Mark the backing run/completion future resolved (once).
    fn mark_run_finished(&self) {
        self.run_finished
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.run_done.notify_waiters();
    }

    /// Wait until the backing run/completion future resolved.
    async fn wait_run_finished(&self) {
        loop {
            if self.run_finished.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            let notified = self.run_done.notified();
            if self.run_finished.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            notified.await;
        }
    }

    fn is_settled(&self) -> bool {
        self.inner.lock().settled
    }

    fn run_ctl(&self) -> Option<Arc<dyn RunCtl>> {
        self.run.lock().clone()
    }

    /// Settle once: the first terminal state wins (a `cancel()` settles the
    /// handle immediately; the killed run's own settle is then ignored).
    fn settle(
        &self,
        status: HandleStatus,
        text: Option<String>,
        data: Option<Value>,
        error: Option<String>,
    ) {
        {
            let mut inner = self.inner.lock();
            if inner.settled {
                return;
            }
            inner.status = status;
            inner.settled = true;
            inner.text = text;
            inner.data = data;
            inner.error = error;
        }
        self.notify.notify_waiters();
    }

    /// TS `EvalHandleSnapshot` (`handle-bridge.ts`): `{kind, id, status}` plus
    /// `text`/`data` on completion and `error` on failure/cancellation.
    fn snapshot(&self) -> Value {
        let inner = self.inner.lock();
        let mut snap = json!({"kind": self.kind, "id": self.id, "status": inner.status.as_str()});
        if let Some(text) = &inner.text {
            snap["text"] = Value::String(text.clone());
        }
        if let Some(data) = &inner.data {
            snap["data"] = data.clone();
        }
        if let Some(error) = &inner.error {
            snap["error"] = Value::String(error.clone());
        }
        snap
    }
}

static AGENT_HANDLES: LazyLock<Mutex<HashMap<String, Arc<Handle>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

static COMPLETION_HANDLES: LazyLock<Mutex<HashMap<String, Arc<Handle>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

static JUDGE_BATCHES: LazyLock<Mutex<HashMap<String, Arc<JudgmentBatchRun>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn mint_id(prefix: &str) -> String {
    format!("{prefix}-{}", NEXT_ID.fetch_add(1, Ordering::Relaxed))
}

fn parse_handle_ref(value: &Value) -> Result<(&'static str, String), String> {
    let item = value
        .as_object()
        .ok_or_else(|| "handle must be an object with kind and id".to_string())?;
    let kind = match item.get("kind").and_then(Value::as_str) {
        Some("agent") => "agent",
        Some("completion") => "completion",
        _ => {
            return Err("handle must contain kind agent|completion and a non-empty id".to_string());
        }
    };
    let id = item
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            "handle must contain kind agent|completion and a non-empty id".to_string()
        })?;
    Ok((kind, id.to_string()))
}

fn lookup_handle(kind: &str, id: &str) -> Result<Arc<Handle>, String> {
    let registry = if kind == "agent" {
        &AGENT_HANDLES
    } else {
        &COMPLETION_HANDLES
    };
    registry.lock().get(id).cloned().ok_or_else(|| {
        if kind == "agent" {
            format!("unknown agent handle {id}; result expired — read agent://{id}")
        } else {
            format!("unknown completion handle {id}; result expired")
        }
    })
}

// ---------------------------------------------------------------------------
// Op dispatch
// ---------------------------------------------------------------------------

/// Answer one eval agent-family host op. `None` = not an agent op (the bridge
/// then treats it as unknown); `Some(Err(..))` = the op is known but failed,
/// never silent success.
pub async fn handle(
    env: &Arc<Env>,
    op: &str,
    args: Value,
    ctx: &OpCtx,
) -> Option<Result<Value, String>> {
    match op {
        "__agent__" => Some(run_agent(env, &args, ctx)),
        "__completion__" => Some(run_completion(&args, ctx).await),
        "__judge__" => Some(run_judgment_op(&args, ctx).await),
        "__judge_batch__" => Some(run_judge_batch_op(&args, ctx).await),
        "__workpool__" => Some(run_workpool(&args, ctx)),
        "__budget__" => Some(Ok(budget_snapshot())),
        "__llm_query__" => Some(super::rlm::llm_query_op(&args, ctx).await),
        "__rlm_stats__" => Some(Ok(super::rlm::stats_snapshot())),
        "__rlm_context__" => Some(Ok(super::rlm::context_op())),
        op if op.starts_with("__lam_") || op.starts_with("__rubric_") => {
            super::rlm_combinators::handle(op, &args, ctx).await
        }
        "__wait__" => Some(run_wait(&args, ctx).await),
        "__status__" => Some(run_status(&args)),
        "__cancel__" => Some(run_cancel(&args)),
        _ => None,
    }
}

/// TS `AgentHandle.send` / `write agent://<id>` (`agent-protocol.ts` →
/// `irc/messaging.ts` `executeSend`): a running eval subagent (`agent()` or a
/// workpool worker) gets the message as a steer through its RPC stdin, injected
/// at its next turn boundary ("Delivered to <id>."); `agent://all` broadcasts
/// to every running one. Deviation: TS keep-alive agents are revived by a
/// message after they finish; omp-rs subagents exit when their prompt
/// settles, so a finished agent is a delivery failure.
///
/// # Errors
/// TS delivery-failure texts (`Failed: <id> is not running. …`) and argument
/// errors (empty content, JSON-path suffix, missing recipient).
pub async fn send_to_agent(id: &str, content: &str) -> Result<Value, String> {
    deliver_message(id, content).await.map(Value::String)
}

pub(crate) async fn deliver_message(to: &str, content: &str) -> Result<String, String> {
    validate_message(to, content)?;
    // Inside a `task` child the parent runs the bus: `parent` is only reachable
    // through it, and ids that are not local handles (siblings) go there too.
    let Some(bus) = super::a2a::Client::from_env() else {
        return deliver_local(to, content, None).await;
    };
    if to == "parent" {
        return bus.send(to, content).await;
    }
    if to == "all" {
        // Local eval peers first (when any), plus every sibling via the parent.
        let local = if has_live_agents() {
            Some(deliver_local(to, content, None).await)
        } else {
            None
        };
        let remote = bus.send(to, content).await;
        return match (local, remote) {
            (Some(Ok(local)), Ok(remote)) => Ok(format!("{local}\n{remote}")),
            (Some(Ok(text)), Err(_)) | (None | Some(Err(_)), Ok(text)) => Ok(text),
            (Some(Err(error)), Err(_)) | (None, Err(error)) => Err(error),
        };
    }
    if agent_id_taken(to) {
        return deliver_local(to, content, None).await;
    }
    bus.send(to, content).await
}

fn validate_message(to: &str, content: &str) -> Result<(), String> {
    if to.is_empty() {
        return Err("agent:// URL requires a recipient: agent://<id>".to_string());
    }
    if to.contains('/') {
        return Err("agent:// message target cannot have a JSON-path suffix.".to_string());
    }
    if content.trim().is_empty() {
        return Err("agent:// messages require non-empty content.".to_string());
    }
    Ok(())
}

/// Deliver a message a child sent over the bus: steer a registered run (or all
/// of them except the sender).
pub(crate) async fn deliver_from_peer(
    to: &str,
    content: &str,
    sender: &str,
) -> Result<String, String> {
    validate_message(to, content)?;
    deliver_local(to, content, Some(sender)).await
}

fn lookup_agent(id: &str) -> Option<Arc<Handle>> {
    AGENT_HANDLES.lock().get(id).cloned()
}

/// How long a local `write agent://<id>` waits for an unknown id to appear.
const UNKNOWN_ID_GRACE: Duration = Duration::from_secs(2);

fn has_live_agents() -> bool {
    AGENT_HANDLES
        .lock()
        .values()
        .any(|handle| !handle.is_settled())
}

async fn deliver_local(to: &str, content: &str, exclude: Option<&str>) -> Result<String, String> {
    if to == "all" {
        return broadcast_message(content, exclude).await;
    }
    let mut handle = lookup_agent(to);
    if handle.is_none() && exclude.is_none() && !super::a2a::looks_like_pane_id(to) {
        // A `write` issued in the same tool batch as the `task` call that
        // creates the recipient can run first: give registration a short,
        // bounded grace before declaring the id unknown. Pane ids are never
        // local eval agents — route them straight to the hub.
        let deadline = tokio::time::Instant::now() + UNKNOWN_ID_GRACE;
        while handle.is_none() && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(25)).await;
            handle = lookup_agent(to);
        }
    }
    match steer_handle(handle.as_deref(), to, content).await {
        Ok(()) => Ok(format!("Delivered to {to}.")),
        Err(error) => {
            // Not a local run: a pending hub question from `to` becomes an
            // `answer{}`, a pane/unknown id becomes a hub `send{}` — the hub
            // seam reports queued/delivered or a clear error, never silence.
            if let Some(outcome) = super::a2a::route_external_message(to, content).await {
                return outcome;
            }
            Err(format!("Failed: {to} is not running. {error}"))
        }
    }
}

/// TS broadcast receipt text over every running eval subagent.
// invariant: write!/writeln! to a String is infallible.
#[allow(clippy::expect_used)]
async fn broadcast_message(content: &str, exclude: Option<&str>) -> Result<String, String> {
    use std::fmt::Write as _;
    let mut targets: Vec<(String, Arc<Handle>)> = AGENT_HANDLES
        .lock()
        .iter()
        .filter(|(id, handle)| !handle.is_settled() && Some(id.as_str()) != exclude)
        .map(|(id, handle)| (id.clone(), handle.clone()))
        .collect();
    if targets.is_empty() {
        return Ok("No live peers to broadcast to.".to_string());
    }
    targets.sort_by(|a, b| a.0.cmp(&b.0));
    let mut text = String::new();
    let mut delivered = 0usize;
    for (id, handle) in &targets {
        match steer_handle(Some(&**handle), id, content).await {
            Ok(()) => {
                delivered += 1;
                write!(text, "\n- {id}: injected").expect("writing to a String is infallible");
            }
            Err(error) => write!(text, "\n- {id}: failed — {error}")
                .expect("writing to a String is infallible"),
        }
    }
    text = format!(
        "Broadcast delivered to {delivered} of {} peer(s).{}",
        targets.len(),
        text
    );
    if delivered == 0 {
        Err(text)
    } else {
        Ok(text)
    }
}

/// Steer one agent handle's live run (the TS bus receipt error on failure).
async fn steer_handle(handle: Option<&Handle>, to: &str, content: &str) -> Result<(), String> {
    let Some(handle) = handle else {
        return Err(format!(
            "Unknown agent \"{to}\" — check the subagent roster or read history:// for known peers."
        ));
    };
    let finished = || {
        format!(
            "Agent \"{to}\" already finished; omt eval subagents exit when their prompt settles and cannot be revived (read its result with wait()/status())."
        )
    };
    if handle.is_settled() {
        return Err(finished());
    }
    // A handle is registered a moment before its run is attached (launch
    // window); give that hand-over a short, bounded grace.
    let mut run = handle.run_ctl();
    for _ in 0..100 {
        if run.is_some() || handle.is_settled() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        run = handle.run_ctl();
    }
    let Some(run) = run else {
        return Err(format!("Agent \"{to}\" has no live session."));
    };
    run.steer(content).await.map_err(|error| match error {
        omp_task::SteerError::NotRunning => finished(),
        omp_task::SteerError::Unsupported => format!("Agent \"{to}\" has no input channel."),
    })
}

// ---------------------------------------------------------------------------
// Subagent runs (omp-task controllable runner)
// ---------------------------------------------------------------------------

/// Kill/steer surface of one launched subagent run ([`omp_task::RunControl`]
/// in production, a mock in tests).
pub(crate) trait RunCtl: Send + Sync {
    /// Queue a steering message for the running child; the future is a
    /// delivery receipt (resolves once the drive loop wrote the frame, or
    /// fails explicitly when the run is finishing/already settled).
    fn steer<'a>(
        &'a self,
        message: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), omp_task::SteerError>> + Send + 'a>>;
    /// Kill the run; `false` when it already settled or was cancelled.
    fn cancel(&self) -> bool;
    /// Output tokens the child session reported once settled.
    fn output_tokens(&self) -> u64;
}

impl RunCtl for omp_task::RunControl {
    fn steer<'a>(
        &'a self,
        message: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), omp_task::SteerError>> + Send + 'a>> {
        Box::pin(omp_task::RunControl::steer(self, message))
    }

    fn cancel(&self) -> bool {
        omp_task::RunControl::cancel(self)
    }

    fn output_tokens(&self) -> u64 {
        omp_task::RunControl::output_tokens(self)
    }
}

type RunFuture = Pin<Box<dyn Future<Output = omp_task::RunOutput> + Send>>;

/// Starts subagent runs (the subprocess runner in production, a mock in tests).
pub(crate) trait Launcher: Send + Sync {
    fn launch(&self, spec: omp_task::TaskSpec) -> (Arc<dyn RunCtl>, RunFuture);
}

/// `omp-rs --mode rpc` children under one shared concurrency ceiling.
pub(crate) struct SubprocessLauncher {
    runner: omp_task::TaskRunner,
}

impl SubprocessLauncher {
    /// Test hook: a launcher over an explicit fake binary, so integration
    /// tests exercise the real subprocess env hand-off.
    #[cfg(test)]
    #[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
    pub(crate) fn for_test(binary: &std::path::Path) -> Self {
        Self {
            runner: omp_task::TaskRunner::new(binary, 4),
        }
    }
}

impl Launcher for SubprocessLauncher {
    fn launch(&self, spec: omp_task::TaskSpec) -> (Arc<dyn RunCtl>, RunFuture) {
        let (control, join) = self.runner.start(spec, omp_task::ChildMode::Rpc);
        let control: Arc<dyn RunCtl> = Arc::new(control);
        let join: RunFuture = Box::pin(join);
        (control, join)
    }
}

/// Process-wide launcher: `agent()` runs and workpool turns share one runner,
/// so the [`subagent_concurrency()`] ceiling holds across both.
static LAUNCHER: LazyLock<Arc<dyn Launcher>> = LazyLock::new(|| {
    let binary = crate::respawn_exe();
    Arc::new(SubprocessLauncher {
        runner: omp_task::TaskRunner::new(binary, subagent_concurrency()),
    }) as Arc<dyn Launcher>
});

/// Output tokens spent by eval subagents (`agent()` + workpool workers) in
/// this process (TS `recordEvalSubagentUsage`).
static EVAL_SUBAGENT_OUTPUT: AtomicU64 = AtomicU64::new(0);

/// Launch `spec` as the run behind agent `handle`; the returned future
/// resolves to the run output once the child settled (usage recorded).
fn launch_agent(handle: &Handle, spec: omp_task::TaskSpec, launcher: &dyn Launcher) -> RunFuture {
    let (run, done) = launcher.launch(spec);
    *handle.run.lock() = Some(run.clone());
    Box::pin(async move {
        let output = done.await;
        EVAL_SUBAGENT_OUTPUT.fetch_add(run.output_tokens(), Ordering::Relaxed);
        output
    })
}

/// Settle an agent handle from its run (TS job settle + `agentSnapshot`); a
/// run whose output violates the declared schema fails in strict mode.
fn settle_agent(
    handle: &Handle,
    output: &omp_task::RunOutput,
    schema_mode: omp_task::SchemaMode,
    schema_wanted: bool,
) {
    let ok = output.status == omp_task::RunStatus::Completed && !output.schema_failed(schema_mode);
    let status = match output.status {
        _ if ok => HandleStatus::Completed,
        omp_task::RunStatus::Cancelled => HandleStatus::Cancelled,
        _ => HandleStatus::Failed,
    };
    let text = if output.output.trim().is_empty() {
        output.stderr.trim().to_string()
    } else {
        output.output.clone()
    };
    if status == HandleStatus::Cancelled {
        // TS agent jobs are cancelled through `manager.cancel(job.id)` with
        // no reason: the snapshot carries the killed run's own abort text,
        // not the completion-only "Cancelled by eval handle". `cancel_handle`
        // may already have settled the handle with the same text, whichever
        // lands first.
        let error = if output.stderr.trim().is_empty() {
            AGENT_CANCELLED_TEXT.to_string()
        } else {
            output.stderr.trim().to_string()
        };
        handle.settle(status, None, None, Some(error));
    } else if status == HandleStatus::Completed {
        let data = if schema_wanted {
            serde_json::from_str::<Value>(output.output.trim()).ok()
        } else {
            None
        };
        handle.settle(status, Some(text), data, None);
    } else {
        let error = omp_task::envelope::format_task_result(output)
            .error
            .unwrap_or_else(|| format!("agent() subagent '{}' failed.", output.agent));
        handle.settle(status, None, None, Some(error));
    }
}

/// TS `cancelResolved`: kill a running agent (settling its handle
/// `cancelled` right away) or abort a running completion. Agent jobs match
/// TS `manager.cancel(job.id)` — no "Cancelled by eval handle" reason (that
/// text is completion-only).
fn cancel_handle(handle: &Handle) -> bool {
    if handle.is_settled() {
        return false;
    }
    if handle.kind == "completion" {
        handle.cancel.cancel();
        return true;
    }
    let cancelled = handle.run_ctl().is_some_and(|run| run.cancel());
    if cancelled {
        handle.settle(
            HandleStatus::Cancelled,
            None,
            None,
            Some(AGENT_CANCELLED_TEXT.to_string()),
        );
    }
    cancelled
}

// ---------------------------------------------------------------------------
// `__agent__` (TS `agent-bridge.ts` `runEvalAgent`)
// ---------------------------------------------------------------------------

fn run_agent(env: &Arc<Env>, args: &Value, ctx: &OpCtx) -> Result<Value, String> {
    // This Env's session token: the child and its deliveries belong to the
    // session whose `agent()` call spawned it.
    run_agent_with(env, args, ctx, &LAUNCHER, &env.a2a_owner())
}

pub(crate) fn run_agent_with(
    _env: &Arc<Env>,
    args: &Value,
    ctx: &OpCtx,
    launcher: &Arc<dyn Launcher>,
    owner: &super::a2a::Owner,
) -> Result<Value, String> {
    let prompt = args
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
        .ok_or_else(|| {
            "agent() received invalid arguments: prompt must be a non-empty string".to_string()
        })?;
    let agent = args
        .get("agent")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|agent| !agent.is_empty())
        .unwrap_or("task");
    let schema = args.get("schema").cloned();
    let schema_mode = match args.get("schemaMode").and_then(Value::as_str) {
        None | Some("permissive") => omp_task::SchemaMode::Permissive,
        Some("strict") => omp_task::SchemaMode::Strict,
        Some(other) => {
            return Err(format!(
                "agent() received invalid arguments: schemaMode must be \"permissive\" or \"strict\", got {other:?}"
            ));
        }
    };
    if args.get("isolated").is_some() || args.get("apply").is_some() || args.get("merge").is_some()
    {
        return Err(
            "agent() isolation controls (isolated/apply/merge) are not supported in this runtime"
                .to_string(),
        );
    }
    if args
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err("Eval-defined tools are unavailable in this runtime.".to_string());
    }
    // TS `assertDepthAndSpawnAllowed` / `canSpawnAtDepth`
    // (`task.maxRecursionDepth`, default 2): refuse unbounded recursion.
    omp_task::recursion_gate()
        .map_err(|error| format!("agent() cannot spawn a subagent: {error}"))?;

    let id = mint_id("agt");
    // Validate exactly like the `task` tool does (`TaskItem::validate`:
    // non-empty task, agent-name and handle-safety patterns), then reuse the
    // shared item→spec conversion.
    let item = omp_task::params::TaskItem {
        name: Some(id.clone()),
        agent: agent.to_string(),
        task: prompt.to_string(),
        solution_space: ".".to_string(),
        model: None,
        output_schema: schema,
        schema_mode,
        tools: None,
        worktree: None,
    };
    item.validate()
        .map_err(|error| format!("agent() received invalid arguments: {error}"))?;
    let schema_wanted = item.output_schema.is_some();
    let mut spec = omp_task::runner::TaskSpec::from_item(&item, None, ctx.cwd.clone());
    apply_child_policy(&mut spec);
    super::task::apply_agent_definition(&mut spec);
    // One channel for every spawner: eval `agent()` children get the same a2a
    // bus as `task` children, so inside the child `write agent://<sibling>` /
    // `agent://parent` / `agent://all` work regardless of how it was spawned.
    // The lease lives until the child settles.
    let lease = super::a2a::BusLease::acquire_for(owner)
        .map_err(|e| format!("agent() cannot start the agent message bus: {e}"))?;
    spec.env.push((
        super::a2a::SOCK_ENV.to_string(),
        lease.sock().to_string_lossy().into_owned(),
    ));
    spec.env
        .push((super::a2a::SELF_ENV.to_string(), id.clone()));

    let handle = Handle::new("agent", id.clone());
    AGENT_HANDLES.lock().insert(id.clone(), handle.clone());

    let run = launch_agent(&handle, spec, &**launcher);
    tokio::spawn(async move {
        let output = run.await;
        drop(lease);
        settle_agent(&handle, &output, schema_mode, schema_wanted);
        handle.mark_run_finished();
    });

    Ok(json!({"id": id, "agent": agent}))
}

// ---------------------------------------------------------------------------
// `__completion__` (TS `completion-bridge.ts` `runEvalCompletion`)
// ---------------------------------------------------------------------------

/// One stateless sub-LLM call for `llm_query` (RLM): resolves `model`, runs
/// under the shared eval request-slot ceiling and the cell's abort signal, and
/// returns the text with the provider-reported usage and the model that
/// actually answered (fallback chains may land on a later candidate).
pub(crate) async fn query_once(
    model: &str,
    prompt: &str,
    system: Option<&str>,
    cell_signal: &AbortSignal,
) -> Result<(String, pa_types::ai::Usage, Model), String> {
    let candidates = resolve_query_candidates(model).await?;
    let _permit = REQUEST_SLOTS
        .acquire()
        .await
        .map_err(|_| "llm_query() request aborted.".to_string())?;
    let cancel = CancellationToken::new();
    tokio::select! {
        () = cell_signal.aborted() => {
            cancel.cancel();
            Err("llm_query() request aborted.".to_string())
        }
        result = execute_completion(prompt, system, None, &candidates, &cancel) => {
            result.map(|(text, _, usage, model)| (text, usage, model))
        }
    }
}

fn agent_id_taken(id: &str) -> bool {
    AGENT_HANDLES.lock().contains_key(id)
}

/// A `task` child's entry in the shared agent registry (the one
/// [`deliver_message`] reads), so `write agent://<id>` steers it and
/// `agent://all` reaches it. The entry is removed when the slot drops.
pub(crate) struct TaskSlot {
    id: String,
    handle: Arc<Handle>,
}

impl TaskSlot {
    /// Reserve `id` in the registry; fails when another live or finished
    /// agent already owns it.
    pub(crate) fn claim(id: &str) -> Result<Self, String> {
        let mut registry = AGENT_HANDLES.lock();
        if registry.contains_key(id) {
            return Err(format!(
                "agent id `{id}` is already in use; pick a different `name`."
            ));
        }
        let handle = Handle::new("agent", id.to_string());
        registry.insert(id.to_string(), handle.clone());
        Ok(Self {
            id: id.to_string(),
            handle,
        })
    }

    /// Make the run steerable/cancellable through `write agent://<id>`.
    pub(crate) fn attach(&self, control: omp_task::RunControl) {
        self.attach_ctl(Arc::new(control));
    }

    fn attach_ctl(&self, run: Arc<dyn RunCtl>) {
        *self.handle.run.lock() = Some(run);
    }

    /// Mark the handle settled from the child's final output.
    pub(crate) fn settle(&self, output: &omp_task::RunOutput) {
        settle_agent(
            &self.handle,
            output,
            omp_task::SchemaMode::Permissive,
            false,
        );
        self.handle.mark_run_finished();
    }
}

impl Drop for TaskSlot {
    fn drop(&mut self) {
        self.handle.settle(
            HandleStatus::Cancelled,
            None,
            None,
            Some(AGENT_CANCELLED_TEXT.to_string()),
        );
        self.handle.mark_run_finished();
        let mut registry = AGENT_HANDLES.lock();
        if registry
            .get(&self.id)
            .is_some_and(|current| Arc::ptr_eq(current, &self.handle))
        {
            registry.remove(&self.id);
        }
    }
}

/// Parent spawn policy captured from this process's own argv
/// (EVAL-CHILD-POLICY-NOT-PROPAGATED): children re-launched from
/// `current_exe()` must inherit the parent's extension loading and tool
/// restriction rather than silently running the full default toolset.
pub(crate) struct ChildSpawnPolicy {
    /// Verbatim argv fragments appended to the child command
    /// (`--no-extensions`, `--extension/-e/--hook/--trusted-extension <p>`).
    pub extra_args: Vec<String>,
    /// The parent's `--tools` restriction; `None` = unrestricted. `__none__`
    /// is the omp convention for "no tools" (`main.rs` `--no-tools`).
    pub allowed_tools: Option<Vec<String>>,
}

pub(crate) fn child_spawn_policy() -> &'static ChildSpawnPolicy {
    static POLICY: LazyLock<ChildSpawnPolicy> = LazyLock::new(|| {
        let mut extra_args = Vec::new();
        let mut allowed_tools = None;
        let mut args = std::env::args_os().skip(1);
        while let Some(arg) = args.next() {
            let Some(flag) = arg.to_str() else { continue };
            match flag {
                "--no-extensions" => extra_args.push(flag.to_string()),
                "--extension" | "-e" | "--hook" | "--trusted-extension" => {
                    extra_args.push(flag.to_string());
                    if let Some(value) = args.next() {
                        extra_args.push(value.to_string_lossy().into_owned());
                    }
                }
                "--no-tools" => allowed_tools = Some(vec!["__none__".to_string()]),
                "--tools" => {
                    if let Some(value) = args.next() {
                        allowed_tools = Some(
                            value
                                .to_string_lossy()
                                .split(',')
                                .map(str::trim)
                                .filter(|name| !name.is_empty())
                                .map(str::to_string)
                                .collect(),
                        );
                    }
                }
                _ if flag.starts_with("--tools=") => {
                    allowed_tools = Some(
                        flag["--tools=".len()..]
                            .split(',')
                            .map(str::trim)
                            .filter(|name| !name.is_empty())
                            .map(str::to_string)
                            .collect(),
                    );
                }
                _ => {}
            }
        }
        ChildSpawnPolicy {
            extra_args,
            allowed_tools,
        }
    });
    &POLICY
}

/// Clamp a requested child tool allow-list to the parent's enabled set
/// (EVAL-CHILD-POLICY-NOT-PROPAGATED, TS "clamp child tools to a subset of
/// the parent's enabled tools"): an unrestricted child of a restricted
/// parent would bypass the restriction entirely.
pub(crate) fn clamp_child_tools(requested: Option<Vec<String>>) -> Option<Vec<String>> {
    clamp_tools(child_spawn_policy().allowed_tools.as_ref(), requested)
}

/// Pure clamp against an explicit parent restriction (testable without argv).
fn clamp_tools(
    allowed: Option<&Vec<String>>,
    requested: Option<Vec<String>>,
) -> Option<Vec<String>> {
    match allowed {
        None => requested,
        Some(allowed) => match requested {
            None => Some(allowed.clone()),
            Some(requested) => {
                let kept: Vec<String> = requested
                    .into_iter()
                    .filter(|tool| allowed.contains(tool))
                    .collect();
                Some(if kept.is_empty() {
                    vec!["__none__".to_string()]
                } else {
                    kept
                })
            }
        },
    }
}

/// Apply the parent policy to a spec about to be launched: child depth stamp
/// plus extension/no-extensions passthrough and tool clamping. Built-in
/// read-only scouts use the lightweight profile: their declared tool policy
/// is unchanged, but they do not pay for the parent's interactive extension
/// host (approval, TUI and session-recall hooks) on every cold start.
pub(crate) fn apply_child_policy(spec: &mut omp_task::TaskSpec) {
    spec.depth = omp_task::current_task_depth() + 1;
    spec.extra_args.clone_from(&child_spawn_policy().extra_args);
    if matches!(
        spec.agent.as_str(),
        "scout-glm" | "scout-kimi" | "scout-codex" | "scout-grok"
    ) && !spec.extra_args.iter().any(|arg| {
        matches!(
            arg.as_str(),
            "--no-extensions" | "--extension" | "-e" | "--hook" | "--trusted-extension"
        )
    }) {
        spec.extra_args.push("--no-extensions".to_string());
    }
    spec.tools = clamp_child_tools(spec.tools.take());
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod tests {
    use super::test_support::*;
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[tokio::test]
    async fn cancel_running_agent_settles_cancelled_once() {
        let launcher = MockLauncher::default();
        let (handle, run) = mock_agent(&launcher, "test-cancel-1");
        assert_eq!(handle.snapshot()["status"], "running");
        assert!(cancel_handle(&handle));
        assert!(run.cancelled.load(Ordering::SeqCst), "runner kill invoked");
        let snap = handle.snapshot();
        assert_eq!(snap["status"], "cancelled");
        assert_eq!(snap["error"], AGENT_CANCELLED_TEXT);
        // The killed run's own settle does not overwrite the cancel.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(handle.snapshot()["error"], AGENT_CANCELLED_TEXT);
        assert!(!cancel_handle(&handle), "second cancel is a no-op");
        let reply = run_cancel(&json!({"item": {"kind": "agent", "id": "test-cancel-1"}})).unwrap();
        assert_eq!(reply, json!({"cancelled": false}));
    }

    #[tokio::test]
    async fn cancel_after_completion_reports_false() {
        let launcher = MockLauncher::default();
        let (handle, run) = mock_agent(&launcher, "test-cancel-2");
        run.finish(omp_task::RunStatus::Completed, "answer");
        settled(&handle).await;
        assert_eq!(handle.snapshot()["text"], "answer");
        let reply = run_cancel(&json!({"item": {"kind": "agent", "id": "test-cancel-2"}})).unwrap();
        assert_eq!(reply, json!({"cancelled": false}));
        assert!(!run.cancelled.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn messages_steer_running_agents_only() {
        // No parent bus: the unknown-id leg must stay local.
        super::super::a2a::tests::without_parent_bus(async {
            inner_messages_steer_running_agents_only().await;
        })
        .await;
    }

    async fn inner_messages_steer_running_agents_only() {
        let launcher = MockLauncher::default();
        let (handle, run) = mock_agent(&launcher, "test-msg-1");
        assert_eq!(
            deliver_message("test-msg-1", "look at foo.rs")
                .await
                .unwrap(),
            "Delivered to test-msg-1."
        );
        assert_eq!(*run.steered.lock(), ["look at foo.rs"]);
        assert!(deliver_message("test-msg-1", "  ").await.is_err());
        assert!(deliver_message("test-msg-1/data", "x")
            .await
            .unwrap_err()
            .contains("JSON-path suffix"));
        let unknown = deliver_message("test-msg-nope", "x").await.unwrap_err();
        assert!(
            unknown.starts_with("Failed: test-msg-nope is not running."),
            "{unknown}"
        );
        run.finish(omp_task::RunStatus::Completed, "done");
        settled(&handle).await;
        let late = deliver_message("test-msg-1", "x").await.unwrap_err();
        assert!(late.contains("already finished"), "{late}");
    }

    #[test]
    fn scout_profile_skips_extensions_without_changing_tool_policy() {
        let mut scout = spec("light-scout");
        scout.agent = "scout-kimi".to_string();
        scout.tools = Some(vec!["read".to_string(), "grep".to_string()]);
        apply_child_policy(&mut scout);
        assert!(scout.extra_args.iter().any(|arg| arg == "--no-extensions"));
        assert_eq!(
            scout.tools,
            Some(vec!["read".to_string(), "grep".to_string()])
        );

        let mut regular = spec("full-worker");
        apply_child_policy(&mut regular);
        assert!(!regular
            .extra_args
            .iter()
            .any(|arg| arg == "--no-extensions"));
    }

    /// Finding 8: at `OMP_TASK_MAX_RECURSION_DEPTH` (TS `task.maxRecursionDepth`,
    /// default 2) agent/workpool/task spawns are refused with the TS message.
    #[test]
    fn recursion_gate_matches_ts_depth_rule() {
        let _env = DEPTH_ENV_LOCK.blocking_lock();
        std::env::set_var(omp_task::MAX_RECURSION_ENV, "2");
        assert!(omp_task::recursion_gate_at(0).is_ok());
        assert!(omp_task::recursion_gate_at(1).is_ok());
        let err = omp_task::recursion_gate_at(2).unwrap_err();
        assert_eq!(
            err,
            "Cannot spawn another agent at task depth 2; maximum depth is 2."
        );
        std::env::remove_var(omp_task::MAX_RECURSION_ENV);
    }

    /// Finding 9: child tool allow-lists clamp to the parent's enabled set.
    #[test]
    fn child_tools_clamp_to_the_parent_allow_list() {
        let parent = Some(vec!["read".to_string(), "bash".to_string()]);
        // Unrestricted request inherits the parent's restriction.
        assert_eq!(
            clamp_tools(parent.as_ref(), None),
            Some(vec!["read".to_string(), "bash".to_string()])
        );
        // Requests keep only tools the parent has.
        assert_eq!(
            clamp_tools(
                parent.as_ref(),
                Some(vec!["read".to_string(), "eval".to_string()])
            ),
            Some(vec!["read".to_string()])
        );
        // Nothing in common: the child gets no tools (omp `__none__`), never
        // the full default toolset.
        assert_eq!(
            clamp_tools(parent.as_ref(), Some(vec!["eval".to_string()])),
            Some(vec!["__none__".to_string()])
        );
        // Unrestricted parent: the request passes through untouched.
        assert_eq!(
            clamp_tools(None, Some(vec!["eval".to_string()])),
            Some(vec!["eval".to_string()])
        );
    }

    fn mock_run(id: &str) -> Arc<MockRun> {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        Arc::new(MockRun {
            id: id.to_string(),
            steered: Mutex::new(Vec::new()),
            cancelled: AtomicBool::new(false),
            done: Mutex::new(Some(tx)),
        })
    }

    /// `task` children share the registry `deliver_message` reads: a parent
    /// `write agent://<id>` steers the live run, ids are exclusive, and the
    /// entry disappears once the slot drops.
    #[tokio::test]
    async fn task_slot_is_steerable_and_unregistered_on_drop() {
        let run = mock_run("slot-t1");
        let slot = TaskSlot::claim("slot-t1").unwrap();
        slot.attach_ctl(run.clone());
        assert!(TaskSlot::claim("slot-t1")
            .err()
            .is_some_and(|e| e.contains("already in use")));

        let receipt = deliver_message("slot-t1", "change course").await.unwrap();
        assert_eq!(receipt, "Delivered to slot-t1.");
        assert_eq!(*run.steered.lock(), ["change course"]);

        drop(slot);
        assert!(!agent_id_taken("slot-t1"));
        let gone = deliver_from_peer("slot-t1", "late", "someone")
            .await
            .unwrap_err();
        assert!(gone.contains("Unknown agent"), "{gone}");
        assert!(
            TaskSlot::claim("slot-t1").is_ok(),
            "id is reusable after the drop"
        );
    }

    /// A child's bus messages reach siblings as steers (sender-prefixed) and
    /// the parent as inbox entries; settled/unknown/self targets fail.
    #[tokio::test]
    async fn bus_routes_child_messages_to_siblings_and_parent() {
        // No inherited parent bus: the unknown-id leg must stay local.
        super::super::a2a::tests::without_parent_bus(async {
            inner_bus_routes_child_messages_to_siblings_and_parent().await;
        })
        .await;
    }

    async fn inner_bus_routes_child_messages_to_siblings_and_parent() {
        let _serial = super::super::a2a::tests::LOCK.lock().await;
        let (run_a, run_b) = (mock_run("bus-a"), mock_run("bus-b"));
        let (slot_a, slot_b) = (
            TaskSlot::claim("bus-a").unwrap(),
            TaskSlot::claim("bus-b").unwrap(),
        );
        slot_a.attach_ctl(run_a.clone());
        slot_b.attach_ctl(run_b.clone());
        // A private owner: this bus's deliveries cannot be consumed by a
        // concurrent test runtime's park/wake legs.
        let owner: super::super::a2a::Owner = Some(crate::persist::uuid_v7().into());
        let lease = super::super::a2a::BusLease::acquire_for(&owner).unwrap();
        let from_a = super::super::a2a::Client::for_test(lease.sock(), "bus-a");

        let receipt = from_a.send("bus-b", "need the schema").await.unwrap();
        assert_eq!(receipt, "Delivered to bus-b.");
        assert_eq!(
            *run_b.steered.lock(),
            ["[message from agent://bus-a]\nneed the schema"]
        );
        assert!(run_a.steered.lock().is_empty(), "sender not steered");

        from_a.send("parent", "blocked on bus-b").await.unwrap();
        let inbox: Vec<_> = super::super::a2a::take_deliveries()
            .into_iter()
            .filter(|m| m.from == "bus-a")
            .collect();
        assert_eq!(inbox.len(), 1);
        assert_eq!(inbox[0].text, "blocked on bus-b");

        assert!(from_a
            .send("bus-a", "me")
            .await
            .unwrap_err()
            .contains("itself"));
        assert!(from_a
            .send("bus-nobody", "hi")
            .await
            .unwrap_err()
            .contains("Unknown agent"));

        // Once the sibling settled, delivery fails instead of vanishing.
        slot_b.settle(&run_output("bus-b", omp_task::RunStatus::Completed, "done"));
        let late = from_a.send("bus-b", "too late").await.unwrap_err();
        assert!(late.contains("already finished"), "{late}");
        assert_eq!(run_b.steered.lock().len(), 1);
    }
}
