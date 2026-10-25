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

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, RwLock};
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
    *slot.write().expect("eval runtime hooks lock") = Some(EvalRuntimeHooks { target });
}

fn hooks() -> Result<std::sync::RwLockReadGuard<'static, Option<EvalRuntimeHooks>>, String> {
    HOOKS
        .get()
        .ok_or_else(|| "eval runtime is not initialized (set_runtime_hooks not called)".to_string())
        .map(|slot| slot.read().expect("eval runtime hooks lock"))
}

/// The live provider target (TS `session.getActiveModel`-backed `default` tier).
fn provider_target() -> Result<crate::adapter::ProviderTarget, String> {
    let guard = hooks()?;
    let hooks = guard.as_ref().ok_or_else(|| {
        "eval runtime is not initialized (set_runtime_hooks not called)".to_string()
    })?;
    let target = hooks.target.read().expect("provider target lock").clone();
    target.ok_or_else(|| "completion() has no provider target set.".to_string())
}

// ---------------------------------------------------------------------------
// Handle registry (agents + completions)
// ---------------------------------------------------------------------------

/// Terminal or current state of one handle (TS `EvalHandleState`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum HandleStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl Default for HandleStatus {
    fn default() -> Self {
        Self::Running
    }
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
        self.inner.lock().expect("eval handle lock").settled
    }

    fn run_ctl(&self) -> Option<Arc<dyn RunCtl>> {
        self.run.lock().expect("eval handle lock").clone()
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
            let mut inner = self.inner.lock().expect("eval handle lock");
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
        let inner = self.inner.lock().expect("eval handle lock");
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
            return Err("handle must contain kind agent|completion and a non-empty id".to_string())
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
    registry
        .lock()
        .expect("eval handles lock")
        .get(id)
        .cloned()
        .ok_or_else(|| {
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
        "__agent__" => Some(run_agent(env, &args, ctx).await),
        "__completion__" => Some(run_completion(&args, ctx).await),
        "__judge__" => Some(run_judgment_op(&args, ctx).await),
        "__judge_batch__" => Some(run_judge_batch_op(&args, ctx).await),
        "__workpool__" => Some(run_workpool(&args, ctx)),
        "__budget__" => Some(Ok(budget_snapshot())),
        "__llm_query__" => Some(super::rlm::llm_query_op(&args, ctx).await),
        "__rlm_stats__" => Some(Ok(super::rlm::stats_snapshot())),
        "__rlm_context__" => Some(super::rlm::context_op()),
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
    if to.is_empty() {
        return Err("agent:// URL requires a recipient: agent://<id>".to_string());
    }
    if to.contains('/') {
        return Err("agent:// message target cannot have a JSON-path suffix.".to_string());
    }
    if content.trim().is_empty() {
        return Err("agent:// messages require non-empty content.".to_string());
    }
    if to == "all" {
        return broadcast_message(content).await;
    }
    let handle = AGENT_HANDLES
        .lock()
        .expect("eval handles lock")
        .get(to)
        .cloned();
    match steer_handle(handle.as_deref(), to, content).await {
        Ok(()) => Ok(format!("Delivered to {to}.")),
        Err(error) => Err(format!("Failed: {to} is not running. {error}")),
    }
}

/// TS broadcast receipt text over every running eval subagent.
async fn broadcast_message(content: &str) -> Result<String, String> {
    let mut targets: Vec<(String, Arc<Handle>)> = AGENT_HANDLES
        .lock()
        .expect("eval handles lock")
        .iter()
        .filter(|(_, handle)| !handle.is_settled())
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
                text.push_str(&format!("\n- {id}: injected"));
            }
            Err(error) => text.push_str(&format!("\n- {id}: failed — {error}")),
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
    let Some(run) = handle.run_ctl() else {
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
trait RunCtl: Send + Sync {
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
trait Launcher: Send + Sync {
    fn launch(&self, spec: omp_task::TaskSpec) -> (Arc<dyn RunCtl>, RunFuture);
}

/// `omp-rs --mode rpc` children under one shared concurrency ceiling.
struct SubprocessLauncher {
    runner: omp_task::TaskRunner,
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
    let binary = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("omt"));
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
    *handle.run.lock().expect("eval handle lock") = Some(run.clone());
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

async fn run_agent(_env: &Arc<Env>, args: &Value, ctx: &OpCtx) -> Result<Value, String> {
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
    };
    item.validate()
        .map_err(|error| format!("agent() received invalid arguments: {error}"))?;
    let schema_wanted = item.output_schema.is_some();
    let mut spec = omp_task::runner::TaskSpec::from_item(&item, None, ctx.cwd.clone());
    apply_child_policy(&mut spec);
    super::task::apply_agent_model(&mut spec);

    let handle = Handle::new("agent", id.clone());
    AGENT_HANDLES
        .lock()
        .expect("eval handles lock")
        .insert(id.clone(), handle.clone());

    let run = launch_agent(&handle, spec, &**LAUNCHER);
    tokio::spawn(async move {
        let output = run.await;
        settle_agent(&handle, &output, schema_mode, schema_wanted);
        handle.mark_run_finished();
    });

    Ok(json!({"id": id, "agent": agent}))
}

// ---------------------------------------------------------------------------
// `__completion__` (TS `completion-bridge.ts` `runEvalCompletion`)
// ---------------------------------------------------------------------------

/// One candidate in the tier's fallback chain (TS `CompletionCandidate`).
struct CompletionCandidate {
    model: Model,
    api_key: Option<String>,
    service_tier: Option<pa_types::ai::ServiceTier>,
    headers: Option<Vec<(String, String)>>,
    /// Effective reasoning from the resolved selector, if any.
    thinking: Option<pa_types::ai::ModelThinkingLevel>,
}

fn candidate_from_resolved(resolved: ResolvedModel) -> CompletionCandidate {
    CompletionCandidate {
        model: resolved.model,
        api_key: resolved.api_key,
        service_tier: None,
        headers: Some(resolved.headers),
        thinking: resolved.thinking,
    }
}

/// Walk a role's fallback chain to more candidates; the walk does blocking
/// config I/O (`resolve_in`), so it runs off the executor threads.
async fn walk_fallbacks(dir: AgentDir, primary: ResolvedModel) -> Vec<CompletionCandidate> {
    tokio::task::spawn_blocking(move || {
        let mut walker = omp_providers::fallback_walker(&dir, &primary);
        let mut out = Vec::new();
        while let Some(resolved) = walker.next_model() {
            out.push(candidate_from_resolved(resolved));
        }
        out
    })
    .await
    .unwrap_or_default()
}

/// TS `resolveTierCandidates`, omp-rs form: `default` answers from the live
/// provider target (then the `@default` role's fallback chain); `smol`/`slow`
/// resolve the role selector and walk its configured fallbacks.
async fn resolve_completion_candidates(tier: &str) -> Result<Vec<CompletionCandidate>, String> {
    let dir = AgentDir::default();
    let mut candidates = Vec::new();
    match tier {
        "default" => {
            // Primary: the session's live target; fallbacks: the `@default`
            // role's configured chain, walked when the primary fails.
            let target = provider_target()?;
            let (api_key, _) = crate::adapter::fresh_stored_key(
                &target.model.provider,
                target.credential_id,
                target.api_key,
            )
            .await;
            candidates.push(CompletionCandidate {
                model: target.model,
                api_key,
                service_tier: target.service_tier,
                headers: target.headers.map(|headers| headers.into_iter().collect()),
                thinking: None,
            });
            if let Ok(primary) =
                omp_providers::resolve_in_async(dir.clone(), "default".to_string()).await
            {
                candidates.extend(walk_fallbacks(dir, primary).await);
            }
        }
        "smol" | "slow" => {
            // `resolve_in` takes a bare role name (TS `@smol` alias form is
            // not a selector it parses).
            let primary = omp_providers::resolve_in_async(dir.clone(), tier.to_string())
                .await
                .map_err(|e| {
                    format!(
                        "completion() could not resolve a model for the \"{tier}\" tier. Configure modelRoles.{tier} or ensure a provider is available. ({e})"
                    )
                })?;
            candidates.push(candidate_from_resolved(primary.clone()));
            candidates.extend(walk_fallbacks(dir, primary).await);
        }
        other => {
            return Err(format!(
                "completion() received invalid arguments: model must be \"smol\", \"default\", or \"slow\", got {other:?}"
            ));
        }
    }
    if candidates.is_empty() {
        return Err(format!(
            "completion() could not resolve a model for the \"{tier}\" tier. Configure modelRoles.{tier} or ensure a provider is available."
        ));
    }
    Ok(candidates)
}

/// TS `executeCompletion`: one stateless, toolless completion over the
/// candidate chain (each fallback that issues a request consumes one retry
/// attempt; keyless candidates are skipped without consuming budget).
async fn execute_completion(
    prompt: &str,
    system: Option<&str>,
    schema: Option<&Value>,
    candidates: &[CompletionCandidate],
    cancel: &CancellationToken,
) -> Result<(String, Option<Value>, pa_types::ai::Usage, Model), String> {
    let tools = schema.map(|schema| {
        vec![Tool {
            name: STRUCTURED_TOOL_NAME.to_string(),
            description:
                "Return your answer by calling this tool with the requested structured fields."
                    .to_string(),
            parameters: schema.clone(),
        }]
    });
    let system_prompt = system.unwrap_or("You are a helpful assistant.").to_string();
    let message = Message::User(UserMessage {
        content: UserContent::Text(prompt.to_string()),
        timestamp: pa_agent::now_ms().cast_unsigned(),
        rest: Map::default(),
    });

    let mut last_error: Option<String> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err("completion() request aborted.".to_string());
        }
        if candidate.api_key.is_none() {
            last_error = Some(format!(
                "completion() has no API key for {}/{}. Configure credentials for this provider or choose another tier.",
                candidate.model.provider, candidate.model.id
            ));
            continue;
        }
        let context = Context {
            system_prompt: Some(system_prompt.clone()),
            messages: vec![message.clone()],
            tools: tools.clone(),
        };
        let options = SimpleStreamOptions {
            base: pa_ai::types::StreamOptions {
                api_key: candidate.api_key.clone(),
                service_tier: candidate.service_tier,
                headers: candidate
                    .headers
                    .clone()
                    .map(|headers| headers.into_iter().collect()),
                signal: Some(cancel.clone()),
                ..pa_ai::types::StreamOptions::default()
            },
            reasoning: candidate.thinking,
            thinking_budgets: None,
            // Utility one-shot: mirror the title/judgment path (no thinking)
            // for tiers without an explicit level; `slow` keeps the resolved
            // selector's level.
            disable_reasoning: candidate.thinking.is_none(),
            tool_choice: schema.map(|_| SimpleToolChoice::Function {
                name: STRUCTURED_TOOL_NAME.to_string(),
            }),
        };
        let response = match pa_ai::complete_simple(&candidate.model, &context, Some(options)).await
        {
            Ok(response) => response,
            Err(error) => {
                last_error = Some(format!("{error:?}"));
                continue;
            }
        };
        if response.stop_reason == StopReason::Aborted {
            return Err("completion() request aborted.".to_string());
        }
        if response.stop_reason == StopReason::Error {
            let message = response
                .error_message
                .clone()
                .unwrap_or_else(|| "completion() request failed.".to_string());
            last_error = Some(message.clone());
            if index + 1 >= candidates.len() {
                return Err(message);
            }
            continue;
        }
        if schema.is_some() {
            let call = response.content.iter().find_map(|block| match block {
                AssistantContentBlock::ToolCall(call) if call.name == STRUCTURED_TOOL_NAME => {
                    Some(call)
                }
                _ => None,
            });
            let value = match call {
                Some(call) => Value::Object(call.arguments.clone()),
                None => {
                    let text = assistant_text(&response.content);
                    if text.is_empty() {
                        return Err("completion() returned no structured response.".to_string());
                    }
                    serde_json::from_str::<Value>(&text).map_err(|_| {
                        "completion() did not return a structured response matching the schema."
                            .to_string()
                    })?
                }
            };
            return Ok((
                serde_json::to_string(&value).unwrap_or_default(),
                Some(value),
                response.usage,
                candidate.model.clone(),
            ));
        }
        let text = assistant_text(&response.content);
        if text.is_empty() {
            return Err("completion() returned no text output.".to_string());
        }
        return Ok((text, None, response.usage, candidate.model.clone()));
    }
    Err(last_error.unwrap_or_else(|| "completion() request failed.".to_string()))
}

/// Resolve the candidate chain for an `llm_query` model: a tier name
/// (`smol`/`default`/`slow`) or any model selector `resolve_in` understands
/// (`provider/id[:level]`, other role names).
async fn resolve_query_candidates(model: &str) -> Result<Vec<CompletionCandidate>, String> {
    if matches!(model, "smol" | "default" | "slow") {
        return resolve_completion_candidates(model).await;
    }
    let dir = AgentDir::default();
    let primary = omp_providers::resolve_in_async(dir.clone(), model.to_string())
        .await
        .map_err(|e| format!("llm_query() could not resolve model {model:?}: {e}"))?;
    let mut candidates = vec![candidate_from_resolved(primary.clone())];
    candidates.extend(walk_fallbacks(dir, primary).await);
    Ok(candidates)
}

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

fn assistant_text(content: &[AssistantContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            AssistantContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

async fn run_completion(args: &Value, ctx: &OpCtx) -> Result<Value, String> {
    let prompt = args
        .get("prompt")
        .and_then(Value::as_str)
        .filter(|prompt| !prompt.is_empty())
        .ok_or_else(|| {
            "completion() received invalid arguments: prompt must be a non-empty string".to_string()
        })?;
    let tier = args
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("default");
    let system = args.get("system").and_then(Value::as_str);
    let schema = args.get("schema").cloned();

    let candidates = resolve_completion_candidates(tier).await?;

    let id = mint_id("cmp");
    let handle = Handle::new("completion", id.clone());
    COMPLETION_HANDLES
        .lock()
        .expect("eval handles lock")
        .insert(id.clone(), handle.clone());

    let cell_signal = ctx.signal.clone();
    let cancel = handle.cancel.clone();
    let (prompt, system) = (prompt.to_string(), system.map(str::to_string));
    tokio::spawn(async move {
        match completion_job(
            prompt,
            system,
            schema,
            candidates,
            cancel.clone(),
            cell_signal,
        )
        .await
        {
            Ok((text, data, _usage)) => {
                handle.settle(HandleStatus::Completed, Some(text), data, None)
            }
            Err(error) => {
                let status = if cancel.is_cancelled() {
                    HandleStatus::Cancelled
                } else {
                    HandleStatus::Failed
                };
                handle.settle(status, None, None, Some(error));
            }
        }
        handle.mark_run_finished();
    });

    Ok(json!({"id": id}))
}

/// One retained completion run under the shared request-slot ceiling (TS
/// `retainCompletionHandle` + `evalRequestSlots`). The handle's cancellation
/// token reaches the provider stream mid-flight (`StreamOptions.signal`) and
/// stops the candidate walk; the cell's abort signal cancels the handle (TS
/// `AbortSignal.any([options.signal, controller.signal])`).
async fn completion_job(
    prompt: String,
    system: Option<String>,
    schema: Option<Value>,
    candidates: Vec<CompletionCandidate>,
    cancel: CancellationToken,
    cell_signal: AbortSignal,
) -> Result<(String, Option<Value>, pa_types::ai::Usage), String> {
    let _permit = REQUEST_SLOTS
        .acquire()
        .await
        .map_err(|_| "completion() request aborted.".to_string())?;
    if cancel.is_cancelled() {
        return Err("completion() request aborted.".to_string());
    }
    tokio::select! {
        _ = cell_signal.aborted() => {
            cancel.cancel();
            Err("completion() request aborted.".to_string())
        }
        result = execute_completion(&prompt, system.as_deref(), schema.as_ref(), &candidates, &cancel) => {
            result.map(|(text, data, usage, _model)| (text, data, usage))
        },
    }
}

// ---------------------------------------------------------------------------
// `__judge__` (TS `judgment-bridge.ts` `runEvalJudgment`)
// ---------------------------------------------------------------------------

/// TS `parseState`: a non-empty string or any JSON value (object/array).
fn parse_state(value: &Value) -> Result<Value, String> {
    match value {
        Value::String(text) if text.is_empty() => {
            Err("judge() received invalid arguments: state must not be empty".to_string())
        }
        Value::String(_) => Ok(value.clone()),
        Value::Array(_) | Value::Object(_) => Ok(value.clone()),
        _ => Err("judge() received invalid arguments: state must be a string, a JSON object, or a JSON array".to_string()),
    }
}

fn parse_instructions(id: &str, value: Option<&Value>) -> Result<String, String> {
    match value.and_then(Value::as_str) {
        Some(text) if !text.is_empty() => Ok(text.to_string()),
        _ => Err(format!(
            "judge() received invalid arguments: question \"{id}\" needs non-empty string instructions"
        )),
    }
}

/// TS `parseQuestions`: keyed questions of type `choice` | `bool` | `score`;
/// the cell-facing `bool` maps to the library's `noul` wire type.
fn parse_questions(value: Option<&Value>) -> Result<Map<String, Value>, String> {
    let Some(questions) = value.and_then(Value::as_object) else {
        return Err(
            "judge() received invalid arguments: questions must be an object keyed by question id"
                .to_string(),
        );
    };
    if questions.is_empty() {
        return Err(
            "judge() received invalid arguments: questions must contain at least one question"
                .to_string(),
        );
    }
    let mut parsed = Map::new();
    for (id, question) in questions {
        let Some(question) = question.as_object() else {
            return Err(format!(
                "judge() received invalid arguments: question \"{id}\" must be an object"
            ));
        };
        let instructions = parse_instructions(id, question.get("instructions"))?;
        let invalid = |detail: &str| format!("judge() received invalid arguments: {detail}");
        let wire = match question.get("type").and_then(Value::as_str) {
            Some("choice") => {
                let Some(criteria) = question.get("criteria").and_then(Value::as_object) else {
                    return Err(invalid(&format!(
                        "choice question \"{id}\" needs criteria: {{ label: rubric | null }}"
                    )));
                };
                if criteria.len() < 2 {
                    return Err(invalid(&format!(
                        "choice question \"{id}\" needs at least two options"
                    )));
                }
                for (label, rubric) in criteria {
                    if !rubric.is_null() && !rubric.is_string() {
                        return Err(invalid(&format!(
                            "choice question \"{id}\" criteria \"{label}\" must be a string or null"
                        )));
                    }
                }
                json!({"type": "choice", "instructions": instructions, "criteria": question.get("criteria").cloned().unwrap_or(Value::Null)})
            }
            Some("bool") => {
                // TS `parseBool`: no `criteria` key when absent (the System One
                // endpoint rejects `criteria: null`), else only the string sides.
                let mut wire = json!({"type": "noul", "instructions": instructions});
                if let Some(criteria) = question.get("criteria") {
                    let Some(criteria) = criteria.as_object() else {
                        return Err(invalid(&format!(
                            "bool question \"{id}\" criteria must be {{ true?: string, false?: string }}"
                        )));
                    };
                    let mut sides = Map::new();
                    for side in ["true", "false"] {
                        if let Some(description) = criteria.get(side) {
                            if !description.is_string() {
                                return Err(invalid(&format!(
                                    "bool question \"{id}\" criteria.{side} must be a string"
                                )));
                            }
                            sides.insert(side.to_string(), description.clone());
                        }
                    }
                    wire["criteria"] = Value::Object(sides);
                }
                wire
            }
            Some("score") => {
                let Some(levels) = question.get("criteria").and_then(Value::as_array) else {
                    return Err(invalid(&format!(
                        "score question \"{id}\" needs criteria: [lowest, ..., highest] level descriptions"
                    )));
                };
                if levels.len() < 2 || !levels.iter().all(Value::is_string) {
                    return Err(invalid(&format!(
                        "score question \"{id}\" needs at least two levels"
                    )));
                }
                json!({"type": "score", "instructions": instructions, "criteria": question.get("criteria").cloned().unwrap_or(Value::Null)})
            }
            _ => {
                return Err(invalid(&format!(
                    "question \"{id}\" type must be \"choice\", \"bool\", or \"score\""
                )));
            }
        };
        parsed.insert(id.clone(), wire);
    }
    Ok(parsed)
}

/// TS `toEvalJudgmentResult`: the cell sees `bool` where the library answers
/// `noul`; plus the `provider/model` backend label.
fn judgment_result_value(result: &omp_providers::JudgeResult) -> Value {
    let mut answers = Map::new();
    for (id, answer) in &result.answers {
        let mut answer = answer.clone();
        if answer.get("type").and_then(Value::as_str) == Some("noul") {
            let mut as_bool = Map::new();
            as_bool.insert("type".to_string(), Value::String("bool".to_string()));
            as_bool.insert(
                "bool".to_string(),
                answer.get("noul").cloned().unwrap_or(Value::Null),
            );
            answer = Value::Object(as_bool);
        }
        answers.insert(id.clone(), answer);
    }
    json!({"answers": Value::Object(answers), "model": format!("{}/{}", result.provider, result.model)})
}

fn judgment_cache() -> Option<JudgmentCache> {
    let dir = AgentDir::default();
    JudgmentCache::open(&omp_providers::judgment_cache_path(&dir)).ok()
}

async fn run_judgment_op(args: &Value, ctx: &OpCtx) -> Result<Value, String> {
    if !args.is_object() {
        return Err(
            "judge() received invalid arguments: expected { state, questions }".to_string(),
        );
    }
    let state = parse_state(args.get("state").unwrap_or(&Value::Null))?;
    let questions = parse_questions(args.get("questions"))?;
    let dir = AgentDir::default();
    let chain = omp_providers::resolve_judge_chain(&dir)
        .map_err(|error| format!("judge() has no model registry: {error}"))?;
    throw_if_aborted(&ctx.signal)?;
    let _permit = REQUEST_SLOTS
        .acquire()
        .await
        .map_err(|_| "judge() request aborted.".to_string())?;
    let result = omp_providers::judge(
        &chain,
        &JudgeRequest { state, questions },
        judgment_cache().as_ref(),
    )
    .await
    .map_err(|error| error.to_string())?;
    throw_if_aborted(&ctx.signal)?;
    Ok(judgment_result_value(&result))
}

fn throw_if_aborted(signal: &AbortSignal) -> Result<(), String> {
    if signal.is_aborted() {
        return Err("aborted".to_string());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `__judge_batch__` (TS `judgment-batch-bridge.ts` `runEvalJudgmentBatch`)
// ---------------------------------------------------------------------------

/// TS `DEFAULT_RETRIES` (`judgment-batch-bridge.ts`).
const DEFAULT_BATCH_RETRIES: u32 = 1;
/// TS `DEFAULT_INTENT`.
const DEFAULT_BATCH_INTENT: &str = "Judging";
/// TS `DRAIN_SETTLE_WINDOW_MS`: once a waiting drain sees its first item it
/// holds this long for more so near-simultaneous settles return together.
const DRAIN_SETTLE_WINDOW: Duration = Duration::from_millis(50);

struct BatchInput {
    key: Value,
    state: Value,
}

/// One settled item as the cell receives it from `drain` (TS
/// `JudgmentBatchItem`).
#[derive(Clone)]
struct BatchItem {
    key: Value,
    answers: Option<Map<String, Value>>,
    error: Option<String>,
    model: Option<String>,
}

#[derive(Default)]
struct BatchInner {
    settled: Vec<BatchItem>,
    /// Number of settled items handed out by `drain` so far (TS drain cursor).
    cursor: usize,
    failed: usize,
    cost: f64,
    last_model: Option<String>,
    running: bool,
    error: Option<String>,
}

struct JudgmentBatchRun {
    id: String,
    intent: String,
    total: usize,
    min_ok: usize,
    inner: Mutex<BatchInner>,
    notify: Notify,
    cancel: CancellationToken,
    started: Instant,
}

impl JudgmentBatchRun {
    /// TS `JudgmentBatchStatus`.
    fn status(&self) -> Value {
        let inner = self.inner.lock().expect("judge batch lock");
        let mut status = json!({
            "id": self.id,
            "intent": self.intent,
            "total": self.total,
            "done": inner.settled.len(),
            "failed": inner.failed,
            "cost": inner.cost,
            "running": inner.running,
            "elapsedS": self.started.elapsed().as_secs(),
        });
        if let Some(model) = &inner.last_model {
            status["model"] = Value::String(model.clone());
        }
        if let Some(error) = &inner.error {
            status["error"] = Value::String(error.clone());
        }
        status
    }

    /// TS `JudgmentBatch.drain`: items settled since the previous drain; waits
    /// up to `timeout` for at least one, then holds a short settle window.
    async fn drain(&self, timeout: Option<Duration>) -> Vec<BatchItem> {
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        loop {
            {
                let inner = self.inner.lock().expect("judge batch lock");
                if inner.cursor < inner.settled.len() || !inner.running {
                    break;
                }
            }
            let wait = match deadline {
                Some(deadline) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return Vec::new();
                    }
                    remaining
                }
                None => Duration::from_secs(3600),
            };
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            // Re-check after arming so a settle that raced the arm is seen.
            {
                let inner = self.inner.lock().expect("judge batch lock");
                if inner.cursor < inner.settled.len() || !inner.running {
                    break;
                }
            }
            let _ = tokio::time::timeout(wait, notified).await;
        }
        // First item is in: hold the settle window once for near-simultaneous
        // settles, then hand out everything past the cursor.
        let have = {
            let inner = self.inner.lock().expect("judge batch lock");
            inner.cursor < inner.settled.len()
        };
        if have {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let _ = tokio::time::timeout(DRAIN_SETTLE_WINDOW, notified).await;
        }
        let mut inner = self.inner.lock().expect("judge batch lock");
        let items: Vec<BatchItem> = inner.settled[inner.cursor..].to_vec();
        inner.cursor = inner.settled.len();
        items
    }

    /// Raise the wholesale run error once the cursor is exhausted (TS
    /// `drain()` rejects with `#error` after delivering every item).
    fn drain_error(&self) -> Option<String> {
        let inner = self.inner.lock().expect("judge batch lock");
        if inner.cursor >= inner.settled.len() && !inner.running {
            inner.error.clone()
        } else {
            None
        }
    }
}

fn batch_invalid(detail: &str) -> String {
    format!("judge_batch() received invalid arguments: {detail}")
}

fn parse_batch_items(value: Option<&Value>) -> Result<Vec<BatchInput>, String> {
    let Some(items) = value.and_then(Value::as_array) else {
        return Err(batch_invalid("items must be an array of { key, state }"));
    };
    if items.is_empty() {
        return Err(batch_invalid("items must not be empty"));
    }
    let mut seen: Vec<String> = Vec::new();
    let mut parsed = Vec::new();
    for (index, entry) in items.iter().enumerate() {
        let Some(entry) = entry.as_object() else {
            return Err(batch_invalid(&format!(
                "item {index} needs a string or number key"
            )));
        };
        let key = entry.get("key").cloned().unwrap_or(Value::Null);
        let valid_key = matches!(&key, Value::String(s) if !s.is_empty()) || key.is_number();
        if !valid_key {
            return Err(batch_invalid(&format!(
                "item {index} needs a string or number key"
            )));
        }
        let key_id = serde_json::to_string(&key).unwrap_or_default();
        if seen.contains(&key_id) {
            return Err(batch_invalid(&format!("duplicate item key {key_id}")));
        }
        seen.push(key_id);
        parsed.push(BatchInput {
            key,
            // TS reuses `judgment-bridge.ts` `parseState`/`parseQuestions`
            // verbatim, so their `judge()`-prefixed messages surface as-is.
            state: parse_state(entry.get("state").unwrap_or(&Value::Null))?,
        });
    }
    Ok(parsed)
}

fn parse_batch_count(value: Option<&Value>, name: &str, fallback: u64) -> Result<u64, String> {
    match value {
        None => Ok(fallback),
        Some(Value::Number(number)) => {
            let count = number
                .as_u64()
                .ok_or_else(|| batch_invalid(&format!("{name} must be a non-negative integer")))?;
            Ok(count.min(if name == "concurrency" {
                EVAL_HANDLE_CONCURRENCY as u64
            } else {
                u64::MAX
            }))
        }
        Some(_) => Err(batch_invalid(&format!(
            "{name} must be a non-negative integer"
        ))),
    }
}

async fn run_judge_batch_op(args: &Value, ctx: &OpCtx) -> Result<Value, String> {
    let Some(record) = args.as_object() else {
        return Err(batch_invalid("expected { op, … }"));
    };
    let op = record
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| batch_invalid("expected { op, … }"))?;

    match op {
        "create" => {
            let items = parse_batch_items(record.get("items"))?;
            let questions = parse_questions(record.get("questions"))?;
            let total = items.len();
            // TS `parseBatchOptions`: an explicit 0 falls back to the default
            // ceiling (`concurrency > 0 ? concurrency : EVAL_HANDLE_CONCURRENCY`).
            let concurrency = parse_batch_count(
                record.get("concurrency"),
                "concurrency",
                EVAL_HANDLE_CONCURRENCY as u64,
            )?;
            let concurrency = if concurrency == 0 {
                EVAL_HANDLE_CONCURRENCY
            } else {
                concurrency as usize
            };
            let retries = parse_batch_count(
                record.get("retries"),
                "retries",
                u64::from(DEFAULT_BATCH_RETRIES),
            )? as u32;
            let min_ok = record
                .get("minOk")
                .map(|_| parse_batch_count(record.get("minOk"), "minOk", 1))
                .transpose()?
                .map_or(1, |min_ok| min_ok as usize)
                .min(total);
            let intent = match record.get("intent") {
                None => DEFAULT_BATCH_INTENT.to_string(),
                Some(Value::String(intent)) if !intent.trim().is_empty() => intent.clone(),
                Some(_) => return Err(batch_invalid("intent must be a non-empty string")),
            };
            let batch = Arc::new(JudgmentBatchRun {
                id: mint_id("jb"),
                intent,
                total,
                min_ok,
                inner: Mutex::new(BatchInner {
                    running: true,
                    ..BatchInner::default()
                }),
                notify: Notify::new(),
                cancel: CancellationToken::new(),
                started: Instant::now(),
            });
            JUDGE_BATCHES
                .lock()
                .expect("judge batches lock")
                .insert(batch.id.clone(), batch.clone());
            start_batch(
                batch.clone(),
                items,
                questions,
                concurrency,
                retries,
                ctx.signal.clone(),
            );
            // `create` answers with the status snapshot (TS returns
            // `batch.status()`; the prelude only requires id/total/intent).
            Ok(batch.status())
        }
        "attach" | "status" => {
            let batch = require_batch(record)?;
            Ok(batch.status())
        }
        "drain" => {
            let batch = require_batch(record)?;
            let timeout_ms = record.get("timeoutMs");
            if let Some(value) = timeout_ms {
                let Some(ms) = value.as_f64() else {
                    return Err(batch_invalid(
                        "timeoutMs must be a non-negative finite number",
                    ));
                };
                if !ms.is_finite() || ms < 0.0 {
                    return Err(batch_invalid(
                        "timeoutMs must be a non-negative finite number",
                    ));
                }
            }
            let timeout = timeout_ms
                .and_then(Value::as_f64)
                .map(|ms| Duration::from_secs_f64(ms / 1000.0));
            let items = batch.drain(timeout).await;
            if let Some(error) = batch.drain_error() {
                return Err(error);
            }
            Ok(json!({
                "items": items.iter().map(|item| {
                    let mut value = json!({"key": item.key});
                    if let Some(answers) = &item.answers {
                        value["answers"] = Value::Object(answers.clone());
                    }
                    if let Some(error) = &item.error {
                        value["error"] = Value::String(error.clone());
                    }
                    if let Some(model) = &item.model {
                        value["model"] = Value::String(model.clone());
                    }
                    value
                }).collect::<Vec<_>>()
            }))
        }
        "results" => {
            let batch = require_batch(record)?;
            let inner = batch.inner.lock().expect("judge batch lock");
            let mut results = Map::new();
            for item in &inner.settled {
                if let Some(answers) = &item.answers {
                    let key = serde_json::to_string(&item.key)
                        .ok()
                        .map(|key| key.trim_matches('"').to_string())
                        .unwrap_or_default();
                    results.insert(key, Value::Object(answers.clone()));
                }
            }
            Ok(json!({"results": Value::Object(results)}))
        }
        "failed" => {
            let batch = require_batch(record)?;
            let inner = batch.inner.lock().expect("judge batch lock");
            let mut failed = Map::new();
            for item in &inner.settled {
                if let Some(error) = &item.error {
                    let key = serde_json::to_string(&item.key)
                        .ok()
                        .map(|key| key.trim_matches('"').to_string())
                        .unwrap_or_default();
                    failed.insert(key, Value::String(error.clone()));
                }
            }
            Ok(json!({"failed": Value::Object(failed)}))
        }
        "cancel" => {
            let batch = require_batch(record)?;
            let was_running = batch.inner.lock().expect("judge batch lock").running;
            batch.cancel.cancel();
            Ok(json!({"cancelled": was_running}))
        }
        "close" => {
            let id = record
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| batch_invalid("op requires a batch id"))?;
            let batch = JUDGE_BATCHES
                .lock()
                .expect("judge batches lock")
                .get(id)
                .cloned()
                .ok_or_else(|| format!("unknown judge_batch \"{id}\""))?;
            batch.cancel.cancel();
            JUDGE_BATCHES.lock().expect("judge batches lock").remove(id);
            Ok(json!({"closed": true}))
        }
        other => Err(batch_invalid(&format!("unknown op \"{other}\""))),
    }
}

fn require_batch(record: &Map<String, Value>) -> Result<Arc<JudgmentBatchRun>, String> {
    let id = record
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| batch_invalid("op requires a batch id"))?;
    JUDGE_BATCHES
        .lock()
        .expect("judge batches lock")
        .get(id)
        .cloned()
        .ok_or_else(|| format!("unknown judge_batch \"{id}\""))
}

/// TS `JudgmentBatch.start()`: `min(concurrency, total)` workers pull items
/// and judge each with up to `retries` retries; per-item failures land in
/// `settled` with `error`, only a run that dies wholesale sets the batch error.
fn start_batch(
    batch: Arc<JudgmentBatchRun>,
    items: Vec<BatchInput>,
    questions: Map<String, Value>,
    concurrency: usize,
    retries: u32,
    cell_signal: AbortSignal,
) {
    let items = Arc::new(items);
    let questions = Arc::new(questions);
    let worker_count = concurrency.clamp(1, items.len());
    let next = Arc::new(AtomicU64::new(0));
    let finished = Arc::new(AtomicU64::new(0));
    for _ in 0..worker_count {
        let batch = batch.clone();
        let items = items.clone();
        let questions = questions.clone();
        let next = next.clone();
        let finished = finished.clone();
        let cell_signal = cell_signal.clone();
        tokio::spawn(async move {
            loop {
                if batch.cancel.is_cancelled() || cell_signal.is_aborted() {
                    break;
                }
                let index = next.fetch_add(1, Ordering::Relaxed) as usize;
                let Some(input) = items.get(index) else {
                    break;
                };
                let settled = judge_one_batch_item(
                    &batch,
                    input,
                    &questions,
                    retries,
                    &AgentDir::default(),
                    &cell_signal,
                )
                .await;
                {
                    let mut inner = batch.inner.lock().expect("judge batch lock");
                    if let Some(model) = &settled.model {
                        inner.last_model = Some(model.clone());
                    }
                    if settled.error.is_some() {
                        inner.failed += 1;
                    }
                    inner.settled.push(settled);
                }
                batch.notify.notify_waiters();
            }
            // Last worker out closes the run and applies the minOk gate (TS
            // `#finish`): enough items judged ok, else the run dies wholesale.
            if finished.fetch_add(1, Ordering::Relaxed) + 1 == worker_count as u64 {
                let mut inner = batch.inner.lock().expect("judge batch lock");
                inner.running = false;
                let ok = inner.settled.len() - inner.failed;
                if ok < batch.min_ok {
                    let last = inner
                        .settled
                        .iter()
                        .rev()
                        .find_map(|item| item.error.as_deref())
                        .filter(|error| *error != "cancelled")
                        .unwrap_or("unknown error");
                    inner.error = Some(format!(
                        "judge_batch: only {ok}/{} item(s) judged (min_ok={}); last error: {last}",
                        batch.total, batch.min_ok
                    ));
                }
                drop(inner);
                batch.notify.notify_waiters();
            }
        });
    }
}

/// Judge one item with the resolved chain, up to `retries` extra attempts
/// (TS `#judgeItem`: `for (let attempt = 0; attempt <= retries; attempt++)`).
async fn judge_one_batch_item(
    batch: &JudgmentBatchRun,
    input: &BatchInput,
    questions: &Map<String, Value>,
    retries: u32,
    dir: &AgentDir,
    cell_signal: &AbortSignal,
) -> BatchItem {
    if batch.cancel.is_cancelled() || cell_signal.is_aborted() {
        return BatchItem {
            key: input.key.clone(),
            answers: None,
            error: Some("cancelled".to_string()),
            model: None,
        };
    }
    let request = JudgeRequest {
        state: input.state.clone(),
        questions: questions.clone(),
    };
    let chain = match omp_providers::resolve_judge_chain(dir) {
        Ok(chain) => chain,
        Err(error) => {
            return BatchItem {
                key: input.key.clone(),
                answers: None,
                error: Some(error.to_string()),
                model: None,
            };
        }
    };
    let cache = judgment_cache();
    let mut last_error: Option<String> = None;
    for attempt in 0..=retries {
        if batch.cancel.is_cancelled() || cell_signal.is_aborted() {
            return BatchItem {
                key: input.key.clone(),
                answers: None,
                error: Some("cancelled".to_string()),
                model: None,
            };
        }
        let _permit = match REQUEST_SLOTS.acquire().await {
            Ok(permit) => permit,
            Err(_) => {
                last_error = Some("judge_batch: request aborted".to_string());
                break;
            }
        };
        match omp_providers::judge(&chain, &request, cache.as_ref()).await {
            Ok(result) => {
                let value = judgment_result_value(&result);
                let answers = value
                    .get("answers")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                batch.inner.lock().expect("judge batch lock").cost += result.usage.cost;
                return BatchItem {
                    key: input.key.clone(),
                    answers: Some(answers),
                    error: None,
                    model: Some(format!("{}/{}", result.provider, result.model)),
                };
            }
            Err(error) => {
                last_error = Some(error.to_string());
                if attempt < retries {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }
    }
    BatchItem {
        key: input.key.clone(),
        answers: None,
        error: Some(last_error.unwrap_or_else(|| "judge_batch: judgment failed".to_string())),
        model: None,
    }
}

// ---------------------------------------------------------------------------
// `__workpool__` (TS `workpool-bridge.ts` + `task/workpool.ts`)
// ---------------------------------------------------------------------------
//
// TS pools keep subagents alive and feed them batches of queued items as
// follow-up turns; omp-rs subagents exit when their prompt settles, so pools
// run in the TS `eval.workpool.freshAgents` mode: every pushed item becomes
// one fresh worker (`<pool>-<n>`, an `agent://` handle that `status`/`cancel`/
// `write agent://` reach) running a one-item batch, at most `limit` at once,
// the rest queued FIFO. The aggregate pool job TS registers with the async
// job manager (final drained-pool delivery to the main agent) has no omp-rs
// counterpart; `status()`/`peek()` expose the same bookkeeping.

/// TS `WorkPoolItem.status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ItemStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl ItemStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

struct PoolItem {
    id: String,
    text: String,
    status: ItemStatus,
}

/// A live worker (fresh mode: dies with its single turn).
struct PoolWorker {
    id: String,
    /// Batch id of the running turn (TS `agent.jobId`).
    current: String,
}

struct PoolBatch {
    id: String,
    agent_id: String,
    items: Vec<usize>,
    status: ItemStatus,
    output: Option<String>,
}

/// One worker turn the pool wants launched.
#[derive(Debug, PartialEq, Eq)]
struct TurnStart {
    /// Index into `PoolState::batches`.
    batch: usize,
    worker_id: String,
    prompt: String,
}

/// Pure pool bookkeeping (TS `WorkPool` minus the job/session plumbing).
struct PoolState {
    name: String,
    agent: String,
    limit: usize,
    closed: bool,
    items: Vec<PoolItem>,
    workers: Vec<PoolWorker>,
    batches: Vec<PoolBatch>,
    fresh_queue: VecDeque<usize>,
    next_seq: u64,
    next_worker: u64,
}

impl PoolState {
    fn new(name: String, agent: String, limit: usize) -> Self {
        Self {
            name,
            agent,
            limit,
            closed: false,
            items: Vec::new(),
            workers: Vec::new(),
            batches: Vec::new(),
            fresh_queue: VecDeque::new(),
            next_seq: 1,
            next_worker: 1,
        }
    }

    /// TS `WorkPool.push`: queue items (ids `<pool>#<seq>`) and dispatch them.
    /// `taken` reports agent ids already in use (worker ids must be unique
    /// `agent://` handles).
    fn push(
        &mut self,
        texts: &[String],
        taken: &dyn Fn(&str) -> bool,
    ) -> Result<(Vec<String>, Vec<TurnStart>), String> {
        if self.closed {
            return Err(format!("workpool {} is closed", self.name));
        }
        let first = self.items.len();
        let mut ids = Vec::with_capacity(texts.len());
        for text in texts {
            let id = format!("{}#{}", self.name, self.next_seq);
            self.next_seq += 1;
            ids.push(id.clone());
            self.items.push(PoolItem {
                id,
                text: text.clone(),
                status: ItemStatus::Queued,
            });
        }
        let mut starts = Vec::new();
        for index in first..self.items.len() {
            self.dispatch(index, &mut starts, taken);
        }
        Ok((ids, starts))
    }

    /// TS `#dispatch` (fresh mode): spawn a worker while under the limit,
    /// otherwise wait in the FIFO queue.
    fn dispatch(
        &mut self,
        index: usize,
        starts: &mut Vec<TurnStart>,
        taken: &dyn Fn(&str) -> bool,
    ) {
        if self.closed || self.items[index].status != ItemStatus::Queued {
            return;
        }
        if self.workers.len() < self.limit {
            self.spawn_worker(index, starts, taken);
        } else {
            self.fresh_queue.push_back(index);
        }
    }

    /// TS `#spawn` + `#drain`: a new worker takes the item as its first batch.
    fn spawn_worker(
        &mut self,
        index: usize,
        starts: &mut Vec<TurnStart>,
        taken: &dyn Fn(&str) -> bool,
    ) {
        let base = format!("{}-{}", self.name, self.next_worker);
        self.next_worker += 1;
        let mut worker_id = base.clone();
        let mut suffix = 2;
        while taken(&worker_id) || self.workers.iter().any(|w| w.id == worker_id) {
            worker_id = format!("{base}-{suffix}");
            suffix += 1;
        }
        let batch_id = format!("{worker_id}-b1");
        self.items[index].status = ItemStatus::Running;
        self.workers.push(PoolWorker {
            id: worker_id.clone(),
            current: batch_id.clone(),
        });
        self.batches.push(PoolBatch {
            id: batch_id,
            agent_id: worker_id.clone(),
            items: vec![index],
            status: ItemStatus::Running,
            output: None,
        });
        let batch = self.batches.len() - 1;
        starts.push(TurnStart {
            batch,
            worker_id,
            prompt: self.batch_prompt(batch),
        });
    }

    /// TS `workpool-batch.md`, minus the keyed incremental `yield` contract
    /// (omp-rs children answer once, with their final text).
    fn batch_prompt(&self, batch: usize) -> String {
        let batch = &self.batches[batch];
        let mut prompt = format!(
            "<workpool pool=\"{}\" batch=\"{}\">\nYou are a worker in pool `{}`. Complete every item below in order. Your final answer is the outcome: for each item, its number and a self-contained outcome/evidence value.\n",
            self.name, batch.id, self.name
        );
        for (n, &index) in batch.items.iter().enumerate() {
            prompt.push_str(&format!("## Item {}\n{}\n", n + 1, self.items[index].text));
        }
        prompt.push_str("</workpool>");
        prompt
    }

    /// TS `#finishTurn` (fresh mode): settle the batch and its items, retire
    /// the worker, and hand the next queued item a new worker.
    fn finish_turn(
        &mut self,
        batch: usize,
        status: ItemStatus,
        output: String,
        taken: &dyn Fn(&str) -> bool,
    ) -> Vec<TurnStart> {
        let Some(entry) = self.batches.get_mut(batch) else {
            return Vec::new();
        };
        if entry.status != ItemStatus::Running {
            return Vec::new();
        }
        entry.status = status;
        entry.output = Some(output);
        let agent_id = entry.agent_id.clone();
        for &index in &entry.items {
            self.items[index].status = status;
        }
        self.workers.retain(|w| w.id != agent_id);
        let mut starts = Vec::new();
        while let Some(next) = self.fresh_queue.pop_front() {
            if !self.closed && self.items[next].status == ItemStatus::Queued {
                self.dispatch(next, &mut starts, taken);
                break;
            }
        }
        starts
    }

    /// TS `WorkPool.status()`.
    fn status(&self) -> Value {
        let mut counts = Map::new();
        for status in [
            ItemStatus::Queued,
            ItemStatus::Running,
            ItemStatus::Completed,
            ItemStatus::Failed,
            ItemStatus::Cancelled,
        ] {
            let n = self
                .items
                .iter()
                .filter(|item| item.status == status)
                .count();
            counts.insert(status.as_str().to_string(), json!(n));
        }
        json!({
            "name": self.name,
            "agent": self.agent,
            "limit": self.limit,
            "closed": self.closed,
            "freshAgents": true,
            "agents": self.workers.iter().map(|w| json!({
                "id": w.id,
                "state": "running",
                "queued": 0,
                "turns": 0,
                "current": w.current,
            })).collect::<Vec<_>>(),
            "items": counts,
            "batches": self.batches.len(),
        })
    }

    /// TS `WorkPool.peek()`: batch results without consuming anything.
    fn peek(&self) -> Value {
        let batches: Vec<Value> = self
            .batches
            .iter()
            .map(|batch| {
                let mut entry = json!({
                    "id": batch.id,
                    "agent": batch.agent_id,
                    "items": batch.items.iter().map(|&i| self.items[i].id.as_str()).collect::<Vec<_>>(),
                    "status": batch.status.as_str(),
                });
                if let Some(output) = &batch.output {
                    entry["output"] = Value::String(output.clone());
                }
                entry
            })
            .collect();
        json!({"batches": batches, "pending": self.pending()})
    }

    fn pending(&self) -> usize {
        self.items
            .iter()
            .filter(|item| matches!(item.status, ItemStatus::Queued | ItemStatus::Running))
            .count()
    }

    /// TS `WorkPool.close()`: stop accepting work and cancel queued items;
    /// running turns finish.
    fn close(&mut self) -> Value {
        self.closed = true;
        let mut dropped = Vec::new();
        for item in &mut self.items {
            if item.status == ItemStatus::Queued {
                item.status = ItemStatus::Cancelled;
                dropped.push(item.id.clone());
            }
        }
        self.fresh_queue.clear();
        json!({"dropped": dropped})
    }
}

/// A registered pool: bookkeeping plus what launching its turns needs.
struct WorkPool {
    state: Mutex<PoolState>,
    context: Option<String>,
    cwd: PathBuf,
    launcher: Arc<dyn Launcher>,
}

/// Process-local pools by name (TS `WorkPoolRegistry`, main-agent owner).
static WORKPOOLS: LazyLock<Mutex<HashMap<String, Arc<WorkPool>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn agent_id_taken(id: &str) -> bool {
    AGENT_HANDLES
        .lock()
        .expect("eval handles lock")
        .contains_key(id)
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
    clamp_tools(&child_spawn_policy().allowed_tools, requested)
}

/// Pure clamp against an explicit parent restriction (testable without argv).
fn clamp_tools(
    allowed: &Option<Vec<String>>,
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
/// plus extension/no-extensions passthrough and tool clamping.
pub(crate) fn apply_child_policy(spec: &mut omp_task::TaskSpec) {
    spec.depth = omp_task::current_task_depth() + 1;
    spec.extra_args = child_spawn_policy().extra_args.clone();
    spec.tools = clamp_child_tools(spec.tools.take());
}

/// Launch the given worker turns; each one, once settled, settles its batch
/// and launches whatever the pool dispatches next.
fn start_turns(pool: &Arc<WorkPool>, starts: Vec<TurnStart>) {
    if starts.is_empty() {
        return;
    }
    let agent = pool.state.lock().expect("workpool lock").agent.clone();
    for start in starts {
        let prompt = match &pool.context {
            Some(context) => format!("{context}\n\n{}", start.prompt),
            None => start.prompt,
        };
        let mut spec = omp_task::TaskSpec {
            id: Some(start.worker_id.clone()),
            agent: agent.clone(),
            prompt,
            cwd: pool.cwd.clone(),
            output_schema: None,
            schema_mode: omp_task::SchemaMode::Permissive,
            model: None,
            tools: None,
            depth: 0,
            extra_args: Vec::new(),
        };
        apply_child_policy(&mut spec);
        super::task::apply_agent_model(&mut spec);
        let handle = Handle::new("agent", start.worker_id.clone());
        // Reserve the worker id atomically: a concurrent registration of the
        // same id (another pool, an agent() call) must not have its handle
        // overwritten and orphaned from status/cancel.
        let worker_id = start.worker_id.clone();
        let inserted = {
            let mut registry = AGENT_HANDLES.lock().expect("eval handles lock");
            if registry.contains_key(&worker_id) {
                false
            } else {
                registry.insert(worker_id.clone(), handle.clone());
                true
            }
        };
        if !inserted {
            handle.settle(
                HandleStatus::Failed,
                None,
                None,
                Some(format!("worker id {worker_id} is already registered; refusing to overwrite its agent handle")),
            );
            let (status, text) = turn_outcome(&handle);
            let next = pool.state.lock().expect("workpool lock").finish_turn(
                start.batch,
                status,
                text,
                &agent_id_taken,
            );
            start_turns(&pool, next);
            continue;
        }
        let run = launch_agent(&handle, spec, &*pool.launcher);
        let pool = pool.clone();
        let batch = start.batch;
        tokio::spawn(async move {
            let output = run.await;
            settle_agent(&handle, &output, omp_task::SchemaMode::Permissive, false);
            handle.mark_run_finished();
            let (status, text) = turn_outcome(&handle);
            let next = pool.state.lock().expect("workpool lock").finish_turn(
                batch,
                status,
                text,
                &agent_id_taken,
            );
            start_turns(&pool, next);
        });
    }
}

/// Batch status + output from a settled worker handle (TS `#finishTurn`
/// status rule; output falls back to the error text).
fn turn_outcome(handle: &Handle) -> (ItemStatus, String) {
    let inner = handle.inner.lock().expect("eval handle lock");
    let status = match inner.status {
        HandleStatus::Completed => ItemStatus::Completed,
        HandleStatus::Cancelled => ItemStatus::Cancelled,
        HandleStatus::Failed | HandleStatus::Running => ItemStatus::Failed,
    };
    let output = inner
        .text
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .or(inner.error.as_deref())
        .unwrap_or("(no output)")
        .to_string();
    (status, output)
}

fn workpool_optional_string(
    record: &Map<String, Value>,
    key: &str,
) -> Result<Option<String>, String> {
    match record.get(key) {
        None => Ok(None),
        Some(Value::String(value)) if !value.trim().is_empty() => {
            Ok(Some(value.trim().to_string()))
        }
        Some(_) => Err(format!("workpool {key} must be a non-empty string")),
    }
}

fn run_workpool(args: &Value, ctx: &OpCtx) -> Result<Value, String> {
    run_workpool_with(args, &ctx.cwd, &LAUNCHER)
}

fn run_workpool_with(
    args: &Value,
    cwd: &std::path::Path,
    launcher: &Arc<dyn Launcher>,
) -> Result<Value, String> {
    let record = args
        .as_object()
        .ok_or_else(|| "workpool() arguments must be an object".to_string())?;
    let op = record
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| "workpool() requires an op".to_string())?;
    if op == "create" {
        return create_workpool(record, cwd, launcher);
    }
    let name = record
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "workpool operation requires a non-empty name".to_string())?;
    let pool = WORKPOOLS
        .lock()
        .expect("workpools lock")
        .get(name)
        .cloned()
        .ok_or_else(|| format!("unknown workpool \"{name}\""))?;
    match op {
        "push" => {
            let texts: Vec<String> = record
                .get("items")
                .and_then(Value::as_array)
                .and_then(|items| {
                    items
                        .iter()
                        .map(|item| item.as_str().map(str::to_string))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| "workpool push requires an items string array".to_string())?;
            let (ids, starts) = pool
                .state
                .lock()
                .expect("workpool lock")
                .push(&texts, &agent_id_taken)?;
            start_turns(&pool, starts);
            Ok(json!({"ids": ids}))
        }
        "status" => Ok(pool.state.lock().expect("workpool lock").status()),
        "peek" => Ok(pool.state.lock().expect("workpool lock").peek()),
        "close" => Ok(pool.state.lock().expect("workpool lock").close()),
        _ => Err(format!("unknown workpool operation \"{op}\"")),
    }
}

/// TS `runEvalWorkpool` `create`: `{name, agent, limit}`.
fn create_workpool(
    record: &Map<String, Value>,
    cwd: &std::path::Path,
    launcher: &Arc<dyn Launcher>,
) -> Result<Value, String> {
    let agent = workpool_optional_string(record, "agent")?;
    let requested = workpool_optional_string(record, "name")?;
    let context = workpool_optional_string(record, "context")?;
    match record.get("tools") {
        None => {}
        Some(Value::Array(tools))
            if tools
                .iter()
                .all(|tool| tool.as_str().is_some_and(|name| !name.is_empty())) =>
        {
            if !tools.is_empty() {
                return Err("Eval-defined tools are unavailable in this runtime.".to_string());
            }
        }
        Some(_) => return Err("workpool tools must be an array of non-empty strings".to_string()),
    }
    let agent = agent.unwrap_or_else(|| "task".to_string());
    if !omp_task::params::is_valid_agent_name(&agent) {
        return Err(format!("workpool() received invalid arguments: `agent` must be a valid agent name, got {agent:?}"));
    }
    // Every pushed item spawns a fresh worker subagent: same recursion gate
    // as agent() (TS `canSpawnAtDepth`).
    omp_task::recursion_gate()
        .map_err(|error| format!("workpool() cannot spawn workers: {error}"))?;
    let mut pools = WORKPOOLS.lock().expect("workpools lock");
    let name = if let Some(name) = requested {
        if pools.contains_key(&name) {
            return Err(format!("workpool \"{name}\" already exists"));
        }
        name
    } else {
        let base = format!("{agent}-pool");
        let mut name = base.clone();
        let mut suffix = 2;
        while pools.contains_key(&name) || agent_id_taken(&name) {
            name = format!("{base}-{suffix}");
            suffix += 1;
        }
        name
    };
    let pool = WorkPool {
        state: Mutex::new(PoolState::new(
            name.clone(),
            agent.clone(),
            subagent_concurrency(),
        )),
        context,
        cwd: cwd.to_path_buf(),
        launcher: launcher.clone(),
    };
    pools.insert(name.clone(), Arc::new(pool));
    Ok(json!({"name": name, "agent": agent, "limit": subagent_concurrency()}))
}

// ---------------------------------------------------------------------------
// `__wait__` / `__status__` / `__cancel__` (TS `handle-bridge.ts`)
// ---------------------------------------------------------------------------

async fn run_wait(args: &Value, ctx: &OpCtx) -> Result<Value, String> {
    let items = args
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| "wait() requires an items array".to_string())?;
    let refs: Vec<(&'static str, String)> = items
        .iter()
        .map(parse_handle_ref)
        .collect::<Result<_, _>>()?;
    let timeout_ms = args.get("timeoutMs").and_then(Value::as_f64);
    if let Some(ms) = timeout_ms {
        if !ms.is_finite() || ms < 0.0 {
            return Err("wait() timeoutMs must be a non-negative finite number".to_string());
        }
    }
    let handles: Vec<Arc<Handle>> = refs
        .iter()
        .map(|(kind, id)| lookup_handle(kind, id))
        .collect::<Result<_, _>>()?;
    let deadline = timeout_ms.map(|ms| Instant::now() + Duration::from_secs_f64(ms / 1000.0));

    loop {
        // TS `runEvalWait`: an aborted cell cancels every handle it waited on
        // and then awaits their settlement (`Promise.allSettled`) before
        // throwing, so an aborted caller never returns while the underlying
        // runs are still tearing down.
        if let Err(error) = throw_if_aborted(&ctx.signal) {
            for handle in &handles {
                cancel_handle(handle);
            }
            for handle in &handles {
                handle.wait_run_finished().await;
            }
            return Err(error);
        }
        let all_settled = handles.iter().all(|handle| handle.is_settled());
        if all_settled {
            break;
        }
        if let Some(deadline) = deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            tokio::time::sleep(WAIT_TICK.min(remaining)).await;
        } else {
            tokio::time::sleep(WAIT_TICK).await;
        }
    }
    Ok(json!({
        "items": handles.iter().map(|handle| handle.snapshot()).collect::<Vec<_>>()
    }))
}

fn run_status(args: &Value) -> Result<Value, String> {
    let item = args
        .get("item")
        .ok_or_else(|| "handle request requires an item".to_string())?;
    let (kind, id) = parse_handle_ref(item)?;
    Ok(lookup_handle(&kind, &id)?.snapshot())
}

fn run_cancel(args: &Value) -> Result<Value, String> {
    let item = args
        .get("item")
        .ok_or_else(|| "handle request requires an item".to_string())?;
    let (kind, id) = parse_handle_ref(item)?;
    let handle = lookup_handle(kind, &id)?;
    Ok(json!({"cancelled": cancel_handle(&handle)}))
}

// ---------------------------------------------------------------------------
// `__budget__` (TS `budget-bridge.ts` `runEvalBudget`)
// ---------------------------------------------------------------------------

/// TS `runEvalBudget` fallback path when neither a per-turn directive nor Goal
/// Mode exists: no ceiling (`total: null`), advisory (`hard: false`), `spent`
/// = output tokens. omp-rs has no `+Nk` turn budget / Goal Mode ledger, and
/// the main session's usage is not reachable from the tool layer, so `spent`
/// counts what TS adds via `recordEvalSubagentUsage`: output tokens reported
/// by settled eval subagents (`agent()` + workpool workers), cumulative for
/// the process rather than per turn.
pub fn budget_snapshot() -> Value {
    json!({
        "total": Value::Null,
        "spent": EVAL_SUBAGENT_OUTPUT.load(Ordering::Relaxed),
        "hard": false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn run_output(id: &str, status: omp_task::RunStatus, output: &str) -> omp_task::RunOutput {
        omp_task::RunOutput {
            id: id.to_string(),
            agent: "task".to_string(),
            status,
            exit_code: Some(0),
            output: output.to_string(),
            stderr: String::new(),
            schema: omp_task::SchemaValidation::NoSchema,
            duration: Duration::ZERO,
            truncated: false,
        }
    }

    /// Runner stand-in: records steers, settles on `finish`/`cancel`.
    struct MockRun {
        id: String,
        steered: Mutex<Vec<String>>,
        cancelled: AtomicBool,
        done: Mutex<Option<tokio::sync::oneshot::Sender<omp_task::RunOutput>>>,
    }

    impl MockRun {
        fn finish(&self, status: omp_task::RunStatus, output: &str) {
            if let Some(tx) = self.done.lock().unwrap().take() {
                let _ = tx.send(run_output(&self.id, status, output));
            }
        }

        fn live(&self) -> bool {
            self.done.lock().unwrap().is_some()
        }
    }

    impl RunCtl for MockRun {
        fn steer<'a>(
            &'a self,
            message: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<(), omp_task::SteerError>> + Send + 'a>> {
            Box::pin(async move {
                if !self.live() || self.cancelled.load(Ordering::SeqCst) {
                    return Err(omp_task::SteerError::NotRunning);
                }
                self.steered.lock().unwrap().push(message.to_string());
                Ok(())
            })
        }

        fn cancel(&self) -> bool {
            if !self.live() || self.cancelled.swap(true, Ordering::SeqCst) {
                return false;
            }
            self.finish(omp_task::RunStatus::Cancelled, "");
            true
        }

        fn output_tokens(&self) -> u64 {
            0
        }
    }

    #[derive(Default)]
    struct MockLauncher {
        runs: Mutex<Vec<(omp_task::TaskSpec, Arc<MockRun>)>>,
    }

    impl MockLauncher {
        fn run(&self, index: usize) -> Arc<MockRun> {
            self.runs.lock().unwrap()[index].1.clone()
        }

        fn launched(&self) -> usize {
            self.runs.lock().unwrap().len()
        }
    }

    impl Launcher for MockLauncher {
        fn launch(&self, spec: omp_task::TaskSpec) -> (Arc<dyn RunCtl>, RunFuture) {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let id = spec.id.clone().unwrap_or_default();
            let run = Arc::new(MockRun {
                id: id.clone(),
                steered: Mutex::new(Vec::new()),
                cancelled: AtomicBool::new(false),
                done: Mutex::new(Some(tx)),
            });
            self.runs.lock().unwrap().push((spec, run.clone()));
            let future: RunFuture = Box::pin(async move {
                rx.await
                    .unwrap_or_else(|_| run_output(&id, omp_task::RunStatus::Cancelled, ""))
            });
            let run: Arc<dyn RunCtl> = run;
            (run, future)
        }
    }

    fn spec(id: &str) -> omp_task::TaskSpec {
        omp_task::TaskSpec {
            id: Some(id.to_string()),
            agent: "task".to_string(),
            prompt: "p".to_string(),
            cwd: PathBuf::from("."),
            output_schema: None,
            schema_mode: omp_task::SchemaMode::Permissive,
            model: None,
            tools: None,
            depth: 0,
            extra_args: Vec::new(),
        }
    }

    /// Register an agent handle backed by a mock run, settled by a
    /// background task exactly like `run_agent` does.
    fn mock_agent(launcher: &MockLauncher, id: &str) -> (Arc<Handle>, Arc<MockRun>) {
        let handle = Handle::new("agent", id.to_string());
        AGENT_HANDLES
            .lock()
            .unwrap()
            .insert(id.to_string(), handle.clone());
        let run = launch_agent(&handle, spec(id), launcher);
        let settle = handle.clone();
        tokio::spawn(async move {
            let output = run.await;
            settle_agent(&settle, &output, omp_task::SchemaMode::Permissive, false);
            settle.mark_run_finished();
        });
        let mock = launcher.run(launcher.launched() - 1);
        (handle, mock)
    }

    async fn settled(handle: &Handle) {
        for _ in 0..200 {
            if handle.is_settled() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("handle {} never settled", handle.id);
    }

    fn no_ids(_: &str) -> bool {
        false
    }

    #[test]
    fn pool_dispatches_up_to_limit_then_queues_fifo() {
        let mut pool = PoolState::new("p".to_string(), "task".to_string(), 2);
        let texts: Vec<String> = ["a", "b", "c"].iter().map(|s| (*s).to_string()).collect();
        let (ids, starts) = pool.push(&texts, &no_ids).unwrap();
        assert_eq!(ids, ["p#1", "p#2", "p#3"]);
        let workers: Vec<&str> = starts.iter().map(|s| s.worker_id.as_str()).collect();
        assert_eq!(workers, ["p-1", "p-2"]);
        assert!(starts[0].prompt.contains("batch=\"p-1-b1\""));
        assert!(starts[0].prompt.contains("## Item 1\na\n"));
        let status = pool.status();
        assert_eq!(status["items"]["running"], 2);
        assert_eq!(status["items"]["queued"], 1);
        assert_eq!(status["agents"].as_array().unwrap().len(), 2);
        assert_eq!(status["agents"][0]["current"], "p-1-b1");
        assert_eq!(status["limit"], 2);
        assert_eq!(status["freshAgents"], true);

        // A finished turn retires its worker and hands the queued item a new one.
        let next = pool.finish_turn(
            starts[0].batch,
            ItemStatus::Completed,
            "done a".to_string(),
            &no_ids,
        );
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].worker_id, "p-3");
        assert!(next[0].prompt.contains("## Item 1\nc\n"));
        let peek = pool.peek();
        assert_eq!(peek["pending"], 2);
        assert_eq!(peek["batches"][0]["status"], "completed");
        assert_eq!(peek["batches"][0]["output"], "done a");
        assert_eq!(peek["batches"][0]["items"], json!(["p#1"]));
        assert!(peek["batches"][1].get("output").is_none());

        // Settling is idempotent.
        assert!(pool
            .finish_turn(
                starts[0].batch,
                ItemStatus::Failed,
                "x".to_string(),
                &no_ids
            )
            .is_empty());
        assert_eq!(pool.peek()["batches"][0]["status"], "completed");

        pool.finish_turn(
            starts[1].batch,
            ItemStatus::Failed,
            "boom".to_string(),
            &no_ids,
        );
        pool.finish_turn(
            next[0].batch,
            ItemStatus::Cancelled,
            "stop".to_string(),
            &no_ids,
        );
        let status = pool.status();
        assert_eq!(status["items"]["completed"], 1);
        assert_eq!(status["items"]["failed"], 1);
        assert_eq!(status["items"]["cancelled"], 1);
        assert_eq!(status["batches"], 3);
        assert_eq!(pool.peek()["pending"], 0);
        assert!(status["agents"].as_array().unwrap().is_empty());
    }

    #[test]
    fn pool_close_drops_queued_and_rejects_pushes() {
        let mut pool = PoolState::new("q".to_string(), "task".to_string(), 1);
        let texts: Vec<String> = ["a", "b", "c"].iter().map(|s| (*s).to_string()).collect();
        let (_, starts) = pool.push(&texts, &no_ids).unwrap();
        assert_eq!(starts.len(), 1);
        assert_eq!(pool.close(), json!({"dropped": ["q#2", "q#3"]}));
        assert_eq!(
            pool.push(&["d".to_string()], &no_ids).unwrap_err(),
            "workpool q is closed"
        );
        // The running turn still finishes; nothing new is dispatched.
        let next = pool.finish_turn(
            starts[0].batch,
            ItemStatus::Completed,
            "ok".to_string(),
            &no_ids,
        );
        assert!(next.is_empty());
        let status = pool.status();
        assert_eq!(status["closed"], true);
        assert_eq!(status["items"]["cancelled"], 2);
        assert_eq!(status["items"]["completed"], 1);
        assert_eq!(pool.close(), json!({"dropped": []}));
    }

    #[test]
    fn pool_worker_ids_avoid_taken_agent_ids() {
        let mut pool = PoolState::new("r".to_string(), "task".to_string(), 4);
        let taken = |id: &str| id == "r-1";
        let (_, starts) = pool
            .push(&["a".to_string(), "b".to_string()], &taken)
            .unwrap();
        let workers: Vec<&str> = starts.iter().map(|s| s.worker_id.as_str()).collect();
        assert_eq!(workers, ["r-1-2", "r-2"]);
        assert!(pool.push(&[], &taken).unwrap().0.is_empty());
    }

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
        let launcher = MockLauncher::default();
        let (handle, run) = mock_agent(&launcher, "test-msg-1");
        assert_eq!(
            deliver_message("test-msg-1", "look at foo.rs")
                .await
                .unwrap(),
            "Delivered to test-msg-1."
        );
        assert_eq!(*run.steered.lock().unwrap(), ["look at foo.rs"]);
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

    #[tokio::test]
    async fn workpool_ops_drive_worker_runs() {
        // Reads the recursion-depth env through `create_workpool`: serialize
        // with the depth-env tests.
        let _depth_env = DEPTH_ENV_LOCK.lock();
        let launcher = Arc::new(MockLauncher::default());
        let dyn_launcher: Arc<dyn Launcher> = launcher.clone();
        let cwd = PathBuf::from(".");
        let created = run_workpool_with(
            &json!({"op": "create", "name": "test-pool-e2e", "context": "Shared ctx"}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap();
        assert_eq!(
            created,
            json!({"name": "test-pool-e2e", "agent": "task", "limit": subagent_concurrency()})
        );
        let dup = run_workpool_with(
            &json!({"op": "create", "name": "test-pool-e2e"}),
            &cwd,
            &dyn_launcher,
        );
        assert_eq!(
            dup.unwrap_err(),
            "workpool \"test-pool-e2e\" already exists"
        );

        let pushed = run_workpool_with(
            &json!({"op": "push", "name": "test-pool-e2e", "items": ["one", "two"]}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap();
        assert_eq!(
            pushed,
            json!({"ids": ["test-pool-e2e#1", "test-pool-e2e#2"]})
        );
        assert_eq!(launcher.launched(), 2);
        {
            let runs = launcher.runs.lock().unwrap();
            assert!(runs[0].0.prompt.starts_with("Shared ctx\n\n<workpool"));
            assert!(runs[0].0.prompt.contains("one"));
            assert_eq!(runs[1].0.id.as_deref(), Some("test-pool-e2e-2"));
        }
        // Workers are agent handles: messageable and cancellable.
        assert_eq!(
            deliver_message("test-pool-e2e-1", "hint").await.unwrap(),
            "Delivered to test-pool-e2e-1."
        );
        launcher
            .run(0)
            .finish(omp_task::RunStatus::Completed, "first done");
        let cancel =
            run_cancel(&json!({"item": {"kind": "agent", "id": "test-pool-e2e-2"}})).unwrap();
        assert_eq!(cancel, json!({"cancelled": true}));

        let peek_args = json!({"op": "peek", "name": "test-pool-e2e"});
        let mut peek = Value::Null;
        for _ in 0..200 {
            peek = run_workpool_with(&peek_args, &cwd, &dyn_launcher).unwrap();
            if peek["pending"] == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(peek["pending"], 0, "{peek}");
        assert_eq!(peek["batches"][0]["status"], "completed");
        assert_eq!(peek["batches"][0]["output"], "first done");
        assert_eq!(peek["batches"][1]["status"], "cancelled");
        assert_eq!(peek["batches"][1]["output"], AGENT_CANCELLED_TEXT);

        let closed = run_workpool_with(
            &json!({"op": "close", "name": "test-pool-e2e"}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap();
        assert_eq!(closed, json!({"dropped": []}));
        let err = run_workpool_with(
            &json!({"op": "push", "name": "test-pool-e2e", "items": ["x"]}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap_err();
        assert_eq!(err, "workpool test-pool-e2e is closed");
        assert_eq!(
            run_workpool_with(
                &json!({"op": "status", "name": "nope-pool"}),
                &cwd,
                &dyn_launcher
            )
            .unwrap_err(),
            "unknown workpool \"nope-pool\""
        );
        assert_eq!(
            run_workpool_with(
                &json!({"op": "create", "tools": ["bash"]}),
                &cwd,
                &dyn_launcher
            )
            .unwrap_err(),
            "Eval-defined tools are unavailable in this runtime."
        );
    }

    #[test]
    fn budget_is_the_ts_fallback_shape() {
        let snap = budget_snapshot();
        assert!(snap["total"].is_null());
        assert_eq!(snap["hard"], false);
        assert!(snap["spent"].is_u64());
    }

    /// Finding 5: registering a worker whose id is already taken (the race
    /// between the pool's taken-check and the handle insert, or any direct
    /// `start_turns` dispatch) must not overwrite the existing handle — the
    /// turn fails explicitly and the original handle keeps owning its run.
    #[tokio::test]
    async fn pool_worker_id_collision_fails_without_overwriting() {
        let _depth_env = DEPTH_ENV_LOCK.lock();
        let launcher = Arc::new(MockLauncher::default());
        let dyn_launcher: Arc<dyn Launcher> = launcher.clone();
        let cwd = PathBuf::from(".");
        run_workpool_with(
            &json!({"op": "create", "name": "test-pool-collision"}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap();
        let pool = WORKPOOLS
            .lock()
            .unwrap()
            .get("test-pool-collision")
            .cloned()
            .unwrap();

        // Occupy the id a turn will try to register.
        let existing = Handle::new("agent", "occupied-worker".to_string());
        AGENT_HANDLES
            .lock()
            .unwrap()
            .insert("occupied-worker".to_string(), existing.clone());

        start_turns(
            &pool,
            vec![TurnStart {
                batch: usize::MAX,
                worker_id: "occupied-worker".to_string(),
                prompt: "do work".to_string(),
            }],
        );
        assert_eq!(launcher.launched(), 0, "no run launches on a collision");

        // The existing handle still owns the id, untouched by the failed turn.
        let registered = AGENT_HANDLES
            .lock()
            .unwrap()
            .get("occupied-worker")
            .cloned();
        assert!(Arc::ptr_eq(&registered.unwrap(), &existing));
        assert_eq!(existing.snapshot()["status"], "running");
        AGENT_HANDLES.lock().unwrap().remove("occupied-worker");
        WORKPOOLS.lock().unwrap().remove("test-pool-collision");
    }

    /// A mock run whose cancel does *not* settle the run: the run resolves
    /// only when `release()` sends its output (like a real child that still
    /// has to be reaped after the kill).
    struct SlowMockRun {
        inner: MockRun,
        release: Mutex<Option<tokio::sync::oneshot::Sender<omp_task::RunOutput>>>,
    }

    impl RunCtl for SlowMockRun {
        fn steer<'a>(
            &'a self,
            message: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<(), omp_task::SteerError>> + Send + 'a>> {
            self.inner.steer(message)
        }

        fn cancel(&self) -> bool {
            // The kill is requested but the run stays unresolved until
            // `release()` (a real child still has to be reaped).
            if self.inner.cancelled.swap(true, Ordering::SeqCst) {
                return false;
            }
            true
        }

        fn output_tokens(&self) -> u64 {
            0
        }
    }

    struct SlowMockLauncher {
        runs: Mutex<Vec<Arc<SlowMockRun>>>,
    }

    impl Launcher for SlowMockLauncher {
        fn launch(&self, spec: omp_task::TaskSpec) -> (Arc<dyn RunCtl>, RunFuture) {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let id = spec.id.clone().unwrap_or_default();
            let run = Arc::new(SlowMockRun {
                inner: MockRun {
                    id: id.clone(),
                    steered: Mutex::new(Vec::new()),
                    cancelled: AtomicBool::new(false),
                    done: Mutex::new(None),
                },
                release: Mutex::new(Some(tx)),
            });
            self.runs.lock().unwrap().push(run.clone());
            let future: RunFuture = Box::pin(async move {
                rx.await
                    .unwrap_or_else(|_| run_output(&id, omp_task::RunStatus::Cancelled, ""))
            });
            (run, future)
        }
    }

    /// Finding 6: an aborted wait() cancels every handle and waits for them
    /// to settle before erroring (TS `handle-bridge.ts` `Promise.allSettled`).
    #[tokio::test]
    async fn aborted_wait_cancels_then_waits_for_settlement() {
        let _depth_env = DEPTH_ENV_LOCK.lock();
        let launcher = Arc::new(SlowMockLauncher {
            runs: Mutex::new(Vec::new()),
        });
        let dyn_launcher: Arc<dyn Launcher> = launcher.clone();
        let cwd = PathBuf::from(".");
        run_workpool_with(
            &json!({"op": "create", "name": "test-pool-abort"}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap();
        run_workpool_with(
            &json!({"op": "push", "name": "test-pool-abort", "items": ["one", "two"]}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap();
        assert_eq!(launcher.runs.lock().unwrap().len(), 2);

        let controller = pa_agent::abort::AbortController::new();
        let signal = controller.signal();
        controller.abort();
        let ctx = OpCtx {
            cwd: cwd.clone(),
            signal,
            on_update: None,
            tool_call_id: "test-abort".to_string(),
        };
        let wait_args = json!({"items": [
            {"kind": "agent", "id": "test-pool-abort-1"},
            {"kind": "agent", "id": "test-pool-abort-2"},
        ]});
        let wait = run_wait(&wait_args, &ctx);
        tokio::pin!(wait);
        tokio::select! {
            _ = &mut wait => panic!("wait() must not return before the cancelled runs finish"),
            _ = tokio::time::sleep(Duration::from_millis(200)) => {}
        }
        // The runs were asked to cancel but are not finished yet; the handles
        // already report cancelled (TS `manager.cancel` flips job status at once).
        assert!(launcher
            .runs
            .lock()
            .unwrap()
            .iter()
            .all(|run| run.inner.cancelled.load(Ordering::SeqCst)));
        let handle = lookup_handle("agent", "test-pool-abort-1").unwrap();
        assert_eq!(handle.snapshot()["status"], "cancelled");
        assert!(!handle
            .run_finished
            .load(std::sync::atomic::Ordering::SeqCst));

        for run in launcher.runs.lock().unwrap().iter() {
            let tx = run.release.lock().unwrap().take().unwrap();
            let _ = tx.send(run_output(
                &run.inner.id,
                omp_task::RunStatus::Cancelled,
                "",
            ));
        }
        let err = tokio::time::timeout(Duration::from_secs(5), &mut wait)
            .await
            .expect("wait() returns once the cancelled runs finish")
            .unwrap_err();
        assert_eq!(err, "aborted");
        for id in ["test-pool-abort-1", "test-pool-abort-2"] {
            let handle = lookup_handle("agent", id).unwrap();
            assert!(
                handle
                    .run_finished
                    .load(std::sync::atomic::Ordering::SeqCst),
                "{id} run finished before the abort error"
            );
        }
        WORKPOOLS.lock().unwrap().remove("test-pool-abort");
        for id in ["test-pool-abort-1", "test-pool-abort-2"] {
            AGENT_HANDLES.lock().unwrap().remove(id);
        }
    }

    /// Serialize the tests that mutate the recursion-depth env: the env is
    /// process-global and `create_workpool` reads it through the gate.
    static DEPTH_ENV_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// Finding 8: at `OMP_TASK_MAX_RECURSION_DEPTH` (TS `task.maxRecursionDepth`,
    /// default 2) agent/workpool/task spawns are refused with the TS message.
    #[test]
    fn recursion_gate_matches_ts_depth_rule() {
        let _env = DEPTH_ENV_LOCK.lock();
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

    #[tokio::test]
    async fn workpool_refuses_to_spawn_at_max_depth() {
        let _env = DEPTH_ENV_LOCK.lock();
        // Simulate a depth-2 session (the TS default ceiling): spawns from
        // here are refused with the TS message.
        std::env::set_var(omp_task::TASK_DEPTH_ENV, "2");
        let launcher = Arc::new(MockLauncher::default());
        let dyn_launcher: Arc<dyn Launcher> = launcher.clone();
        let cwd = PathBuf::from(".");
        let err = run_workpool_with(
            &json!({"op": "create", "name": "test-pool-depth"}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap_err();
        assert_eq!(
            err,
            "workpool() cannot spawn workers: Cannot spawn another agent at task depth 2; maximum depth is 2."
        );
        assert_eq!(launcher.launched(), 0);
        std::env::remove_var(omp_task::TASK_DEPTH_ENV);
        WORKPOOLS.lock().unwrap().remove("test-pool-depth");
    }

    /// Finding 9: child tool allow-lists clamp to the parent's enabled set.
    #[test]
    fn child_tools_clamp_to_the_parent_allow_list() {
        let parent = Some(vec!["read".to_string(), "bash".to_string()]);
        // Unrestricted request inherits the parent's restriction.
        assert_eq!(
            clamp_tools(&parent, None),
            Some(vec!["read".to_string(), "bash".to_string()])
        );
        // Requests keep only tools the parent has.
        assert_eq!(
            clamp_tools(&parent, Some(vec!["read".to_string(), "eval".to_string()])),
            Some(vec!["read".to_string()])
        );
        // Nothing in common: the child gets no tools (omp `__none__`), never
        // the full default toolset.
        assert_eq!(
            clamp_tools(&parent, Some(vec!["eval".to_string()])),
            Some(vec!["__none__".to_string()])
        );
        // Unrestricted parent: the request passes through untouched.
        assert_eq!(
            clamp_tools(&None, Some(vec!["eval".to_string()])),
            Some(vec!["eval".to_string()])
        );
    }
}
