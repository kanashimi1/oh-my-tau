//! Agent-family host ops for eval cells: the host half of the `agent()`,
//! `completion()`, `judge()`, `judge_batch()`, `workpool()`, `budget`, `wait()`,
//! `status()` and `cancel()` prelude helpers (TS `eval/agent-bridge.ts`,
//! `eval/completion-bridge.ts`, `eval/judgment-bridge.ts`,
//! `eval/judgment-batch-bridge.ts`, `eval/workpool-bridge.ts`,
//! `eval/budget-bridge.ts`, `eval/handle-bridge.ts`; payload shapes verbatim
//! from `crates/omp-eval/src/prelude_agents.py`).
//!
//! omp-rs subagents are one-shot `omp-rs -p` subprocesses driven by
//! [`omp_task`]'s [`TaskRunner`] (the same machinery the `task` tool uses), so
//! agent handles are process-local registry entries whose jobs run in the
//! background; there is no live input channel into a running child, which
//! limits what `cancel` and agent messaging can do (explicit errors instead of
//! silent success). Judgments reuse [`omp_providers::judge`]'s resolved chain;
//! completions run one-shot against the session's live provider target
//! ([`crate::agent::Target`], installed by [`set_runtime_hooks`]) or a resolved
//! `@smol`/`@slow` role.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use omp_providers::{JudgeRequest, JudgmentCache};
use omp_providers::{AgentDir, ResolvedModel};
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
static REQUEST_SLOTS: LazyLock<Semaphore> = LazyLock::new(|| Semaphore::new(EVAL_HANDLE_CONCURRENCY));

/// Synthetic tool the model is forced to call when a `schema` is supplied
/// (TS `STRUCTURED_TOOL_NAME`).
const STRUCTURED_TOOL_NAME: &str = "respond";

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
    let hooks = guard
        .as_ref()
        .ok_or_else(|| "eval runtime is not initialized (set_runtime_hooks not called)".to_string())?;
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
    /// Completion handles are abortable through the pa-ai cancellation token;
    /// agent jobs are subprocesses with no kill channel (see `cancel_handle`).
    cancel: CancellationToken,
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
            cancel: CancellationToken::new(),
        })
    }

    fn settle(&self, status: HandleStatus, text: Option<String>, data: Option<Value>, error: Option<String>) {
        {
            let mut inner = self.inner.lock().expect("eval handle lock");
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
        _ => return Err("handle must contain kind agent|completion and a non-empty id".to_string()),
    };
    let id = item
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| "handle must contain kind agent|completion and a non-empty id".to_string())?;
    Ok((kind, id.to_string()))
}

fn lookup_handle(kind: &str, id: &str) -> Result<Arc<Handle>, String> {
    let registry = if kind == "agent" { &AGENT_HANDLES } else { &COMPLETION_HANDLES };
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
        "__workpool__" => Some(Err(
            "workpool() is not supported in this runtime: it needs in-process keep-alive subagents; use agent() fan-out instead".to_string(),
        )),
        "__budget__" => Some(Ok(budget_snapshot())),
        "__wait__" => Some(run_wait(&args, ctx).await),
        "__status__" => Some(run_status(&args)),
        "__cancel__" => Some(run_cancel(&args)),
        _ => None,
    }
}

/// TS `AgentHandle.send` (`agent://<id>` write target): posts a steering
/// message into a *running* eval subagent's input queue. omp-rs subagents are
/// one-shot `omp-rs -p` subprocesses with no live input channel, so this is an
/// explicit failure rather than a silent no-op.
pub async fn send_to_agent(id: &str, _content: &str) -> Result<Value, String> {
    let running = AGENT_HANDLES
        .lock()
        .expect("eval handles lock")
        .get(id)
        .map(|handle| !handle.inner.lock().expect("eval handle lock").settled);
    match running {
        Some(true) => Err(format!(
            "agent messaging (write agent://{id}) is not supported in this runtime: eval subagents are one-shot subprocesses without a live input channel"
        )),
        _ => Err(format!(
            "unknown agent handle {id}; result expired — read agent://{id}"
        )),
    }
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
        .ok_or_else(|| "agent() received invalid arguments: prompt must be a non-empty string".to_string())?;
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
    if args.get("isolated").is_some() || args.get("apply").is_some() || args.get("merge").is_some() {
        return Err(
            "agent() isolation controls (isolated/apply/merge) are not supported in this runtime".to_string(),
        );
    }
    if args
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err("Eval-defined tools are unavailable in this runtime.".to_string());
    }

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
    let spec = omp_task::runner::TaskSpec::from_item(&item, None, ctx.cwd.clone());

    let handle = Handle::new("agent", id.clone());
    AGENT_HANDLES
        .lock()
        .expect("eval handles lock")
        .insert(id.clone(), handle.clone());

    let binary = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("omp-rs"));
    let runner = omp_task::runner::TaskRunner::new(binary, EVAL_HANDLE_CONCURRENCY);

    tokio::spawn(async move {
        let output = runner.spawn(spec).await;
        // A run whose output violates the declared schema fails in strict
        // mode (TS `structured-subagent` schema enforcement).
        let ok = output.status == omp_task::runner::RunStatus::Completed
            && !output.schema_failed(schema_mode);
        let status = match output.status {
            _ if ok => HandleStatus::Completed,
            omp_task::runner::RunStatus::Cancelled => HandleStatus::Cancelled,
            _ => HandleStatus::Failed,
        };
        let text = if output.output.trim().is_empty() {
            output.stderr.trim().to_string()
        } else {
            output.output.clone()
        };
        let (text, data, error) = if status == HandleStatus::Completed {
            let data = if schema_wanted {
                serde_json::from_str::<Value>(output.output.trim()).ok()
            } else {
                None
            };
            (Some(text), data, None)
        } else {
            let error = omp_task::envelope::format_task_result(&output)
                .error
                .unwrap_or_else(|| format!("agent() subagent '{}' failed.", output.agent));
            (None, None, Some(error))
        };
        handle.settle(status, text, data, error);
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
            candidates.push(CompletionCandidate {
                model: target.model,
                api_key: target.api_key,
                service_tier: target.service_tier,
                headers: target.headers.map(|headers| headers.into_iter().collect()),
                thinking: None,
            });
            if let Ok(primary) = omp_providers::resolve_in_async(dir.clone(), "@default".to_string()).await {
                candidates.extend(walk_fallbacks(dir, primary).await);
            }
        }
        "smol" | "slow" => {
            let primary = omp_providers::resolve_in_async(dir.clone(), format!("@{tier}"))
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
) -> Result<(String, Option<Value>), String> {
    let tools = schema.map(|schema| {
        vec![Tool {
            name: STRUCTURED_TOOL_NAME.to_string(),
            description: "Return your answer by calling this tool with the requested structured fields.".to_string(),
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
        let response = match pa_ai::complete_simple(&candidate.model, &context, Some(options)).await {
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
                AssistantContentBlock::ToolCall(call) if call.name == STRUCTURED_TOOL_NAME => Some(call),
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
                        "completion() did not return a structured response matching the schema.".to_string()
                    })?
                }
            };
            return Ok((serde_json::to_string(&value).unwrap_or_default(), Some(value)));
        }
        let text = assistant_text(&response.content);
        if text.is_empty() {
            return Err("completion() returned no text output.".to_string());
        }
        return Ok((text, None));
    }
    Err(last_error.unwrap_or_else(|| "completion() request failed.".to_string()))
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
    let tier = args.get("model").and_then(Value::as_str).unwrap_or("default");
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
            Ok((text, data)) => handle.settle(HandleStatus::Completed, Some(text), data, None),
            Err(error) => {
                let status = if cancel.is_cancelled() {
                    HandleStatus::Cancelled
                } else {
                    HandleStatus::Failed
                };
                handle.settle(status, None, None, Some(error));
            }
        }
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
) -> Result<(String, Option<Value>), String> {
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
        result = execute_completion(&prompt, system.as_deref(), schema.as_ref(), &candidates, &cancel) => result,
    }
}

// ---------------------------------------------------------------------------
// `__judge__` (TS `judgment-bridge.ts` `runEvalJudgment`)
// ---------------------------------------------------------------------------

/// TS `parseState`: a non-empty string or any JSON value (object/array).
fn parse_state(value: &Value) -> Result<Value, String> {
    match value {
        Value::String(text) if !text.is_empty() => Ok(value.clone()),
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
        return Err("judge() received invalid arguments: questions must be an object keyed by question id".to_string());
    };
    if questions.is_empty() {
        return Err("judge() received invalid arguments: questions must contain at least one question".to_string());
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
                    return Err(invalid(&format!("choice question \"{id}\" needs at least two options")));
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
                if let Some(criteria) = question.get("criteria") {
                    let Some(criteria) = criteria.as_object() else {
                        return Err(invalid(&format!(
                            "bool question \"{id}\" criteria must be {{ true?: string, false?: string }}"
                        )));
                    };
                    for side in ["true", "false"] {
                        if let Some(description) = criteria.get(side) {
                            if !description.is_string() {
                                return Err(invalid(&format!(
                                    "bool question \"{id}\" criteria.{side} must be a string"
                                )));
                            }
                        }
                    }
                }
                json!({"type": "noul", "instructions": instructions, "criteria": question.get("criteria").cloned().unwrap_or(Value::Null)})
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
        return Err("judge() received invalid arguments: expected { state, questions }".to_string());
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
    let result = omp_providers::judge(&chain, &JudgeRequest { state, questions }, judgment_cache().as_ref())
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
            return Err(batch_invalid(&format!("item {index} needs a string or number key")));
        };
        let key = entry.get("key").cloned().unwrap_or(Value::Null);
        let valid_key = matches!(&key, Value::String(s) if !s.is_empty()) || key.is_number();
        if !valid_key {
            return Err(batch_invalid(&format!("item {index} needs a string or number key")));
        }
        let key_id = serde_json::to_string(&key).unwrap_or_default();
        if seen.contains(&key_id) {
            return Err(batch_invalid(&format!("duplicate item key {key_id}")));
        }
        seen.push(key_id);
        parsed.push(BatchInput {
            key,
            state: parse_state(entry.get("state").unwrap_or(&Value::Null))
                .map_err(|_| batch_invalid(&format!("item {index} state must be a string, a JSON object, or a JSON array")))?,
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
        Some(_) => Err(batch_invalid(&format!("{name} must be a non-negative integer"))),
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
            let questions = parse_questions(record.get("questions"))
                .map_err(|error| error.replace("judge()", "judge_batch()"))?;
            let total = items.len();
            // TS `parseBatchOptions`: an explicit 0 falls back to the default
            // ceiling (`concurrency > 0 ? concurrency : EVAL_HANDLE_CONCURRENCY`).
            let concurrency = parse_batch_count(record.get("concurrency"), "concurrency", EVAL_HANDLE_CONCURRENCY as u64)?;
            let concurrency = if concurrency == 0 {
                EVAL_HANDLE_CONCURRENCY
            } else {
                concurrency as usize
            };
            let retries = parse_batch_count(record.get("retries"), "retries", u64::from(DEFAULT_BATCH_RETRIES))? as u32;
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
            start_batch(batch.clone(), items, questions, concurrency, retries, ctx.signal.clone());
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
                    return Err(batch_invalid("timeoutMs must be a non-negative finite number"));
                };
                if !ms.is_finite() || ms < 0.0 {
                    return Err(batch_invalid("timeoutMs must be a non-negative finite number"));
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
                let settled =
                    judge_one_batch_item(&batch, input, &questions, retries, &AgentDir::default(), &cell_signal)
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
        throw_if_aborted(&ctx.signal)?;
        let all_settled = handles
            .iter()
            .all(|handle| handle.inner.lock().expect("eval handle lock").settled);
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
    let handle = lookup_handle(&kind, &id)?;
    let settled = handle.inner.lock().expect("eval handle lock").settled;
    if kind == "completion" {
        if settled {
            return Ok(json!({"cancelled": false}));
        }
        handle.cancel.cancel();
        return Ok(json!({"cancelled": true}));
    }
    // Agent jobs are one-shot subprocesses with no kill channel in this
    // runtime; refusing beats reporting a cancellation that never happens.
    if settled {
        return Ok(json!({"cancelled": false}));
    }
    Err(format!(
        "cancel() of running eval agent {id} is not supported in this runtime: subagents are one-shot subprocesses without a kill channel"
    ))
}

// ---------------------------------------------------------------------------
// `__budget__` (TS `budget-bridge.ts` `runEvalBudget`)
// ---------------------------------------------------------------------------

/// TS `runEvalBudget` fallback path when neither a per-turn directive nor Goal
/// Mode exists: no ceiling (`total: null`), `spent` from output usage where
/// available (none is wired here), advisory (`hard: false`). omp-rs has no
/// turn-budget / Goal Mode ledger yet, so the snapshot is the fallback shape.
pub fn budget_snapshot() -> Value {
    json!({"total": Value::Null, "spent": 0, "hard": false})
}
