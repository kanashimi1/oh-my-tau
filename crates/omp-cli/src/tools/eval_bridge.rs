//! Host-side bridge for eval cells: answers the runners' host ops
//! (`{"type":"host","op":...}` NDJSON frames) by dispatching to this
//! process's tools, so cell code can call `read`/`write`/`bash`/…, message
//! subagents and query the runtime. TS reference: `eval/py/tool-bridge.ts`
//! (`/v1/tool` loopback, `callSessionTool`, `bridgeValueFromToolResult`),
//! `eval/js/tool-bridge.ts`; agent-family ops (`__agent__`, `__wait__`, …)
//! are delegated to [`super::eval_agents`] unchanged.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use pa_agent::abort::{race_with_abort, AbortSignal};
use pa_agent::agent_loop::{AfterToolCallFn, BeforeToolCallFn};
use pa_agent::types::{
    AfterToolCallContext, AgentContext, AgentTool, AgentToolResult, AgentToolUpdateCallback,
    AssistantContent, AssistantMessage, AssistantMeta, BeforeToolCallContext, StopReason, ToolCall,
    ToolResultContent, Usage,
};
use pa_agent::validation::validate_tool_arguments;
use serde_json::{json, Value};

use super::Env;

/// Per-cell execution context handed down from the `eval` tool: the cell's
/// abort signal (a bridged tool call dies with its cell), the partial-result
/// callback, and the parent tool_call_id child calls derive their ids from.
pub struct OpCtx {
    pub cwd: PathBuf,
    pub signal: AbortSignal,
    pub on_update: Option<AgentToolUpdateCallback>,
    pub tool_call_id: String,
}

/// The host half of the cell↔host wire protocol: one instance per running
/// cell, passed to [`omp_eval::HostBridge`]. `call` runs on the kernel's
/// spawn-blocking thread and drives async tool work through the stored tokio
/// Handle (`rt.block_on`), so tools observe a normal multi-thread runtime.
pub struct EvalBridge {
    env: Arc<Env>,
    ctx: OpCtx,
    rt: tokio::runtime::Handle,
    policy: BridgePolicy,
}

/// Tools the bridge may dispatch to. Set once per process by the runtime
/// (the session's enabled tool list, so cell-side `read`/`bash`/… resolve
/// exactly like model-side calls); before that (tests), the bridge falls back
/// to [`super::builtin_tools`]. Hidden tools the session never enabled are not
/// reachable (TS `getToolForEvalBridge` accepts enabled names only).
static SESSION_TOOLS: OnceLock<RwLock<Vec<Arc<dyn AgentTool>>>> = OnceLock::new();

/// Install the process-global session tool list (TS `ToolSession` the py
/// bridge's `entry.toolSession` resolves against).
pub fn set_session_tools(tools: Vec<Arc<dyn AgentTool>>) {
    let slot = SESSION_TOOLS.get_or_init(|| RwLock::new(Vec::new()));
    *slot.write().expect("session tools lock") = tools;
}

fn session_tools(env: &Env) -> Vec<Arc<dyn AgentTool>> {
    if let Some(slot) = SESSION_TOOLS.get() {
        let tools = slot.read().expect("session tools lock").clone();
        if !tools.is_empty() {
            return tools;
        }
    }
    super::builtin_tools(&env.cwd, env.edit_mode)
}

/// The session's extension tool-call hooks (`tool_call` block/input rewrite,
/// `tool_result` rewrite), installed by `agent::build_agent` next to the
/// agent loop's own copy so bridged calls pass the same policy (TS
/// `ExtensionToolWrapper` around every tool `getToolForEvalBridge` returns).
type SessionHooks = (Option<BeforeToolCallFn>, Option<AfterToolCallFn>);
static SESSION_HOOKS: RwLock<SessionHooks> = RwLock::new((None, None));

/// Install the process-global tool-call hooks the bridge applies.
pub fn set_session_hooks(before: Option<BeforeToolCallFn>, after: Option<AfterToolCallFn>) {
    *SESSION_HOOKS.write().expect("session hooks lock") = (before, after);
}

/// Tools a cell may never call through the bridge. `checkpoint`/`rewind`
/// only take effect as direct tool results (TS `callSessionTool` refusal);
/// a nested `eval` would wait on the per-cwd kernel lock its own cell holds.
fn refused_reason(name: &str) -> Option<String> {
    match name {
        "checkpoint" | "rewind" => Some(format!(
            "`{name}` cannot run through the eval bridge; call the direct `{name}` tool."
        )),
        "eval" => Some(
            "`eval` cannot run from inside an eval cell (the kernel is busy running this \
             cell); run the code directly in this cell instead."
                .to_string(),
        ),
        _ => None,
    }
}

/// Tool-call policy one bridge instance enforces: the session's enabled
/// tools plus the extension hooks the agent loop runs around model calls.
#[derive(Clone)]
pub(crate) struct BridgePolicy {
    tools: Vec<Arc<dyn AgentTool>>,
    before: Option<BeforeToolCallFn>,
    after: Option<AfterToolCallFn>,
}

impl BridgePolicy {
    /// The process session's policy (see [`set_session_tools`] /
    /// [`set_session_hooks`]).
    fn session(env: &Env) -> Self {
        let (before, after) = SESSION_HOOKS.read().expect("session hooks lock").clone();
        Self {
            tools: session_tools(env),
            before,
            after,
        }
    }

    /// An enabled, bridge-callable tool by name.
    fn find(&self, name: &str) -> Option<Arc<dyn AgentTool>> {
        if refused_reason(name).is_some() {
            return None;
        }
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    fn callable(&self) -> impl Iterator<Item = &Arc<dyn AgentTool>> {
        self.tools
            .iter()
            .filter(|t| refused_reason(t.name()).is_none())
    }
}

/// Synthetic single-call assistant turn for the hook contexts (bridged calls
/// have no model message; extension hooks read only the tool call).
fn synthetic_turn(call: &ToolCall) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantContent::ToolCall(call.clone())],
        api: String::new(),
        provider: String::new(),
        model: String::new(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        stop_reason_raw: None,
        error_status: None,
        error_message: None,
        timestamp: 0,
        meta: AssistantMeta::default(),
    }
}

/// Ops owned by [`super::eval_agents`] (`agent-bridge.ts`,
/// `completion-bridge.ts`, `judgment-bridge.ts`, `judgment-batch-bridge.ts`,
/// `workpool-bridge.ts`, `budget-bridge.ts`, `handle-bridge.ts`).
const AGENT_OPS: [&str; 9] = [
    "__agent__",
    "__completion__",
    "__judge__",
    "__judge_batch__",
    "__workpool__",
    "__budget__",
    "__wait__",
    "__status__",
    "__cancel__",
];

impl EvalBridge {
    #[must_use]
    pub fn new(env: Arc<Env>, ctx: OpCtx, rt: tokio::runtime::Handle) -> Self {
        let policy = BridgePolicy::session(&env);
        Self {
            env,
            ctx,
            rt,
            policy,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_policy(mut self, policy: BridgePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Publish one structured status event (TS `_emit_status` /
    /// `JsStatusEvent`) as a partial tool result, in the TS display-mime
    /// bundle shape the TUI already renders.
    fn emit_status(&self, event: Value) {
        let Some(on_update) = self.ctx.on_update.as_ref() else {
            return;
        };
        on_update(AgentToolResult {
            content: Vec::new(),
            details: json!({"application/x-omp-status": event}),
            terminate: None,
            is_error: None,
        });
    }

    async fn dispatch(&self, op: &str, args: Value) -> Result<Value, String> {
        if op == "__prelude__" {
            // Kernel-side capability query: answer with the tool descriptor
            // when this host can serve `name`, else refuse explicitly.
            let name = args.get("name").and_then(Value::as_str).unwrap_or_default();
            return self.policy.find(name).map_or_else(
                || Err(format!("prelude capability not supported: {name}")),
                |tool| Ok(tool_descriptor(&tool)),
            );
        }
        if AGENT_OPS.contains(&op) || op.starts_with("__") {
            // Agent-family ops are F's; a declined op is an unknown op.
            return match super::eval_agents::handle(&self.env, op, args, &self.ctx).await {
                Some(result) => result,
                None => Err(format!("unknown eval host op: {op}")),
            };
        }
        match op {
            "read" => self.read_helper(args).await,
            "write" => self.write_helper(args).await,
            "output" => self.output_helper(&args),
            "env" => env_helper(&args),
            "tool.list" => Ok(Value::Array(
                self.policy.callable().map(tool_descriptor).collect(),
            )),
            "tool.describe" => {
                let name = args
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "tool.describe requires a string `name`".to_string())?;
                self.policy.find(name).map_or_else(
                    || Err(format!("no such tool: {name}")),
                    |tool| Ok(tool_descriptor(&tool)),
                )
            }
            _ if op.strip_prefix("tool.").is_some_and(|n| !n.is_empty()) => {
                self.call_tool(op.trim_start_matches("tool."), args).await
            }
            _ if self.policy.find(op).is_some() || refused_reason(op).is_some() => {
                self.call_tool(op, args).await
            }
            _ => Err(format!("unknown eval host op: {op}")),
        }
    }

    /// `read` host op: `{path, offset?, limit?}` → file/URL contents as a
    /// plain string. 1-indexed `offset`/`limit` become a read-tool line
    /// selector (TS `_read_line_selector`); internal URLs and `artifact://`
    /// ids route through the read tool like TS `_read_through_tool`.
    async fn read_helper(&self, args: Value) -> Result<Value, String> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| "read requires a string `path`".to_string())?;
        let offset = args.get("offset").and_then(Value::as_i64);
        let limit = args.get("limit").and_then(Value::as_i64);
        if limit.is_some_and(|l| l <= 0) {
            return Ok(Value::String(String::new()));
        }
        let mut target = path.to_string();
        if offset.is_some() || limit.is_some() {
            let start = offset.unwrap_or(1).max(1);
            target = match limit {
                Some(n) => format!("{target}:{start}-{}", start + n - 1),
                None => format!("{target}:{start}-"),
            };
        }
        let value = self.call_tool("read", json!({"path": target})).await?;
        Ok(Value::String(bridge_text(&value)))
    }

    /// `write` host op: `{path, data | content, encoding?}` → resolved path
    /// string. `encoding: "base64"` decodes `data` (JS helper parity).
    /// `agent://<id>` targets are subagent messages, not files (TS
    /// `AgentHandle.send`), so they go to the agent bridge.
    async fn write_helper(&self, args: Value) -> Result<Value, String> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| "write requires a string `path`".to_string())?;
        if let Some(id) = path.strip_prefix("agent://") {
            let content = args
                .get("content")
                .or_else(|| args.get("data"))
                .and_then(Value::as_str)
                .ok_or_else(|| "agent message requires string `content`".to_string())?;
            return super::eval_agents::send_to_agent(id, content).await;
        }
        let raw = args
            .get("content")
            .or_else(|| args.get("data"))
            .and_then(Value::as_str)
            .ok_or_else(|| "write requires string `content`".to_string())?;
        let content = if args.get("encoding").and_then(Value::as_str) == Some("base64") {
            String::from_utf8(
                base64_decode(raw).map_err(|e| format!("invalid base64 write data: {e}"))?,
            )
            .map_err(|_| "base64 write data is not UTF-8 text".to_string())?
        } else {
            raw.to_string()
        };
        self.call_tool("write", json!({"path": path, "content": content}))
            .await?;
        if path.contains("://") {
            return Ok(Value::String(path.to_string()));
        }
        Ok(Value::String(
            super::util::resolve_to_cwd(path, &self.env.cwd)
                .to_string_lossy()
                .into_owned(),
        ))
    }

    /// `output` host op: read recorded task/agent outputs (`<id>.md` files in
    /// the session artifacts dir) — TS `output()` in `py/prelude.py`, moved
    /// host-side so both preludes share one implementation.
    fn output_helper(&self, args: &Value) -> Result<Value, String> {
        let ids: Vec<String> = match args.get("ids") {
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(items)) => items
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect(),
            _ => args
                .get("id")
                .and_then(Value::as_str)
                .map_or_else(Vec::new, |s| vec![s.to_string()]),
        };
        if ids.is_empty() {
            return Err("output requires at least one id".to_string());
        }
        let format = args.get("format").and_then(Value::as_str).unwrap_or("raw");
        let query = args.get("query").and_then(Value::as_str);
        let offset = args.get("offset").and_then(Value::as_i64);
        let limit = args.get("limit").and_then(Value::as_i64);
        if query.is_some() && (offset.is_some() || limit.is_some()) {
            return Err("output: query cannot be combined with offset/limit".to_string());
        }
        let dir = artifacts_dir()?;
        if !dir.is_dir() {
            return Err(format!("no artifacts directory found: {}", dir.display()));
        }

        let mut results: Vec<Value> = Vec::new();
        let mut not_found: Vec<&str> = Vec::new();
        for id in &ids {
            let path = dir.join(format!("{id}.md"));
            let Ok(raw) = std::fs::read_to_string(&path) else {
                not_found.push(id);
                continue;
            };
            let raw_lines: Vec<&str> = raw.lines().collect();
            let total_lines = raw_lines.len();
            let mut selected = raw.clone();
            let mut range: Option<Value> = None;

            if let Some(query) = query {
                let parsed: Value = serde_json::from_str(&raw)
                    .map_err(|e| format!("output {id} is not valid JSON: {e}"))?;
                let value = apply_query(&parsed, query).unwrap_or(Value::Null);
                selected = match value {
                    Value::Null => "null".to_string(),
                    other => {
                        serde_json::to_string_pretty(&other).unwrap_or_else(|_| other.to_string())
                    }
                };
            } else if offset.is_some() || limit.is_some() {
                let start = offset.unwrap_or(1).max(1);
                let total = total_lines as i64;
                if start > total.max(1) {
                    return Err(format!(
                        "offset {start} is beyond end of output ({total_lines} lines) for {id}"
                    ));
                }
                let effective = limit.map_or(total - start + 1, |l| l.max(0));
                let end = total.min(start + effective - 1).max(0);
                selected = raw_lines
                    .get((start - 1).max(0) as usize..end as usize)
                    .map_or_else(String::new, |ls| ls.join("\n"));
                range = Some(json!({
                    "start_line": start,
                    "end_line": end,
                    "total_lines": total_lines,
                }));
            }

            if format == "stripped" {
                selected = strip_ansi(&selected);
            }

            if format == "json" {
                let mut data = json!({
                    "id": id,
                    "path": path.display().to_string(),
                    "line_count": if query.is_some() { selected.lines().count() } else { total_lines },
                    "char_count": if query.is_some() { selected.chars().count() } else { raw.chars().count() },
                    "content": selected,
                });
                if let Some(range) = range {
                    data["range"] = range;
                }
                if let Some(query) = query {
                    data["query"] = Value::String(query.to_string());
                }
                results.push(data);
            } else {
                results.push(json!({"id": id, "content": selected}));
            }
        }
        if !not_found.is_empty() {
            let mut available: Vec<String> = std::fs::read_dir(&dir)
                .map(|rd| {
                    rd.flatten()
                        .filter_map(|e| {
                            e.path()
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .map(ToOwned::to_owned)
                        })
                        .collect()
                })
                .unwrap_or_default();
            available.sort();
            return Err(format!(
                "output not found: {}; available: {}",
                not_found.join(", "),
                available.join(", ")
            ));
        }
        if results.len() == 1 {
            let single = results.pop().expect("one result");
            if format == "json" {
                return Ok(single);
            }
            return Ok(single
                .get("content")
                .cloned()
                .unwrap_or_else(|| String::new().into()));
        }
        Ok(Value::Array(results))
    }

    /// Dispatch one host-side tool call by name (TS `/v1/tool` →
    /// `callSessionTool` through `getToolForEvalBridge`'s
    /// `ExtensionToolWrapper`): only enabled tools, validate arguments, run the
    /// session's `tool_call` hook (block / input rewrite, revalidated), execute
    /// under the cell's abort signal, run the `tool_result` hook (result
    /// rewrite), and convert to the TS bridge value shape
    /// (`bridgeValueFromToolResult`).
    async fn call_tool(&self, name: &str, args: Value) -> Result<Value, String> {
        if let Some(reason) = refused_reason(name) {
            return Err(reason);
        }
        let Some(tool) = self.policy.find(name) else {
            return Err(format!(
                "unknown eval host op: {name} (not a tool enabled for this session)"
            ));
        };
        let mut params = args;
        if let Some(obj) = params.as_object_mut() {
            // The model-facing intent field is not part of the tool schema.
            obj.remove("i");
        }
        let mut params = validate_tool_arguments(tool.name(), tool.parameters(), &params)
            .map_err(|e| e.to_string())?;
        let signal = &self.ctx.signal;
        let mut call = ToolCall {
            id: format!("{}:{name}", self.ctx.tool_call_id),
            name: name.to_string(),
            arguments: params.clone(),
            thought_signature: None,
        };
        let context = || AgentContext {
            tools: self.policy.tools.clone(),
            ..AgentContext::default()
        };
        if let Some(before) = self.policy.before.as_ref() {
            let hook = before(
                BeforeToolCallContext {
                    assistant_message: synthetic_turn(&call),
                    tool_call: call.clone(),
                    args: params.clone(),
                    context: context(),
                },
                signal.clone(),
            );
            // A failing hook fails closed, like the agent loop's error result.
            let outcome = race_with_abort(hook, signal)
                .await
                .map_err(|e| format!("{e:#}"))?
                .map_err(|e| format!("{e:#}"))?;
            if let Some(outcome) = outcome {
                if outcome.block {
                    return Err(outcome
                        .reason
                        .unwrap_or_else(|| "Tool execution was blocked".to_string()));
                }
                if let Some(revised) = outcome.args {
                    // Hook revisions are untrusted input to the tool schema.
                    params = validate_tool_arguments(tool.name(), tool.parameters(), &revised)
                        .map_err(|e| e.to_string())?;
                    call.arguments = revised;
                }
            }
        }
        let on_update: AgentToolUpdateCallback = match self.ctx.on_update.as_ref() {
            Some(cb) => {
                let cb = Arc::clone(cb);
                Arc::new(move |partial| cb(partial))
            }
            None => Arc::new(|_| {}),
        };
        let executed = race_with_abort(
            Arc::clone(&tool).execute(call.id.clone(), params.clone(), signal.clone(), on_update),
            signal,
        )
        .await
        .and_then(|r| r);
        let (mut result, mut is_error, mut failure) = match executed {
            Ok(result) => {
                let is_error = result.is_error == Some(true);
                (result, is_error, None)
            }
            Err(e) => {
                let message = format!("{e:#}");
                (AgentToolResult::error(message.clone()), true, Some(message))
            }
        };
        if let Some(after) = self.policy.after.as_ref() {
            let hook = after(
                AfterToolCallContext {
                    assistant_message: synthetic_turn(&call),
                    tool_call: call.clone(),
                    args: params,
                    result: result.clone(),
                    is_error,
                    context: context(),
                },
                signal.clone(),
            );
            match race_with_abort(hook, signal).await.and_then(|r| r) {
                Ok(Some(overrides)) => {
                    // Field-by-field merge (TS semantics). Any override turns
                    // an execution failure into the handler's result.
                    failure = None;
                    if let Some(content) = overrides.content {
                        result.content = content;
                    }
                    if let Some(details) = overrides.details {
                        result.details = details;
                    }
                    if let Some(error) = overrides.is_error {
                        is_error = error;
                    }
                    result.is_error = is_error.then_some(true);
                }
                Ok(None) => {}
                Err(e) => return Err(format!("{e:#}")),
            }
        }
        if let Some(message) = failure {
            return Err(message);
        }
        self.emit_status(json!({"op": name}));
        Ok(bridge_value(&result))
    }
}

impl omp_eval::HostBridge for EvalBridge {
    fn call(&self, op: &str, args: Value) -> Result<Value, String> {
        // The kernel invokes this from its spawn-blocking cell loop; drive
        // the async tool graph on the stored runtime handle.
        self.rt.block_on(self.dispatch(op, args))
    }

    fn notify(&self, op: &str, args: Value) {
        // Fire-and-forget status: `log`/`phase`/`progress`/`display`/
        // `tool_define` and unknown ops all travel as one JsStatusEvent.
        let mut event = serde_json::Map::new();
        event.insert("op".to_string(), Value::String(op.to_string()));
        if let Value::Object(map) = args {
            event.extend(map);
        }
        if op == "display" {
            // Raw JSON display: a mime bundle, mirroring how the same value
            // would render as a cell display frame.
            let value = event.remove("value").unwrap_or(Value::Null);
            let bundle = match &value {
                Value::String(s) => json!({"text/plain": s}),
                other => json!({"application/json": other, "text/plain": other.to_string()}),
            };
            let Some(on_update) = self.ctx.on_update.as_ref() else {
                return;
            };
            on_update(AgentToolResult {
                content: Vec::new(),
                details: bundle,
                terminate: None,
                is_error: None,
            });
            return;
        }
        self.emit_status(Value::Object(event));
    }
}

fn tool_descriptor(tool: &Arc<dyn AgentTool>) -> Value {
    json!({
        "name": tool.name(),
        "description": tool.description(),
        "parameters": tool.parameters(),
    })
}

/// TS `bridgeValueFromToolResult` (js/tool-bridge.ts): bare text when the
/// result carries no details or images, otherwise the object form.
fn bridge_value(result: &AgentToolResult) -> Value {
    let mut text = String::new();
    let mut images: Vec<Value> = Vec::new();
    for content in &result.content {
        match content {
            ToolResultContent::Text(t) => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&t.text);
            }
            ToolResultContent::Image(image) => images.push(json!({
                "data": image.data,
                "mimeType": image.mime_type,
            })),
        }
    }
    let details_empty =
        result.details.is_null() || result.details.as_object().is_some_and(|o| o.is_empty());
    if result.is_error != Some(true) && images.is_empty() && details_empty {
        return Value::String(text);
    }
    let mut value = json!({"text": text});
    if !details_empty {
        value["details"] = result.details.clone();
    }
    if !images.is_empty() {
        value["images"] = Value::Array(images);
    }
    if result.is_error == Some(true) {
        value["isError"] = Value::Bool(true);
    }
    value
}

fn bridge_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Object(map) => map
            .get("text")
            .and_then(Value::as_str)
            .map_or_else(|| value.to_string(), ToOwned::to_owned),
        other => other.to_string(),
    }
}

/// Per-session environment overlay the `env` host op writes (TS JS runtime
/// `ctx.env` map). The host process environment is never mutated after
/// startup (`set_var` races concurrent `getenv`/`execve` on worker threads);
/// later eval kernel spawns receive the overlay through `Command::envs`.
static ENV_OVERLAY: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

/// The overlay entries, for kernel spawns (see `eval::kernel_env`).
pub(crate) fn env_overlay() -> Vec<(String, String)> {
    ENV_OVERLAY
        .lock()
        .expect("env overlay lock")
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// `env` host op: `{key?, value?}` → merged environment (no key), the set
/// value (`value` given; recorded in the overlay only), or the overlay/host
/// value / null (TS `env()`: `ctx.env.get(key) ?? Bun.env[key]`).
fn env_helper(args: &Value) -> Result<Value, String> {
    let key = args.get("key").and_then(Value::as_str);
    let value = args.get("value").and_then(Value::as_str);
    let mut overlay = ENV_OVERLAY.lock().expect("env overlay lock");
    Ok(match (key, value) {
        (None, _) => {
            let mut merged: BTreeMap<String, String> = std::env::vars().collect();
            merged.extend(overlay.iter().map(|(k, v)| (k.clone(), v.clone())));
            Value::Object(
                merged
                    .into_iter()
                    .map(|(k, v)| (k, Value::String(v)))
                    .collect(),
            )
        }
        (Some(k), Some(v)) => {
            if k.is_empty() || k.contains(['=', '\0']) || v.contains('\0') {
                return Err(format!("env: invalid variable name or value for {k:?}"));
            }
            overlay.insert(k.to_string(), v.to_string());
            Value::String(v.to_string())
        }
        (Some(k), None) => overlay
            .get(k)
            .cloned()
            .or_else(|| std::env::var(k).ok())
            .map_or(Value::Null, Value::String),
    })
}

/// This session's artifacts dir: `PI_ARTIFACTS_DIR`, else the global session
/// file minus its `.jsonl` suffix (TS `output()` / `artifactsDirectoryFor`).
fn artifacts_dir() -> Result<PathBuf, String> {
    if let Ok(dir) = std::env::var("PI_ARTIFACTS_DIR") {
        if !dir.is_empty() {
            return Ok(PathBuf::from(dir));
        }
    }
    omp_urls::global_session_file()
        .and_then(|f| {
            f.to_str()
                .and_then(|s| s.strip_suffix(".jsonl"))
                .map(PathBuf::from)
        })
        .ok_or_else(|| "no session - output artifacts unavailable".to_string())
}

/// Minimal jq-like dotted query (`.a.b[0]`, TS `_apply_query`): keys and
/// numeric indexes only.
fn apply_query(value: &Value, query: &str) -> Option<Value> {
    let mut current = value.clone();
    let mut chars = query.chars().peekable();
    while chars.peek().is_some() {
        match chars.next() {
            Some('.') => {
                let mut key = String::new();
                while let Some(&c) = chars.peek() {
                    if c == '.' || c == '[' {
                        break;
                    }
                    key.push(c);
                    chars.next();
                }
                current = current.get(key)?.clone();
            }
            Some('[') => {
                let mut num = String::new();
                for c in chars.by_ref() {
                    if c == ']' {
                        break;
                    }
                    num.push(c);
                }
                let index: usize = num.trim().parse().ok()?;
                current = current.get(index)?.clone();
            }
            Some(c) if c.is_whitespace() => continue,
            _ => return None,
        }
    }
    Some(current)
}

/// Strip SGR sequences (`\x1b[…m`, TS `format == "stripped"`); other CSI
/// final bytes are consumed generically so nested styling can't leak through.
fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        if chars.peek() == Some(&'[') {
            chars.next();
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        }
    }
    out
}

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(input.trim())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pa_agent::types::{AfterToolCallResult, BeforeToolCallResult};

    fn env(tag: &str) -> Arc<Env> {
        let dir =
            std::env::temp_dir().join(format!("omp-rs-bridge-test-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Arc::new(Env::new(&dir, false))
    }

    fn bridge(env: &Arc<Env>, policy: BridgePolicy) -> EvalBridge {
        let ctx = OpCtx {
            cwd: env.cwd.clone(),
            signal: AbortSignal::never(),
            on_update: None,
            tool_call_id: "cell".into(),
        };
        EvalBridge::new(Arc::clone(env), ctx, tokio::runtime::Handle::current()).with_policy(policy)
    }

    fn all_tools(env: &Env) -> Vec<Arc<dyn AgentTool>> {
        super::super::builtin_tools(&env.cwd, env.edit_mode)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn blocking_extension_hook_denies_bridged_bash() {
        let env = env("block");
        let marker = env.cwd.join("ran.txt");
        let _ = std::fs::remove_file(&marker);
        let before: BeforeToolCallFn = Arc::new(|ctx: BeforeToolCallContext, _| {
            Box::pin(async move {
                Ok(
                    (ctx.tool_call.name == "bash").then(|| BeforeToolCallResult {
                        block: true,
                        reason: Some("bash denied by policy".into()),
                        ..BeforeToolCallResult::default()
                    }),
                )
            })
        });
        let b = bridge(
            &env,
            BridgePolicy {
                tools: all_tools(&env),
                before: Some(before),
                after: None,
            },
        );
        for op in ["tool.bash", "bash"] {
            let err = b
                .dispatch(op, json!({"command": "touch ran.txt"}))
                .await
                .unwrap_err();
            assert!(err.contains("bash denied by policy"), "{op}: {err}");
        }
        assert!(!marker.exists(), "blocked bash must not run");
        // Unblocked tools still pass through the hook and run.
        std::fs::write(env.cwd.join("ok.txt"), "fine\n").unwrap();
        let v = b.dispatch("read", json!({"path": "ok.txt"})).await.unwrap();
        assert!(v.as_str().unwrap().contains("fine"), "{v}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hook_input_and_result_rewrites_apply_to_bridged_calls() {
        let env = env("rewrite");
        std::fs::write(env.cwd.join("secret.txt"), "TOKEN=abc\n").unwrap();
        std::fs::write(env.cwd.join("public.txt"), "hello\n").unwrap();
        let before: BeforeToolCallFn = Arc::new(|ctx: BeforeToolCallContext, _| {
            Box::pin(async move {
                let wants_secret = ctx.args["path"]
                    .as_str()
                    .is_some_and(|p| p.starts_with("secret.txt"));
                Ok(wants_secret.then(|| BeforeToolCallResult {
                    args: Some(json!({"path": "public.txt"})),
                    ..BeforeToolCallResult::default()
                }))
            })
        });
        let after: AfterToolCallFn = Arc::new(|ctx: AfterToolCallContext, _| {
            Box::pin(async move {
                Ok(
                    (ctx.tool_call.name == "write").then(|| AfterToolCallResult {
                        content: Some(vec![ToolResultContent::text("[redacted]")]),
                        ..AfterToolCallResult::default()
                    }),
                )
            })
        });
        let b = bridge(
            &env,
            BridgePolicy {
                tools: all_tools(&env),
                before: Some(before),
                after: Some(after),
            },
        );
        let v = b
            .dispatch("tool.read", json!({"path": "secret.txt"}))
            .await
            .unwrap();
        let text = bridge_text(&v);
        assert!(text.contains("hello") && !text.contains("TOKEN"), "{text}");
        let v = b
            .dispatch("tool.write", json!({"path": "out.txt", "content": "x"}))
            .await
            .unwrap();
        assert_eq!(bridge_text(&v), "[redacted]");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn tools_not_enabled_for_the_session_are_refused() {
        let env = env("gate");
        std::fs::write(env.cwd.join("a.txt"), "a\n").unwrap();
        let tools: Vec<_> = all_tools(&env)
            .into_iter()
            .filter(|t| t.name() == "read")
            .collect();
        let b = bridge(
            &env,
            BridgePolicy {
                tools,
                before: None,
                after: None,
            },
        );
        for op in ["tool.bash", "bash", "tool.think", "think", "tool.yield"] {
            let err = b
                .dispatch(op, json!({"command": "true"}))
                .await
                .unwrap_err();
            assert!(err.contains("unknown eval host op"), "{op}: {err}");
        }
        assert!(b
            .dispatch("tool.describe", json!({"name": "bash"}))
            .await
            .is_err());
        let listed = b.dispatch("tool.list", json!({})).await.unwrap();
        assert_eq!(listed.as_array().unwrap().len(), 1, "{listed}");
        assert!(b.dispatch("read", json!({"path": "a.txt"})).await.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn eval_checkpoint_and_rewind_are_refused_through_the_bridge() {
        let env = env("refuse");
        let b = bridge(
            &env,
            BridgePolicy {
                tools: all_tools(&env),
                before: None,
                after: None,
            },
        );
        let started = std::time::Instant::now();
        for op in ["eval", "tool.eval", "checkpoint", "tool.rewind"] {
            let err = b
                .dispatch(op, json!({"language": "py", "code": "1"}))
                .await
                .unwrap_err();
            assert!(err.contains("cannot run"), "{op}: {err}");
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        let listed = b.dispatch("tool.list", json!({})).await.unwrap();
        assert!(listed
            .as_array()
            .unwrap()
            .iter()
            .all(|t| !matches!(t["name"].as_str(), Some("eval" | "checkpoint" | "rewind"))));
    }

    #[tokio::test]
    async fn env_set_goes_to_the_session_overlay_not_the_host_environment() {
        let env = env("envop");
        let b = bridge(
            &env,
            BridgePolicy {
                tools: Vec::new(),
                before: None,
                after: None,
            },
        );
        let key = format!("OMP_RS_BRIDGE_ENV_TEST_{}", std::process::id());
        let set = b
            .dispatch("env", json!({"key": key, "value": "overlay"}))
            .await
            .unwrap();
        assert_eq!(set, json!("overlay"));
        assert!(
            std::env::var(&key).is_err(),
            "host environment must not change"
        );
        assert_eq!(
            b.dispatch("env", json!({"key": key})).await.unwrap(),
            json!("overlay")
        );
        let all = b.dispatch("env", json!({})).await.unwrap();
        assert_eq!(all[&key], json!("overlay"));
        assert!(env_overlay().contains(&(key.clone(), "overlay".to_string())));
        // Unset overlay keys fall back to the host environment.
        let path = b.dispatch("env", json!({"key": "PATH"})).await.unwrap();
        assert_eq!(path.as_str(), std::env::var("PATH").ok().as_deref());
        for bad in ["", "A=B", "A\0B"] {
            assert!(
                b.dispatch("env", json!({"key": bad, "value": "v"}))
                    .await
                    .is_err(),
                "{bad:?}"
            );
        }
    }
}
