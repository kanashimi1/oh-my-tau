//! Host-side bridge for eval cells: answers the runners' host ops
//! (`{"type":"host","op":...}` NDJSON frames) by dispatching to this
//! process's tools, so cell code can call `read`/`write`/`bash`/…, message
//! subagents and query the runtime. TS reference: `eval/py/tool-bridge.ts`
//! (`/v1/tool` loopback, `callSessionTool`, `bridgeValueFromToolResult`),
//! `eval/js/tool-bridge.ts`; agent-family ops (`__agent__`, `__wait__`, …)
//! are delegated to [`super::eval_agents`] unchanged.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback, ToolResultContent};
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
}

/// Tools the bridge may dispatch to. Set once per process by the runtime
/// (the session's real tool list, so cell-side `read`/`bash`/… behave exactly
/// like model-side calls, including hidden tools); before that, the bridge
/// falls back to [`super::builtin_tools`] plus [`super::hidden_tool_by_name].
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

fn find_tool(env: &Env, name: &str) -> Option<Arc<dyn AgentTool>> {
    session_tools(env)
        .into_iter()
        .find(|t| t.name() == name)
        .or_else(|| super::hidden_tool_by_name(name))
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
        Self { env, ctx, rt }
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
            let name = args
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            return find_tool(&self.env, name).map_or_else(
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
            "env" => Ok(env_helper(&args)),
            "tool.list" => Ok(Value::Array(
                session_tools(&self.env).iter().map(tool_descriptor).collect(),
            )),
            "tool.describe" => {
                let name = args
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "tool.describe requires a string `name`".to_string())?;
                find_tool(&self.env, name).map_or_else(
                    || Err(format!("no such tool: {name}")),
                    |tool| Ok(tool_descriptor(&tool)),
                )
            }
            _ if op.strip_prefix("tool.").is_some_and(|n| !n.is_empty()) => {
                self.call_tool(op.trim_start_matches("tool."), args).await
            }
            _ => match find_tool(&self.env, op) {
                Some(_) => self.call_tool(op, args).await,
                None => Err(format!("unknown eval host op: {op}")),
            },
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
                    other => serde_json::to_string_pretty(&other)
                        .unwrap_or_else(|_| other.to_string()),
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
    /// `callSessionTool`): validate arguments, execute under the cell's
    /// abort signal, forward partial updates, and convert the tool result to
    /// the TS bridge value shape (`bridgeValueFromToolResult`).
    async fn call_tool(&self, name: &str, args: Value) -> Result<Value, String> {
        let Some(tool) = find_tool(&self.env, name) else {
            return Err(format!("unknown eval host op: {name}"));
        };
        let mut params = args;
        if let Some(obj) = params.as_object_mut() {
            // The model-facing intent field is not part of the tool schema.
            obj.remove("i");
        }
        let params = validate_tool_arguments(tool.name(), tool.parameters(), &params)
            .map_err(|e| e.to_string())?;
        let child_id = format!("{}:{name}", self.ctx.tool_call_id);
        let on_update: AgentToolUpdateCallback = match self.ctx.on_update.as_ref() {
            Some(cb) => {
                let cb = Arc::clone(cb);
                Arc::new(move |partial| cb(partial))
            }
            None => Arc::new(|_| {}),
        };
        let result = tool
            .execute(child_id, params, self.ctx.signal.clone(), on_update)
            .await
            .map_err(|e| e.to_string())?;
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
    let details_empty = result.details.is_null()
        || result.details.as_object().is_some_and(|o| o.is_empty());
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

/// `env` host op: `{key?, value?}` → whole environment (no key), the set
/// value (`value` given), or the current value / null (TS `env()`).
fn env_helper(args: &Value) -> Value {
    let key = args.get("key").and_then(Value::as_str);
    let value = args.get("value").and_then(Value::as_str);
    match (key, value) {
        (None, _) => Value::Object(
            std::env::vars()
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .map(|(k, v)| (k, Value::String(v)))
                .collect(),
        ),
        (Some(k), Some(v)) => {
            // Cell-time set: children spawned by later host ops inherit it.
            std::env::set_var(k, v);
            Value::String(v.to_string())
        }
        (Some(k), None) => std::env::var(k).map_or(Value::Null, Value::String),
    }
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
