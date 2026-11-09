//! `omt hub mcp` — a stdio MCP server (protocol revision 2025-06-18) that
//! exposes the a2a hub as MCP tools. Wire format mirrors the client side in
//! `crates/omp-mcp` (transport.rs/client.rs): newline-delimited JSON-RPC 2.0
//! over stdio, `initialize` → `notifications/initialized` → `tools/list` →
//! `tools/call`. Implemented with plain `serde_json`/tokio like the hub itself
//! (no extra server-framework dependency).
//!
//! Endpoint identity (who `send`/`answer` are from, whose mail `inbox` and
//! `status` read, who `claim`/`release` belong to) resolves in this order:
//! `--agent <name>`, then `OMP_HUB_AGENT`, then `mcp:<client name from
//! initialize>`.
//!
//! Large bodies: `payload_path` is read by this server; bodies over 8 KiB go
//! as `payload_ref {path, sha256, bytes}` (hash computed here), small bodies
//! inline. Inline `payload` over 8 KiB is refused (use `payload_path`).

use parking_lot::Mutex;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use omp_hub::{Envelope, HubClient, HubDir, PayloadRef};
use serde_json::{json, Value};

/// MCP protocol revision this server implements (`SPEC.md` item 1).
pub(crate) const PROTOCOL_VERSION: &str = "2025-06-18";
/// Bodies larger than this never go inline (`SPEC.md`).
pub(crate) const INLINE_LIMIT: usize = 8 * 1024;

/// Dispatch `omt hub mcp [flags]`; returns the process exit code.
///
/// # Errors
/// Flag parse or startup failures (printed by `main`).
#[allow(clippy::disallowed_methods)] // entry point: dedicated runtime on its own thread
pub fn run(args: &[String]) -> Result<i32> {
    let (flags, rest) = super::hub::parse_flags(args);
    let _ = rest;
    let flags = std::sync::Arc::new(flags);
    let agent = super::hub::flag(&flags, "agent")
        .map(ToOwned::to_owned)
        .or_else(|| {
            std::env::var("OMP_HUB_AGENT")
                .ok()
                .filter(|v| !v.is_empty())
        });
    let hub = match super::hub::flag(&flags, "dir") {
        Some(dir) => HubDir::explicit(
            PathBuf::from(dir),
            super::hub::flag(&flags, "sock").map(PathBuf::from),
        ),
        None => HubDir::resolve(),
    };
    let auto_start = std::env::var_os("OMP_HUB_NO_AUTOSTART").is_none();
    eprintln!(
        "omt hub mcp: stdio MCP server ({PROTOCOL_VERSION}) dir={} agent={}",
        hub.dir.display(),
        agent
            .as_deref()
            .unwrap_or("mcp:<client name from initialize>"),
    );
    // `subcmd::run` executes on the CLI's own runtime thread; drive ours on a
    // dedicated current-thread runtime (the `subcmd::hub::cmd_serve` pattern).
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("create tokio runtime")
            .and_then(|rt| rt.block_on(serve_stdio(hub, agent, auto_start)))
    })
    .join()
    .unwrap_or_else(|e| anyhow::bail!("hub mcp thread panicked: {e:?}"))
}

/// Server state shared across requests.
struct McpState {
    hub_dir: HubDir,
    auto_start: bool,
    /// `--agent` / `OMP_HUB_AGENT`, when set (wins over the initialize name).
    agent_override: Option<String>,
    /// `clientInfo.name` from `initialize`.
    client_name: Mutex<Option<String>>,
    /// Hub connection, created lazily on the first hub-touching tool call.
    client: tokio::sync::OnceCell<HubClient>,
}

impl McpState {
    fn new(hub_dir: HubDir, agent_override: Option<String>, auto_start: bool) -> Arc<Self> {
        Arc::new(Self {
            hub_dir,
            auto_start,
            agent_override,
            client_name: Mutex::new(None),
            client: tokio::sync::OnceCell::new(),
        })
    }

    /// Resolved endpoint identity (see module doc).
    fn identity(&self) -> String {
        if let Some(a) = &self.agent_override {
            return a.clone();
        }
        let name = self
            .client_name
            .lock()
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        format!("mcp:{name}")
    }

    async fn hub(&self) -> Result<&HubClient> {
        self.client
            .get_or_try_init(|| HubClient::connect_with(self.hub_dir.clone(), self.auto_start))
            .await
    }
}

/// Run the stdio protocol loop until stdin closes or an I/O error.
async fn serve_stdio(hub: HubDir, agent: Option<String>, auto_start: bool) -> Result<i32> {
    let state = McpState::new(hub, agent, auto_start);
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    let mut line = String::new();
    loop {
        line.clear();
        match input.read_line(&mut line) {
            Ok(0) => return Ok(0), // EOF: client gone, exit quietly.
            Ok(_) => {}
            Err(e) => anyhow::bail!("hub mcp: read stdin: {e}"),
        }
        if line.trim().is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            write_frame(
                &mut output,
                &json!({"jsonrpc": "2.0", "id": Value::Null, "error": {
                    "code": -32700, "message": "parse error"} }),
            )?;
            continue;
        };
        if let Some(response) = handle_message(&state, msg).await {
            write_frame(&mut output, &response)?;
        }
    }
}

fn write_frame(out: &mut impl Write, v: &Value) -> Result<()> {
    let mut s = serde_json::to_string(v)?;
    s.push('\n');
    out.write_all(s.as_bytes())?;
    out.flush()?;
    Ok(())
}

/// Handle one JSON-RPC message; `Some(response)` for requests, `None` for
/// notifications.
async fn handle_message(state: &Arc<McpState>, msg: Value) -> Option<Value> {
    let id = msg.get("id").cloned();
    let method = msg.get("method").and_then(Value::as_str);
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let Some(method) = method else {
        return None; // A response or stray frame: nothing to do.
    };
    let Some(id) = id else {
        // Notification (`notifications/initialized`, `notifications/cancelled`, ...).
        if method == "initialize" {
            remember_client_name(state, &params);
        }
        return None;
    };
    let result = match method {
        "initialize" => {
            remember_client_name(state, &params);
            Ok(initialize_result(state))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tool_definitions() })),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match handle_tool_call(state, name, &args).await {
                Ok(v) => Ok(tool_result(&v, false)),
                Err(e) => Ok(tool_result(&json!({"error": e}), true)),
            }
        }
        other => Err(json!({
            "code": -32601,
            "message": format!("method not found: {other}"),
        })),
    };
    Some(match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err(error) => json!({"jsonrpc": "2.0", "id": id, "error": error}),
    })
}

fn remember_client_name(state: &Arc<McpState>, params: &Value) {
    let name = params
        .get("clientInfo")
        .and_then(|c| c.get("name"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    if let Some(name) = name {
        *state.client_name.lock() = Some(name);
    }
}

/// `initialize` result: advertise only the tools capability, report the
/// negotiated revision (`SPEC.md`: 2025-06-18) and this server identity.
fn initialize_result(state: &Arc<McpState>) -> Value {
    let client_name = state
        .client_name
        .lock()
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": { "tools": { "listChanged": false } },
        "serverInfo": { "name": "omt-hub", "version": env!("CARGO_PKG_VERSION") },
        "instructions": format!(
            "omt a2a hub endpoint. Identity: `{}`. Unread mail via the inbox tool; \
             large bodies via payload_path; claims carry fencing tokens.",
            state.identity_or_client(&client_name),
        ),
    })
}

impl McpState {
    /// Identity for the `instructions` field (no mutation: pre-initialize).
    fn identity_or_client(&self, client_name: &str) -> String {
        if let Some(a) = &self.agent_override {
            return a.clone();
        }
        format!("mcp:{client_name}")
    }
}

/// Build a `tools/call` result: one text block (compact JSON) plus
/// `structuredContent` (2025-06-18).
fn tool_result(v: &Value, is_error: bool) -> Value {
    let mut r = json!({
        "content": [{ "type": "text", "text": serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string()) }],
        "structuredContent": v,
    });
    if is_error {
        r["isError"] = Value::Bool(true);
    }
    r
}

// --------------------------------------------------------------------- tools

fn tool_definitions() -> Value {
    let obj = |properties: Value, required: &[&str]| json!({"type": "object", "properties": properties, "required": required});
    json!([
        {
            "name": "send",
            "description": "Send one envelope (idempotent on message_id). `to` is a plain agent id or herdr pane id (w4:pXX). Replies are matched via the inbox tool.",
            "inputSchema": obj(json!({
                "to": { "type": "string", "description": "recipient address" },
                "kind": { "type": "string", "description": "envelope kind, default agent.message" },
                "payload": { "type": "string", "description": "inline body, <= 8 KiB" },
                "payload_path": { "type": "string", "description": "body file; > 8 KiB goes by payload_ref (sha256 computed by the server)" },
                "expected": { "type": "string", "description": "free-text reply expectation; non-empty means reply-expected" },
                "reply_schema": { "type": "object", "description": "JSON Schema the answer is validated against" },
                "deadline_s": { "type": "number", "description": "reply deadline in seconds from now" },
                "correlation_id": { "type": "string", "description": "thread id" },
                "priority": { "type": "integer", "enum": [0, 1, 2], "description": "0 urgent, 1 normal, 2 bulk" },
                "message_id": { "type": "string", "description": "uuid v7; generated when omitted" }
            }), &["to"]),
        },
        {
            "name": "inbox",
            "description": "List this endpoint's envelopes with seq > since_seq, urgent first (at-least-once; dedup by message_id). Default re-lists everything; pass the last seen seq for deltas. mark_read acknowledges them read.",
            "inputSchema": obj(json!({
                "since_seq": { "type": "number", "description": "only envelopes with seq > since_seq (default 0 = all)" },
                "limit": { "type": "number", "description": "max envelopes (default 50)" },
                "mark_read": { "type": "boolean", "description": "acknowledge every returned envelope read (default false)" }
            }), &[]),
        },
        {
            "name": "ack",
            "description": "Acknowledge one envelope delivered or read.",
            "inputSchema": obj(json!({
                "message_id": { "type": "string" },
                "state": { "type": "string", "enum": ["delivered", "read"] }
            }), &["message_id", "state"]),
        },
        {
            "name": "answer",
            "description": "Answer a reply-expected envelope (causation_id = the message being answered). The hub validates the payload against its reply_schema and routes the answer to the original sender.",
            "inputSchema": obj(json!({
                "causation_id": { "type": "string", "description": "message_id of the question" },
                "payload": { "type": "string", "description": "inline answer body, <= 8 KiB" },
                "payload_path": { "type": "string", "description": "answer body file" }
            }), &["causation_id"]),
        },
        {
            "name": "status",
            "description": "Unread envelopes, pending replies and active claims for this endpoint; or one envelope by message_id.",
            "inputSchema": obj(json!({
                "message_id": { "type": "string", "description": "when set, show just this envelope" }
            }), &[]),
        },
        {
            "name": "claim",
            "description": "Claim a resource with a ttl lease. Returns granted, a fencing token and the expiry; a conflict returns the current holder instead.",
            "inputSchema": obj(json!({
                "resource": { "type": "string" },
                "ttl_s": { "type": "number", "description": "lease seconds" },
                "intent": { "type": "string", "description": "why the claim is taken" }
            }), &["resource", "ttl_s"]),
        },
        {
            "name": "release",
            "description": "Release a claim. Requires the fencing token returned by claim; a wrong token is refused.",
            "inputSchema": obj(json!({
                "resource": { "type": "string" },
                "fencing_token": { "type": "number" }
            }), &["resource", "fencing_token"]),
        },
        {
            "name": "check",
            "description": "Who holds a resource right now (owner, fencing token, expiry) or free.",
            "inputSchema": obj(json!({
                "resource": { "type": "string" }
            }), &["resource"]),
        },
        {
            "name": "claims",
            "description": "List active claims, optionally filtered by resource prefix.",
            "inputSchema": obj(json!({
                "prefix": { "type": "string", "description": "resource name prefix filter" }
            }), &[]),
        },
    ])
}

/// Execute one tool; `Err(message)` becomes an isError tool result.
async fn handle_tool_call(
    state: &Arc<McpState>,
    name: &str,
    args: &Value,
) -> std::result::Result<Value, String> {
    match name {
        "send" => tool_send(state, args).await,
        "inbox" => tool_inbox(state, args).await,
        "ack" => tool_ack(state, args).await,
        "answer" => tool_answer(state, args).await,
        "status" => tool_status(state, args).await,
        "claim" => tool_claim(state, args).await,
        "release" => tool_release(state, args).await,
        "check" => tool_check(state, args).await,
        "claims" => tool_claims(state, args).await,
        other => Err(format!("unknown tool: {other}")),
    }
}

// --------------------------------------------------------------- arg helpers

fn arg_str<'v>(
    args: &'v Value,
    key: &str,
    required: bool,
) -> std::result::Result<Option<&'v str>, String> {
    match args.get(key) {
        None | Some(Value::Null) if !required => Ok(None),
        None | Some(Value::Null) => Err(format!("missing required argument: {key}")),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(other) => Err(format!("argument {key} must be a string, got {other}")),
    }
}

fn arg_u64(args: &Value, key: &str, required: bool) -> std::result::Result<Option<u64>, String> {
    match args.get(key) {
        None | Some(Value::Null) if !required => Ok(None),
        None | Some(Value::Null) => Err(format!("missing required argument: {key}")),
        Some(Value::Number(n)) => n
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("argument {key} must be a non-negative integer")),
        Some(other) => Err(format!("argument {key} must be a number, got {other}")),
    }
}

fn arg_bool(args: &Value, key: &str) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// Resolve the body: `payload` inline or `payload_path` (hash computed here).
/// Bodies over [`INLINE_LIMIT`] go by `payload_ref`; both-set is an error.
fn resolve_payload(
    args: &Value,
) -> std::result::Result<(Option<String>, Option<PayloadRef>), String> {
    let inline = arg_str(args, "payload", false)?;
    let path = arg_str(args, "payload_path", false)?;
    match (inline, path) {
        (Some(_), Some(_)) => Err("pass either payload or payload_path, not both".to_string()),
        (Some(p), None) => {
            if p.len() > INLINE_LIMIT {
                return Err(format!(
                    "inline payload exceeds {INLINE_LIMIT} bytes; use payload_path"
                ));
            }
            Ok((Some(p.to_string()), None))
        }
        (None, Some(p)) => {
            let raw = std::fs::read(p).map_err(|e| format!("read payload_path {p}: {e}"))?;
            if raw.len() > INLINE_LIMIT {
                let canonical = std::fs::canonicalize(p).unwrap_or_else(|_| PathBuf::from(p));
                Ok((
                    None,
                    Some(PayloadRef {
                        path: canonical.display().to_string(),
                        sha256: sha256_hex(&raw),
                        bytes: raw.len() as u64,
                    }),
                ))
            } else {
                let text = String::from_utf8(raw)
                    .map_err(|_| format!("payload_path {p}: not UTF-8 text"))?;
                Ok((Some(text), None))
            }
        }
        (None, None) => Ok((None, None)),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

// ------------------------------------------------------------------ tools

async fn tool_send(state: &Arc<McpState>, args: &Value) -> std::result::Result<Value, String> {
    let to = arg_str(args, "to", true)?.unwrap_or_default();
    let (payload, payload_ref) = resolve_payload(args)?;
    let reply_schema = match args.get("reply_schema") {
        None | Some(Value::Null) => None,
        Some(v @ Value::Object(_)) => Some(v.clone()),
        Some(_) => return Err("reply_schema must be a JSON object".to_string()),
    };
    let priority = arg_u64(args, "priority", false)?
        .map(|p| u8::try_from(p).map_err(|_| "priority must be 0, 1 or 2".to_string()))
        .transpose()?;
    if let Some(p) = priority {
        if p > 2 {
            return Err("priority must be 0, 1 or 2".to_string());
        }
    }
    let deadline = arg_u64(args, "deadline_s", false)?
        .map(|s| omp_hub::now_ms().saturating_add(s.saturating_mul(1000)));
    let env = Envelope {
        message_id: arg_str(args, "message_id", false)?
            .unwrap_or_default()
            .to_string(),
        correlation_id: arg_str(args, "correlation_id", false)?.map(ToOwned::to_owned),
        sender: state.identity(),
        recipient: to.to_string(),
        kind: arg_str(args, "kind", false)?
            .unwrap_or(omp_hub::Kind::AGENT_MESSAGE)
            .to_string(),
        priority: omp_hub::Priority(priority.unwrap_or(1)),
        expected: arg_str(args, "expected", false)?.map(ToOwned::to_owned),
        reply_schema,
        deadline,
        payload,
        payload_ref,
        ..Envelope::default()
    };
    let receipt = state
        .hub()
        .await
        .map_err(|e| e.to_string())?
        .send(env)
        .await
        .map_err(|e| e.to_string())?;
    serde_json::to_value(receipt).map_err(|e| e.to_string())
}

async fn tool_inbox(state: &Arc<McpState>, args: &Value) -> std::result::Result<Value, String> {
    let since = arg_u64(args, "since_seq", false)?.unwrap_or(0);
    let limit = arg_u64(args, "limit", false)?.unwrap_or(50);
    if limit == 0 {
        return Err("limit must be >= 1".to_string());
    }
    let me = state.identity();
    let hub = state.hub().await.map_err(|e| e.to_string())?;
    let envelopes = hub
        .inbox(&me, Some(since), Some(limit))
        .await
        .map_err(|e| e.to_string())?;
    let mut mark_read_errors: Vec<Value> = Vec::new();
    if arg_bool(args, "mark_read") {
        for env in &envelopes {
            if let Err(e) = hub.ack(&env.message_id, omp_hub::State::READ).await {
                mark_read_errors
                    .push(json!({"message_id": env.message_id, "error": e.to_string()}));
            }
        }
    }
    let envelopes: Vec<Value> = envelopes
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "recipient": me,
        "envelopes": envelopes,
        "mark_read_errors": mark_read_errors,
    }))
}

async fn tool_ack(state: &Arc<McpState>, args: &Value) -> std::result::Result<Value, String> {
    let id = arg_str(args, "message_id", true)?.unwrap_or_default();
    let s = arg_str(args, "state", true)?.unwrap_or_default();
    if !matches!(s, omp_hub::State::DELIVERED | omp_hub::State::READ) {
        return Err("state must be delivered|read".to_string());
    }
    state
        .hub()
        .await
        .map_err(|e| e.to_string())?
        .ack(id, s)
        .await
        .map_err(|e| e.to_string())?;
    Ok(json!({"message_id": id, "state": s, "ok": true}))
}

async fn tool_answer(state: &Arc<McpState>, args: &Value) -> std::result::Result<Value, String> {
    let causation_id = arg_str(args, "causation_id", true)?.unwrap_or_default();
    let (payload, payload_ref) = resolve_payload(args)?;
    let answer = Envelope {
        message_id: arg_str(args, "message_id", false)?
            .unwrap_or_default()
            .to_string(),
        sender: state.identity(),
        kind: omp_hub::KIND_ANSWER.to_string(),
        priority: omp_hub::Priority::URGENT,
        payload,
        payload_ref,
        ..Envelope::default()
    };
    state
        .hub()
        .await
        .map_err(|e| e.to_string())?
        .answer(causation_id, answer)
        .await
        .map_err(|e| e.to_string())
}

async fn tool_status(state: &Arc<McpState>, args: &Value) -> std::result::Result<Value, String> {
    let hub = state.hub().await.map_err(|e| e.to_string())?;
    if let Some(id) = arg_str(args, "message_id", false)? {
        return hub.status_message(id).await.map_err(|e| e.to_string());
    }
    let me = state.identity();
    hub.status(Some(&me)).await.map_err(|e| e.to_string())
}

async fn tool_claim(state: &Arc<McpState>, args: &Value) -> std::result::Result<Value, String> {
    let resource = arg_str(args, "resource", true)?.unwrap_or_default();
    let ttl = arg_u64(args, "ttl_s", true)?
        .ok_or_else(|| "missing required argument: ttl_s".to_string())?;
    if ttl == 0 {
        return Err("ttl_s must be >= 1".to_string());
    }
    let me = state.identity();
    let outcome = state
        .hub()
        .await
        .map_err(|e| e.to_string())?
        .claim(resource, &me, ttl, arg_str(args, "intent", false)?)
        .await
        .map_err(|e| e.to_string())?;
    serde_json::to_value(outcome).map_err(|e| e.to_string())
}

async fn tool_release(state: &Arc<McpState>, args: &Value) -> std::result::Result<Value, String> {
    let resource = arg_str(args, "resource", true)?.unwrap_or_default();
    let token = arg_u64(args, "fencing_token", true)?
        .ok_or_else(|| "missing required argument: fencing_token".to_string())?;
    let me = state.identity();
    state
        .hub()
        .await
        .map_err(|e| e.to_string())?
        .release(resource, &me, token)
        .await
        .map_err(|e| e.to_string())?;
    Ok(json!({"resource": resource, "released": true}))
}

async fn tool_check(state: &Arc<McpState>, args: &Value) -> std::result::Result<Value, String> {
    let resource = arg_str(args, "resource", true)?.unwrap_or_default();
    state
        .hub()
        .await
        .map_err(|e| e.to_string())?
        .check(resource)
        .await
        .map_err(|e| e.to_string())
}

async fn tool_claims(state: &Arc<McpState>, args: &Value) -> std::result::Result<Value, String> {
    let prefix = arg_str(args, "prefix", false)?;
    state
        .hub()
        .await
        .map_err(|e| e.to_string())?
        .claims(prefix, None)
        .await
        .map_err(|e| e.to_string())
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod tests {
    use super::*;

    fn state(agent: Option<&str>) -> Arc<McpState> {
        McpState::new(
            HubDir::explicit(PathBuf::from("/nonexistent-hub-dir"), None),
            agent.map(ToOwned::to_owned),
            false,
        )
    }

    fn call(state: &Arc<McpState>, name: &str, args: Value) -> std::result::Result<Value, String> {
        // Synchronous wrapper: these tests never reach the hub (validation
        // errors fire first), so no runtime is needed.
        let state = state.clone();
        let name = name.to_string();
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(handle_tool_call(&state, &name, &args))
        })
        .join()
        .unwrap()
    }

    #[test]
    fn tools_list_has_all_nine_with_schemas() {
        let tools = tool_definitions();
        let names: Vec<&str> = tools
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            ["send", "inbox", "ack", "answer", "status", "claim", "release", "check", "claims"]
        );
        for t in tools.as_array().unwrap() {
            assert!(t["description"].as_str().unwrap().len() < 300);
            assert_eq!(t["inputSchema"]["type"], "object");
        }
    }

    #[test]
    fn initialize_negotiates_2025_06_18() {
        let st = state(None);
        let id = json!(1);
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "tester", "version": "0" },
        }});
        let out = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(handle_message(&st, msg))
            .unwrap();
        assert_eq!(out["id"], id);
        assert_eq!(out["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(out["result"]["serverInfo"]["name"], "omt-hub");
        assert!(out["result"]["capabilities"]["tools"].is_object());
        // Client name remembered -> default identity mcp:tester.
        assert_eq!(st.identity(), "mcp:tester");
    }

    #[test]
    fn identity_precedence_flag_env_initialize() {
        // --agent wins over remembered client name.
        let st = state(Some("flag-agent"));
        *st.client_name.lock() = Some("claude".to_string());
        assert_eq!(st.identity(), "flag-agent");
        // No flag: initialize name is used.
        let st = state(None);
        *st.client_name.lock() = Some("claude".to_string());
        assert_eq!(st.identity(), "mcp:claude");
        // Pre-initialize fallback.
        let st = state(None);
        assert_eq!(st.identity(), "mcp:unknown");
    }

    #[test]
    fn notifications_get_no_response() {
        let st = state(None);
        let msg = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        let out = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(handle_message(&st, msg));
        assert!(out.is_none());
    }

    #[test]
    fn unknown_method_is_jsonrpc_error() {
        let st = state(None);
        let msg = json!({"jsonrpc": "2.0", "id": 7, "method": "resources/list"});
        let out = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(handle_message(&st, msg))
            .unwrap();
        assert_eq!(out["error"]["code"], -32601);
        assert_eq!(out["id"], 7);
    }

    #[test]
    fn stray_frames_get_no_response() {
        let st = state(None);
        // A response (has id, no method) and a bare value: both ignored.
        let out = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(handle_message(
                &st,
                json!({"jsonrpc": "2.0", "id": 3, "result": {}}),
            ));
        assert!(out.is_none());
        let out = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(handle_message(&st, json!(42)));
        assert!(out.is_none());
    }

    #[test]
    fn send_requires_to_and_validates_priority() {
        let st = state(Some("a"));
        let err = call(&st, "send", json!({})).unwrap_err();
        assert!(err.contains("to"), "{err}");
        let err = call(&st, "send", json!({"to": "b", "priority": 5})).unwrap_err();
        assert!(err.contains("priority"), "{err}");
        let err = call(&st, "send", json!({"to": "b", "priority": -1})).unwrap_err();
        assert!(err.contains("priority"), "{err}");
    }

    #[test]
    fn send_rejects_oversize_inline_payload() {
        let st = state(Some("a"));
        let big = "x".repeat(INLINE_LIMIT + 1);
        let err = call(&st, "send", json!({"to": "b", "payload": big})).unwrap_err();
        assert!(err.contains("payload_path"), "{err}");
    }

    #[test]
    fn send_rejects_payload_and_path_together() {
        let st = state(Some("a"));
        let err = call(
            &st,
            "send",
            json!({"to": "b", "payload": "x", "payload_path": "/tmp/nope"}),
        )
        .unwrap_err();
        assert!(err.contains("either"), "{err}");
    }

    #[test]
    fn send_rejects_non_object_reply_schema() {
        let st = state(Some("a"));
        let err = call(&st, "send", json!({"to": "b", "reply_schema": "nope"})).unwrap_err();
        assert!(err.contains("reply_schema"), "{err}");
    }

    #[test]
    fn ack_validates_state() {
        let st = state(Some("a"));
        let err = call(&st, "ack", json!({"message_id": "m", "state": "answered"})).unwrap_err();
        assert!(err.contains("delivered|read"), "{err}");
        let err = call(&st, "ack", json!({"state": "read"})).unwrap_err();
        assert!(err.contains("message_id"), "{err}");
    }

    #[test]
    fn answer_requires_causation_id() {
        let st = state(Some("a"));
        let err = call(&st, "answer", json!({"payload": "x"})).unwrap_err();
        assert!(err.contains("causation_id"), "{err}");
    }

    #[test]
    fn claim_validates_ttl() {
        let st = state(Some("a"));
        let err = call(&st, "claim", json!({"resource": "r"})).unwrap_err();
        assert!(err.contains("ttl_s"), "{err}");
        let err = call(&st, "claim", json!({"resource": "r", "ttl_s": 0})).unwrap_err();
        assert!(err.contains("ttl_s"), "{err}");
    }

    #[test]
    fn release_requires_fencing_token() {
        let st = state(Some("a"));
        let err = call(&st, "release", json!({"resource": "r"})).unwrap_err();
        assert!(err.contains("fencing_token"), "{err}");
    }

    #[test]
    fn inbox_rejects_zero_limit() {
        let st = state(Some("a"));
        let err = call(&st, "inbox", json!({"limit": 0})).unwrap_err();
        assert!(err.contains("limit"), "{err}");
    }

    #[test]
    fn payload_path_small_file_inlines_large_refs() {
        let dir = tempfile::tempdir().unwrap();
        let small = dir.path().join("small.txt");
        std::fs::write(&small, "hello").unwrap();
        let (payload, refr) =
            resolve_payload(&json!({"payload_path": small.to_str().unwrap()})).unwrap();
        assert_eq!(payload.as_deref(), Some("hello"));
        assert!(refr.is_none());

        let big = dir.path().join("big.txt");
        std::fs::write(&big, "y".repeat(INLINE_LIMIT + 1)).unwrap();
        let (payload, refr) =
            resolve_payload(&json!({"payload_path": big.to_str().unwrap()})).unwrap();
        assert!(payload.is_none());
        let refr = refr.unwrap();
        assert_eq!(refr.bytes, (INLINE_LIMIT + 1) as u64);
        assert_eq!(refr.sha256.len(), 64);
        assert!(refr.path.ends_with("big.txt"));
        // sha256 of "y" * (8KiB+1) sanity: stable, not zero.
        assert_ne!(refr.sha256, sha256_hex(b""));
    }

    #[test]
    fn tool_result_shape() {
        let r = tool_result(&json!({"ok": true}), false);
        assert_eq!(r["content"][0]["type"], "text");
        assert!(r.get("isError").is_none());
        assert_eq!(r["structuredContent"]["ok"], true);
        let r = tool_result(&json!({"error": "x"}), true);
        assert_eq!(r["isError"], true);
    }

    #[test]
    fn parse_flags_used_from_hub() {
        // The dispatcher parses with subcmd::hub's flag parser; make sure the
        // re-exports see `--agent x` / boolean-flag forms.
        let (flags, rest) = super::super::hub::parse_flags(&[
            "mcp".to_string(),
            "--agent".to_string(),
            "alice".to_string(),
            "--json".to_string(),
        ]);
        assert_eq!(rest, ["mcp"]);
        let flags = std::sync::Arc::new(flags);
        assert_eq!(super::super::hub::flag(&flags, "agent"), Some("alice"));
        // Boolean flags parse to an empty value; presence is the check.
        assert!(flags.contains_key("json"));
    }
}
