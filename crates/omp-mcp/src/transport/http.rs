//! Streamable-HTTP transport: JSON-RPC 2.0 over HTTP POST with SSE
//! responses, a port of `transports/http.ts` (also serves legacy `sse`
//! configs through the same POST+SSE surface, which is what omp's SSE
//! transport degrades to for requests).
//!
//! Carries the negotiated `MCP-Protocol-Version` header after `initialize`
//! and the server's `Mcp-Session-Id` on subsequent requests, sends
//! `Accept: application/json, text/event-stream`, parses both plain JSON and
//! `text/event-stream` bodies, and issues the spec's DELETE on close.

use std::sync::Arc;

use futures::StreamExt;
use parking_lot::Mutex;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::config::McpServerConfig;
use crate::errors::{
    may_have_run, Failure, JsonRpcError, McpError, McpTransportError, TransportKind,
};
use crate::transport::{McpTransport, PendingRequests, TransportFut};

/// One SSE event.
#[derive(Debug, Default)]
struct SseEvent {
    data: String,
}

/// Read `data:` lines from an SSE byte stream into JSON values, following
/// the standard `readSseEvents` shape (blank line terminates an event).
async fn read_sse_events(body: reqwest::Response) -> Vec<Result<Value, String>> {
    let mut events = Vec::new();
    let mut stream = body.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else { break };
        buf.extend_from_slice(&chunk);
        while let Some(pos) = find_newline(&buf) {
            let line_bytes = buf.drain(..=pos).collect::<Vec<_>>();
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches(['\r', '\n']);
            if let Some(data) = line.strip_prefix("data:") {
                let event = SseEvent {
                    data: data.trim_start().to_string(),
                };
                if let Ok(value) = serde_json::from_str::<Value>(&event.data) {
                    events.push(Ok(value));
                } else {
                    events.push(Err(format!("malformed SSE data: {}", event.data)));
                }
            }
        }
    }
    events
}

fn find_newline(buf: &[u8]) -> Option<usize> {
    buf.iter().position(|&b| b == b'\n')
}

/// Mutable connection state shared with in-flight requests.
#[derive(Default)]
struct HttpState {
    session_id: Option<String>,
    protocol_version: Option<String>,
    next_id: u64,
}

/// HTTP(SSE) transport state.
pub struct HttpTransport {
    config: McpServerConfig,
    timeout_ms: u64,
    client: reqwest::Client,
    state: Arc<Mutex<HttpState>>,
}

impl HttpTransport {
    /// Build a transport for an `http` (Streamable) or legacy `sse` server.
    pub fn new(_server_name: &str, config: McpServerConfig, _cwd: Arc<std::path::PathBuf>) -> Self {
        let timeout_ms = config.timeout_ms();
        Self {
            config,
            timeout_ms,
            client: reqwest::Client::new(),
            state: Arc::new(Mutex::new(HttpState::default())),
        }
    }

    /// Record the protocol version negotiated by `initialize`.
    fn set_protocol_version_sync(&self, version: &str) {
        self.state.lock().protocol_version = Some(version.to_string());
    }

    fn headers(&self, state: &HttpState) -> HeaderMap {
        let mut map = HeaderMap::new();
        map.insert(
            ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream"),
        );
        map.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(session) = &state.session_id {
            map.insert(
                "mcp-session-id",
                HeaderValue::from_str(session).unwrap_or_else(|_| HeaderValue::from_static("")),
            );
        }
        if let Some(version) = &state.protocol_version {
            map.insert(
                "mcp-protocol-version",
                HeaderValue::from_str(version).unwrap_or_else(|_| HeaderValue::from_static("")),
            );
        }
        for (name, value) in self.config.headers.clone().unwrap_or_default() {
            if let (Ok(name), Ok(value)) = (
                HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(&value),
            ) {
                if name != "mcp-protocol-version" {
                    map.insert(name, value);
                }
            }
        }
        map
    }
}

impl HttpTransport {
    fn http_error(status: reqwest::StatusCode, body_text: &str) -> McpError {
        McpError::Transport(McpTransportError {
            transport: TransportKind::Http,
            stage: Failure::Receive,
            failure: Failure::HttpStatus,
            retryable: matches!(status.as_u16(), 404 | 502 | 503),
            message: format!("HTTP {status}: {}", truncate(body_text, 512)),
            code: Some(status.to_string()),
            data: None,
        })
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..s.floor_char_boundary(max)])
    }
}

/// Best-effort MCP cancellation for an abandoned HTTP request. The POST may
/// already have reached the server, so abort and timeout owe one notification
/// with the original id; `initialize` is exempt and sending is capped.
struct HttpCancelGuard {
    client: reqwest::Client,
    url: String,
    headers: HeaderMap,
    id: Value,
    method: String,
    settled: bool,
}

impl HttpCancelGuard {
    fn settle(&mut self) {
        self.settled = true;
    }
}

impl Drop for HttpCancelGuard {
    fn drop(&mut self) {
        if self.settled || self.method == "initialize" {
            return;
        }
        let body = json!({
            "jsonrpc": "2.0", "method": "notifications/cancelled",
            "params": {"requestId": self.id.clone()},
        });
        let request = self
            .client
            .post(&self.url)
            .headers(self.headers.clone())
            .json(&body);
        tokio::spawn(async move {
            let _ =
                tokio::time::timeout(std::time::Duration::from_millis(500), request.send()).await;
        });
    }
}

/// Send one JSON-RPC request over HTTP and return the response `result`.
#[allow(clippy::too_many_arguments)]
async fn send_request(
    client: reqwest::Client,
    url: String,
    headers: HeaderMap,
    id: Value,
    method: String,
    params: Value,
    timeout_ms: u64,
    state: Arc<Mutex<HttpState>>,
) -> Result<Value, McpError> {
    // Long-running calls ask the server for `notifications/progress` keyed
    // by the request id (MCP `_meta.progressToken`).
    let mut params = params;
    if method == "tools/call" {
        if let Some(meta) = params
            .as_object_mut()
            .map(|m| m.entry("_meta").or_insert_with(|| json!({})))
            .and_then(Value::as_object_mut)
        {
            meta.entry("progressToken").or_insert_with(|| id.clone());
        }
    }
    let body = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
    let mut cancel = HttpCancelGuard {
        client: client.clone(),
        url: url.clone(),
        headers: headers.clone(),
        id: id.clone(),
        method: method.clone(),
        settled: false,
    };
    let request = client.post(&url).headers(headers).json(&body);
    let expected_key = PendingRequests::key(&id);
    let session_cell: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let session_cell_inner = session_cell.clone();
    let send = async move {
        let response = request.send().await.map_err(|err| {
            // `send()` also waits for the response headers. A connect failure
            // happened before this request reached a server and is safe to
            // replay; later failures are not, even when reqwest labels them
            // as timeouts.
            let retryable = err.is_connect();
            let error = McpError::Transport(McpTransportError {
                transport: TransportKind::Http,
                stage: if retryable {
                    Failure::Connect
                } else {
                    Failure::Receive
                },
                failure: if retryable {
                    Failure::Connect
                } else {
                    Failure::Receive
                },
                retryable,
                message: format!("fetch failed: {err}"),
                code: err.status().map(|s| s.to_string()),
                data: None,
            });
            if retryable {
                error
            } else {
                may_have_run(error)
            }
        })?;
        let status = response.status();
        let session: Option<String> = response
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        if let Some(s) = &session {
            *session_cell_inner.lock() = Some(s.clone());
        }
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        if content_type.contains("text/event-stream") {
            // Streamable HTTP SSE response: scan events for our response.
            let events = read_sse_events(response).await;
            for event in events {
                match event {
                    Ok(msg) => {
                        if let Some(response) = extract_response(&msg, &expected_key) {
                            return response;
                        }
                    }
                    Err(err) => tracing::debug!("ignoring malformed SSE event: {err}"),
                }
            }
            return Err(may_have_run(McpError::Transport(McpTransportError {
                transport: TransportKind::Http,
                stage: Failure::Receive,
                failure: Failure::Eof,
                retryable: true,
                message: "SSE stream closed before a response arrived".to_string(),
                code: None,
                data: None,
            })));
        }
        let text = response.text().await.map_err(|err| {
            may_have_run(McpError::Transport(McpTransportError {
                transport: TransportKind::Http,
                stage: Failure::Receive,
                failure: Failure::Receive,
                retryable: true,
                message: format!("failed to read response body: {err}"),
                code: None,
                data: None,
            }))
        })?;
        if !status.is_success() {
            return Err(may_have_run(HttpTransport::http_error(status, &text)));
        }
        parse_json_rpc(&text)
    };
    let result = if crate::timeout::is_timeout_enabled(timeout_ms) {
        match tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), send).await {
            Ok(outcome) => outcome,
            Err(_elapsed) => Err(McpError::Transport(McpTransportError {
                transport: TransportKind::Http,
                stage: Failure::Timeout,
                failure: Failure::Timeout,
                retryable: false,
                message: format!("Request timeout after {timeout_ms}ms"),
                code: None,
                data: None,
            })),
        }
    } else {
        send.await
    };
    match &result {
        Ok(_) => cancel.settle(),
        Err(McpError::Transport(error)) if error.retryable && error.failure == Failure::Connect => {
            // The request never reached a server; no cancellation is owed.
            cancel.settle();
        }
        Err(_) => {
            // EOF, timeout, status and decode failures are ambiguous after
            // the POST started. Tell the server not to continue the call.
        }
    }
    if result.is_ok() {
        if let Some(session) = session_cell.lock().take() {
            state.lock().session_id = Some(session);
        }
    }
    result
}

impl Clone for HttpState {
    fn clone(&self) -> Self {
        Self {
            session_id: self.session_id.clone(),
            protocol_version: self.protocol_version.clone(),
            next_id: self.next_id,
        }
    }
}

impl McpTransport for HttpTransport {
    fn set_protocol_version(&mut self, version: &str) {
        self.set_protocol_version_sync(version);
    }

    fn request(&mut self, method: &str, params: Value) -> TransportFut<Result<Value, McpError>> {
        // Claim the id before the request future exists: ids stay unique for
        // concurrent requests and are never reused after a failed one (MCP:
        // an id must not be reused within a session).
        let state = {
            let mut guard = self.state.lock();
            guard.next_id += 1;
            guard.clone()
        };
        let id = Value::from(state.next_id);
        let url = self.config.url.clone().unwrap_or_default();
        let headers = self.headers(&state);
        let client = self.client.clone();
        let timeout_ms = self.timeout_ms;
        let state = self.state.clone();
        let method = method.to_string();
        Box::pin(send_request(
            client, url, headers, id, method, params, timeout_ms, state,
        ))
    }

    fn notify(&mut self, method: &str, params: Value) -> TransportFut<Result<(), McpError>> {
        let url = self.config.url.clone().unwrap_or_default();
        let body = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        let state = self.state.lock().clone();
        let request = self
            .client
            .post(&url)
            .headers(self.headers(&state))
            .json(&body);
        Box::pin(async move {
            let response = request.send().await.map_err(|err| {
                McpError::Transport(McpTransportError {
                    transport: TransportKind::Http,
                    stage: Failure::Connect,
                    failure: Failure::Connect,
                    retryable: true,
                    message: format!("fetch failed: {err}"),
                    code: err.status().map(|s| s.to_string()),
                    data: None,
                })
            })?;
            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return Err(Self::http_error(status, &text));
            }
            Ok(())
        })
    }

    fn close(&mut self) -> TransportFut<()> {
        let url = self.config.url.clone().unwrap_or_default();
        let state = self.state.clone();
        let client = self.client.clone();
        Box::pin(async move {
            let session = state.lock().session_id.take();
            if let Some(session) = session {
                let _ = client
                    .delete(&url)
                    .header("mcp-session-id", session)
                    .send()
                    .await;
            }
        })
    }
}

/// Extract a JSON-RPC response destined for our request from one SSE event;
/// server→client requests and notifications are ignored here (the streaming
/// GET listener that answers them is not ported).
fn extract_response(msg: &Value, expected_key: &str) -> Option<Result<Value, McpError>> {
    let obj = msg.as_object()?;
    if obj.contains_key("method") {
        return None;
    }
    let id = obj.get("id")?;
    if PendingRequests::key(id) != expected_key {
        return None;
    }
    if let Some(error) = obj.get("error") {
        return Some(Err(McpError::JsonRpc(
            serde_json::from_value::<JsonRpcError>(error.clone()).unwrap_or(JsonRpcError {
                code: 0,
                message: "malformed JSON-RPC error response".to_string(),
                data: None,
            }),
        )));
    }
    Some(Ok(obj.get("result").cloned().unwrap_or(Value::Null)))
}

/// Parse a plain JSON JSON-RPC response body.
fn parse_json_rpc(text: &str) -> Result<Value, McpError> {
    let value: Value = serde_json::from_str(text).map_err(|err| {
        McpError::Transport(McpTransportError {
            transport: TransportKind::Http,
            stage: Failure::Decode,
            failure: Failure::Decode,
            retryable: false,
            message: format!("malformed JSON-RPC response: {err}"),
            code: None,
            data: None,
        })
    })?;
    let Some(obj) = value.as_object() else {
        return Err(McpError::Config("malformed JSON-RPC response".to_string()));
    };
    if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(McpError::Config("malformed JSON-RPC response".to_string()));
    }
    if let Some(error) = obj.get("error") {
        return Err(McpError::JsonRpc(
            serde_json::from_value::<JsonRpcError>(error.clone()).unwrap_or(JsonRpcError {
                code: 0,
                message: "malformed JSON-RPC error response".to_string(),
                data: None,
            }),
        ));
    }
    Ok(obj.get("result").cloned().unwrap_or(Value::Null))
}
