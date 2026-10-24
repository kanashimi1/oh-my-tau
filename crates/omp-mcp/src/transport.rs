//! MCP transports: the `MCPTransport` trait plus stdio and Streamable-HTTP
//! implementations, ports of `transports/stdio.ts` and `transports/http.ts`.
//!
//! Both are JSON-RPC 2.0: requests carry a per-transport incrementing id,
//! responses resolve the pending request, server→client requests get default
//! answers (`ping` → `{}`, `roots/list` → the project root), notifications
//! are dispatched to a callback, and every request obeys the configured
//! timeout (`timeout: 0` / `OMP_MCP_TIMEOUT_MS=0` disables it).

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::sync::{oneshot, Mutex};

use crate::errors::{Failure, JsonRpcError, McpError, McpTransportError, TransportKind};
use crate::timeout::is_timeout_enabled;

/// Latest MCP protocol revision this client negotiates (omp `MCP_PROTOCOL_VERSION`).
pub const PROTOCOL_VERSION: &str = "2025-11-25";

/// Boxed future returned by transport methods.
pub type TransportFut<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Transport abstraction over stdio/http, like omp's `MCPTransport`.
pub trait McpTransport: Send {
    /// Send a JSON-RPC request and await its response `result`.
    fn request(&mut self, method: &str, params: Value) -> TransportFut<Result<Value, McpError>>;
    /// Send a JSON-RPC notification (no response expected).
    fn notify(&mut self, method: &str, params: Value) -> TransportFut<Result<(), McpError>>;
    /// Close the transport and release its resources.
    fn close(&mut self) -> TransportFut<()>;
    /// Record the protocol version negotiated by `initialize`; HTTP sends it
    /// as the `MCP-Protocol-Version` header afterwards (default: no-op).
    fn set_protocol_version(&mut self, _version: &str) {}
}

/// Default answer for server→client requests, port of `defaultRequestHandler`.
async fn default_request_handler(
    method: &str,
    cwd: &std::path::Path,
) -> Result<Value, JsonRpcError> {
    match method {
        "ping" => Ok(json!({})),
        "roots/list" => {
            let name = cwd.file_name().and_then(|n| n.to_str()).unwrap_or("");
            Ok(json!({ "roots": [{ "uri": cwd.to_string_lossy(), "name": name }] }))
        }
        other => Err(JsonRpcError {
            code: -32601,
            message: format!("Unsupported server request: {other}"),
            data: None,
        }),
    }
}

/// Shared request-id allocation (`RequestIdAllocator`): monotonic integers.
#[derive(Debug, Default)]
pub struct RequestIdAllocator {
    next: u64,
}

impl RequestIdAllocator {
    /// Allocate the next id; `string` format stringifies (omp
    /// `requestIdFormat: "string"` for collision-resistant snowflake ids —
    /// modeled here as the same counter rendered as a string).
    pub fn next(&mut self, string_format: bool) -> Value {
        self.next += 1;
        if string_format {
            Value::String(self.next.to_string())
        } else {
            Value::from(self.next)
        }
    }
}

/// Per-transport pending-request table shared between the request path and
/// the read/dispatch loop.
#[derive(Debug, Default)]
pub struct PendingRequests {
    pub map: HashMap<String, oneshot::Sender<Result<Value, McpError>>>,
}

impl PendingRequests {
    /// Key used to match responses to requests (JSON value equality on the id).
    #[must_use]
    pub fn key(id: &Value) -> String {
        serde_json::to_string(id).unwrap_or_default()
    }
}

/// How an incoming server→client request is answered.
pub type RequestHandler =
    Arc<dyn Fn(&str, Value) -> TransportFut<Result<Value, JsonRpcError>> + Send + Sync>;

/// Callback invoked for server→client notifications.
pub type NotificationCallback = Arc<dyn Fn(&str, Value) + Send + Sync>;

/// Sink used to send JSON-RPC responses back to the server.
pub type RespondCallback = Arc<dyn Fn(Value) + Send + Sync>;

/// Build the default handler bound to `cwd`.
#[must_use]
pub fn default_handler(cwd: Arc<std::path::PathBuf>) -> RequestHandler {
    Arc::new(move |method, _params| {
        let cwd = cwd.clone();
        let method = method.to_string();
        Box::pin(async move { default_request_handler(&method, &cwd).await })
    })
}

/// Dispatch one incoming JSON-RPC message: responses resolve pending
/// requests, requests get answered via `handler` and `respond`, plain
/// notifications go to `on_notification`.
#[allow(clippy::too_many_arguments)]
pub async fn dispatch_message(
    msg: Value,
    _kind: TransportKind,
    pending: &Arc<Mutex<PendingRequests>>,
    handler: &RequestHandler,
    respond: &RespondCallback,
    on_notification: &NotificationCallback,
) {
    let Some(obj) = msg.as_object() else { return };
    let id = obj.get("id").cloned();
    if let Some(method) = obj.get("method").and_then(Value::as_str) {
        if let Some(id) = id {
            // Server→client request: answer with result or JSON-RPC error.
            let method = method.to_string();
            let params = obj.get("params").cloned().unwrap_or(Value::Null);
            let handler = handler.clone();
            let respond = respond.clone();
            tokio::spawn(async move {
                let mut response = match handler(&method, params).await {
                    Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
                    Err(error) => json!({ "jsonrpc": "2.0", "id": id, "error": error }),
                };
                if let Some(map) = response.as_object_mut() {
                    map.insert("jsonrpc".to_string(), json!("2.0"));
                }
                respond(response);
            });
            return;
        }
        on_notification(method, obj.get("params").cloned().unwrap_or(Value::Null));
        return;
    }
    // Response to one of our requests.
    let Some(id) = id else { return };
    let key = PendingRequests::key(&id);
    let result = match obj.get("error") {
        Some(error) => Err(McpError::JsonRpc(
            serde_json::from_value::<JsonRpcError>(error.clone()).unwrap_or(JsonRpcError {
                code: 0,
                message: "malformed JSON-RPC error".to_string(),
                data: None,
            }),
        )),
        None => Ok(obj.get("result").cloned().unwrap_or(Value::Null)),
    };
    let mut guard = pending.lock().await;
    if let Some(tx) = guard.map.remove(&key) {
        let _ = tx.send(result);
    }
}

/// Enforce the request deadline around a pending response, port of the
/// per-request timeout in the stdio/http `request` implementations.
pub async fn await_with_timeout(
    rx: oneshot::Receiver<Result<Value, McpError>>,
    timeout_ms: u64,
    kind: TransportKind,
) -> Result<Value, McpError> {
    if !is_timeout_enabled(timeout_ms) {
        return rx.await.unwrap_or_else(|_| Err(connection_closed(kind)));
    }
    match tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), rx).await {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => Err(connection_closed(kind)),
        Err(_elapsed) => Err(McpError::Transport(kind_error(
            kind,
            Failure::Timeout,
            format!("Request timeout after {timeout_ms}ms"),
            false,
        ))),
    }
}

fn kind_error(
    kind: TransportKind,
    failure: Failure,
    message: String,
    retryable: bool,
) -> McpTransportError {
    McpTransportError {
        transport: kind,
        stage: failure,
        failure,
        retryable,
        message,
        code: None,
        data: None,
    }
}

fn connection_closed(kind: TransportKind) -> McpError {
    McpError::Transport(kind_error(
        kind,
        Failure::Closed,
        "Transport closed".to_string(),
        true,
    ))
}

pub mod http;
pub mod stdio;
