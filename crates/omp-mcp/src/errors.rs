//! Error types: a port of `mcp/errors.ts` (`MCPTransportError`,
//! `formatMCPToolFailure`) and the JSON-RPC error shape.

use std::fmt;

/// Transport / stage / failure classification, mirroring omp's
/// `MCPTransportError` fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportKind {
    Stdio,
    Http,
    Unknown,
}

impl fmt::Display for TransportKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stdio => write!(f, "stdio"),
            Self::Http => write!(f, "http"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Failure classification for transport errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    Connect,
    Send,
    Receive,
    Decode,
    Timeout,
    Closed,
    Eof,
    HttpStatus,
    MalformedResponse,
    Protocol,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Connect => "connect",
            Self::Send => "send",
            Self::Receive => "receive",
            Self::Decode => "decode",
            Self::Timeout => "timeout",
            Self::Closed => "closed",
            Self::Eof => "eof",
            Self::HttpStatus => "http_status",
            Self::MalformedResponse => "malformed_response",
            Self::Protocol => "protocol",
        };
        write!(f, "{s}")
    }
}

/// A structured MCP transport error, like omp's `MCPTransportError`.
#[derive(Debug, Clone)]
pub struct McpTransportError {
    pub transport: TransportKind,
    pub stage: Failure,
    pub failure: Failure,
    pub retryable: bool,
    pub message: String,
    pub code: Option<String>,
    pub data: Option<String>,
}

impl fmt::Display for McpTransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MCP {} {} error ({}): {}",
            self.transport, self.stage, self.failure, self.message
        )
    }
}

impl std::error::Error for McpTransportError {}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// All MCP client errors.
#[derive(Debug)]
pub enum McpError {
    Transport(McpTransportError),
    JsonRpc(JsonRpcError),
    /// Connection handshake exceeded the configured timeout.
    Timeout(McpConnectionTimeoutError),
    /// The server configuration is unusable.
    Config(String),
}

impl fmt::Display for McpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "{e}"),
            Self::JsonRpc(e) => write!(f, "JSON-RPC error {}: {}", e.code, e.message),
            Self::Timeout(e) => write!(f, "{e}"),
            Self::Config(m) => write!(f, "invalid MCP server config: {m}"),
        }
    }
}

impl std::error::Error for McpError {}

impl From<McpTransportError> for McpError {
    fn from(e: McpTransportError) -> Self {
        Self::Transport(e)
    }
}

impl From<JsonRpcError> for McpError {
    fn from(e: JsonRpcError) -> Self {
        Self::JsonRpc(e)
    }
}

/// Identifies a server whose initial handshake exceeded its timeout.
#[derive(Debug, Clone)]
pub struct McpConnectionTimeoutError {
    pub server_name: String,
    pub timeout_ms: u64,
}

impl fmt::Display for McpConnectionTimeoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Connection to MCP server \"{}\" timed out after {}",
            self.server_name,
            crate::timeout::describe_timeout(self.timeout_ms)
        )
    }
}

impl std::error::Error for McpConnectionTimeoutError {}

/// Render the user/model-facing failure block, a port of
/// `formatMCPToolFailure`: one `key: value` line per diagnostic field,
/// ending with a `next:` hint.
#[must_use]
pub fn format_tool_failure(error: &McpError, server_name: &str, tool_name: &str) -> String {
    let (transport, stage, failure, retryable, message, code) = match error {
        McpError::Transport(e) => (
            e.transport.to_string(),
            e.stage.to_string(),
            e.failure.to_string(),
            e.retryable,
            e.message.clone(),
            e.code.clone(),
        ),
        McpError::JsonRpc(e) => (
            "unknown".to_string(),
            "protocol".to_string(),
            "jsonrpc_error".to_string(),
            false,
            format!("{} (code {})", e.message, e.code),
            Some(e.code.to_string()),
        ),
        McpError::Timeout(e) => (
            "unknown".to_string(),
            "connect".to_string(),
            "timeout".to_string(),
            true,
            e.to_string(),
            None,
        ),
        McpError::Config(m) => (
            "unknown".to_string(),
            "connect".to_string(),
            "config".to_string(),
            false,
            m.clone(),
            None,
        ),
    };
    let mut lines = vec![
        "MCP failure".to_string(),
        format!("server: {server_name}"),
        format!("tool: {tool_name}"),
        format!("transport: {transport}"),
        format!("stage: {stage}"),
        format!("failure: {failure}"),
        format!("retryable: {}", if retryable { "yes" } else { "no" }),
        format!("message: {message}"),
    ];
    if let Some(code) = code {
        lines.push(format!("code: {code}"));
    }
    lines.push(format!("next: {}", next_step(retryable)));
    lines.join("\n")
}

fn next_step(retryable: bool) -> &'static str {
    if retryable {
        "retry the call; if it keeps failing, reconnect the MCP server"
    } else {
        "check the server and its configuration"
    }
}
