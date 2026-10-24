//! MCP client for omp-rs: a Rust port of `pi-coding-agent/src/mcp`.
//!
//! Covers the pieces the agent loop needs: config loading (user
//! `~/.omp/agent/mcp.json` + project `mcp.json`/`.mcp.json`), stdio and
//! Streamable-HTTP(SSE) transports, `initialize`/`tools/list`/`tools/call`,
//! omp's `mcp__<server>_<tool>` name minting, schema passthrough
//! (`normalizeSchemaForMCP`), result formatting, per-request timeouts, lazy
//! connect and per-server failure isolation.
//!
//! Not ported (omp features outside the agent-tool surface): OAuth flows,
//! Smithery, tool caching across sessions, prompts/resources, the TUI render
//! callbacks, and Bun-compatible hashing for overlong tool names (see
//! [`names::create_tool_name`]).

pub mod client;
pub mod config;
pub mod errors;
pub mod format;
pub mod manager;
pub mod names;
pub mod schema;
pub mod timeout;
pub mod transport;

pub use client::{call_tool, connect_to_server, list_tools, Connection};
pub use config::{load_configs, McpConfigFile, McpServerConfig, ServerKind};
pub use errors::{JsonRpcError, McpConnectionTimeoutError, McpError, McpTransportError};
pub use manager::{DiscoveredTool, McpManager, ServerStatus};
pub use names::{create_legacy_tool_name, create_tool_name};
pub use timeout::{describe_timeout, is_timeout_enabled, resolve_timeout_ms, DEFAULT_TIMEOUT_MS};
pub use transport::{McpTransport, PROTOCOL_VERSION};

use serde_json::Value;

/// One MCP tool definition (`tools/list` entry), schema kept verbatim.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub input_schema: Value,
    pub output_schema: Option<Value>,
    #[allow(dead_code)]
    pub annotations: Option<Value>,
}

/// `tools/call` result.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCallResult {
    pub content: Vec<Content>,
    pub is_error: bool,
    pub structured_content: Option<Value>,
    #[allow(dead_code)]
    pub meta: Option<Value>,
}

/// One MCP content block.
#[derive(Debug, Clone, PartialEq)]
pub enum Content {
    Text {
        text: String,
    },
    Image {
        data: String,
        mime_type: String,
    },
    Resource {
        uri: String,
        text: Option<String>,
    },
    /// Any other content type, kept as raw JSON.
    Other(Value),
}
