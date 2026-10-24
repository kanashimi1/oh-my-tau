//! Connection lifecycle: a port of `client.ts` (`connectToServer`,
//! `listTools`, `callTool`, `initializeConnection`).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use crate::config::{McpServerConfig, ServerKind};
use crate::errors::{McpConnectionTimeoutError, McpError};
use crate::timeout::is_timeout_enabled;
use crate::transport::http::HttpTransport;
use crate::transport::stdio::StdioTransport;
use crate::transport::McpTransport;
use crate::transport::PROTOCOL_VERSION;
use crate::{Content, ToolCallResult, ToolDefinition};

/// Client info sent during initialization (omp `CLIENT_INFO`).
const CLIENT_INFO: (&str, &str) = ("omt", env!("CARGO_PKG_VERSION"));

/// An initialized MCP server connection, like omp's `MCPServerConnection`.
pub struct Connection {
    pub name: String,
    pub config: McpServerConfig,
    pub transport: Box<dyn McpTransport>,
    pub server_info: Value,
    pub capabilities: Value,
    pub instructions: Option<String>,
    pub tools: Option<Vec<ToolDefinition>>,
}

/// Connect to an MCP server: create the transport, run `initialize`, send
/// `notifications/initialized`, all under the configured timeout (omp
/// `connectToServer`; the deadline defaults to 30s, `0` disables it).
pub async fn connect_to_server(
    name: &str,
    config: &McpServerConfig,
    cwd: Arc<PathBuf>,
) -> Result<Connection, McpError> {
    let timeout_ms = config.timeout_ms();
    let connect = async {
        let mut transport = create_transport(name, config, cwd)?;
        let init_result = initialize_connection(&mut *transport).await?;
        if let Some(version) = init_result.get("protocolVersion").and_then(Value::as_str) {
            transport.set_protocol_version(version);
        }
        Ok(Connection {
            name: name.to_string(),
            config: config.clone(),
            transport,
            server_info: init_result
                .get("serverInfo")
                .cloned()
                .unwrap_or(Value::Null),
            capabilities: init_result
                .get("capabilities")
                .cloned()
                .unwrap_or_else(|| json!({})),
            instructions: init_result
                .get("instructions")
                .and_then(Value::as_str)
                .map(str::to_string),
            tools: None,
        })
    };
    if is_timeout_enabled(timeout_ms) {
        match tokio::time::timeout(Duration::from_millis(timeout_ms), connect).await {
            Ok(result) => result,
            Err(_elapsed) => Err(McpError::Timeout(McpConnectionTimeoutError {
                server_name: name.to_string(),
                timeout_ms,
            })),
        }
    } else {
        connect.await
    }
}

/// Build the transport for a server config (omp `createTransport`). The stdio
/// subprocess is spawned here so spawn failures surface as connect errors.
fn create_transport(
    name: &str,
    config: &McpServerConfig,
    cwd: Arc<PathBuf>,
) -> Result<Box<dyn McpTransport>, McpError> {
    match config.kind {
        ServerKind::Stdio => {
            let mut transport = StdioTransport::new(name, config.clone(), cwd);
            futures::executor::block_on(transport.connect())?;
            Ok(Box::new(transport))
        }
        ServerKind::Http | ServerKind::Sse => {
            Ok(Box::new(HttpTransport::new(name, config.clone(), cwd)))
        }
    }
}

/// `initialize` then `notifications/initialized` (omp `initializeConnection`).
async fn initialize_connection(transport: &mut dyn McpTransport) -> Result<Value, McpError> {
    let params = json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": { "roots": { "listChanged": false } },
        "clientInfo": { "name": CLIENT_INFO.0, "version": CLIENT_INFO.1 },
    });
    let result = transport.request("initialize", params).await?;
    transport
        .notify("notifications/initialized", json!({}))
        .await?;
    Ok(result)
}

/// List tools from a connected server with cursor pagination, like omp
/// `listTools`. Servers without tool support yield an empty list.
pub async fn list_tools(connection: &mut Connection) -> Result<Vec<ToolDefinition>, McpError> {
    if connection.capabilities.get("tools").is_none() && connection.tools.is_none() {
        return Ok(Vec::new());
    }
    if let Some(tools) = &connection.tools {
        return Ok(tools.clone());
    }
    let mut all_tools = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let mut params = json!({});
        if let Some(c) = &cursor {
            params["cursor"] = Value::String(c.clone());
        }
        let result = connection.transport.request("tools/list", params).await?;
        if let Some(tools) = result.get("tools").and_then(Value::as_array) {
            for tool in tools {
                all_tools.push(parse_tool_definition(tool));
            }
        }
        cursor = result
            .get("nextCursor")
            .and_then(Value::as_str)
            .map(str::to_string);
        if cursor.is_none() {
            break;
        }
    }
    connection.tools = Some(all_tools.clone());
    Ok(all_tools)
}

/// Parse one `tools/list` entry, keeping the input schema verbatim.
fn parse_tool_definition(value: &Value) -> ToolDefinition {
    ToolDefinition {
        name: value
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        title: value
            .get("title")
            .and_then(Value::as_str)
            .map(str::to_string),
        description: value
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_string),
        input_schema: value
            .get("inputSchema")
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object" })),
        output_schema: value.get("outputSchema").cloned(),
        annotations: value.get("annotations").cloned(),
    }
}

/// Call a tool on a connected server (omp `callTool`).
pub async fn call_tool(
    connection: &mut Connection,
    tool_name: &str,
    args: Value,
) -> Result<ToolCallResult, McpError> {
    let params = json!({ "name": tool_name, "arguments": args });
    let result = connection.transport.request("tools/call", params).await?;
    Ok(parse_tool_call_result(&result))
}

/// Parse a `tools/call` result into content blocks + flags.
pub(crate) fn parse_tool_call_result(result: &Value) -> ToolCallResult {
    let content = result
        .get("content")
        .and_then(Value::as_array)
        .map(|items| items.iter().map(parse_content).collect())
        .unwrap_or_default();
    ToolCallResult {
        content,
        is_error: result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        structured_content: result.get("structuredContent").cloned(),
        meta: result.get("_meta").cloned(),
    }
}

fn parse_content(value: &Value) -> Content {
    match value.get("type").and_then(Value::as_str) {
        Some("text") => Content::Text {
            text: value
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        Some("image") => Content::Image {
            data: value
                .get("data")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            mime_type: value
                .get("mimeType")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        Some("resource") => {
            let resource = value.get("resource").cloned().unwrap_or_else(|| json!({}));
            Content::Resource {
                uri: resource
                    .get("uri")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                text: resource
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            }
        }
        _ => Content::Other(value.clone()),
    }
}
