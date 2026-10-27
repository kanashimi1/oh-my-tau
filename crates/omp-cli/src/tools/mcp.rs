//! MCP tool registration: bridges `omp-mcp` discovered tools into the
//! omp-rs tool registry with omp's names (`mcp__<server>_<tool>`), schema
//! passthrough (`normalize_schema_for_mcp`) and result formatting
//! (`format_tool_result`), matching pi-coding-agent's `tool-bridge.ts`.
//!
//! Discovery is eager but connection-failure-isolated: servers that fail to
//! connect simply contribute no tools (their failure is logged), exactly
//! like omp's startup. Individual calls are lazy in the sense that the
//! manager reconnects on demand and isolates per-server errors.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use omp_mcp::manager::McpManager;
use omp_mcp::names::create_tool_name;
use omp_mcp::schema::normalize_schema_for_mcp;
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback, ToolResultContent};
use pa_agent::BoxFut;
use serde_json::Value;

/// Discover MCP tools from omp's config locations (`~/.omp/agent/mcp.json`
/// plus project `mcp.json`/`.mcp.json`) and wrap them as agent tools.
///
/// This performs network/subprocess I/O (servers connect eagerly, exactly
/// like omp's startup) and the transports spawn `tokio::spawn` tasks (stdio
/// read loop, subprocess pipes) that must outlive this call, so it must be
/// awaited on the process's main runtime — never on a throwaway runtime.
#[must_use]
pub async fn mcp_tools(cwd: &Path) -> Vec<Arc<dyn AgentTool>> {
    let requested = REQUESTED_TOOLS.lock().clone();
    register_tools(Arc::new(McpManager::from_configs_for(cwd, &requested))).await
}

/// Connect `manager`'s servers and wrap their tools (records the connected
/// servers for the status line).
async fn register_tools(manager: Arc<McpManager>) -> Vec<Arc<dyn AgentTool>> {
    let discovered = manager.tools().await;
    let mut connected = Vec::new();
    for (name, status) in manager.statuses().await {
        match status {
            omp_mcp::ServerStatus::Failed(message) => {
                eprintln!("omt: MCP server \"{name}\" unavailable: {message}");
            }
            omp_mcp::ServerStatus::Connected => {
                let tools = discovered.iter().filter(|t| t.server == name).count();
                connected.push((name, tools));
            }
            omp_mcp::ServerStatus::Pending => {}
        }
    }
    *CONNECTED_SERVERS.lock() = connected;
    discovered
        .into_iter()
        .map(|tool| {
            let definition = tool.definition.clone();
            let name = create_tool_name(&tool.server, &definition.name);
            let description = definition
                .description
                .clone()
                .unwrap_or_else(|| format!("MCP tool from {}", tool.server));
            let parameters = normalize_schema_for_mcp(&definition.input_schema);
            Arc::new(McpTool {
                name,
                description,
                parameters,
                server: tool.server,
                tool_name: definition.name,
                manager: manager.clone(),
            }) as Arc<dyn AgentTool>
        })
        .collect::<Vec<_>>()
}

/// Tool names this run is restricted to (`--tools <csv>`, an agent's `tools:`).
/// MCP servers marked `enabled: false` still load when one of their tools is
/// named here; set once at startup, read by every later (re)discovery.
static REQUESTED_TOOLS: parking_lot::Mutex<Vec<String>> = parking_lot::Mutex::new(Vec::new());

/// Record the run's `--tools <csv>` list for MCP discovery (see [`REQUESTED_TOOLS`]).
pub fn set_requested_tools(csv: &str) {
    *REQUESTED_TOOLS.lock() = csv
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect();
}

/// Servers connected by the latest discovery, with their tool counts (TS
/// `MCPManager.getConnectedServers` + `getConnection(name).tools.length`).
static CONNECTED_SERVERS: parking_lot::Mutex<Vec<(String, usize)>> =
    parking_lot::Mutex::new(Vec::new());

/// `(server, tool count)` for every MCP server connected by the latest discovery.
#[must_use]
pub fn connected_servers() -> Vec<(String, usize)> {
    CONNECTED_SERVERS.lock().clone()
}

/// One registered MCP tool: schema/description come from `tools/list`, the
/// registry key is the omp-minted `mcp__<server>_<tool>` name, and executing
/// calls through the shared [`McpManager`] (lazy connect + one reconnect
/// retry on stale connections).
struct McpTool {
    name: String,
    description: String,
    parameters: Value,
    server: String,
    tool_name: String,
    manager: Arc<McpManager>,
}

impl AgentTool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters(&self) -> &Value {
        &self.parameters
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        params: Value,
        signal: AbortSignal,
        on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move {
            if signal.is_aborted() {
                anyhow::bail!("aborted");
            }
            // omp `prepareOutboundArgs`: non-object args normalize to `{}`.
            let args = match params {
                Value::Object(_) => params,
                _ => Value::Object(serde_json::Map::new()),
            };
            // Server→client progress updates refresh the running tool card
            // through the agent loop's `tool_execution_update` events.
            let (server, tool_name) = (self.server.clone(), self.tool_name.clone());
            let progress_update = {
                let on_update = on_update.clone();
                move |update: omp_mcp::McpProgress| {
                    let text = match (&update.message, update.progress, update.total) {
                        (Some(message), _, _) => format!("mcp progress: {message}"),
                        (None, Some(done), Some(total)) => {
                            format!("mcp progress: {done}/{total}")
                        }
                        (None, Some(done), None) => format!("mcp progress: {done}"),
                        (None, None, _) => "mcp progress".to_string(),
                    };
                    on_update(AgentToolResult {
                        content: vec![ToolResultContent::text(text)],
                        details: serde_json::json!({
                            "serverName": server,
                            "toolName": tool_name,
                            "isError": false,
                            "mcpProgress": {
                                "progress": update.progress,
                                "total": update.total,
                                "message": update.message,
                            },
                        }),
                        terminate: None,
                        is_error: None,
                    });
                }
            };
            match self
                .manager
                .call_with_progress(&self.server, &self.tool_name, args, progress_update)
                .await
            {
                Ok(result) => {
                    let text = omp_mcp::format::format_tool_result(&result);
                    let content = vec![ToolResultContent::text(text)];
                    let details = serde_json::json!({
                        "serverName": self.server,
                        "toolName": self.tool_name,
                        "isError": result.is_error,

                    });
                    Ok(AgentToolResult {
                        content,
                        details,
                        terminate: None,
                        is_error: None,
                    })
                }
                Err(error) => {
                    let text =
                        omp_mcp::errors::format_tool_failure(&error, &self.server, &self.tool_name);
                    Ok(AgentToolResult {
                        content: vec![ToolResultContent::text(text)],
                        details: serde_json::json!({
                            "serverName": self.server,
                            "toolName": self.tool_name,
                            "isError": true,
                        }),
                        terminate: None,
                        is_error: None,
                    })
                }
            }
        })
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod tests {
    use super::*;
    use std::io::Write;

    /// Same fake MCP stdio server as the omp-mcp fixture.
    const FAKE_SERVER_PY: &str = r#"
import json, sys
TOOLS = [{"name": "echo", "description": "Echo args",
          "inputSchema": {"type": "object", "properties": {
              "text": {"type": "string"}}, "required": ["text"]}}]
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2025-11-25",
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake", "version": "0.1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        args = msg["params"].get("arguments", {})
        if args.get("text") == "progress":
            print(json.dumps({"jsonrpc": "2.0", "method": "notifications/progress",
                "params": {"progressToken": msg["params"]["_meta"]["progressToken"],
                           "progress": 1, "total": 2, "message": "halfway"}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
            "result": {"content": [{"type": "text", "text": json.dumps(args)}]}}), flush=True)
        continue
    else:
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
"#;

    /// Registration matches omp: `mcp__<server>_<tool>` names, verbatim
    /// schema passthrough, `MCP tool from <server>` fallback description,
    /// and `execute()` result formatting.
    #[tokio::test(flavor = "multi_thread")]
    async fn mcp_tools_register_with_omp_names_and_schemas() {
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("python3 not available; skipping");
            return;
        }
        let root = tempfile::tempdir().expect("tempdir");
        let project = root.path().join("project");
        std::fs::create_dir_all(&project).unwrap();
        let script = project.join("fake.py");
        std::fs::File::create(&script)
            .unwrap()
            .write_all(FAKE_SERVER_PY.as_bytes())
            .unwrap();
        // Explicit server config: no process env (OMP_CONFIG_DIR) or
        // `--tools` global, which parallel tests would race on.
        let config: omp_mcp::McpServerConfig = serde_json::from_value(serde_json::json!({
            "command": "python3", "args": [script.to_string_lossy()], "timeout": 15000
        }))
        .expect("server config");
        let configs = std::iter::once(("fake".to_string(), config)).collect();
        let manager = McpManager::new(Arc::new(project.clone()), configs);
        let tools = register_tools(Arc::new(manager)).await;

        assert_eq!(tools.len(), 1);
        let tool = &tools[0];
        assert_eq!(tool.name(), "mcp__fake_echo");
        assert_eq!(tool.description(), "Echo args");
        // Schema passthrough: server-provided inputSchema is kept verbatim.
        assert_eq!(
            tool.parameters()["properties"]["text"]["type"],
            serde_json::json!("string")
        );
        assert_eq!(tool.parameters()["required"], serde_json::json!(["text"]));

        // Execute through the AgentTool surface (on the same runtime that
        // ran discovery, as the production main runtime does).
        let result = tool
            .clone()
            .execute(
                "call-1".to_string(),
                serde_json::json!({"text": "hi"}),
                AbortSignal::never(),
                Arc::new(|_| {}),
            )
            .await
            .expect("execute");
        let text = match &result.content[0] {
            ToolResultContent::Text(t) => t.text.clone(),
            ToolResultContent::Image(_) => panic!("unexpected content: image"),
        };
        assert!(text.contains("\"text\": \"hi\""), "result text: {text}");
        assert_eq!(result.details["serverName"], serde_json::json!("fake"));
        assert_eq!(result.details["toolName"], serde_json::json!("echo"));
        assert_eq!(result.details["isError"], serde_json::json!(false));

        let updates = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let updates_in_callback = Arc::clone(&updates);
        tool.clone()
            .execute(
                "call-progress".to_string(),
                serde_json::json!({"text": "progress"}),
                AbortSignal::never(),
                Arc::new(move |update| {
                    updates_in_callback.lock().push(update);
                }),
            )
            .await
            .expect("execute with progress");
        let updates = updates.lock();
        assert_eq!(updates.len(), 1, "one ordered progress update");
        assert_eq!(updates[0].details["mcpProgress"]["message"], "halfway");
        let event = pa_agent::types::AgentEvent::ToolExecutionUpdate {
            tool_call_id: "call-progress".to_string(),
            tool_name: tool.name().to_string(),
            args: serde_json::json!({"text": "progress"}),
            partial_result: updates[0].clone(),
        };
        let frame = crate::events::frame(&event);
        assert_eq!(frame["type"], "tool_execution_update");
        assert_eq!(
            frame["partialResult"]["details"]["mcpProgress"]["progress"],
            1.0
        );
    }
}
