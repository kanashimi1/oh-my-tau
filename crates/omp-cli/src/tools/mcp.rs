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
/// This performs blocking network/subprocess I/O; call from a sync context
/// within (or outside) a Tokio runtime.
#[must_use]
pub fn mcp_tools(cwd: &Path) -> Vec<Arc<dyn AgentTool>> {
    block_on_discover(cwd).unwrap_or_default()
}

/// Servers connected by the latest discovery, with their tool counts (TS
/// `MCPManager.getConnectedServers` + `getConnection(name).tools.length`).
static CONNECTED_SERVERS: std::sync::Mutex<Vec<(String, usize)>> =
    std::sync::Mutex::new(Vec::new());

/// `(server, tool count)` for every MCP server connected by the latest discovery.
#[must_use]
pub fn connected_servers() -> Vec<(String, usize)> {
    CONNECTED_SERVERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

#[allow(clippy::unnecessary_wraps)]
fn block_on_discover(cwd: &Path) -> Option<Vec<Arc<dyn AgentTool>>> {
    let cwd = cwd.to_path_buf();
    let discover = move || async move {
        let manager = Arc::new(McpManager::from_default_configs(&cwd));
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
        *CONNECTED_SERVERS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = connected;
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
    };
    #[allow(clippy::result_large_err)]
    match tokio::runtime::Handle::try_current() {
        // `block_in_place` panics on a current-thread runtime; fall through
        // to the shared fallback runtime there (rpc/print modes).
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            Some(tokio::task::block_in_place(|| handle.block_on(discover())))
        }
        // `block_in_place` and in-place `block_on` both panic or deadlock on
        // a current-thread runtime; run discovery on the shared fallback
        // runtime's own thread (rpc/print modes).
        Ok(_current_thread) => {
            let (tx, rx) = std::sync::mpsc::channel::<Option<Vec<Arc<dyn AgentTool>>>>();
            std::thread::spawn(move || {
                let result: Option<Vec<Arc<dyn AgentTool>>> =
                    Some(fallback_runtime().block_on(discover()));
                let _ = tx.send(result);
            });
            rx.recv().unwrap_or(None)
        }
        // No ambient runtime (sync host): run discovery on a shared
        // process-lifetime runtime so the transports' spawned tasks (stdio
        // read loop, subprocess pipes) outlive this call — MCP tools are
        // invoked long after registration.
        Err(_no_runtime) => Some(fallback_runtime().block_on(discover())),
    }
}

/// Process-lifetime runtime used when the host has no Tokio context.
fn fallback_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: std::sync::LazyLock<tokio::runtime::Runtime> = std::sync::LazyLock::new(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("MCP runtime")
    });
    &RUNTIME
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
        _on_update: AgentToolUpdateCallback,
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
            match self.manager.call(&self.server, &self.tool_name, args).await {
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
    #[test]
    fn mcp_tools_register_with_omp_names_and_schemas() {
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("python3 not available; skipping");
            return;
        }
        let root = tempfile::tempdir().expect("tempdir");
        let empty_user = root.path().join("usercfg");
        std::fs::create_dir_all(&empty_user).unwrap();
        let project = root.path().join("project");
        std::fs::create_dir_all(&project).unwrap();
        let script = project.join("fake.py");
        std::fs::File::create(&script)
            .unwrap()
            .write_all(FAKE_SERVER_PY.as_bytes())
            .unwrap();
        std::fs::write(
            project.join("mcp.json"),
            serde_json::json!({"mcpServers": {"fake": {
                "command": "python3", "args": [script.to_string_lossy()], "timeout": 15000
            }}})
            .to_string(),
        )
        .unwrap();

        std::env::set_var("OMP_CONFIG_DIR", &empty_user);
        let tools = mcp_tools(&project);
        std::env::remove_var("OMP_CONFIG_DIR");

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

        // Execute through the AgentTool surface.
        let result = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                tool.clone()
                    .execute(
                        "call-1".to_string(),
                        serde_json::json!({"text": "hi"}),
                        AbortSignal::never(),
                        Arc::new(|_| {}),
                    )
                    .await
                    .expect("execute")
            });
        let text = match &result.content[0] {
            ToolResultContent::Text(t) => t.text.clone(),
            ToolResultContent::Image(_) => panic!("unexpected content: image"),
        };
        assert!(text.contains("\"text\": \"hi\""), "result text: {text}");
        assert_eq!(result.details["serverName"], serde_json::json!("fake"));
        assert_eq!(result.details["toolName"], serde_json::json!("echo"));
        assert_eq!(result.details["isError"], serde_json::json!(false));
    }
}
