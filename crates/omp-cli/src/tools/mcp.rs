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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use omp_mcp::manager::McpManager;
use omp_mcp::names::create_tool_name;
use omp_mcp::schema::normalize_schema_for_mcp;
use pa_agent::abort::AbortSignal;
use pa_agent::stream::ToolDefinition;
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
    // Lazy-mode catalog: names + one-line descriptions of everything
    // discovered, independent of what the provider projection withholds.
    *CATALOG.lock() = discovered
        .iter()
        .map(|tool| {
            let description = tool
                .definition
                .description
                .clone()
                .unwrap_or_else(|| format!("MCP tool from {}", tool.server));
            (
                create_tool_name(&tool.server, &tool.definition.name),
                description,
            )
        })
        .collect();
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

/// Lazy MCP loading (opt-in via `tools.mcpLazy: true` or `OMP_MCP_LAZY=1`):
/// discovery and the registry stay exactly as before — every discovered tool
/// is registered and fully executable — but the provider-facing tool list
/// withholds `mcp__*` schemas until the model loads them through
/// [`LoadTools`]. The startup catalog is one compact system-prompt line.
static LAZY: AtomicBool = AtomicBool::new(false);

/// Discovered MCP tools in registration order: `(registry name, description)`.
static CATALOG: parking_lot::Mutex<Vec<(String, String)>> = parking_lot::Mutex::new(Vec::new());

/// Registry names whose schemas the model has requested via `load_tools`.
static LOADED: parking_lot::Mutex<Vec<String>> = parking_lot::Mutex::new(Vec::new());

/// Record whether this run opted into lazy MCP. Called once by the agent
/// builder; `false` (the default) leaves every behavior unchanged.
pub fn set_lazy(active: bool) {
    LAZY.store(active, Ordering::Relaxed);
    // A newly built agent starts a fresh projection lifecycle. Model switches
    // rediscover tools without calling this, preserving deliberate loads.
    LOADED.lock().clear();
}

/// Whether lazy MCP loading is active for the current run.
#[must_use]
pub fn lazy_active() -> bool {
    LAZY.load(Ordering::Relaxed)
}

/// Opt-in probe: `OMP_MCP_LAZY=1` or the `tools.mcpLazy` setting. An
/// explicit `--tools <csv>` list wins over lazy loading, exactly like it
/// wins over the `tools.profile` projections (checked by the caller).
#[must_use]
pub fn lazy_requested(settings: &omp_settings::Settings) -> bool {
    std::env::var("OMP_MCP_LAZY").is_ok_and(|v| v == "1")
        || settings.get_bool("tools.mcpLazy").unwrap_or(false)
}

/// Provider-side projection for lazy MCP: drop not-yet-loaded `mcp__*`
/// schemas from the serialized tool list. The registry keeps the full
/// persistent set, so eval-bridge execution and post-load structured calls
/// are unaffected.
pub fn project_lazy(tools: &mut Vec<ToolDefinition>) {
    if !lazy_active() {
        return;
    }
    let loaded = LOADED.lock();
    tools.retain(|tool| {
        !tool.name.starts_with("mcp__") || loaded.iter().any(|name| name == &tool.name)
    });
}

/// Whether the current provider request includes this tool's full schema.
/// Kept in sync with [`project_lazy`] so startup context accounting uses the
/// same projection as provider serialization.
#[must_use]
pub fn wire_includes(name: &str) -> bool {
    !lazy_active()
        || !name.starts_with("mcp__")
        || LOADED.lock().iter().any(|loaded| loaded == name)
}

/// Compact startup catalog: every discovered MCP tool as
/// `name — first description sentence`, with the load instruction.
#[must_use]
pub fn catalog_note() -> Option<String> {
    if !lazy_active() {
        return None;
    }
    let catalog = CATALOG.lock();
    if catalog.is_empty() {
        return None;
    }
    let items = catalog
        .iter()
        .map(|(name, description)| {
            let first = description.split(['.', '\n']).next().unwrap_or(description);
            let truncated: String = first.chars().take(80).collect();
            format!("{name} — {truncated}")
        })
        .collect::<Vec<_>>()
        .join("; ");
    Some(format!(
        "MCP tools are lazy. Call `load_tools` with any of these names to expose their full schemas: {items}."
    ))
}

/// The `load_tools` agent tool; registered only under lazy MCP.
#[must_use]
pub fn load_tools_tool() -> Option<Arc<dyn AgentTool>> {
    lazy_active().then(|| {
        Arc::new(LoadTools {
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Registry names to load: `mcp__<server>_<tool>`, or `mcp__<server>` to load every discovered tool of that server.",
                    },
                },
                "required": ["names"],
            }),
        }) as Arc<dyn AgentTool>
    })
}

/// `load_tools`: move selected MCP tools from the compact catalog into the
/// provider tool list. Marks names loaded for [`project_lazy`]; unknown
/// names are reported per entry without failing the batch.
struct LoadTools {
    parameters: Value,
}

impl AgentTool for LoadTools {
    fn name(&self) -> &'static str {
        "load_tools"
    }
    fn description(&self) -> &'static str {
        "Load MCP tools by registry name (mcp__<server>_<tool>, or mcp__<server> for all of a server's tools) so their full schemas are sent to the model from the next request on."
    }
    fn parameters(&self) -> &Value {
        &self.parameters
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        params: Value,
        _signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move {
            let requested = match params.get("names") {
                Some(Value::Array(items)) => items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect::<Vec<_>>(),
                Some(Value::String(csv)) => csv
                    .split(',')
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            };
            if requested.is_empty() {
                let names = CATALOG
                    .lock()
                    .iter()
                    .map(|(name, _)| name.clone())
                    .collect::<Vec<_>>();
                let text = format!(
                    "No names given. Discovered MCP tools: {}.",
                    if names.is_empty() {
                        "(none)".to_string()
                    } else {
                        names.join(", ")
                    }
                );
                return Ok(AgentToolResult {
                    content: vec![ToolResultContent::text(text)],
                    details: serde_json::json!({"loaded": [], "unknown": []}),
                    terminate: None,
                    is_error: None,
                });
            }
            let catalog = CATALOG.lock();
            let mut loaded = LOADED.lock();
            let mut loaded_names = Vec::new();
            let mut unknown = Vec::new();
            for name in requested {
                if catalog.iter().any(|(known, _)| known == &name) {
                    if !loaded_names.contains(&name) {
                        if !loaded.contains(&name) {
                            loaded.push(name.clone());
                        }
                        loaded_names.push(name);
                    }
                } else if name.starts_with("mcp__") {
                    // Server prefix `mcp__<server>`: load every tool of it.
                    let prefix = format!("{name}_");
                    let matches = catalog
                        .iter()
                        .map(|(known, _)| known.clone())
                        .filter(|known| known.starts_with(&prefix))
                        .collect::<Vec<_>>();
                    if matches.is_empty() {
                        unknown.push(name);
                    } else {
                        for known in matches {
                            if !loaded_names.contains(&known) {
                                if !loaded.contains(&known) {
                                    loaded.push(known.clone());
                                }
                                loaded_names.push(known);
                            }
                        }
                    }
                } else {
                    unknown.push(name);
                }
            }
            let mut text = if loaded_names.is_empty() {
                "No MCP tools loaded.".to_string()
            } else {
                format!(
                    "Loaded {} MCP tool(s): {}.",
                    loaded_names.len(),
                    loaded_names.join(", ")
                )
            };
            if !unknown.is_empty() {
                text.push_str(" Unknown (not discovered): ");
                text.push_str(&unknown.join(", "));
                text.push('.');
            }
            Ok(AgentToolResult {
                content: vec![ToolResultContent::text(text)],
                details: serde_json::json!({"loaded": loaded_names, "unknown": unknown}),
                terminate: None,
                is_error: None,
            })
        })
    }
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

    // ----- lazy MCP loading -----

    /// Serializes the lazy tests: they flip process-global LAZY/CATALOG/LOADED.
    static LAZY_TEST_LOCK: std::sync::LazyLock<tokio::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

    fn reset_lazy_state() {
        set_lazy(false);
        CATALOG.lock().clear();
        LOADED.lock().clear();
    }

    /// Fake manager wired to the shared stdio fixture (same as the
    /// registration test above).
    fn fake_manager(project: &std::path::Path, script: &std::path::Path) -> Arc<McpManager> {
        let config: omp_mcp::McpServerConfig = serde_json::from_value(serde_json::json!({
            "command": "python3", "args": [script.to_string_lossy()], "timeout": 15000
        }))
        .expect("server config");
        let configs = std::iter::once(("fake".to_string(), config)).collect();
        Arc::new(McpManager::new(Arc::new(project.to_path_buf()), configs))
    }

    fn def(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: String::new(),
            parameters: serde_json::json!({"type": "object"}),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn lazy_projection_withholds_until_load_tools_then_executes() {
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("python3 not available; skipping");
            return;
        }
        let _guard = LAZY_TEST_LOCK.lock().await;
        reset_lazy_state();
        let root = tempfile::tempdir().expect("tempdir");
        let script = root.path().join("fake.py");
        std::fs::File::create(&script)
            .unwrap()
            .write_all(FAKE_SERVER_PY.as_bytes())
            .unwrap();
        set_lazy(true);

        // Discovery registers the full persistent list and fills the catalog.
        let tools = register_tools(fake_manager(root.path(), &script)).await;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "mcp__fake_echo");
        assert_eq!(
            CATALOG.lock().as_slice(),
            [("mcp__fake_echo".to_string(), "Echo args".to_string())]
        );
        assert!(load_tools_tool().is_some(), "lazy on registers load_tools");
        assert!(catalog_note().is_some());
        assert!(!wire_includes("mcp__fake_echo"));
        assert!(wire_includes("read"));

        // Provider projection withholds unloaded MCP schemas, keeps built-ins.
        let mut provider_tools = vec![def("mcp__fake_echo"), def("read")];
        project_lazy(&mut provider_tools);
        assert_eq!(
            provider_tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            ["read"],
            "unloaded mcp__* withheld, built-ins kept"
        );

        // load_tools by server prefix marks the tool loaded.
        let loader = load_tools_tool().expect("loader");
        let result = loader
            .execute(
                "load-1".to_string(),
                serde_json::json!({"names": ["mcp__fake"]}),
                AbortSignal::never(),
                Arc::new(|_| {}),
            )
            .await
            .expect("load_tools execute");
        assert_eq!(
            result.details["loaded"],
            serde_json::json!(["mcp__fake_echo"])
        );
        assert_eq!(result.details["unknown"], serde_json::json!([]));
        let text = match &result.content[0] {
            ToolResultContent::Text(t) => t.text.clone(),
            ToolResultContent::Image(_) => panic!("unexpected image"),
        };
        assert!(text.contains("mcp__fake_echo"), "result text: {text}");

        // After loading, the full schema is serialized: structured calls work.
        let mut provider_tools = vec![def("mcp__fake_echo"), def("read")];
        project_lazy(&mut provider_tools);
        assert_eq!(provider_tools.len(), 2, "loaded tool projected");
        assert!(wire_includes("mcp__fake_echo"));

        // The persistent registry still executes the tool end to end.
        let executed = tools[0]
            .clone()
            .execute(
                "call-e2e".to_string(),
                serde_json::json!({"text": "E2E"}),
                AbortSignal::never(),
                Arc::new(|_| {}),
            )
            .await
            .expect("execute after load");
        let executed_text = match &executed.content[0] {
            ToolResultContent::Text(t) => t.text.clone(),
            ToolResultContent::Image(_) => panic!("unexpected image"),
        };
        assert!(executed_text.contains("E2E"));

        reset_lazy_state();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn load_tools_reports_unknown_and_is_idempotent() {
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("python3 not available; skipping");
            return;
        }
        let _guard = LAZY_TEST_LOCK.lock().await;
        reset_lazy_state();
        let root = tempfile::tempdir().expect("tempdir");
        let script = root.path().join("fake.py");
        std::fs::File::create(&script)
            .unwrap()
            .write_all(FAKE_SERVER_PY.as_bytes())
            .unwrap();
        set_lazy(true);
        register_tools(fake_manager(root.path(), &script)).await;

        let loader = load_tools_tool().expect("loader");
        // Unknown names are reported without failing the batch.
        let miss = loader
            .clone()
            .execute(
                "load-miss".to_string(),
                serde_json::json!({"names": ["mcp__ghost", "read"]}),
                AbortSignal::never(),
                Arc::new(|_| {}),
            )
            .await
            .expect("execute");
        assert_eq!(miss.details["loaded"], serde_json::json!([]));
        assert_eq!(
            miss.details["unknown"],
            serde_json::json!(["mcp__ghost", "read"])
        );
        assert!(LOADED.lock().is_empty());

        // Direct name + repeat stays single-entry (idempotent).
        let hit = loader
            .clone()
            .execute(
                "load-hit".to_string(),
                serde_json::json!({"names": ["mcp__fake_echo", "mcp__fake_echo"]}),
                AbortSignal::never(),
                Arc::new(|_| {}),
            )
            .await
            .expect("execute");
        assert_eq!(hit.details["loaded"], serde_json::json!(["mcp__fake_echo"]));
        assert_eq!(LOADED.lock().as_slice(), ["mcp__fake_echo"]);

        // Empty request lists the catalog instead of loading anything.
        let none = loader
            .execute(
                "load-none".to_string(),
                serde_json::json!({}),
                AbortSignal::never(),
                Arc::new(|_| {}),
            )
            .await
            .expect("execute");
        assert!(match &none.content[0] {
            ToolResultContent::Text(t) => t.text.contains("Discovered MCP tools"),
            ToolResultContent::Image(_) => false,
        });
        assert_eq!(LOADED.lock().as_slice(), ["mcp__fake_echo"]);

        reset_lazy_state();
    }

    #[test]
    fn lazy_off_keeps_projection_and_tool_registration_noop() {
        let _guard = LAZY_TEST_LOCK.blocking_lock();
        reset_lazy_state();
        let mut provider_tools = vec![def("mcp__fake_echo"), def("read")];
        project_lazy(&mut provider_tools);
        assert_eq!(provider_tools.len(), 2, "lazy off: nothing withheld");
        assert!(wire_includes("mcp__fake_echo"));
        assert!(load_tools_tool().is_none(), "lazy off: no load_tools");
        assert!(catalog_note().is_none(), "lazy off: no catalog note");
    }

    #[test]
    fn lazy_requested_reads_setting_and_defaults_false() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.yml"),
            "tools:\n  codeMode: false\n  mcpLazy: true\n",
        )
        .unwrap();
        let settings = omp_settings::Settings::load_from(None, dir.path());
        assert!(lazy_requested(&settings), "opt-in via tools.mcpLazy");
        let empty = omp_settings::Settings::load_from(None, std::path::Path::new("/"));
        assert!(!lazy_requested(&empty), "default off without opt-in");
    }
}
