//! WASM plugin registration: bridges `omp-wasm` discovered plugins into the
//! agent tool registry as `<plugin>_<tool>` tools (the same mechanism MCP
//! tools use) and provides `/plugins` listing + `/<plugin>:<command>` slash
//! dispatch helpers.
//!
//! Discovery is cached per working directory (first call wins) so the agent
//! tool list, `/plugins`, and slash dispatch all agree on the same registry.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use omp_wasm::{CallKind, Discovery};
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback, ToolResultContent};
use pa_agent::BoxFut;
use serde_json::Value;

/// Cache of the most recently discovered registry, keyed by absolute cwd.
static REGISTRY: Mutex<Option<(PathBuf, Arc<Discovery>)>> = Mutex::new(None);

fn discover_once(cwd: &Path) -> Arc<Discovery> {
    let mut guard = REGISTRY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((cached, registry)) = guard.as_ref() {
        if cached == cwd {
            return registry.clone();
        }
    }
    let registry = Arc::new(omp_wasm::discover(cwd));
    *guard = Some((cwd.to_path_buf(), registry.clone()));
    registry
}

/// Discover WASM plugins for `cwd` and wrap each manifest tool as an agent
/// tool named `<plugin>_<tool>`.
#[must_use]
pub fn wasm_tools(cwd: &Path) -> Vec<Arc<dyn AgentTool>> {
    let registry = discover_once(cwd);
    let mut tools: Vec<Arc<dyn AgentTool>> = Vec::new();
    for entry in &registry.plugins {
        let module = entry.module.clone();
        let plugin_name = module.name().to_string();
        for tool in &module.manifest().tools {
            tools.push(Arc::new(WasmTool {
                plugin: module.clone(),
                name: format!("{plugin_name}_{}", tool.name),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                tool_name: tool.name.clone(),
            }));
        }
    }
    tools
}

/// Text for the `/plugins` command listing WASM plugins (name, version,
/// tools, commands, path, and load errors).
pub fn plugins_list_text(cwd: &Path) -> String {
    let registry = discover_once(cwd);
    let mut lines = Vec::new();
    for entry in &registry.plugins {
        let module = &entry.module;
        let manifest = module.manifest();
        let tools = if manifest.tools.is_empty() {
            "-".to_string()
        } else {
            manifest
                .tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let commands = if manifest.commands.is_empty() {
            "-".to_string()
        } else {
            manifest
                .commands
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        lines.push(format!(
            "  {} v{} [wasm]\n    path: {}\n    tools: {}\n    commands: {}",
            manifest.name,
            manifest.version,
            module.path().display(),
            tools,
            commands,
        ));
    }
    for (path, error) in &registry.failures {
        lines.push(format!("  {} [load error]: {error}", path.display()));
    }
    if lines.is_empty() {
        "No WASM plugins found (look for ~/.omp/agent/plugins/*.wasm and .omp/plugins/*.wasm)."
            .to_string()
    } else {
        format!("WASM plugins:\n{}", lines.join("\n"))
    }
}

/// Render the command catalog entries for WASM plugin commands
/// (`/<plugin>:<command>`).
pub fn command_catalog(cwd: &Path) -> Vec<Value> {
    let registry = discover_once(cwd);
    let mut out = Vec::new();
    for entry in &registry.plugins {
        let module = entry.module.clone();
        let plugin_name = module.name().to_string();
        for command in &module.manifest().commands {
            out.push(serde_json::json!({
                "name": format!("{plugin_name}:{}", command.name),
                "description": if command.description.is_empty() {
                    format!("WASM plugin command from {plugin_name}")
                } else {
                    command.description.clone()
                },
                "input": {"hint": "arguments"},
                "source": "wasm",
            }));
        }
    }
    out
}

/// Dispatch `/<plugin>:<command> args` for a WASM plugin. `plugin` and
/// `args` come from `parse_slash_command` (which already split at the first
/// colon), so `args` is `"<command> <args>"`. Returns `Some(rendered output)`
/// when the plugin + command exist.
pub async fn try_command(cwd: &Path, plugin: &str, args: &str) -> Option<String> {
    let registry = discover_once(cwd);
    let entry = registry
        .plugins
        .iter()
        .find(|e| e.module.name() == plugin)?;
    let module = &entry.module;
    let (command, rest) = match args.split_once(char::is_whitespace) {
        Some((cmd, rest)) => (cmd, rest.trim()),
        None => (args, ""),
    };
    if !module.manifest().commands.iter().any(|c| c.name == command) {
        return None;
    }
    let request_args = serde_json::json!(rest);
    let command = command.to_owned();
    let module = module.clone();
    // Same off-executor rule as the tool path above.
    let result = tokio::task::spawn_blocking(move || {
        module.call(CallKind::Command, &command, &request_args)
    })
    .await;
    let result = match result {
        Ok(result) => result,
        Err(join_error) => Err(anyhow::anyhow!("plugin task failed: {join_error}")),
    };
    match result {
        Ok(out) => Some(if out.ok {
            out.text
        } else {
            format!(
                "Plugin command error: {}",
                out.error.unwrap_or_else(|| "see plugin output".into())
            )
        }),
        Err(error) => Some(format!("WASM plugin error: {error:#}")),
    }
}

/// One registered WASM plugin tool. Executing a call instantiates the plugin
/// fresh and invokes `omt_call` with `{"kind":"tool","name":...,"args":...}`.
struct WasmTool {
    plugin: Arc<omp_wasm::WasmModule>,
    name: String,
    description: String,
    parameters: Value,
    tool_name: String,
}

impl AgentTool for WasmTool {
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
            let args = match params {
                Value::Object(_) => params,
                _ => Value::Object(serde_json::Map::new()),
            };
            // `call` blocks on a dedicated plugin thread (up to the wall
            // timeout); keep it off the async executor with spawn_blocking.
            let plugin = self.plugin.clone();
            let tool_name = self.tool_name.clone();
            let call =
                tokio::task::spawn_blocking(move || plugin.call(CallKind::Tool, &tool_name, &args))
                    .await;
            let result = match call {
                Ok(result) => result,
                Err(join_error) => Err(anyhow::anyhow!("plugin task failed: {join_error}")),
            };
            match result {
                Ok(out) if out.ok => Ok(AgentToolResult {
                    content: vec![ToolResultContent::text(out.text)],
                    details: serde_json::json!({
                        "plugin": self.plugin.name(),
                        "tool": self.tool_name,
                        "ok": true,
                    }),
                    terminate: None,
                    is_error: None,
                }),
                Ok(out) => Ok(AgentToolResult {
                    content: vec![ToolResultContent::text(format!(
                        "Plugin error: {}",
                        out.error.unwrap_or_else(|| "see plugin output".into())
                    ))],
                    details: serde_json::json!({
                        "plugin": self.plugin.name(),
                        "tool": self.tool_name,
                        "ok": false,
                    }),
                    terminate: None,
                    is_error: Some(true),
                }),
                Err(error) => Ok(AgentToolResult {
                    content: vec![ToolResultContent::text(format!(
                        "WASM plugin error: {error:#}"
                    ))],
                    details: serde_json::json!({
                        "plugin": self.plugin.name(),
                        "tool": self.tool_name,
                        "ok": false,
                    }),
                    terminate: None,
                    is_error: Some(true),
                }),
            }
        })
    }
}
