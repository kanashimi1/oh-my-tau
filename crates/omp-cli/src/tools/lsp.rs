//! `lsp`: symbol-aware code intelligence from language servers. Port of omp
//! `lsp/tool.ts` (`v18.6.1`): diagnostics, definition/references/hover/symbols,
//! rename + `rename_file`, code actions, capabilities, status, reload, raw
//! request. The machinery lives in the `omp-lsp` crate.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use omp_lsp::client::{self, LspError};
use omp_lsp::Cancel;
use pa_agent::abort::{is_abort_error, AbortSignal};
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use pa_agent::BoxFut;
use serde::Deserialize;
use serde_json::{json, Value};

use super::Env;

/// omp `prompts/tools/lsp.md`, byte-identical.
pub const DESCRIPTION: &str = r#"Symbol-aware code intelligence from language servers — navigation, refactors, and diagnostics where text tools miss callsites.

<operations>
- Position-based: `file` + `line` + `symbol` (substring; `#N` for Nth match). `line` is 1-indexed.
- `rename` — applies by default; `apply: false` previews. Project-aware lookups ERROR without `symbol` — no silent fallback on missing/ambiguous matches.
- `code_actions` — lists by default; apply ONE with `apply: true` + `query` (title substring or index).
- `rename_file` — moves file AND rewrites all imports/references; applies by default.
- `diagnostics` — path, glob (`src/**/*.ts`), or `file: "*"` for workspace.
- `symbols` — `file` lists file symbols; `file: "*"` + `query` searches workspace.
- `reload` — restart one server (`file`) or all (`*`); `reload *` re-reads LSP config.
- `request` — raw: `query` = method, `payload` = JSON params (else auto-built).
</operations>

<critical>
- Symbol-aware work (rename, references, definition, code actions) MUST use `lsp` whenever a server is available.
  It follows shadowing, re-exports, and cross-file usages text tools miss.
- NEVER do a cross-file rename with `ast_edit`/`sed`/hand edits when `lsp` `rename`/`rename_file` can — text renames silently drop callsites.
- Reach for `code_actions` on imports, quick-fixes, and server-known refactors before editing by hand.
</critical>"#;

/// Parameter schema, byte-equal (as JSON) to omp's `lspSchema`.
#[must_use]
pub fn schema() -> Value {
    json!({"type":"object","properties":{
        "action":{"enum":["diagnostics","definition","references","hover","symbols","rename","rename_file","code_actions","type_definition","implementation","status","reload","capabilities","request"],"type":"string"},
        "file":{"type":"string"},
        "line":{"type":"number"},
        "symbol":{"type":"string"},
        "query":{"type":"string"},
        "new_name":{"type":"string"},
        "apply":{"type":"boolean"},
        "timeout":{"type":"number","minimum":5,"maximum":300,"description":"Timeout in seconds (default 20; range 5–300)."},
        "payload":{"type":"string"}
    },"required":["action"],"additionalProperties":false})
}

/// Tool timeout config (omp `TOOL_TIMEOUTS.lsp`).
const TIMEOUT_DEFAULT_SECS: u64 = 20;
const TIMEOUT_MIN_SECS: u64 = 5;
const TIMEOUT_MAX_SECS: u64 = 300;

const MAX_RENAME_PAIRS: usize = 1000;

/// The `lsp` tool, bound to the session working directory.
#[must_use]
pub fn tool(env: Arc<Env>) -> Arc<dyn AgentTool> {
    Arc::new(LspTool {
        schema: schema(),
        env,
    })
}

struct LspTool {
    schema: Value,
    env: Arc<Env>,
}

/// Combined caller-cancel + wall-clock deadline.
struct DeadlineCancel<'a> {
    abort: &'a AbortSignal,
    deadline: Instant,
}

impl omp_lsp::Cancel for DeadlineCancel<'_> {
    fn is_cancelled(&self) -> bool {
        self.abort.is_aborted() || Instant::now() >= self.deadline
    }
    fn cancelled(&self) -> std::pin::Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let sleep = tokio::time::sleep_until(self.deadline.into());
        Box::pin(async move {
            tokio::select! {
                () = self.abort.aborted() => {}
                () = sleep => {}
            }
        })
    }
}

impl AgentTool for LspTool {
    fn name(&self) -> &'static str {
        "lsp"
    }
    fn description(&self) -> &str {
        DESCRIPTION
    }
    fn parameters(&self) -> &Value {
        &self.schema
    }
    fn label(&self) -> &'static str {
        "LSP"
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        params: Value,
        signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move { self.run(params, &signal).await })
    }
}

#[derive(Debug, Deserialize)]
struct Params {
    action: String,
    file: Option<String>,
    line: Option<u32>,
    symbol: Option<String>,
    query: Option<String>,
    new_name: Option<String>,
    apply: Option<bool>,
    timeout: Option<u64>,
    payload: Option<String>,
}

impl Params {
    fn request_json(&self) -> Value {
        json!({
            "action": self.action,
            "file": self.file,
            "line": self.line,
            "symbol": self.symbol,
            "query": self.query,
            "new_name": self.new_name,
            "apply": self.apply,
            "timeout": self.timeout,
            "payload": self.payload,
        })
    }
}

fn text_result(text: impl Into<String>, details: Value) -> AgentToolResult {
    AgentToolResult {
        content: vec![pa_agent::types::ToolResultContent::text(text.into())],
        details,
        terminate: None,
        is_error: None,
    }
}

impl LspTool {
    async fn run(self: &Arc<Self>, raw: Value, signal: &AbortSignal) -> Result<AgentToolResult> {
        // Broker-shared language servers (`lsp.shared`, default on, gated by
        // `lsp.enabled`): re-applied on every tool call like omp's settings
        // listener, consulted only at client cold-start, so running clients
        // keep their transport until they exit or idle out.
        let settings = omp_settings::Settings::load_from(None, &self.env.cwd);
        let shared = settings.get_bool("lsp.enabled").unwrap_or(true)
            && settings.get_bool("lsp.shared").unwrap_or(true);
        omp_lsp::mux::set_shared_lsp_enabled(shared);

        let params: Params =
            serde_json::from_value(raw).map_err(|e| anyhow!("Invalid lsp arguments: {e}"))?;
        let timeout_secs = params
            .timeout
            .unwrap_or(TIMEOUT_DEFAULT_SECS)
            .clamp(TIMEOUT_MIN_SECS, TIMEOUT_MAX_SECS);
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        let cwd = self.env.cwd.clone();
        let cancel = DeadlineCancel {
            abort: signal,
            deadline,
        };

        let result = self
            .dispatch(&params, &cwd, &cancel, signal, timeout_secs)
            .await;
        match result {
            Ok(r) => Ok(r),
            Err(e) => {
                if is_abort_error(&e) {
                    return Err(e);
                }
                // Distinguish the wall-clock budget from a caller cancel.
                if signal.is_aborted() {
                    return Err(e);
                }
                Err(e)
            }
        }
    }

    #[allow(clippy::too_many_lines)] // TS-parity per-action dispatch; splitting would diverge from omp tool.ts
    async fn dispatch(
        self: &Arc<Self>,
        params: &Params,
        cwd: &Path,
        cancel: &DeadlineCancel<'_>,
        signal: &AbortSignal,
        timeout_secs: u64,
    ) -> Result<AgentToolResult> {
        let action = params.action.as_str();
        let config = omp_lsp::config::get_config(cwd);

        // Status doesn't need a file.
        if action == "status" {
            let configured: Vec<&String> = config.servers.keys().collect();
            let started = client::get_active_clients();
            let mut lines: Vec<String> = Vec::new();
            if configured.is_empty() {
                lines.push("No language servers configured for this project".to_string());
            } else {
                let labelled: Vec<String> = configured
                    .iter()
                    .map(|name| {
                        let cfg = &config.servers[*name];
                        match started.iter().find(|c| c.name == cfg.command) {
                            Some(s) => format!("{} ({:?})", name, s.status),
                            None => format!("{name} (configured, not started)"),
                        }
                    })
                    .collect();
                lines.push(format!("Language servers: {}", labelled.join(", ")));
                lines.push(
                    "  note: 'configured, not started' means the binary resolves on PATH but no request has spawned it yet; 'ready' means a client process is live for this cwd.".to_string(),
                );
            }
            return Ok(text_result(
                lines.join("\n"),
                json!({"action": action, "success": true, "request": params.request_json()}),
            ));
        }

        if action == "diagnostics" {
            return self
                .action_diagnostics(params, cwd, cancel, timeout_secs)
                .await;
        }

        if action == "rename_file" {
            return self.action_rename_file(params, cwd, cancel).await;
        }

        if action == "capabilities" {
            let server_list: Vec<(String, omp_lsp::types::ServerConfig)> = match &params.file {
                Some(file) if file != "*" => {
                    let resolved = omp_lsp::utils::resolve_to_cwd(file, cwd);
                    let list = omp_lsp::servers::get_lsp_servers_for_file(&config, &resolved);
                    if list.is_empty() {
                        return Ok(text_result(
                            "No language server found for this file",
                            json!({"action": action, "success": false, "request": params.request_json()}),
                        ));
                    }
                    list
                }
                _ => omp_lsp::servers::get_lsp_servers(&config),
            };
            if server_list.is_empty() {
                return Ok(text_result(
                    "No language servers configured",
                    json!({"action": action, "success": false, "request": params.request_json()}),
                ));
            }
            let mut sections: Vec<String> = Vec::new();
            let mut responding: Vec<String> = Vec::new();
            for (server_name, server_config) in &server_list {
                if cancel.is_cancelled() {
                    return Err(abort_err());
                }
                match client::get_or_create_client(server_config, cwd, None, Some(cancel)).await {
                    Ok(c) => {
                        responding.push(server_name.clone());
                        let caps = c.server_capabilities();
                        let pretty = serde_json::to_string_pretty(&caps)
                            .unwrap_or_else(|_| caps.to_string());
                        sections.push(format!("{server_name}:"));
                        sections.push(format!("  capabilities: {}", pretty.replace('\n', "\n  ")));
                    }
                    Err(LspError {
                        message,
                        cancelled: true,
                    }) => {
                        return Err(if signal.is_aborted() {
                            abort_err()
                        } else {
                            anyhow!(message)
                        })
                    }
                    Err(e) => {
                        sections.push(format!("{server_name}: failed to start ({})", e.message))
                    }
                }
            }
            return Ok(text_result(
                sections.join("\n"),
                json!({"action": action, "serverName": responding.join(", "), "success": true, "request": params.request_json()}),
            ));
        }

        if action == "request" {
            return self.action_request(params, cwd, cancel, signal).await;
        }

        // `*` means workspace scope for symbols/reload; others need a concrete file.
        let is_workspace = params.file.as_deref() == Some("*");
        let requires_file = params.file.is_none() && action != "reload";
        if requires_file {
            return Ok(text_result(
                "Error: file parameter required. Use `*` for workspace scope where supported.",
                json!({"action": action, "success": false}),
            ));
        }
        let resolved_file: Option<PathBuf> = match &params.file {
            Some(f) if !is_workspace => Some(omp_lsp::utils::resolve_to_cwd(f, cwd)),
            _ => None,
        };

        if action == "symbols" && (is_workspace || resolved_file.is_none()) {
            return self
                .action_workspace_symbols(params, cwd, cancel, signal)
                .await;
        }

        if action == "reload" && (is_workspace || resolved_file.is_none()) {
            return self.action_reload_all(params, cwd, cancel, signal).await;
        }

        let Some(resolved_file) = resolved_file else {
            return Ok(text_result(
                "No language server found for this action",
                json!({"action": action, "success": false}),
            ));
        };
        let Some((server_name, server_config)) =
            omp_lsp::servers::get_lsp_server_for_file(&config, &resolved_file)
        else {
            return Ok(text_result(
                "No language server found for this action",
                json!({"action": action, "success": false}),
            ));
        };

        if action == "reload" {
            omp_lsp::client::clear_initialization_failure(&server_config, cwd);
        }

        match self
            .action_file_based(
                action,
                params,
                cwd,
                &resolved_file,
                &server_name,
                &server_config,
                cancel,
                timeout_secs,
            )
            .await
        {
            Ok(r) => Ok(r),
            Err(e) => Err(map_action_error(e, action, &server_name, signal)),
        }
    }
}

fn abort_err() -> anyhow::Error {
    pa_agent::abort::aborted_error()
}

fn map_action_error(
    e: anyhow::Error,
    action: &str,
    server_name: &str,
    signal: &AbortSignal,
) -> anyhow::Error {
    if let Some(lsp_err) = e.downcast_ref::<LspError>() {
        if lsp_err.cancelled {
            return if signal.is_aborted() {
                abort_err()
            } else {
                anyhow!("LSP {action} timed out or was cancelled on {server_name}")
            };
        }
        return anyhow!("LSP error: {}", lsp_err.message);
    }
    e
}

impl LspTool {
    /// `diagnostics`: single file, glob batch, or `*` workspace-wide.
    #[allow(clippy::too_many_lines)] // TS-parity per-action dispatch; splitting would diverge from omp tool.ts
    async fn action_diagnostics(
        self: &Arc<Self>,
        params: &Params,
        cwd: &Path,
        cancel: &DeadlineCancel<'_>,
        timeout_secs: u64,
    ) -> Result<AgentToolResult> {
        use omp_lsp::diagnostics as diag;
        let action = "diagnostics";
        let Some(file) = params.file.clone() else {
            return Ok(text_result(
                "Error: file parameter required. Use `*` for workspace-wide diagnostics or a path/glob for specific files.",
                json!({"action": action, "success": false, "request": params.request_json()}),
            ));
        };

        if file == "*" {
            return self.action_workspace_diagnostics(params, cwd, cancel).await;
        }

        let (targets, truncated_glob) = match diag::resolve_diagnostic_targets(
            &file,
            cwd,
            diag::MAX_GLOB_DIAGNOSTIC_TARGETS,
        )
        .await
        {
            Ok(t) => t,
            Err(e) => {
                return Ok(text_result(
                    format!("Error: {e}"),
                    json!({"action": action, "success": false, "request": params.request_json()}),
                ))
            }
        };
        if targets.is_empty() {
            return Ok(text_result(
                format!("No files matched pattern: {file}"),
                json!({"action": action, "success": true, "request": params.request_json()}),
            ));
        }

        let config = omp_lsp::config::get_config(cwd);
        let detailed = targets.len() > 1 || truncated_glob;
        let mut results: Vec<String> = Vec::new();
        let mut all_server_names: Vec<String> = Vec::new();
        let mut total_attempts = 0usize;
        let mut total_successes = 0usize;
        if truncated_glob {
            results.push(format!(
                "⚠ Pattern matched more than {} files; showing first {}. Narrow the glob or use workspace diagnostics.",
                diag::MAX_GLOB_DIAGNOSTIC_TARGETS,
                diag::MAX_GLOB_DIAGNOSTIC_TARGETS
            ));
        }

        for target in &targets {
            if cancel.is_cancelled() {
                return Err(pa_agent::abort::aborted_error());
            }
            let resolved = omp_lsp::utils::resolve_to_cwd(target, cwd);
            let servers = omp_lsp::config::get_servers_for_file(&config, &resolved);
            if servers.is_empty() {
                results.push(format!("✘ {target}: No language server found"));
                continue;
            }
            let uri = omp_lsp::utils::file_to_uri(&resolved);
            let rel_path = omp_lsp::utils::format_path_relative_to_cwd(&resolved, cwd);
            let mut all_diagnostics: Vec<omp_lsp::types::Diagnostic> = Vec::new();
            let mut failed_servers: Vec<String> = Vec::new();
            let mut succeeded = 0usize;

            for (server_name, server_config) in &servers {
                if !all_server_names.contains(server_name) {
                    all_server_names.push(server_name.clone());
                }
                total_attempts += 1;
                match self
                    .diagnostics_from_server(
                        &resolved,
                        &uri,
                        server_config,
                        detailed,
                        cancel,
                        timeout_secs,
                    )
                    .await
                {
                    Ok(mut ds) => {
                        all_diagnostics.append(&mut ds);
                        succeeded += 1;
                        total_successes += 1;
                    }
                    Err(LspError {
                        cancelled: true, ..
                    }) => return Err(pa_agent::abort::aborted_error()),
                    Err(e) => {
                        failed_servers.push(server_name.clone());
                        tracing::debug!(
                            "LSP diagnostics server failed: {server_name} {rel_path}: {}",
                            e.message
                        );
                    }
                }
            }

            // Deduplicate by position + message.
            let mut seen = std::collections::HashSet::new();
            let mut unique: Vec<omp_lsp::types::Diagnostic> = Vec::new();
            for d in all_diagnostics {
                let key = format!(
                    "{}:{}:{}:{}:{}",
                    d.range.start.line,
                    d.range.start.character,
                    d.range.end.line,
                    d.range.end.character,
                    d.message
                );
                if seen.insert(key) {
                    unique.push(d);
                }
            }
            omp_lsp::utils::sort_diagnostics(&mut unique);

            if !detailed && targets.len() == 1 {
                let server_names = all_server_names.join(", ");
                if succeeded == 0 {
                    return Ok(text_result(
                        format!(
                            "✘ {rel_path}: all language servers failed ({})",
                            failed_servers.join(", ")
                        ),
                        json!({"action": action, "serverName": server_names, "success": false}),
                    ));
                }
                if unique.is_empty() {
                    let text = if failed_servers.is_empty() {
                        "OK".to_string()
                    } else {
                        format!("OK\n⚠ some servers failed: {}", failed_servers.join(", "))
                    };
                    return Ok(text_result(
                        text,
                        json!({"action": action, "serverName": server_names, "success": true}),
                    ));
                }
                let summary = omp_lsp::utils::format_diagnostics_summary(&unique);
                let formatted: Vec<String> = unique
                    .iter()
                    .map(|d| omp_lsp::utils::format_diagnostic(d, &rel_path))
                    .collect();
                let mut output = format!(
                    "{summary}:\n{}",
                    omp_lsp::utils::format_grouped_diagnostic_messages(&formatted)
                );
                if !failed_servers.is_empty() {
                    output.push_str("\n⚠ some servers failed: ");
                    output.push_str(&failed_servers.join(", "));
                }
                return Ok(text_result(
                    output,
                    json!({"action": action, "serverName": server_names, "success": true}),
                ));
            }

            if unique.is_empty() {
                if succeeded == 0 {
                    results.push(format!(
                        "✘ {rel_path}: all language servers failed ({})",
                        failed_servers.join(", ")
                    ));
                } else {
                    results.push(format!("✔ {rel_path}: no issues"));
                    if !failed_servers.is_empty() {
                        results.push(format!(
                            "⚠ {rel_path}: some servers failed ({})",
                            failed_servers.join(", ")
                        ));
                    }
                }
            } else {
                let summary = omp_lsp::utils::format_diagnostics_summary(&unique);
                results.push(format!("✘ {rel_path}: {summary}"));
                let formatted: Vec<String> = unique
                    .iter()
                    .map(|d| omp_lsp::utils::format_diagnostic(d, &rel_path))
                    .collect();
                results.push(omp_lsp::utils::format_grouped_diagnostic_messages(
                    &formatted,
                ));
                if !failed_servers.is_empty() {
                    results.push(format!(
                        "⚠ {rel_path}: some servers failed ({})",
                        failed_servers.join(", ")
                    ));
                }
            }
        }

        let all_failed = total_attempts > 0 && total_successes == 0;
        Ok(text_result(
            results.join("\n"),
            json!({"action": action, "serverName": all_server_names.join(", "), "success": !all_failed, "request": params.request_json()}),
        ))
    }

    /// Diagnostics for one file from one server (publish-wait path).
    async fn diagnostics_from_server(
        self: &Arc<Self>,
        resolved: &Path,
        uri: &str,
        server_config: &omp_lsp::types::ServerConfig,
        detailed: bool,
        cancel: &DeadlineCancel<'_>,
        timeout_secs: u64,
    ) -> std::result::Result<Vec<omp_lsp::types::Diagnostic>, LspError> {
        use omp_lsp::diagnostics as diag;
        let client =
            omp_lsp::client::get_or_create_client(server_config, &self.env.cwd, None, Some(cancel))
                .await?;
        if omp_lsp::servers::is_project_aware_lsp_server(server_config) {
            client.wait_for_project_loaded(Some(cancel)).await;
            if cancel.is_cancelled() {
                return Err(LspError {
                    message: "operation cancelled".to_string(),
                    cancelled: true,
                });
            }
        }
        let min_version = client.diagnostics_version();
        client.refresh_file(resolved).await;
        let expected_version = client.open_file_version(uri);
        let wait_cap = if detailed {
            diag::BATCH_DIAGNOSTICS_WAIT_TIMEOUT_MS
        } else if omp_lsp::servers::is_project_aware_lsp_server(server_config) {
            diag::PROJECT_DIAGNOSTICS_WAIT_TIMEOUT_MS
        } else {
            diag::SINGLE_DIAGNOSTICS_WAIT_TIMEOUT_MS
        };
        let wait = wait_cap.min(timeout_secs * 1000);
        diag::wait_for_diagnostics(
            &client,
            uri,
            wait,
            Some(cancel),
            Some(min_version),
            expected_version,
        )
        .await
        .map_err(|message| LspError {
            message,
            cancelled: false,
        })
    }
}

impl LspTool {
    /// `diagnostics file:"*"` — workspace-wide diagnostics across all servers.
    async fn action_workspace_diagnostics(
        self: &Arc<Self>,
        params: &Params,
        cwd: &Path,
        cancel: &DeadlineCancel<'_>,
    ) -> Result<AgentToolResult> {
        let config = omp_lsp::config::get_config(cwd);
        let servers = omp_lsp::servers::get_lsp_servers(&config);
        if servers.is_empty() {
            return Ok(text_result(
                "No language servers configured",
                json!({"action": "diagnostics", "success": false, "request": params.request_json()}),
            ));
        }
        // Collect candidate files per server (bounded walk of the workspace).
        let mut results: Vec<String> = Vec::new();
        let mut any = false;
        for (server_name, server_config) in &servers {
            if cancel.is_cancelled() {
                return Err(pa_agent::abort::aborted_error());
            }
            let files = collect_workspace_files(cwd, &server_config.file_types, 100);
            if files.is_empty() {
                continue;
            }
            let client =
                match omp_lsp::client::get_or_create_client(server_config, cwd, None, Some(cancel))
                    .await
                {
                    Ok(c) => c,
                    Err(LspError {
                        cancelled: true, ..
                    }) => return Err(pa_agent::abort::aborted_error()),
                    Err(e) => {
                        results.push(format!("{server_name}: failed to start ({})", e.message));
                        continue;
                    }
                };
            client.wait_for_project_loaded(Some(cancel)).await;
            let mut rows: Vec<String> = Vec::new();
            for file in &files {
                let uri = omp_lsp::utils::file_to_uri(file);
                client.refresh_file(file).await;
                let expected = client.open_file_version(&uri);
                match omp_lsp::diagnostics::wait_for_diagnostics(
                    &client,
                    &uri,
                    omp_lsp::diagnostics::BATCH_DIAGNOSTICS_WAIT_TIMEOUT_MS,
                    Some(cancel),
                    None,
                    expected,
                )
                .await
                {
                    Ok(ds) if !ds.is_empty() => {
                        any = true;
                        let rel = omp_lsp::utils::format_path_relative_to_cwd(file, cwd);
                        for d in &ds {
                            rows.push(omp_lsp::utils::format_diagnostic(d, &rel));
                        }
                    }
                    Ok(_) | Err(_) => {}
                }
            }
            if !rows.is_empty() {
                results.push(format!("{server_name}:"));
                results.push(omp_lsp::utils::format_grouped_diagnostic_messages(&rows));
            }
        }
        if !any && results.is_empty() {
            results.push("no issues".to_string());
        }
        Ok(text_result(
            format!("Workspace diagnostics:\n{}", results.join("\n")),
            json!({"action": "diagnostics", "success": true, "request": params.request_json()}),
        ))
    }

    /// `symbols` with `file:"*"` — workspace-wide symbol search.
    async fn action_workspace_symbols(
        self: &Arc<Self>,
        params: &Params,
        cwd: &Path,
        cancel: &DeadlineCancel<'_>,
        signal: &AbortSignal,
    ) -> Result<AgentToolResult> {
        let action = "symbols";
        let normalized_query = params.query.as_deref().unwrap_or("").trim().to_string();
        if normalized_query.is_empty() {
            return Ok(text_result(
                "Error: query parameter required for workspace symbol search",
                json!({"action": action, "success": false, "request": params.request_json()}),
            ));
        }
        let config = omp_lsp::config::get_config(cwd);
        let servers = omp_lsp::servers::get_lsp_servers(&config);
        if servers.is_empty() {
            return Ok(text_result(
                "No language server found for this action",
                json!({"action": action, "success": false, "request": params.request_json()}),
            ));
        }
        let mut aggregated: Vec<omp_lsp::types::SymbolInformation> = Vec::new();
        let mut responding: Vec<String> = Vec::new();
        let mut failures: Vec<String> = Vec::new();
        for (name, server_config) in &servers {
            if cancel.is_cancelled() {
                return Err(if signal.is_aborted() {
                    pa_agent::abort::aborted_error()
                } else {
                    anyhow!("LSP {action} timed out")
                });
            }
            match omp_lsp::client::get_or_create_client(server_config, cwd, None, Some(cancel))
                .await
            {
                Ok(client) => match client
                    .send_request(
                        "workspace/symbol",
                        json!({ "query": normalized_query }),
                        Some(cancel),
                        None,
                    )
                    .await
                {
                    Ok(result) => {
                        responding.push(name.clone());
                        let symbols: Vec<omp_lsp::types::SymbolInformation> = result
                            .as_array()
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                                    .collect()
                            })
                            .unwrap_or_default();
                        aggregated.extend(omp_lsp::utils::filter_workspace_symbols(
                            &symbols,
                            &normalized_query,
                        ));
                    }
                    Err(LspError {
                        cancelled: true, ..
                    }) => {
                        return Err(if signal.is_aborted() {
                            pa_agent::abort::aborted_error()
                        } else {
                            anyhow!("LSP {action} timed out")
                        });
                    }
                    Err(e) => failures.push(format!("  {name}: {}", e.message)),
                },
                Err(LspError {
                    cancelled: true, ..
                }) => {
                    return Err(if signal.is_aborted() {
                        pa_agent::abort::aborted_error()
                    } else {
                        anyhow!("LSP {action} timed out")
                    });
                }
                Err(e) => failures.push(format!("  {name}: {}", e.message)),
            }
        }
        let failure_section = if failures.is_empty() {
            String::new()
        } else {
            format!("\nServer failures:\n{}", failures.join("\n"))
        };
        if responding.is_empty() {
            return Ok(text_result(
                format!(
                    "Workspace symbol search failed: all language servers failed{failure_section}"
                ),
                json!({"action": action, "serverName": "", "success": false, "request": params.request_json()}),
            ));
        }
        let deduped = omp_lsp::utils::dedupe_workspace_symbols(&aggregated);
        if deduped.is_empty() {
            return Ok(text_result(
                format!("No symbols matching \"{normalized_query}\"{failure_section}"),
                json!({"action": action, "serverName": responding.join(", "), "success": true, "request": params.request_json()}),
            ));
        }
        let limited: Vec<omp_lsp::types::SymbolInformation> = deduped
            .iter()
            .take(omp_lsp::diagnostics::WORKSPACE_SYMBOL_LIMIT)
            .cloned()
            .collect();
        let lines: Vec<String> = limited
            .iter()
            .map(|s| format!("  {}", omp_lsp::utils::format_symbol_information(s, cwd)))
            .collect();
        let truncation = if deduped.len() > omp_lsp::diagnostics::WORKSPACE_SYMBOL_LIMIT {
            format!(
                "\n[…{} symbols elided…]",
                deduped.len() - omp_lsp::diagnostics::WORKSPACE_SYMBOL_LIMIT
            )
        } else {
            String::new()
        };
        Ok(text_result(
            format!(
                "Found {} symbol(s) matching \"{normalized_query}\":\n{}{}{failure_section}",
                deduped.len(),
                lines.join("\n"),
                truncation
            ),
            json!({"action": action, "serverName": responding.join(", "), "success": true, "request": params.request_json()}),
        ))
    }

    /// `reload *` — re-read config, stop stale servers, reload the rest.
    async fn action_reload_all(
        self: &Arc<Self>,
        params: &Params,
        cwd: &Path,
        cancel: &DeadlineCancel<'_>,
        signal: &AbortSignal,
    ) -> Result<AgentToolResult> {
        omp_lsp::config::invalidate_config(cwd);
        let config = omp_lsp::config::get_config(cwd);
        omp_lsp::client::reconcile_idle_checker();
        let servers = omp_lsp::servers::get_lsp_servers(&config);
        if servers.is_empty() {
            return Ok(text_result(
                "No language server found for this action",
                json!({"action": "reload", "success": false, "request": params.request_json()}),
            ));
        }
        let mut outputs: Vec<String> = Vec::new();
        for (name, server_config) in &servers {
            if cancel.is_cancelled() {
                return Err(if signal.is_aborted() {
                    pa_agent::abort::aborted_error()
                } else {
                    anyhow!("LSP reload timed out")
                });
            }
            omp_lsp::client::clear_initialization_failure(server_config, cwd);
            match omp_lsp::client::get_or_create_client(server_config, cwd, None, Some(cancel))
                .await
            {
                Ok(client) => {
                    match omp_lsp::servers::reload_server(&client, name, Some(cancel)).await {
                        Ok(line) => outputs.push(line),
                        Err(LspError {
                            cancelled: true, ..
                        }) => {
                            return Err(if signal.is_aborted() {
                                pa_agent::abort::aborted_error()
                            } else {
                                anyhow!("LSP reload timed out")
                            });
                        }
                        Err(e) => outputs.push(format!("Failed to reload {name}: {}", e.message)),
                    }
                }
                Err(LspError {
                    cancelled: true, ..
                }) => {
                    return Err(if signal.is_aborted() {
                        pa_agent::abort::aborted_error()
                    } else {
                        anyhow!("LSP reload timed out")
                    });
                }
                Err(e) => outputs.push(format!("Failed to reload {name}: {}", e.message)),
            }
        }
        Ok(text_result(
            outputs.join("\n"),
            json!({"action": "reload", "serverName": servers.iter().map(|(n, _)| n).cloned().collect::<Vec<_>>().join(", "), "success": true, "request": params.request_json()}),
        ))
    }
}

/// Bounded recursive walk of `cwd` collecting files matching `file_types`
/// (both `.ext` and `ext` spellings).
fn collect_workspace_files(cwd: &Path, file_types: &[String], max: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = vec![cwd.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if out.len() >= max {
            break;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            if out.len() >= max {
                break;
            }
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_lowercase();
            if path.is_dir() {
                if !matches!(
                    name.as_str(),
                    "node_modules" | ".git" | "target" | ".venv" | "venv"
                ) {
                    stack.push(path);
                }
                continue;
            }
            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            let matches = file_types.iter().any(|ft| {
                let f = ft.to_lowercase();
                f.trim_start_matches('.') == ext || f == name
            });
            if matches {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

impl LspTool {
    /// `request` — raw JSON-RPC method passthrough.
    async fn action_request(
        self: &Arc<Self>,
        params: &Params,
        cwd: &Path,
        cancel: &DeadlineCancel<'_>,
        signal: &AbortSignal,
    ) -> Result<AgentToolResult> {
        let action = "request";
        let method = params.query.as_deref().unwrap_or("").trim().to_string();
        if method.is_empty() {
            return Ok(text_result(
                "Error: action=request requires `query` to specify the LSP method name (e.g., 'rust-analyzer/expandMacro')",
                json!({"action": action, "success": false, "request": params.request_json()}),
            ));
        }
        let config = omp_lsp::config::get_config(cwd);
        let (chosen_name, chosen_config);
        let mut resolved_target: Option<PathBuf> = None;
        match &params.file {
            Some(file) if file != "*" => {
                let resolved = omp_lsp::utils::resolve_to_cwd(file, cwd);
                let Some(pair) = omp_lsp::servers::get_lsp_server_for_file(&config, &resolved)
                else {
                    return Ok(text_result(
                        "No language server found for this file",
                        json!({"action": action, "success": false, "request": params.request_json()}),
                    ));
                };
                resolved_target = Some(resolved);
                chosen_name = pair.0;
                chosen_config = pair.1;
            }
            _ => {
                let all = omp_lsp::servers::get_lsp_servers(&config);
                if all.is_empty() {
                    return Ok(text_result(
                        "No language servers configured",
                        json!({"action": action, "success": false, "request": params.request_json()}),
                    ));
                }
                let (n, c) = all.into_iter().next().unwrap();
                chosen_name = n;
                chosen_config = c;
            }
        }

        let request_params: Value = if let Some(payload) = &params.payload {
            match serde_json::from_str::<Value>(payload) {
                Ok(v) => v,
                Err(e) => {
                    return Ok(text_result(
                        format!("Error: invalid JSON in payload: {e}"),
                        json!({"action": action, "serverName": chosen_name, "success": false, "request": params.request_json()}),
                    ));
                }
            }
        } else if let Some(target) = &resolved_target {
            let uri = omp_lsp::utils::file_to_uri(target);
            if let Some(line) = params.line {
                let character = omp_lsp::utils::resolve_symbol_column(
                    target,
                    line as usize,
                    params.symbol.as_deref(),
                )
                .await?;
                json!({ "textDocument": { "uri": uri }, "position": { "line": line - 1, "character": character } })
            } else {
                json!({ "textDocument": { "uri": uri } })
            }
        } else {
            json!({})
        };

        let client = match omp_lsp::client::get_or_create_client(
            &chosen_config,
            cwd,
            None,
            Some(cancel),
        )
        .await
        {
            Ok(c) => c,
            Err(LspError {
                cancelled: true, ..
            }) => {
                return Err(if signal.is_aborted() {
                    pa_agent::abort::aborted_error()
                } else {
                    anyhow!("LSP {action} timed out")
                });
            }
            Err(e) => {
                return Ok(text_result(
                    format!("LSP error: {}", e.message),
                    json!({"action": action, "serverName": chosen_name, "success": false, "request": params.request_json()}),
                ))
            }
        };
        if let Some(target) = &resolved_target {
            client.refresh_file(target).await;
        }
        match client
            .send_request(&method, request_params.clone(), Some(cancel), None)
            .await
        {
            Ok(result) => {
                let formatted = if result.is_null() {
                    "null".to_string()
                } else if let Some(s) = result.as_str() {
                    s.to_string()
                } else {
                    serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string())
                };
                Ok(text_result(
                    format!("{chosen_name} ← {method}:\n{formatted}"),
                    json!({"action": action, "serverName": chosen_name, "success": true, "request": params.request_json()}),
                ))
            }
            Err(LspError {
                cancelled: true, ..
            }) => Err(if signal.is_aborted() {
                pa_agent::abort::aborted_error()
            } else {
                anyhow!("LSP {action} timed out")
            }),
            Err(e) => {
                let preview_raw =
                    serde_json::to_string(&request_params).unwrap_or_else(|_| "null".to_string());
                let preview = if preview_raw.len() > 400 {
                    format!("{}...", &preview_raw[..397])
                } else {
                    preview_raw
                };
                Ok(text_result(
                    format!(
                        "LSP error from {chosen_name} on {method}: {}\n  params: {preview}",
                        e.message
                    ),
                    json!({"action": action, "serverName": chosen_name, "success": false, "request": params.request_json()}),
                ))
            }
        }
    }
}

/// Enumerate the {oldUri, newUri} pairs for willRenameFiles/didRenameFiles.
async fn enumerate_rename_pairs(
    source: &Path,
    dest: &Path,
) -> std::io::Result<(Vec<(String, String)>, bool, bool)> {
    let stat = tokio::fs::metadata(source).await?;
    if !stat.is_dir() {
        return Ok((
            vec![(
                omp_lsp::utils::file_to_uri(source),
                omp_lsp::utils::file_to_uri(dest),
            )],
            false,
            false,
        ));
    }
    let mut pairs = Vec::new();
    let mut stack: Vec<(PathBuf, PathBuf)> = vec![(source.to_path_buf(), dest.to_path_buf())];
    while let Some((dir, new_dir)) = stack.pop() {
        let mut rd = tokio::fs::read_dir(&dir).await?;
        while let Some(entry) = rd.next_entry().await? {
            let path = entry.path();
            let rel_new = new_dir.join(entry.file_name());
            if path.is_dir() {
                stack.push((path, rel_new));
                continue;
            }
            if pairs.len() >= MAX_RENAME_PAIRS {
                return Ok((pairs, true, true));
            }
            pairs.push((
                omp_lsp::utils::file_to_uri(&path),
                omp_lsp::utils::file_to_uri(&rel_new),
            ));
        }
    }
    Ok((pairs, true, false))
}

struct Bucket {
    primary_server: String,
    edits: Vec<omp_lsp::types::TextEdit>,
    discarded: usize,
    conflict_servers: Vec<String>,
}

impl LspTool {
    /// `rename_file` — move a file/dir and rewrite references via
    /// workspace/willRenameFiles; preview with `apply:false`.
    #[allow(clippy::too_many_lines)] // TS-parity per-action dispatch; splitting would diverge from omp tool.ts
    async fn action_rename_file(
        self: &Arc<Self>,
        params: &Params,
        cwd: &Path,
        cancel: &DeadlineCancel<'_>,
    ) -> Result<AgentToolResult> {
        let action = "rename_file";
        let err = |text: &str| {
            Ok(text_result(
                text,
                json!({"action": action, "success": false, "request": params.request_json()}),
            ))
        };
        let (Some(file), Some(new_name)) = (&params.file, &params.new_name) else {
            return err("Error: rename_file requires both `file` (source path) and `new_name` (destination path)");
        };
        let source = omp_lsp::utils::resolve_to_cwd(file, cwd);
        let dest = omp_lsp::utils::resolve_to_cwd(new_name, cwd);
        if source == dest {
            return err("Error: source and destination paths are identical");
        }
        let _source_stat = match tokio::fs::metadata(&source).await {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return err(&format!(
                    "Error: source path does not exist: {}",
                    source.display()
                ));
            }
            Err(e) => {
                return err(&format!(
                    "Error: cannot read source path {}: {e}",
                    source.display()
                ))
            }
        };
        match tokio::fs::symlink_metadata(&dest).await {
            Ok(_) => {
                return err(&format!(
                    "Error: destination already exists: {}",
                    dest.display()
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return err(&format!(
                    "Error: cannot read destination path {}: {e}",
                    dest.display()
                ))
            }
        }

        let (pairs, directory, exceeded) = match enumerate_rename_pairs(&source, &dest).await {
            Ok(p) => p,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return err(&format!(
                    "Error: source path does not exist: {}",
                    source.display()
                ));
            }
            Err(e) => {
                return err(&format!(
                    "Error: cannot read source path {}: {e}",
                    source.display()
                ))
            }
        };
        if exceeded {
            return err(&format!(
                "Error: directory contains more than {MAX_RENAME_PAIRS} files; rename in smaller batches to keep LSP edits accurate"
            ));
        }
        if pairs.is_empty() {
            return err("Error: no files to rename");
        }

        let lsp_params = json!({ "files": pairs.iter().map(|(o, n)| json!({"oldUri": o, "newUri": n})).collect::<Vec<_>>() });
        let config = omp_lsp::config::get_config(cwd);
        let all_lsp_servers = omp_lsp::servers::get_lsp_servers(&config);
        let mut relevant: Vec<String> = Vec::new();
        let collect_relevant = |path: &Path, relevant: &mut Vec<String>| {
            for (name, _) in omp_lsp::servers::get_lsp_servers_for_file(&config, path) {
                if !relevant.contains(&name) {
                    relevant.push(name);
                }
            }
        };
        collect_relevant(&source, &mut relevant);
        collect_relevant(&dest, &mut relevant);
        for (old, new) in &pairs {
            collect_relevant(&omp_lsp::utils::uri_to_file(old), &mut relevant);
            collect_relevant(&omp_lsp::utils::uri_to_file(new), &mut relevant);
        }
        let servers: Vec<(String, omp_lsp::types::ServerConfig)> = all_lsp_servers
            .into_iter()
            .filter(|(name, _)| relevant.contains(name))
            .collect();

        let mut responding: Vec<String> = Vec::new();
        let mut per_server_edits: Vec<(String, omp_lsp::types::WorkspaceEdit)> = Vec::new();
        let mut server_notes: Vec<String> = Vec::new();
        let mut hard_failures: Vec<String> = Vec::new();

        for (server_name, server_config) in &servers {
            if cancel.is_cancelled() {
                return Err(pa_agent::abort::aborted_error());
            }
            let client =
                match omp_lsp::client::get_or_create_client(server_config, cwd, None, Some(cancel))
                    .await
                {
                    Ok(c) => c,
                    Err(LspError {
                        cancelled: true, ..
                    }) => return Err(pa_agent::abort::aborted_error()),
                    Err(e) => {
                        server_notes.push(format!("  {server_name}: {}", e.message));
                        continue;
                    }
                };
            if omp_lsp::servers::is_project_aware_lsp_server(server_config) {
                client.wait_for_project_loaded(Some(cancel)).await;
            }
            match client
                .send_request(
                    "workspace/willRenameFiles",
                    lsp_params.clone(),
                    Some(cancel),
                    None,
                )
                .await
            {
                Ok(result) => {
                    responding.push(server_name.clone());
                    if let Ok(edit) =
                        serde_json::from_value::<omp_lsp::types::WorkspaceEdit>(result)
                    {
                        if edit.changes.is_some() || edit.document_changes.is_some() {
                            per_server_edits.push((server_name.clone(), edit));
                        }
                    }
                }
                Err(LspError {
                    cancelled: true, ..
                }) => return Err(pa_agent::abort::aborted_error()),
                Err(e) => {
                    if !omp_lsp::servers::is_method_not_found_error(&e) {
                        server_notes.push(format!("  {server_name}: {}", e.message));
                        hard_failures.push(server_name.clone());
                    }
                }
            }
        }

        let source_label = omp_lsp::utils::format_path_relative_to_cwd(&source, cwd);
        let dest_label = omp_lsp::utils::format_path_relative_to_cwd(&dest, cwd);
        let file_count_label_owned;
        let file_count_label = if directory {
            format!(
                "{} file{} under {source_label}",
                pairs.len(),
                if pairs.len() == 1 { "" } else { "s" }
            )
        } else {
            file_count_label_owned = source_label.clone();
            file_count_label_owned
        };

        if params.apply != Some(true) {
            let mut lines: Vec<String> =
                vec![format!("Rename preview: {file_count_label} → {dest_label}")];
            if per_server_edits.is_empty() {
                lines.push("  No LSP edits would be applied".to_string());
            } else {
                for (server_name, edit) in &per_server_edits {
                    let edits = format_workspace_edit_lines(edit, cwd);
                    if edits.is_empty() {
                        continue;
                    }
                    lines.push(format!("  {server_name}:"));
                    for e in edits {
                        lines.push(format!("    {e}"));
                    }
                }
            }
            if !server_notes.is_empty() {
                lines.push("  Server notes:".to_string());
                lines.extend(server_notes.iter().cloned());
            }
            return Ok(text_result(
                lines.join("\n"),
                json!({"action": action, "serverName": responding.join(", "), "success": true, "request": params.request_json()}),
            ));
        }

        if !hard_failures.is_empty() {
            let mut lines = vec![format!(
                "Error: aborted rename; workspace/willRenameFiles failed on {}, so semantic references would not be updated. No files were moved.",
                hard_failures.join(", ")
            )];
            lines.push("  Server notes:".to_string());
            lines.extend(server_notes.iter().cloned());
            return Ok(text_result(
                lines.join("\n"),
                json!({"action": action, "serverName": responding.join(", "), "success": false, "request": params.request_json()}),
            ));
        }

        // Coalesce per-URI edits across servers: primary (project-aware) wins overlaps.
        let server_config_by_name: std::collections::HashMap<
            String,
            &omp_lsp::types::ServerConfig,
        > = servers.iter().map(|(n, c)| (n.clone(), c)).collect();
        let mut accepted: Vec<(String, Bucket)> = Vec::new();
        for (server_name, edit) in &per_server_edits {
            let incoming_primary = server_config_by_name
                .get(server_name)
                .is_some_and(|c| omp_lsp::servers::is_project_aware_lsp_server(c));
            for (uri, edits) in omp_lsp::edits::flatten_workspace_text_edits(edit) {
                let existing = accepted.iter_mut().find(|(u, _)| *u == uri);
                match existing {
                    None => accepted.push((
                        uri.clone(),
                        Bucket {
                            primary_server: server_name.clone(),
                            edits: edits.clone(),
                            discarded: 0,
                            conflict_servers: Vec::new(),
                        },
                    )),
                    Some((_, bucket)) => {
                        let existing_primary = server_config_by_name
                            .get(&bucket.primary_server)
                            .is_some_and(|c| omp_lsp::servers::is_project_aware_lsp_server(c));
                        if incoming_primary && !existing_primary {
                            let mut kept_old = Vec::new();
                            let mut discarded_old = 0;
                            for oe in &bucket.edits {
                                if edits
                                    .iter()
                                    .any(|ne| omp_lsp::edits::ranges_overlap(&ne.range, &oe.range))
                                {
                                    discarded_old += 1;
                                } else {
                                    kept_old.push(oe.clone());
                                }
                            }
                            if discarded_old > 0
                                && !bucket.conflict_servers.contains(&bucket.primary_server)
                            {
                                bucket.conflict_servers.push(bucket.primary_server.clone());
                            }
                            bucket.discarded += discarded_old;
                            bucket.primary_server.clone_from(server_name);
                            let mut merged = edits.clone();
                            merged.extend(kept_old);
                            bucket.edits = merged;
                        } else {
                            let mut discarded_new = 0;
                            for ne in &edits {
                                if bucket
                                    .edits
                                    .iter()
                                    .any(|ae| omp_lsp::edits::ranges_overlap(&ae.range, &ne.range))
                                {
                                    discarded_new += 1;
                                } else {
                                    bucket.edits.push(ne.clone());
                                }
                            }
                            if discarded_new > 0 {
                                if !bucket.conflict_servers.contains(server_name) {
                                    bucket.conflict_servers.push(server_name.clone());
                                }
                                bucket.discarded += discarded_new;
                            }
                        }
                    }
                }
            }
        }

        // Validate every bucket before writing any file.
        for (_, bucket) in &accepted {
            let mut edits = bucket.edits.clone();
            if let Err(e) = omp_lsp::edits::sort_and_validate_text_edits(&mut edits) {
                return Ok(text_result(
                    format!("Error: {e}"),
                    json!({"action": action, "success": false, "request": params.request_json()}),
                ));
            }
        }

        let mut summary: Vec<String> = Vec::new();
        let mut reference_edits: Vec<omp_lsp::edits::RenameReferenceEdit> = Vec::new();
        for (uri, bucket) in &accepted {
            let file_path = omp_lsp::utils::uri_to_file(uri);
            reference_edits.push(omp_lsp::edits::RenameReferenceEdit {
                file_path: file_path.clone(),
                edits: bucket.edits.clone(),
            });
            let rel = omp_lsp::utils::format_path_relative_to_cwd(&file_path, cwd);
            summary.push(format!(
                "  {}: applied {} edit(s) to {rel}",
                bucket.primary_server,
                bucket.edits.len()
            ));
            if bucket.discarded > 0 {
                let others = bucket.conflict_servers.join(", ");
                summary.push(format!(
                    "    note: discarded {} overlapping edit(s) from {others} (kept {})",
                    bucket.discarded, bucket.primary_server
                ));
            }
        }

        if let Err(e) =
            omp_lsp::edits::apply_edits_then_rename(&reference_edits, &source, &dest).await
        {
            return Ok(text_result(
                format!("Error: {e}"),
                json!({"action": action, "success": false, "request": params.request_json()}),
            ));
        }
        summary.push(format!("  Renamed {source_label} → {dest_label}"));

        // Announce the rename to every relevant server.
        let old_uris: Vec<String> = pairs.iter().map(|(o, _)| o.clone()).collect();
        for (server_name, server_config) in &servers {
            if let Ok(client) =
                omp_lsp::client::get_or_create_client(server_config, cwd, None, Some(cancel)).await
            {
                if let Err(e) =
                    omp_lsp::servers::notify_did_rename(&client, lsp_params.clone(), &old_uris)
                        .await
                {
                    server_notes.push(format!("  {server_name}: {}", e.message));
                }
            }
        }
        if !server_notes.is_empty() {
            summary.push("  Server notes:".to_string());
            summary.extend(server_notes.iter().cloned());
        }

        Ok(text_result(
            format!(
                "Renamed {file_count_label} → {dest_label}\n{}",
                summary.join("\n")
            ),
            json!({"action": action, "serverName": responding.join(", "), "success": true, "request": params.request_json()}),
        ))
    }
}

fn format_workspace_edit_lines(edit: &omp_lsp::types::WorkspaceEdit, cwd: &Path) -> Vec<String> {
    let mut results = Vec::new();
    if let Some(changes) = &edit.changes {
        for (uri, edits_value) in changes {
            let count = edits_value.as_array().map_or(0, Vec::len);
            let file =
                omp_lsp::utils::format_path_relative_to_cwd(&omp_lsp::utils::uri_to_file(uri), cwd);
            results.push(format!(
                "{file}: {count} edit{}",
                if count > 1 { "s" } else { "" }
            ));
        }
    }
    if let Some(document_changes) = &edit.document_changes {
        use omp_lsp::types::DocumentChange;
        for change in document_changes {
            match change {
                DocumentChange::TextDocumentEdit(tdc) => {
                    let file = omp_lsp::utils::format_path_relative_to_cwd(
                        &omp_lsp::utils::uri_to_file(&tdc.text_document.uri),
                        cwd,
                    );
                    results.push(format!(
                        "{file}: {} edit{}",
                        tdc.edits.len(),
                        if tdc.edits.len() > 1 { "s" } else { "" }
                    ));
                }
                DocumentChange::CreateFile(c) => {
                    results.push(format!(
                        "CREATE: {}",
                        omp_lsp::utils::format_path_relative_to_cwd(
                            &omp_lsp::utils::uri_to_file(&c.uri),
                            cwd
                        )
                    ));
                }
                DocumentChange::RenameFile(r) => {
                    results.push(format!(
                        "RENAME: {} {} {}",
                        omp_lsp::utils::format_path_relative_to_cwd(
                            &omp_lsp::utils::uri_to_file(&r.old_uri),
                            cwd
                        ),
                        omp_lsp::utils::NAV_CURSOR,
                        omp_lsp::utils::format_path_relative_to_cwd(
                            &omp_lsp::utils::uri_to_file(&r.new_uri),
                            cwd
                        )
                    ));
                }
                DocumentChange::DeleteFile(d) => {
                    results.push(format!(
                        "DELETE: {}",
                        omp_lsp::utils::format_path_relative_to_cwd(
                            &omp_lsp::utils::uri_to_file(&d.uri),
                            cwd
                        )
                    ));
                }
            }
        }
    }
    results
}

impl LspTool {
    /// File-based actions: `definition`/`type_definition`/`implementation`/`references`/
    /// `hover`/`symbols`/`rename`/`code_actions`/`reload`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)] // TS-parity per-action dispatch; splitting would diverge from omp tool.ts
    async fn action_file_based(
        self: &Arc<Self>,
        action: &str,
        params: &Params,
        cwd: &Path,
        target_file: &Path,
        server_name: &str,
        server_config: &omp_lsp::types::ServerConfig,
        cancel: &DeadlineCancel<'_>,
        timeout_secs: u64,
    ) -> Result<AgentToolResult> {
        use omp_lsp::diagnostics as diag;
        let client =
            omp_lsp::client::get_or_create_client(server_config, cwd, None, Some(cancel)).await?;
        let is_rust_analyzer_server = client.is_rust_analyzer() || server_name == "rust-analyzer";
        let project_aware = omp_lsp::servers::is_project_aware_lsp_server(server_config);
        let needs_project_index = diag::PROJECT_INDEXED_ACTIONS.contains(&action) && project_aware;
        let rust_workspace_wait = needs_project_index
            && is_rust_analyzer_server
            && diag::has_rust_workspace_ancestor(target_file);

        client.refresh_file(target_file).await;
        let reconciled_from_disk = false;
        if rust_workspace_wait {
            client.wait_for_project_loaded(Some(cancel)).await;
        }
        if cancel.is_cancelled() {
            return Err(pa_agent::abort::aborted_error());
        }

        // Project-aware position lookups without a symbol silently hit the
        // wrong identifier; require `symbol` explicitly.
        if params.line.is_some()
            && params.symbol.is_none()
            && matches!(action, "references" | "rename" | "definition")
            && project_aware
        {
            return Err(anyhow!(
                "symbol is required for project-aware {action}; pass symbol=<name>, optionally symbol#N for repeated occurrences"
            ));
        }
        let uri = omp_lsp::utils::file_to_uri(target_file);
        let resolved_line = params.line.unwrap_or(1);
        let resolved_character = if target_file.exists() {
            omp_lsp::utils::resolve_symbol_column(
                target_file,
                resolved_line as usize,
                params.symbol.as_deref(),
            )
            .await?
        } else {
            0
        };
        let position = json!({ "line": resolved_line - 1, "character": resolved_character });

        if needs_project_index && !is_rust_analyzer_server {
            client.wait_for_project_loaded(Some(cancel)).await;
        }

        let mut useless = false;
        let output: String = match action {
            "definition" | "type_definition" | "implementation" => {
                let method = match action {
                    "definition" => "textDocument/definition",
                    "type_definition" => "textDocument/typeDefinition",
                    _ => "textDocument/implementation",
                };
                let result = client
                    .send_request(
                        method,
                        json!({ "textDocument": { "uri": uri }, "position": position }),
                        Some(cancel),
                        None,
                    )
                    .await?;
                let locations = diag::normalize_location_result(&result);
                if locations.is_empty() {
                    useless = true;
                    format!("No {action} found").replace("type_definition", "type definition")
                } else {
                    let mut lines = Vec::new();
                    for location in &locations {
                        lines.push(diag::format_location_with_context(location, cwd).await);
                    }
                    let label = match action {
                        "definition" => "definition",
                        "type_definition" => "type definition",
                        _ => "implementation",
                    };
                    format!(
                        "Found {} {label}(s):\n{}",
                        locations.len(),
                        lines.join("\n")
                    )
                }
            }
            "references" => {
                let mut result: Option<Vec<omp_lsp::types::Location>> = None;
                for attempt in 0..=diag::REFERENCES_RETRY_COUNT {
                    let raw = client
                        .send_request(
                            "textDocument/references",
                            json!({
                                "textDocument": { "uri": uri },
                                "position": position,
                                "context": { "includeDeclaration": true }
                            }),
                            Some(cancel),
                            None,
                        )
                        .await?;
                    let locations: Vec<omp_lsp::types::Location> = raw
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| serde_json::from_value(v.clone()).ok())
                                .collect()
                        })
                        .unwrap_or_default();
                    if !project_aware || attempt == diag::REFERENCES_RETRY_COUNT {
                        result = Some(locations);
                        break;
                    }
                    let pos = omp_lsp::types::Position {
                        line: resolved_line - 1,
                        character: u32::try_from(resolved_character).unwrap_or(u32::MAX),
                    };
                    if !locations.is_empty()
                        && !diag::is_only_queried_declaration(&locations, &uri, &pos)
                    {
                        result = Some(locations);
                        break;
                    }
                    client.wait_for_project_loaded(Some(cancel)).await;
                    if cancel.is_cancelled() {
                        return Err(pa_agent::abort::aborted_error());
                    }
                    tokio::time::sleep(Duration::from_millis(diag::REFERENCES_RETRY_DELAY_MS))
                        .await;
                }
                let locations = result.unwrap_or_default();
                if locations.is_empty() {
                    useless = true;
                    "No references found".to_string()
                } else {
                    let contextual: Vec<_> = locations
                        .iter()
                        .take(diag::REFERENCE_CONTEXT_LIMIT)
                        .cloned()
                        .collect();
                    let plain: Vec<_> = locations
                        .iter()
                        .skip(diag::REFERENCE_CONTEXT_LIMIT)
                        .cloned()
                        .collect();
                    let mut lines: Vec<String> = Vec::new();
                    for location in &contextual {
                        lines.push(diag::format_location_with_context(location, cwd).await);
                    }
                    let mut plain_lines: Vec<String> = plain
                        .iter()
                        .map(|l| format!("  {}", omp_lsp::utils::format_location(l, cwd)))
                        .collect();
                    if !plain_lines.is_empty() {
                        lines.push(format!(
                            "  ... {} additional reference(s) shown without context",
                            plain_lines.len()
                        ));
                        lines.append(&mut plain_lines);
                    }
                    format!(
                        "Found {} reference(s):\n{}",
                        locations.len(),
                        lines.join("\n")
                    )
                }
            }
            "hover" => {
                let result = client
                    .send_request(
                        "textDocument/hover",
                        json!({ "textDocument": { "uri": uri }, "position": position }),
                        Some(cancel),
                        None,
                    )
                    .await?;
                let hover: Option<omp_lsp::types::Hover> = serde_json::from_value(result).ok();
                match hover {
                    Some(h) if !h.contents.is_null() => {
                        omp_lsp::utils::extract_hover_text(&h.contents)
                    }
                    _ => "No hover information".to_string(),
                }
            }
            "code_actions" => {
                self.action_code_actions(
                    params,
                    &client,
                    &uri,
                    &position,
                    reconciled_from_disk,
                    cwd,
                    cancel,
                    timeout_secs,
                    project_aware,
                )
                .await?
            }
            "symbols" => {
                let result = client
                    .send_request(
                        "textDocument/documentSymbol",
                        json!({ "textDocument": { "uri": uri } }),
                        Some(cancel),
                        None,
                    )
                    .await?;
                let arr = result.as_array().cloned().unwrap_or_default();
                if arr.is_empty() {
                    useless = true;
                    "No symbols found".to_string()
                } else {
                    let rel_path = omp_lsp::utils::format_path_relative_to_cwd(target_file, cwd);
                    let first = &arr[0];
                    if first.get("selectionRange").is_some() {
                        let symbols: Vec<omp_lsp::types::DocumentSymbol> = arr
                            .iter()
                            .filter_map(|v| serde_json::from_value(v.clone()).ok())
                            .collect();
                        let mut lines = Vec::new();
                        for s in &symbols {
                            lines.extend(format_document_symbol(s, 0));
                        }
                        format!("Symbols in {rel_path}:\n{}", lines.join("\n"))
                    } else {
                        let infos: Vec<omp_lsp::types::SymbolInformation> = arr
                            .iter()
                            .filter_map(|v| serde_json::from_value(v.clone()).ok())
                            .collect();
                        let lines: Vec<String> = infos
                            .iter()
                            .map(|s| {
                                let line = s.location.range.start.line + 1;
                                format!("{} {} @ line {line}", symbol_icon(s.kind), s.name)
                            })
                            .collect();
                        format!("Symbols in {rel_path}:\n{}", lines.join("\n"))
                    }
                }
            }
            "rename" => {
                let Some(new_name) = &params.new_name else {
                    return Ok(text_result(
                        "Error: new_name parameter required for rename",
                        json!({"action": action, "serverName": server_name, "success": false}),
                    ));
                };
                let result = client
                    .send_request(
                        "textDocument/rename",
                        json!({ "textDocument": { "uri": uri }, "position": position, "newName": new_name }),
                        Some(cancel),
                        None,
                    )
                    .await?;
                if result.is_null() {
                    "Rename returned no edits".to_string()
                } else {
                    let edit: omp_lsp::types::WorkspaceEdit = serde_json::from_value(result)?;
                    if params.apply == Some(false) {
                        let preview = format_workspace_edit_lines(&edit, cwd);
                        format!(
                            "Rename preview:\n{}",
                            preview
                                .iter()
                                .map(|p| format!("  {p}"))
                                .collect::<Vec<_>>()
                                .join("\n")
                        )
                    } else {
                        let applied = omp_lsp::edits::apply_workspace_edit(&edit, cwd).await?;
                        format!(
                            "Applied rename:\n{}",
                            applied
                                .iter()
                                .map(|a| format!("  {a}"))
                                .collect::<Vec<_>>()
                                .join("\n")
                        )
                    }
                }
            }
            "reload" => omp_lsp::servers::reload_server(&client, server_name, Some(cancel)).await?,
            other => format!("Unknown action: {other}"),
        };

        let mut details = json!({"serverName": server_name, "action": action, "success": true, "request": params.request_json()});
        if useless {
            details["useless"] = json!(true);
        }
        Ok(text_result(output, details))
    }

    /// `code_actions` listing + single-action application.
    #[allow(clippy::too_many_arguments)]
    async fn action_code_actions(
        self: &Arc<Self>,
        params: &Params,
        client: &omp_lsp::client::LspClient,
        uri: &str,
        position: &Value,
        reconciled_from_disk: bool,
        cwd: &Path,
        cancel: &DeadlineCancel<'_>,
        timeout_secs: u64,
        project_aware: bool,
    ) -> Result<String> {
        use omp_lsp::diagnostics as diag;
        let mut diagnostics = client
            .published_diagnostics(uri)
            .map(|p| p.diagnostics)
            .unwrap_or_default();
        if reconciled_from_disk {
            let wait_cap = if project_aware {
                diag::PROJECT_DIAGNOSTICS_WAIT_TIMEOUT_MS
            } else {
                diag::SINGLE_DIAGNOSTICS_WAIT_TIMEOUT_MS
            }
            .min(timeout_secs * 1000);
            let expected = client.open_file_version(uri);
            match diag::wait_for_diagnostics(client, uri, wait_cap, Some(cancel), None, expected)
                .await
            {
                Ok(ds) => diagnostics = ds,
                Err(_) => {
                    if let Some(published) = client.published_diagnostics(uri) {
                        diagnostics = published.diagnostics;
                    }
                }
            }
        }
        let only: Option<String> = if params.apply == Some(true) {
            None
        } else {
            params.query.clone()
        };
        let context = json!({
            "diagnostics": diagnostics,
            "only": only.map(|q| vec![q]),
            "triggerKind": 1,
        });
        let result = client
            .send_request(
                "textDocument/codeAction",
                json!({ "textDocument": { "uri": uri }, "range": { "start": position, "end": position }, "context": context }),
                Some(cancel),
                None,
            )
            .await?;
        let actions: Vec<Value> = result.as_array().cloned().unwrap_or_default();
        if actions.is_empty() {
            return Ok("No code actions available".to_string());
        }

        if params.apply == Some(true) {
            let Some(query) = params.query.as_deref().map(str::trim) else {
                return Ok(
                    "Error: query parameter required when apply=true for code_actions".to_string(),
                );
            };
            if query.is_empty() {
                return Ok(
                    "Error: query parameter required when apply=true for code_actions".to_string(),
                );
            }
            let selected: Option<Value> = if query.chars().all(|c| c.is_ascii_digit()) {
                let idx: usize = query.parse().unwrap_or(usize::MAX);
                actions.get(idx).cloned()
            } else {
                let needle = query.to_lowercase();
                actions
                    .iter()
                    .find(|a| {
                        a.get("title")
                            .and_then(Value::as_str)
                            .is_some_and(|t| t.to_lowercase().contains(&needle))
                    })
                    .cloned()
            };
            let Some(selected) = selected else {
                let lines: Vec<String> = actions
                    .iter()
                    .enumerate()
                    .map(|(i, a)| format!("  {}", format_code_action_line(a, i)))
                    .collect();
                return Ok(format!(
                    "No code action matches \"{query}\". Available actions:\n{}",
                    lines.join("\n")
                ));
            };
            return self.apply_code_action(&selected, client, cwd, cancel).await;
        }

        let lines: Vec<String> = actions
            .iter()
            .enumerate()
            .map(|(i, a)| format!("  {}", format_code_action_line(a, i)))
            .collect();
        Ok(format!(
            "{} code action(s):\n{}",
            actions.len(),
            lines.join("\n")
        ))
    }

    /// Apply one code action: resolve (if needed), apply its workspace edit,
    /// then execute its command (omp `applyCodeAction`).
    async fn apply_code_action(
        self: &Arc<Self>,
        action: &Value,
        client: &omp_lsp::client::LspClient,
        cwd: &Path,
        cancel: &DeadlineCancel<'_>,
    ) -> Result<String> {
        let title = action
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        // A Command item has a string `command` field.
        if let Some(command_name) = action.get("command").and_then(Value::as_str) {
            client
                .send_request(
                    "workspace/executeCommand",
                    json!({ "command": command_name, "arguments": action.get("arguments").cloned().unwrap_or(json!([])) }),
                    Some(cancel),
                    None,
                )
                .await?;
            return Ok(format!(
                "Applied \"{title}\":\n  Executed command(s):\n    {command_name}"
            ));
        }
        let mut resolved = action.clone();
        if resolved.get("edit").is_none() {
            if let Ok(r) = client
                .send_request("codeAction/resolve", action.clone(), Some(cancel), None)
                .await
            {
                resolved = r;
            }
        }
        let mut edits: Vec<String> = Vec::new();
        let mut executed: Vec<String> = Vec::new();
        if let Some(edit_value) = resolved.get("edit") {
            let edit: omp_lsp::types::WorkspaceEdit = serde_json::from_value(edit_value.clone())?;
            edits = omp_lsp::edits::apply_workspace_edit(&edit, cwd).await?;
        }
        if let Some(command) = resolved.get("command").and_then(Value::as_object) {
            let name = command.get("command").and_then(Value::as_str).unwrap_or("");
            client
                .send_request(
                    "workspace/executeCommand",
                    json!({
                        "command": name,
                        "arguments": command.get("arguments").cloned().unwrap_or(json!([]))
                    }),
                    Some(cancel),
                    None,
                )
                .await?;
            executed.push(name.to_string());
        }
        if edits.is_empty() && executed.is_empty() {
            return Ok(format!(
                "Action \"{title}\" has no workspace edit or command to apply"
            ));
        }
        let mut lines: Vec<String> = Vec::new();
        if !edits.is_empty() {
            lines.push("  Workspace edit:".to_string());
            lines.extend(edits.iter().map(|e| format!("    {e}")));
        }
        if !executed.is_empty() {
            lines.push("  Executed command(s):".to_string());
            lines.extend(executed.iter().map(|c| format!("    {c}")));
        }
        Ok(format!("Applied \"{title}\":\n{}", lines.join("\n")))
    }
}

fn symbol_icon(kind: u32) -> &'static str {
    match kind {
        1 => omp_lsp::utils::ICON_FILE,
        2 | 3 | 23 => omp_lsp::utils::ICON_FOLDER,
        4 => omp_lsp::utils::ICON_PACKAGE,
        5 | 11 => omp_lsp::utils::ICON_MODEL,
        6 | 9 | 12 => omp_lsp::utils::ICON_AUTO,
        _ => omp_lsp::utils::FORMAT_BULLET,
    }
}

fn format_document_symbol(symbol: &omp_lsp::types::DocumentSymbol, indent: usize) -> Vec<String> {
    let prefix = "  ".repeat(indent);
    let line = symbol.range.start.line + 1;
    let detail = symbol
        .detail
        .as_deref()
        .map(|d| format!(" {d}"))
        .unwrap_or_default();
    let mut out = vec![format!(
        "{}{} {}{detail} @ line {line}",
        prefix,
        symbol_icon(symbol.kind),
        symbol.name
    )];
    if let Some(children) = &symbol.children {
        for child in children {
            out.extend(format_document_symbol(child, indent + 1));
        }
    }
    out
}

fn format_code_action_line(action: &Value, index: usize) -> String {
    let title = action.get("title").and_then(Value::as_str).unwrap_or("");
    let kind = action
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("action");
    let preferred = action
        .get("isPreferred")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let disabled = action
        .get("disabled")
        .and_then(|d| d.get("reason"))
        .and_then(Value::as_str);
    omp_lsp::utils::format_code_action(title, Some(kind), preferred, disabled, index)
}
