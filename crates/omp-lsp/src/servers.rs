//! Server selection, read-only action classification and reload
//! (omp `lsp/servers.ts`, minus warmup/lspmux which the omp-rs host does not use).

use serde_json::{json, Value};

use crate::client::{self, LspClient, LspError, LspServerStatus};
use crate::config;
use crate::types::LspConfig;
use crate::types::ServerConfig;
use crate::types::WorkspaceEdit;
use crate::utils::uri_to_file;
use crate::Cancel;

/// LSP actions that do not mutate the workspace or language-server state.
pub const LSP_READONLY_ACTIONS: [&str; 9] = [
    "diagnostics",
    "definition",
    "type_definition",
    "implementation",
    "references",
    "hover",
    "symbols",
    "status",
    "capabilities",
];

/// Whether `action` is a read-only LSP action.
#[must_use]
pub fn is_readonly_action(action: &str) -> bool {
    LSP_READONLY_ACTIONS.contains(&action)
}

/// Status of currently active LSP servers.
#[must_use]
pub fn get_lsp_status() -> Vec<LspServerStatus> {
    client::get_active_clients()
}

/// True when a server is a plain LSP server (not a custom linter client).
fn is_custom_linter(_server_config: &ServerConfig) -> bool {
    // Custom linter clients (biome/swiftlint) are not ported; `load_config`
    // already dropped them, so no live config counts as one.
    false
}

/// All non-linter configured servers.
#[must_use]
pub fn get_lsp_servers(config: &LspConfig) -> Vec<(String, ServerConfig)> {
    config
        .servers
        .iter()
        .filter(|(_, c)| !is_custom_linter(c))
        .map(|(n, c)| (n.clone(), c.clone()))
        .collect()
}

/// Non-linter servers that can handle `file_path`.
#[must_use]
pub fn get_lsp_servers_for_file(
    config: &LspConfig,
    file_path: &std::path::Path,
) -> Vec<(String, ServerConfig)> {
    config::get_servers_for_file(config, file_path)
        .into_iter()
        .filter(|(_, c)| !is_custom_linter(c))
        .collect()
}

/// Primary non-linter server for `file_path`.
#[must_use]
pub fn get_lsp_server_for_file(
    config: &LspConfig,
    file_path: &std::path::Path,
) -> Option<(String, ServerConfig)> {
    get_lsp_servers_for_file(config, file_path)
        .into_iter()
        .next()
}

/// Project-aware servers compute pull diagnostics on demand and need an index
/// wait before semantic queries.
#[must_use]
pub fn is_project_aware_lsp_server(server_config: &ServerConfig) -> bool {
    !is_custom_linter(server_config) && server_config.is_linter != Some(true)
}

/// True when an LSP error indicates the server doesn't implement the method.
#[must_use]
pub fn is_method_not_found_error(err: &LspError) -> bool {
    let msg = err.message.to_lowercase();
    msg.contains("method not found")
        || msg.contains("unhandled method")
        || msg.contains("not supported")
        || msg.contains("-32601")
}

/// Build the params for `workspace/didChangeConfiguration` reload.
#[must_use]
pub fn reload_configuration_params(config: &ServerConfig) -> Value {
    json!({ "settings": config.settings.clone().unwrap_or(json!({})) })
}

/// Reload one server (omp `reloadServer`): rust-analyzer gets its real
/// reload request; everyone else is re-sent the didChangeConfiguration
/// notification. A wedged connection tears the client down so the next call
/// cold-starts a fresh one.
///
/// # Errors
///
/// Returns the teardown failure when the server process did not exit after kill.
pub async fn reload_server(
    lsp_client: &LspClient,
    server_name: &str,
    cancel: Option<&dyn Cancel>,
) -> Result<String, LspError> {
    if let Some(c) = cancel {
        if c.is_cancelled() {
            return Err(LspError {
                message: "operation cancelled".to_string(),
                cancelled: true,
            });
        }
    }
    if lsp_client.is_rust_analyzer() || server_name == "rust-analyzer" {
        match lsp_client
            .send_request("rust-analyzer/reloadWorkspace", Value::Null, cancel, None)
            .await
        {
            Ok(_) => return Ok(format!("Reloaded {server_name}")),
            Err(e) => {
                if let Some(c) = cancel {
                    if c.is_cancelled() {
                        return Err(LspError {
                            message: "operation cancelled".to_string(),
                            cancelled: true,
                        });
                    }
                }
                if !is_method_not_found_error(&e) {
                    return Err(e);
                }
                // Fall through to the generic reload.
            }
        }
    }
    let params = reload_configuration_params(&lsp_client.config().clone());
    if let Ok(()) = lsp_client
        .send_notification("workspace/didChangeConfiguration", params)
        .await
    {
    } else {
        if let Some(c) = cancel {
            if c.is_cancelled() {
                return Err(LspError {
                    message: "operation cancelled".to_string(),
                    cancelled: true,
                });
            }
        }
        // On a broker-shared link a per-session teardown only detaches this
        // process while the wedged server keeps serving everyone else — ask
        // the mux to kill the shared server first (best-effort; it also
        // severs us), like omp's `MUX_RESTART_METHOD` reload path.
        if lsp_client.is_shared_mux() {
            let _ = lsp_client
                .send_notification(crate::mux::protocol::MUX_RESTART_METHOD, Value::Null)
                .await;
        }
        if !lsp_client.shutdown_instance().await {
            return Err(LspError {
                message: format!(
                    "Failed to restart {server_name}: server process did not exit after kill"
                ),
                cancelled: false,
            });
        }
        return Ok(format!("Restarted {server_name}"));
    }
    // Refresh every open file from disk, like the TS reload path.
    let open_uris: Vec<String> = {
        // `open_files` is keyed canonically; rebuild URIs from disk paths.
        Vec::new()
    };
    let _ = open_uris;
    for open_path in lsp_client.open_file_paths() {
        let () = lsp_client.refresh_file(&open_path).await;
    }
    Ok(format!("Reloaded {server_name}"))
}

/// Warm a single server configuration up (initialize eagerly); used by hosts
/// that want startup warmup. Returns the display name on success.
///
/// # Errors
///
/// Initialization failures.
pub async fn warmup_server(
    server_config: &ServerConfig,
    cwd: &std::path::Path,
    warmup_timeout_ms: Option<u64>,
) -> Result<String, LspError> {
    let client = client::get_or_create_client(
        server_config,
        cwd,
        Some(std::time::Duration::from_millis(
            warmup_timeout_ms.unwrap_or(client::WARMUP_TIMEOUT_MS),
        )),
        None,
    )
    .await?;
    Ok(client.name().to_string())
}

/// Apply a workspace edit delivered through the LSP link (rename/code action),
/// formatting results like omp `applyWorkspaceEditWithLsp`.
///
/// # Errors
///
/// Propagates edit application failures.
pub async fn apply_workspace_edit_with_lsp(
    edit: &WorkspaceEdit,
    cwd: &std::path::Path,
) -> anyhow::Result<Vec<String>> {
    crate::edits::apply_workspace_edit(edit, cwd)
        .await
        .map_err(anyhow::Error::from)
}

/// Notify servers that a file was renamed, closing the old URIs (used by
/// `rename_file` after a successful move).
///
/// # Errors
///
/// Propagates `didClose`/`didRenameFiles` write failures.
pub async fn notify_did_rename(
    lsp_client: &LspClient,
    params: Value,
    old_uris: &[String],
) -> Result<(), LspError> {
    for old_uri in old_uris {
        if lsp_client.is_open(old_uri) {
            lsp_client.close_file(old_uri).await?;
        }
    }
    lsp_client
        .send_notification("workspace/didRenameFiles", params)
        .await
}

/// Convert a willRenameFiles/didRenameFiles `files` entry pair back to paths.
#[must_use]
pub fn rename_pair_paths(pair: &Value) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let old_uri = pair.get("oldUri").and_then(Value::as_str)?;
    let new_uri = pair.get("newUri").and_then(Value::as_str)?;
    Some((uri_to_file(old_uri), uri_to_file(new_uri)))
}
