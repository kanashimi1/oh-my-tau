//! `/mcp` slash command: MCP server management (ACP mode).

use super::{
    json, output, subcommand, usage, Handled, HashSet, Outcome, Path, PathBuf, Runtime, Value,
};

// =============================================================================
// /mcp (helpers/mcp.ts `handleMcpAcp`)
// =============================================================================

const MCP_HELP: &str = "MCP server management (ACP mode)
  /mcp list                                               List configured servers
  /mcp enable <name>                                      Enable a server
  /mcp disable <name>                                     Disable a server
  /mcp remove <name> [--scope project|user]               Remove a server
  /mcp reload                                             Reload MCP runtime
  /mcp resources                                          List resources from all servers
  /mcp prompts                                            List prompts from all servers
  /mcp test <name>                                        Test connection to a server
  /mcp add <name> [--scope project|user] [--url <url>]    Add a server (non-interactive)
  /mcp add <name> [-- <command...>]                       Add a stdio server
  /mcp smithery-search <kw> [--scope project|user]        Search Smithery registry
  /mcp help                                               Show this help";

/// Read one mcp.json file as a raw JSON value, preserving unknown fields.
fn read_mcp_file(path: &Path) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_else(|| json!({}))
}

fn mcp_servers_of(file: &Value) -> &serde_json::Map<String, Value> {
    static EMPTY: std::sync::LazyLock<serde_json::Map<String, Value>> =
        std::sync::LazyLock::new(serde_json::Map::new);
    file.get("mcpServers")
        .and_then(Value::as_object)
        .unwrap_or_else(|| &*EMPTY)
}

/// Write an mcp.json file back, creating the parent dir.
fn write_mcp_file(path: &Path, file: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| e.to_string())
}

/// `/mcp` scoped config path: user (`~/.omp/agent/mcp.json`) or the first
/// project candidate (`mcp.json` / `.mcp.json` / `.omp/mcp.json`).
// invariant: omp_mcp::config::project_config_paths always returns at least one candidate path.
#[allow(clippy::expect_used)]
fn mcp_scope_path(rt: &Runtime, scope: &str, create: bool) -> Result<PathBuf, String> {
    match scope {
        "user" => omp_mcp::config::user_config_path()
            .ok_or_else(|| "no HOME for user MCP config".to_string()),
        "project" => {
            let paths = omp_mcp::config::project_config_paths(&rt.cwd);
            if create {
                Ok(paths.first().cloned().expect("project paths non-empty"))
            } else {
                paths
                    .into_iter()
                    .find(|p| p.is_file())
                    .ok_or_else(|| "no project MCP config found".to_string())
            }
        }
        _ => Err("Invalid --scope value. Use project or user.".to_string()),
    }
}

/// One configured MCP server: `(name, scope, enabled, summary line)` (user entries shadow
/// project ones of the same name).
fn mcp_server_rows(rt: &Runtime) -> Vec<(String, &'static str, bool, String)> {
    let user_path = mcp_scope_path(rt, "user", false);
    let project_path = mcp_scope_path(rt, "project", false);
    let (Ok(user_path), Ok(project_path)) = (user_path, project_path) else {
        return Vec::new();
    };
    let user = read_mcp_file(&user_path);
    let project = read_mcp_file(&project_path);
    let disabled: HashSet<String> = user
        .get("disabledServers")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let mut entries: Vec<(String, Value, &'static str)> = Vec::new();
    for (name, config) in mcp_servers_of(&user) {
        entries.push((name.clone(), config.clone(), "user"));
    }
    for (name, config) in mcp_servers_of(&project) {
        if entries.iter().all(|(n, _, _)| n != name) {
            entries.push((name.clone(), config.clone(), "project"));
        }
    }
    entries
        .into_iter()
        .map(|(name, config, scope)| {
            let ty = config
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("stdio");
            let enabled = config.get("enabled").and_then(Value::as_bool) != Some(false)
                && !disabled.contains(&name);
            let location = match ty {
                "http" | "sse" => config
                    .get("url")
                    .and_then(Value::as_str)
                    .map_or("(unknown)", |u| u.split('?').next().unwrap_or(u)),
                _ => config
                    .get("command")
                    .and_then(Value::as_str)
                    .unwrap_or("(unknown)"),
            };
            let summary = format!(
                "[{scope}] ({ty}, {}) — {location}",
                if enabled { "enabled" } else { "disabled" }
            );
            (name, scope, enabled, summary)
        })
        .collect()
}

fn mcp_list(rt: &Runtime) -> Outcome {
    let rows = mcp_server_rows(rt);
    if rows.is_empty() {
        output(rt, "No MCP servers configured.");
        return Handled {
            agent_invoked: false,
        };
    }
    let lines: Vec<String> = rows
        .iter()
        .map(|(name, _, _, summary)| format!("* {name} {summary}"))
        .collect();
    output(rt, &lines.join("\n"));
    Handled {
        agent_invoked: false,
    }
}

/// Bare `/mcp` with a dialog host (TS interactive MCP menu): pick a server, then an action.
async fn mcp_menu(rt: &Runtime) -> Outcome {
    use crate::slash_extra::dialog::{self, Choice, SelectStyle};
    let rows = mcp_server_rows(rt);
    let mut choices: Vec<Choice> = rows
        .iter()
        .map(|(name, _, _, summary)| Choice::with(name.clone(), summary.clone()))
        .collect();
    choices.push(Choice::with(
        "Add server…",
        "/mcp add <name> [--scope project|user] -- <command…> | --url <url>",
    ));
    choices.push(Choice::with("Help", "MCP subcommands"));
    let title = if rows.is_empty() {
        "MCP servers (none configured)"
    } else {
        "MCP servers"
    };
    let Some(index) = dialog::select_with(rt, title, &choices, SelectStyle::default()).await else {
        return Handled {
            agent_invoked: false,
        };
    };
    if index == rows.len() {
        let Some(spec) = dialog::input(
            rt,
            "Add MCP server",
            Some("<name> [--scope project|user] -- <command…> | --url <url>"),
        )
        .await
        else {
            return Handled {
                agent_invoked: false,
            };
        };
        if spec.trim().is_empty() {
            return Handled {
                agent_invoked: false,
            };
        }
        return mcp_add(rt, spec.trim());
    }
    if index > rows.len() {
        output(rt, MCP_HELP);
        return Handled {
            agent_invoked: false,
        };
    }
    let (name, scope, enabled, _) = &rows[index];
    let actions = [
        "Test connection".to_string(),
        if *enabled {
            "Disable".to_string()
        } else {
            "Enable".to_string()
        },
        "Remove".to_string(),
    ];
    let Some(action) = dialog::select(rt, &format!("MCP server: {name}"), &actions).await else {
        return Handled {
            agent_invoked: false,
        };
    };
    match action {
        0 => Box::pin(mcp_command(rt, &format!("test {name}"))).await,
        1 => mcp_set_enabled(rt, name, scope, !enabled),
        _ => {
            if dialog::confirm(
                rt,
                "Remove MCP server",
                &format!("Remove \"{name}\" from the {scope} config?"),
            )
            .await
                == Some(true)
            {
                mcp_remove(rt, name, scope)
            } else {
                Handled {
                    agent_invoked: false,
                }
            }
        }
    }
}

// invariant: the or_insert(json!({})) above guarantees mcpServers is an object.
#[allow(clippy::expect_used)]
fn mcp_set_enabled(rt: &Runtime, name: &str, scope: &str, enabled: bool) -> Outcome {
    let Ok(path) = mcp_scope_path(rt, scope, true) else {
        return usage(rt, &format!("No {scope} MCP config to modify."));
    };
    let mut file = read_mcp_file(&path);
    let servers = file
        .as_object_mut()
        .map(|o| {
            o.entry("mcpServers".to_string())
                .or_insert_with(|| json!({}))
        })
        .and_then(Value::as_object_mut)
        .expect("mcpServers inserted as object");
    let Some(server) = servers.get_mut(name) else {
        return usage(rt, &format!("No {scope} MCP server named \"{name}\"."));
    };
    let Some(obj) = server.as_object_mut() else {
        return usage(
            rt,
            &format!("MCP server \"{name}\" config is not an object."),
        );
    };
    if enabled {
        obj.remove("enabled");
        if let Some(arr) = file
            .get_mut("disabledServers")
            .and_then(Value::as_array_mut)
        {
            arr.retain(|v| v.as_str() != Some(name));
        }
    } else {
        obj.insert("enabled".into(), Value::Bool(false));
    }
    match write_mcp_file(&path, &file) {
        Ok(()) => output(
            rt,
            &format!("{} {name}.", if enabled { "Enabled" } else { "Disabled" }),
        ),
        Err(e) => return usage(rt, &format!("Failed to write {}: {e}", path.display())),
    }
    Handled {
        agent_invoked: false,
    }
}

fn mcp_remove(rt: &Runtime, name: &str, scope: &str) -> Outcome {
    let Ok(path) = mcp_scope_path(rt, scope, false) else {
        return usage(rt, &format!("No {scope} MCP config found."));
    };
    let mut file = read_mcp_file(&path);
    let removed = file
        .as_object_mut()
        .and_then(|o| o.get_mut("mcpServers"))
        .and_then(Value::as_object_mut)
        .is_some_and(|servers| servers.remove(name).is_some());
    if !removed {
        return usage(rt, &format!("No {scope} MCP server named \"{name}\"."));
    }
    match write_mcp_file(&path, &file) {
        Ok(()) => {
            output(rt, &format!("Removed {name} from {}.", path.display()));
            Handled {
                agent_invoked: false,
            }
        }
        Err(e) => usage(rt, &format!("Failed to write {}: {e}", path.display())),
    }
}

// invariant: the or_insert(json!({})) above guarantees mcpServers is an object.
#[allow(clippy::expect_used)]
fn mcp_add(rt: &Runtime, rest: &str) -> Outcome {
    const ADD_USAGE: &str = "Usage: /mcp add <name> [--scope project|user] [--url <url> --transport http|sse] [--token <token>] [-- <command...>]";
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    let mut scope = "project";
    let mut url: Option<String> = None;
    let mut transport = "http";
    let mut command: Vec<String> = Vec::new();
    let mut name: Option<String> = None;
    let mut i = 0;
    if let Some(first) = tokens.first() {
        if !first.starts_with('-') {
            name = Some((*first).to_string());
            i = 1;
        }
    }
    while i < tokens.len() {
        match tokens[i] {
            "--" => {
                command = tokens[i + 1..].iter().map(|s| (*s).to_string()).collect();
                break;
            }
            "--scope" => match tokens.get(i + 1) {
                Some(&v) if v == "project" || v == "user" => {
                    scope = v;
                    i += 2;
                }
                _ => return usage(rt, "Invalid --scope value. Use project or user."),
            },
            "--url" => match tokens.get(i + 1) {
                Some(v) => {
                    url = Some((*v).to_string());
                    i += 2;
                }
                None => return usage(rt, "Missing value for --url."),
            },
            "--transport" => match tokens.get(i + 1) {
                Some(&v) if v == "http" || v == "sse" => {
                    transport = v;
                    i += 2;
                }
                _ => return usage(rt, "Invalid --transport value. Use http or sse."),
            },
            "--token" => {
                return usage(
                    rt,
                    "--token is only persisted by the TUI MCP add flow in omt; use headers in the config file.",
                );
            }
            other => return usage(rt, &format!("Unknown option: {other}")),
        }
    }
    let Some(name) = name else {
        return usage(rt, "Server name required. Usage: /mcp add <name> ...");
    };
    if url.is_none() && command.is_empty() {
        return usage(
            rt,
            &format!("Provide --url or -- <command...> for non-interactive add. {ADD_USAGE}"),
        );
    }
    if url.is_some() && !command.is_empty() {
        return usage(rt, "Use either --url or -- <command...>, not both.");
    }
    let Ok(path) = mcp_scope_path(rt, scope, true) else {
        return usage(rt, "No MCP config path available.");
    };
    let mut file = read_mcp_file(&path);
    let servers = file
        .as_object_mut()
        .map(|o| {
            o.entry("mcpServers".to_string())
                .or_insert_with(|| json!({}))
        })
        .and_then(Value::as_object_mut)
        .expect("mcpServers inserted as object");
    let config = if let Some(url) = url {
        json!({"type": transport, "url": url})
    } else {
        let mut cmd = command.into_iter();
        let head = cmd.next().unwrap_or_default();
        let args: Vec<String> = cmd.collect();
        let mut v = json!({"command": head});
        if !args.is_empty() {
            v["args"] = json!(args);
        }
        v
    };
    let existed = servers.insert(name.clone(), config).is_some();
    match write_mcp_file(&path, &file) {
        Ok(()) => output(
            rt,
            &format!(
                "{} {name} in {} ({}). Restart to load its tools.",
                if existed { "Updated" } else { "Added" },
                path.display(),
                scope
            ),
        ),
        Err(e) => return usage(rt, &format!("Failed to write {}: {e}", path.display())),
    }
    Handled {
        agent_invoked: false,
    }
}

/// `/mcp <subcommand>` (TS `handleMcpAcp`; smithery/oauth verbs are TUI-only).
pub(super) async fn mcp_command(rt: &Runtime, args: &str) -> Outcome {
    let (verb, rest) = subcommand(args);
    match verb.as_str() {
        "" if crate::slash_extra::dialog::has_ui() => mcp_menu(rt).await,
        "" | "help" => {
            output(rt, MCP_HELP);
            Handled {
                agent_invoked: false,
            }
        }
        "notifications" => usage(
            rt,
            "MCP notifications require the TUI client (live MCPManager). Use /mcp list to see server status.",
        ),
        "reauth" | "unauth" | "smithery-login" | "smithery-logout" | "reconnect" => usage(
            rt,
            &format!(
                "/mcp {verb} requires OAuth or browser flows only available in the TUI client."
            ),
        ),
        "smithery-search" => usage(
            rt,
            "/mcp smithery-search needs the Smithery registry client, which is not ported to omt.",
        ),
        "list" => mcp_list(rt),
        "add" => mcp_add(rt, &rest),
        "remove" | "rm" => {
            let (name, scope) = match rest.split_once("--scope") {
                Some((n, s)) => (n.trim().to_string(), s.trim().to_string()),
                None => (rest.trim().to_string(), "project".to_string()),
            };
            if name.is_empty() {
                return usage(rt, "Usage: /mcp remove <name> [--scope project|user]");
            }
            mcp_remove(rt, &name, &scope)
        }
        "enable" | "disable" => {
            let enabled = verb == "enable";
            let (name, scope) = match rest.split_once("--scope") {
                Some((n, s)) => (n.trim().to_string(), s.trim().to_string()),
                None => (rest.trim().to_string(), "project".to_string()),
            };
            if name.is_empty() {
                return usage(
                    rt,
                    &format!("Usage: /mcp {verb} <name> [--scope project|user]"),
                );
            }
            mcp_set_enabled(rt, &name, &scope, enabled)
        }
        "reload" => {
            output(
                rt,
                "MCP runtime reload requested. (New tools apply on next session; the live agent toolset is fixed at startup.)",
            );
            Handled {
                agent_invoked: false,
            }
        }
        "test" => {
            let name = rest.trim();
            if name.is_empty() {
                return usage(rt, "Usage: /mcp test <name>");
            }
            let configs = omp_mcp::config::load_configs(&rt.cwd);
            let Some(config) = configs.get(name) else {
                return usage(rt, &format!("No MCP server named \"{name}\"."));
            };
            let cwd = std::sync::Arc::new(rt.cwd.clone());
            match omp_mcp::client::connect_to_server(name, config, cwd).await {
                Ok(mut connection) => {
                    let tools = omp_mcp::client::list_tools(&mut connection).await;
                    match tools {
                        Ok(tools) => {
                            output(
                                rt,
                                &format!("Connected to {name}. {} tool(s) available.", tools.len()),
                            );
                            Handled {
                                agent_invoked: false,
                            }
                        }
                        Err(e) => {
                            output(
                                rt,
                                &format!("Connected to {name}, but listing tools failed: {e}"),
                            );
                            Handled {
                                agent_invoked: false,
                            }
                        }
                    }
                }
                Err(e) => usage(rt, &format!("Failed to connect to {name}: {e}")),
            }
        }
        "resources" | "prompts" => usage(
            rt,
            &format!(
                "/mcp {verb} needs the MCPManager resource/prompt listing, which the omt MCP client does not implement (tools only)."
            ),
        ),
        _ => usage(
            rt,
            &format!("Unknown /mcp subcommand: {verb}. Use /mcp help for available subcommands."),
        ),
    }
}
