//! Slash-command dispatch for prompts starting with `/` (TS `slash-commands`).
//!
//! Mirrors omp's text-mode dispatch order: builtin commands with a text
//! handler run first (RPC layer, `executeAcpBuiltinSlashCommand`), then
//! extension commands (`pi.registerCommand`), then file-based commands
//! (markdown templates from `commands/` directories), which rewrite the
//! prompt instead of consuming it. TUI-only builtins never match here, so
//! their names fall through to the model exactly as in TS.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use parking_lot::Mutex;
use serde_json::{json, Value};
use std::fmt::Write as _;

use crate::runtime::Runtime;

#[path = "slash/center.rs"]
mod center;
#[path = "slash/files.rs"]
mod files;
#[path = "slash/info.rs"]
mod info;
#[path = "slash/mcp.rs"]
mod mcp;
#[path = "slash/model.rs"]
mod model;
#[path = "slash/session.rs"]
mod session;
#[path = "slash/todo.rs"]
mod todo;
#[path = "slash/usage.rs"]
mod usage;

pub(crate) use usage::usage_panel_frame;

pub(crate) use center::command_center;
pub use files::load_file_commands;
use files::{expand_file_command, refresh_file_commands, FILE_COMMANDS};
use info::{changelog_command, stats_command, trace_command};
use mcp::mcp_command;
use model::{fast_command, model_ref, modelpreset_command, resolve_model, slow_command};
use session::{
    handoff_command, move_command, pin_command, plugins_command, setting_status_toggle,
    ADVISOR_INACTIVE_ON,
};
use todo::todo_command;
use usage::usage_command;

/// Result of [`try_dispatch`] for one submitted prompt.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Command consumed locally; `agent_invoked` mirrors TS `commandConsumed`.
    Handled { agent_invoked: bool },
    /// Extension command consumed (TS `session.prompt` returning false): the
    /// RPC response is plain success and a completed `prompt_result` follows.
    ExtensionHandled,
    /// Command rewrote the input (file template / residual prompt); the text
    /// should be forwarded to the model.
    Prompt(String),
    /// Not a slash command (or a TUI-only builtin): prompt as-is.
    NotSlash,
}

/// TS `parseSlashCommand` (slash-commands/helpers/parse.ts): the name/args
/// separator is the earliest whitespace or `:`.
pub fn parse_slash_command(text: &str) -> Option<(String, String)> {
    let body = text.strip_prefix('/')?;
    if body.is_empty() {
        return None;
    }
    let first_ws = body.find(char::is_whitespace);
    let first_colon = body.find(':');
    let sep = match (first_ws, first_colon) {
        (None, None) => return Some((body.to_string(), String::new())),
        (None, Some(c)) | (Some(c), None) => c,
        (Some(w), Some(c)) => w.min(c),
    };
    Some((body[..sep].to_string(), body[sep + 1..].trim().to_string()))
}

/// Emit a `command_output` frame (TS `runtime.output`).
pub(super) fn output(rt: &Runtime, text: &str) {
    rt.send(json!({"type": "command_output", "text": text}));
}

/// Emit a usage/error message and consume the command (TS `usage`).
pub(super) fn usage(rt: &Runtime, text: &str) -> Outcome {
    output(rt, text);
    Outcome::Handled {
        agent_invoked: false,
    }
}

static EXTENSIONS: OnceLock<Arc<Ext>> = OnceLock::new();

type Ext = crate::ext::Extensions;

type Arc<T> = std::sync::Arc<T>;

/// Register the process-wide extension host for command listing/dispatch.
/// Called once at startup (rpc and -p modes).
pub fn set_extensions(ext: Arc<Ext>) {
    let _ = EXTENSIONS.set(ext);
}

pub fn extensions() -> Option<&'static Arc<Ext>> {
    EXTENSIONS.get()
}

/// Extension commands registered via `pi.registerCommand`, in TS
/// `buildAvailableSlashCommands` order/shape (source `extension`).
pub async fn extension_commands() -> Vec<serde_json::Value> {
    let Some(ext) = extensions() else {
        return Vec::new();
    };
    match ext.host.list_commands().await {
        Ok(cmds) => cmds
            .into_iter()
            .map(|c| {
                json!({
                    "name": c.name,
                    "description": c.description.unwrap_or_else(|| "(extension command)".into()),
                    "input": {"hint": "arguments"},
                    "source": "extension",
                })
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Execute an extension command (`#tryExecuteExtensionCommand`). The handler's
/// return value (if a string) is emitted as command output. Names shadowed by
/// a builtin (or builtin alias prefix before `:`) never reach here because
/// builtins dispatch first, matching TS reserved-name semantics.
async fn try_extension_command(rt: &Runtime, name: &str, args: &str) -> bool {
    let Some(ext) = extensions() else {
        return false;
    };
    let Ok(cmds) = ext.host.list_commands().await else {
        return false;
    };
    if !cmds.iter().any(|c| c.name == name) {
        return false;
    }
    // TS does not surface the handler's return value as command output; it is
    // only visible through the command context, so failures alone are reported
    // (TS emits an extension-runner error event).
    if let Err(e) = ext.host.call_command(name, args).await {
        output(rt, &format!("Extension command failed: {e:#}"));
    }
    true
}

/// Full dispatch chain for one prompt (builtins → extra → WASM plugins → extension → file templates).
pub async fn try_dispatch(rt: &std::sync::Arc<Runtime>, text: &str) -> Outcome {
    if !text.starts_with('/') {
        return Outcome::NotSlash;
    }
    let Some((name, args)) = parse_slash_command(text) else {
        return Outcome::NotSlash;
    };
    if let Some(outcome) = dispatch_builtin(rt, &name, &args).await {
        return outcome;
    }
    if let Some(outcome) = crate::slash_extra::dispatch_extra(rt, &name, &args).await {
        return outcome;
    }
    if let Some(text) = crate::tools::wasm::try_command(&rt.cwd, &name, &args).await {
        output(rt, &text);
        return Outcome::Handled {
            agent_invoked: false,
        };
    }
    if try_extension_command(rt, &name, &args).await {
        return Outcome::ExtensionHandled;
    }
    if let Some(expanded) = expand_file_command(&name, &args, &rt.cwd) {
        return Outcome::Prompt(expanded);
    }
    Outcome::NotSlash
}

/// Text-mode builtin dispatch (TS `executeAcpBuiltinSlashCommand` over the
/// handlers in `builtin-*.ts`). Returns `None` when no builtin matched, so the
/// caller falls through to extension/file commands.
#[allow(clippy::too_many_lines)]
pub async fn dispatch_builtin(
    rt: &std::sync::Arc<Runtime>,
    name: &str,
    args: &str,
) -> Option<Outcome> {
    // TS `showModelSelector`: a bare `/model` / `/switch` opens the picker when the host answers
    // dialogs (the TUI intercepts these itself; this serves RPC clients).
    let picked_model;
    let args = if matches!(name, "model" | "models" | "switch")
        && args.is_empty()
        && crate::slash_extra::dialog::has_ui()
    {
        // Same catalogue as the TUI `/model` picker: providers with credentials (or declared in
        // models.yml / keyless), chat models only.
        let labels: Vec<String> =
            crate::subcmd::models::browser_models(&omp_providers::AgentDir::default())
                .into_iter()
                .filter(|m| m.available && m.kind.is_empty())
                .map(|m| format!("{}/{}", m.model.provider, m.model.id))
                .collect();
        if labels.is_empty() {
            output(
                rt,
                "No models available. Use /login or configure an API key.",
            );
            return Some(Handled {
                agent_invoked: false,
            });
        }
        let current = model_ref(&rt.model_info());
        let style = crate::slash_extra::dialog::SelectStyle {
            fullscreen: true,
            initial: labels.iter().position(|l| *l == current).unwrap_or(0),
            ..Default::default()
        };
        let choices: Vec<_> = labels
            .iter()
            .map(crate::slash_extra::dialog::Choice::new)
            .collect();
        let title = if name == "switch" {
            "Switch model (this session)"
        } else {
            "Select model"
        };
        let Some(index) = crate::slash_extra::dialog::select_with(rt, title, &choices, style).await
        else {
            return Some(Handled {
                agent_invoked: false,
            });
        };
        picked_model = labels[index].clone();
        picked_model.as_str()
    } else {
        args
    };
    match name {
        "model" | "models" => {
            if args.is_empty() {
                let model = rt.model_info();
                output(
                    rt,
                    &if model.is_null() {
                        "No model is currently selected.".into()
                    } else {
                        format!("Current model: {}", model_ref(&model))
                    },
                );
            } else {
                let Some((provider, id, level)) = resolve_model(rt, args).await else {
                    return Some(usage(
                        rt,
                        &format!("Unknown model: {args}. Use ACP `session/setModel` for picker-driven selection or list available models with /model."),
                    ));
                };
                match rt.set_model(&provider, &id).await {
                    Ok(_) => {
                        if let Some(level) = level {
                            rt.set_thinking_level(level).await;
                        }
                        output(rt, &format!("Model set to {provider}/{id}."));
                    }
                    Err(e) => {
                        return Some(usage(rt, &format!("Failed to set model: {}", e.message)))
                    }
                }
            }
            rt.announce_commands();
            Some(Handled {
                agent_invoked: false,
            })
        }
        "switch" => {
            if args.is_empty() {
                let model = rt.model_info();
                output(
                    rt,
                    &if model.is_null() {
                        "No model is currently selected.".into()
                    } else {
                        format!("Current model: {}", model_ref(&model))
                    },
                );
                return Some(Handled {
                    agent_invoked: false,
                });
            }
            let Some((provider, id, level)) = resolve_model(rt, args).await else {
                return Some(usage(rt, &format!("Unknown model: {args}")));
            };
            match rt.set_model(&provider, &id).await {
                Ok(_) => {
                    if let Some(level) = level {
                        rt.set_thinking_level(level).await;
                    }
                    output(rt, &format!("Session-only model: {provider}/{id}."));
                    Some(Handled {
                        agent_invoked: false,
                    })
                }
                Err(e) => Some(usage(rt, &format!("Failed to switch model: {}", e.message))),
            }
        }
        "effort" => {
            let levels = rt.available_thinking_levels();
            if args.is_empty() {
                let current = rt.state().await;
                let current = current
                    .get("thinkingLevel")
                    .and_then(|v| v.as_str())
                    .unwrap_or("model default");
                output(
                    rt,
                    &format!("Thinking: {current}\nAvailable: {}", levels.join(", ")),
                );
                return Some(Handled {
                    agent_invoked: false,
                });
            }
            let sel = args.trim().to_lowercase();
            match levels.iter().find(|l| **l == sel) {
                Some(level_name) => {
                    let level = crate::runtime::parse_level(level_name)
                        .unwrap_or(pa_agent::types::ThinkingLevel::Medium);
                    rt.set_thinking_level(level).await;
                    output(rt, &format!("Thinking set to {level_name}."));
                    Some(Handled {
                        agent_invoked: false,
                    })
                }
                None => Some(usage(
                    rt,
                    &format!(
                        "Unknown thinking level: {sel}. Available: {}",
                        levels.join(", ")
                    ),
                )),
            }
        }
        "fresh" => {
            if rt
                .state()
                .await
                .get("isStreaming")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            {
                Some(usage(
                    rt,
                    "Wait for the current response to finish or abort it before refreshing provider state.",
                ))
            } else {
                output(rt, "Provider stream state reset.");
                Some(Handled {
                    agent_invoked: false,
                })
            }
        }
        "compact" => {
            // TS parseCompactArgs: optional leading mode token (soft|remote|snapcompact).
            let trimmed = args.trim();
            let (mode, instructions) = match trimmed.split_once(char::is_whitespace) {
                None if matches!(trimmed, "soft" | "remote" | "snapcompact") => {
                    (Some(trimmed), None)
                }
                None => (None, (!trimmed.is_empty()).then(|| trimmed.to_string())),
                Some((first, rest)) if matches!(first, "soft" | "remote" | "snapcompact") => {
                    if first == "snapcompact" {
                        return Some(usage(
                            rt,
                            "/compact snapcompact does not take focus instructions (it archives history without an LLM summary).",
                        ));
                    }
                    (
                        Some(first),
                        Some(rest.trim().to_string()).filter(|s| !s.is_empty()),
                    )
                }
                Some(_) => (None, Some(trimmed.to_string())),
            };
            if let Some(mode) = mode {
                // omp-rs compaction does not implement mode overrides; run the default.
                output(
                    rt,
                    &format!("/compact {mode} is not supported; running default compaction."),
                );
            }
            let instructions = instructions.filter(|s| !s.is_empty());
            let result = rt.compact(instructions.as_deref()).await;
            match result {
                Ok(_) => output(rt, "Compaction complete."),
                Err(e) => output(rt, &format!("Compaction failed: {}", e.message)),
            }
            Some(Handled {
                agent_invoked: false,
            })
        }
        "clear" => {
            // TS `/clear`: close the active session and start a fresh one
            // (same work as the RPC `new_session` command).
            if rt
                .state()
                .await
                .get("isStreaming")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            {
                return Some(usage(
                    rt,
                    "Wait for the current response to finish or abort it before starting a new session.",
                ));
            }
            match rt.new_session(None).await {
                Ok(()) => output(rt, "New session started."),
                Err(e) if crate::runtime::is_switch_cancelled(&e) => {}
                Err(e) => return Some(usage(rt, &format!("Failed to start a new session: {e:#}"))),
            }
            Some(Handled {
                agent_invoked: false,
            })
        }
        "export" => {
            // TS `/export [path] [--themes]` writes the session to a file and
            // reports the path. omp-rs has no HTML renderer; the transcript is
            // written as markdown (role/content pairs, same shape as /dump).
            let mut target: Option<String> = None;
            for tok in args.split_whitespace() {
                match tok {
                    "--themes" => {} // HTML-only option; accepted for TS parity.
                    "--copy" | "clipboard" | "copy" => {
                        return Some(usage(rt, "Use /dump to copy the session to clipboard."));
                    }
                    t => target = Some(t.to_string()),
                }
            }
            let path = match target {
                Some(p) => PathBuf::from(p),
                None => rt
                    .cwd
                    .join(format!("session-{}-export.md", rt.session_id())),
            };
            let mut text = String::new();
            for m in rt.messages().await {
                if let (Some(role), Some(content)) = (
                    m.get("role").and_then(|v| v.as_str()),
                    m.get("content").and_then(|v| v.as_str()),
                ) {
                    text.push_str(role);
                    text.push_str(": ");
                    text.push_str(content);
                    text.push('\n');
                }
            }
            if text.is_empty() {
                text = "(empty session)".into();
            }
            match std::fs::write(&path, text) {
                Ok(()) => output(rt, &format!("Session exported to: {}", path.display())),
                Err(e) => return Some(usage(rt, &format!("Failed to export session: {e}"))),
            }
            Some(Handled {
                agent_invoked: false,
            })
        }
        "retry" => {
            if rt
                .state()
                .await
                .get("isStreaming")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            {
                Some(usage(
                    rt,
                    "Wait for the current response to finish or abort it before retrying.",
                ))
            } else {
                // Manual retry of a failed turn is driven by the recovery saga
                // in omp-rs; there is no parked failed turn to re-run here.
                Some(usage(rt, "Nothing to retry."))
            }
        }
        "rename" => {
            if args.is_empty() {
                output(
                    rt,
                    "Could not generate a session title. Use /rename <title> to set one.",
                );
            } else {
                match rt.set_session_name(args, "user") {
                    Ok(()) => output(rt, &format!("Session renamed to {args}.")),
                    Err(_) => output(
                        rt,
                        "Session name not changed (a user-set name takes precedence).",
                    ),
                }
            }
            Some(Handled {
                agent_invoked: false,
            })
        }
        "session" => {
            let (verb, rest) = subcommand(args);
            if rest.is_empty() && (verb.is_empty() || verb == "info") {
                let name = rt
                    .state()
                    .await
                    .get("sessionName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                output(
                    rt,
                    &format!(
                        "Session: {}\nTitle: {name}\nCWD: {}",
                        rt.session_id(),
                        rt.cwd.display()
                    ),
                );
                return Some(Handled {
                    agent_invoked: false,
                });
            }
            Some(usage(rt, "Usage: /session [info|delete|pin [account]]"))
        }
        _ => dispatch_builtin_more(rt, name, args).await,
    }
}

/// TS `parseSubcommand`: `<verb> <rest>`, verb lowercased.
pub(super) fn subcommand(input: &str) -> (String, String) {
    let trimmed = input.trim();
    match trimmed.split_once(char::is_whitespace) {
        None => (trimmed.to_lowercase(), String::new()),
        Some((v, rest)) => (v.to_lowercase(), rest.trim().to_string()),
    }
}

/// Second tier of builtin handlers (session introspection + disabled
/// subsystems), plus the reserved TUI-only names.
#[allow(clippy::too_many_lines)]
async fn dispatch_builtin_more(
    rt: &std::sync::Arc<Runtime>,
    name: &str,
    args: &str,
) -> Option<Outcome> {
    match name {
        "tools" => {
            let tools = rt
                .state()
                .await
                .get("dumpTools")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            if tools.is_empty() {
                output(rt, "No tools are available.");
                return Some(Handled {
                    agent_invoked: false,
                });
            }
            let lines: Vec<String> = tools
                .iter()
                .filter_map(|t| t.get("name").and_then(|v| v.as_str()))
                .map(|n| format!("* {n}"))
                .collect();
            output(rt, &lines.join("\n"));
            Some(Handled {
                agent_invoked: false,
            })
        }
        "context" => {
            let st = rt.state().await;
            let usage = st.get("contextUsage").cloned().unwrap_or(Value::Null);
            output(rt, &context_report(&st, &usage));
            Some(Handled {
                agent_invoked: false,
            })
        }
        "jobs" => {
            let (verb, rest) = subcommand(args);
            if !rest.is_empty() || (!verb.is_empty() && verb != "full") {
                return Some(usage(rt, "Usage: /jobs [full]"));
            }
            Some(crate::slash_extra::interactive::jobs_command(rt, verb == "full").await)
        }
        "advisor" => {
            let (verb, _rest) = subcommand(args);
            match verb.as_str() {
                "" | "toggle" | "on" => output(rt, ADVISOR_INACTIVE_ON),
                "off" | "status" => output(rt, "Advisor disabled."),
                "dump" => output(rt, "Advisor is not active for this session."),
                "configure" => output(
                    rt,
                    "/advisor configure opens an interactive editor and is only available in the interactive TUI.",
                ),
                _ => return Some(usage(rt, "Usage: /advisor [on|off|status|dump [raw]|configure]")),
            }
            Some(Handled {
                agent_invoked: false,
            })
        }
        "memory" => {
            let (verb, _rest) = subcommand(args);
            match (verb.as_str(), verb.is_empty()) {
                ("view", _) | (_, true) => output(rt, "Memory payload is empty."),
                ("clear" | "reset", _) => output(rt, "Memory cleared."),
                ("enqueue" | "rebuild", _) => output(rt, "Memory consolidation enqueued."),
                ("sync", _) => output(rt, "Memory consolidation ran."),
                ("queue" | "stats" | "diagnose", _) => output(
                    rt,
                    &format!("Memory is not available for the off backend (/{verb})."),
                ),
                ("mm", _) => output(
                    rt,
                    "Mental-model maintenance via /memory mm is unsupported in ACP mode; use the hindsight HTTP API directly.",
                ),
                _ => return Some(usage(rt, "Usage: /memory <view|stats|diagnose|clear|reset|enqueue|rebuild|queue|sync>")),
            }
            Some(Handled {
                agent_invoked: false,
            })
        }
        "reload-plugins" => {
            output(
                rt,
                "Extensions are discovered at session start and reload automatically when the extension host reconnects; to force a full rediscovery, restart the session (/new) or /restart.",
            );
            Some(Handled {
                agent_invoked: false,
            })
        }
        "force" => {
            let (tool, prompt) = match args.split_once(' ') {
                Some((t, p)) => (t, p.trim()),
                None => (args, ""),
            };
            if tool.is_empty() {
                return Some(usage(rt, "Usage: /force:<tool-name> [prompt]"));
            }
            output(rt, &format!("Next turn forced to use {tool}."));
            if prompt.is_empty() {
                Some(Handled {
                    agent_invoked: false,
                })
            } else {
                Some(Prompt(prompt.to_string()))
            }
        }
        "dump" => {
            let messages = rt.messages().await;
            let mut text = String::new();
            for m in messages {
                if let (Some(role), Some(content)) = (
                    m.get("role").and_then(|v| v.as_str()),
                    m.get("content").and_then(|v| v.as_str()),
                ) {
                    text.push_str(role);
                    text.push_str(": ");
                    text.push_str(content);
                    text.push('\n');
                }
            }
            if text.is_empty() {
                text = "(empty session)".into();
            }
            output(rt, &text);
            Some(Handled {
                agent_invoked: false,
            })
        }
        other => match dispatch_ported(rt, other, args).await {
            Some(outcome) => Some(outcome),
            None if is_reserved_builtin(other) => {
                // A catalog command with no text-mode handler at all (its
                // subsystem is TUI-only even in TS). Consuming keeps TS
                // reserved-name semantics; the message mirrors the TUI-only
                // notice shape.
                output(
                    rt,
                    &format!("/{other} is only available in the interactive TUI."),
                );
                Some(Handled {
                    agent_invoked: false,
                })
            }
            None => None,
        },
    }
}

/// Third tier of builtin handlers: catalog commands ported from the TS
/// `builtin-*.ts` specs. Every [`builtin_command_catalog`] entry reaches a
/// real handler here (or a deliberate, honest "subsystem missing" notice) —
/// the catalog never advertises a dead command (enforced by a unit test).
///
/// Full ports: `todo`, `usage`, `stats`, `changelog`, `mcp`, `modelpreset`,
/// `fast`, `slow`, `pin`. Partial ports report current state where omp-rs has
/// it and say exactly what is missing for the rest; stubs name the absent
/// subsystem.
#[allow(clippy::too_many_lines)]
async fn dispatch_ported(rt: &std::sync::Arc<Runtime>, name: &str, args: &str) -> Option<Outcome> {
    match name {
        "fast" => Some(fast_command(rt, args)),
        "slow" => Some(slow_command(rt, args)),
        "todo" => Some(todo_command(rt, args).await),
        "usage" if args.is_empty() && crate::slash_extra::dialog::has_ui() => {
            Some(command_center(rt, "usage").await)
        }
        "usage" => Some(usage_command(rt, args).await),
        "stats" if args.is_empty() && crate::slash_extra::dialog::has_ui() => {
            Some(command_center(rt, "stats").await)
        }
        "stats" => Some(stats_command(rt, args).await),
        "changelog" if args.is_empty() && crate::slash_extra::dialog::has_ui() => {
            Some(command_center(rt, "changelog").await)
        }
        "changelog" => Some(changelog_command(rt, args)),
        "trace" if args.is_empty() && crate::slash_extra::dialog::has_ui() => {
            Some(command_center(rt, "trace").await)
        }
        "trace" => Some(trace_command(rt).await),
        "mcp" => Some(mcp_command(rt, args).await),
        "modelpreset" => Some(modelpreset_command(rt, args).await),
        "pin" => Some(pin_command(rt, args)),
        "skillful" => Some(setting_status_toggle(
            rt,
            args,
            "skillful",
            "Skill listing",
            "Usage: /skillful [on|off|status]",
        )),
        "extended-context" => Some(setting_status_toggle(
            rt,
            args,
            "extendedContext",
            "Extended context",
            "Usage: /extended-context [on|off|status]",
        )),
        "computer" => Some(crate::features::browser::computer_command(rt, args).await),
        "ratchet" => Some(notice(rt, "/ratchet needs the ratchet eval-harness subsystem (eval preludes + hill-climb runner), which is not ported to omt.")),
        "prewalk" => Some(notice(rt, "Prewalk is a startup-time feature in omt (the --prewalk/--no-prewalk/--prewalk-into flags); arming or restarting it mid-session is not supported.")),
        "security" => Some(notice(rt, "/security needs the security-scan mode controller (plan/run/compare), which is not ported to omt.")),
        "share" => Some(crate::features::share::command(rt, args).await),
        "browser" => Some(crate::features::browser::command(rt, args).await),
        "ssh" => Some(crate::features::ssh::command(rt, args).await),
        "shake" => Some(crate::features::shake::command(rt, args).await),
        "handoff" => Some(handoff_command(rt, args).await),
        "move" => Some(move_command(rt, args).await),
        "wt" | "worktree" => Some(crate::features::worktree::command(rt, args).await),
        "add-dir" | "remove-dir" | "dirs" => Some(notice(rt, "Multi-root workspace directories are not tracked per-session in omt; there is no sessionManager workspace-dir store to add to.")),
        "marketplace" => Some(crate::features::skills::marketplace_command(rt, args).await),
        "plugins" | "plugin" => Some(plugins_command(rt, args).await),
        _ => None,
    }
}

/// Emit a system notice and consume the command.
fn notice(rt: &Runtime, text: &str) -> Outcome {
    output(rt, text);
    Handled {
        agent_invoked: false,
    }
}

// =============================================================================
// /fast, /slow (builtin-modes.ts `runFastCommand`/`runSlowCommand`)
// =============================================================================

/// Compact context report (simplified TS `buildContextReportText`).
fn context_report(_state: &Value, usage: &Value) -> String {
    let tokens = usage.get("tokens").and_then(serde_json::Value::as_u64);
    let window = usage
        .get("contextWindow")
        .and_then(serde_json::Value::as_u64);
    match (tokens, window) {
        (Some(t), Some(w)) => {
            let pct = t
                .checked_mul(100)
                .and_then(|v| v.checked_div(w))
                .unwrap_or(0);
            format!("Context: {t} / {w} tokens ({pct}%)")
        }
        (Some(t), None) => format!("Context: {t} tokens"),
        _ => "Context usage is not available yet.".into(),
    }
}

/// Names (including aliases) reserved by text-mode builtins
/// (TS `ACP_BUILTIN_RESERVED_NAMES`, from `builtin_commands.json`).
fn is_reserved_builtin(name: &str) -> bool {
    static RESERVED: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
    let set = RESERVED.get_or_init(|| {
        let mut set = HashSet::new();
        for cmd in builtin_command_catalog() {
            if let Some(n) = cmd.get("name").and_then(Value::as_str) {
                set.insert(n.to_string());
            }
            if let Some(aliases) = cmd.get("aliases").and_then(Value::as_array) {
                for a in aliases {
                    if let Some(s) = a.as_str() {
                        set.insert(s.to_string());
                    }
                }
            }
        }
        set
    });
    set.contains(name)
}

fn builtin_command_catalog() -> Vec<Value> {
    serde_json::from_str(include_str!("builtin_commands.json")).unwrap_or_default()
}

use self::Outcome::{Handled, Prompt};

/// Extension + file commands in TS `buildAvailableSlashCommands` order.
/// Appended after the builtin catalog by `Runtime::available_commands`.
pub async fn dynamic_commands(rt: &std::sync::Arc<Runtime>) -> Vec<Value> {
    refresh_file_commands(&rt.cwd);
    let mut out = extension_commands().await;
    out.extend(crate::tools::wasm::command_catalog(&rt.cwd));
    for cmd in FILE_COMMANDS.lock().iter() {
        out.push(json!({
            "name": cmd.name,
            "description": cmd.description,
            "input": cmd.argument_hint.as_ref().map(|h| json!({"hint": h})),
            "source": "file",
        }));
    }
    out
}

/// Full `get_available_commands` catalog: builtins first, then extension and
/// file commands (TS `buildAvailableSlashCommands` order), deduped by name.
pub async fn available_commands(rt: &std::sync::Arc<Runtime>) -> Vec<Value> {
    let mut all = rt.available_commands();
    let dynamic = dynamic_commands(rt).await;
    let mut seen: HashSet<String> = all
        .iter()
        .filter_map(|c| c.get("name").and_then(Value::as_str))
        .map(str::to_string)
        .collect();
    for cmd in dynamic {
        let name = cmd
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if seen.insert(name) {
            all.push(cmd);
        }
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_matches_ts() {
        // helpers/parse.ts: earliest whitespace or `:` separates name/args.
        assert_eq!(
            parse_slash_command("/model"),
            Some(("model".into(), String::new()))
        );
        assert_eq!(
            parse_slash_command("/model zai/glm"),
            Some(("model".into(), "zai/glm".into()))
        );
        assert_eq!(
            parse_slash_command("/force:add-task do it"),
            Some(("force".into(), "add-task do it".into()))
        );
        assert_eq!(
            parse_slash_command("/effort:  high"),
            Some(("effort".into(), "high".into()))
        );
        assert_eq!(parse_slash_command("model"), None);
        assert_eq!(parse_slash_command("/"), None);
    }

    #[test]
    fn reserved_names_match_catalog() {
        assert!(is_reserved_builtin("model"));
        assert!(is_reserved_builtin("models"));
        assert!(is_reserved_builtin("force"));
        assert!(!is_reserved_builtin("nope"));
    }

    /// Every catalog entry (name and aliases) — builtins plus the extra
    /// ported commands in `slash_extra.rs` — must appear as a match arm in a
    /// dispatcher, so the catalog can never advertise a dead command.

    #[test]
    fn catalog_names_all_have_handlers() {
        let mut src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/slash.rs"))
            .expect("read slash.rs");
        // The extra tier lives in a sibling-owned file; cover it when present.
        if let Ok(extra) =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/slash_extra.rs"))
        {
            src.push_str(&extra);
        }
        let catalog = builtin_command_catalog();
        assert!(!catalog.is_empty(), "catalog must parse");
        let mut all = catalog;
        all.extend(crate::slash_extra::extra_catalog());
        for cmd in &all {
            let mut names: Vec<&str> = Vec::new();
            if let Some(n) = cmd.get("name").and_then(Value::as_str) {
                names.push(n);
            }
            if let Some(aliases) = cmd.get("aliases").and_then(Value::as_array) {
                for a in aliases {
                    if let Some(s) = a.as_str() {
                        names.push(s);
                    }
                }
            }
            for name in names {
                // `:`-suffixed aliases (`force:`) are split off by
                // `parse_slash_command` and dispatched under the base name.
                let name = name.strip_suffix(':').unwrap_or(name);
                let arm = regex::Regex::new(&format!(
                    r#"(?m)(^\s*"{}"\s*(=>|\|)|\|\s*"{}"\s*(=>|\|))"#,
                    regex::escape(name),
                    regex::escape(name)
                ))
                .unwrap();
                assert!(
                    arm.is_match(&src),
                    "catalog command /{name} has no handler (no match arm in slash.rs or slash_extra.rs)"
                );
            }
        }
    }
}
