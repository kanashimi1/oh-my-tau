//! Extra slash commands ported from TS (`slash-commands/builtin-*.ts`).
//!
//! Provides [`dispatch_extra`] and [`extra_catalog`] for commands that extend
//! the core text-mode builtin set. Commands with runtime backing in `omp-rs`
//! execute their real handlers (`/new`, `/fork`, `/queue`); commands whose
//! subsystems are not ported headlessly (clipboards, TUI
//! dashboards) emit an informative notice explaining what is missing and return
//! [`Outcome::Handled`].

use std::path::Path;
use std::sync::Arc;

mod login;
pub use login::mark_interactive_ui;

use serde_json::{json, Value};

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Emit a `command_output` frame, mirroring `slash::output`.
fn output(rt: &Runtime, text: &str) {
    rt.send(json!({
        "type": "command_output",
        "text": text,
    }));
}

/// `/rlm <question> @file | -- text` (opt-in RLM phase 1, design §3.3): loads
/// the context into the Python eval kernel as `context`, resets the sub-call
/// ledger and hands the RLM loop prompt to the model.
async fn rlm_command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    use crate::tools::rlm;
    let trimmed = args.trim();
    if trimmed.is_empty() || trimmed == "help" {
        output(rt, rlm::USAGE);
        return Outcome::Handled { agent_invoked: false };
    }
    if trimmed == "stats" {
        let stats = rlm::stats_snapshot();
        output(rt, &serde_json::to_string_pretty(&stats).unwrap_or_default());
        return Outcome::Handled { agent_invoked: false };
    }
    let parsed = match rlm::parse_args(trimmed) {
        Ok(parsed) => parsed,
        Err(message) => {
            output(rt, &message);
            return Outcome::Handled { agent_invoked: false };
        }
    };
    let context = match parsed.context {
        rlm::ContextSource::Text(text) => text,
        rlm::ContextSource::File(path) => {
            let full = if Path::new(&path).is_absolute() { path.clone().into() } else { rt.cwd.join(&path) };
            match std::fs::read_to_string(&full) {
                Ok(text) => text,
                Err(e) => {
                    output(rt, &format!("/rlm: cannot read {}: {e}", full.display()));
                    return Outcome::Handled { agent_invoked: false };
                }
            }
        }
    };
    rlm::SESSION.reset();
    rlm::set_override(parsed.max_depth, parsed.max_calls, parsed.max_tokens);
    let context_chars = context.chars().count();
    rlm::SESSION.set_context(Some(context));
    if let Err(e) = rlm::load_context_cell(&rt.cwd).await {
        output(rt, &format!("/rlm: failed to load context into the eval kernel: {e}"));
        return Outcome::Handled { agent_invoked: false };
    }
    Outcome::Prompt(rlm::loop_prompt(&parsed.question, context_chars, rlm::Limits::effective()))
}

/// Check whether `name` is one of the extra builtin commands or their aliases
/// (only the catalog⊆handlers test needs it; dispatch matches directly).
#[cfg(test)]
#[must_use]
pub fn is_extra_command(name: &str) -> bool {
    matches!(
        name,
        "new"
            | "quit"
            | "q"
            | "copy"
            | "fork"
            | "branch"
            | "rewind"
            | "plan"
            | "goal"
            | "loop"
            | "queue"
            | "skills"
            | "agents"
            | "extensions"
            | "status"
            | "login"
            | "logout"
            | "debug"
            | "restart"
            | "btw"
            | "rlm"
    )
}

/// Catalog of extra slash commands ported from TS.
///
/// Shape: `{"name", "description", "aliases"?, "input": {"hint"}?, "source": "builtin"}`.
#[must_use]
pub fn extra_catalog() -> Vec<Value> {
    vec![
        json!({
            "name": "rlm",
            "description": "RLM (opt-in): load @file/text as `context` in the eval kernel; model inspects it with code and delegates chunks to llm_query (smol)",
            "input": {
                "hint": "[--max-calls=N] <question> @file | -- text"
            },
            "source": "builtin",
        }),
        json!({
            "name": "new",
            "description": "Start a new session",
            "source": "builtin",
        }),
        json!({
            "name": "quit",
            "aliases": ["q"],
            "description": "Quit the application",
            "source": "builtin",
        }),
        json!({
            "name": "copy",
            "description": "Pick text or code from the conversation to copy",
            "input": {
                "hint": "[code|cmd|link]"
            },
            "source": "builtin",
        }),
        json!({
            "name": "fork",
            "description": "Create a new fork from a previous message",
            "source": "builtin",
        }),
        json!({
            "name": "branch",
            "aliases": ["rewind"],
            "description": "Rewind to a previous message, keeping the old path as a branch",
            "source": "builtin",
        }),
        json!({
            "name": "plan",
            "description": "Toggle plan mode (agent plans before executing)",
            "input": {
                "hint": "[prompt]"
            },
            "source": "builtin",
        }),
        json!({
            "name": "goal",
            "description": "Toggle goal mode (persistent autonomous objective for this session)",
            "input": {
                "hint": "[objective]"
            },
            "source": "builtin",
        }),
        json!({
            "name": "loop",
            "description": "Toggle loop mode (repeats prompt after each yield)",
            "input": {
                "hint": "[count|duration] [--while|--until '<cmd>'] [prompt]"
            },
            "source": "builtin",
        }),
        json!({
            "name": "queue",
            "description": "Queue a message for after the agent yields",
            "input": {
                "hint": "<message>"
            },
            "source": "builtin",
        }),
        json!({
            "name": "skills",
            "description": "Search, install, and update skills from the skills.omp.sh registry",
            "input": {
                "hint": "<search|install|installed|update> [args]"
            },
            "source": "builtin",
        }),
        json!({
            "name": "agents",
            "description": "Open the agents hub (per-agent model, prewalk, and advisor)",
            "source": "builtin",
        }),
        json!({
            "name": "extensions",
            "aliases": ["status"],
            "description": "Open Extension Control Center dashboard",
            "source": "builtin",
        }),
        json!({
            "name": "login",
            "description": "Login with OAuth provider",
            "input": {
                "hint": "[provider|redirect URL]"
            },
            "source": "builtin",
        }),
        json!({
            "name": "logout",
            "description": "Logout from OAuth provider",
            "input": {
                "hint": "[provider]"
            },
            "source": "builtin",
        }),
        json!({
            "name": "debug",
            "description": "Open debug tools selector",
            "source": "builtin",
        }),
        json!({
            "name": "restart",
            "description": "Restart omp with the same launch flags, resuming this session",
            "source": "builtin",
        }),
        json!({
            "name": "btw",
            "description": "Ask a side question, or browse this session's BTW history",
            "input": {
                "hint": "[question]"
            },
            "source": "builtin",
        }),
    ]
}

/// Text-mode dispatch for the extra builtin slash commands.
///
/// Returns `None` when `name` is not recognized, allowing fallback to
/// extension and file commands.
pub async fn dispatch_extra(rt: &Arc<Runtime>, name: &str, args: &str) -> Option<Outcome> {
    match name {
        "rlm" => Some(rlm_command(rt, args).await),
        "new" => {
            if rt.state().await.get("isStreaming").and_then(Value::as_bool) == Some(true) {
                output(
                    rt,
                    "Wait for the current response to finish or abort it before starting a new session.",
                );
                return Some(Outcome::Handled {
                    agent_invoked: false,
                });
            }
            let dir = if args.trim().is_empty() {
                None
            } else {
                Some(Path::new(args.trim()))
            };
            match rt.new_session(dir).await {
                Ok(()) => output(rt, "New session started."),
                Err(e) if crate::runtime::is_switch_cancelled(&e) => {}
                Err(e) => output(rt, &format!("Failed to start a new session: {e:#}")),
            }
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "quit" => {
            output(
                rt,
                "/quit: to exit the interactive session use /exit, Ctrl+D, or double-tap Ctrl+C; RPC sessions terminate when the client disconnects.",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "q" => {
            output(
                rt,
                "/quit: to exit the interactive session use /exit, Ctrl+D, or double-tap Ctrl+C; RPC sessions terminate when the client disconnects.",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "copy" => {
            output(
                rt,
                "/copy: clipboard integration is not ported to omp-rs; copy directly from the terminal or export session with /export",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "fork" => {
            if rt.state().await.get("isStreaming").and_then(Value::as_bool) == Some(true) {
                output(
                    rt,
                    "Wait for the current response to finish or abort it before forking.",
                );
                return Some(Outcome::Handled {
                    agent_invoked: false,
                });
            }
            let path = rt.session_file();
            if !path.exists() {
                output(rt, "No active session file to fork.");
                return Some(Outcome::Handled {
                    agent_invoked: false,
                });
            }
            match rt.fork_session(&path, None).await {
                Ok(()) => {
                    rt.publish_session_file();
                    output(
                        rt,
                        &format!("Forked session to new session file: {}", rt.session_id()),
                    );
                }
                Err(e) => output(rt, &format!("Fork failed: {e:#}")),
            }
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "branch" => {
            output(
                rt,
                "/branch: interactive message rewinding is not ported to omp-rs; use checkpoint/rewind tools in agent turns or /fork to branch from the current tip",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "rewind" => {
            output(
                rt,
                "/branch: interactive message rewinding is not ported to omp-rs; use checkpoint/rewind tools in agent turns or /fork to branch from the current tip",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "plan" => {
            output(
                rt,
                "/plan: plan mode is not ported to omp-rs; prompt the model directly with planning instructions or use the TS CLI",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "goal" => {
            output(
                rt,
                "/goal: goal mode is not enabled in this session; configure goal.enabled or run interactively in the TS CLI",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "loop" => {
            output(
                rt,
                "/loop: loop mode is not ported to omp-rs; use the TS CLI to run prompt loops",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "queue" => {
            let text = args.trim();
            if text.is_empty() {
                output(rt, "Usage: /queue <message>");
            } else {
                rt.follow_up(text);
                output(rt, &format!("Queued message for after yield: {text}"));
            }
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "skills" => {
            output(
                rt,
                "/skills: skills registry (skills.omp.sh) is not ported to omp-rs; use `omp skills` from the TS CLI",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "agents" => {
            output(
                rt,
                "/agents: agents dashboard is only available in the interactive TUI",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "extensions" => {
            output(
                rt,
                "/extensions: extension dashboard is only available in the interactive TUI",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "status" => {
            output(
                rt,
                "/extensions: extension dashboard is only available in the interactive TUI",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "login" => {
            login::login_command(rt, args).await;
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "logout" => {
            login::logout_command(rt, args).await;
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "debug" => {
            output(
                rt,
                "/debug: debug tools selector is only available in the interactive TUI",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "restart" => {
            output(
                rt,
                "/restart: process restart is not ported to omp-rs; exit and re-run with --resume",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "btw" => {
            output(
                rt,
                "/btw: side-question conversation (btw) is not ported to omp-rs; prompt the model directly in the main session",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extra_catalog_names_all_handled() {
        let catalog = extra_catalog();
        assert_eq!(catalog.len(), 18, "expected 18 extra commands in catalog");
        for cmd in &catalog {
            let name = cmd["name"].as_str().expect("command must have name");
            assert!(
                is_extra_command(name),
                "Command /{name} from extra_catalog is not recognized by is_extra_command"
            );
            if let Some(aliases) = cmd.get("aliases").and_then(Value::as_array) {
                for a in aliases {
                    let alias = a.as_str().expect("alias must be a string");
                    assert!(
                        is_extra_command(alias),
                        "Alias /{alias} for /{name} from extra_catalog is not recognized by is_extra_command"
                    );
                }
            }
        }
    }

    #[test]
    fn test_catalog_structure() {
        for cmd in extra_catalog() {
            assert!(cmd.get("name").is_some());
            assert!(cmd.get("description").is_some());
            assert_eq!(cmd.get("source").and_then(Value::as_str), Some("builtin"));
        }
    }

    #[test]
    fn test_unhandled_command_returns_false() {
        assert!(!is_extra_command("nonexistent_command"));
        assert!(!is_extra_command("help"));
    }
}
