//! Extra slash commands ported from TS (`slash-commands/builtin-*.ts`).
//!
//! Provides [`dispatch_extra`] and [`extra_catalog`] for commands that extend
//! the core text-mode builtin set. Commands with runtime backing in `omp-rs`
//! execute their real handlers (`/new`, `/fork`, `/queue`); interactive selectors and
//! dashboards (`/branch`, `/copy`, `/agents`, `/extensions`, `/debug`, `/jobs`) run as runtime
//! dialogs (`interactive`, on `dialog`); commands whose subsystems are not ported emit an
//! informative notice explaining what is missing and return [`Outcome::Handled`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod dialog;
pub(crate) mod interactive;
mod login;
pub use dialog::mark_interactive_ui;

use serde_json::{json, Value};

use omp_collab::controller::{CollabController, CollabStartOptions, SettingsGetter};
use omp_collab::host::CollabHostBackend;
use omp_collab::registry::{self, CollabAccess};

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
/// The conversation as plain text for `/rlm <question>` without a context
/// source: one `[role]` header per message, text and tool output bodies.
fn session_transcript(messages: &[Value]) -> String {
    let mut out = String::new();
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("?");
        let body = match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                    Some("text") => p.get("text").and_then(Value::as_str).map(str::to_owned),
                    Some("toolCall") => Some(format!(
                        "<call {}> {}",
                        p.get("name").and_then(Value::as_str).unwrap_or(""),
                        p.get("arguments").map(Value::to_string).unwrap_or_default()
                    )),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if body.trim().is_empty() {
            continue;
        }
        out.push_str(&format!("[{role}]\n{body}\n\n"));
    }
    out
}

async fn rlm_command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    use crate::tools::rlm;
    let trimmed = args.trim();
    if trimmed.is_empty() || trimmed == "help" {
        output(rt, rlm::USAGE);
        return Outcome::Handled {
            agent_invoked: false,
        };
    }
    if trimmed == "stats" {
        let stats = rlm::stats_snapshot();
        output(
            rt,
            &serde_json::to_string_pretty(&stats).unwrap_or_default(),
        );
        return Outcome::Handled {
            agent_invoked: false,
        };
    }
    let parsed = match rlm::parse_args(trimmed) {
        Ok(parsed) => parsed,
        Err(message) => {
            output(rt, &message);
            return Outcome::Handled {
                agent_invoked: false,
            };
        }
    };
    let context = match parsed.context {
        rlm::ContextSource::Text(text) => text,
        rlm::ContextSource::Session => {
            let text = session_transcript(&rt.messages().await);
            if text.trim().is_empty() {
                output(rt, &format!("/rlm: this session has no conversation yet; pass a context with @<file> or -- <text>.\n{}", rlm::USAGE));
                return Outcome::Handled {
                    agent_invoked: false,
                };
            }
            output(
                rt,
                &format!(
                    "/rlm: context = this session's conversation ({} chars)",
                    text.chars().count()
                ),
            );
            text
        }
        rlm::ContextSource::File(path) => {
            let full = if Path::new(&path).is_absolute() {
                path.clone().into()
            } else {
                rt.cwd.join(&path)
            };
            match std::fs::read_to_string(&full) {
                Ok(text) => text,
                Err(e) => {
                    output(rt, &format!("/rlm: cannot read {}: {e}", full.display()));
                    return Outcome::Handled {
                        agent_invoked: false,
                    };
                }
            }
        }
    };
    rlm::SESSION.reset();
    rlm::set_override(parsed.max_depth, parsed.max_calls, parsed.max_tokens);
    let context_chars = context.chars().count();
    rlm::SESSION.set_context(Some(context));
    if let Err(e) = rlm::load_context_cell(&rt.cwd).await {
        output(
            rt,
            &format!("/rlm: failed to load context into the eval kernel: {e}"),
        );
        return Outcome::Handled {
            agent_invoked: false,
        };
    }
    Outcome::Prompt(rlm::loop_prompt(
        &parsed.question,
        context_chars,
        rlm::Limits::effective(),
    ))
}

/// `/ultra` usage text.
pub const ULTRA_USAGE: &str = concat!(
    "Usage: /ultra [code|deep|long|hyper|off]\n",
    "  code   ultracode: max thinking + workflow (eval workpool) by default on every substantial task\n",
    "  deep   ultradeep: ultracode + RLM (llm_query, λ-RLM, SERA rubrics) for huge inputs\n",
    "  long   ultralong: ultracode + CLM (clm-host context strategy) for long sessions\n",
    "  hyper  hypercode: ultracode + RLM + CLM\n",
    "  off    leave ultra mode (thinking level stays)",
);

/// What an `/ultra` argument asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UltraArg {
    /// No argument: show the mode and usage.
    Show,
    /// Enter (`Some`) or leave (`None`) an ultra mode.
    Set(Option<omp_tui::magic::Ultra>),
}

/// Parse an `/ultra` argument (`code|deep|long|hyper|off`, full mode names too).
///
/// # Errors
/// Usage error for anything else.
pub fn parse_ultra_arg(args: &str) -> Result<UltraArg, String> {
    match args.trim() {
        "" => Ok(UltraArg::Show),
        "off" => Ok(UltraArg::Set(None)),
        other => omp_tui::magic::Ultra::parse(other)
            .map(|u| UltraArg::Set(Some(u)))
            .ok_or_else(|| format!("/ultra: unknown mode {other:?}\n{ULTRA_USAGE}")),
    }
}

async fn ultra_command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    match parse_ultra_arg(args) {
        Ok(UltraArg::Show) => {
            let current = rt.ultra().map_or("off", omp_tui::magic::Ultra::name);
            output(rt, &format!("Ultra mode: {current}\n{ULTRA_USAGE}"));
        }
        Ok(UltraArg::Set(None)) if rt.ultra().is_none() => output(rt, "Ultra mode is already off."),
        Ok(UltraArg::Set(mode)) => {
            rt.set_ultra(mode).await;
            match mode {
                Some(m) => output(rt, &format!("Ultra mode: {}", m.name())),
                None => output(rt, "Ultra mode off."),
            }
        }
        Err(message) => output(rt, &message),
    }
    Outcome::Handled {
        agent_invoked: false,
    }
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
            | "ultra"
            | "collab"
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
        json!({
            "name": "ultra",
            "description": "Ultra mode: max thinking + workflow by default (code), with RLM (deep), CLM (long) or both (hyper)",
            "input": {
                "hint": "[code|deep|long|hyper|off]"
            },
            "source": "builtin",
        }),
        json!({
            "name": "collab",
            "description": "Share this session live via a relay",
            "input": {
                "hint": "[start|view|list|stop|status] [relayUrl]"
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
        "ultra" => Some(ultra_command(rt, args).await),
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
        "copy" => Some(interactive::copy_command(rt, args).await),
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
        "branch" => Some(interactive::branch_command(rt).await),
        "rewind" => Some(interactive::branch_command(rt).await),
        "plan" => {
            output(
                rt,
                "/plan: plan mode is not ported to omt; prompt the model directly with planning instructions or use the TS CLI",
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
                "/loop: loop mode is not ported to omt; use the TS CLI to run prompt loops",
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
                "/skills: skills registry (skills.omp.sh) is not ported to omt; use `omp skills` from the TS CLI",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "agents" => Some(interactive::agents_command(rt).await),
        "extensions" => Some(interactive::extensions_command(rt).await),
        "status" => Some(interactive::extensions_command(rt).await),
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
        "debug" => Some(interactive::debug_command(rt).await),
        "restart" => {
            output(
                rt,
                "/restart: process restart is not ported to omt; exit and re-run with --resume",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "btw" => {
            output(
                rt,
                "/btw: side-question conversation (btw) is not ported to omt; prompt the model directly in the main session",
            );
            Some(Outcome::Handled {
                agent_invoked: false,
            })
        }
        "collab" => Some(collab_command(rt, args).await),
        _ => None,
    }
}

/// Process-wide collab controller — one per interactive process, mirroring TS
/// `ctx.collabController` (built lazily on the first `/collab`).
static COLLAB_CONTROLLER: std::sync::OnceLock<Arc<CollabController>> = std::sync::OnceLock::new();

/// Settings getter wired the way the runtime reads config: reload per call
/// (runtime.rs does `Settings::load(&self.cwd)` at every read site).
fn collab_settings_getter(cwd: PathBuf) -> SettingsGetter {
    Arc::new(move |id: &str| omp_settings::Settings::load(&cwd).get_string(id))
}

/// [`Runtime`] as a collab host backend: registry snapshot identity plus
/// guest-prompt injection into the live session.
struct RuntimeCollabBackend {
    rt: Arc<Runtime>,
    /// Handle of the CLI's runtime, captured at construction: guest prompts
    /// must run there even though the host's relay event loop is driven on its
    /// own thread (controller spawns `CollabHost::start` off-runtime).
    prompt_runtime: tokio::runtime::Handle,
}

impl CollabHostBackend for RuntimeCollabBackend {
    fn session_id(&self) -> String {
        self.rt.session_id()
    }

    fn cwd(&self) -> String {
        self.rt.cwd.display().to_string()
    }

    fn session_name(&self) -> Option<String> {
        // settings.ts `collab.displayName` overrides the registry session name.
        let name =
            collab_settings_getter(self.rt.cwd.clone())("collab.displayName").unwrap_or_default();
        (!name.trim().is_empty()).then_some(name)
    }

    fn inject_guest_prompt(&self, text: &str, images: Vec<Value>) -> anyhow::Result<()> {
        let rt = Arc::clone(&self.rt);
        let text = text.to_string();
        let images: Vec<pa_agent::types::ImageContent> = images
            .into_iter()
            .filter_map(|v| serde_json::from_value(v).ok())
            .collect();
        // `Runtime::prompt` self-queues while streaming; `followUp` keeps a
        // guest message from stealing the live turn (TS queued-message drain).
        self.prompt_runtime.spawn(async move {
            if let Err(err) = rt.prompt(None, &text, Some("followUp"), images).await {
                rt.send(json!({
                    "type": "command_output",
                    "text": format!("collab: guest prompt failed: {}", err.message),
                }));
            }
        });
        Ok(())
    }
}

/// The process collab controller, built on first use against this Runtime.
fn collab_controller(rt: &Arc<Runtime>) -> Arc<CollabController> {
    COLLAB_CONTROLLER
        .get_or_init(|| {
            Arc::new(CollabController::new(
                Arc::new(RuntimeCollabBackend {
                    rt: Arc::clone(rt),
                    prompt_runtime: tokio::runtime::Handle::current(),
                }),
                collab_settings_getter(rt.cwd.clone()),
            ))
        })
        .clone()
}

/// `/collab [start|view|list|stop|status] [relayUrl]` (TS
/// `slash-commands/builtin-collaboration.ts` "collab", text-mode port).
async fn collab_command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let controller = collab_controller(rt);
    let args = args.trim();
    let (verb, rest) = match args.split_once(char::is_whitespace) {
        Some((v, r)) => (v, r.trim()),
        None => (args, ""),
    };
    match verb {
        "stop" => {
            controller.stop_host("host stopped").await;
            output(rt, "Collab stopped");
        }
        "status" => match controller.current_host() {
            Some(host) => {
                let guests = host.participant_count().saturating_sub(1);
                let link = if host.access() == CollabAccess::View {
                    controller.web_view_link()
                } else {
                    controller.web_link()
                };
                output(
                    rt,
                    &format!(
                        "Collab: {guests} guest{} — {}",
                        if guests == 1 { "" } else { "s" },
                        link.as_deref().unwrap_or("(link unavailable)"),
                    ),
                );
            }
            None => output(rt, "Not in a collab session"),
        },
        "list" => {
            // Same registry as `omp collab list`: metadata only, never a link
            // (TS: a link is a deliberate per-host act).
            if !rest.is_empty() {
                output(
                    rt,
                    "Usage: /collab list — for links use `omt collab link <id|pid>`",
                );
                return Outcome::Handled {
                    agent_invoked: false,
                };
            }
            match registry::list_collab_hosts(None).await {
                Ok(hosts) if hosts.is_empty() => output(rt, "No active Collab hosts"),
                Ok(hosts) => {
                    let mut lines = vec![format!(
                        "{} active local Collab host{}",
                        hosts.len(),
                        if hosts.len() == 1 { "" } else { "s" }
                    )];
                    for h in &hosts {
                        let guests = h.participants.saturating_sub(1);
                        lines.push(format!(
                            "  {} — pid {}, gen {}, {} guest{}, {}, {}",
                            h.instance_id,
                            h.pid,
                            h.generation,
                            guests,
                            if guests == 1 { "" } else { "s" },
                            h.access,
                            if h.relay_connected {
                                "relay connected"
                            } else {
                                "relay reconnecting"
                            },
                        ));
                    }
                    lines.push(
                        "Get a link: omt collab link <instanceId|pid> [--view]".to_string(),
                    );
                    output(rt, &lines.join("\n"));
                }
                Err(err) => output(rt, &format!("Failed to list collab hosts: {err:#}")),
            }
        }
        _ => {
            // `/collab`, `/collab start [relayUrl]`, `/collab view [relayUrl]`;
            // a first token that is not a verb is a relay override (TS
            // `knownStartVerb ? rest : args`).
            let known_verb = matches!(verb, "start" | "view");
            let view = verb == "view";
            let relay_arg = if known_verb { rest } else { args };
            let access = if view {
                CollabAccess::View
            } else {
                CollabAccess::Control
            };
            let existing = controller.current_host();
            match controller
                .start_host(CollabStartOptions {
                    access,
                    relay: (!relay_arg.is_empty()).then(|| relay_arg.to_string()),
                })
                .await
            {
                Ok(host) => {
                    let heading = if existing.as_ref().is_some_and(|e| {
                        e.instance_id() == host.instance_id() && e.generation() == host.generation()
                    }) {
                        if view {
                            "Read-only collab session active"
                        } else {
                            "Collab session active"
                        }
                    } else {
                        "Collab session started!"
                    };
                    let mut lines = vec![heading.to_string()];
                    if let Some(full) = controller.full_link() {
                        lines.push(format!("  Control link: {full}"));
                    }
                    if let Some(view_link) = controller.view_link() {
                        lines.push(format!("  View-only link: {view_link}"));
                    }
                    if let Some(web) = controller.web_link() {
                        lines.push(format!("  Browser: {web}"));
                    }
                    lines.push(
                        "Guests open the browser link; share the control link only with trusted guests."
                            .to_string(),
                    );
                    output(rt, &lines.join("\n"));
                }
                Err(err) => output(rt, &format!("Failed to start collab session: {err:#}")),
            }
        }
    }
    Outcome::Handled {
        agent_invoked: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extra_catalog_names_all_handled() {
        let catalog = extra_catalog();
        assert_eq!(catalog.len(), 20, "expected 20 extra commands in catalog");
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
    fn ultra_args_parse() {
        use omp_tui::magic::Ultra;
        assert_eq!(parse_ultra_arg(""), Ok(UltraArg::Show));
        assert_eq!(parse_ultra_arg("  "), Ok(UltraArg::Show));
        assert_eq!(parse_ultra_arg("off"), Ok(UltraArg::Set(None)));
        assert_eq!(
            parse_ultra_arg("code"),
            Ok(UltraArg::Set(Some(Ultra::Code)))
        );
        assert_eq!(
            parse_ultra_arg("deep"),
            Ok(UltraArg::Set(Some(Ultra::Deep)))
        );
        assert_eq!(
            parse_ultra_arg(" long "),
            Ok(UltraArg::Set(Some(Ultra::Long)))
        );
        assert_eq!(
            parse_ultra_arg("hyper"),
            Ok(UltraArg::Set(Some(Ultra::Hyper)))
        );
        assert_eq!(
            parse_ultra_arg("hypercode"),
            Ok(UltraArg::Set(Some(Ultra::Hyper)))
        );
        let err = parse_ultra_arg("max").unwrap_err();
        assert!(
            err.contains("unknown mode") && err.contains("Usage: /ultra"),
            "{err}"
        );
        assert!(extra_catalog()
            .iter()
            .any(|c| c["name"] == "ultra" && c["input"]["hint"] == "[code|deep|long|hyper|off]"));
    }

    #[test]
    fn test_unhandled_command_returns_false() {
        assert!(!is_extra_command("nonexistent_command"));
        assert!(!is_extra_command("help"));
    }
}

#[cfg(test)]
mod rlm_session_context_tests {
    use super::*;

    #[test]
    fn session_transcript_keeps_roles_text_and_calls() {
        let t = session_transcript(&[
            json!({"role": "user", "content": [{"type": "text", "text": "the code is ZEBRA-42"}]}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "toolCall", "name": "bash", "arguments": {"command": "ls"}},
            ]}),
            json!({"role": "toolResult", "content": [{"type": "text", "text": "a.txt"}]}),
            json!({"role": "assistant", "content": []}),
        ]);
        assert!(t.contains("[user]\nthe code is ZEBRA-42"), "{t}");
        assert!(t.contains("<call bash>"), "{t}");
        assert!(t.contains("[toolResult]\na.txt"), "{t}");
        assert!(!t.contains("hidden"), "{t}");
    }
}
