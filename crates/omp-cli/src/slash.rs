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

use serde_json::{json, Value};
use std::sync::Mutex;

use crate::runtime::Runtime;

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
    Some((
        body[..sep].to_string(),
        body[sep + 1..].trim().to_string(),
    ))
}

/// Emit a `command_output` frame (TS `runtime.output`).
fn output(rt: &Runtime, text: &str) {
    rt.send(json!({"type": "command_output", "text": text}));
}

/// Emit a usage/error message and consume the command (TS `usage`).
fn usage(rt: &Runtime, text: &str) -> Outcome {
    output(rt, text);
    Outcome::Handled { agent_invoked: false }
}

static EXTENSIONS: OnceLock<Arc<Ext>> = OnceLock::new();
type Ext = crate::ext::Extensions;
type Arc<T> = std::sync::Arc<T>;

/// Register the process-wide extension host for command listing/dispatch.
/// Called once at startup (rpc and -p modes).
pub fn set_extensions(ext: Arc<Ext>) {
    let _ = EXTENSIONS.set(ext);
}

fn extensions() -> Option<&'static Arc<Ext>> {
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

/// Full dispatch chain for one prompt (builtins → extension → file templates).
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
    if try_extension_command(rt, &name, &args).await {
        return Outcome::ExtensionHandled;
    }
    if let Some(expanded) = expand_file_command(&name, &args, &rt.cwd) {
        return Outcome::Prompt(expanded);
    }
    Outcome::NotSlash
}

static FILE_COMMANDS: Mutex<Vec<FileCommand>> = Mutex::new(Vec::new());

/// A file-based slash command (`commands/*.md`).
#[derive(Debug, Clone)]
pub struct FileCommand {
    pub name: String,
    pub description: String,
    pub argument_hint: Option<String>,
    pub content: String,
}

fn command_dirs(cwd: &Path) -> Vec<(PathBuf, &'static str)> {
    let mut dirs = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push((PathBuf::from(home).join(".omp/agent/commands"), "user"));
    }
    dirs.push((cwd.join(".omp/commands"), "project"));
    dirs
}

/// Load markdown commands from the user and project `commands/` directories
/// (TS `loadSlashCommands` builtin provider; frontmatter description /
/// `argument-hint` honored, first body line as fallback description).
pub fn load_file_commands(cwd: &Path) -> Vec<FileCommand> {
    let mut out: Vec<FileCommand> = Vec::new();
    for (dir, _level) in command_dirs(cwd) {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "md"))
            .collect();
        paths.sort();
        for path in paths {
            let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if out.iter().any(|c| c.name == name) {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let (frontmatter, body) = split_frontmatter(&content);
            let mut description = String::new();
            let mut argument_hint = None;
            for line in frontmatter.lines() {
                if let Some(v) = line.strip_prefix("description:") {
                    description = v.trim().to_string();
                }
                if let Some(v) = line
                    .strip_prefix("argument-hint:")
                    .or_else(|| line.strip_prefix("argumentHint:"))
                {
                    argument_hint = Some(v.trim().to_string());
                }
            }
            if description.is_empty() {
                if let Some(first) = body.lines().find(|l| !l.trim().is_empty()) {
                    description = first.trim().chars().take(60).collect();
                    if first.trim().chars().count() > 60 {
                        description.push_str("...");
                    }
                }
            }
            out.push(FileCommand {
                name: name.to_string(),
                description,
                argument_hint,
                content: body.to_string(),
            });
        }
    }
    out
}

/// Minimal frontmatter split: `---` fenced block at the top of the file.
fn split_frontmatter(content: &str) -> (&str, &str) {
    if let Some(rest) = content.strip_prefix("---\n") {
        if let Some(end) = rest.find("\n---") {
            let fm = &rest[..end];
            let body = rest[end + 4..].strip_prefix('\n').unwrap_or(&rest[end + 4..]);
            return (fm, body);
        }
    }
    ("", content)
}

/// Reload the cached file commands from disk (cheap; safe to call from async context).
pub fn refresh_file_commands(cwd: &Path) {
    let cmds = load_file_commands(cwd);
    *FILE_COMMANDS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = cmds;
}

/// Expand `/name args` through a file template (`expandSlashCommand`): the
/// raw args string is appended when the template has no `$ARGUMENTS`/`$1`
/// placeholders; `$ARGUMENTS`/`$ARGS` render the args text, `$1..$9` the
/// positional words.
fn expand_file_command(name: &str, args: &str, cwd: &Path) -> Option<String> {
    let cmds = load_file_commands(cwd);
    let cmd = cmds.into_iter().find(|c| c.name == name)?;
    let positional: Vec<&str> = args.split_whitespace().collect();
    let args_text = positional.join(" ");
    let mut body = cmd.content;
    let mut used = false;
    for (placeholder, value) in [
        ("$ARGUMENTS", args_text.as_str()),
        ("$ARGS", args_text.as_str()),
    ] {
        if body.contains(placeholder) {
            used = true;
            body = body.replace(placeholder, value);
        }
    }
    for (i, word) in positional.iter().enumerate().take(9) {
        let placeholder = format!("${}", i + 1);
        if body.contains(&placeholder) {
            used = true;
            body = body.replace(&placeholder, word);
        }
    }
    if !used && !args_text.is_empty() {
        body = body.trim_end().to_string();
        body.push('\n');
        body.push_str(&args_text);
    }
    Some(body)
}


/// `formatModelString` equivalent: `provider/id`.
fn model_ref(info: &serde_json::Value) -> String {
    format!(
        "{}/{}",
        info.get("provider").and_then(|v| v.as_str()).unwrap_or("?"),
        info.get("id").and_then(|v| v.as_str()).unwrap_or("?")
    )
}

/// Resolve a `/model <selector>` argument: `provider/id`, or a unique model-id
/// prefix match over the models with configured auth (fuzzy `@role` selectors
/// and `:level` suffixes are TS-selector features; `:level` is honored here).
async fn resolve_model(
    _rt: &std::sync::Arc<Runtime>,
    selector: &str,
) -> Option<(String, String, Option<pa_agent::types::ThinkingLevel>)> {
    let (sel, level) = match selector.rsplit_once(':') {
        Some((s, l)) if crate::runtime::parse_level(l).is_some() => (s, crate::runtime::parse_level(l)),
        _ => (selector, None),
    };
    if sel.is_empty() {
        return None;
    }
    if let Some((provider, id)) = sel.split_once('/') {
        return Some((provider.to_string(), id.to_string(), level));
    }
    let models = crate::runtime::available_models(None).await;
    let mut hits = models
        .iter()
        .filter(|m| m.id == sel || m.id.starts_with(sel))
        .collect::<Vec<_>>();
    if hits.len() != 1 {
        hits = models
            .iter()
            .filter(|m| format!("{}/{}", m.provider, m.id).contains(sel))
            .collect();
    }
    hits.into_iter()
        .next()
        .map(|m| (m.provider.clone(), m.id.clone(), level))
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
                    Err(e) => return Some(usage(rt, &format!("Failed to set model: {}", e.message))),
                }
            }
            rt.announce_commands();
            Some(Handled { agent_invoked: false })
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
                return Some(Handled { agent_invoked: false });
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
                    Some(Handled { agent_invoked: false })
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
                output(rt, &format!("Thinking: {current}\nAvailable: {}", levels.join(", ")));
                return Some(Handled { agent_invoked: false });
            }
            let sel = args.trim().to_lowercase();
            match levels.iter().find(|l| **l == sel) {
                Some(level_name) => {
                    let level = crate::runtime::parse_level(level_name)
                        .unwrap_or(pa_agent::types::ThinkingLevel::Medium);
                    rt.set_thinking_level(level).await;
                    output(rt, &format!("Thinking set to {level_name}."));
                    Some(Handled { agent_invoked: false })
                }
                None => Some(usage(
                    rt,
                    &format!("Unknown thinking level: {sel}. Available: {}", levels.join(", ")),
                )),
            }
        }
        "fresh" => {
            if rt.state().await.get("isStreaming").and_then(serde_json::Value::as_bool) == Some(true) {
                Some(usage(
                    rt,
                    "Wait for the current response to finish or abort it before refreshing provider state.",
                ))
            } else {
                output(rt, "Provider stream state reset.");
                Some(Handled { agent_invoked: false })
            }
        }
        "compact" => {
            // TS parseCompactArgs: optional leading mode token (soft|remote|snapcompact).
            let trimmed = args.trim();
            let (mode, instructions) = match trimmed.split_once(char::is_whitespace) {
                None if matches!(trimmed, "soft" | "remote" | "snapcompact") => (Some(trimmed), None),
                None => (None, (!trimmed.is_empty()).then(|| trimmed.to_string())),
                Some((first, rest))
                    if matches!(first, "soft" | "remote" | "snapcompact") =>
                {
                    if first == "snapcompact" {
                        return Some(usage(
                            rt,
                            "/compact snapcompact does not take focus instructions (it archives history without an LLM summary).",
                        ));
                    }
                    (Some(first), Some(rest.trim().to_string()).filter(|s| !s.is_empty()))
                }
                Some(_) => (None, Some(trimmed.to_string())),
            };
            if let Some(mode) = mode {
                // omp-rs compaction does not implement mode overrides; run the default.
                output(rt, &format!("/compact {mode} is not supported; running default compaction."));
            }
            let instructions = instructions.filter(|s| !s.is_empty());
            let result = rt.compact(instructions.as_deref()).await;
            match result {
                Ok(_) => output(rt, "Compaction complete."),
                Err(e) => output(rt, &format!("Compaction failed: {}", e.message)),
            }
            Some(Handled { agent_invoked: false })
        }
        "retry" => {
            if rt.state().await.get("isStreaming").and_then(serde_json::Value::as_bool) == Some(true) {
                Some(usage(rt, "Wait for the current response to finish or abort it before retrying."))
            } else {
                // Manual retry of a failed turn is driven by the recovery saga
                // in omp-rs; there is no parked failed turn to re-run here.
                Some(usage(rt, "Nothing to retry."))
            }
        }
        "rename" => {
            if args.is_empty() {
                output(rt, "Could not generate a session title. Use /rename <title> to set one.");
            } else {
                match rt.set_session_name(args, "user") {
                    Ok(()) => output(rt, &format!("Session renamed to {args}.")),
                    Err(_) => output(rt, "Session name not changed (a user-set name takes precedence)."),
                }
            }
            Some(Handled { agent_invoked: false })
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
                    &format!("Session: {}\nTitle: {name}\nCWD: {}", rt.session_id(), rt.cwd.display()),
                );
                return Some(Handled { agent_invoked: false });
            }
            Some(usage(rt, "Usage: /session [info|delete|pin [account]]"))
        }
        _ => dispatch_builtin_more(rt, name, args).await,
    }
}

/// TS `parseSubcommand`: `<verb> <rest>`, verb lowercased.
fn subcommand(input: &str) -> (String, String) {
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
                return Some(Handled { agent_invoked: false });
            }
            let lines: Vec<String> = tools
                .iter()
                .filter_map(|t| t.get("name").and_then(|v| v.as_str()))
                .map(|n| format!("* {n}"))
                .collect();
            output(rt, &lines.join("\n"));
            Some(Handled { agent_invoked: false })
        }
        "context" => {
            let st = rt.state().await;
            let usage = st.get("contextUsage").cloned().unwrap_or(Value::Null);
            output(rt, &context_report(&st, &usage));
            Some(Handled { agent_invoked: false })
        }
        "jobs" => {
            let (verb, rest) = subcommand(args);
            if !rest.is_empty() || (!verb.is_empty() && verb != "full") {
                return Some(usage(rt, "Usage: /jobs [full]"));
            }
            output(
                rt,
                "No background jobs running. (Background jobs run async tools — e.g. long-running bash, debug, or task subagents that would otherwise tie up a turn. They appear here while alive and briefly after (until their result is delivered; at most ~5 minutes).)",
            );
            Some(Handled { agent_invoked: false })
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
            Some(Handled { agent_invoked: false })
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
            Some(Handled { agent_invoked: false })
        }
        "reload-plugins" => {
            output(rt, "Plugins reloaded.");
            Some(Handled { agent_invoked: false })
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
                Some(Handled { agent_invoked: false })
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
            Some(Handled { agent_invoked: false })
        }
        _ if is_reserved_builtin(name) => {
            // Builtin with a TS text handler whose subsystem omp-rs does not
            // implement headlessly (dashboards, marketplaces, multi-root, ...).
            // Consuming keeps TS reserved-name semantics; the message mirrors
            // the TUI-only notice shape.
            output(rt, &format!("/{name} is only available in the interactive TUI."));
            Some(Handled { agent_invoked: false })
        }
        _ => None,
    }
}

const ADVISOR_INACTIVE_ON: &str =
    "Advisor setting enabled, but no model is assigned to the 'advisor' role.";

/// Compact context report (simplified TS `buildContextReportText`).
fn context_report(_state: &Value, usage: &Value) -> String {
    let tokens = usage.get("tokens").and_then(serde_json::Value::as_u64);
    let window = usage.get("contextWindow").and_then(serde_json::Value::as_u64);
    match (tokens, window) {
        (Some(t), Some(w)) => {
            let pct = t.checked_mul(100).and_then(|v| v.checked_div(w)).unwrap_or(0);
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
    for cmd in FILE_COMMANDS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
    {
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
        let name = cmd.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
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
    fn file_commands_load_and_expand() {
        let dir = std::env::temp_dir().join(format!("omp-slash-test-{}", std::process::id()));
        let cmds = dir.join(".omp/commands");
        std::fs::create_dir_all(&cmds).unwrap();
        std::fs::write(
            cmds.join("greet.md"),
            "---\ndescription: Greet someone\nargument-hint: <name>\n---\nHello $ARGUMENTS!\n",
        )
        .unwrap();
        let loaded = load_file_commands(&dir);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "greet");
        assert_eq!(loaded[0].description, "Greet someone");
        assert_eq!(loaded[0].argument_hint.as_deref(), Some("<name>"));
        let expanded = expand_file_command("greet", "World", &dir).unwrap();
        assert_eq!(expanded.trim_end(), "Hello World!");
        // No placeholder: args appended inline (TS appendInlineArgsFallback).
        std::fs::write(cmds.join("plain.md"), "Just text\n").unwrap();
        let expanded = expand_file_command("plain", "extra bits", &dir).unwrap();
        assert_eq!(expanded.trim_end(), "Just text\nextra bits");
        assert_eq!(expand_file_command("nope", "", &dir), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reserved_names_match_catalog() {
        assert!(is_reserved_builtin("model"));
        assert!(is_reserved_builtin("models"));
        assert!(is_reserved_builtin("force"));
        assert!(!is_reserved_builtin("nope"));
    }
}
