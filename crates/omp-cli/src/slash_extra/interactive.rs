//! Dialog-driven ports of TS interactive-mode selectors and dashboards
//! (`modes/controllers/selector-controller.ts`): `/branch` (`showUserMessageSelector`),
//! `/copy` (`showCopySelector`), `/agents` (`showAgentsDashboard`), `/extensions`
//! (`showExtensionsDashboard`), `/debug` (`showDebugSelector`) and `/jobs` (`showJobsSheet`).
//!
//! Each runs on [`super::dialog`] so the TUI and RPC clients that answer
//! `extension_ui_request` dialogs both get it; without a dialog host the commands fall back to
//! text output.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value};

use super::dialog::{self, Choice, SelectStyle};
use super::output;
use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

const HANDLED: Outcome = Outcome::Handled {
    agent_invoked: false,
};

/// First non-empty line of `text`, clipped to `max` chars.
fn headline(text: &str, max: usize) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    if line.chars().count() > max {
        format!(
            "{}…",
            line.chars().take(max.saturating_sub(1)).collect::<String>()
        )
    } else {
        line.to_string()
    }
}

/// Plain text of a message `content` (string, or the `text` parts).
fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

// =============================================================================
// /branch
// =============================================================================

/// `/branch` (TS `showUserMessageSelector` → `#rewindFromTranscript`): pick a message on the
/// active branch, move the leaf there (user targets rewind past themselves) and put the user
/// message back into the prompt draft.
pub async fn branch_command(rt: &Arc<Runtime>) -> Outcome {
    if !dialog::has_ui() {
        output(rt, "/branch needs an interactive UI (message selector); use the rpc `branch` / `navigate_tree` commands or /fork.");
        return HANDLED;
    }
    if rt.state().await.get("isStreaming").and_then(Value::as_bool) == Some(true) {
        output(
            rt,
            "Wait for the current response to finish or abort it before branching.",
        );
        return HANDLED;
    }
    let mut targets: Vec<(String, Choice, bool)> = Vec::new();
    for entry in rt.branch_entries() {
        if entry.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            continue;
        };
        let role = entry
            .pointer("/message/role")
            .and_then(Value::as_str)
            .unwrap_or("");
        let text = content_text(entry.pointer("/message/content"));
        match role {
            "user" => targets.push((
                id.to_string(),
                Choice::with(format!("› {}", headline(&text, 100)), "user"),
                true,
            )),
            "assistant" if !text.trim().is_empty() => {
                targets.push((
                    id.to_string(),
                    Choice::with(format!("  {}", headline(&text, 100)), "assistant"),
                    false,
                ));
            }
            _ => {}
        }
    }
    if targets.is_empty() {
        output(rt, "No messages to branch from");
        return HANDLED;
    }
    let choices: Vec<Choice> = targets.iter().map(|(_, c, _)| c.clone()).collect();
    let style = SelectStyle {
        fullscreen: true,
        initial: choices.len() - 1,
        ..SelectStyle::default()
    };
    let Some(index) = dialog::select_with(rt, "Branch from message", &choices, style).await else {
        return HANDLED;
    };
    let (entry_id, _, is_user) = &targets[index];
    match crate::tree::navigate_tree(rt, entry_id, &json!({"summarize": false}), None).await {
        Ok(Some(data)) if data.get("cancelled").and_then(Value::as_bool) == Some(true) => {
            output(rt, "Navigation cancelled");
        }
        Ok(data) => {
            let editor_text = data
                .as_ref()
                .and_then(|d| d.get("editorText"))
                .and_then(Value::as_str);
            if let (Some(text), true) = (editor_text, *is_user) {
                dialog::set_editor_text(rt, text);
            }
            rt.send(json!({"type": "session_loaded", "messages": rt.messages().await, "name": rt.session_name()}));
            output(rt, "Rewound to selected point");
        }
        Err(e) => output(rt, &format!("Branch failed: {}", e.message)),
    }
    HANDLED
}

// =============================================================================
// /copy
// =============================================================================

/// Fenced code blocks of `text`: `(language, code)`.
fn code_blocks(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut current: Option<(String, String, String)> = None; // (fence, lang, body)
    for line in text.lines() {
        let trimmed = line.trim_start();
        match &mut current {
            None => {
                let fence: String = trimmed
                    .chars()
                    .take_while(|c| *c == '`' || *c == '~')
                    .collect();
                if fence.len() >= 3
                    && fence
                        .chars()
                        .all(|c| c == fence.chars().next().unwrap_or('`'))
                {
                    let lang = trimmed[fence.len()..]
                        .trim()
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_string();
                    current = Some((fence, lang, String::new()));
                }
            }
            Some((fence, lang, body)) => {
                if trimmed.starts_with(fence.as_str())
                    && trimmed
                        .trim_start_matches(fence.chars().next().unwrap_or('`'))
                        .trim()
                        .is_empty()
                {
                    out.push((std::mem::take(lang), std::mem::take(body)));
                    current = None;
                } else {
                    body.push_str(line);
                    body.push('\n');
                }
            }
        }
    }
    out.into_iter()
        .map(|(l, b)| (l, b.trim_end_matches('\n').to_string()))
        .filter(|(_, b)| !b.is_empty())
        .collect()
}

/// `http(s)://` links of `text`, in order.
fn links(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find("http") {
        let tail = &rest[pos..];
        if tail.starts_with("https://") || tail.starts_with("http://") {
            let end = tail
                .find(|c: char| {
                    c.is_whitespace() || matches!(c, ')' | ']' | '>' | '"' | '\'' | '`' | '<')
                })
                .unwrap_or(tail.len());
            let url = tail[..end].trim_end_matches(['.', ',', ';', ':', '!', '?']);
            if url.len() > "https://".len() {
                out.push(url.to_string());
            }
            rest = &tail[end.max(1)..];
        } else {
            rest = &tail[4..];
        }
    }
    out
}

/// The last `bash`/`eval` tool call's code: `(kind, code)`.
fn last_command(messages: &[Value]) -> Option<(String, String)> {
    messages.iter().rev().find_map(|m| {
        if m.get("role").and_then(Value::as_str) != Some("assistant") {
            return None;
        }
        m.get("content")?.as_array()?.iter().rev().find_map(|p| {
            if p.get("type").and_then(Value::as_str) != Some("toolCall") {
                return None;
            }
            let name = p.get("name").and_then(Value::as_str)?;
            let args = p.get("arguments")?;
            let code = match name {
                "bash" => args.get("command"),
                "eval" | "python" | "js" => args.get("code").or_else(|| args.get("cells")),
                _ => None,
            }?;
            let code = code
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| code.to_string());
            Some((name.to_string(), code))
        })
    })
}

/// Copy candidates, newest first: `(choice, text, status message)`.
fn copy_candidates(messages: &[Value]) -> Vec<(Choice, String, String)> {
    let mut out = Vec::new();
    let assistant_texts: Vec<String> = messages
        .iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
        .map(|m| content_text(m.get("content")))
        .filter(|t| !t.trim().is_empty())
        .collect();
    if let Some(last) = assistant_texts.last() {
        out.push((
            Choice::with("Last assistant message", headline(last, 80)),
            last.clone(),
            "Copied last message to clipboard".into(),
        ));
    }
    let mut blocks = 0;
    'outer: for text in assistant_texts.iter().rev() {
        for (lang, code) in code_blocks(text).into_iter().rev() {
            let label = if lang.is_empty() {
                "Code block".to_string()
            } else {
                format!("Code block ({lang})")
            };
            out.push((
                Choice::with(label, headline(&code, 80)),
                code,
                "Copied code block to clipboard".into(),
            ));
            blocks += 1;
            if blocks >= 20 {
                break 'outer;
            }
        }
    }
    if let Some((kind, code)) = last_command(messages) {
        let what = if kind == "bash" {
            "bash command"
        } else {
            "eval code"
        };
        out.push((
            Choice::with(format!("Last {what}"), headline(&code, 80)),
            code,
            format!("Copied {what} to clipboard"),
        ));
    }
    let mut seen = Vec::new();
    for text in assistant_texts.iter().rev() {
        for url in links(text).into_iter().rev() {
            if !seen.contains(&url) && seen.len() < 10 {
                seen.push(url.clone());
                out.push((
                    Choice::with("Link", url.clone()),
                    url,
                    "Copied link to clipboard".into(),
                ));
            }
        }
    }
    for m in messages
        .iter()
        .rev()
        .filter(|m| {
            matches!(
                m.get("role").and_then(Value::as_str),
                Some("user" | "assistant")
            )
        })
        .skip(1)
        .take(20)
    {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("");
        let text = content_text(m.get("content"));
        if text.trim().is_empty() {
            continue;
        }
        let label = if role == "user" {
            "User message"
        } else {
            "Assistant message"
        };
        out.push((
            Choice::with(label, headline(&text, 80)),
            text,
            format!("Copied {} to clipboard", label.to_lowercase()),
        ));
    }
    out
}

/// `/copy [code|cmd|link]` (TS builtin-collaboration.ts `copy`).
pub async fn copy_command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let messages = rt.messages().await;
    let arg = args.trim().to_lowercase();
    let pick: Option<(String, String)> = match arg.as_str() {
        "" => {
            let candidates = copy_candidates(&messages);
            if candidates.is_empty() {
                output(rt, "Nothing to copy yet.");
                return HANDLED;
            }
            if !dialog::has_ui() {
                let (_, text, _) = &candidates[0];
                output(rt, &format!("/copy needs an interactive UI (copy selector); last assistant message:\n\n{text}"));
                return HANDLED;
            }
            let choices: Vec<Choice> = candidates.iter().map(|(c, _, _)| c.clone()).collect();
            let Some(index) = dialog::select_with(
                rt,
                "Copy to clipboard",
                &choices,
                SelectStyle {
                    searchable: true,
                    ..SelectStyle::default()
                },
            )
            .await
            else {
                return HANDLED;
            };
            let (_, text, status) = &candidates[index];
            Some((text.clone(), status.clone()))
        }
        "code" => {
            let block = messages
                .iter()
                .rev()
                .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
                .find_map(|m| code_blocks(&content_text(m.get("content"))).pop());
            let Some((_, code)) = block else {
                output(rt, "No code block to copy.");
                return HANDLED;
            };
            Some((code, "Copied code block to clipboard".into()))
        }
        "cmd" | "command" => {
            let Some((kind, code)) = last_command(&messages) else {
                output(rt, "No command to copy.");
                return HANDLED;
            };
            Some((
                code,
                format!(
                    "Copied {} to clipboard",
                    if kind == "bash" {
                        "bash command"
                    } else {
                        "eval code"
                    }
                ),
            ))
        }
        "link" | "url" => {
            let link = messages
                .iter()
                .rev()
                .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
                .find_map(|m| links(&content_text(m.get("content"))).pop());
            let Some(link) = link else {
                output(rt, "No link to copy.");
                return HANDLED;
            };
            Some((link, "Copied link to clipboard".into()))
        }
        _ => {
            output(rt, "Usage: /copy [code|cmd|link]");
            return HANDLED;
        }
    };
    if let Some((text, status)) = pick {
        if dialog::has_ui() {
            dialog::copy_to_clipboard(rt, &text);
            output(rt, &status);
        } else {
            output(
                rt,
                &format!("/copy: no clipboard host in this mode; text:\n\n{text}"),
            );
        }
    }
    HANDLED
}

// =============================================================================
// /agents
// =============================================================================

fn discovered_agents(cwd: &std::path::Path) -> Vec<omp_task::agent::AgentDefinition> {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from);
    let mut agents: Vec<_> = omp_task::discovery::discover_user_agents(&home)
        .agents
        .into_iter()
        .chain(omp_task::discovery::discover_project_agents(cwd).agents)
        .collect();
    agents.sort_by(|a, b| a.fields.name.cmp(&b.fields.name));
    agents
}

fn source_name(source: omp_task::agent::AgentSource) -> &'static str {
    use omp_task::agent::AgentSource;
    match source {
        AgentSource::User => "user",
        AgentSource::Project => "project",
        AgentSource::Extension => "extension",
        AgentSource::Bundled => "bundled",
    }
}

fn agent_details(agent: &omp_task::agent::AgentDefinition) -> String {
    let f = &agent.fields;
    let mut lines = vec![
        format!("Agent: {}", f.name),
        format!(
            "Source: {} ({})",
            source_name(agent.source),
            agent.file_path.display()
        ),
    ];
    if !f.description.is_empty() {
        lines.push(format!("Description: {}", f.description));
    }
    lines.push(format!(
        "Model: {}",
        f.model
            .as_ref()
            .filter(|m| !m.is_empty())
            .map_or_else(|| "(default)".to_string(), |m| m.join(", "))
    ));
    if let Some(level) = &f.thinking_level {
        lines.push(format!(
            "Thinking: {}",
            serde_json::to_value(level)
                .ok()
                .and_then(|v| v.as_str().map(str::to_owned))
                .unwrap_or_default()
        ));
    }
    lines.push(format!(
        "Tools: {}",
        f.tools.as_ref().map_or_else(
            || "(all)".to_string(),
            |t| if t.is_empty() {
                "(none)".into()
            } else {
                t.join(", ")
            }
        )
    ));
    if let Some(spawns) = &f.spawns {
        lines.push(format!(
            "Spawns: {}",
            serde_json::to_string(spawns).unwrap_or_default()
        ));
    }
    if !f.listed {
        lines.push("Listed: no (spawnable, hidden from the task summary)".into());
    }
    let prompt = headline(&agent.system_prompt, 200);
    if !prompt.is_empty() {
        lines.push(format!("Prompt: {prompt}"));
    }
    lines.join("\n")
}

/// `/agents` (TS `showAgentsDashboard`): the discovered agent definitions; picking one shows
/// its definition.
pub async fn agents_command(rt: &Arc<Runtime>) -> Outcome {
    let agents = discovered_agents(&rt.cwd);
    if agents.is_empty() {
        output(
            rt,
            "No agent definitions found (~/.omp/agent/agents, .omp/agents).",
        );
        return HANDLED;
    }
    if !dialog::has_ui() {
        let lines: Vec<String> = agents
            .iter()
            .map(|a| {
                format!(
                    "- {} [{}] {}",
                    a.fields.name,
                    source_name(a.source),
                    headline(&a.fields.description, 100)
                )
            })
            .collect();
        output(
            rt,
            &format!("Agents ({}):\n{}", agents.len(), lines.join("\n")),
        );
        return HANDLED;
    }
    let choices: Vec<Choice> = agents
        .iter()
        .map(|a| {
            Choice::with(
                a.fields.name.clone(),
                format!(
                    "[{}] {}",
                    source_name(a.source),
                    headline(&a.fields.description, 100)
                ),
            )
        })
        .collect();
    let style = SelectStyle {
        fullscreen: true,
        ..SelectStyle::default()
    };
    if let Some(index) =
        dialog::select_with(rt, &format!("Agents ({})", agents.len()), &choices, style).await
    {
        output(rt, &agent_details(&agents[index]));
    }
    HANDLED
}

// =============================================================================
// /extensions
// =============================================================================

/// One dashboard row: label, detail line, full report.
struct ExtRow {
    label: String,
    detail: String,
    report: String,
}

async fn extension_rows(rt: &Runtime) -> Vec<ExtRow> {
    let mut rows = Vec::new();
    if let Some(ext) = crate::rpc::slash::extensions() {
        let tools = ext.host.list_tools().await.unwrap_or_default();
        let commands = ext.host.list_commands().await.unwrap_or_default();
        let mut names: Vec<String> = tools
            .iter()
            .map(|t| t.extension.clone())
            .chain(commands.iter().map(|c| c.extension.clone()))
            .collect();
        names.sort();
        names.dedup();
        for name in names {
            let ext_tools: Vec<&omp_ext::ToolInfo> =
                tools.iter().filter(|t| t.extension == name).collect();
            let ext_cmds: Vec<&omp_ext::CommandInfo> =
                commands.iter().filter(|c| c.extension == name).collect();
            let mut report = vec![format!("Extension: {name}")];
            report.push(format!("Tools ({}):", ext_tools.len()));
            report.extend(ext_tools.iter().map(|t| {
                format!(
                    "  - {}: {}",
                    t.name,
                    headline(t.description.as_deref().unwrap_or(""), 100)
                )
            }));
            report.push(format!("Commands ({}):", ext_cmds.len()));
            report.extend(ext_cmds.iter().map(|c| {
                format!(
                    "  - /{}: {}",
                    c.name,
                    headline(c.description.as_deref().unwrap_or(""), 100)
                )
            }));
            rows.push(ExtRow {
                label: name.clone(),
                detail: format!(
                    "extension · {} tool(s), {} command(s)",
                    ext_tools.len(),
                    ext_cmds.len()
                ),
                report: report.join("\n"),
            });
        }
    }
    let configs = omp_mcp::config::load_configs(&rt.cwd);
    let mut mcp: Vec<&String> = configs.keys().collect();
    mcp.sort();
    for name in mcp {
        rows.push(ExtRow {
            label: format!("mcp: {name}"),
            detail: "MCP server (manage with /mcp)".into(),
            report: format!("MCP server: {name}\nManage with /mcp (test, enable/disable, remove)."),
        });
    }
    for cmd in crate::rpc::slash::load_file_commands(&rt.cwd) {
        rows.push(ExtRow {
            label: format!("/{}", cmd.name),
            detail: format!("file command · {}", headline(&cmd.description, 80)),
            report: format!("File command /{}\n{}", cmd.name, cmd.description),
        });
    }
    rows
}

/// `/extensions` and `/status` (TS `showExtensionsDashboard`): loaded extensions with their
/// tools and commands, MCP servers and file commands.
pub async fn extensions_command(rt: &Arc<Runtime>) -> Outcome {
    let rows = extension_rows(rt).await;
    let header = if crate::rpc::slash::extensions().is_some() {
        "Extensions"
    } else {
        "Extensions (extension host not running)"
    };
    if rows.is_empty() {
        output(
            rt,
            &format!("{header}: nothing loaded (no extensions, MCP servers or file commands)."),
        );
        return HANDLED;
    }
    if !dialog::has_ui() {
        let lines: Vec<String> = rows
            .iter()
            .map(|r| format!("- {} — {}", r.label, r.detail))
            .collect();
        output(rt, &format!("{header}:\n{}", lines.join("\n")));
        return HANDLED;
    }
    let choices: Vec<Choice> = rows
        .iter()
        .map(|r| Choice::with(r.label.clone(), r.detail.clone()))
        .collect();
    let style = SelectStyle {
        fullscreen: true,
        ..SelectStyle::default()
    };
    if let Some(index) = dialog::select_with(rt, header, &choices, style).await {
        output(rt, &rows[index].report);
    }
    HANDLED
}

// =============================================================================
// /debug
// =============================================================================

fn system_info(rt: &Runtime) -> String {
    let model = rt.model_info();
    let ext = crate::rpc::slash::extensions().map_or("not running", |_| "running");
    [
        format!("omt {}", env!("CARGO_PKG_VERSION")),
        format!("OS: {} {}", std::env::consts::OS, std::env::consts::ARCH),
        format!("CWD: {}", rt.cwd.display()),
        format!(
            "Session: {} ({})",
            rt.session_id(),
            rt.session_file().display()
        ),
        format!(
            "Model: {}/{}",
            model.get("provider").and_then(Value::as_str).unwrap_or("?"),
            model.get("id").and_then(Value::as_str).unwrap_or("?")
        ),
        format!("Extension host: {ext}"),
    ]
    .join("\n")
}

fn terminal_info() -> String {
    let vars = [
        "TERM",
        "COLORTERM",
        "TERM_PROGRAM",
        "TERM_PROGRAM_VERSION",
        "TMUX",
        "STY",
        "WT_SESSION",
        "WSL_DISTRO_NAME",
        "LANG",
        "COLUMNS",
        "LINES",
    ];
    vars.iter()
        .map(|k| {
            format!(
                "{k}={}",
                std::env::var(k).unwrap_or_else(|_| "(unset)".into())
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// TS "Report: dump session": a JSON bundle (system info + session entries) under
/// `~/.omp-rs/reports/`.
fn dump_report(rt: &Runtime) -> Result<PathBuf, String> {
    let home = std::env::var_os("HOME").ok_or("no HOME")?;
    let dir = PathBuf::from(home).join(".omp-rs").join("reports");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let path = dir.join(format!("report-{}-{stamp}.json", rt.session_id()));
    let bundle = json!({
        "system": system_info(rt),
        "terminal": terminal_info(),
        "sessionFile": rt.session_file(),
        "entries": rt.entries().unwrap_or_default(),
    });
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&bundle).unwrap_or_default(),
    )
    .map_err(|e| e.to_string())?;
    Ok(path)
}

/// `/debug` (TS `DebugSelectorComponent`): the debug actions omp-rs can back.
pub async fn debug_command(rt: &Arc<Runtime>) -> Outcome {
    let actions = [
        ("Report: dump session", "Create report bundle immediately"),
        ("View: system info", "Show environment details"),
        ("View: terminal state", "Terminal-related environment"),
        ("Open: artifact folder", "Show the session artifact folder"),
    ];
    if !dialog::has_ui() {
        output(rt, &system_info(rt));
        return HANDLED;
    }
    let choices: Vec<Choice> = actions.iter().map(|(l, d)| Choice::with(*l, *d)).collect();
    let Some(index) =
        dialog::select_with(rt, "Debug Tools", &choices, SelectStyle::default()).await
    else {
        return HANDLED;
    };
    match index {
        0 => match dump_report(rt) {
            Ok(path) => output(rt, &format!("Report bundle written to {}", path.display())),
            Err(e) => output(rt, &format!("Report failed: {e}")),
        },
        1 => output(rt, &system_info(rt)),
        2 => output(rt, &terminal_info()),
        _ => {
            let file = rt.session_file();
            let dir = file.with_extension("");
            let shown = if dir.is_dir() {
                dir
            } else {
                file.parent().map(PathBuf::from).unwrap_or(file)
            };
            output(rt, &format!("Artifact folder: {}", shown.display()));
        }
    }
    HANDLED
}

// =============================================================================
// /jobs
// =============================================================================

fn elapsed(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{:.1}s", d.as_secs_f64())
    } else if secs < 3600 {
        format!("{}m{}s", secs / 60, secs % 60)
    } else {
        format!("{}h{}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// `/jobs [full]` (TS `showJobsSheet`): background bash jobs and named services; picking one
/// shows its output.
pub async fn jobs_command(rt: &Arc<Runtime>, full: bool) -> Outcome {
    let registry = crate::tools::session_jobs();
    let mut rows: Vec<(Choice, String)> = Vec::new();
    if let Some(reg) = &registry {
        for (id, job) in reg.sorted() {
            let state = match (&job.ended, &job.error) {
                (None, _) => format!("running {}", elapsed(job.started.elapsed())),
                (Some(d), None) => format!("completed in {}", elapsed(*d)),
                (Some(d), Some(_)) => format!("failed after {}", elapsed(*d)),
            };
            let out = job
                .result
                .clone()
                .or_else(|| job.error.clone())
                .unwrap_or_else(|| job.output.lock().map(|o| o.clone()).unwrap_or_default());
            let report = format!("Job {id}: {}\n{state}\n\n{}", job.command, out.trim_end());
            rows.push((
                Choice::with(format!("{id}  {}", headline(&job.command, 70)), state),
                report,
            ));
        }
        for (name, service) in reg.services_sorted() {
            let Ok(s) = service.lock() else { continue };
            let state = format!(
                "service · {}{}",
                s.state.as_str(),
                s.pid.map(|p| format!(" · pid {p}")).unwrap_or_default()
            );
            let report = format!("Service {name}: {state}\n\n{}", s.log());
            rows.push((Choice::with(name.clone(), state), report));
        }
    }
    if rows.is_empty() {
        output(
            rt,
            "No background jobs running. (Background jobs run async tools — e.g. long-running bash, debug, or task subagents that would otherwise tie up a turn. They appear here while alive and briefly after (until their result is delivered; at most ~5 minutes).)",
        );
        return HANDLED;
    }
    if !dialog::has_ui() || full {
        let text: Vec<String> = rows
            .iter()
            .map(|(c, report)| {
                if full {
                    report.clone()
                } else {
                    format!(
                        "- {} — {}",
                        c.label,
                        c.description.clone().unwrap_or_default()
                    )
                }
            })
            .collect();
        output(rt, &text.join(if full { "\n\n" } else { "\n" }));
        return HANDLED;
    }
    let choices: Vec<Choice> = rows.iter().map(|(c, _)| c.clone()).collect();
    if let Some(index) =
        dialog::select_with(rt, "Background jobs", &choices, SelectStyle::default()).await
    {
        output(rt, &rows[index].1);
    }
    HANDLED
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_blocks_and_links_are_extracted() {
        let text = "see https://example.com/a.\n```rust\nfn main() {}\n```\nand\n~~~\nplain\n~~~\n";
        assert_eq!(
            code_blocks(text),
            vec![
                ("rust".into(), "fn main() {}".into()),
                (String::new(), "plain".into())
            ]
        );
        assert_eq!(links(text), vec!["https://example.com/a".to_string()]);
    }

    #[test]
    fn copy_candidates_lead_with_the_last_assistant_message() {
        let messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": [{"type": "text", "text": "answer\n```sh\nls\n```"},
                {"type": "toolCall", "name": "bash", "arguments": {"command": "echo hi"}}]}),
        ];
        let c = copy_candidates(&messages);
        assert_eq!(c[0].0.label, "Last assistant message");
        assert!(c.iter().any(|(_, text, _)| text == "ls"));
        assert!(c.iter().any(|(_, text, _)| text == "echo hi"));
        assert!(c.iter().any(|(_, text, _)| text == "hi"));
    }
}
