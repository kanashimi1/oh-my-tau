//! `history://` provider (TS `internal-urls/history-protocol.ts`): agent
//! transcript index and concise markdown transcripts from `.jsonl` session
//! files. The live-registry paths need the agent registry omp-rs lacks; the
//! on-disk fallback (TS parity after resume/unregister) is implemented.

use std::path::PathBuf;

use crate::parse::InternalUrl;
use crate::spec::{InternalResource, SchemeBacking, SchemeSelectors, SchemeSpec};
use crate::{UrlCtx, UrlProvider};

use crate::schemes::core::session_files_from_disk;

/// TS `isCurrentFullRoute`: exactly `history://current/full`.
fn is_current_full_route(url: &InternalUrl) -> anyhow::Result<bool> {
    let id = url.host.to_lowercase();
    if id != "current" || url.pathname.is_empty() || url.pathname == "/" {
        return Ok(false);
    }
    if url.pathname != "/full" || !url.query.is_empty() || !url.fragment.is_empty() {
        anyhow::bail!("Invalid history://current route; use exactly history://current/full (selectors may follow it)");
    }
    Ok(true)
}

/// TS `#findOnDisk`: exact id wins, else the last case-insensitive match.
fn find_on_disk(agent_id: &str, preferred: Option<&std::path::Path>) -> Option<(String, PathBuf)> {
    let files = session_files_from_disk(preferred);
    let lower = agent_id.to_lowercase();
    let mut matched: Option<(String, PathBuf)> = None;
    for (id, file) in files {
        if id == agent_id {
            return Some((id, file));
        }
        if id.to_lowercase() == lower {
            matched = Some((id, file));
        }
    }
    matched
}

/// Load the `message` payloads of a session file as raw JSON values. Like
/// omp's lenient loader, a file without a valid leading `{"type":"session"}`
/// header yields no entries at all.
fn load_messages(path: &std::path::Path) -> anyhow::Result<Vec<serde_json::Value>> {
    let raw = std::fs::read_to_string(path)?;
    let mut messages = Vec::new();
    let mut saw_header = false;
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if !saw_header {
            if value.get("type").and_then(|t| t.as_str()) == Some("session") {
                saw_header = true;
            } else {
                // First record is not a session header: the file is not a
                // loadable session (TS `loadEntriesFromFile` drops it).
                break;
            }
            continue;
        }
        if let Some(message) = value.get("message") {
            messages.push(message.clone());
        }
    }
    Ok(messages)
}

/// TS `#renderIndex`: registry rows merged with on-disk-only transcripts.
fn render_index(ctx: &UrlCtx) -> String {
    let disk = session_files_from_disk(ctx.artifacts_dir().as_deref());
    let mut lines = vec!["# Agents".to_string(), String::new()];
    if disk.is_empty() {
        lines.push("No agents registered.".to_string());
        return format!("{}\n", lines.join("\n"));
    }
    lines.push("| id | status | kind | parent | last activity |".to_string());
    lines.push("|---|---|---|---|---|".to_string());
    for (id, _file) in &disk {
        lines.push(format!("| {id} | on disk | — | — | — |"));
    }
    lines.push(String::new());
    lines.push("Read a transcript with `read history://<id>`.".to_string());
    format!("{}\n", lines.join("\n"))
}

/// `history://` provider.
pub struct HistoryProvider;

impl UrlProvider for HistoryProvider {
    fn scheme(&self) -> &'static str {
        "history"
    }

    fn spec(&self) -> &SchemeSpec {
        const SPEC: SchemeSpec = SchemeSpec {
            backing: SchemeBacking::Virtual,
            selectors: SchemeSelectors::Lines,
            immutable: false,
            single_slash_alias: false,
            linkable: true,
            port_authority: false,
            write: None,
        };
        &SPEC
    }

    fn impl_locate(&self) -> bool {
        true
    }

    fn read(&self, url: &InternalUrl, ctx: &UrlCtx) -> anyhow::Result<InternalResource> {
        if is_current_full_route(url)? {
            anyhow::bail!(
                "history://current/full is available only when compaction.experimentalContextManagement is enabled"
            );
        }
        let agent_id = url.host.as_str();
        if agent_id.is_empty() {
            return Ok(InternalResource::text(render_index(ctx)));
        }
        let Some((matched_id, file)) = find_on_disk(agent_id, ctx.artifacts_dir().as_deref())
        else {
            // TS lists registered refs only; omp-rs has no live registry yet.
            anyhow::bail!(
                "Unknown agent: {agent_id}\nKnown agents: none\nList all with history://"
            );
        };
        let messages = load_messages(&file)?;
        let content =
            format_session_history_markdown(&messages, Some(&format!("{matched_id} (on disk)")));
        let mut r = InternalResource::text(content);
        r.path = Some(file.to_string_lossy().into_owned());
        Ok(r)
    }

    fn locate(&self, url: &InternalUrl, ctx: &UrlCtx) -> Option<PathBuf> {
        if is_current_full_route(url).ok()? {
            return None;
        }
        let agent_id = url.host.as_str();
        if agent_id.is_empty() {
            return None;
        }
        find_on_disk(agent_id, ctx.artifacts_dir().as_deref()).map(|(_, file)| file)
    }

    fn complete(&self, _query: &str, ctx: &UrlCtx) -> Vec<crate::UrlCompletion> {
        session_files_from_disk(ctx.artifacts_dir().as_deref())
            .into_iter()
            .map(|(id, _)| crate::UrlCompletion {
                label: id,
                detail: Some("on disk".to_string()),
            })
            .collect()
    }
}

/// TS `session-history-format.ts` `formatSessionHistoryMarkdown` (default
/// options: no thinking, no expanded IO, tool calls collapsed to one line),
/// operating on raw message JSON.
pub fn format_session_history_markdown(
    messages: &[serde_json::Value],
    title: Option<&str>,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    if let Some(title) = title {
        lines.push(format!("# {title}"));
        lines.push(String::new());
    }
    // Index tool results by call id so each toolCall collapses to one line.
    let mut results_by_call: Vec<(String, &serde_json::Value)> = Vec::new();
    for msg in messages {
        if msg.get("role").and_then(|r| r.as_str()) == Some("toolResult") {
            if let Some(call_id) = msg.get("toolCallId").and_then(|c| c.as_str()) {
                results_by_call.push((call_id.to_string(), msg));
            }
        }
    }
    let mut consumed: Vec<String> = Vec::new();

    for msg in messages {
        let Some(role) = msg.get("role").and_then(|r| r.as_str()) else {
            continue;
        };
        match role {
            "user" | "developer" => {
                let text = content_to_text(msg.get("content"));
                if text.trim().is_empty() {
                    continue;
                }
                lines.push(format!("## {role}"));
                lines.push(String::new());
                lines.push(text);
                lines.push(String::new());
            }
            "assistant" => {
                let mut body: Vec<String> = Vec::new();
                let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) else {
                    continue;
                };
                for block in blocks {
                    match block.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            let text = block.get("text").and_then(|t| t.as_str()).unwrap_or("");
                            if !text.trim().is_empty() {
                                body.push(text.to_string());
                            }
                        }
                        Some("toolCall") => {
                            let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                            let call_id = block.get("id").and_then(|i| i.as_str()).unwrap_or("");
                            let args = block.get("arguments");
                            let result = results_by_call
                                .iter()
                                .find(|(id, _)| id == call_id)
                                .map(|(_, r)| *r);
                            if result.is_some() {
                                consumed.push(call_id.to_string());
                            }
                            body.push(tool_call_line(name, args, result));
                        }
                        _ => {}
                    }
                }
                if body.is_empty() {
                    continue;
                }
                lines.push("## assistant".to_string());
                lines.push(String::new());
                lines.extend(body);
                lines.push(String::new());
            }
            "toolResult" => {
                let call_id = msg.get("toolCallId").and_then(|c| c.as_str()).unwrap_or("");
                if consumed.iter().any(|c| c == call_id) {
                    continue;
                }
                let tool_name = msg.get("toolName").and_then(|t| t.as_str()).unwrap_or("");
                lines.push(tool_call_line(tool_name, None, Some(msg)));
                lines.push(String::new());
            }
            "bashExecution" => {
                if msg.get("excludeFromContext").and_then(|v| v.as_bool()) == Some(true) {
                    continue;
                }
                let command = msg.get("command").and_then(|c| c.as_str()).unwrap_or("");
                let exit = msg.get("exitCode").and_then(|e| e.as_i64()).unwrap_or(0);
                let cancelled = msg.get("cancelled").and_then(|c| c.as_bool()) == Some(true);
                let status = if cancelled {
                    "cancelled".to_string()
                } else {
                    format!("exit {exit}")
                };
                lines.push(format!("$ {command}  [{status}]"));
                lines.push(String::new());
            }
            "custom" | "hookMessage" => {
                if msg.get("display").and_then(|v| v.as_bool()) == Some(false) {
                    continue;
                }
                let custom_type = msg
                    .get("customType")
                    .and_then(|t| t.as_str())
                    .unwrap_or("custom");
                lines.push(format!(
                    "[{custom_type}] {}",
                    one_line(&content_to_text(msg.get("content")), 120)
                ));
                lines.push(String::new());
            }
            "branchSummary" => {
                let from = msg.get("fromId").and_then(|f| f.as_str()).unwrap_or("");
                let summary = msg.get("summary").and_then(|s| s.as_str()).unwrap_or("");
                lines.push(format!("[branch] from {from}: {}", one_line(summary, 120)));
                lines.push(String::new());
            }
            "compactionSummary" => {
                let summary = msg.get("summary").and_then(|s| s.as_str()).unwrap_or("");
                lines.push(format!("[compaction] {}", one_line(summary, 120)));
                lines.push(String::new());
            }
            "fileMention" => {
                let files = msg
                    .get("files")
                    .and_then(|f| f.as_array())
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|f| f.get("path").and_then(|p| p.as_str()))
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                lines.push(format!("[file-mention] {}", one_line(&files, 120)));
                lines.push(String::new());
            }
            _ => {}
        }
    }
    format!("{}\n", lines.join("\n").trim())
}

/// Join the text blocks of a string-or-blocks content field; images become `[image]`.
fn content_to_text(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(serde_json::Value::Array(blocks)) => blocks
            .iter()
            .map(|block| match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => block
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
                Some("image") => "[image]".to_string(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// TS `oneLine`: collapse whitespace runs, truncate to `max` chars with an ellipsis.
fn one_line(text: &str, max: usize) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > max {
        let truncated: String = flat.chars().take(max - 1).collect();
        format!("{truncated}\u{2026}")
    } else {
        flat
    }
}

/// TS `formatToolCallPrimaryArg` over the PRIMARY_ARG_KEYS preference order.
fn primary_arg(args: Option<&serde_json::Value>) -> String {
    let Some(args) = args else {
        return String::new();
    };
    const KEYS: &[&str] = &[
        "path",
        "file_path",
        "filePath",
        "command",
        "cmd",
        "pattern",
        "url",
        "query",
        "prompt",
        "assignment",
        "note",
        "message",
        "op",
        "name",
        "id",
    ];
    let scalar = |value: Option<&serde_json::Value>| -> Option<String> {
        match value {
            Some(serde_json::Value::String(s)) if !s.is_empty() => Some(s.clone()),
            Some(serde_json::Value::Array(items))
                if !items.is_empty() && items.iter().all(|i| i.is_string()) =>
            {
                Some(
                    items
                        .iter()
                        .filter_map(|i| i.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            }
            _ => None,
        }
    };
    for key in KEYS {
        if let Some(summary) = scalar(args.get(key)) {
            return one_line(&summary, 120);
        }
    }
    // Fallback: first string arg, else compact JSON of the rest.
    if let Some(map) = args.as_object() {
        for (_key, value) in map {
            if let Some(s) = scalar(Some(value)) {
                return one_line(&s, 120);
            }
        }
        one_line(&serde_json::to_string(map).unwrap_or_default(), 120)
    } else {
        String::new()
    }
}

/// TS `toolCallLine` (no intent, no expanded IO): `→ name(arg) ⇒ ok · N lines`.
fn tool_call_line(
    name: &str,
    args: Option<&serde_json::Value>,
    result: Option<&serde_json::Value>,
) -> String {
    let head = format!("\u{2192} {name}({})", primary_arg(args));
    let Some(result) = result else {
        return format!("{head} \u{21d2} pending");
    };
    let raw_text = content_to_text(result.get("content"));
    let count = if raw_text.is_empty() {
        0
    } else {
        raw_text.split('\n').count()
    };
    let count_label = if count == 1 {
        "1 line".to_string()
    } else {
        format!("{count} lines")
    };
    if result.get("isError").and_then(|e| e.as_bool()) == Some(true) {
        let first_line = one_line(raw_text.split('\n').next().unwrap_or(""), 120);
        if first_line.is_empty() {
            format!("{head} \u{21d2} error \u{b7} {count_label}")
        } else {
            format!("{head} \u{21d2} error \u{b7} {count_label} \u{2014} {first_line}")
        }
    } else {
        format!("{head} \u{21d2} ok \u{b7} {count_label}")
    }
}
