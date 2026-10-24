//! Conversation → plain text for summarization prompts (`compaction/utils.ts`).
//!
//! The dialect-less serializer (`serializeConversation(messages)` with no dialect:
//! `[User]:` / `[Assistant]:` / `[Think]:` / `[Tool Call]:` / `[Tool Result]:` blocks). omp's
//! summarization callers always pass a dialect, so requests use [`crate::dialect`] instead; this
//! plain form is kept for completeness.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;

use crate::json::{message_role, str_field, truthy};

const TOOL_RESULT_MAX_CHARS: usize = 2000;

/// JS `String.length` (UTF-16 code units).
pub(crate) fn utf16_len(s: &str) -> usize {
    s.encode_utf16().count()
}

/// `s.slice(0, n)` over UTF-16 code units (never splits a surrogate pair: trims it).
pub(crate) fn utf16_prefix(s: &str, n: usize) -> String {
    let mut units = 0;
    let mut out = String::new();
    for ch in s.chars() {
        let w = ch.len_utf16();
        if units + w > n {
            break;
        }
        units += w;
        out.push(ch);
    }
    out
}

/// `truncateToolResultForSummary`.
#[must_use]
pub fn truncate_tool_result_for_summary(text: &str) -> String {
    let len = utf16_len(text);
    if len <= TOOL_RESULT_MAX_CHARS {
        return text.to_owned();
    }
    format!(
        "{}\n\n[... {} more characters truncated]",
        utf16_prefix(text, TOOL_RESULT_MAX_CHARS),
        len - TOOL_RESULT_MAX_CHARS
    )
}

/// `escapeSummaryBoundaryTags`: stop untrusted text from closing harness-owned boundaries.
#[must_use]
pub fn escape_summary_boundary_tags(text: &str) -> String {
    static CELL: OnceLock<Regex> = OnceLock::new();
    let re = CELL.get_or_init(|| {
        Regex::new(r"(?i)<\s*/?\s*(?:conversation|previous-summary)\s*>").expect("boundary regex")
    });
    re.replace_all(text, |caps: &regex::Captures<'_>| {
        format!("&lt;{}", &caps[0][1..])
    })
    .into_owned()
}

fn text_blocks(content: &Value, sep: &str) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|b| str_field(b, "type") == Some("text"))
            .filter_map(|b| str_field(b, "text"))
            .collect::<Vec<_>>()
            .join(sep),
        _ => String::new(),
    }
}

fn render_tool_calls(calls: &[&Value]) -> String {
    calls
        .iter()
        .map(|call| {
            let args = call
                .get("arguments")
                .and_then(Value::as_object)
                .map(|o| {
                    o.iter()
                        .map(|(k, v)| {
                            format!(
                                "{k}={}",
                                omp_session::entries::to_js_string(v)
                                    .unwrap_or_else(|_| "null".into())
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            format!("{}({args})", str_field(call, "name").unwrap_or(""))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// `serializeConversation(messages)` without a dialect.
#[must_use]
pub fn serialize_conversation(messages: &[Value]) -> String {
    let mut useless: Vec<&str> = Vec::new();
    for msg in messages {
        if message_role(msg) == "toolResult"
            && msg.get("useless") == Some(&Value::Bool(true))
            && msg.get("isError") != Some(&Value::Bool(true))
        {
            if let Some(id) = str_field(msg, "toolCallId") {
                useless.push(id);
            }
        }
    }
    let mut parts: Vec<String> = Vec::new();
    for msg in messages {
        match message_role(msg) {
            "user" => {
                let content = msg
                    .get("content")
                    .map(|c| text_blocks(c, ""))
                    .unwrap_or_default();
                if !content.is_empty() {
                    parts.push(format!("[User]: {content}"));
                }
            }
            "assistant" => {
                let (mut text, mut thinking, mut calls) = (Vec::new(), Vec::new(), Vec::new());
                for block in msg
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    match str_field(block, "type") {
                        Some("text") => text.push(str_field(block, "text").unwrap_or("")),
                        Some("thinking") => {
                            thinking.push(str_field(block, "thinking").unwrap_or(""));
                        }
                        Some("toolCall")
                            if !str_field(block, "id").is_some_and(|id| useless.contains(&id)) =>
                        {
                            calls.push(block);
                        }
                        _ => {}
                    }
                }
                if !thinking.is_empty() {
                    parts.push(format!("[Think]: {}", thinking.join("\n")));
                }
                if !text.is_empty() {
                    parts.push(format!("[Assistant]: {}", text.join("\n")));
                }
                if !calls.is_empty() {
                    parts.push(format!("[Tool Call]: {}", render_tool_calls(&calls)));
                }
            }
            "toolResult" => {
                if str_field(msg, "toolCallId").is_some_and(|id| useless.contains(&id)) {
                    continue;
                }
                let content = msg
                    .get("content")
                    .map(|c| text_blocks(c, ""))
                    .unwrap_or_default();
                if truthy(Some(&Value::String(content.clone()))) {
                    parts.push(format!(
                        "[Tool Result]: {}",
                        truncate_tool_result_for_summary(&content)
                    ));
                }
            }
            _ => {}
        }
    }
    parts.join("\n\n")
}

/// `serializeConversationForSummary(messages)` (no dialect): serialized and boundary-escaped.
#[must_use]
pub fn serialize_conversation_for_summary(messages: &[Value]) -> String {
    escape_summary_boundary_tags(&serialize_conversation(messages))
}
