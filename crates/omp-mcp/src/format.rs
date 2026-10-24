//! Result formatting: a port of `tool-bridge.ts` `buildResult`/
//! `formatMCPContent`/`formatStructuredContent`.
//!
//! Text blocks are joined with blank lines; resources render as
//! `[Resource: <uri>]` (+ text); errors prefix the first text block with
//! `Error: `; `structuredContent` appends a fenced `json` block unless a text
//! block already carries that JSON verbatim.

use serde_json::Value;

use crate::{Content, ToolCallResult};

/// Format MCP content blocks into a single model-facing text, per omp's
/// `formatMCPContent` (text joined by `\n\n`, resources bracketed).
#[must_use]
pub fn format_content(content: &[Content]) -> String {
    let mut blocks: Vec<String> = Vec::new();
    let mut text = String::new();
    let flush = |text: &mut String, blocks: &mut Vec<String>| {
        if !text.is_empty() {
            blocks.push(std::mem::take(text));
        }
    };
    for item in content {
        match item {
            Content::Text { text: t } => {
                if !text.is_empty() {
                    text.push_str("\n\n");
                }
                text.push_str(t);
            }
            Content::Resource { uri, text: Some(t) } => {
                if !text.is_empty() {
                    text.push_str("\n\n");
                }
                text.push_str(&format!("[Resource: {uri}]\n{t}"));
            }
            Content::Resource { uri, text: None } => {
                if !text.is_empty() {
                    text.push_str("\n\n");
                }
                text.push_str(&format!("[Resource: {uri}]"));
            }
            Content::Image { .. } | Content::Other(_) => {}
        }
    }
    flush(&mut text, &mut blocks);
    if blocks.is_empty() {
        String::new()
    } else {
        blocks.join("\n\n")
    }
}

/// Serialize structured content as omp's fenced JSON block.
#[must_use]
pub fn format_structured_content(structured: &Value) -> String {
    format!(
        "```json\n{}\n```",
        serde_json::to_string_pretty(structured).unwrap_or_default()
    )
}

/// True when a text block already carries the structured payload verbatim.
fn structured_already_in_text(structured: &Value, content: &[Content]) -> bool {
    let expected = serde_json::to_string(structured).unwrap_or_default();
    content.iter().any(|item| match item {
        Content::Text { text } => {
            let trimmed = text.trim();
            !trimmed.is_empty()
                && serde_json::from_str::<Value>(trimmed)
                    .map(|v| serde_json::to_string(&v).unwrap_or_default() == expected)
                    .unwrap_or(false)
        }
        _ => false,
    })
}

/// Build the model-facing text of a tool result, following omp `buildResult`:
/// error prefix, then content, then (if not duplicated) structured content.
#[must_use]
pub fn format_tool_result(result: &ToolCallResult) -> String {
    let body = format_content(&result.content);
    let body = if result.is_error && !body.is_empty() {
        format!("Error: {body}")
    } else if result.is_error {
        "Error:".to_string()
    } else {
        body
    };
    match &result.structured_content {
        Some(structured)
            if !structured.is_null()
                && !structured_already_in_text(structured, &result.content) =>
        {
            let rendered = format_structured_content(structured);
            if body.is_empty() {
                rendered
            } else {
                format!("{body}\n\n{rendered}")
            }
        }
        _ => body,
    }
}
