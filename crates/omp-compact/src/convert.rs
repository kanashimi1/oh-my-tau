//! `convertToLlm`: agent messages → the provider-facing messages the model is sent
//! (pi-agent-core `compaction/messages.ts#convertMessageToLlm` + coding-agent
//! `session/messages.ts#convertToLlm`).
//!
//! Ported roles: `user`, `developer`, `assistant`, `toolResult`, `custom`, `hookMessage`,
//! `branchSummary`, `compactionSummary`, `bashExecution`, `pythonExecution`. Not ported (they
//! pass through as plain developer/user messages): steering / user-invoked-skill custom messages,
//! image-bearing custom-message splitting, `fileMention`, interrupted-thinking demotion,
//! `xd://` tool-name canonicalisation, and the `OutputMeta` notice appended to bash/python text.

use serde_json::{Map, Value};

use crate::json::{message_role, str_field, truthy};
use crate::messages::{
    render_branch_summary_context, render_compaction_summary_context,
    render_handoff_summary_context,
};

fn text_block(text: &str) -> Value {
    let mut b = Map::new();
    b.insert("type".into(), "text".into());
    b.insert("text".into(), text.into());
    Value::Object(b)
}

fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// `bashExecutionToText` (without the `OutputMeta` notice).
#[must_use]
pub fn bash_execution_to_text(msg: &Value) -> String {
    let mut text = format!("Ran `{}`\n", str_field(msg, "command").unwrap_or(""));
    match str_field(msg, "output").filter(|o| !o.is_empty()) {
        Some(output) => text.push_str(&format!("```\n{output}\n```")),
        None => text.push_str("(no output)"),
    }
    if truthy(msg.get("cancelled")) {
        text.push_str("\n\n(command cancelled)");
    } else if let Some(code) = msg
        .get("exitCode")
        .filter(|c| !c.is_null())
        .filter(|c| c.as_f64() != Some(0.0))
    {
        text.push_str(&format!(
            "\n\nCommand exited with code {}",
            code.as_f64().map_or_else(|| code.to_string(), js_number)
        ));
    }
    text
}

/// `pythonExecutionToText` (without the `OutputMeta` notice).
#[must_use]
pub fn python_execution_to_text(msg: &Value) -> String {
    let mut text = format!(
        "Ran Python:\n```python\n{}\n```\n",
        str_field(msg, "code").unwrap_or("")
    );
    match str_field(msg, "output").filter(|o| !o.is_empty()) {
        Some(output) => text.push_str(&format!("Output:\n```\n{output}\n```")),
        None => text.push_str("(no output)"),
    }
    if truthy(msg.get("cancelled")) {
        text.push_str("\n\n(execution cancelled)");
    } else if let Some(code) = msg
        .get("exitCode")
        .filter(|c| !c.is_null())
        .filter(|c| c.as_f64() != Some(0.0))
    {
        text.push_str(&format!(
            "\n\nExecution failed with code {}",
            code.as_f64().map_or_else(|| code.to_string(), js_number)
        ));
    }
    text
}

/// JS `String(number)` for the integral exit codes tools record.
pub(crate) fn js_number(f: f64) -> String {
    if f.fract() == 0.0 && f.abs() < 1e15 {
        format!("{f:.0}")
    } else {
        f.to_string()
    }
}

/// `getPrunedToolResultContent`: a pruned tool result keeps its text (or `[Output truncated]`)
/// collapsed into the first text block's slot.
fn pruned_tool_result_content(message: &Value) -> Value {
    let content = message
        .get("content")
        .cloned()
        .unwrap_or(Value::Array(Vec::new()));
    if message.get("prunedAt").is_none_or(Value::is_null) {
        return content;
    }
    let Value::Array(blocks) = &content else {
        return content;
    };
    let text: String = blocks
        .iter()
        .filter(|b| str_field(b, "type") == Some("text"))
        .filter_map(|b| str_field(b, "text"))
        .collect();
    let text = if text.is_empty() {
        "[Output truncated]".to_owned()
    } else {
        text
    };
    let Some(first_text) = blocks
        .iter()
        .position(|b| str_field(b, "type") == Some("text"))
    else {
        let mut out = vec![text_block(&text)];
        out.extend(blocks.iter().cloned());
        return Value::Array(out);
    };
    let mut out = Vec::new();
    for (i, block) in blocks.iter().enumerate() {
        if str_field(block, "type") != Some("text") {
            out.push(block.clone());
        } else if i == first_text {
            out.push(text_block(&text));
        }
    }
    Value::Array(out)
}

/// Clone a message and default its `attribution` (`message.attribution ?? default`).
fn with_default_attribution(message: &Value, default: &str) -> Value {
    let mut m = message.as_object().cloned().unwrap_or_default();
    if m.get("attribution").is_none_or(Value::is_null) {
        m.insert("attribution".into(), default.into());
    }
    Value::Object(m)
}

fn custom_content(message: &Value) -> Option<Value> {
    match message.get("content")? {
        Value::String(s) => Some(Value::Array(vec![text_block(s)])),
        c @ Value::Array(_) => Some(c.clone()),
        _ => None,
    }
}

/// Convert one agent message to its provider fragment (zero or one message in the ported roles).
#[must_use]
pub fn convert_message(message: &Value) -> Option<Value> {
    let timestamp = message.get("timestamp").cloned().unwrap_or(Value::Null);
    match message_role(message) {
        "bashExecution" | "pythonExecution" => {
            if truthy(message.get("excludeFromContext")) {
                return None;
            }
            let text = if message_role(message) == "bashExecution" {
                bash_execution_to_text(message)
            } else {
                python_execution_to_text(message)
            };
            let mut content = vec![text_block(&text)];
            if let Some(images) = message.get("images").and_then(Value::as_array) {
                content.extend(images.iter().cloned());
            }
            Some(obj(vec![
                ("role", "user".into()),
                ("content", Value::Array(content)),
                ("attribution", "user".into()),
                ("timestamp", timestamp),
            ]))
        }
        "custom" | "hookMessage" => {
            let content = custom_content(message)?;
            let mut pairs = vec![("role", Value::from("developer")), ("content", content)];
            if let Some(a) = message.get("attribution").filter(|a| !a.is_null()) {
                pairs.push(("attribution", a.clone()));
            }
            pairs.push(("timestamp", timestamp));
            Some(obj(pairs))
        }
        "branchSummary" => Some(obj(vec![
            ("role", "user".into()),
            (
                "content",
                Value::Array(vec![text_block(&render_branch_summary_context(
                    str_field(message, "summary").unwrap_or(""),
                ))]),
            ),
            ("attribution", "agent".into()),
            ("historyRewriteAt", timestamp.clone()),
            ("timestamp", timestamp),
        ])),
        "compactionSummary" => {
            let summary = str_field(message, "summary").unwrap_or("");
            let content = if let Some(blocks) = message.get("blocks").and_then(Value::as_array) {
                let mut c = vec![text_block(summary)];
                c.extend(blocks.iter().cloned());
                c
            } else {
                let rendered = if str_field(message, "method") == Some("handoff") {
                    render_handoff_summary_context(summary)
                } else {
                    render_compaction_summary_context(summary)
                };
                let mut c = vec![text_block(&rendered)];
                if let Some(images) = message.get("images").and_then(Value::as_array) {
                    c.extend(images.iter().cloned());
                }
                c
            };
            let rewrite = message
                .get("historyRewriteAt")
                .filter(|v| !v.is_null())
                .cloned()
                .unwrap_or_else(|| timestamp.clone());
            let mut pairs = vec![
                ("role", Value::from("user")),
                ("content", Value::Array(content)),
                ("attribution", "agent".into()),
                ("historyRewriteAt", rewrite),
            ];
            if let Some(p) = message.get("providerPayload").filter(|p| !p.is_null()) {
                pairs.push(("providerPayload", p.clone()));
            }
            pairs.push(("timestamp", timestamp));
            Some(obj(pairs))
        }
        "user" => Some(with_default_attribution(message, "user")),
        "developer" => Some(with_default_attribution(message, "agent")),
        "assistant" => Some(message.clone()),
        "toolResult" => {
            let mut m = message.as_object().cloned().unwrap_or_default();
            m.insert("content".into(), pruned_tool_result_content(message));
            if m.get("attribution").is_none_or(Value::is_null) {
                m.insert("attribution".into(), "agent".into());
            }
            Some(Value::Object(m))
        }
        _ => None,
    }
}

/// [`convert_message`] that borrows the messages it would copy verbatim
/// (assistant turns, user/developer turns that already carry an attribution),
/// so converting a long history does not deep-copy it.
pub(crate) fn convert_message_cow(message: &Value) -> Option<std::borrow::Cow<'_, Value>> {
    let verbatim = match message_role(message) {
        "assistant" => true,
        "user" | "developer" => {
            message.is_object() && !message.get("attribution").is_none_or(Value::is_null)
        }
        _ => false,
    };
    if verbatim {
        Some(std::borrow::Cow::Borrowed(message))
    } else {
        convert_message(message).map(std::borrow::Cow::Owned)
    }
}

/// `convertToLlm`: the messages sent to the provider for these agent messages.
#[must_use]
pub fn convert_to_llm(messages: &[Value]) -> Vec<Value> {
    messages.iter().filter_map(convert_message).collect()
}
