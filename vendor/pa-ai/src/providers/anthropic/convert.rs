//! Anthropic Messages conversion: content blocks, messages, and tools.
//!
//! Messages and tools are built as borrowing [`Wire`] trees over the transformed history and the
//! context: the request body is serialized straight from them, byte-identical to the former
//! `serde_json::Value` trees (`sanitize_surrogates` is the identity on Rust strings, so text is
//! borrowed as-is).

use std::borrow::Cow;

use serde_json::json;

use crate::providers::anthropic::{to_claude_code_name, CacheControl};
use crate::types::{
    user_block_payload, AssistantContent, Context, Message, Model, StopReason, Tool,
    UserBlockPayload, UserMessageContent, UserOrToolContent,
};
use crate::utils_inner::wire_json::{obj, Wire, WireObject};

pub(crate) fn normalize_tool_call_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

/// The transformed history a request borrows from (TS `transformMessages` with Anthropic's
/// tool-call ID normalizer).
pub(crate) fn transform_for_request(context: &Context, model: &Model) -> Vec<Message> {
    crate::providers::transform_messages::transform_messages_with_normalizer(
        &context.messages,
        model,
        &|id, _model, _source| Some(normalize_tool_call_id(id)),
    )
}

fn text_part(text: Wire<'_>) -> Wire<'_> {
    obj([("type", Wire::str("text")), ("text", text)])
}

fn image_part<'a>(data: &'a str, mime_type: &'a str) -> Wire<'a> {
    obj([
        ("type", Wire::str("image")),
        (
            "source",
            obj([
                ("type", Wire::str("base64")),
                ("media_type", Wire::str(mime_type)),
                ("data", Wire::str(data)),
            ]),
        ),
    ])
}

pub(crate) fn convert_content_blocks(content: &[UserOrToolContent]) -> Wire<'_> {
    let has_images = content
        .iter()
        .any(|block| matches!(block, UserOrToolContent::Image(_)));
    if !has_images {
        return Wire::join(
            content.iter().map(|block| match user_block_payload(block) {
                UserBlockPayload::Text(text) => Cow::Borrowed(text),
                UserBlockPayload::Image { data, .. } => Cow::Borrowed(data),
                UserBlockPayload::Opaque(json) => Cow::Owned(json),
            }),
            "\n",
        );
    }
    let mut has_text = false;
    let mut blocks: Vec<Wire<'_>> = content
        .iter()
        .map(|block| match user_block_payload(block) {
            UserBlockPayload::Text(text) => {
                has_text = true;
                text_part(Wire::str(text))
            }
            UserBlockPayload::Image { data, mime_type } => image_part(data, mime_type),
            UserBlockPayload::Opaque(json) => {
                has_text = true;
                text_part(Wire::from(json))
            }
        })
        .collect();
    if !has_text {
        blocks.insert(0, text_part(Wire::str("(see attached image)")));
    }
    Wire::Array(blocks)
}

/// Anthropic `messages` params for `transformed` ([`transform_for_request`] of the context).
// Long by design: mirrors the provider's stream shape.
#[allow(clippy::too_many_lines)]
pub(crate) fn convert_messages<'a>(
    transformed: &'a [Message],
    is_oauth_token: bool,
    cache_control: Option<&CacheControl>,
) -> Vec<Wire<'a>> {
    let mut params: Vec<Wire<'a>> = Vec::new();

    let mut index = 0usize;
    while index < transformed.len() {
        let msg: &'a Message = &transformed[index];
        match msg {
            Message::User(user) => match &user.content {
                UserMessageContent::Text(text) => {
                    if !text.trim().is_empty() {
                        params.push(obj([
                            ("role", Wire::str("user")),
                            ("content", Wire::str(text)),
                        ]));
                    }
                }
                UserMessageContent::Blocks(blocks) => {
                    // Text parts that are blank are dropped; images always stay.
                    let filtered: Vec<Wire<'a>> = blocks
                        .iter()
                        .filter_map(|item| match user_block_payload(item) {
                            UserBlockPayload::Text(text) => {
                                (!text.trim().is_empty()).then(|| text_part(Wire::str(text)))
                            }
                            UserBlockPayload::Image { data, mime_type } => {
                                Some(image_part(data, mime_type))
                            }
                            UserBlockPayload::Opaque(json) => {
                                (!json.trim().is_empty()).then(|| text_part(Wire::from(json)))
                            }
                        })
                        .collect();
                    if filtered.is_empty() {
                        index += 1;
                        continue;
                    }
                    params.push(obj([
                        ("role", Wire::str("user")),
                        ("content", Wire::Array(filtered)),
                    ]));
                }
            },
            Message::Assistant(assistant) => {
                let mut blocks: Vec<Wire<'a>> = Vec::new();
                for block in &assistant.content {
                    match block {
                        AssistantContent::Text(text) => {
                            if text.text.trim().is_empty() {
                                continue;
                            }
                            blocks.push(text_part(Wire::str(&text.text)));
                        }
                        AssistantContent::Thinking(thinking) => {
                            if thinking.redacted.unwrap_or(false) {
                                blocks.push(obj([
                                    ("type", Wire::str("redacted_thinking")),
                                    (
                                        "data",
                                        Wire::str(
                                            thinking.thinking_signature.as_deref().unwrap_or(""),
                                        ),
                                    ),
                                ]));
                                continue;
                            }
                            if thinking.thinking.trim().is_empty() {
                                continue;
                            }
                            let signature = thinking.thinking_signature.as_deref().unwrap_or("");
                            if signature.trim().is_empty() {
                                blocks.push(text_part(Wire::str(&thinking.thinking)));
                            } else {
                                blocks.push(obj([
                                    ("type", Wire::str("thinking")),
                                    ("thinking", Wire::str(&thinking.thinking)),
                                    ("signature", Wire::str(signature)),
                                ]));
                            }
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            blocks.push(obj([
                                ("type", Wire::str("tool_use")),
                                ("id", Wire::str(&tool_call.id)),
                                (
                                    "name",
                                    if is_oauth_token {
                                        Wire::from(to_claude_code_name(&tool_call.name))
                                    } else {
                                        Wire::str(&tool_call.name)
                                    },
                                ),
                                ("input", Wire::RefMap(&tool_call.arguments)),
                            ]));
                        }
                    }
                }
                if blocks.is_empty() {
                    index += 1;
                    continue;
                }
                params.push(obj([
                    ("role", Wire::str("assistant")),
                    ("content", Wire::Array(blocks)),
                ]));
            }
            Message::ToolResult(_tool_result) => {
                // Collect all consecutive toolResult messages into one user turn.
                let mut tool_results: Vec<Wire<'a>> = Vec::new();
                let mut j = index;
                while j < transformed.len() {
                    let Message::ToolResult(result) = &transformed[j] else {
                        break;
                    };
                    tool_results.push(obj([
                        ("type", Wire::str("tool_result")),
                        ("tool_use_id", Wire::str(&result.tool_call_id)),
                        ("content", convert_content_blocks(&result.content)),
                        ("is_error", Wire::Bool(result.is_error)),
                    ]));
                    j += 1;
                }
                index = j;
                params.push(obj([
                    ("role", Wire::str("user")),
                    ("content", Wire::Array(tool_results)),
                ]));
                continue;
            }
        }
        index += 1;
    }

    // Add cache_control to the last user message to cache conversation history.
    if let Some(cache_control) = cache_control {
        if let Some(Wire::Object(last)) = params.last_mut() {
            if last.get("role").and_then(Wire::as_str) == Some("user") {
                if let Some(content) = last.get_mut("content") {
                    if content.string_is_empty().is_some() {
                        // A string becomes one text part carrying the marker (moved, not copied).
                        let text = std::mem::replace(content, Wire::Null);
                        *content = Wire::Array(vec![obj([
                            ("type", Wire::str("text")),
                            ("text", text),
                            ("cache_control", Wire::Owned(cache_control.to_json())),
                        ])]);
                    } else if let Some(last_block) =
                        content.as_array_mut().and_then(|blocks| blocks.last_mut())
                    {
                        let block_type = last_block.get("type").and_then(Wire::as_str).unwrap_or("");
                        if block_type == "text"
                            || block_type == "image"
                            || block_type == "tool_result"
                        {
                            last_block
                                .as_object_mut()
                                .expect("content blocks are objects")
                                .insert("cache_control", cache_control.to_json());
                        }
                    }
                }
            }
        }
    }

    params
}

pub(crate) fn convert_tools<'a>(
    tools: &'a [Tool],
    is_oauth_token: bool,
    supports_eager_tool_input_streaming: bool,
    cache_control: Option<&CacheControl>,
) -> Vec<Wire<'a>> {
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let mut entry = WireObject::new();
            entry.insert(
                "name",
                if is_oauth_token {
                    Wire::from(to_claude_code_name(&tool.name))
                } else {
                    Wire::str(&tool.name)
                },
            );
            entry.insert("description", Wire::str(&tool.description));
            if supports_eager_tool_input_streaming {
                entry.insert("eager_input_streaming", true);
            }
            entry.insert(
                "input_schema",
                obj([
                    ("type", Wire::str("object")),
                    (
                        "properties",
                        tool.parameters
                            .get("properties")
                            .map_or_else(|| Wire::Owned(json!({})), Wire::Ref),
                    ),
                    (
                        "required",
                        tool.parameters
                            .get("required")
                            .map_or_else(|| Wire::Owned(json!([])), Wire::Ref),
                    ),
                ]),
            );
            if let Some(cache_control) = cache_control {
                if index == tools.len() - 1 {
                    entry.insert("cache_control", cache_control.to_json());
                }
            }
            Wire::Object(entry)
        })
        .collect()
}

pub(crate) fn map_stop_reason(reason: &str) -> Result<StopReason, String> {
    match reason {
        "end_turn" | "pause_turn" | "stop_sequence" => Ok(StopReason::Stop),
        "max_tokens" => Ok(StopReason::Length),
        "tool_use" => Ok(StopReason::ToolUse),
        "refusal" | "sensitive" => Ok(StopReason::Error),
        other => Err(format!("Unhandled stop reason: {other}")),
    }
}
