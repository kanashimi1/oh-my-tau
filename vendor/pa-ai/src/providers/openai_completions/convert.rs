//! `OpenAI` Completions conversion: reasoning-details signatures, messages, and tools.

use std::borrow::Cow;

use serde_json::Value;

use crate::models::{calculate_cost, CostOverrides};
use crate::providers::openai_completions::compat::ResolvedCompat;
use crate::providers::openai_completions::decode_reasoning_details;
use crate::providers::transform_messages::transform_messages_with_options;
use crate::types::{
    user_block_payload, AssistantContent, Context, Message, MessageExt, Model, StopReason,
    TextContent, ThinkingContent, Tool, ToolCall, Usage, UsageCost, UserBlockPayload,
    UserMessageContent, UserOrToolContent,
};
use crate::utils_inner::wire_json::{obj, Wire, WireObject};

/// The transformed history a request borrows from: tool-call IDs normalized for the provider,
/// unanswered calls answered, errored turns dropped (TS `transformMessages`).
pub(crate) fn transform_for_request(
    model: &Model,
    context: &Context,
    compat: &ResolvedCompat,
) -> Vec<Message> {
    transform_messages_with_options(
        &context.messages,
        model,
        &|id, _, _| Some(normalize_tool_call_id(id, compat)),
        false,
    )
}

/// The `assistant` filler some providers need between tool results and a user turn.
fn processed_tool_results() -> Wire<'static> {
    obj([
        ("role", Wire::str("assistant")),
        ("content", Wire::str("I have processed the tool results.")),
    ])
}

/// An `image_url` part carrying `data:{mime_type};base64,{data}`, written without copying the
/// image data into a fresh URL string.
fn image_url_part<'a>(mime_type: &'a str, data: &'a str) -> Wire<'a> {
    obj([
        ("type", Wire::str("image_url")),
        (
            "image_url",
            obj([(
                "url",
                Wire::Concat(vec![
                    Cow::Borrowed("data:"),
                    Cow::Borrowed(mime_type),
                    Cow::Borrowed(";base64,"),
                    Cow::Borrowed(data),
                ]),
            )]),
        ),
    ])
}

fn text_part(text: Wire<'_>) -> Wire<'_> {
    obj([("type", Wire::str("text")), ("text", text)])
}

/// Placeholder text standing in for images a text-only endpoint cannot take (TS
/// `NON_VISION_IMAGE_PLACEHOLDER`).
const NON_VISION_IMAGE_PLACEHOLDER: &str = "[image omitted: model does not support vision]";

/// The reasoning field names a streamed signature may carry (TS `reasoning_content` /
/// `reasoning` / `reasoning_text`).
fn is_reasoning_field(name: &str) -> bool {
    matches!(name, "reasoning_content" | "reasoning" | "reasoning_text")
}

/// Convert a conversation into Chat Completions `messages` params (TS `convertMessages`), including
/// tool-result bridging and image replay. `transformed` is [`transform_for_request`] of the
/// context; the entries borrow its strings. `system_prompts` are the ordered system prompt blocks:
/// one message each, or one `\n\n`-joined message on hosts whose templates reject follow-up system
/// messages (`compat.supportsMultipleSystemMessages === false`).
#[cfg(test)]
pub(crate) fn convert_messages<'a>(
    model: &Model,
    system_prompts: &[&'a str],
    transformed: &'a [Message],
    compat: &ResolvedCompat,
) -> Vec<Wire<'a>> {
    let mut params: Vec<Wire<'a>> = Vec::new();
    push_system_messages(model, system_prompts, compat, &mut params);
    convert_message_range(model, transformed, compat, 0, None, &mut params);
    params
}

/// The system prompt blocks as leading `messages` entries (TS `convertMessages` system section);
/// extracted from [`convert_messages`] so the wire-fragment cache can re-run a message range
/// without re-deriving them.
pub(crate) fn push_system_messages<'a>(
    model: &Model,
    system_prompts: &[&'a str],
    compat: &ResolvedCompat,
    params: &mut Vec<Wire<'a>>,
) {
    let system_prompts: Vec<&'a str> = system_prompts
        .iter()
        .copied()
        .filter(|p| !p.trim().is_empty())
        .collect();
    if system_prompts.is_empty() {
        return;
    }
    let role = if model.reasoning && compat.supports_developer_role {
        "developer"
    } else {
        "system"
    };
    if compat.supports_multiple_system_messages.unwrap_or(true) {
        for prompt in &system_prompts {
            params.push(obj([
                ("role", Wire::str(role)),
                ("content", Wire::str(prompt)),
            ]));
        }
    } else {
        params.push(obj([
            ("role", Wire::str(role)),
            (
                "content",
                Wire::join(system_prompts.iter().map(|p| Cow::Borrowed(*p)), "\n\n"),
            ),
        ]));
    }
}

/// Convert `transformed[start..]` into `messages` params appended to `params`, starting from the
/// carried `last_role` of everything before `start` (`None` at the beginning of the history).
/// Returns the `last_role` after the range, which a continuation conversion must carry. Extracted
/// from [`convert_messages`] so the wire-fragment cache can convert only the messages added since
/// the cached prefix.
// Long by design: mirrors the provider's stream shape.
#[allow(clippy::too_many_lines)]
pub(crate) fn convert_message_range<'a>(
    model: &Model,
    transformed: &'a [Message],
    compat: &ResolvedCompat,
    start: usize,
    initial_last_role: Option<&'static str>,
    params: &mut Vec<Wire<'a>>,
) -> Option<&'static str> {
    let supports_images = compat.vision_supported(model);
    let mut last_role: Option<&'static str> = initial_last_role;
    let mut index = start;
    while index < transformed.len() {
        let msg: &'a Message = &transformed[index];
        // Some providers (e.g. Mistral/Devstral) don't allow user messages directly after tool
        // results: bridge the gap with a synthetic assistant message.
        if compat.requires_assistant_after_tool_result
            && last_role == Some("toolResult")
            && matches!(msg, Message::User(_) | Message::Developer(_))
        {
            params.push(processed_tool_results());
        }

        match msg {
            Message::User(user) | Message::Developer(user) => match &user.content {
                UserMessageContent::Text(text) => {
                    if text.trim().is_empty() {
                        index += 1;
                        continue;
                    }
                    // TS: only plain-string content keeps the `developer` role (and only when the host
                    // supports it); array content is always sent as `user`.
                    let role =
                        if matches!(msg, Message::Developer(_)) && compat.supports_developer_role {
                            "developer"
                        } else {
                            "user"
                        };
                    params.push(obj([
                        ("role", Wire::str(role)),
                        ("content", Wire::str(text)),
                    ]));
                }
                UserMessageContent::Blocks(blocks) => {
                    let mut content_blocks: Vec<Wire<'a>> = Vec::new();
                    let mut omitted_images = false;
                    for item in blocks {
                        match user_block_payload(item) {
                            UserBlockPayload::Text(text) => {
                                if !text.trim().is_empty() {
                                    content_blocks.push(text_part(Wire::str(text)));
                                }
                            }
                            UserBlockPayload::Image { data, mime_type } => {
                                if supports_images {
                                    content_blocks.push(image_url_part(mime_type, data));
                                } else {
                                    omitted_images = true;
                                }
                            }
                            UserBlockPayload::Opaque(json) => {
                                content_blocks.push(text_part(Wire::from(json)))
                            }
                        }
                    }
                    if omitted_images {
                        content_blocks.push(text_part(Wire::str(NON_VISION_IMAGE_PLACEHOLDER)));
                    }
                    if content_blocks.is_empty() {
                        index += 1;
                        continue;
                    }
                    params.push(obj([
                        ("role", Wire::str("user")),
                        ("content", Wire::Array(content_blocks)),
                    ]));
                }
            },
            Message::Assistant(assistant) => {
                let mut assistant_msg = WireObject::new();
                assistant_msg.insert("role", Wire::str("assistant"));
                assistant_msg.insert("content", Wire::Null);

                let text_blocks: Vec<&'a TextContent> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Text(text) if !text.text.trim().is_empty() => Some(text),
                        _ => None,
                    })
                    .collect();
                // Assistant text is always a plain string; adjacent blocks join without separator.
                let assistant_text_pieces =
                    || text_blocks.iter().map(|&b| Cow::Borrowed(b.text.as_str()));
                let has_assistant_text = !text_blocks.is_empty();
                if has_assistant_text {
                    assistant_msg
                        .insert("content", Wire::Concat(assistant_text_pieces().collect()));
                }

                let replay_reasoning_details: Vec<Value> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Thinking(thinking) => {
                            decode_reasoning_details(thinking.thinking_signature.as_deref())
                        }
                        _ => None,
                    })
                    .flatten()
                    .collect();
                let has_replay_reasoning_details = !replay_reasoning_details.is_empty();
                if has_replay_reasoning_details {
                    assistant_msg
                        .insert("reasoning_details", Value::Array(replay_reasoning_details));
                }

                let all_thinking_blocks: Vec<&'a ThinkingContent> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Thinking(thinking)
                            if decode_reasoning_details(thinking.thinking_signature.as_deref())
                                .is_none() =>
                        {
                            Some(thinking)
                        }
                        _ => None,
                    })
                    .collect();
                let non_empty_thinking_blocks: Vec<&'a ThinkingContent> = all_thinking_blocks
                    .iter()
                    .copied()
                    .filter(|t| !t.thinking.trim().is_empty())
                    .collect();
                let thinking_pieces = || {
                    non_empty_thinking_blocks
                        .iter()
                        .map(|&b| Cow::Borrowed(b.thinking.as_str()))
                };
                let first_signature = non_empty_thinking_blocks
                    .first()
                    .and_then(|t| t.thinking_signature.as_deref());
                let reasoning_field_default = compat
                    .reasoning_content_field
                    .as_deref()
                    .unwrap_or("reasoning_content");

                // Reasoning wire fields the replay may set (`reasoning_content` / `reasoning` /
                // `reasoning_text`), in first-set order.
                let mut reasoning_fields: Vec<(&'static str, Wire<'a>)> = Vec::new();
                let field_name = |name: &str| -> &'static str {
                    match name {
                        "reasoning" => "reasoning",
                        "reasoning_text" => "reasoning_text",
                        _ => "reasoning_content",
                    }
                };

                if compat.mistral_reasoning_content_parts == Some(true)
                    && !non_empty_thinking_blocks.is_empty()
                {
                    // Mistral reads thinking from ordered content parts, not a top-level field.
                    let mut parts: Vec<Wire<'a>> = Vec::new();
                    for block in &assistant.content {
                        match block {
                            AssistantContent::Thinking(t) if !t.thinking.trim().is_empty() => parts
                                .push(obj([
                                    ("type", Wire::str("thinking")),
                                    (
                                        "thinking",
                                        Wire::Array(vec![text_part(Wire::str(&t.thinking))]),
                                    ),
                                ])),
                            AssistantContent::Text(t) if !t.text.trim().is_empty() => {
                                parts.push(text_part(Wire::str(&t.text)));
                            }
                            _ => {}
                        }
                    }
                    assistant_msg.insert("content", Wire::Array(parts));
                } else if !non_empty_thinking_blocks.is_empty() {
                    if compat.requires_thinking_as_text {
                        // TS `renderDemotedThinking` (xml dialect, the fallback every
                        // `requiresThinkingAsText` host resolves to), joined with a space.
                        let mut pieces: Vec<Cow<'a, str>> = Vec::new();
                        for (i, block) in non_empty_thinking_blocks.iter().enumerate() {
                            if i > 0 {
                                pieces.push(Cow::Borrowed(" "));
                            }
                            pieces.push(Cow::Borrowed("<thinking>\n"));
                            pieces.push(Cow::Borrowed(block.thinking.as_str()));
                            pieces.push(Cow::Borrowed("\n</thinking>"));
                        }
                        if has_assistant_text {
                            pieces.push(Cow::Borrowed(" "));
                            pieces.extend(assistant_text_pieces());
                        }
                        assistant_msg.insert("content", Wire::Concat(pieces));
                    } else if compat.requires_reasoning_content_for_tool_calls {
                        let wire_field = match first_signature {
                            Some(sig) if is_reasoning_field(sig) => Some(
                                if compat.allows_synthetic_reasoning_content_for_tool_calls {
                                    field_name(sig)
                                } else {
                                    field_name(reasoning_field_default)
                                },
                            ),
                            _ => None,
                        };
                        if let Some(field) = wire_field {
                            reasoning_fields.push((field, Wire::join(thinking_pieces(), "\n")));
                        }
                    } else if compat.thinking_format == "zai" && model.reasoning {
                        reasoning_fields.push((
                            field_name(reasoning_field_default),
                            Wire::join(thinking_pieces(), "\n"),
                        ));
                    } else if compat.replay_reasoning_content {
                        let field = match first_signature {
                            Some(sig) if is_reasoning_field(sig) => field_name(sig),
                            _ => field_name(reasoning_field_default),
                        };
                        reasoning_fields.push((field, Wire::join(thinking_pieces(), "\n")));
                    }
                }

                if compat.requires_reasoning_content_for_tool_calls {
                    let streamed = non_empty_thinking_blocks
                        .first()
                        .and_then(|t| t.thinking_signature.as_deref());
                    let field = match streamed {
                        Some(sig)
                            if compat.allows_synthetic_reasoning_content_for_tool_calls
                                && is_reasoning_field(sig) =>
                        {
                            field_name(sig)
                        }
                        _ => field_name(reasoning_field_default),
                    };
                    let present = |fields: &[(&'static str, Wire<'a>)], name: &str| {
                        fields
                            .iter()
                            .any(|(n, w)| *n == name && w.string_is_empty() != Some(true))
                    };
                    if !present(&reasoning_fields, field) {
                        if present(&reasoning_fields, "reasoning") && field != "reasoning" {
                            let moved = reasoning_fields
                                .iter()
                                .find(|(n, _)| *n == "reasoning")
                                .map(|(_, w)| w.clone());
                            if let Some(w) = moved {
                                reasoning_fields.retain(|(n, _)| *n != field);
                                reasoning_fields.push((field, w));
                            }
                        } else if present(&reasoning_fields, "reasoning_text")
                            && field != "reasoning_text"
                        {
                            let moved = reasoning_fields
                                .iter()
                                .find(|(n, _)| *n == "reasoning_text")
                                .map(|(_, w)| w.clone());
                            if let Some(w) = moved {
                                reasoning_fields.retain(|(n, _)| *n != field);
                                reasoning_fields.push((field, w));
                            }
                        } else if !non_empty_thinking_blocks.is_empty() {
                            reasoning_fields.retain(|(n, _)| *n != field);
                            reasoning_fields.push((field, Wire::join(thinking_pieces(), "\n")));
                        }
                    }
                }

                let tool_calls: Vec<&'a ToolCall> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::ToolCall(tool_call) => Some(tool_call),
                        _ => None,
                    })
                    .collect();

                // Replay reasoning_content on assistant turns for backends that validate
                // thinking-mode history, in three tiers (see the TS `buildParams` notes).
                let can_use_synthetic = compat.requires_reasoning_content_for_tool_calls
                    && compat.allows_synthetic_reasoning_content_for_tool_calls
                    && matches!(
                        compat.thinking_format.as_str(),
                        "openai" | "openrouter" | "zai"
                    );
                let needs_reasoning_field = compat
                    .requires_reasoning_content_for_all_assistant_turns
                    || !tool_calls.is_empty();
                let mut has_reasoning_field = !reasoning_fields.is_empty();
                if needs_reasoning_field
                    && !has_reasoning_field
                    && compat.requires_reasoning_content_for_tool_calls
                    && !compat.allows_synthetic_reasoning_content_for_tool_calls
                {
                    // Tier 1: recover from ALL thinking blocks (including empty-text ones) whose
                    // signature names a recognized wire field.
                    if let Some(first) = all_thinking_blocks.first() {
                        if first
                            .thinking_signature
                            .as_deref()
                            .is_some_and(is_reasoning_field)
                        {
                            let field = field_name(reasoning_field_default);
                            let joined = Wire::join(
                                all_thinking_blocks
                                    .iter()
                                    .map(|&b| Cow::Borrowed(b.thinking.as_str())),
                                "\n",
                            );
                            reasoning_fields.push((field, joined));
                            has_reasoning_field = true;
                        }
                    }
                    // Tier 2: genuinely no thinking at all — the configured fallback (empty string).
                    if !has_reasoning_field {
                        let fallback = compat
                            .synthetic_reasoning_content_fallback
                            .clone()
                            .unwrap_or_default();
                        reasoning_fields.push((
                            field_name(reasoning_field_default),
                            Wire::from(Value::String(fallback)),
                        ));
                        has_reasoning_field = true;
                    }
                }
                // Tier 3: providers that accept synthetic placeholders (Kimi, OpenRouter).
                if !tool_calls.is_empty() && can_use_synthetic && !has_reasoning_field {
                    reasoning_fields.push((field_name(reasoning_field_default), Wire::str(".")));
                    has_reasoning_field = true;
                }
                for (name, value) in reasoning_fields {
                    assistant_msg.insert(name, value);
                }

                if !tool_calls.is_empty() {
                    assistant_msg.insert(
                        "tool_calls",
                        Wire::Array(
                            tool_calls
                                .iter()
                                .map(|&tool_call| {
                                    obj([
                                        (
                                            "id",
                                            emit_tool_call_id(
                                                &tool_call.id,
                                                compat,
                                                Some((model, assistant)),
                                            ),
                                        ),
                                        ("type", Wire::str("function")),
                                        (
                                            "function",
                                            obj([
                                                ("name", Wire::str(&tool_call.name)),
                                                ("arguments", Wire::JsonText(&tool_call.arguments)),
                                            ]),
                                        ),
                                    ])
                                })
                                .collect(),
                        ),
                    );
                    let reasoning_details: Vec<Value> = tool_calls
                        .iter()
                        .filter_map(|tool_call| {
                            tool_call
                                .thought_signature
                                .as_ref()
                                .and_then(|signature| serde_json::from_str(signature).ok())
                        })
                        .collect();
                    if !reasoning_details.is_empty() && !has_replay_reasoning_details {
                        assistant_msg.insert("reasoning_details", Value::Array(reasoning_details));
                    }
                }
                // Some backends concatenate assistant content as a string even for tool-call
                // replay: an empty string, never null.
                let has_tool_calls = assistant_msg.contains_key("tool_calls");
                if assistant_msg.get("content").is_some_and(Wire::is_null)
                    && (has_reasoning_field || has_tool_calls || has_replay_reasoning_details)
                {
                    assistant_msg.insert("content", Wire::str(""));
                }
                let has_content = assistant_msg.get("content").is_some_and(|content| {
                    !content.is_null()
                        && match (content.string_is_empty(), content.array_len()) {
                            (Some(is_empty), _) => !is_empty,
                            (None, Some(len)) => len > 0,
                            (None, None) => true,
                        }
                });
                if !has_content
                    && has_tool_calls
                    && compat.requires_assistant_content_for_tool_calls
                {
                    assistant_msg.insert("content", Wire::str("."));
                }
                if !has_content
                    && !has_tool_calls
                    && !has_reasoning_field
                    && !has_replay_reasoning_details
                {
                    index += 1;
                    continue;
                }
                params.push(Wire::Object(assistant_msg));
            }
            Message::ToolResult(_) => {
                let mut image_blocks: Vec<Wire<'a>> = Vec::new();
                let mut j = index;
                while j < transformed.len() {
                    let Message::ToolResult(tool_msg) = &transformed[j] else {
                        break;
                    };

                    let text_result = Wire::join(
                        tool_msg.content.iter().filter_map(|block| match block {
                            UserOrToolContent::Text(text) => {
                                Some(Cow::Borrowed(text.text.as_str()))
                            }
                            _ => None,
                        }),
                        "\n",
                    );
                    let has_images = tool_msg
                        .content
                        .iter()
                        .any(|block| matches!(block, UserOrToolContent::Image(_)));
                    let omitted_images = has_images && !supports_images;
                    let has_text = text_result.string_is_empty() == Some(false);
                    let mut tool_result_msg = WireObject::new();
                    tool_result_msg.insert("role", Wire::str("tool"));
                    let content = if omitted_images {
                        if has_text {
                            let Wire::Concat(mut pieces) = text_result else {
                                unreachable!("Wire::join builds a Concat")
                            };
                            pieces.push(Cow::Borrowed("\n"));
                            pieces.push(Cow::Borrowed(NON_VISION_IMAGE_PLACEHOLDER));
                            Wire::Concat(pieces)
                        } else {
                            Wire::str(NON_VISION_IMAGE_PLACEHOLDER)
                        }
                    } else if has_text {
                        text_result
                    } else if has_images {
                        Wire::str("(see attached image)")
                    } else {
                        Wire::str("")
                    };
                    tool_result_msg.insert("content", content);
                    tool_result_msg.insert(
                        "tool_call_id",
                        emit_tool_call_id(&tool_msg.tool_call_id, compat, None),
                    );
                    if compat.requires_tool_result_name && !tool_msg.tool_name.is_empty() {
                        tool_result_msg.insert("name", Wire::str(&tool_msg.tool_name));
                    }
                    params.push(Wire::Object(tool_result_msg));

                    if has_images && supports_images {
                        for block in &tool_msg.content {
                            if let UserOrToolContent::Image(image) = block {
                                image_blocks.push(image_url_part(&image.mime_type, &image.data));
                            }
                        }
                    }
                    j += 1;
                }

                index = j;
                if image_blocks.is_empty() {
                    last_role = Some("toolResult");
                } else {
                    if compat.requires_assistant_after_tool_result {
                        params.push(processed_tool_results());
                    }
                    let mut content_items =
                        vec![text_part(Wire::str("Attached image(s) from tool result:"))];
                    content_items.extend(image_blocks);
                    params.push(obj([
                        ("role", Wire::str("user")),
                        ("content", Wire::Array(content_items)),
                    ]));
                    last_role = Some("user");
                }
                continue;
            }
        }

        last_role = Some(msg.role());
        index += 1;
    }

    last_role
}

/// The wire form of a tool call id at emission time (TS `ensureToolCallId` +
/// `normalizeMistralToolId`): Mistral ids are exactly nine alphanumerics; a cross-model composite
/// `call|item` id keeps a sanitized `call_id`; OpenAI-limit endpoints cap ids at 40 characters.
fn emit_tool_call_id<'a>(
    id: &'a str,
    compat: &ResolvedCompat,
    source: Option<(&Model, &crate::types::AssistantMessage)>,
) -> Wire<'a> {
    if compat.requires_mistral_tool_ids {
        return Wire::from(normalize_mistral_tool_id(id));
    }
    let same_model = source.is_some_and(|(model, a)| {
        a.provider == model.provider && a.api == model.api && a.model == model.id
    });
    if !same_model && id.contains('|') {
        let call_id = id.split('|').next().unwrap_or("");
        return Wire::from(
            call_id
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .take(40)
                .collect::<String>(),
        );
    }
    if compat.uses_open_ai_tool_call_id_limit && id.chars().count() > 40 {
        return Wire::from(id.chars().take(40).collect::<String>());
    }
    Wire::str(id)
}

/// Normalize a tool call ID for the target endpoint (TS `normalizeToolCallId` in `convertMessages`):
/// Mistral takes exactly nine alphanumerics; cross-model OpenAI Responses composite ids
/// (`call|item`) keep only a sanitized `call_id`; OpenAI-limit endpoints cap ids at 40 chars.
pub fn normalize_tool_call_id(id: &str, compat: &ResolvedCompat) -> String {
    if compat.requires_mistral_tool_ids {
        return normalize_mistral_tool_id(id);
    }
    if id.contains('|') {
        let call_id = id.split('|').next().unwrap_or("");
        return call_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .take(40)
            .collect();
    }
    if compat.uses_open_ai_tool_call_id_limit {
        return id.chars().take(40).collect();
    }
    id.to_string()
}

/// TS `normalizeMistralToolId`: strip non-alphanumerics, pad/truncate to exactly nine characters.
fn normalize_mistral_tool_id(id: &str) -> String {
    let mut normalized: String = id.chars().filter(char::is_ascii_alphanumeric).collect();
    if normalized.len() < 9 {
        let padding = "ABCDEFGHI";
        normalized.push_str(&padding[..9 - normalized.len()]);
    } else if normalized.len() > 9 {
        normalized.truncate(9);
    }
    normalized
}

/// Tools as the Chat Completions `tools` array, with the strictness bookkeeping the stream layer
/// needs (TS `BuiltOpenAICompletionTools`).
pub(crate) struct BuiltTools<'a> {
    pub tools: Vec<(&'a str, Wire<'a>)>,
}

/// TS `convertTools`: strict-mode adaptation per tool, `toolStrictMode` collapse, and the
/// host-specific schema flavors.
pub(crate) fn convert_tools<'a>(tools: &'a [Tool], compat: &ResolvedCompat) -> BuiltTools<'a> {
    use crate::providers::openai_completions::schema::{
        adapt_schema_for_strict, flatten_exclusive_required_root_union,
        normalize_schema_for_moonshot, sanitize_schema_for_grammar, tool_wire_schema,
    };
    struct Adapted<'t> {
        tool: &'t Tool,
        base_parameters: Value,
        parameters: Value,
        strict: bool,
    }
    let adapted: Vec<Adapted<'a>> = tools
        .iter()
        .map(|tool| {
            let wire = tool_wire_schema(&tool.parameters);
            let base_parameters = if compat.reject_root_object_union {
                flatten_exclusive_required_root_union(&wire)
            } else {
                wire
            };
            let (parameters, strict) =
                adapt_schema_for_strict(&base_parameters, compat.strict_mode_allowed());
            Adapted {
                tool,
                base_parameters,
                parameters,
                strict,
            }
        })
        .collect();
    let requested = compat.tool_strict_mode.as_deref().unwrap_or("mixed");
    let tool_strict_mode = match requested {
        "none" => "none",
        "all_strict" => {
            if adapted.iter().all(|a| a.strict) {
                "all_strict"
            } else {
                "none"
            }
        }
        _ => "mixed",
    };
    let mut wire_tools = Vec::with_capacity(adapted.len());
    for a in adapted {
        let include_strict =
            tool_strict_mode == "all_strict" || (tool_strict_mode == "mixed" && a.strict);
        let wire_parameters = if include_strict {
            a.parameters
        } else {
            a.base_parameters
        };
        let emitted = match compat.tool_schema_flavor.as_deref() {
            Some("moonshot-mfjs") => normalize_schema_for_moonshot(&wire_parameters),
            Some("grammar") => sanitize_schema_for_grammar(&wire_parameters),
            _ => wire_parameters,
        };
        let mut function = WireObject::new();
        function.insert("name", Wire::str(&a.tool.name));
        function.insert("description", Wire::str(&a.tool.description));
        function.insert("parameters", Wire::Owned(emitted));
        if include_strict {
            function.insert("strict", true);
        }
        wire_tools.push((
            a.tool.name.as_str(),
            obj([
                ("type", Wire::str("function")),
                ("function", Wire::Object(function)),
            ]),
        ));
    }
    BuiltTools { tools: wire_tools }
}

pub(crate) fn parse_chunk_usage(
    raw_usage: &Value,
    model: &Model,
    cache_write_cost: Option<f64>,
) -> Usage {
    let get_u64 = |value: &Value| value.as_u64().unwrap_or(0);
    let prompt_tokens = raw_usage.get("prompt_tokens").map_or(0, get_u64);
    let reported_cached_tokens = raw_usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .map(get_u64)
        .or_else(|| raw_usage.get("prompt_cache_hit_tokens").map(get_u64))
        .unwrap_or(0);
    let cache_write_tokens = raw_usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cache_write_tokens"))
        .map_or(0, get_u64);

    // Normalize to the provider-layer usage accounting semantics:
    // - cacheRead: hits from cache created by previous requests only
    // - cacheWrite: tokens written to cache in this request
    // Some OpenAI-compatible providers (observed on OpenRouter) report cached_tokens as (previous
    // hits + current writes). Remove cacheWrite from cacheRead in that case.
    let cache_read_tokens = if cache_write_tokens > 0 {
        reported_cached_tokens.saturating_sub(cache_write_tokens)
    } else {
        reported_cached_tokens
    };

    let input = prompt_tokens
        .saturating_sub(cache_read_tokens)
        .saturating_sub(cache_write_tokens);
    // OpenAI completion_tokens already includes reasoning_tokens.
    let output_tokens = raw_usage.get("completion_tokens").map_or(0, get_u64);
    let mut usage = Usage {
        input,
        output: output_tokens,
        cache_read: cache_read_tokens,
        cache_write: cache_write_tokens,
        total_tokens: input + output_tokens + cache_read_tokens + cache_write_tokens,
        // omp always records the counter for chat completions (0 when not reported).
        reasoning_tokens: Some(
            raw_usage
                .get("completion_tokens_details")
                .and_then(|details| details.get("reasoning_tokens"))
                .map_or(0, get_u64),
        ),
        cost: UsageCost::default(),
    };
    calculate_cost(
        model,
        &mut usage,
        cache_write_cost
            .map(|cache_write| CostOverrides {
                cache_write: Some(cache_write),
            })
            .as_ref(),
    );
    // OpenRouter reports billing truth in usage, already priced by the endpoint and service tier
    // that served the request (https://openrouter.ai/docs/api-reference/overview). Trust it over
    // the catalog-rate estimate, scaling the component breakdown to match.
    let reported_cost = if model.provider == "openrouter" {
        openrouter_reported_cost(raw_usage)
    } else {
        None
    };
    // Token counts sit far below f64's 2^53 exact-integer range; the cost apportioning math is f64 by design.
    #[allow(clippy::cast_precision_loss)]
    if let Some(reported_cost) = reported_cost {
        if usage.cost.total.as_f64() > 0.0 {
            let scale = reported_cost / usage.cost.total.as_f64();
            usage.cost.input = (usage.cost.input.as_f64() * scale).into();
            usage.cost.output = (usage.cost.output.as_f64() * scale).into();
            usage.cost.cache_read = (usage.cost.cache_read.as_f64() * scale).into();
            usage.cost.cache_write = (usage.cost.cache_write.as_f64() * scale).into();
        } else if usage.total_tokens > 0 {
            // No catalog rates to apportion by: attribute by token counts instead.
            let total_tokens = usage.total_tokens as f64;
            usage.cost.input = (reported_cost * usage.input as f64 / total_tokens).into();
            usage.cost.output = (reported_cost * usage.output as f64 / total_tokens).into();
            usage.cost.cache_read = (reported_cost * usage.cache_read as f64 / total_tokens).into();
            usage.cost.cache_write =
                (reported_cost * usage.cache_write as f64 / total_tokens).into();
        }
        usage.cost.total = reported_cost.into();
    }
    usage
}

/// The user's real spend for an `OpenRouter` request, or `None` to keep the catalog estimate.
/// `usage.cost` only carries what `OpenRouter` charged the account's credits: for BYOK requests
/// that is just `OpenRouter`'s fee, so real spend is the upstream provider's bill plus that fee. A
/// cost of 0 can mean not-billed-via-credits (e.g. `:free` endpoints) rather than free, so it keeps
/// the catalog estimate.
fn openrouter_reported_cost(raw_usage: &Value) -> Option<f64> {
    let credits = raw_usage
        .get("cost")
        .and_then(Value::as_f64)
        .filter(|cost| *cost > 0.0);
    if raw_usage.get("is_byok").and_then(Value::as_bool) == Some(true) {
        let upstream = raw_usage
            .get("cost_details")
            .and_then(|details| details.get("upstream_inference_cost"))
            .and_then(Value::as_f64);
        return upstream
            .filter(|upstream| *upstream > 0.0)
            .map(|upstream| upstream + credits.unwrap_or(0.0));
    }
    credits
}

pub(crate) fn map_stop_reason(reason: &str) -> (StopReason, Option<String>) {
    match reason {
        "stop" | "end" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "function_call" | "tool_calls" => (StopReason::ToolUse, None),
        "content_filter" => (
            StopReason::Error,
            Some("Provider finish_reason: content_filter".to_string()),
        ),
        "network_error" => (
            StopReason::Error,
            Some("Provider finish_reason: network_error".to_string()),
        ),
        other => (
            StopReason::Error,
            Some(format!("Provider finish_reason: {other}")),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelCost, ModelInput};
    use pa_types::JsNumber;
    use serde_json::json;

    fn model(provider: &str, input: f64, output: f64) -> Model {
        Model {
            id: "m".into(),
            name: "m".into(),
            api: "openai-completions".into(),
            provider: provider.into(),
            base_url: "http://localhost".into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![ModelInput::Text],
            cost: ModelCost {
                input: JsNumber::from(input),
                output: JsNumber::from(output),
                cache_read: JsNumber::from(0.0),
                cache_write: JsNumber::from(0.0),
            },
            context_window: 128_000,
            max_tokens: 8192,
            max_tokens_explicit: false,
            omit_max_output_tokens: None,
            request_model_id: None,
            featured: None,
            headers: None,
            compat: None,
        }
    }

    enum ByokBilling {
        NotByok,
        FeeOnly,
        WithUpstreamBill(f64),
    }

    // Test-only helper; adapting its signature and call sites would churn test fixtures.
    #[allow(clippy::needless_pass_by_value)]
    fn raw_usage(cost: f64, byok: ByokBilling) -> Value {
        let mut raw = json!({
            "prompt_tokens": 50_000,
            "completion_tokens": 50_000,
            "total_tokens": 100_000,
            "cost": cost,
        });
        match byok {
            ByokBilling::NotByok => {}
            ByokBilling::FeeOnly => raw["is_byok"] = json!(true),
            ByokBilling::WithUpstreamBill(upstream) => {
                raw["is_byok"] = json!(true);
                raw["cost_details"] = json!({ "upstream_inference_cost": upstream });
            }
        }
        raw
    }

    #[test]
    fn openrouter_zero_reported_cost_keeps_catalog_estimate() {
        let model = model("openrouter", 0.5, 0.5);
        // A cost of 0 can mean not-billed-via-credits (:free endpoints) rather than free, so the
        // catalog estimate stays.
        let usage = parse_chunk_usage(&raw_usage(0.0, ByokBilling::NotByok), &model, None);
        assert!((usage.cost.total.as_f64() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn openrouter_byok_without_upstream_keeps_catalog_estimate() {
        let model = model("openrouter", 0.5, 0.5);
        // BYOK credits are only OpenRouter's fee; without the upstream bill the real spend is
        // unknown, so the catalog estimate stays.
        let usage = parse_chunk_usage(&raw_usage(0.003, ByokBilling::FeeOnly), &model, None);
        assert!((usage.cost.total.as_f64() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn openrouter_byok_upstream_cost_replaces_catalog_estimate() {
        let model = model("openrouter", 0.5, 0.5);
        // Credits charged by OpenRouter plus the upstream provider's bill.
        let usage = parse_chunk_usage(
            &raw_usage(0.003, ByokBilling::WithUpstreamBill(0.2)),
            &model,
            None,
        );
        assert!((usage.cost.input.as_f64() - 0.1015).abs() < 1e-9);
        assert!((usage.cost.output.as_f64() - 0.1015).abs() < 1e-9);
        assert!((usage.cost.total.as_f64() - 0.203).abs() < 1e-9);
    }

    #[test]
    fn openrouter_reported_cost_apportioned_by_tokens_without_rates() {
        let model = model("openrouter", 0.0, 0.0);
        let raw = json!({"prompt_tokens": 40_000, "completion_tokens": 10_000, "total_tokens": 50_000, "cost": 0.15});
        let usage = parse_chunk_usage(&raw, &model, None);
        assert!((usage.cost.input.as_f64() - 0.12).abs() < 1e-9);
        assert!((usage.cost.output.as_f64() - 0.03).abs() < 1e-9);
        assert!((usage.cost.cache_read.as_f64() - 0.0).abs() < 1e-9);
        assert!((usage.cost.total.as_f64() - 0.15).abs() < 1e-9);
    }

    #[test]
    fn non_openrouter_provider_ignores_reported_cost() {
        let model = model("zai", 0.5, 0.5);
        let usage = parse_chunk_usage(&raw_usage(0.07, ByokBilling::NotByok), &model, None);
        assert!((usage.cost.total.as_f64() - 0.05).abs() < 1e-9);
    }
}
