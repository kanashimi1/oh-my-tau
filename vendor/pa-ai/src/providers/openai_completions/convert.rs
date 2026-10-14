//! `OpenAI` Completions conversion: reasoning-details signatures, messages, and tools.

use std::borrow::Cow;

use serde_json::Value;

use crate::models::{calculate_cost, CostOverrides};
use crate::providers::openai_completions::{decode_reasoning_details, ResolvedCompat};
use crate::providers::transform_messages::transform_messages_with_normalizer;
use crate::types::{
    user_block_payload, AssistantContent, Context, Message, MessageExt, Model, ModelInput,
    StopReason, TextContent, ThinkingContent, Tool, ToolCall, Usage, UsageCost, UserBlockPayload,
    UserMessageContent, UserOrToolContent,
};
use crate::utils_inner::wire_json::{obj, Wire, WireObject};

/// The transformed history a request borrows from: tool-call IDs normalized for the provider,
/// unanswered calls answered, errored turns dropped (TS `transformMessages`).
pub(crate) fn transform_for_request(model: &Model, context: &Context) -> Vec<Message> {
    transform_messages_with_normalizer(&context.messages, model, &|id, model, _| {
        Some(normalize_tool_call_id(id, model))
    })
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

/// Convert a conversation into Chat Completions `messages` params, including tool-result bridging
/// and image replay. `transformed` is [`transform_for_request`] of the context; the entries borrow
/// its strings (`sanitize_surrogates` is the identity on Rust strings, so text is borrowed as-is).
// Long by design: mirrors the provider's stream shape.
#[allow(clippy::too_many_lines)]
pub(crate) fn convert_messages<'a>(
    model: &Model,
    system_prompt: Option<&'a str>,
    transformed: &'a [Message],
    compat: &ResolvedCompat,
) -> Vec<Wire<'a>> {
    let mut params: Vec<Wire<'a>> = Vec::new();

    if let Some(system_prompt) = system_prompt {
        let use_developer_role = model.reasoning && compat.supports_developer_role;
        let role = if use_developer_role {
            "developer"
        } else {
            "system"
        };
        params.push(obj([
            ("role", Wire::str(role)),
            ("content", Wire::str(system_prompt)),
        ]));
    }

    let mut last_role: Option<&'static str> = None;
    let mut index = 0usize;
    while index < transformed.len() {
        let msg: &'a Message = &transformed[index];
        // Some providers don't allow user messages directly after tool results.
        if compat.requires_assistant_after_tool_result
            && last_role == Some("toolResult")
            && matches!(msg, Message::User(_))
        {
            params.push(processed_tool_results());
        }

        match msg {
            Message::User(user) => match &user.content {
                UserMessageContent::Text(text) => params.push(obj([
                    ("role", Wire::str("user")),
                    ("content", Wire::str(text)),
                ])),
                UserMessageContent::Blocks(blocks) => {
                    let content_blocks: Vec<Wire<'a>> = blocks
                        .iter()
                        .map(|item| match user_block_payload(item) {
                            UserBlockPayload::Text(text) => text_part(Wire::str(text)),
                            UserBlockPayload::Image { data, mime_type } => {
                                image_url_part(mime_type, data)
                            }
                            UserBlockPayload::Opaque(json) => text_part(Wire::from(json)),
                        })
                        .collect();
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
                // Some providers don't accept null content; use empty string instead.
                assistant_msg.insert(
                    "content",
                    if compat.requires_assistant_after_tool_result {
                        Wire::str("")
                    } else {
                        Wire::Null
                    },
                );

                let text_blocks: Vec<&'a TextContent> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Text(text) if !text.text.trim().is_empty() => Some(text),
                        _ => None,
                    })
                    .collect();
                // The concatenated assistant text, as one JSON string written from its pieces.
                let assistant_text_pieces = || {
                    text_blocks
                        .iter()
                        .map(|&block| Cow::Borrowed(block.text.as_str()))
                };
                let assistant_text_is_empty = text_blocks.iter().all(|block| block.text.is_empty());

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
                    assistant_msg.insert("reasoning_details", Value::Array(replay_reasoning_details));
                }

                let non_empty_thinking_blocks: Vec<&'a ThinkingContent> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Thinking(thinking)
                            if decode_reasoning_details(thinking.thinking_signature.as_deref())
                                .is_none()
                                && !thinking.thinking.trim().is_empty() =>
                        {
                            Some(thinking)
                        }
                        _ => None,
                    })
                    .collect();
                let thinking_pieces = || {
                    non_empty_thinking_blocks
                        .iter()
                        .map(|&block| Cow::Borrowed(block.thinking.as_str()))
                };

                if let Some(&first_thinking) = non_empty_thinking_blocks.first() {
                    if compat.requires_thinking_as_text {
                        let mut content = vec![text_part(Wire::join(thinking_pieces(), "\n\n"))];
                        for &block in &text_blocks {
                            content.push(text_part(Wire::str(&block.text)));
                        }
                        assistant_msg.insert("content", Wire::Array(content));
                    } else {
                        // Always send assistant text as a plain string.
                        if !assistant_text_is_empty {
                            assistant_msg
                                .insert("content", Wire::Concat(assistant_text_pieces().collect()));
                        }

                        let reasoning_field: Option<&'a str> =
                            if compat.requires_reasoning_content_on_assistant_messages {
                                Some("reasoning_content")
                            } else {
                                first_thinking.thinking_signature.as_deref()
                            };
                        match reasoning_field {
                            Some(field) => {
                                assistant_msg.insert(field, Wire::join(thinking_pieces(), "\n"));
                            }
                            None => {
                                let content = if assistant_text_is_empty {
                                    Wire::join(thinking_pieces(), "\n")
                                } else {
                                    // `{reasoning_text}\n\n{assistant_text}`
                                    let Wire::Concat(mut pieces) =
                                        Wire::join(thinking_pieces(), "\n")
                                    else {
                                        unreachable!("Wire::join builds a Concat")
                                    };
                                    pieces.push(Cow::Borrowed("\n\n"));
                                    pieces.extend(assistant_text_pieces());
                                    Wire::Concat(pieces)
                                };
                                assistant_msg.insert("content", content);
                            }
                        }
                    }
                } else if !assistant_text_is_empty {
                    assistant_msg.insert("content", Wire::Concat(assistant_text_pieces().collect()));
                }

                let tool_calls: Vec<&'a ToolCall> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::ToolCall(tool_call) => Some(tool_call),
                        _ => None,
                    })
                    .collect();
                if !tool_calls.is_empty() {
                    assistant_msg.insert(
                        "tool_calls",
                        Wire::Array(
                            tool_calls
                                .iter()
                                .map(|&tool_call| {
                                    obj([
                                        ("id", Wire::str(&tool_call.id)),
                                        ("type", Wire::str("function")),
                                        (
                                            "function",
                                            obj([
                                                ("name", Wire::str(&tool_call.name)),
                                                (
                                                    "arguments",
                                                    Wire::JsonText(&tool_call.arguments),
                                                ),
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
                if compat.requires_reasoning_content_on_assistant_messages
                    && model.reasoning
                    && !assistant_msg.contains_key("reasoning_content")
                {
                    assistant_msg.insert("reasoning_content", Wire::str(""));
                }
                if has_replay_reasoning_details
                    && assistant_msg.get("content").is_some_and(Wire::is_null)
                    && !assistant_msg.contains_key("tool_calls")
                {
                    assistant_msg.insert("content", Wire::str(""));
                }
                // Skip assistant messages that have no content and no tool calls.
                let has_content = assistant_msg.get("content").is_some_and(|content| {
                    !content.is_null()
                        && match (content.string_is_empty(), content.array_len()) {
                            (Some(is_empty), _) => !is_empty,
                            (None, Some(len)) => len > 0,
                            (None, None) => true,
                        }
                });
                if !has_content
                    && !assistant_msg.contains_key("tool_calls")
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
                            UserOrToolContent::Text(text) => Some(Cow::Borrowed(text.text.as_str())),
                            _ => None,
                        }),
                        "\n",
                    );
                    let has_images = tool_msg
                        .content
                        .iter()
                        .any(|block| matches!(block, UserOrToolContent::Image(_)));

                    let has_text = text_result.string_is_empty() == Some(false);
                    let mut tool_result_msg = WireObject::new();
                    tool_result_msg.insert("role", Wire::str("tool"));
                    tool_result_msg.insert(
                        "content",
                        if has_text {
                            text_result
                        } else if has_images {
                            Wire::str("(see attached image)")
                        } else {
                            Wire::str("")
                        },
                    );
                    tool_result_msg.insert("tool_call_id", Wire::str(&tool_msg.tool_call_id));
                    if compat.requires_tool_result_name && !tool_msg.tool_name.is_empty() {
                        tool_result_msg.insert("name", Wire::str(&tool_msg.tool_name));
                    }
                    params.push(Wire::Object(tool_result_msg));

                    if has_images
                        && model
                            .input
                            .iter()
                            .any(|mode| matches!(mode, ModelInput::Image))
                    {
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

    params
}

/// Normalize a tool call ID for providers that reject long/pipe-separated IDs.
pub fn normalize_tool_call_id(id: &str, model: &Model) -> String {
    if id.contains('|') {
        let call_id = id.split('|').next().unwrap_or("");
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
            .collect()
    } else if model.provider == "openai" {
        id.chars().take(40).collect()
    } else {
        id.to_string()
    }
}

pub(crate) fn convert_tools<'a>(tools: &'a [Tool], compat: &ResolvedCompat) -> Vec<Wire<'a>> {
    tools
        .iter()
        .map(|tool| {
            let mut function = WireObject::new();
            function.insert("name", Wire::str(&tool.name));
            function.insert("description", Wire::str(&tool.description));
            function.insert("parameters", Wire::Ref(&tool.parameters));
            // Only include strict if provider supports it. Some reject unknown fields.
            if compat.supports_strict_mode {
                function.insert("strict", false);
            }
            obj([
                ("type", Wire::str("function")),
                ("function", Wire::Object(function)),
            ])
        })
        .collect()
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
    use serde_json::json;
    use pa_types::JsNumber;

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
