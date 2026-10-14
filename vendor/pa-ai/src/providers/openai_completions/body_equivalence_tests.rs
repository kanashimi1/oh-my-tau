//! Byte-for-byte equivalence of the borrowing request body ([`build_params`] serialized through
//! [`request_body`]) with the former `serde_json::Value`-tree body, across representative
//! histories, compat flag combinations, image support and cache-control markers.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::providers::openai_completions::convert::transform_for_request;
use crate::providers::openai_completions::params::build_params;
use crate::providers::openai_completions::{
    encode_reasoning_details, get_compat, OpenAICompatCacheControl, OpenAICompletionsOptions,
    ResolvedCompat,
};
use crate::providers::request_body_fixtures::{contexts, model};
use crate::types::{CacheRetention, Context, Model, OnPayloadHook, StreamOptions};
use crate::utils_inner::wire_json::request_body;

mod legacy {
    //! The `Value`-tree request assembly as of HEAD before the borrowing body (verbatim), kept as
    //! the byte-for-byte reference.
    #![allow(clippy::all, clippy::pedantic, dead_code, unused_imports)]

    use serde_json::{json, Map, Value};

    use crate::models::supports_thinking;
    use crate::providers::openai_completions::has_tool_history;
    use crate::providers::openai_completions::{
        decode_reasoning_details, OpenAICompatCacheControl, OpenAICompletionsOptions,
        ResolvedCompat,
    };
    use crate::providers::transform_messages::transform_messages_with_normalizer;
    use crate::types::{
        AssistantContent, CacheRetention, Context, MessageExt, Model, ModelExt, ModelInput,
        ModelThinkingLevel, TextContent, ThinkingContent, Tool, ToolCall, UserMessageContent,
        UserOrToolContent,
    };
    use crate::utils_inner::sanitize_unicode::sanitize_surrogates;

    /// Convert a conversation into Chat Completions `messages` params, including tool-result bridging
    /// and image replay.
    // Long by design: mirrors the provider's stream shape.
    #[allow(clippy::too_many_lines)]
    pub fn convert_messages(model: &Model, context: &Context, compat: &ResolvedCompat) -> Vec<Value> {
        use crate::types::Message;
        let mut params: Vec<Value> = Vec::new();

        let transformed =
            transform_messages_with_normalizer(&context.messages, model, &|id, model, _| {
                Some(normalize_tool_call_id(id, model))
            });

        if let Some(system_prompt) = &context.system_prompt {
            let use_developer_role = model.reasoning && compat.supports_developer_role;
            let role = if use_developer_role {
                "developer"
            } else {
                "system"
            };
            params.push(json!({
                "role": role,
                "content": sanitize_surrogates(system_prompt),
            }));
        }

        let mut last_role: Option<&'static str> = None;
        let mut index = 0usize;
        while index < transformed.len() {
            let msg = &transformed[index];
            // Some providers don't allow user messages directly after tool results.
            if compat.requires_assistant_after_tool_result
                && last_role == Some("toolResult")
                && matches!(msg, Message::User(_))
            {
                params.push(json!({
                    "role": "assistant",
                    "content": "I have processed the tool results.",
                }));
            }

            match msg {
                Message::User(user) => match &user.content {
                    UserMessageContent::Text(text) => params.push(json!({
                        "role": "user",
                        "content": sanitize_surrogates(text),
                    })),
                    UserMessageContent::Blocks(blocks) => {
                        let content_blocks: Vec<Value> = blocks
                            .iter()
                            .map(|item| match crate::types::user_block_payload(item) {
                                crate::types::UserBlockPayload::Text(text) => json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(text),
                                }),
                                crate::types::UserBlockPayload::Image { data, mime_type } => json!({
                                    "type": "image_url",
                                    "image_url": { "url": format!("data:{mime_type};base64,{data}") },
                                }),
                                crate::types::UserBlockPayload::Opaque(json) => json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&json),
                                }),
                            })
                            .collect();
                        if content_blocks.is_empty() {
                            index += 1;
                            continue;
                        }
                        params.push(json!({
                            "role": "user",
                            "content": content_blocks,
                        }));
                    }
                },
                Message::Assistant(assistant) => {
                    let mut assistant_msg = Map::new();
                    assistant_msg.insert("role".into(), json!("assistant"));
                    // Some providers don't accept null content; use empty string instead.
                    assistant_msg.insert(
                        "content".into(),
                        if compat.requires_assistant_after_tool_result {
                            json!("")
                        } else {
                            Value::Null
                        },
                    );

                    let text_blocks: Vec<&TextContent> = assistant
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            AssistantContent::Text(text) if !text.text.trim().is_empty() => Some(text),
                            _ => None,
                        })
                        .collect();
                    let assistant_text = text_blocks
                        .iter()
                        .map(|block| sanitize_surrogates(&block.text))
                        .collect::<String>();

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
                    if !replay_reasoning_details.is_empty() {
                        assistant_msg
                            .insert("reasoning_details".into(), json!(replay_reasoning_details));
                    }

                    let non_empty_thinking_blocks: Vec<&ThinkingContent> = assistant
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

                    if !non_empty_thinking_blocks.is_empty() {
                        if compat.requires_thinking_as_text {
                            let thinking_text = non_empty_thinking_blocks
                                .iter()
                                .map(|block| sanitize_surrogates(&block.thinking))
                                .collect::<Vec<_>>()
                                .join("\n\n");
                            assistant_msg.insert(
                                "content".into(),
                                json!([{
                                    "type": "text",
                                    "text": thinking_text,
                                }]),
                            );
                            for block in &text_blocks {
                                assistant_msg["content"]
                                    .as_array_mut()
                                    .expect("content is an array")
                                    .push(json!({
                                        "type": "text",
                                        "text": sanitize_surrogates(&block.text),
                                    }));
                            }
                        } else {
                            // Always send assistant text as a plain string.
                            if !assistant_text.is_empty() {
                                assistant_msg.insert("content".into(), json!(assistant_text));
                            }

                            let reasoning_text = non_empty_thinking_blocks
                                .iter()
                                .map(|block| sanitize_surrogates(&block.thinking))
                                .collect::<Vec<_>>()
                                .join("\n");
                            let reasoning_field =
                                if compat.requires_reasoning_content_on_assistant_messages {
                                    Some("reasoning_content")
                                } else {
                                    non_empty_thinking_blocks[0].thinking_signature.as_deref()
                                };
                            match reasoning_field {
                                Some(field) => {
                                    assistant_msg.insert(field.to_string(), json!(reasoning_text));
                                }
                                None => {
                                    assistant_msg.insert(
                                        "content".into(),
                                        if assistant_text.is_empty() {
                                            json!(reasoning_text)
                                        } else {
                                            json!(format!("{reasoning_text}\n\n{assistant_text}"))
                                        },
                                    );
                                }
                            }
                        }
                    } else if !assistant_text.is_empty() {
                        assistant_msg.insert("content".into(), json!(assistant_text));
                    }

                    let tool_calls: Vec<&ToolCall> = assistant
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            AssistantContent::ToolCall(tool_call) => Some(tool_call),
                            _ => None,
                        })
                        .collect();
                    if !tool_calls.is_empty() {
                        assistant_msg.insert(
                            "tool_calls".into(),
                            json!(tool_calls
                                .iter()
                                .map(|tool_call| json!({
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.name,
                                        "arguments": serde_json::Value::Object(tool_call.arguments.clone()).to_string(),
                                    },
                                }))
                                .collect::<Vec<_>>()),
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
                        if !reasoning_details.is_empty() && replay_reasoning_details.is_empty() {
                            assistant_msg.insert("reasoning_details".into(), json!(reasoning_details));
                        }
                    }
                    if compat.requires_reasoning_content_on_assistant_messages
                        && model.reasoning
                        && !assistant_msg.contains_key("reasoning_content")
                    {
                        assistant_msg.insert("reasoning_content".into(), json!(""));
                    }
                    if !replay_reasoning_details.is_empty()
                        && assistant_msg.get("content") == Some(&Value::Null)
                        && !assistant_msg.contains_key("tool_calls")
                    {
                        assistant_msg.insert("content".into(), json!(""));
                    }
                    // Skip assistant messages that have no content and no tool calls.
                    let content_value = assistant_msg.get("content");
                    let has_content = match content_value {
                        Some(Value::Null) | None => false,
                        Some(Value::String(text)) => !text.is_empty(),
                        Some(Value::Array(array)) => !array.is_empty(),
                        Some(_) => true,
                    };
                    if !has_content
                        && !assistant_msg.contains_key("tool_calls")
                        && replay_reasoning_details.is_empty()
                    {
                        index += 1;
                        continue;
                    }
                    params.push(Value::Object(assistant_msg));
                }
                Message::ToolResult(_) => {
                    let mut image_blocks: Vec<Value> = Vec::new();
                    let mut j = index;
                    while j < transformed.len() {
                        let Message::ToolResult(tool_msg) = &transformed[j] else {
                            break;
                        };

                        let text_result = tool_msg
                            .content
                            .iter()
                            .filter_map(|block| match block {
                                UserOrToolContent::Text(text) => Some(text.text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        let has_images = tool_msg
                            .content
                            .iter()
                            .any(|block| matches!(block, UserOrToolContent::Image(_)));

                        let has_text = !text_result.is_empty();
                        let mut tool_result_msg = Map::new();
                        tool_result_msg.insert("role".into(), json!("tool"));
                        tool_result_msg.insert(
                            "content".into(),
                            json!(sanitize_surrogates(if has_text {
                                text_result.as_str()
                            } else if has_images {
                                "(see attached image)"
                            } else {
                                ""
                            })),
                        );
                        tool_result_msg.insert("tool_call_id".into(), json!(tool_msg.tool_call_id));
                        if compat.requires_tool_result_name && !tool_msg.tool_name.is_empty() {
                            tool_result_msg.insert("name".into(), json!(tool_msg.tool_name));
                        }
                        params.push(Value::Object(tool_result_msg));

                        if has_images
                            && model
                                .input
                                .iter()
                                .any(|mode| matches!(mode, ModelInput::Image))
                        {
                            for block in &tool_msg.content {
                                if let UserOrToolContent::Image(image) = block {
                                    image_blocks.push(json!({
                                        "type": "image_url",
                                        "image_url": { "url": format!("data:{};base64,{}", image.mime_type, image.data) },
                                    }));
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
                            params.push(json!({
                                "role": "assistant",
                                "content": "I have processed the tool results.",
                            }));
                        }
                        let mut content_items = vec![json!({
                            "type": "text",
                            "text": "Attached image(s) from tool result:",
                        })];
                        content_items.extend(image_blocks);
                        params.push(json!({
                            "role": "user",
                            "content": content_items,
                        }));
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

    pub fn convert_tools(tools: &[Tool], compat: &ResolvedCompat) -> Vec<Value> {
        tools
            .iter()
            .map(|tool| {
                let mut function = Map::new();
                function.insert("name".into(), json!(tool.name));
                function.insert("description".into(), json!(tool.description));
                function.insert("parameters".into(), tool.parameters.clone());
                // Only include strict if provider supports it. Some reject unknown fields.
                if compat.supports_strict_mode {
                    function.insert("strict".into(), json!(false));
                }
                json!({
                    "type": "function",
                    "function": Value::Object(function),
                })
            })
            .collect()
    }

    // Long by design: mirrors the provider's stream shape.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn build_params(
        model: &Model,
        context: &Context,
        options: Option<&OpenAICompletionsOptions>,
        compat: &ResolvedCompat,
        cache_retention: CacheRetention,
        cache_control: Option<&OpenAICompatCacheControl>,
    ) -> Value {
        let options = options.cloned().unwrap_or_default();
        let messages = convert_messages(model, context, compat);
        let mut params = Map::new();
        params.insert("model".into(), json!(model.id));
        // Move the converted messages into the params tree: `json!(messages)`
        // would deep-clone the whole per-turn context (the largest transient
        // allocation of the turn), which mimalloc then retains committed.
        params.insert("messages".into(), Value::Array(messages));
        params.insert("stream".into(), json!(true));

        let prompt_cache_key = if (model.base_url.contains("api.openai.com")
            && cache_retention != CacheRetention::None)
            || (cache_retention == CacheRetention::Long && compat.supports_long_cache_retention)
        {
            options.base.session_id.clone().map(Value::String)
        } else {
            None
        };
        if let Some(key) = prompt_cache_key {
            params.insert("prompt_cache_key".into(), key);
        }
        if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
            params.insert("prompt_cache_retention".into(), json!("24h"));
        }

        if compat.supports_usage_in_streaming {
            params.insert("stream_options".into(), json!({ "include_usage": true }));
        }

        if compat.supports_store {
            params.insert("store".into(), json!(false));
        }

        if let Some(max_tokens) = options.base.max_tokens {
            if compat.max_tokens_field == crate::types::MaxTokensField::MaxTokens {
                params.insert("max_tokens".into(), json!(max_tokens));
            } else {
                params.insert("max_completion_tokens".into(), json!(max_tokens));
            }
        }

        if let Some(temperature) = options.base.temperature {
            params.insert("temperature".into(), json!(temperature));
        }

        let mut tools: Option<Vec<Value>> = None;
        if let Some(context_tools) = &context.tools {
            if !context_tools.is_empty() {
                tools = Some(convert_tools(context_tools, compat));
                if compat.zai_tool_stream {
                    params.insert("tool_stream".into(), json!(true));
                }
            }
        }
        if tools.is_none() && has_tool_history(&context.messages) {
            // Anthropic (via LiteLLM/proxy) requires the tools param when the conversation has
            // tool_calls/tool_results.
            tools = Some(Vec::new());
        }
        if let Some(tools) = tools {
            params.insert("tools".into(), Value::Array(tools));
        }

        if let Some(cache_control) = cache_control {
            apply_anthropic_cache_control(&mut params, cache_control);
        }

        if let Some(tool_choice) = &options.tool_choice {
            params.insert("tool_choice".into(), tool_choice.to_json());
        }

        if supports_thinking(model) {
            match compat.thinking_format {
                crate::types::ThinkingFormat::Zai | crate::types::ThinkingFormat::Qwen => {
                    params.insert(
                        "enable_thinking".into(),
                        json!(options.reasoning_effort.is_some()),
                    );
                }
                crate::types::ThinkingFormat::QwenChatTemplate => {
                    params.insert(
                        "chat_template_kwargs".into(),
                        json!({
                            "enable_thinking": options.reasoning_effort.is_some(),
                            "preserve_thinking": true,
                        }),
                    );
                }
                crate::types::ThinkingFormat::Deepseek => {
                    params.insert(
                        "thinking".into(),
                        json!({
                            "type": if options.reasoning_effort.is_some() { "enabled" } else { "disabled" },
                        }),
                    );
                    if let Some(effort) = options.reasoning_effort {
                        let mapped = model
                            .thinking_level_map_value(effort)
                            .flatten()
                            .cloned()
                            .unwrap_or_else(|| effort.wire_name().to_string());
                        params.insert("reasoning_effort".into(), json!(mapped));
                    }
                }
                crate::types::ThinkingFormat::Openrouter => {
                    // OpenRouter distinguishes an omitted reasoning preference (use the model default),
                    // an explicit toggle, and an explicit effort selection.
                    let declared_effort = options
                        .reasoning_effort
                        .filter(|_| compat.supports_reasoning_effort);
                    if let Some(effort) = declared_effort {
                        let mapped = model
                            .thinking_level_map_value(effort)
                            .flatten()
                            .cloned()
                            .unwrap_or_else(|| effort.wire_name().to_string());
                        params.insert("reasoning".into(), json!({ "effort": mapped }));
                    } else if options.reasoning_enabled == Some(true) {
                        params.insert("reasoning".into(), json!({ "enabled": true }));
                    } else if options.reasoning_enabled == Some(false) {
                        let off = model
                            .thinking_level_map
                            .as_ref()
                            .and_then(|map| map.get(&ModelThinkingLevel::Off));
                        // TS `thinkingLevelMap?.off !== null`: only an explicit null suppresses the
                        // disable; a missing key or map still disables reasoning.
                        if !off.is_some_and(std::option::Option::is_none) {
                            if compat.supports_reasoning_effort {
                                let off_value = off
                                    .and_then(|value| value.as_deref())
                                    .unwrap_or("none")
                                    .to_string();
                                params.insert("reasoning".into(), json!({ "effort": off_value }));
                            } else {
                                params.insert("reasoning".into(), json!({ "enabled": false }));
                            }
                        }
                    }
                }
                crate::types::ThinkingFormat::Openai => {
                    if let Some(effort) = options.reasoning_effort {
                        if compat.supports_reasoning_effort {
                            let mapped = model
                                .thinking_level_map_value(effort)
                                .flatten()
                                .cloned()
                                .unwrap_or_else(|| effort.wire_name().to_string());
                            params.insert("reasoning_effort".into(), json!(mapped));
                        }
                    } else if options.reasoning_enabled == Some(false)
                        && compat.supports_reasoning_effort
                    {
                        let off = model
                            .thinking_level_map
                            .as_ref()
                            .and_then(|map| map.get(&ModelThinkingLevel::Off));
                        // TS `thinkingLevelMap?.off !== null`: only an explicit null suppresses the
                        // disable; a missing key or map still sends the off value.
                        if !off.is_some_and(std::option::Option::is_none) {
                            let off_value = off
                                .and_then(|value| value.as_deref())
                                .unwrap_or("none")
                                .to_string();
                            params.insert("reasoning_effort".into(), json!(off_value));
                        }
                    }
                }
            }
        }

        // OpenAI and OpenRouter accept a top-level service_tier
        // (https://openrouter.ai/docs/guides/features/service-tiers). Prime Inference tolerates but
        // ignores the field (probed 2026-09-01), so it is not forwarded; other OpenAI-compatible
        // gateways may reject unknown fields.
        if let Some(service_tier) = options.base.service_tier {
            if model.provider == "openai" || model.provider == "openrouter" {
                params.insert(
                    "service_tier".into(),
                    serde_json::to_value(service_tier).unwrap_or(Value::Null),
                );
            }
        }

        if model.base_url.contains("openrouter.ai") {
            if let Some(crate::types::CompatKind::OpenAiCompletions(compat)) = model.compat_kind() {
                if let Some(routing) = &compat.as_ref().open_router_routing {
                    params.insert(
                        "provider".into(),
                        serde_json::to_value(routing).unwrap_or(Value::Null),
                    );
                }
            }
        }

        if model.base_url.contains("ai-gateway.vercel.sh") {
            if let Some(crate::types::CompatKind::OpenAiCompletions(compat)) = model.compat_kind() {
                if let Some(routing) = &compat.as_ref().vercel_gateway_routing {
                    let mut gateway_options = Map::new();
                    if let Some(only) = &routing.only {
                        gateway_options.insert("only".into(), json!(only));
                    }
                    if let Some(order) = &routing.order {
                        gateway_options.insert("order".into(), json!(order));
                    }
                    if !gateway_options.is_empty() {
                        params.insert(
                            "providerOptions".into(),
                            json!({ "gateway": Value::Object(gateway_options) }),
                        );
                    }
                }
            }
        }

        Value::Object(params)
    }

    fn apply_anthropic_cache_control(
        params: &mut Map<String, Value>,
        cache_control: &OpenAICompatCacheControl,
    ) {
        // Last tool.
        if let Some(tools) = params
            .get_mut("tools")
            .and_then(|value| value.as_array_mut())
        {
            if let Some(last_tool) = tools.last_mut() {
                last_tool
                    .as_object_mut()
                    .expect("tools entries are objects")
                    .insert("cache_control".into(), cache_control.to_json());
            }
        }
        let Some(messages) = params
            .get_mut("messages")
            .and_then(|value| value.as_array_mut())
        else {
            return;
        };
        // System prompt.
        for message in messages.iter_mut() {
            let role = message.get("role").and_then(|value| value.as_str());
            if role == Some("system") || role == Some("developer") {
                add_cache_control_to_message(message, cache_control);
                break;
            }
        }
        // Last conversation message (user/assistant/tool), from the end.
        for message in messages.iter_mut().rev() {
            let role = message
                .get("role")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            if (role == "user" || role == "assistant" || role == "tool")
                && add_cache_control_to_message(message, cache_control)
            {
                break;
            }
        }
    }

    fn add_cache_control_to_message(
        message: &mut Value,
        cache_control: &OpenAICompatCacheControl,
    ) -> bool {
        let cache_json = cache_control.to_json();
        match message.get_mut("content") {
            Some(Value::String(content)) => {
                if content.is_empty() {
                    return false;
                }
                let text = content.clone();
                message
                    .as_object_mut()
                    .expect("messages are objects")
                    .insert(
                        "content".into(),
                        json!([{
                            "type": "text",
                            "text": text,
                            "cache_control": cache_json,
                        }]),
                    );
                true
            }
            Some(Value::Array(content)) => {
                for part in content.iter_mut().rev() {
                    if part.get("type").and_then(|value| value.as_str()) == Some("text") {
                        part.as_object_mut()
                            .expect("text parts are objects")
                            .insert("cache_control".into(), cache_json);
                        return true;
                    }
                }
                false
            }
            _ => false,
        }
    }

}

fn reasoning_signature() -> String {
    encode_reasoning_details(&[
        json!({ "type": "reasoning.text", "text": "r1", "index": 0 }),
        json!({ "type": "reasoning.encrypted", "data": "e==", "index": 1 }),
    ])
}

fn test_model(reasoning: bool, images: bool) -> Model {
    model("openai-completions", "test", "m", reasoning, images)
}

/// The compat flags that steer message conversion, one bit each.
fn compat_with(model: &Model, bits: u8) -> ResolvedCompat {
    let mut compat = get_compat(model);
    compat.requires_assistant_after_tool_result = bits & 1 != 0;
    compat.requires_thinking_as_text = bits & 2 != 0;
    compat.requires_reasoning_content_on_assistant_messages = bits & 4 != 0;
    compat.requires_tool_result_name = bits & 8 != 0;
    compat.supports_strict_mode = bits & 16 != 0;
    compat.supports_developer_role = bits & 32 != 0;
    compat
}

fn cache_controls() -> [Option<OpenAICompatCacheControl>; 3] {
    [
        None,
        Some(OpenAICompatCacheControl { ttl: None }),
        Some(OpenAICompatCacheControl { ttl: Some("1h") }),
    ]
}

fn options_variants() -> [Option<OpenAICompletionsOptions>; 2] {
    [
        None,
        Some(OpenAICompletionsOptions::from_base(StreamOptions {
            temperature: Some(0.25),
            max_tokens: Some(1000),
            session_id: Some("sess-1".into()),
            ..Default::default()
        })),
    ]
}

fn assert_same_body(
    model: &Model,
    context: &Context,
    options: Option<&OpenAICompletionsOptions>,
    compat: &ResolvedCompat,
    cache_control: Option<&OpenAICompatCacheControl>,
) {
    let legacy = legacy::build_params(
        model,
        context,
        options,
        compat,
        CacheRetention::Short,
        cache_control,
    );
    let transformed = transform_for_request(model, context);
    let wire = build_params(
        model,
        context,
        &transformed,
        options,
        compat,
        CacheRetention::Short,
        cache_control,
    );
    let body = request_body(&wire, None, model);
    assert_eq!(
        body,
        legacy.to_string(),
        "body bytes differ (compat {compat:?}, cache {cache_control:?})"
    );
    assert_eq!(body.capacity(), body.len(), "the body is allocated at its exact size");
    assert_eq!(wire.to_value(), legacy, "the hook-facing tree differs");
}

#[test]
fn every_compat_combination_matches_the_value_tree_body() {
    let signature = reasoning_signature();
    for reasoning in [false, true] {
        for images in [false, true] {
            let model = test_model(reasoning, images);
            let all = contexts(&model, &signature);
            // Full history with system prompt + tools, and the tool-result tail without either.
            let picked = [&all[5], &all[27]];
            for bits in 0..64u8 {
                let compat = compat_with(&model, bits);
                for context in picked {
                    for cache_control in &cache_controls() {
                        assert_same_body(&model, context, None, &compat, cache_control.as_ref());
                    }
                }
            }
        }
    }
}

#[test]
fn every_history_shape_matches_the_value_tree_body() {
    let signature = reasoning_signature();
    for images in [false, true] {
        let model = test_model(true, images);
        for context in &contexts(&model, &signature) {
            for bits in [0u8, 63, 0b01_0101, 0b10_1010] {
                let compat = compat_with(&model, bits);
                for cache_control in &cache_controls() {
                    for options in &options_variants() {
                        assert_same_body(
                            &model,
                            context,
                            options.as_ref(),
                            &compat,
                            cache_control.as_ref(),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn payload_hook_receives_and_replaces_the_same_tree() {
    let model = test_model(true, true);
    let signature = reasoning_signature();
    let context = &contexts(&model, &signature)[5];
    let compat = compat_with(&model, 0);
    let replacing: OnPayloadHook = Arc::new(|mut payload: Value, _model: &Model| {
        payload["injected"] = json!({ "n": 1 });
        Some(payload)
    });
    let keeping: OnPayloadHook = Arc::new(|_payload: Value, _model: &Model| None);
    for hook in [replacing, keeping] {
        let mut legacy = legacy::build_params(
            &model,
            context,
            None,
            &compat,
            CacheRetention::Short,
            None,
        );
        if let Some(next) = hook(legacy.clone(), &model) {
            legacy = next;
        }
        let transformed = transform_for_request(&model, context);
        let wire = build_params(
            &model,
            context,
            &transformed,
            None,
            &compat,
            CacheRetention::Short,
            None,
        );
        assert_eq!(request_body(&wire, Some(&hook), &model), legacy.to_string());
    }
}
