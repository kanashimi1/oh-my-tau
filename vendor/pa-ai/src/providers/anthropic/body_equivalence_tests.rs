//! Byte-for-byte equivalence of the borrowing request body ([`build_params`] serialized through
//! [`request_body`]) with the former `serde_json::Value`-tree body, across representative
//! histories, OAuth naming, thinking options, eager tool streaming and cache-control markers.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::providers::anthropic::convert::{convert_tools, transform_for_request};
use crate::providers::anthropic::params::build_params;
use crate::providers::anthropic::{
    AnthropicEffort, AnthropicOptions, AnthropicThinkingDisplay, AnthropicToolChoice,
    CacheControl,
};
use crate::providers::request_body_fixtures::{contexts, model, tools};
use crate::types::{Context, Model, OnPayloadHook, StreamOptions};
use crate::utils_inner::wire_json::{request_body, to_json_string, Wire};

mod legacy {
    //! The `Value`-tree request assembly as of HEAD before the borrowing body (verbatim), kept as
    //! the byte-for-byte reference.
    #![allow(clippy::all, clippy::pedantic, dead_code, unused_imports)]

    use serde_json::{json, Map, Value};

    use crate::providers::anthropic::{
        get_anthropic_compat, is_always_on_adaptive_thinking_model, supports_adaptive_thinking,
        to_claude_code_name, AnthropicOptions, AnthropicThinkingDisplay, CacheControl,
    };
    use crate::types::{
        AssistantContent, Context, Message, Model, Tool, UserMessageContent, UserOrToolContent,
    };
    use crate::utils_inner::sanitize_unicode::sanitize_surrogates;

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

    pub(crate) fn convert_content_blocks(content: &[UserOrToolContent]) -> Value {
        let has_images = content
            .iter()
            .any(|block| matches!(block, UserOrToolContent::Image(_)));
        if !has_images {
            let text = content
                .iter()
                .map(|block| match crate::types::user_block_payload(block) {
                    crate::types::UserBlockPayload::Text(text) => text.to_string(),
                    crate::types::UserBlockPayload::Image { data, .. } => data.to_string(),
                    crate::types::UserBlockPayload::Opaque(json) => json,
                })
                .collect::<Vec<_>>()
                .join("\n");
            return json!(sanitize_surrogates(&text));
        }
        let mut blocks: Vec<Value> = content
            .iter()
            .map(|block| match crate::types::user_block_payload(block) {
                crate::types::UserBlockPayload::Text(text) => json!({
                    "type": "text",
                    "text": sanitize_surrogates(text),
                }),
                crate::types::UserBlockPayload::Image { data, mime_type } => json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": data,
                    },
                }),
                crate::types::UserBlockPayload::Opaque(json) => json!({
                    "type": "text",
                    "text": sanitize_surrogates(&json),
                }),
            })
            .collect();
        let has_text = blocks
            .iter()
            .any(|block| block.get("type").and_then(|value| value.as_str()) == Some("text"));
        if !has_text {
            blocks.insert(0, json!({ "type": "text", "text": "(see attached image)" }));
        }
        json!(blocks)
    }

    // Long by design: mirrors the provider's stream shape.
    #[allow(clippy::too_many_lines)]
    pub fn convert_messages(
        context: &Context,
        model: &Model,
        is_oauth_token: bool,
        cache_control: Option<&CacheControl>,
    ) -> Vec<Value> {
        let mut params: Vec<Value> = Vec::new();
        let transformed = crate::providers::transform_messages::transform_messages_with_normalizer(
            &context.messages,
            model,
            &|id, _model, _source| Some(normalize_tool_call_id(id)),
        );

        let mut index = 0usize;
        while index < transformed.len() {
            let msg = &transformed[index];
            match msg {
                Message::User(user) => match &user.content {
                    UserMessageContent::Text(text) => {
                        if !text.trim().is_empty() {
                            params.push(json!({
                                "role": "user",
                                "content": sanitize_surrogates(text),
                            }));
                        }
                    }
                    UserMessageContent::Blocks(blocks) => {
                        let converted: Vec<Value> = blocks
                            .iter()
                            .map(|item| match crate::types::user_block_payload(item) {
                                crate::types::UserBlockPayload::Text(text) => json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(text),
                                }),
                                crate::types::UserBlockPayload::Image { data, mime_type } => json!({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime_type,
                                        "data": data,
                                    },
                                }),
                                crate::types::UserBlockPayload::Opaque(json) => json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&json),
                                }),
                            })
                            .collect();
                        let filtered: Vec<Value> = converted
                            .into_iter()
                            .filter(|block| {
                                if block.get("type").and_then(|value| value.as_str()) == Some("text") {
                                    block["text"]
                                        .as_str()
                                        .is_some_and(|text| !text.trim().is_empty())
                                } else {
                                    true
                                }
                            })
                            .collect();
                        if filtered.is_empty() {
                            index += 1;
                            continue;
                        }
                        params.push(json!({
                            "role": "user",
                            "content": filtered,
                        }));
                    }
                },
                Message::Assistant(assistant) => {
                    let mut blocks: Vec<Value> = Vec::new();
                    for block in &assistant.content {
                        match block {
                            AssistantContent::Text(text) => {
                                if text.text.trim().is_empty() {
                                    continue;
                                }
                                blocks.push(json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&text.text),
                                }));
                            }
                            AssistantContent::Thinking(thinking) => {
                                if thinking.redacted.unwrap_or(false) {
                                    blocks.push(json!({
                                        "type": "redacted_thinking",
                                        "data": thinking.thinking_signature.clone().unwrap_or_default(),
                                    }));
                                    continue;
                                }
                                if thinking.thinking.trim().is_empty() {
                                    continue;
                                }
                                let signature = thinking.thinking_signature.as_deref().unwrap_or("");
                                if signature.trim().is_empty() {
                                    blocks.push(json!({
                                        "type": "text",
                                        "text": sanitize_surrogates(&thinking.thinking),
                                    }));
                                } else {
                                    blocks.push(json!({
                                        "type": "thinking",
                                        "thinking": sanitize_surrogates(&thinking.thinking),
                                        "signature": signature,
                                    }));
                                }
                            }
                            AssistantContent::ToolCall(tool_call) => {
                                blocks.push(json!({
                                    "type": "tool_use",
                                    "id": tool_call.id,
                                    "name": if is_oauth_token {
                                        to_claude_code_name(&tool_call.name)
                                    } else {
                                        tool_call.name.clone()
                                    },
                                    "input": Value::Object(tool_call.arguments.clone()),
                                }));
                            }
                        }
                    }
                    if blocks.is_empty() {
                        index += 1;
                        continue;
                    }
                    params.push(json!({
                        "role": "assistant",
                        "content": blocks,
                    }));
                }
                Message::ToolResult(_tool_result) => {
                    // Collect all consecutive toolResult messages into one user turn.
                    let mut tool_results: Vec<Value> = Vec::new();
                    let mut j = index;
                    while j < transformed.len() {
                        let Message::ToolResult(result) = &transformed[j] else {
                            break;
                        };
                        tool_results.push(json!({
                            "type": "tool_result",
                            "tool_use_id": result.tool_call_id,
                            "content": convert_content_blocks(&result.content),
                            "is_error": result.is_error,
                        }));
                        j += 1;
                    }
                    index = j;
                    params.push(json!({
                        "role": "user",
                        "content": tool_results,
                    }));
                    continue;
                }
            }
            index += 1;
        }

        // Add cache_control to the last user message to cache conversation history.
        if let Some(cache_control) = cache_control {
            if let Some(last) = params.last_mut() {
                if last.get("role").and_then(|value| value.as_str()) == Some("user") {
                    let content_value = &mut last["content"];
                    if let Value::String(text) = content_value {
                        let text = text.clone();
                        *last = json!({
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": text,
                                "cache_control": cache_control.to_json(),
                            }],
                        });
                    } else if let Value::Array(blocks) = content_value {
                        if let Some(last_block) = blocks.last_mut() {
                            let block_type = last_block
                                .get("type")
                                .and_then(|value| value.as_str())
                                .unwrap_or("");
                            if block_type == "text"
                                || block_type == "image"
                                || block_type == "tool_result"
                            {
                                last_block
                                    .as_object_mut()
                                    .expect("content blocks are objects")
                                    .insert("cache_control".into(), cache_control.to_json());
                            }
                        }
                    }
                }
            }
        }

        params
    }

    pub fn convert_tools(
        tools: &[Tool],
        is_oauth_token: bool,
        supports_eager_tool_input_streaming: bool,
        cache_control: Option<&CacheControl>,
    ) -> Vec<Value> {
        tools
            .iter()
            .enumerate()
            .map(|(index, tool)| {
                let mut entry = Map::new();
                entry.insert(
                    "name".into(),
                    json!(if is_oauth_token {
                        to_claude_code_name(&tool.name)
                    } else {
                        tool.name.clone()
                    }),
                );
                entry.insert("description".into(), json!(tool.description));
                if supports_eager_tool_input_streaming {
                    entry.insert("eager_input_streaming".into(), json!(true));
                }
                entry.insert(
                    "input_schema".into(),
                    json!({
                        "type": "object",
                        "properties": tool.parameters.get("properties").cloned().unwrap_or_else(|| json!({})),
                        "required": tool.parameters.get("required").cloned().unwrap_or_else(|| json!([])),
                    }),
                );
                if let Some(cache_control) = cache_control {
                    if index == tools.len() - 1 {
                        entry.insert("cache_control".into(), cache_control.to_json());
                    }
                }
                Value::Object(entry)
            })
            .collect()
    }

    // Long by design: mirrors the provider's stream shape.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn build_params(
        model: &Model,
        context: &Context,
        is_oauth_token: bool,
        options: Option<&AnthropicOptions>,
        cache_control: Option<&CacheControl>,
    ) -> Value {
        let base = options
            .map(|options| options.base.clone())
            .unwrap_or_default();
        let mut params = Map::new();
        params.insert("model".into(), json!(model.id));
        params.insert(
            "messages".into(),
            json!(convert_messages(
                context,
                model,
                is_oauth_token,
                cache_control
            )),
        );
        params.insert(
            "max_tokens".into(),
            json!(base.max_tokens.unwrap_or(model.max_tokens / 3)),
        );
        params.insert("stream".into(), json!(true));

        // For OAuth tokens, we MUST include Claude Code identity.
        if is_oauth_token {
            let mut system = vec![json!({
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            })];
            if let Some(cache_control) = cache_control {
                system[0]
                    .as_object_mut()
                    .expect("system entry is an object")
                    .insert("cache_control".into(), cache_control.to_json());
            }
            if let Some(system_prompt) = &context.system_prompt {
                let mut entry = json!({
                    "type": "text",
                    "text": sanitize_surrogates(system_prompt),
                });
                if let Some(cache_control) = cache_control {
                    entry
                        .as_object_mut()
                        .expect("system entry is an object")
                        .insert("cache_control".into(), cache_control.to_json());
                }
                system.push(entry);
            }
            params.insert("system".into(), json!(system));
        } else if let Some(system_prompt) = &context.system_prompt {
            let mut entry = json!({
                "type": "text",
                "text": sanitize_surrogates(system_prompt),
            });
            if let Some(cache_control) = cache_control {
                entry
                    .as_object_mut()
                    .expect("system entry is an object")
                    .insert("cache_control".into(), cache_control.to_json());
            }
            params.insert("system".into(), json!([entry]));
        }

        // Temperature is incompatible with extended thinking (adaptive or budget-based), and always-on
        // models reject sampling params outright.
        if let Some(temperature) = base.temperature {
            if options.map(|options| options.thinking_enabled) != Some(Some(true))
                && !is_always_on_adaptive_thinking_model(&model.id)
            {
                params.insert("temperature".into(), json!(temperature));
            }
        }

        if let Some(tools) = &context.tools {
            if !tools.is_empty() {
                params.insert(
                    "tools".into(),
                    json!(convert_tools(
                        tools,
                        is_oauth_token,
                        get_anthropic_compat(model).supports_eager_tool_input_streaming,
                        cache_control,
                    )),
                );
            }
        }

        // Configure thinking mode: adaptive, budget-based, or explicitly disabled.
        if model.reasoning {
            if options.map(|options| options.thinking_enabled) == Some(Some(true)) {
                let display = options
                    .and_then(|options| options.thinking_display)
                    .unwrap_or(AnthropicThinkingDisplay::Summarized);
                if supports_adaptive_thinking(&model.id) {
                    params.insert(
                        "thinking".into(),
                        json!({ "type": "adaptive", "display": display.as_str() }),
                    );
                    if let Some(effort) = options.and_then(|options| options.effort) {
                        params.insert("output_config".into(), json!({ "effort": effort.as_str() }));
                    }
                } else {
                    params.insert(
                        "thinking".into(),
                        json!({
                            "type": "enabled",
                            "budget_tokens": options.and_then(|options| options.thinking_budget_tokens).unwrap_or(1024),
                            "display": display.as_str(),
                        }),
                    );
                }
            } else if options.map(|options| options.thinking_enabled) == Some(Some(false))
                && !is_always_on_adaptive_thinking_model(&model.id)
            {
                params.insert("thinking".into(), json!({ "type": "disabled" }));
            }
        }

        if let Some(metadata) = &base.metadata {
            if let Some(user_id) = metadata.get("user_id").and_then(|value| value.as_str()) {
                params.insert("metadata".into(), json!({ "user_id": user_id }));
            }
        }

        if let Some(tool_choice) = options.and_then(|options| options.tool_choice.as_ref()) {
            params.insert("tool_choice".into(), tool_choice.to_json());
        }

        Value::Object(params)
    }
}

fn test_model(id: &str, reasoning: bool, images: bool) -> Model {
    model("anthropic-messages", "anthropic", id, reasoning, images)
}

fn cache_controls() -> [Option<CacheControl>; 3] {
    [
        None,
        Some(CacheControl { ttl: None }),
        Some(CacheControl { ttl: Some("1h") }),
    ]
}

fn options_variants() -> Vec<Option<AnthropicOptions>> {
    let base = StreamOptions {
        temperature: Some(0.5),
        max_tokens: Some(2048),
        metadata: Some(std::collections::HashMap::from([(
            "user_id".to_string(),
            json!("user-1"),
        )])),
        ..Default::default()
    };
    vec![
        None,
        Some(AnthropicOptions {
            base: base.clone(),
            thinking_enabled: Some(true),
            thinking_budget_tokens: Some(4096),
            effort: Some(AnthropicEffort::High),
            thinking_display: Some(AnthropicThinkingDisplay::Summarized),
            tool_choice: Some(AnthropicToolChoice::Tool {
                name: "read".into(),
            }),
            ..Default::default()
        }),
        Some(AnthropicOptions {
            base,
            thinking_enabled: Some(false),
            tool_choice: Some(AnthropicToolChoice::Auto),
            ..Default::default()
        }),
    ]
}

fn assert_same_body(
    model: &Model,
    context: &Context,
    is_oauth: bool,
    options: Option<&AnthropicOptions>,
    cache_control: Option<&CacheControl>,
) {
    let legacy = legacy::build_params(model, context, is_oauth, options, cache_control);
    let transformed = transform_for_request(context, model);
    let wire = build_params(model, context, &transformed, is_oauth, options, cache_control);
    let body = request_body(&wire, None, model);
    assert_eq!(
        body,
        legacy.to_string(),
        "body bytes differ (oauth {is_oauth}, cache ttl {:?})",
        cache_control.map(|cache| cache.ttl)
    );
    assert_eq!(body.capacity(), body.len(), "the body is allocated at its exact size");
    assert_eq!(wire.to_value(), legacy, "the hook-facing tree differs");
}

#[test]
fn every_history_shape_matches_the_value_tree_body() {
    for (id, reasoning) in [
        ("claude-opus-4-6", true),
        ("claude-sonnet-4-5", true),
        ("claude-opus-5-5", true),
        ("claude-3-haiku", false),
    ] {
        for images in [false, true] {
            let model = test_model(id, reasoning, images);
            for context in &contexts(&model, "{\"type\":\"not-reasoning-details\"}") {
                for is_oauth in [false, true] {
                    for cache_control in &cache_controls() {
                        for options in &options_variants() {
                            assert_same_body(
                                &model,
                                context,
                                is_oauth,
                                options.as_ref(),
                                cache_control.as_ref(),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn tools_match_with_and_without_eager_input_streaming() {
    let tools = tools();
    for is_oauth in [false, true] {
        for eager in [false, true] {
            for cache_control in &cache_controls() {
                let legacy = Value::Array(legacy::convert_tools(
                    &tools,
                    is_oauth,
                    eager,
                    cache_control.as_ref(),
                ));
                let wire = Wire::Array(convert_tools(&tools, is_oauth, eager, cache_control.as_ref()));
                assert_eq!(to_json_string(&wire), legacy.to_string());
            }
        }
    }
}

#[test]
fn payload_hook_receives_and_replaces_the_same_tree() {
    let model = test_model("claude-sonnet-4-5", true, true);
    let context = &contexts(&model, "sig")[5];
    let replacing: OnPayloadHook = Arc::new(|mut payload: Value, _model: &Model| {
        payload["injected"] = json!([1, "two"]);
        Some(payload)
    });
    let keeping: OnPayloadHook = Arc::new(|_payload: Value, _model: &Model| None);
    let cache_control = CacheControl { ttl: None };
    for hook in [replacing, keeping] {
        let mut legacy = legacy::build_params(&model, context, true, None, Some(&cache_control));
        if let Some(next) = hook(legacy.clone(), &model) {
            legacy = next;
        }
        let transformed = transform_for_request(context, &model);
        let wire = build_params(&model, context, &transformed, true, None, Some(&cache_control));
        assert_eq!(request_body(&wire, Some(&hook), &model), legacy.to_string());
    }
}
