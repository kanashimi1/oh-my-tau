//! Anthropic Messages request params assembly.

use serde_json::json;

use crate::providers::anthropic::convert::{convert_messages, convert_tools};
use crate::providers::anthropic::{
    get_anthropic_compat, is_always_on_adaptive_thinking_model, supports_adaptive_thinking,
    AnthropicOptions, AnthropicThinkingDisplay, CacheControl,
};
use crate::types::{Context, Message, Model};
use crate::utils_inner::wire_json::{Wire, WireObject};

/// The request params for `context`, whose history was transformed into `transformed`
/// ([`transform_for_request`](crate::providers::anthropic::convert::transform_for_request)). The
/// system prompt, messages and tools borrow from them instead of being copied into a `Value` tree:
/// the body is serialized straight from the result.
// Long by design: mirrors the provider's stream shape.
#[allow(clippy::too_many_lines)]
pub(crate) fn build_params<'a>(
    model: &Model,
    context: &'a Context,
    transformed: &'a [Message],
    is_oauth_token: bool,
    options: Option<&AnthropicOptions>,
    cache_control: Option<&CacheControl>,
) -> Wire<'a> {
    let base = options
        .map(|options| options.base.clone())
        .unwrap_or_default();
    let mut params = WireObject::new();
    params.insert("model", json!(model.id));
    params.insert(
        "messages",
        Wire::Array(convert_messages(transformed, is_oauth_token, cache_control)),
    );
    params.insert(
        "max_tokens",
        json!(base.max_tokens.unwrap_or(model.max_tokens / 3)),
    );
    params.insert("stream", json!(true));

    let system_entry = |text: &'a str| {
        let mut entry =
            WireObject::from_entries([("type", Wire::str("text")), ("text", Wire::str(text))]);
        if let Some(cache_control) = cache_control {
            entry.insert("cache_control", cache_control.to_json());
        }
        Wire::Object(entry)
    };
    // For OAuth tokens, we MUST include Claude Code identity.
    if is_oauth_token {
        let mut system =
            vec![system_entry("You are Claude Code, Anthropic's official CLI for Claude.")];
        if let Some(system_prompt) = &context.system_prompt {
            system.push(system_entry(system_prompt.as_str()));
        }
        params.insert("system", Wire::Array(system));
    } else if let Some(system_prompt) = &context.system_prompt {
        params.insert("system", Wire::Array(vec![system_entry(system_prompt.as_str())]));
    }

    // Temperature is incompatible with extended thinking (adaptive or budget-based), and always-on
    // models reject sampling params outright.
    if let Some(temperature) = base.temperature {
        if options.map(|options| options.thinking_enabled) != Some(Some(true))
            && !is_always_on_adaptive_thinking_model(&model.id)
        {
            params.insert("temperature", json!(temperature));
        }
    }

    if let Some(tools) = &context.tools {
        if !tools.is_empty() {
            params.insert(
                "tools",
                Wire::Array(convert_tools(
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
                    "thinking",
                    json!({ "type": "adaptive", "display": display.as_str() }),
                );
                if let Some(effort) = options.and_then(|options| options.effort) {
                    params.insert("output_config", json!({ "effort": effort.as_str() }));
                }
            } else {
                params.insert(
                    "thinking",
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
            params.insert("thinking", json!({ "type": "disabled" }));
        }
    }

    if let Some(metadata) = &base.metadata {
        if let Some(user_id) = metadata.get("user_id").and_then(|value| value.as_str()) {
            params.insert("metadata", json!({ "user_id": user_id }));
        }
    }

    if let Some(tool_choice) = options.and_then(|options| options.tool_choice.as_ref()) {
        params.insert("tool_choice", tool_choice.to_json());
    }

    Wire::Object(params)
}
