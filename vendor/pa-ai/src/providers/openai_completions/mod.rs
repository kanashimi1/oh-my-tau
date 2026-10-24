//! `OpenAI` Chat Completions streaming provider: compat detection and options here,
//! message/tool/usage conversion in [`convert`], params and header assembly in [`params`], the
//! SDK-shaped error surface in [`errors`], and the chunk-driven streaming core in [`stream`].

use serde_json::{json, Map, Value};

use crate::env_api_keys::get_env_api_key;
use crate::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEvent, AssistantMessageEventStream,
};
use crate::models::clamp_thinking_level;
use crate::providers::simple_options::build_base_options;
use crate::registry::Provider;
use crate::types::{
    AssistantContent, AssistantMessage, CacheRetention, Context, Model, ModelThinkingLevel,
    SimpleStreamOptions, SimpleToolChoice, StopReason, StreamOptions, Usage, UsageCost,
};

pub(crate) mod compat;
mod convert;
mod errors;
pub(crate) mod params;
pub(crate) mod schema;
mod stream;
mod wire_cache;

#[cfg(test)]
mod body_parity_tests;

pub(crate) use compat::resolve_compat;
pub(crate) use params::normalize_stable_id;
pub use stream::stream_openai_completions;

pub const API_OPENAI_COMPLETIONS: &str = "openai-completions";

/// Unified tool choice → openai-completions wire form (TS `mapToOpenAICompletionsToolChoice`):
/// `any` becomes `required`, a function pin becomes `{type:"function", function:{name}}`.
fn map_tool_choice(choice: &SimpleToolChoice) -> ToolChoice {
    match choice {
        SimpleToolChoice::Auto => ToolChoice::Auto,
        SimpleToolChoice::None => ToolChoice::None,
        SimpleToolChoice::Any | SimpleToolChoice::Required => ToolChoice::Required,
        SimpleToolChoice::Function { name } => ToolChoice::Function { name: name.clone() },
    }
}

const REASONING_DETAILS_SIGNATURE_TYPE: &str = "openai-completions.reasoning_details.v1";
pub(crate) const REASONING_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_text"];

/// Tool selection passed to the API.
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)] // full TS option surface; variants set by callers
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function { name: String },
}

impl ToolChoice {
    fn to_json(&self) -> Value {
        match self {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::None => json!("none"),
            ToolChoice::Required => json!("required"),
            ToolChoice::Function { name } => json!({
                "type": "function",
                "function": { "name": name }
            }),
        }
    }
}

/// Provider-native options.
#[derive(Clone, Default)]
pub struct OpenAICompletionsOptions {
    pub base: StreamOptions,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning_effort: Option<ModelThinkingLevel>,
    /// Explicit reasoning toggle. `None` preserves the provider/model default; `Some(false)` is
    /// the TS `disableReasoning`.
    pub reasoning_enabled: Option<bool>,
    /// The caller set `max_tokens` itself (TS `maxTokensExplicit`): hosts that route across
    /// upstreams with different output caps omit catalog-default caps.
    pub max_tokens_explicit: bool,
}

impl OpenAICompletionsOptions {
    pub fn from_base(base: StreamOptions) -> Self {
        let max_tokens_explicit = base.max_tokens.is_some();
        Self {
            base,
            tool_choice: None,
            reasoning_effort: None,
            reasoning_enabled: None,
            max_tokens_explicit,
        }
    }
}

pub fn resolve_cache_retention(cache_retention: Option<CacheRetention>) -> CacheRetention {
    if let Some(retention) = cache_retention {
        return retention;
    }
    if std::env::var("PI_CACHE_RETENTION").as_deref() == Ok("long") {
        return CacheRetention::Long;
    }
    CacheRetention::Short
}

pub(crate) fn has_tool_history(messages: &[crate::types::Message]) -> bool {
    use crate::types::Message;
    for msg in messages {
        if let Message::ToolResult(_) = msg {
            return true;
        }
        if let Message::Assistant(assistant) = msg {
            if assistant
                .content
                .iter()
                .any(|block| matches!(block, AssistantContent::ToolCall(_)))
            {
                return true;
            }
        }
    }
    false
}

pub(crate) fn encode_reasoning_details(details: &[Value]) -> String {
    json!({
        "type": REASONING_DETAILS_SIGNATURE_TYPE,
        "details": details,
    })
    .to_string()
}

pub(crate) fn decode_reasoning_details(signature: Option<&str>) -> Option<Vec<Value>> {
    let signature = signature?;
    if !signature.starts_with('{') {
        return None;
    }
    let parsed: Value = serde_json::from_str(signature).ok()?;
    if parsed.get("type")?.as_str()? != REASONING_DETAILS_SIGNATURE_TYPE {
        return None;
    }
    let details = parsed.get("details")?.as_array()?;
    for detail in details {
        if !detail.is_object() {
            return None;
        }
    }
    Some(details.clone())
}

pub fn stream_simple_openai_completions(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let api_key = options
        .and_then(|options| options.base.api_key.clone())
        .or_else(|| get_env_api_key(&model.provider));
    let Some(api_key) = api_key else {
        let (writer, reader) = create_assistant_message_event_stream();
        let mut message = AssistantMessage {
            content: Vec::new(),
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Error,
            stop_reason_raw: None,
            error_status: None,
            error_message: Some(format!("No API key for provider: {}", model.provider)),
            timestamp: crate::utils_inner::diagnostics::now_ms(),
            rest: Map::default(),
        };
        message.usage.cost = UsageCost::default();
        writer.push(AssistantMessageEvent::Error {
            reason: crate::types::ErrorStopReason::Error,
            error: message.clone(),
        });
        writer.end(Some(message));
        return reader;
    };

    let base = build_base_options(model, options, Some(&api_key));
    let requested_reasoning = options.and_then(|options| options.reasoning);
    let reasoning_specified = requested_reasoning.is_some();
    let clamped_reasoning =
        requested_reasoning.map(|reasoning| clamp_thinking_level(model, reasoning));
    // TS `disableReasoning`: wins over `reasoning` — no effort is sent and the toggle is
    // explicitly off, so effort-declared models do not fall back to their default.
    let disable_reasoning = options.is_some_and(|options| options.disable_reasoning)
        || clamped_reasoning == Some(ModelThinkingLevel::Off);
    let reasoning_effort = if disable_reasoning {
        None
    } else {
        clamped_reasoning.filter(|level| *level != ModelThinkingLevel::Off)
    };
    let reasoning_enabled = if disable_reasoning {
        Some(false)
    } else if reasoning_specified {
        Some(clamped_reasoning != Some(ModelThinkingLevel::Off))
    } else {
        None
    };
    let tool_choice = options
        .and_then(|options| options.tool_choice.as_ref())
        .map(map_tool_choice);

    let stream_options = OpenAICompletionsOptions {
        base,
        tool_choice,
        reasoning_effort,
        reasoning_enabled,
        max_tokens_explicit: options.is_some_and(|o| o.base.max_tokens.is_some()),
    };
    stream_openai_completions(model, context, Some(&stream_options))
}

/// Registry provider for the `openai-completions` API.
pub struct OpenAICompletionsProvider;

impl Provider for OpenAICompletionsProvider {
    fn api(&self) -> &str {
        API_OPENAI_COMPLETIONS
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let options = options.map(|base| OpenAICompletionsOptions::from_base(base.clone()));
        stream_openai_completions(model, context, options.as_ref())
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        stream_simple_openai_completions(model, context, options)
    }
}

/// The request body the chat-completions transport sends for `options`.
///
/// # Errors
/// The model's compat cannot be resolved.
pub(crate) fn build_request_body(
    model: &Model,
    context: &Context,
    options: &OpenAICompletionsOptions,
) -> Result<Value, String> {
    let compat = resolve_compat(model).map_err(|e| format!("{e:?}"))?;
    let retention = resolve_cache_retention(options.base.cache_retention);
    let transformed = convert::transform_for_request(model, context, &compat);
    Ok(params::build_params(
        model,
        context,
        &transformed,
        Some(options),
        &compat,
        retention,
    )
    .to_value())
}
