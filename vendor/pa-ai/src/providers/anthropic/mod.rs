//! Anthropic Messages streaming provider: request headers and options here, message/tool conversion
//! in [`convert`], params assembly in [`params`], and the SSE streaming core in [`stream`].
//! OAuth/Claude-Code header modes, beta headers, and adaptive vs budget-based thinking selection
//! live here.

use serde_json::{json, Map, Value};

use crate::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEvent, AssistantMessageEventStream,
};
use crate::registry::Provider;
use crate::types::{
    AssistantMessage, Context, Model, ModelThinkingLevel, SimpleStreamOptions, SimpleToolChoice,
    StopReason, StreamOptions, Usage,
};
use crate::utils_inner::diagnostics::now_ms;

mod convert;
pub(crate) mod request;
mod stream;

pub use stream::stream_anthropic;

#[cfg(test)]
mod stream_tests;

pub const API_ANTHROPIC_MESSAGES: &str = "anthropic-messages";

/// Claude Code version mimicked in OAuth mode. The API gates newer models on the claimed client
/// version (e.g. claude-opus-5.5 requires >= 2.280), so keep this at or above the latest released
/// Claude Code.
const CLAUDE_CODE_VERSION: &str = "2.1.281";
const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const CONTEXT_MANAGEMENT_BETA: &str = "context-management-2025-06-27";
const EFFORT_BETA: &str = "effort-2025-11-24";
const THINKING_BINDING_CONTROLS_BETA: &str = "thinking-binding-controls-2026-08-01";
const MID_CONVERSATION_SYSTEM_CLEAR_AT_BETA: &str = "mid-conversation-system-clear-at-2026-08-21";
const MID_CONVERSATION_TOOL_CHANGES_BETA: &str = "mid-conversation-tool-changes-2026-07-01";
const MID_CONVERSATION_OUTPUT_CONFIG_BETA: &str = "mid-conversation-output-config-2026-07-01";

/// Effort levels for adaptive thinking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnthropicEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
    /// `thinking.type: "adaptive"` without `output_config.effort` (MiniMax's binary tag).
    Adaptive,
}

impl AnthropicEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            AnthropicEffort::Low => "low",
            AnthropicEffort::Medium => "medium",
            AnthropicEffort::High => "high",
            AnthropicEffort::Xhigh => "xhigh",
            AnthropicEffort::Max => "max",
            AnthropicEffort::Adaptive => "adaptive",
        }
    }
}

/// Thinking display mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // full TS option surface; variants set by callers
pub enum AnthropicThinkingDisplay {
    Summarized,
    Omitted,
}

impl AnthropicThinkingDisplay {
    pub fn as_str(self) -> &'static str {
        match self {
            AnthropicThinkingDisplay::Summarized => "summarized",
            AnthropicThinkingDisplay::Omitted => "omitted",
        }
    }
}

/// Tool selection passed to the API.
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)] // full TS option surface; variants set by callers
pub enum AnthropicToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

/// Provider-native options.
#[derive(Clone, Default)]
pub struct AnthropicOptions {
    pub base: StreamOptions,
    pub thinking_enabled: Option<bool>,
    pub thinking_budget_tokens: Option<u64>,
    pub effort: Option<AnthropicEffort>,
    pub thinking_display: Option<AnthropicThinkingDisplay>,
    pub interleaved_thinking: Option<bool>,
    pub tool_choice: Option<AnthropicToolChoice>,
    /// Reasoning level fallback when `effort` is unset (TS `AnthropicOptions.reasoning`).
    pub reasoning: Option<ModelThinkingLevel>,
    /// Upstream wire model id override (TS `requestModelId`).
    pub request_model_id: Option<String>,
    /// The policy view the options were mapped against (a rebuilt model such as the Kimi shim
    /// carries an authored thinking config a fresh resolution would not reproduce).
    pub view: Option<crate::compat::ModelView>,
}

/// Check if a model supports adaptive thinking (Opus 4.6+, Sonnet 4.6+).
pub(crate) fn supports_adaptive_thinking(model_id: &str) -> bool {
    model_id.contains("opus-4-6")
        || model_id.contains("opus-4.6")
        || model_id.contains("opus-5")
        || model_id.contains("opus-4-7")
        || model_id.contains("opus-4.7")
        || model_id.contains("opus-4-8")
        || model_id.contains("opus-4.8")
        || model_id.contains("sonnet-4-6")
        || model_id.contains("sonnet-4.6")
        || model_id.contains("sonnet-5")
        || model_id.contains("fable-5")
        || model_id.contains("mythos-5")
        || model_id.contains("mythos-preview")
}

pub(crate) fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

pub(crate) fn should_use_fine_grained_tool_streaming_beta(
    model: &Model,
    context: &Context,
) -> bool {
    context
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        && !crate::compat::ModelView::of(model).is("supportsEagerToolInputStreaming")
}

/// TS `resolveAnthropicControlBetas`: the betas a request opts into through the resolved compat
/// (thinking-binding controls, mid-conversation system/tool/effort changes). The binding beta must
/// accompany `thinking.block_binding`, which the API rejects without it.
pub(crate) fn control_betas(view: &crate::compat::ModelView) -> Vec<&'static str> {
    let mut betas = Vec::new();
    if view.thinking_flag("prefixBinding") == Some(true)
        && view.is("supportsThinkingBindingControls")
    {
        betas.push(THINKING_BINDING_CONTROLS_BETA);
    }
    if view.is("supportsTurnScopedSystem") {
        betas.push(MID_CONVERSATION_SYSTEM_CLEAR_AT_BETA);
    }
    if view.is("supportsMidConversationToolChanges") {
        betas.push(MID_CONVERSATION_TOOL_CHANGES_BETA);
    }
    if view.is("supportsPerMessageEffort") {
        betas.push(MID_CONVERSATION_OUTPUT_CONFIG_BETA);
    }
    betas
}

/// Betas the built request body requires: the compat control betas plus the field-gated ones TS
/// attaches alongside the field (`context_management` needs the context-management beta,
/// `output_config.effort` the effort beta). The API rejects each field without its beta.
pub(crate) fn body_betas(view: &crate::compat::ModelView, body: &Value) -> Vec<&'static str> {
    let mut betas = control_betas(view);
    if body.get("context_management").is_some() {
        betas.push(CONTEXT_MANAGEMENT_BETA);
    }
    if body.pointer("/output_config/effort").is_some() {
        betas.push(EFFORT_BETA);
    }
    betas
}

pub(crate) fn merge_headers(sources: &[Option<Map<String, Value>>]) -> Map<String, Value> {
    let mut merged = Map::new();
    for source in sources.iter().flatten() {
        for (key, value) in source {
            merged.insert(key.clone(), value.clone());
        }
    }
    merged
}

pub(crate) fn headers_to_pairs(headers: &Map<String, Value>) -> Vec<(String, String)> {
    headers
        .iter()
        .filter_map(|(key, value)| match value {
            Value::String(text) => Some((key.clone(), text.clone())),
            _ => None,
        })
        .collect()
}

/// Build the request headers for the messages endpoint, mirroring the SDK client configurations
/// (OAuth/Claude Code mode, cloudflare gateway, github-copilot, plain API key).
pub(crate) fn build_request_headers(
    model: &Model,
    api_key: &str,
    interleaved_thinking: bool,
    use_fine_grained_tool_streaming_beta: bool,
    extra_betas: &[&str],
    options_headers: Option<&std::collections::HashMap<String, String>>,
    session_id: Option<&str>,
) -> (Vec<(String, String)>, bool) {
    let is_oauth = is_oauth_token(api_key);
    let needs_interleaved_beta = interleaved_thinking && !supports_adaptive_thinking(&model.id);
    let mut beta_features: Vec<&str> = Vec::new();
    if use_fine_grained_tool_streaming_beta {
        beta_features.push(FINE_GRAINED_TOOL_STREAMING_BETA);
    }
    if needs_interleaved_beta {
        beta_features.push(INTERLEAVED_THINKING_BETA);
    }
    for beta in extra_betas {
        if !beta_features.contains(beta) {
            beta_features.push(beta);
        }
    }
    let beta_header = if beta_features.is_empty() {
        None
    } else {
        Some(beta_features.join(","))
    };

    let model_headers: Option<Map<String, Value>> = model.headers.as_ref().map(|headers| {
        headers
            .iter()
            .map(|(key, value)| (key.clone(), json!(value)))
            .collect()
    });
    let options_headers_json: Option<Map<String, Value>> = options_headers.map(|headers| {
        headers
            .iter()
            .map(|(key, value)| (key.clone(), json!(value)))
            .collect()
    });

    let mut default_headers = match model.provider.as_str() {
        "cloudflare-ai-gateway" => {
            let mut headers = Map::new();
            headers.insert("accept".into(), json!("application/json"));
            headers.insert(
                "anthropic-dangerous-direct-browser-access".into(),
                json!("true"),
            );
            headers.insert(
                "cf-aig-authorization".into(),
                json!(format!("Bearer {api_key}")),
            );
            headers.insert("x-api-key".into(), Value::Null);
            headers.insert("Authorization".into(), Value::Null);
            if let Some(beta) = &beta_header {
                headers.insert("anthropic-beta".into(), json!(beta));
            }
            merge_headers(&[Some(headers), model_headers, options_headers_json])
        }
        "github-copilot" => {
            let mut headers = Map::new();
            headers.insert("accept".into(), json!("application/json"));
            headers.insert(
                "anthropic-dangerous-direct-browser-access".into(),
                json!("true"),
            );
            if let Some(beta) = &beta_header {
                headers.insert("anthropic-beta".into(), json!(beta));
            }
            merge_headers(&[Some(headers), model_headers, options_headers_json])
        }
        _ => {
            let mut headers = Map::new();
            headers.insert("accept".into(), json!("application/json"));
            headers.insert(
                "anthropic-dangerous-direct-browser-access".into(),
                json!("true"),
            );
            if is_oauth {
                headers.insert(
                    "anthropic-beta".into(),
                    json!(["claude-code-20250219", "oauth-2025-04-20"]
                        .iter()
                        .chain(beta_features.iter())
                        .copied()
                        .collect::<Vec<_>>()
                        .join(",")),
                );
                headers.insert(
                    "user-agent".into(),
                    json!(format!("claude-cli/{CLAUDE_CODE_VERSION}")),
                );
                headers.insert("x-app".into(), json!("cli"));
            } else if let Some(beta) = &beta_header {
                headers.insert("anthropic-beta".into(), json!(beta));
            }
            merge_headers(&[Some(headers), model_headers, options_headers_json])
        }
    };

    // withOpenCodeHeaders: session header for opencode providers.
    if model.provider == "opencode" || model.provider == "opencode-go" {
        if let Some(session_id) = session_id {
            default_headers.insert("session_id".into(), json!(session_id));
        }
    }

    let mut pairs = headers_to_pairs(&default_headers);
    // The Anthropic SDK always sends the API version; mirror it.
    pairs.insert(0, ("anthropic-version".into(), "2023-06-01".into()));
    if model.provider == "cloudflare-ai-gateway" || model.provider == "github-copilot" || is_oauth {
        pairs.push(("Authorization".into(), format!("Bearer {api_key}")));
    } else {
        pairs.push(("x-api-key".into(), api_key.to_string()));
    }
    (pairs, is_oauth)
}

/// Thinking budgets per effort when the caller sets none (TS `ANTHROPIC_THINKING`).
pub(crate) fn thinking_budget_for(options: &SimpleStreamOptions, level: ModelThinkingLevel) -> u64 {
    let custom = options.thinking_budgets.as_ref().and_then(|b| match level {
        ModelThinkingLevel::Minimal => b.minimal,
        ModelThinkingLevel::Low => b.low,
        ModelThinkingLevel::Medium => b.medium,
        ModelThinkingLevel::High => b.high,
        _ => None,
    });
    custom.unwrap_or(match level {
        ModelThinkingLevel::Minimal => 1024,
        ModelThinkingLevel::Low => 4096,
        ModelThinkingLevel::Medium => 8192,
        ModelThinkingLevel::High => 16384,
        _ => 32768,
    })
}

impl AnthropicEffort {
    pub(crate) fn from_wire(name: &str) -> Option<Self> {
        Some(match name {
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::Xhigh,
            "max" => Self::Max,
            "adaptive" => Self::Adaptive,
            _ => return None,
        })
    }
}

/// TS `mapAnthropicToolChoice`.
pub(crate) fn map_tool_choice(choice: Option<&SimpleToolChoice>) -> Option<AnthropicToolChoice> {
    Some(match choice? {
        SimpleToolChoice::Auto => AnthropicToolChoice::Auto,
        SimpleToolChoice::None => AnthropicToolChoice::None,
        // The anthropic wire has no "required": a forced call is `any`.
        SimpleToolChoice::Any | SimpleToolChoice::Required => AnthropicToolChoice::Any,
        SimpleToolChoice::Function { name } if !name.is_empty() => {
            AnthropicToolChoice::Tool { name: name.clone() }
        }
        SimpleToolChoice::Function { .. } => return None,
    })
}

/// TS `normalizeMandatoryReasoningOptions`: mandatory-reasoning endpoints reject disabled thinking,
/// so a request without an effort clamps to the lowest supported one.
pub(crate) fn normalize_mandatory_reasoning(
    model: &Model,
    view: &crate::compat::ModelView,
    options: &SimpleStreamOptions,
) -> SimpleStreamOptions {
    let explicit = options
        .reasoning
        .filter(|l| *l != ModelThinkingLevel::Off)
        .is_some()
        && !options.disable_reasoning;
    if !model.reasoning
        || view.thinking_flag("requiresEffort") != Some(true)
        || view.thinking_flag("suppressWhenOff") == Some(true)
        || explicit
    {
        return options.clone();
    }
    let efforts = view.efforts();
    let floor = ["minimal", "low", "medium", "high", "xhigh", "max"]
        .into_iter()
        .find(|e| efforts.contains(e));
    let Some(floor) = floor.and_then(crate::models::thinking_level_from_str) else {
        return options.clone();
    };
    let mut next = options.clone();
    next.reasoning = Some(floor);
    next.disable_reasoning = false;
    next
}

/// TS `mapOptionsForApi` for `anthropic-messages`: simple options -> provider-native options.
///
/// # Errors
/// The requested effort is not supported by the model, or the thinking budget leaves no room.
pub(crate) fn map_simple_options(
    model: &Model,
    view: &crate::compat::ModelView,
    raw: &SimpleStreamOptions,
) -> Result<AnthropicOptions, String> {
    let options = normalize_mandatory_reasoning(model, view, raw);
    let model_max = (model.max_tokens > 0).then_some(model.max_tokens);
    let mut base = options.base.clone();
    base.max_tokens = options.base.max_tokens.or(model_max);
    let tool_choice = map_tool_choice(options.tool_choice.as_ref());
    let wire_id = |effort: Option<ModelThinkingLevel>| -> Option<String> {
        view.effort_routing(effort.map(ModelThinkingLevel::wire_name))
            .map(str::to_string)
    };
    let off = |base: StreamOptions| {
        let mut o = AnthropicOptions::from_base(base);
        o.request_model_id = wire_id(None);
        o.thinking_enabled = Some(false);
        o.tool_choice = tool_choice.clone();
        o
    };
    let reasoning = options.reasoning.filter(|l| *l != ModelThinkingLevel::Off);
    let Some(reasoning) = reasoning.filter(|_| model.reasoning && !options.disable_reasoning)
    else {
        return Ok(off(base));
    };
    let mut budget = thinking_budget_for(&options, reasoning) as i64;
    if budget <= 0 {
        return Ok(off(base));
    }
    let mode = view.thinking_str("mode");
    let effort = if matches!(mode, Some("anthropic-adaptive" | "anthropic-budget-effort")) {
        AnthropicEffort::from_wire(&request::map_effort_to_adaptive(
            model,
            view,
            reasoning.wire_name(),
        )?)
    } else {
        None
    };
    let cap = |uncapped: u64| model_max.map_or(uncapped, |m| uncapped.min(m));
    let with_thinking = |max: u64, budget: i64| cap(max.saturating_add(budget.max(0) as u64));
    let max_with_thinking = base.max_tokens.map(|m| with_thinking(m, budget));
    let finish = |mut base: StreamOptions, max: Option<u64>, budget_tokens: Option<u64>| {
        base.max_tokens = max;
        let mut o = AnthropicOptions::from_base(base);
        o.request_model_id = wire_id(Some(reasoning));
        o.thinking_enabled = Some(true);
        o.thinking_budget_tokens = budget_tokens;
        o.effort = effort;
        o.tool_choice = tool_choice.clone();
        o
    };
    if mode == Some("anthropic-adaptive") {
        return Ok(finish(base, max_with_thinking, None));
    }
    if std::env::var("PI_NO_INTERLEAVED_THINKING").as_deref() != Ok("1") {
        if let Some(m) = model_max {
            if (m as i64) < budget + request::OUTPUT_FALLBACK_BUFFER as i64 {
                budget = m as i64 - request::OUTPUT_FALLBACK_BUFFER as i64;
            }
        }
        if budget >= 1024 {
            return Ok(finish(base, max_with_thinking, Some(budget as u64)));
        }
    }
    let max_tokens = match base.max_tokens {
        None => cap(64_000),
        Some(m) => with_thinking(m, budget),
    };
    if (max_tokens as i64) < budget + request::OUTPUT_FALLBACK_BUFFER as i64 {
        budget = max_tokens as i64 - request::OUTPUT_FALLBACK_BUFFER as i64;
    }
    if budget < 1024 {
        return Ok(off(base));
    }
    Ok(finish(base, Some(max_tokens), Some(budget as u64)))
}

/// The request body the anthropic transport sends for `options` (`api_key` decides OAuth mode).
///
/// # Errors
/// The options cannot form a valid request (unsupported effort, impossible thinking budget).
pub(crate) fn build_request_body(
    model: &Model,
    view: &crate::compat::ModelView,
    context: &Context,
    options: &AnthropicOptions,
    api_key: &str,
) -> Result<Value, String> {
    request::build_body(&request::BodyInput {
        model,
        view,
        context,
        is_oauth: is_oauth_token(api_key),
        options,
    })
}

pub(crate) fn error_stream_for(model: &Model, message: String) -> AssistantMessageEventStream {
    let (writer, reader) = create_assistant_message_event_stream();
    let message = AssistantMessage {
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
        error_message: Some(message),
        timestamp: now_ms(),
        rest: Map::default(),
    };
    writer.push(AssistantMessageEvent::Error {
        reason: crate::types::ErrorStopReason::Error,
        error: message.clone(),
    });
    writer.end(Some(message));
    reader
}

pub fn stream_simple_anthropic(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    crate::transport::stream_simple_for_api(model, context, options)
}

impl AnthropicOptions {
    pub fn from_base(base: StreamOptions) -> Self {
        Self {
            base,
            thinking_enabled: None,
            thinking_budget_tokens: None,
            effort: None,
            thinking_display: None,
            interleaved_thinking: None,
            tool_choice: None,
            reasoning: None,
            request_model_id: None,
            view: None,
        }
    }
}

/// Registry provider for the `anthropic-messages` API.
pub struct AnthropicMessagesProvider;

impl Provider for AnthropicMessagesProvider {
    fn api(&self) -> &str {
        API_ANTHROPIC_MESSAGES
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let options = options.map(|base| AnthropicOptions::from_base(base.clone()));
        stream_anthropic(model, context, options.as_ref())
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        stream_simple_anthropic(model, context, options)
    }
}

#[cfg(test)]
mod subscription_identity_tests {
    use super::build_request_headers;
    use crate::types::{zero_model_cost, Model, ModelInput};

    // TS #2645's wire-contract assertions (anthropic-thinking-disable.test.ts): subscription
    // requests claim the Claude Code client identity, and the claimed version must stay at or above
    // what the API's model gates require (the opus-5.5 family rejects anything below 2.280).
    fn test_model() -> Model {
        Model {
            id: "claude-opus-5-5".into(),
            name: "Claude Opus 5.5".into(),
            api: "anthropic-messages".into(),
            provider: "anthropic".into(),
            base_url: "https://api.anthropic.com".into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![ModelInput::Text],
            cost: zero_model_cost(),
            context_window: 200_000,
            max_tokens: 32_000,
            max_tokens_explicit: false,
            omit_max_output_tokens: None,
            request_model_id: None,
            featured: None,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn oauth_requests_claim_the_claude_code_identity() {
        let (headers, is_oauth) = build_request_headers(
            &test_model(),
            "sk-ant-oat-test",
            false,
            false,
            &[],
            None,
            None,
        );
        assert!(is_oauth);
        let header = |name: &str| {
            headers
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, value)| value.as_str())
        };
        let user_agent = header("user-agent").expect("the OAuth request sends a user-agent");
        assert!(user_agent.starts_with("claude-cli/"), "got {user_agent}");
        assert_eq!(header("x-app"), Some("cli"));
        let beta = header("anthropic-beta").expect("the OAuth request sends the claude-code beta");
        assert!(beta.contains("claude-code-20250219"));
    }

    /// `thinking.block_binding` is rejected (400 "Extra inputs are not permitted") unless the
    /// request carries the thinking-binding-controls beta.
    #[test]
    fn binding_controls_model_opts_into_the_beta() {
        let mut model = test_model();
        model.id = "claude-haiku-5-5".into();
        model.reasoning = true;
        let view = crate::compat::ModelView::of(&model);
        let betas = super::body_betas(&view, &serde_json::json!({}));
        assert!(
            betas.contains(&"thinking-binding-controls-2026-08-01"),
            "{betas:?}"
        );
        let (headers, _) =
            build_request_headers(&model, "sk-ant-oat-test", false, false, &betas, None, None);
        let beta = headers
            .iter()
            .find(|(k, _)| k == "anthropic-beta")
            .map(|(_, v)| v.as_str())
            .unwrap();
        assert!(
            beta.contains("thinking-binding-controls-2026-08-01"),
            "{beta}"
        );
        let body =
            serde_json::json!({"context_management": {}, "output_config": {"effort": "high"}});
        let betas = super::body_betas(&crate::compat::ModelView::of(&test_model()), &body);
        assert!(
            betas.contains(&"context-management-2025-06-27")
                && betas.contains(&"effort-2025-11-24"),
            "{betas:?}"
        );
        let plain = crate::compat::ModelView::of(&test_model());
        assert!(!super::body_betas(&plain, &serde_json::json!({}))
            .contains(&"thinking-binding-controls-2026-08-01"));
    }
}
