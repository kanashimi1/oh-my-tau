//! Resolved chat-completions compat and the reasoning/compat policy (TS `resolveOpenAICompatPolicy`
//! and `applyChatCompletionsCompatPolicy`, `ai/src/providers/openai-shared.ts`).
//!
//! The compat record is never detected here: it is the output of the shared policy resolver
//! ([`crate::compat`]), the Rust port of TS `resolveModelPolicy`, run over the model's sparse
//! `compat` overrides. This module only types that record and interprets it.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::compat::resolve_for_model;
use crate::models::get_supported_thinking_levels;
use crate::providers::openai_completions::ToolChoice;
use crate::types::{Model, ModelThinkingLevel};
use crate::utils_inner::stream_failure::ProviderError;

/// Fully resolved chat-completions compat (`ResolvedOpenAICompat`), the complete record the TS
/// resolver produces. Absent keys are the record's `undefined` values.
// A flat capability-flag contract mirroring the TS record; reshaping is an API change.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct ResolvedCompat {
    pub supports_store: bool,
    pub supports_developer_role: bool,
    /// TS reads `?? true`.
    pub supports_multiple_system_messages: Option<bool>,
    pub supports_reasoning_effort: bool,
    pub supports_reasoning_params: bool,
    pub supports_sampling_params: bool,
    pub supports_penalty_and_stop_params: bool,
    pub reasoning_effort_map: BTreeMap<String, String>,
    /// TS reads `!== false`.
    pub supports_usage_in_streaming: Option<bool>,
    pub always_send_max_tokens: bool,
    pub disable_reasoning_on_forced_tool_choice: bool,
    pub disable_reasoning_on_tool_choice: bool,
    pub disable_reasoning_with_tools: bool,
    pub supports_tool_choice: bool,
    pub supports_forced_tool_choice: bool,
    pub supports_named_tool_choice: bool,
    pub max_tokens_field: String,
    pub requires_tool_result_name: bool,
    pub requires_assistant_after_tool_result: bool,
    pub requires_thinking_as_text: bool,
    pub requires_mistral_tool_ids: bool,
    pub thinking_format: String,
    pub reasoning_disable_mode: String,
    pub omit_reasoning_effort: bool,
    pub include_encrypted_reasoning: bool,
    pub filter_reasoning_history: bool,
    pub thinking_keep: Option<String>,
    pub reasoning_content_field: Option<String>,
    pub mistral_reasoning_content_parts: Option<bool>,
    pub requires_reasoning_content_for_tool_calls: bool,
    pub requires_reasoning_content_for_all_assistant_turns: bool,
    pub allows_synthetic_reasoning_content_for_tool_calls: bool,
    pub synthetic_reasoning_content_fallback: Option<String>,
    pub replay_reasoning_content: bool,
    pub qwen_preserve_thinking: bool,
    pub qwen_template_reasoning_effort: bool,
    pub requires_assistant_content_for_tool_calls: bool,
    pub cache_control_format: Option<String>,
    pub supports_prompt_cache_breakpoints: bool,
    pub open_router_routing: Option<Value>,
    pub vercel_gateway_routing: Option<Value>,
    pub is_open_router_host: bool,
    pub is_vercel_gateway_host: bool,
    pub wire_model_id_mode: Option<String>,
    pub supports_strict_mode: Option<bool>,
    pub extra_body: Option<Map<String, Value>>,
    pub tool_strict_mode: Option<String>,
    pub tool_schema_flavor: Option<String>,
    pub clamp_output_to_model_max: bool,
    pub native_kimi_k3_reasoning: bool,
    pub zai_reasoning_effort_dialect: bool,
    pub drop_thinking_when_reasoning_effort: bool,
    pub uses_open_ai_tool_call_id_limit: bool,
    pub reject_root_object_union: bool,
    pub supports_prompt_cache_key: bool,
    pub strip_image_input: bool,
    pub prompt_cache_session_header: Option<String>,
    pub session_header: Option<String>,
    pub when_thinking: Option<Box<ResolvedCompat>>,
}

impl ResolvedCompat {
    /// The wire field the output-token cap rides on.
    #[must_use]
    pub fn max_tokens_is_legacy_field(&self) -> bool {
        self.max_tokens_field == "max_tokens"
    }

    /// `supportsStrictMode !== false`.
    #[must_use]
    pub fn strict_mode_allowed(&self) -> bool {
        self.supports_strict_mode != Some(false)
    }

    /// Whether the endpoint takes image parts (TS `isOpenAICompletionsVisionSupported`).
    #[must_use]
    pub fn vision_supported(&self, model: &Model) -> bool {
        use crate::types::ModelInput;
        model.input.iter().any(|m| matches!(m, ModelInput::Image)) && !self.strip_image_input
    }
}

/// The complete resolved compat for a model: the shared resolver run over the model's sparse
/// `compat` overrides.
///
/// # Errors
/// The resolver reports a rule-authoring defect (equal-rank cascade tie).
pub fn resolve_compat(model: &Model) -> Result<ResolvedCompat, ProviderError> {
    let policy = resolve_for_model(model).map_err(|e| ProviderError::Message(e.to_string()))?;
    let record = policy.compat.unwrap_or_default();
    serde_json::from_value(Value::Object(record)).map_err(|e| {
        ProviderError::Message(format!(
            "resolved compat for {}/{}: {e}",
            model.provider, model.id
        ))
    })
}

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisableReason {
    Caller,
    ForcedToolChoice,
    ToolChoice,
    Tools,
    NotRequested,
}

/// TS `OpenAICompatPolicy.reasoning`.
#[derive(Clone, Debug)]
pub struct ReasoningPolicy {
    pub model_supported: bool,
    pub supports_params: bool,
    pub requested_effort: Option<String>,
    pub wire_effort: Option<String>,
    pub enabled: bool,
    pub disabled: bool,
    pub disable_reason: Option<DisableReason>,
    pub dialect: String,
    pub disable_mode: String,
    pub omit_reasoning_effort: bool,
}

/// TS `OpenAICompatPolicy`: the (possibly `whenThinking`) compat the request uses plus the
/// reasoning decision.
#[derive(Clone, Debug)]
pub struct CompatPolicy {
    pub compat: ResolvedCompat,
    pub reasoning: ReasoningPolicy,
}

/// TS `isForcedToolChoice`.
#[must_use]
pub fn is_forced(choice: Option<&ToolChoice>) -> bool {
    matches!(
        choice,
        Some(ToolChoice::Required | ToolChoice::Function { .. })
    )
}

/// TS `mapOpenAIReasoningEffort`: explicit compat map, then the model's baked effort map, else
/// the effort itself.
#[must_use]
pub fn map_reasoning_effort(model: &Model, compat: &ResolvedCompat, effort: &str) -> String {
    if let Some(v) = compat.reasoning_effort_map.get(effort) {
        return v.clone();
    }
    crate::models::thinking_level_from_str(effort)
        .and_then(|level| {
            crate::types::ModelExt::thinking_level_map_value(model, level)
                .flatten()
                .cloned()
        })
        .unwrap_or_else(|| effort.to_string())
}

fn is_implicit_disable_when_not_requested(mode: &str) -> bool {
    matches!(
        mode,
        "zai-thinking-disabled"
            | "qwen-enable-thinking-false"
            | "qwen-template-false"
            | "chat-template-thinking-false"
    )
}

/// TS `shouldDropAutoToolChoiceForReasoning`.
#[must_use]
pub fn should_drop_auto_tool_choice_for_reasoning(
    model: &Model,
    compat: &ResolvedCompat,
    tool_choice: Option<&ToolChoice>,
    reasoning: Option<&str>,
    disable_reasoning: bool,
) -> bool {
    matches!(tool_choice, Some(ToolChoice::Auto))
        && compat.disable_reasoning_on_tool_choice
        && model.reasoning
        && reasoning.is_some()
        && !disable_reasoning
}

/// TS `resolveOpenAICompatPolicy` for the chat-completions endpoint.
#[must_use]
pub fn resolve_policy(
    model: &Model,
    base: &ResolvedCompat,
    reasoning: Option<&str>,
    disable_reasoning: bool,
    tool_choice: Option<&ToolChoice>,
    has_tools: bool,
) -> CompatPolicy {
    let requested = reasoning;
    let model_supported = model.reasoning;
    let forced_suppresses = base.disable_reasoning_on_forced_tool_choice
        && base.supports_forced_tool_choice
        && is_forced(tool_choice);
    let any_tool_choice_suppresses =
        !forced_suppresses && base.disable_reasoning_on_tool_choice && tool_choice.is_some();
    let tools_suppress = !forced_suppresses
        && !any_tool_choice_suppresses
        && base.disable_reasoning_with_tools
        && has_tools;
    let requested_and_allowed = requested.is_some() && !disable_reasoning && model_supported;
    let conflict_disable_reason = if forced_suppresses {
        Some(DisableReason::ForcedToolChoice)
    } else if any_tool_choice_suppresses {
        Some(DisableReason::ToolChoice)
    } else if tools_suppress {
        Some(DisableReason::Tools)
    } else {
        None
    };
    let disable_reason = if disable_reasoning {
        Some(DisableReason::Caller)
    } else {
        conflict_disable_reason
    };
    let enabled_before_variant = requested_and_allowed && disable_reason.is_none();
    let base_wire_effort = if enabled_before_variant {
        requested.map(|e| map_reasoning_effort(model, base, e))
    } else {
        None
    };
    let disabled_by_none_effort = enabled_before_variant
        && base.reasoning_disable_mode == "zai-thinking-disabled"
        && base_wire_effort.as_deref() == Some("none");
    let enabled = enabled_before_variant && !disabled_by_none_effort;
    let compat: &ResolvedCompat = match (&base.when_thinking, enabled) {
        (Some(v), true) => v,
        _ => base,
    };
    let omit_reasoning_effort = compat.omit_reasoning_effort || !compat.supports_reasoning_effort;
    let disable_mode = compat.reasoning_disable_mode.clone();
    let mut wire_effort = if enabled {
        requested.map(|e| map_reasoning_effort(model, compat, e))
    } else {
        None
    };
    let disabled_without_request = model_supported
        && requested.is_none()
        && !disable_reasoning
        && is_implicit_disable_when_not_requested(&disable_mode);
    let disabled = (model_supported && disable_reason == Some(DisableReason::Caller))
        || conflict_disable_reason.is_some()
        || (model_supported && disabled_without_request)
        || disabled_by_none_effort;
    if disabled && compat.supports_reasoning_effort && !omit_reasoning_effort {
        if disable_mode == "none-effort" {
            wire_effort = Some("none".to_string());
        } else if disable_reason == Some(DisableReason::Caller)
            && requested.is_none()
            && disable_mode == "lowest-effort"
        {
            let min_effort = get_supported_thinking_levels(model)
                .into_iter()
                .find(|l| *l != ModelThinkingLevel::Off);
            if let Some(min) = min_effort {
                wire_effort = Some(map_reasoning_effort(model, compat, min.wire_name()));
            }
        }
    }
    CompatPolicy {
        compat: compat.clone(),
        reasoning: ReasoningPolicy {
            model_supported,
            supports_params: compat.supports_reasoning_params,
            requested_effort: requested.map(str::to_string),
            wire_effort,
            enabled,
            disabled,
            disable_reason: disable_reason.or(
                if disabled_without_request || disabled_by_none_effort {
                    Some(DisableReason::NotRequested)
                } else {
                    None
                },
            ),
            dialect: compat.thinking_format.clone(),
            disable_mode,
            omit_reasoning_effort,
        },
    }
}

fn merge_kwargs(params: &mut Map<String, Value>, entries: &[(&str, Value)]) {
    let mut kwargs = params
        .get("chat_template_kwargs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (k, v) in entries {
        kwargs.insert((*k).to_string(), v.clone());
    }
    params.insert("chat_template_kwargs".into(), Value::Object(kwargs));
}

/// TS `encodeChatCompletionsDisabledReasoning`.
fn encode_disabled_reasoning(params: &mut Map<String, Value>, mode: &str) {
    params.shift_remove("reasoning_effort");
    match mode {
        "none-effort" => {
            params.insert("reasoning_effort".into(), json!("none"));
        }
        "zai-thinking-disabled" => {
            params.insert("thinking".into(), json!({ "type": "disabled" }));
        }
        "qwen-enable-thinking-false" => {
            params.insert("enable_thinking".into(), json!(false));
        }
        "qwen-template-false" => merge_kwargs(params, &[("enable_thinking", json!(false))]),
        "chat-template-thinking-false" => merge_kwargs(params, &[("thinking", json!(false))]),
        "openrouter-enabled-false" | "cline-enabled-false" => {
            params.insert("reasoning".into(), json!({ "enabled": false }));
        }
        "venice-disable-thinking" => {
            let mut venice = params
                .get("venice_parameters")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            venice.insert("disable_thinking".into(), json!(true));
            params.insert("venice_parameters".into(), Value::Object(venice));
        }
        _ => {
            params.shift_remove("reasoning");
        }
    }
}

/// TS `applyChatCompletionsCompatPolicy`: the reasoning-related request parameters, written into
/// `params`.
pub fn apply_compat_policy(params: &mut Map<String, Value>, policy: &CompatPolicy) {
    let compat = &policy.compat;
    // `preserve_thinking` is a chat-template HISTORY knob; emitted before the reasoning branches
    // and on every early return (see the TS comment).
    if compat.qwen_preserve_thinking {
        if compat.thinking_format == "qwen" {
            params.insert("preserve_thinking".into(), json!(true));
        }
        merge_kwargs(params, &[("preserve_thinking", json!(true))]);
    }
    let reasoning = &policy.reasoning;
    if (!reasoning.model_supported && !reasoning.disabled) || !reasoning.supports_params {
        return;
    }
    if reasoning.enabled {
        let wire = reasoning.wire_effort.as_deref();
        match reasoning.disable_mode.as_str() {
            "zai-thinking-disabled" => {
                if wire == Some("none") {
                    encode_disabled_reasoning(params, &reasoning.disable_mode);
                    return;
                }
                if reasoning.dialect == "kimi" && wire.is_some() {
                    let mut t = Map::new();
                    t.insert("type".into(), json!("enabled"));
                    t.insert("effort".into(), json!(wire));
                    if let Some(keep) = &compat.thinking_keep {
                        t.insert("keep".into(), json!(keep));
                    }
                    params.insert("thinking".into(), Value::Object(t));
                    return;
                }
                let mut t = Map::new();
                t.insert("type".into(), json!("enabled"));
                if let Some(keep) = &compat.thinking_keep {
                    t.insert("keep".into(), json!(keep));
                }
                params.insert("thinking".into(), Value::Object(t));
                if compat.supports_reasoning_effort {
                    if let Some(w) = wire {
                        params.insert("reasoning_effort".into(), json!(w));
                    }
                }
            }
            "qwen-enable-thinking-false" => {
                params.insert("enable_thinking".into(), json!(true));
                if compat.qwen_template_reasoning_effort {
                    if let Some(w) = wire {
                        params.insert("reasoning_effort".into(), json!(w));
                        merge_kwargs(params, &[("reasoning_effort", json!(w))]);
                    }
                }
            }
            "qwen-template-false" => {
                let mut entries = vec![("enable_thinking", json!(true))];
                if compat.qwen_template_reasoning_effort {
                    if let Some(w) = wire {
                        entries.push(("reasoning_effort", json!(w)));
                    }
                }
                merge_kwargs(params, &entries);
            }
            "chat-template-thinking-false" => {
                let mut entries = vec![("thinking", json!(true))];
                if let Some(w) = wire {
                    entries.push(("reasoning_effort", json!(w)));
                }
                merge_kwargs(params, &entries);
            }
            "openrouter-enabled-false" => {
                if let Some(w) = wire {
                    params.insert("reasoning".into(), json!({ "effort": w }));
                }
            }
            _ => {
                if !reasoning.omit_reasoning_effort {
                    if let Some(w) = wire {
                        params.insert("reasoning_effort".into(), json!(w));
                    }
                }
            }
        }
        return;
    }
    if !reasoning.disabled {
        return;
    }
    if reasoning.disable_reason == Some(DisableReason::Caller)
        && reasoning.requested_effort.is_none()
        && matches!(
            reasoning.disable_mode.as_str(),
            "lowest-effort" | "none-effort"
        )
    {
        if let Some(w) = &reasoning.wire_effort {
            params.insert("reasoning_effort".into(), json!(w));
            return;
        }
    }
    encode_disabled_reasoning(params, &reasoning.disable_mode);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai_completions::convert::transform_for_request;
    use crate::providers::openai_completions::params::build_params;
    use crate::providers::openai_completions::OpenAICompletionsOptions;
    use crate::types::{
        CacheRetention, Context, Message, StreamOptions, UserMessage, UserMessageContent,
    };
    use serde_json::json;

    fn model(
        provider: &str,
        id: &str,
        base_url: &str,
        reasoning: bool,
        compat: Option<Value>,
    ) -> Model {
        let mut m: Model = serde_json::from_value(json!({
            "id": id, "name": id, "api": "openai-completions", "provider": provider,
            "baseUrl": base_url, "reasoning": reasoning, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 128_000, "maxTokens": 16_384,
            "compat": compat,
        }))
        .unwrap();
        crate::compat::materialize(&mut m, None).unwrap();
        m
    }

    fn body(model: &Model, options: OpenAICompletionsOptions, system: Option<&str>) -> Value {
        let context = Context {
            system_prompt: system.map(str::to_string),
            messages: vec![Message::User(UserMessage {
                content: UserMessageContent::Text("hi".into()),
                timestamp: 1,
                rest: Map::default(),
            })],
            tools: None,
        };
        let compat = resolve_compat(model).unwrap();
        let transformed = transform_for_request(model, &context, &compat);
        build_params(
            model,
            &context,
            &transformed,
            Some(&options),
            &compat,
            CacheRetention::Short,
        )
        .to_value()
    }

    /// Regression: `400 role 'developer' is not allowed` on kimi-code. Only OpenAI/Azure hosts
    /// take the developer role; every other host — including one the resolver has never seen —
    /// gets `system`.
    #[test]
    fn only_openai_and_azure_hosts_use_the_developer_role() {
        for (provider, url, want) in [
            ("openai", "https://api.openai.com/v1", "developer"),
            ("azure", "https://x.openai.azure.com/openai", "developer"),
            ("groq", "https://api.groq.com/openai/v1", "system"),
            ("my-proxy", "http://localhost:1234/v1", "system"),
            ("kimi-code", "https://api.kimi.com/coding/v1", "system"),
            ("zai", "https://api.z.ai/api/coding/paas/v4", "system"),
        ] {
            let m = model(provider, "gpt-5.5", url, true, None);
            let b = body(&m, OpenAICompletionsOptions::default(), Some("SYS"));
            assert_eq!(b["messages"][0]["role"], want, "{provider}");
        }
    }

    /// An explicit `supportsDeveloperRole: true` override (models.yml `compat:`) wins over host
    /// detection, an unknown override key is ignored, and a spelling this build never saw does
    /// not discard the rest of the record.
    #[test]
    fn sparse_overrides_apply_over_the_resolved_record() {
        let m = model(
            "my-proxy",
            "gpt-5.5",
            "https://llm.corp.example/v1",
            true,
            Some(
                json!({"supportsDeveloperRole": true, "thinkingFormat": "some-future-dialect", "bogus": 1}),
            ),
        );
        let c = resolve_compat(&m).unwrap();
        assert!(c.supports_developer_role);
        assert_eq!(c.thinking_format, "some-future-dialect");
        assert_eq!(
            body(&m, OpenAICompletionsOptions::default(), Some("S"))["messages"][0]["role"],
            "developer"
        );
    }

    /// `supportsMultipleSystemMessages: false` hosts coalesce system prompt blocks into one
    /// `\n\n`-joined message; other hosts emit one message per block.
    #[test]
    fn system_prompt_blocks_follow_supports_multiple_system_messages() {
        let blocks = ["static prefix", "dynamic suffix"];
        let context = Context {
            system_prompt: None,
            messages: vec![],
            tools: None,
        };
        for (provider, url, multiple) in [
            ("openai", "https://api.openai.com/v1", true),
            ("kimi-code", "https://api.kimi.com/coding/v1", false),
            (
                "alibaba-coding-plan",
                "https://coding.dashscope.aliyuncs.com/v1",
                false,
            ),
        ] {
            let m = model(provider, "some-model", url, false, None);
            let compat = resolve_compat(&m).unwrap();
            assert_eq!(
                compat.supports_multiple_system_messages.unwrap_or(true),
                multiple,
                "{provider}"
            );
            let transformed = transform_for_request(&m, &context, &compat);
            let messages = crate::providers::openai_completions::convert::convert_messages(
                &m,
                &blocks,
                &transformed,
                &compat,
            );
            let wire = serde_json::to_value(&messages).unwrap();
            let system: Vec<&Value> = wire
                .as_array()
                .unwrap()
                .iter()
                .filter(|m| m["role"] == "system" || m["role"] == "developer")
                .collect();
            if multiple {
                assert_eq!(system.len(), 2, "{provider}");
            } else {
                assert_eq!(system.len(), 1, "{provider}");
                assert_eq!(system[0]["content"], "static prefix\n\ndynamic suffix");
            }
        }
    }

    /// The Kimi dialect sends `thinking: {type, effort}` with the resolver's effort map; a
    /// non-reasoning request disables thinking explicitly.
    #[test]
    fn kimi_dialect_body() {
        // The authored `compatConfig` the catalog rows of kimi-code carry.
        let m = model(
            "kimi-code",
            "k3",
            "https://api.kimi.com/coding/v1",
            true,
            Some(
                json!({"thinkingFormat": "kimi", "reasoningContentField": "reasoning_content", "supportsDeveloperRole": false}),
            ),
        );
        let mut o = OpenAICompletionsOptions::from_base(StreamOptions::default());
        o.reasoning_effort = Some(ModelThinkingLevel::Medium);
        let b = body(&m, o, None);
        assert_eq!(b["thinking"], json!({"type": "enabled", "effort": "high"}));
        assert_eq!(b["max_tokens"], 16_384, "kimi always sends a cap");
        let mut off = OpenAICompletionsOptions::from_base(StreamOptions::default());
        off.reasoning_enabled = Some(false);
        assert_eq!(body(&m, off, None)["thinking"], json!({"type": "disabled"}));
    }
}
