//! `OpenAI` Completions request params assembly (TS `buildParams`,
//! `ai/src/providers/openai-completions.ts`): every compat-dependent decision reads the resolved
//! compat record ([`ResolvedCompat`], the output of the shared policy resolver) and the
//! reasoning policy ([`resolve_policy`]); nothing here detects a host or model family.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::providers::openai_completions::compat::{
    apply_compat_policy, is_forced, resolve_policy, should_drop_auto_tool_choice_for_reasoning,
    DisableReason, ResolvedCompat,
};
use crate::providers::openai_completions::convert::{
    convert_message_range, convert_tools, push_system_messages,
};
use crate::providers::openai_completions::has_tool_history;
use crate::providers::openai_completions::{OpenAICompletionsOptions, ToolChoice};
use crate::types::{CacheRetention, Context, Message, Model};
use crate::utils_inner::wire_json::{obj, to_json_string, Wire, WireObject};

/// How the `messages` array reaches the params tree in [`build_params_streaming`].
pub(crate) enum MsgsInput {
    /// Convert from `context`/`transformed` as usual.
    Convert,
    /// Wire-fragment cache hit: splice the pre-serialized array `prefix` (system blocks included,
    /// covering `count` transformed messages) and convert only `transformed[count..]`, carried
    /// from the prefix's `last_role`. Skips the Anthropic cache-control post-pass, which the
    /// cache never stores fragments for.
    Raw {
        prefix: Arc<serde_json::value::RawValue>,
        count: usize,
        last_role: Option<&'static str>,
    },
}

/// How the `tools` array reaches the params tree in [`build_params_streaming`].
pub(crate) enum ToolsInput {
    /// Convert from `context.tools` as usual (including the named-choice narrowing).
    Convert,
    /// Splice this pre-serialized array (`None` = the request carries no `tools` key at all).
    Raw(Option<Arc<serde_json::value::RawValue>>),
}

/// Serialized fragments a [`build_params_streaming`] call built, for the wire-fragment cache to
/// store verbatim and splice into later requests of the same session.
pub(crate) struct BuiltFrags {
    /// The serialized `messages` array (system blocks included).
    pub messages: Arc<serde_json::value::RawValue>,
    /// Message count the prefix covers; `last_role` after it (a continuation conversion's carry).
    pub msg_count: usize,
    pub last_role: Option<&'static str>,
    /// The serialized `tools` array, or `None` when the request omits the key.
    pub tools: Option<Arc<serde_json::value::RawValue>>,
}

/// TS `OPENAI_MAX_OUTPUT_TOKENS`.
const OPENAI_MAX_OUTPUT_TOKENS: u64 = 64_000;

/// TS `applyWireModelIdTransform`: how the catalog id maps onto the wire model id.
fn wire_model_id(id: &str, mode: Option<&str>) -> String {
    fn dot_to_p(id: &str) -> String {
        let bytes: Vec<char> = id.chars().collect();
        let mut out = String::with_capacity(id.len());
        for (i, c) in bytes.iter().enumerate() {
            let between_digits = *c == '.'
                && i > 0
                && bytes[i - 1].is_ascii_digit()
                && bytes.get(i + 1).is_some_and(char::is_ascii_digit);
            out.push(if between_digits { 'p' } else { *c });
        }
        out
    }
    match mode {
        Some("cline-pass") => {
            if id.starts_with("cline-pass/") {
                id.to_string()
            } else {
                format!("cline-pass/{id}")
            }
        }
        Some("firepass") => {
            let stripped = id.strip_prefix("accounts/fireworks/routers/").unwrap_or(id);
            format!("accounts/fireworks/routers/{}", dot_to_p(stripped))
        }
        Some("fireworks") => {
            let stripped = id.strip_prefix("accounts/fireworks/models/").unwrap_or(id);
            format!("accounts/fireworks/models/{}", dot_to_p(stripped))
        }
        _ => id.to_string(),
    }
}

/// TS `resolveOpenAIOutputTokenParam` for Chat Completions: the output cap and the wire field it
/// rides on, or `None` when the request carries no cap.
fn output_token_param(
    model: &Model,
    compat: &ResolvedCompat,
    options: &OpenAICompletionsOptions,
) -> Option<(&'static str, u64)> {
    if model.omit_max_output_tokens == Some(true) {
        return None;
    }
    let model_max = (model.max_tokens > 0).then_some(model.max_tokens);
    let requested = options.base.max_tokens.or_else(|| {
        compat
            .always_send_max_tokens
            .then(|| model_max.unwrap_or(OPENAI_MAX_OUTPUT_TOKENS))
    })?;
    let explicit = options.max_tokens_explicit;
    if compat.is_open_router_host && !compat.always_send_max_tokens && !explicit {
        return None;
    }
    let provider_clamp = if model.provider == "cline-pass"
        || (compat.thinking_format == "zai" && compat.zai_reasoning_effort_dialect)
        || compat.clamp_output_to_model_max
    {
        model_max.unwrap_or(OPENAI_MAX_OUTPUT_TOKENS)
    } else {
        OPENAI_MAX_OUTPUT_TOKENS
    };
    let value = requested
        .min(model_max.unwrap_or(u64::MAX))
        .min(provider_clamp);
    if value == 0 {
        return None;
    }
    Some((
        if compat.max_tokens_is_legacy_field() {
            "max_tokens"
        } else {
            "max_completion_tokens"
        },
        value,
    ))
}

/// TS `getOpenAIPromptCacheKey`: the cache identity, normalized to the 64-char wire limit.
fn prompt_cache_key(
    options: &OpenAICompletionsOptions,
    cache_retention: CacheRetention,
) -> Option<String> {
    if cache_retention == CacheRetention::None {
        return None;
    }
    let value = options
        .base
        .session_id
        .as_deref()
        .filter(|v| !v.is_empty())?;
    normalize_stable_id(value, 64, "pc_")
}

/// TS `normalizeOpenAIStableId`: identities past the wire limit hash down to `{prefix}{base36}`.
pub(crate) fn normalize_stable_id(
    value: &str,
    max_len: usize,
    hash_prefix: &str,
) -> Option<String> {
    if value.is_empty() {
        return None;
    }
    if value.chars().count() <= max_len {
        return Some(value.to_string());
    }
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(value.as_bytes());
    let n = u64::from_be_bytes(digest[..8].try_into().expect("8 bytes"));
    Some(format!("{hash_prefix}{}", to_base36(n)))
}

fn to_base36(mut n: u64) -> String {
    if n == 0 {
        return "0".into();
    }
    let mut out = Vec::new();
    while n > 0 {
        out.push(b"0123456789abcdefghijklmnopqrstuvwxyz"[(n % 36) as usize]);
        n /= 36;
    }
    out.reverse();
    String::from_utf8(out).expect("ascii")
}

/// TS `shouldSendServiceTier` for the chat-completions transports.
fn should_send_service_tier(model: &Model, tier: crate::types::ServiceTier) -> bool {
    use crate::types::ServiceTier;
    if tier == ServiceTier::Auto {
        return false;
    }
    if model.provider == "openai" {
        return true;
    }
    model.provider == "openrouter"
        && matches!(
            tier,
            ServiceTier::Flex | ServiceTier::Scale | ServiceTier::Priority
        )
}

/// The request params for `context`, whose history was transformed into `transformed`
/// ([`transform_for_request`](crate::providers::openai_completions::convert::transform_for_request)).
/// Messages and tools borrow from the context and the transformed history instead of being copied
/// into a `Value` tree: the body is serialized straight from them.
// Long by design: mirrors the provider's stream shape.
#[allow(clippy::too_many_lines)]
pub(crate) fn build_params<'a>(
    model: &Model,
    context: &'a Context,
    transformed: &'a [Message],
    options: Option<&OpenAICompletionsOptions>,
    compat: &ResolvedCompat,
    cache_retention: CacheRetention,
) -> Wire<'a> {
    build_params_streaming(
        model,
        context,
        transformed,
        options,
        compat,
        cache_retention,
        MsgsInput::Convert,
        ToolsInput::Convert,
        None,
    )
}

/// [`build_params`] with the wire-fragment cache hooks the stream layer drives: `msgs_in` /
/// `tools_in` replace the two conversions with verbatim splices on a cache hit, and `frags`, when
/// given, receives the serialized fragments a conversion produced (the cache stores them). Every
/// policy decision is made here exactly once, so cached and uncached requests emit identical
/// bytes.
pub(crate) fn build_params_streaming<'a>(
    model: &Model,
    context: &'a Context,
    transformed: &'a [Message],
    options: Option<&OpenAICompletionsOptions>,
    compat: &ResolvedCompat,
    cache_retention: CacheRetention,
    msgs_in: MsgsInput,
    tools_in: ToolsInput,
    mut frags: Option<&mut BuiltFrags>,
) -> Wire<'a> {
    let options = options.cloned().unwrap_or_default();
    let reasoning = options.reasoning_effort.map(|level| level.wire_name());
    // `reasoning_enabled == Some(false)` is the TS `disableReasoning` (an explicit off).
    let disable_reasoning = options.reasoning_enabled == Some(false);
    let context_tools = context.tools.as_deref().filter(|t| !t.is_empty());

    let initial = resolve_policy(
        model,
        compat,
        reasoning,
        disable_reasoning,
        options.tool_choice.as_ref(),
        context_tools.is_some(),
    );
    let initial_compat = &initial.compat;

    let mut params = WireObject::new();
    params.insert(
        "model",
        json!(wire_model_id(
            model.request_model_id.as_deref().unwrap_or(&model.id),
            compat.wire_model_id_mode.as_deref()
        )),
    );
    params.insert("messages", Wire::Array(Vec::new()));
    params.insert("stream", json!(true));

    if initial_compat.supports_usage_in_streaming != Some(false) {
        params.insert("stream_options", json!({ "include_usage": true }));
    }
    if initial_compat.supports_store {
        params.insert("store", json!(false));
    }
    // OpenAI proprietary reasoning models reject explicit sampling params with a 400 (#5606).
    if initial_compat.supports_sampling_params {
        if let Some(temperature) = options.base.temperature {
            params.insert("temperature", json!(temperature));
        }
    }
    if let Some(service_tier) = options.base.service_tier {
        if should_send_service_tier(model, service_tier) {
            params.insert(
                "service_tier",
                serde_json::to_value(service_tier).unwrap_or(Value::Null),
            );
        }
    }

    // `Some(None)` = splice mode with no `tools` key; `Some(Some(_))` = splice the array verbatim.
    let tools_raw_in = match &tools_in {
        ToolsInput::Raw(raw) => Some(raw.clone()),
        ToolsInput::Convert => None,
    };
    let mut tools: Option<Vec<(&'a str, Wire<'a>)>> = None;
    if tools_raw_in.is_none() {
        if let Some(context_tools) = context_tools {
            tools = Some(convert_tools(context_tools, initial_compat).tools);
        } else if context.tools.is_none() && has_tool_history(&context.messages) {
            // Anthropic (via LiteLLM/proxy) requires the `tools` param when the conversation contains
            // tool_calls/tool_results, even when no tools are offered this turn. An explicit empty
            // list means the caller opted out of tools: emit nothing.
            tools = Some(Vec::new());
        }
    }

    let mut tool_choice: Option<ToolChoice> = None;
    if let Some(choice) = options.tool_choice.clone() {
        if initial_compat.supports_tool_choice {
            tool_choice = Some(choice);
        }
    }
    let forced_tool_name: Option<String> = match &tool_choice {
        Some(ToolChoice::Function { name }) => Some(name.clone()),
        _ => None,
    };
    let tools_contain = |tools: &Option<Vec<(&'a str, Wire<'a>)>>, name: &str| {
        tools
            .as_ref()
            .is_some_and(|t| t.iter().any(|(n, _)| *n == name))
    };
    if matches!(tool_choice, Some(ToolChoice::Function { .. }))
        && !initial_compat.supports_named_tool_choice
    {
        // String-only hosts (llama.cpp, LM Studio) accept only none/auto/required: narrow the
        // advertised tools to the forced one so "required" still enforces that call.
        if let Some(name) = &forced_tool_name {
            if tools_contain(&tools, name) {
                if let Some(t) = tools.as_mut() {
                    t.retain(|(n, _)| n == name);
                }
            }
        }
        tool_choice = Some(ToolChoice::Required);
    }
    if let Some(name) = &forced_tool_name {
        let k3_reasoning = initial_compat.native_kimi_k3_reasoning
            && model.reasoning
            && reasoning.is_some()
            && !disable_reasoning;
        if tools_contain(&tools, name) && k3_reasoning {
            // Native K3 reasoning is incompatible with selecting a specific function.
            tool_choice = Some(ToolChoice::Required);
        }
    }
    if is_forced(tool_choice.as_ref()) && !initial_compat.supports_forced_tool_choice {
        // Some thinking-required models reject forced `tool_choice` while still accepting tools.
        tool_choice = Some(ToolChoice::Auto);
    }
    // The named-choice narrowing below only runs under `ToolChoice::Function`, which the
    // wire-fragment cache never splices (`tools_raw_in.is_some()` implies a non-Function choice),
    // so it is a no-op in splice mode.
    let tools_empty = match &tools_raw_in {
        Some(raw) => raw.is_none(),
        None => tools.as_ref().is_none_or(Vec::is_empty),
    };
    if tools_empty
        && (matches!(tool_choice, Some(ToolChoice::None)) || is_forced(tool_choice.as_ref()))
    {
        // `tool_choice: "none"`/forced with no tools to gate is redundant (and trips LiteLLM → Bedrock).
        tool_choice = None;
    }
    if let Some(name) = &forced_tool_name {
        if !tools_contain(&tools, name) {
            // A forced named choice is only valid when the same request offers that function.
            tool_choice = None;
        }
    }
    if should_drop_auto_tool_choice_for_reasoning(
        model,
        initial_compat,
        tool_choice.as_ref(),
        reasoning,
        disable_reasoning,
    ) {
        tool_choice = None;
    }

    let has_tools = match &tools_raw_in {
        Some(raw) => raw.is_some(),
        None => tools.as_ref().is_some_and(|t| !t.is_empty()),
    };
    let final_policy = resolve_policy(
        model,
        compat,
        reasoning,
        disable_reasoning,
        tool_choice.as_ref(),
        has_tools,
    );
    let final_compat = &final_policy.compat;

    let messages_wire = match msgs_in {
        MsgsInput::Raw {
            prefix,
            count,
            last_role,
        } => {
            if transformed.len() == count {
                Wire::Raw(prefix)
            } else {
                let mut tail: Vec<Wire<'a>> = Vec::new();
                convert_message_range(
                    model,
                    transformed,
                    final_compat,
                    count,
                    last_role,
                    &mut tail,
                );
                if tail.is_empty() {
                    Wire::Raw(prefix)
                } else {
                    Wire::Raw(splice_messages_prefix(&prefix, &tail))
                }
            }
        }
        MsgsInput::Convert => {
            let mut wires: Vec<Wire<'a>> = Vec::new();
            push_system_messages(
                model,
                context.system_prompt.as_deref().as_slice(),
                final_compat,
                &mut wires,
            );
            let last_role =
                convert_message_range(model, transformed, final_compat, 0, None, &mut wires);
            maybe_add_anthropic_cache_control(final_compat, &mut wires);
            if let Some(f) = frags.as_mut() {
                // The caller caches the fragment: report it and splice the bytes we already
                // serialized (identical output, and the body write becomes a verbatim copy).
                let raw = serde_json::value::RawValue::from_string(to_json_string(&wires))
                    .map(Arc::<serde_json::value::RawValue>::from)
                    .expect("serialized messages are valid JSON");
                f.messages = raw.clone();
                f.msg_count = transformed.len();
                f.last_role = last_role;
                Wire::Raw(raw)
            } else {
                Wire::Array(wires)
            }
        }
    };
    params.insert("messages", messages_wire);

    if let Some((field, value)) = output_token_param(model, final_compat, &options) {
        params.insert(field, json!(value));
    }

    // Z.AI/GLM-5.2 reasoning-effort dialect: stream tool-call arguments incrementally.
    let zai_tool_stream = final_compat.thinking_format == "zai"
        && final_compat.zai_reasoning_effort_dialect
        && final_compat.supports_reasoning_effort
        && has_tools;
    if zai_tool_stream {
        params.insert("tool_stream", json!(true));
    }

    match tools_raw_in {
        Some(raw) => {
            if let Some(raw) = raw {
                params.insert("tools", Wire::Raw(raw));
            }
        }
        None => {
            if let Some(tools) = tools {
                let array: Vec<Wire<'a>> = tools.into_iter().map(|(_, w)| w).collect();
                if let Some(f) = frags.as_mut() {
                    let raw = serde_json::value::RawValue::from_string(to_json_string(&array))
                        .map(Arc::<serde_json::value::RawValue>::from)
                        .expect("serialized tools are valid JSON");
                    f.tools = Some(raw.clone());
                    params.insert("tools", Wire::Raw(raw));
                } else {
                    params.insert("tools", Wire::Array(array));
                }
            } else if let Some(f) = frags.as_mut() {
                f.tools = None;
            }
        }
    }
    if let Some(choice) = &tool_choice {
        params.insert("tool_choice", choice.to_json());
    }

    // Reasoning parameters (TS `applyChatCompletionsReasoningParams` + the OpenRouter-Kimi drop).
    let reasoning_policy = resolve_policy(
        model,
        compat,
        reasoning,
        disable_reasoning,
        tool_choice.as_ref(),
        has_tools,
    );
    let mut reasoning_params = Map::new();
    apply_compat_policy(&mut reasoning_params, &reasoning_policy);
    if final_policy.reasoning.disable_reason == Some(DisableReason::ForcedToolChoice)
        && final_policy.reasoning.disable_mode == "openrouter-enabled-false"
        && final_policy.compat.is_open_router_host
    {
        reasoning_params.shift_remove("reasoning");
    }
    for (key, value) in reasoning_params {
        params.insert(key, Wire::Owned(value));
    }

    // Gateway routing: OpenRouter `provider`; Vercel `providerOptions.gateway`.
    if final_compat.is_open_router_host {
        if let Some(routing) = &final_compat.open_router_routing {
            params.insert("provider", Wire::Owned(routing.clone()));
        }
    }
    if final_compat.is_vercel_gateway_host {
        if let Some(routing) = final_compat
            .vercel_gateway_routing
            .as_ref()
            .and_then(Value::as_object)
        {
            let cache_enabled = cache_retention != CacheRetention::None;
            let mut gateway = Map::new();
            if let Some(only) = routing.get("only").filter(|v| !v.is_null()) {
                gateway.insert("only".into(), only.clone());
            }
            if let Some(order) = routing.get("order").filter(|v| !v.is_null()) {
                gateway.insert("order".into(), order.clone());
            }
            if cache_enabled {
                if let Some(caching) = routing.get("caching").filter(|v| !v.is_null()) {
                    gateway.insert("caching".into(), caching.clone());
                }
            }
            if !gateway.is_empty() {
                params.insert(
                    "providerOptions",
                    json!({ "gateway": Value::Object(gateway) }),
                );
            }
        }
    }

    // `extraBody` merges last; an encoded Venice disable signal outranks static venice_parameters.
    if let Some(extra) = &final_compat.extra_body {
        let encoded_venice = params.get("venice_parameters").and_then(|w| match w {
            Wire::Owned(Value::Object(o)) => Some(o.clone()),
            _ => None,
        });
        for (k, v) in extra {
            params.insert(k.clone(), Wire::Owned(v.clone()));
        }
        if let Some(encoded) = encoded_venice {
            if encoded.get("disable_thinking") == Some(&Value::Bool(true)) {
                let mut merged = extra
                    .get("venice_parameters")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                for (k, v) in encoded {
                    merged.insert(k, v);
                }
                params.insert("venice_parameters", Wire::Owned(Value::Object(merged)));
            }
        }
        if final_compat.drop_thinking_when_reasoning_effort
            && params.contains_key("reasoning_effort")
        {
            params.remove("thinking");
        }
    }

    if final_compat.supports_prompt_cache_key {
        if let Some(key) = prompt_cache_key(&options, cache_retention) {
            params.insert("prompt_cache_key", Value::String(key));
        }
    }

    Wire::Object(params)
}

/// `prefix` (a serialized JSON array) with `tail` appended as elements: `"[" + inner(prefix) +
/// "," + inner(tail) + "]"`. The pieces come from `serde_json`, so the join is valid JSON; the
/// one re-validation pass (`RawValue::from_string`) is far cheaper than re-serializing the prefix.
fn splice_messages_prefix(
    prefix: &serde_json::value::RawValue,
    tail: &[Wire<'_>],
) -> Arc<serde_json::value::RawValue> {
    let prefix = prefix.get();
    let inner = &prefix[1..prefix.len() - 1];
    let tail_json = to_json_string(tail);
    let tail_inner = &tail_json[1..tail_json.len() - 1];
    let mut joined = String::with_capacity(inner.len() + tail_inner.len() + 3);
    joined.push('[');
    joined.push_str(inner);
    joined.push(',');
    joined.push_str(tail_inner);
    joined.push(']');
    serde_json::value::RawValue::from_string(joined)
        .map(Arc::<serde_json::value::RawValue>::from)
        .expect("spliced messages are valid JSON")
}

/// TS `maybeAddAnthropicCacheControl`: one `cache_control` breakpoint on the last non-empty text
/// of the final user/assistant/developer message.
fn maybe_add_anthropic_cache_control(compat: &ResolvedCompat, messages: &mut [Wire<'_>]) {
    if compat.cache_control_format.as_deref() != Some("anthropic") {
        return;
    }
    let ephemeral = || json!({ "type": "ephemeral" });
    for message in messages.iter_mut().rev() {
        let role = message.get("role").and_then(Wire::as_str).unwrap_or("");
        if role != "user" && role != "assistant" && role != "developer" {
            continue;
        }
        let Some(content) = message.as_object_mut().and_then(|m| m.get_mut("content")) else {
            continue;
        };
        if let Some(is_empty) = content.string_is_empty() {
            let blank = content.as_str().is_some_and(|t| t.trim().is_empty()) || is_empty;
            if blank {
                continue;
            }
            let text = std::mem::replace(content, Wire::Null);
            *content = Wire::Array(vec![obj([
                ("type", Wire::str("text")),
                ("text", text),
                ("cache_control", Wire::Owned(ephemeral())),
            ])]);
            return;
        }
        if let Some(parts) = content.as_array_mut() {
            for part in parts.iter_mut().rev() {
                if part.get("type").and_then(Wire::as_str) != Some("text") {
                    continue;
                }
                let non_blank = part
                    .get("text")
                    .and_then(Wire::as_str)
                    .is_some_and(|t| !t.trim().is_empty());
                if non_blank {
                    part.as_object_mut()
                        .expect("text parts are objects")
                        .insert("cache_control", Wire::Owned(ephemeral()));
                    return;
                }
            }
        }
    }
}

pub(crate) fn build_headers(
    model: &Model,
    api_key: &str,
    options_headers: Option<&HashMap<String, String>>,
    cache_session_id: Option<&str>,
    compat: &ResolvedCompat,
    conversation_id: Option<&str>,
) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = Vec::new();
    for (name, value) in model.headers.iter().flatten() {
        headers.push((name.clone(), value.clone()));
    }

    if let Some(options_headers) = options_headers {
        for (name, value) in options_headers {
            headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
            headers.push((name.clone(), value.clone()));
        }
    }

    // TS `setHeaderIfAbsent`: the compat-named session headers never override explicit ones.
    let mut set_if_absent = |name: &str, value: &str| {
        if !headers
            .iter()
            .any(|(existing, _)| existing.eq_ignore_ascii_case(name))
        {
            headers.push((name.to_string(), value.to_string()));
        }
    };
    if let (Some(name), Some(session_id)) = (
        compat.prompt_cache_session_header.as_deref(),
        cache_session_id,
    ) {
        set_if_absent(name, session_id);
    }
    if let (Some(name), Some(session_id)) = (compat.session_header.as_deref(), conversation_id) {
        set_if_absent(name, session_id);
    }

    // Keyless providers (`auth: none`) carry an empty key: TS sends no `Authorization` header.
    if !api_key.is_empty() {
        headers.insert(0, ("Authorization".into(), format!("Bearer {api_key}")));
    }
    headers
}
