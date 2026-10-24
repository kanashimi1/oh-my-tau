//! Per-session wire-fragment cache for the Chat Completions request body (the stream layer's hot
//! path). Converting the tool schemas and re-serializing the whole message history dominates the
//! per-turn wire cost once a session grows: the tools JSON (~15 schemas) and the messages prefix
//! are byte-identical between turns of one session, appends only. This cache stores the
//! serialized fragments and splices them verbatim, converting and serializing only the messages
//! added since the previous request.
//!
//! Safety: a hit is served only when the full cache key matches (model, compat record, system
//! prompt, tools, reasoning/tool-choice options) and the transformed history still begins with
//! the exact prefix snapshot stored at the previous request (deep equality). A miss (different
//! key, shorter/compacted history, or a mutated prefix) falls back to the full conversion and
//! replaces the slot, so the cache can never emit stale bytes.
//!
//! Disabled (`OMP_RS_WIRE_FRAG_CACHE=0`), or bypassed per request for formats the cache cannot
//! serve: Anthropic `cache_control` (mutates the last message), named `ToolChoice::Function`
//! (narrows the tools array), and requests carrying a payload hook (handled by the caller).

use std::sync::{Arc, Mutex, OnceLock};

use serde_json::value::RawValue;

use crate::providers::openai_completions::compat::ResolvedCompat;
use crate::providers::openai_completions::params::{
    build_params, build_params_streaming, BuiltFrags, MsgsInput, ToolsInput,
};
use crate::providers::openai_completions::{OpenAICompletionsOptions, ToolChoice};
use crate::types::{CacheRetention, Context, Message, Model, Tool};
use crate::utils_inner::wire_json::Wire;

/// Everything the serialized fragments depend on outside the message history. `Model` and
/// `ResolvedCompat` are compared whole, so a field this module does not know about still
/// invalidates the cache.
struct CacheKey {
    model: Model,
    compat: ResolvedCompat,
    system_prompt: Option<String>,
    tools: Option<Vec<Tool>>,
    reasoning: Option<&'static str>,
    disable_reasoning: bool,
    tool_choice: Option<ToolChoice>,
}

/// The serialized `messages` array (system blocks included) covering `count` transformed
/// messages, plus the snapshot needed to prove the next history extends it unchanged.
struct MsgsFrag {
    prefix: Arc<RawValue>,
    transformed_prefix: Vec<Message>,
    count: usize,
    last_role: Option<&'static str>,
}

struct WireFragCache {
    key: CacheKey,
    /// `None` when the request carried no tools (fragment not cached; converted per request).
    /// `Some(None)` when the request omits the `tools` key; `Some(Some(_))` the array text.
    tools: Option<Option<Arc<RawValue>>>,
    messages: MsgsFrag,
}

/// One process hosts a handful of concurrent request streams (the rpc turn itself plus side
/// requests); a single slot would thrash between them, so keep a few and evict round-robin.
struct CacheSet {
    slots: Vec<WireFragCache>,
    next: usize,
}

const SLOTS: usize = 4;

static CACHE: Mutex<CacheSet> = Mutex::new(CacheSet {
    slots: Vec::new(),
    next: 0,
});

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("OMP_RS_WIRE_FRAG_CACHE").is_none_or(|v| v != "0"))
}

/// Whether two messages convert to identical wire bytes. [`transform_for_request`] stamps
/// synthetic "No result provided" tool results with the current time, and the agent layer may
/// restamp real ones; `timestamp` never reaches the request body, so it is the one field
/// excluded here. Every other field of every `Message` variant participates — adding a field to
/// a message struct or a new variant requires updating this comparison, or the cache could
/// serve a stale prefix.
fn wire_relevant_eq(a: &Message, b: &Message) -> bool {
    use crate::types::{ToolResultMessage, UserMessage};
    fn user_eq(a: &UserMessage, b: &UserMessage) -> bool {
        a.content == b.content && a.rest == b.rest
    }
    fn tool_result_eq(a: &ToolResultMessage, b: &ToolResultMessage) -> bool {
        a.tool_call_id == b.tool_call_id
            && a.tool_name == b.tool_name
            && a.content == b.content
            && a.details == b.details
            && a.is_error == b.is_error
            && a.rest == b.rest
    }
    match (a, b) {
        (Message::User(a), Message::User(b)) | (Message::Developer(a), Message::Developer(b)) => {
            user_eq(a, b)
        }
        (Message::Assistant(a), Message::Assistant(b)) => {
            a.content == b.content
                && a.api == b.api
                && a.provider == b.provider
                && a.model == b.model
                && a.response_model == b.response_model
                && a.response_id == b.response_id
                && a.diagnostics == b.diagnostics
                && a.usage == b.usage
                && a.stop_reason == b.stop_reason
                && a.stop_reason_raw == b.stop_reason_raw
                && a.error_message == b.error_message
                && a.error_status == b.error_status
                && a.rest == b.rest
        }
        (Message::ToolResult(a), Message::ToolResult(b)) => tool_result_eq(a, b),
        _ => false,
    }
}

/// [`build_params`] with the per-session fragment cache; returns byte-identical params.
pub(crate) fn build_params_cached<'a>(
    model: &Model,
    context: &'a Context,
    transformed: &'a [Message],
    options: Option<&OpenAICompletionsOptions>,
    compat: &ResolvedCompat,
    cache_retention: CacheRetention,
) -> Wire<'a> {
    let opts = options.cloned().unwrap_or_default();
    let reasoning = opts.reasoning_effort.map(|level| level.wire_name());
    let disable_reasoning = opts.reasoning_enabled == Some(false);
    let tool_choice = opts.tool_choice.clone();

    let cacheable = enabled()
        && compat.cache_control_format.as_deref() != Some("anthropic")
        && !matches!(tool_choice, Some(ToolChoice::Function { .. }));
    if !cacheable {
        return build_params(
            model,
            context,
            transformed,
            options,
            compat,
            cache_retention,
        );
    }

    let key = CacheKey {
        model: model.clone(),
        compat: compat.clone(),
        system_prompt: context.system_prompt.clone(),
        tools: context.tools.clone(),
        reasoning,
        disable_reasoning,
        tool_choice: tool_choice.clone(),
    };
    let hit = {
        let cache = CACHE.lock().expect("wire frag cache");
        cache.slots.iter().find_map(|cached| {
            let key = &cached.key;
            if key.model != *model
                || key.compat != *compat
                || key.system_prompt != context.system_prompt
                || key.tools != context.tools
                || key.reasoning != reasoning
                || key.disable_reasoning != disable_reasoning
                || key.tool_choice != tool_choice
            {
                return None;
            }
            let frag = &cached.messages;
            if transformed.len() < frag.count
                || !frag
                    .transformed_prefix
                    .iter()
                    .zip(&transformed[..frag.count])
                    .all(|(a, b)| wire_relevant_eq(a, b))
            {
                return None;
            }
            Some((
                cached.tools.clone(),
                frag.prefix.clone(),
                frag.count,
                frag.last_role,
            ))
        })
    };

    match hit {
        Some((tools_frag, prefix, count, last_role)) => build_params_streaming(
            model,
            context,
            transformed,
            options,
            compat,
            cache_retention,
            MsgsInput::Raw {
                prefix,
                count,
                last_role,
            },
            match tools_frag {
                Some(raw) => ToolsInput::Raw(raw),
                None => ToolsInput::Convert,
            },
            None,
        ),
        None => {
            let mut frags = BuiltFrags {
                // Placeholders until the conversion below fills them; "[]" is always valid JSON.
                messages: RawValue::from_string("[]".to_string())
                    .map(Arc::<serde_json::value::RawValue>::from)
                    .expect("valid JSON"),
                msg_count: 0,
                last_role: None,
                tools: None,
            };
            let wire = build_params_streaming(
                model,
                context,
                transformed,
                options,
                compat,
                cache_retention,
                MsgsInput::Convert,
                ToolsInput::Convert,
                Some(&mut frags),
            );
            let entry = WireFragCache {
                key,
                // The tools fragment is only valid while the tools option set is the cached one;
                // a `None` tools context converts cheaply, so it is never cached.
                tools: if context
                    .tools
                    .as_deref()
                    .is_some_and(|tools| !tools.is_empty())
                {
                    Some(frags.tools)
                } else {
                    None
                },
                messages: MsgsFrag {
                    prefix: frags.messages,
                    transformed_prefix: transformed[..frags.msg_count].to_vec(),
                    count: frags.msg_count,
                    last_role: frags.last_role,
                },
            };
            let mut cache = CACHE.lock().expect("wire frag cache");
            if cache.slots.len() < SLOTS {
                cache.slots.push(entry);
            } else {
                let next = cache.next;
                cache.slots[next] = entry;
                cache.next = (next + 1) % SLOTS;
            }
            wire
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::compat::materialize;
    use crate::providers::openai_completions::convert::transform_for_request;
    use crate::providers::openai_completions::params::build_params;
    use crate::providers::openai_completions::resolve_compat;
    use crate::types::{ModelThinkingLevel, StreamOptions};
    use crate::utils_inner::wire_json::to_json_string;

    const BODIES: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../runs/omp-rust/compat-oracle/bodies.jsonl"
    );

    fn model_of(spec: &Value) -> Model {
        let mut model: Model =
            serde_json::from_value(spec.clone()).expect("spec parses as a Model");
        materialize(&mut model, None).expect("policy resolves");
        model
    }

    fn options_of(case: &Value) -> OpenAICompletionsOptions {
        let o = &case["options"];
        let mut base = StreamOptions::default();
        base.max_tokens = o.get("maxTokens").and_then(Value::as_u64);
        base.temperature = o.get("temperature").and_then(Value::as_f64);
        base.session_id = o
            .get("sessionId")
            .and_then(Value::as_str)
            .map(str::to_string);
        let mut opts = OpenAICompletionsOptions::from_base(base);
        opts.reasoning_effort = o
            .get("reasoning")
            .and_then(Value::as_str)
            .and_then(crate::models::thinking_level_from_str)
            .filter(|l| *l != ModelThinkingLevel::Off);
        if o.get("disableReasoning") == Some(&serde_json::json!(true)) {
            opts.reasoning_enabled = Some(false);
        }
        if let Some(name) = o
            .pointer("/toolChoice/function/name")
            .and_then(Value::as_str)
        {
            opts.tool_choice = Some(ToolChoice::Function {
                name: name.to_string(),
            });
        }
        opts
    }

    /// The emitted request body for a context through the cached builder.
    fn cached_body(
        model: &Model,
        context: &Context,
        options: Option<&OpenAICompletionsOptions>,
    ) -> String {
        let compat = resolve_compat(model).expect("compat resolves");
        let transformed = transform_for_request(model, context, &compat);
        to_json_string(&build_params_cached(
            model,
            context,
            &transformed,
            options,
            &compat,
            CacheRetention::Short,
        ))
    }

    fn uncached_body(
        model: &Model,
        context: &Context,
        options: Option<&OpenAICompletionsOptions>,
    ) -> String {
        let compat = resolve_compat(model).expect("compat resolves");
        let transformed = transform_for_request(model, context, &compat);
        to_json_string(&build_params(
            model,
            context,
            &transformed,
            options,
            &compat,
            CacheRetention::Short,
        ))
    }

    /// Every oracle case (all compat flavors: anthropic cache-control, mistral ids, zai, strict
    /// tools, forced choices…), exercised as: cold miss, warm hit (same context again), and a
    /// growing prefix (earlier message subset, then the full history — the splice path). The
    /// cached body must equal the uncached body byte-for-byte in all three, including the
    /// `to_value` tree a payload hook would see.
    #[test]
    fn cached_body_matches_uncached_for_every_oracle_case() {
        let text = match std::fs::read_to_string(BODIES) {
            Ok(text) => text,
            Err(_) => return, // oracle file not generated here; body_parity_tests covers it
        };
        let mut total = 0usize;
        for line in text.lines().filter(|l| !l.is_empty()) {
            let case: Value = serde_json::from_str(line).expect("case parses");
            if case.get("error").is_some_and(|e| !e.is_null()) {
                continue;
            }
            total += 1;
            let model = model_of(&case["spec"]);
            let context: Context =
                serde_json::from_value(case["context"].clone()).expect("context parses");
            let options = options_of(&case);
            let want = uncached_body(&model, &context, Some(&options));

            // Cold call: whatever the global slot holds, this key must produce identical bytes.
            assert_eq!(
                cached_body(&model, &context, Some(&options)),
                want,
                "miss: case {}",
                case["case"]
            );
            // Warm hit: same request again serves the stored prefix.
            assert_eq!(
                cached_body(&model, &context, Some(&options)),
                want,
                "hit: case {}",
                case["case"]
            );
            // Value-tree parity for the payload-hook path.
            let compat = resolve_compat(&model).expect("compat resolves");
            let transformed = transform_for_request(&model, &context, &compat);
            let cached_value = build_params_cached(
                &model,
                &context,
                &transformed,
                Some(&options),
                &compat,
                CacheRetention::Short,
            )
            .to_value();
            let uncached_value = build_params(
                &model,
                &context,
                &transformed,
                Some(&options),
                &compat,
                CacheRetention::Short,
            )
            .to_value();
            assert_eq!(
                cached_value, uncached_value,
                "to_value: case {}",
                case["case"]
            );

            // Growing prefix: earlier subset first (seeds the cache), then the full history.
            let messages = context.messages.clone();
            if messages.len() >= 2 {
                let mut prefix = context.clone();
                prefix.messages = messages[..messages.len() / 2].to_vec();
                let _ = cached_body(&model, &prefix, Some(&options));
                assert_eq!(
                    cached_body(&model, &context, Some(&options)),
                    want,
                    "splice: case {}",
                    case["case"]
                );
            }
        }
        assert!(total > 1000, "expected the full oracle matrix, got {total}");
    }

    /// Key changes mid-session (system prompt, tools, options) must invalidate: the cached body
    /// still has to equal the uncached body after every change.
    #[test]
    fn cached_body_tracks_key_changes() {
        let text = match std::fs::read_to_string(BODIES) {
            Ok(text) => text,
            Err(_) => return,
        };
        let line = text.lines().find(|l| !l.is_empty()).expect("a case");
        let case: Value = serde_json::from_str(line).expect("case parses");
        let model = model_of(&case["spec"]);
        let base: Context =
            serde_json::from_value(case["context"].clone()).expect("context parses");
        let options = options_of(&case);

        let _ = cached_body(&model, &base, Some(&options));
        assert_eq!(
            cached_body(&model, &base, Some(&options)),
            uncached_body(&model, &base, Some(&options))
        );

        let mut changed = base.clone();
        changed.system_prompt = Some("a different system prompt".to_string());
        assert_eq!(
            cached_body(&model, &changed, Some(&options)),
            uncached_body(&model, &changed, Some(&options))
        );
        assert_eq!(
            cached_body(&model, &changed, Some(&options)),
            uncached_body(&model, &changed, Some(&options))
        );

        changed.tools = Some(vec![Tool {
            name: "new_tool".to_string(),
            description: "appears mid-session".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }]);
        assert_eq!(
            cached_body(&model, &changed, Some(&options)),
            uncached_body(&model, &changed, Some(&options))
        );
        assert_eq!(
            cached_body(&model, &changed, Some(&options)),
            uncached_body(&model, &changed, Some(&options))
        );

        changed.tools = None;
        changed.messages.clear();
        assert_eq!(
            cached_body(&model, &changed, Some(&options)),
            uncached_body(&model, &changed, Some(&options))
        );
        assert_eq!(
            cached_body(&model, &changed, Some(&options)),
            uncached_body(&model, &changed, Some(&options))
        );
    }
}
