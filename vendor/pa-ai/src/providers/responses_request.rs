//! OpenAI Responses request bodies (TS `openai-responses.ts` / `azure-openai-responses.ts`
//! `buildParams`, `buildResponsesInput` and `convertTools` in `openai-shared.ts`), shared by the
//! `openai-responses`, `azure-openai-responses` and `openrouter` (Responses default) transports.

use serde_json::{json, Map, Value};

use crate::compat::ModelView;
use crate::providers::openai_completions::compat::{
    is_forced, resolve_policy, should_drop_auto_tool_choice_for_reasoning, CompatPolicy,
    ResolvedCompat,
};
use crate::providers::openai_completions::schema::{
    adapt_schema_for_strict, flatten_exclusive_required_root_union, normalize_schema_for_moonshot,
    tool_wire_schema,
};
use crate::providers::openai_completions::{normalize_stable_id, ToolChoice};
use crate::providers::transform_ts::transform_messages;
use crate::types::{
    AssistantContent, CacheRetention, Context, Message, Model, ModelExt, ServiceTier,
    StreamOptions, Tool, UserMessageContent, UserOrToolContent,
};

const OPENAI_MAX_OUTPUT_TOKENS: u64 = 64_000;
const NON_VISION_IMAGE_PLACEHOLDER: &str = "[image omitted: model does not support vision]";
const SYNTHETIC_REASONING_PLACEHOLDER: &str = "reasoning unavailable";

/// TS `reasoningSummary` option: unset, explicitly off (`null`) or a summary mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Summary {
    #[default]
    Unset,
    Null,
    #[allow(dead_code)] // set once the simple options carry a reasoning-summary choice
    Mode(crate::providers::openai_responses_hooks::ReasoningSummary),
}

/// Provider-native options of a Responses request (TS `OpenAIResponsesOptions` subset).
#[derive(Clone, Debug, Default)]
pub struct ResponsesOptions {
    pub base: StreamOptions,
    /// Effort resolved by `resolveOpenAiReasoningEffort` (wire vocabulary of the ladder).
    pub reasoning: Option<String>,
    pub reasoning_summary: Summary,
    pub disable_reasoning: bool,
    pub tool_choice: Option<ToolChoice>,
    pub max_tokens_explicit: bool,
    pub openrouter_variant: Option<String>,
}

impl ResponsesOptions {
    #[must_use]
    pub fn from_base(base: StreamOptions) -> Self {
        let max_tokens_explicit = base.max_tokens.is_some();
        Self {
            base,
            max_tokens_explicit,
            ..Self::default()
        }
    }
}

/// Request headers (TS `resolveOpenAIRequestSetup` + `applyInferenceHeaders`).
pub(crate) fn responses_headers(
    model: &Model,
    view: &ModelView,
    api_key: &str,
    extra: Option<&std::collections::HashMap<String, String>>,
    session_id: Option<&str>,
) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = Vec::new();
    fn set(headers: &mut Vec<(String, String)>, name: &str, value: &str) {
        headers.retain(|(n, _)| !n.eq_ignore_ascii_case(name));
        headers.push((name.to_string(), value.to_string()));
    }
    fn set_if_absent(headers: &mut Vec<(String, String)>, name: &str, value: &str) {
        if !headers.iter().any(|(n, _)| n.eq_ignore_ascii_case(name)) {
            headers.push((name.to_string(), value.to_string()));
        }
    }
    for (k, v) in model.headers.iter().flatten() {
        headers.push((k.clone(), v.clone()));
    }
    if model.provider == "openrouter" {
        for (k, v) in [
            ("User-Agent", "omp/18.9.1"),
            ("HTTP-Referer", "https://omp.sh/"),
            ("X-OpenRouter-Title", "omp"),
            ("X-OpenRouter-Categories", "cli-agent"),
            ("X-OpenRouter-Cache", "true"),
            ("X-OpenRouter-Cache-TTL", "3600"),
        ] {
            set(&mut headers, k, v);
        }
    }
    for (k, v) in extra.into_iter().flatten() {
        set(&mut headers, k, v);
    }
    if model.provider == "vercel-ai-gateway" {
        set_if_absent(&mut headers, "http-referer", "https://omp.sh/");
        set_if_absent(&mut headers, "x-title", "omp");
    }
    if let Some(id) = session_id.filter(|s| !s.is_empty()) {
        if model.provider == "openai" {
            set(&mut headers, "session_id", id);
            set(&mut headers, "x-client-request-id", id);
        }
        if matches!(model.provider.as_str(), "opencode-go" | "opencode-zen") {
            set_if_absent(&mut headers, "User-Agent", "omp/18.9.1");
            set(&mut headers, "x-opencode-session", id);
        }
        if let Some(cache_id) = prompt_cache_key_for(session_id) {
            if let Some(name) = view.compat_str("promptCacheSessionHeader") {
                set_if_absent(&mut headers, name, &cache_id);
            }
        }
        if let Some(name) = view.compat_str("sessionHeader") {
            set_if_absent(&mut headers, name, id);
        }
    }
    set_if_absent(&mut headers, "Authorization", &format!("Bearer {api_key}"));
    headers
}

fn prompt_cache_key_for(session_id: Option<&str>) -> Option<String> {
    normalize_stable_id(session_id.filter(|s| !s.is_empty())?, 64, "pc_")
}

fn resolve_retention(explicit: Option<CacheRetention>) -> CacheRetention {
    if let Some(r) = explicit {
        return r;
    }
    match std::env::var("PI_CACHE_RETENTION").as_deref() {
        Ok("long") => CacheRetention::Long,
        Ok("none") => CacheRetention::None,
        _ => CacheRetention::Short,
    }
}

fn prompt_cache_key(base: &StreamOptions) -> Option<String> {
    if resolve_retention(base.cache_retention) == CacheRetention::None {
        return None;
    }
    normalize_stable_id(
        base.session_id.as_deref().filter(|s| !s.is_empty())?,
        64,
        "pc_",
    )
}

fn compat_of(view: &ModelView) -> ResolvedCompat {
    serde_json::from_value(Value::Object(view.compat.clone())).unwrap_or_default()
}

// ---------------------------------------------------------------- input

fn clamp_image_detail(detail: Option<&str>, supports_original: bool) -> String {
    match detail {
        Some("original") if !supports_original => "high".into(),
        Some(d @ ("low" | "high" | "auto" | "original")) => d.into(),
        _ => "auto".into(),
    }
}

fn input_image(img: &crate::types::ImageContent, supports_original: bool) -> Value {
    let detail = clamp_image_detail(
        img.rest.get("detail").and_then(Value::as_str),
        supports_original,
    );
    let url = img
        .rest
        .get("url")
        .and_then(Value::as_str)
        .filter(|u| !u.is_empty())
        .map_or_else(
            || format!("data:{};base64,{}", img.mime_type, img.data),
            str::to_string,
        );
    json!({ "type": "input_image", "detail": detail, "image_url": url })
}

/// TS `convertResponsesInputContent`.
fn convert_input_content(
    content: &UserMessageContent,
    supports_images: bool,
    supports_original: bool,
) -> Option<Vec<Value>> {
    match content {
        UserMessageContent::Text(t) => {
            (!t.trim().is_empty()).then(|| vec![json!({ "type": "input_text", "text": t })])
        }
        UserMessageContent::Blocks(blocks) => {
            let mut out: Vec<Value> = Vec::new();
            for b in blocks {
                match b {
                    UserOrToolContent::Text(t) if !t.text.trim().is_empty() => {
                        out.push(json!({ "type": "input_text", "text": t.text }));
                    }
                    UserOrToolContent::Raw(v) => {
                        let text = v.to_string();
                        if !text.trim().is_empty() {
                            out.push(json!({ "type": "input_text", "text": text }));
                        }
                    }
                    _ => {}
                }
            }
            let images: Vec<&crate::types::ImageContent> = blocks
                .iter()
                .filter_map(|b| {
                    if let UserOrToolContent::Image(i) = b {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            if supports_images {
                out.extend(images.iter().map(|i| input_image(i, supports_original)));
            } else if !images.is_empty() {
                out.push(json!({ "type": "input_text", "text": NON_VISION_IMAGE_PLACEHOLDER }));
            }
            (!out.is_empty()).then_some(out)
        }
    }
}

fn base36_hash(value: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(value.as_bytes());
    let mut n = u64::from_be_bytes(digest[..8].try_into().expect("8 bytes"));
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

fn sanitize_call_id(id: &str) -> String {
    let s: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    s.chars().take(64).collect()
}

/// TS `normalizeResponsesToolCallIdForTransform`.
fn normalize_call_id_for_transform(
    id: &str,
    model: &Model,
    source: &crate::types::AssistantMessage,
) -> Option<String> {
    let sep = id.find(['\n', '|']);
    if sep.is_none()
        && id.len() <= 64
        && !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Some(id.to_string());
    }
    let foreign = source.provider != model.provider || source.api != model.api;
    if foreign || sep.is_some() || id.len() > 64 {
        let (call, item) = match sep {
            Some(s) if s > 0 => (&id[..s], Some(&id[s + 1..])),
            _ => (id, None),
        };
        let mut item_id = format!(
            "fc_{}",
            base36_hash(item.filter(|i| !i.is_empty()).unwrap_or(id))
        );
        item_id.truncate(64);
        return Some(format!("{}|{item_id}", sanitize_call_id(call)));
    }
    let (call, item) = split_call_id(id);
    Some(format!("{call}|{item}"))
}

/// TS `normalizeResponsesToolCallId` (`fc` prefix): `(callId, itemId)`.
fn split_call_id(id: &str) -> (String, String) {
    if let Some(sep) = id.find(['\n', '|']).filter(|s| *s > 0) {
        let call = id[..sep].to_string();
        let item = &id[sep + 1..];
        return (call, normalize_item_id(item, "fc"));
    }
    (id.to_string(), format!("fc_{}", base36_hash(id)))
}

fn normalize_item_id(item: &str, prefix: &str) -> String {
    let sanitized: String = item
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let mut s: String = sanitized.chars().take(64).collect();
    while s.ends_with('_') {
        s.pop();
    }
    if s.starts_with(&format!("{prefix}_")) || s.starts_with("fcr_") || s.starts_with("ctc_") {
        s
    } else {
        format!("{prefix}_{s}")
    }
}

/// TS `parseTextSignature`: `(id, phase)`.
fn parse_text_signature(sig: Option<&str>) -> Option<(String, Option<String>)> {
    let sig = sig.filter(|s| !s.is_empty())?;
    if sig.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<Value>(sig) {
            if v.get("v") == Some(&json!(1)) {
                if let Some(id) = v.get("id").and_then(Value::as_str) {
                    let phase = v
                        .get("phase")
                        .and_then(Value::as_str)
                        .filter(|p| matches!(*p, "commentary" | "final_answer"));
                    return Some((id.to_string(), phase.map(str::to_string)));
                }
            }
        }
    }
    Some((sig.to_string(), None))
}

fn parse_reasoning_replay_item(sig: Option<&str>) -> Option<Value> {
    let v: Value = serde_json::from_str(sig.filter(|s| !s.is_empty())?).ok()?;
    (v.get("type") == Some(&json!("reasoning")) && v.get("id").is_some_and(Value::is_string))
        .then_some(v)
}

fn synthetic_reasoning_item(text: &str, id: Option<&str>) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), json!("reasoning"));
    if let Some(id) = id.filter(|i| !i.is_empty()) {
        m.insert("id".into(), json!(id));
    }
    m.insert("summary".into(), json!([]));
    m.insert(
        "content".into(),
        json!([{ "type": "reasoning_text", "text": text }]),
    );
    Value::Object(m)
}

fn with_required_reasoning_text(item: Value) -> Value {
    if item.get("encrypted_content").is_some_and(Value::is_string) {
        return item;
    }
    let has_text = item
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|c| {
            c.iter().any(|p| {
                p["type"] == "reasoning_text"
                    && p["text"].as_str().is_some_and(|t| !t.trim().is_empty())
            })
        });
    if has_text {
        return item;
    }
    let summary: Vec<&str> = item
        .get("summary")
        .and_then(Value::as_array)
        .map(|s| {
            s.iter()
                .filter_map(|p| p["text"].as_str())
                .filter(|t| !t.trim().is_empty())
                .collect()
        })
        .unwrap_or_default();
    let text = summary.join("\n\n");
    let mut item = item;
    item["content"] = json!([{ "type": "reasoning_text", "text": if text.is_empty() { SYNTHETIC_REASONING_PLACEHOLDER } else { &text } }]);
    item
}

struct InputOptions {
    strict_pairing: bool,
    supports_original: bool,
    include_thinking_signatures: bool,
    preserve_message_ids: bool,
    reasoning_for_all_turns: bool,
    reasoning_for_tool_calls: bool,
    system_role: Option<&'static str>,
    /// Azure: a plain-string developer message goes out as a string `content` (TS `developerStringContent`).
    developer_string_content: bool,
}

/// TS `buildResponsesInput` for the shapes this runtime carries.
#[allow(clippy::too_many_lines)]
fn build_input(model: &Model, view: &ModelView, context: &Context, o: &InputOptions) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::new();
    if let Some(role) = o.system_role {
        for sp in context
            .system_prompt
            .iter()
            .filter(|p| !p.trim().is_empty())
        {
            messages.push(json!({ "role": role, "content": sp }));
        }
    }
    let transformed = transform_messages(
        &context.messages,
        model,
        view,
        &normalize_call_id_for_transform,
    );
    let supports_images = model.supports_image_input();
    let mut known_call_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut msg_index = 0usize;
    for msg in &transformed {
        match msg {
            Message::User(user) | Message::Developer(user) => {
                let Some(content) =
                    convert_input_content(&user.content, supports_images, o.supports_original)
                else {
                    continue;
                };
                let content = match (&msg, &user.content) {
                    (Message::Developer(_), UserMessageContent::Text(text))
                        if o.developer_string_content =>
                    {
                        json!(text)
                    }
                    _ => json!(content),
                };
                messages.push(json!({ "role": "user", "content": content }));
            }
            Message::Assistant(a) => {
                let mut out: Vec<Value> = Vec::new();
                let has_replayable = o.include_thinking_signatures
                    && a.stop_reason != crate::types::StopReason::Error
                    && a.content.iter().any(|b| {
                        matches!(b, AssistantContent::Thinking(t) if parse_reasoning_replay_item(t.thinking_signature.as_deref()).is_some())
                    });
                let is_different_model =
                    a.model != model.id && a.provider == model.provider && a.api == model.api;
                let requires_reasoning_item = a.stop_reason != crate::types::StopReason::Error
                    && (o.reasoning_for_all_turns
                        || (o.reasoning_for_tool_calls
                            && a.content
                                .iter()
                                .any(|b| matches!(b, AssistantContent::ToolCall(_)))));
                let mut reasoning_emitted = false;
                let mut carried: Vec<String> = Vec::new();
                let mut unsigned_text = 0usize;
                for block in &a.content {
                    match block {
                        AssistantContent::Thinking(t)
                            if a.stop_reason != crate::types::StopReason::Error =>
                        {
                            if requires_reasoning_item && !t.thinking.trim().is_empty() {
                                carried.push(t.thinking.clone());
                            }
                            if !o.include_thinking_signatures {
                                continue;
                            }
                            if let Some(item) =
                                parse_reasoning_replay_item(t.thinking_signature.as_deref())
                            {
                                out.push(if requires_reasoning_item {
                                    with_required_reasoning_text(item)
                                } else {
                                    item
                                });
                                reasoning_emitted = true;
                            }
                        }
                        AssistantContent::Thinking(_) => {}
                        AssistantContent::Text(t) => {
                            let parsed = parse_text_signature(t.text_signature.as_deref());
                            let mut msg_id = parsed.as_ref().map(|p| p.0.clone());
                            match &msg_id {
                                None => {
                                    if has_replayable {
                                        msg_id = Some(if unsigned_text == 0 {
                                            format!("msg_{msg_index}")
                                        } else {
                                            format!("msg_{msg_index}_{unsigned_text}")
                                        });
                                        unsigned_text += 1;
                                    }
                                }
                                Some(_) if !o.preserve_message_ids && !has_replayable => {
                                    msg_id = None
                                }
                                Some(id) if id.len() > 64 => {
                                    msg_id = Some(format!("msg_{}", base36_hash(id)))
                                }
                                _ => {}
                            }
                            let mut item = Map::new();
                            item.insert("type".into(), json!("message"));
                            item.insert("role".into(), json!("assistant"));
                            item.insert("content".into(), json!([{ "type": "output_text", "text": t.text, "annotations": [] }]));
                            item.insert("status".into(), json!("completed"));
                            if let Some(id) = msg_id {
                                item.insert("id".into(), json!(id));
                            }
                            if let Some(phase) = parsed.and_then(|p| p.1) {
                                item.insert("phase".into(), json!(phase));
                            }
                            out.push(Value::Object(item));
                        }
                        AssistantContent::ToolCall(call) => {
                            let (call_id, mut item_id) = split_call_id(&call.id);
                            let prefixed = |s: &str| {
                                s.starts_with("fc_")
                                    || s.starts_with("fcr_")
                                    || s.starts_with("ctc_")
                            };
                            let drop_item = (!has_replayable && prefixed(&item_id))
                                || (is_different_model && prefixed(&item_id));
                            known_call_ids.insert(call_id.clone());
                            let mut item = Map::new();
                            item.insert("type".into(), json!("function_call"));
                            if !drop_item && !item_id.is_empty() {
                                item.insert("id".into(), json!(std::mem::take(&mut item_id)));
                            }
                            item.insert("call_id".into(), json!(call_id));
                            item.insert("name".into(), json!(call.name));
                            item.insert(
                                "arguments".into(),
                                json!(Value::Object(call.arguments.clone()).to_string()),
                            );
                            out.push(Value::Object(item));
                        }
                    }
                }
                if requires_reasoning_item && !reasoning_emitted && !out.is_empty() {
                    let text = if carried.is_empty() {
                        SYNTHETIC_REASONING_PLACEHOLDER.to_string()
                    } else {
                        carried.join("\n")
                    };
                    out.insert(0, synthetic_reasoning_item(&text, None));
                }
                if out.is_empty() {
                    continue;
                }
                messages.extend(out);
            }
            Message::ToolResult(r) => {
                let text_result: Vec<&str> = r
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let UserOrToolContent::Text(t) = b {
                            Some(t.text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
                let text_result = text_result.join("\n");
                let has_images = r
                    .content
                    .iter()
                    .any(|b| matches!(b, UserOrToolContent::Image(_)));
                let omitted = has_images && !supports_images;
                let raw_output = if omitted {
                    if text_result.is_empty() {
                        NON_VISION_IMAGE_PLACEHOLDER.to_string()
                    } else {
                        format!("{text_result}\n{NON_VISION_IMAGE_PLACEHOLDER}")
                    }
                } else if !text_result.is_empty() {
                    text_result.clone()
                } else if has_images {
                    "(see attached image)".to_string()
                } else {
                    String::new()
                };
                let output: Value = if has_images && supports_images {
                    Value::Array(
                        r.content
                            .iter()
                            .map(|b| match b {
                                UserOrToolContent::Image(i) => input_image(i, o.supports_original),
                                UserOrToolContent::Text(t) => {
                                    json!({ "type": "input_text", "text": t.text })
                                }
                                UserOrToolContent::Raw(v) => {
                                    json!({ "type": "input_text", "text": v.to_string() })
                                }
                            })
                            .collect(),
                    )
                } else {
                    Value::String(raw_output.clone())
                };
                let call_id = split_call_id(&r.tool_call_id).0;
                if o.strict_pairing && !known_call_ids.contains(&call_id) {
                    let note: String = if raw_output.chars().count() > 16_000 {
                        format!(
                            "{}\n...[truncated]",
                            raw_output.chars().take(16_000).collect::<String>()
                        )
                    } else {
                        raw_output
                    };
                    let name = if r.tool_name.is_empty() {
                        "tool"
                    } else {
                        r.tool_name.as_str()
                    };
                    messages.push(json!({
                        "type": "message", "role": "assistant",
                        "content": format!("[Orphan {name} result; call_id={call_id}]: {note}"),
                    }));
                } else {
                    messages.push(json!({ "type": "function_call_output", "call_id": call_id, "output": output }));
                }
            }
        }
        msg_index += 1;
    }
    messages
}

// ---------------------------------------------------------------- tools

fn has_unsupported_lookaround(pattern: &str) -> bool {
    let b: Vec<char> = pattern.chars().collect();
    let mut i = 0;
    while i + 2 < b.len() {
        if b[i] == '(' && b[i + 1] == '?' {
            let escapes = b[..i].iter().rev().take_while(|c| **c == '\\').count();
            if escapes % 2 == 0 {
                let op: String = if b[i + 2] == '<' {
                    b[i + 2..(i + 4).min(b.len())].iter().collect()
                } else {
                    b[i + 2].to_string()
                };
                if matches!(op.as_str(), "=" | "!" | "<=" | "<!") {
                    return true;
                }
            }
        }
        i += 1;
    }
    false
}

const SCHEMA_MAP_KEYS: [&str; 4] = ["properties", "patternProperties", "$defs", "definitions"];
const SCHEMA_ARRAY_KEYS: [&str; 3] = ["anyOf", "allOf", "prefixItems"];
const SCHEMA_VALUE_KEYS: [&str; 10] = [
    "additionalProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "items",
    "additionalItems",
    "contains",
    "propertyNames",
    "if",
    "then",
    "else",
];

fn sanitize_responses_node(value: &Value) -> Value {
    let Value::Object(obj) = value else {
        return value.clone();
    };
    if obj.is_empty() {
        return Value::Bool(true);
    }
    let mut out = Map::new();
    let mut changed = false;
    for (key, child) in obj {
        if key == "oneOf" && child.is_array() {
            changed = true;
            continue;
        }
        if key == "pattern" && child.as_str().is_some_and(has_unsupported_lookaround) {
            changed = true;
            continue;
        }
        let next = if key == "patternProperties" && child.is_object() {
            sanitize_responses_map(child.as_object().expect("object"), true)
        } else if SCHEMA_MAP_KEYS.contains(&key.as_str()) && child.is_object() {
            sanitize_responses_map(child.as_object().expect("object"), false)
        } else if SCHEMA_ARRAY_KEYS.contains(&key.as_str()) && child.is_array() {
            Value::Array(
                child
                    .as_array()
                    .expect("array")
                    .iter()
                    .map(sanitize_responses_node)
                    .collect(),
            )
        } else if SCHEMA_VALUE_KEYS.contains(&key.as_str()) && child.is_object() {
            sanitize_responses_node(child)
        } else {
            child.clone()
        };
        if &next != child {
            changed = true;
        }
        out.insert(key.clone(), next);
    }
    if let Some(Value::Array(one_of)) = obj.get("oneOf") {
        let rewritten: Vec<Value> = one_of.iter().map(sanitize_responses_node).collect();
        let merged = match out.get("anyOf") {
            Some(Value::Array(existing)) => existing.iter().cloned().chain(rewritten).collect(),
            _ => rewritten,
        };
        out.insert("anyOf".into(), Value::Array(merged));
    }
    let declares_object = match obj.get("type") {
        Some(Value::String(t)) => t == "object",
        Some(Value::Array(a)) => a.iter().any(|v| v == "object"),
        _ => false,
    };
    if declares_object && !obj.contains_key("properties") {
        out.insert("properties".into(), json!({}));
        changed = true;
    }
    if changed {
        if out.is_empty() {
            Value::Bool(true)
        } else {
            Value::Object(out)
        }
    } else {
        value.clone()
    }
}

fn sanitize_responses_map(map: &Map<String, Value>, strip_regex_keys: bool) -> Value {
    let mut out = Map::new();
    for (key, child) in map {
        let next = sanitize_responses_node(child);
        if strip_regex_keys && has_unsupported_lookaround(key) {
            match out.get_mut(".*") {
                None => {
                    out.insert(".*".into(), next);
                }
                Some(Value::Object(existing))
                    if existing.len() == 1
                        && existing.get("anyOf").is_some_and(Value::is_array) =>
                {
                    if let Some(Value::Array(a)) = existing.get_mut("anyOf") {
                        a.push(next);
                    }
                }
                Some(existing) => {
                    let prev = std::mem::take(existing);
                    *existing = json!({ "anyOf": [prev, next] });
                }
            }
            continue;
        }
        out.insert(key.clone(), next);
    }
    Value::Object(out)
}

fn json_matches_type(v: &Value, ty: &str) -> bool {
    match ty {
        "string" => v.is_string(),
        "number" => v.is_number(),
        "integer" => v.as_f64().is_some_and(|n| n.fract() == 0.0) && v.is_number(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        "object" => v.is_object(),
        "array" => v.is_array(),
        _ => true,
    }
}

fn declared_types(node: &Map<String, Value>) -> Vec<String> {
    const NAMES: [&str; 7] = [
        "string", "number", "integer", "boolean", "object", "array", "null",
    ];
    match node.get("type") {
        Some(Value::String(t)) if NAMES.contains(&t.as_str()) => vec![t.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_str)
            .filter(|t| NAMES.contains(t))
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

/// TS `findStrictToolSchemaViolation`.
fn find_strict_violation(schema: &Value, path: &str, reject_root_union: bool) -> Option<String> {
    match schema {
        Value::Array(a) => a
            .iter()
            .enumerate()
            .find_map(|(i, s)| find_strict_violation(s, &format!("{path}/{i}"), reject_root_union)),
        Value::Object(node) => {
            let types = declared_types(node);
            if !types.is_empty() {
                if let Some(Value::Array(e)) = node.get("enum") {
                    if e.iter()
                        .any(|v| !types.iter().any(|t| json_matches_type(v, t)))
                    {
                        return Some(format!("{path}/enum"));
                    }
                }
                if let Some(c) = node.get("const") {
                    if !types.iter().any(|t| json_matches_type(c, t)) {
                        return Some(format!("{path}/const"));
                    }
                }
            }
            if reject_root_union
                && path == "#"
                && (types.iter().any(|t| t == "object")
                    || node.get("properties").is_some_and(Value::is_object))
            {
                for key in ["anyOf", "oneOf"] {
                    if let Some(Value::Array(arr)) = node.get(key) {
                        if arr.is_empty() {
                            continue;
                        }
                        let non_object = arr.iter().any(|b| match b {
                            Value::Object(o) => !declared_types(o).iter().any(|t| t == "object"),
                            _ => true,
                        });
                        if non_object {
                            return Some(format!("{path}/{key}"));
                        }
                    }
                }
            }
            for key in [
                "properties",
                "patternProperties",
                "$defs",
                "definitions",
                "dependentSchemas",
            ] {
                if let Some(Value::Object(sub)) = node.get(key) {
                    for (k, v) in sub {
                        if let Some(hit) = find_strict_violation(
                            v,
                            &format!("{path}/{key}/{k}"),
                            reject_root_union,
                        ) {
                            return Some(hit);
                        }
                    }
                }
            }
            for key in [
                "items",
                "contains",
                "not",
                "if",
                "then",
                "else",
                "propertyNames",
                "additionalProperties",
                "unevaluatedProperties",
                "unevaluatedItems",
            ] {
                if let Some(v) = node.get(key) {
                    if let Some(hit) =
                        find_strict_violation(v, &format!("{path}/{key}"), reject_root_union)
                    {
                        return Some(hit);
                    }
                }
            }
            for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
                if let Some(v @ Value::Array(_)) = node.get(key) {
                    if let Some(hit) =
                        find_strict_violation(v, &format!("{path}/{key}"), reject_root_union)
                    {
                        return Some(hit);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// TS Responses `convertTools`: `(wire tools, surviving source tools)`.
fn convert_tools<'a>(
    tools: &'a [Tool],
    strict_mode: bool,
    view: &ModelView,
) -> (Vec<Value>, Vec<&'a Tool>) {
    let no_strict = matches!(std::env::var("PI_NO_STRICT").as_deref(), Ok("1" | "true"));
    let reject_root_union = view.is("rejectRootObjectUnion");
    let mut out = Vec::new();
    let mut surviving = Vec::new();
    for tool in tools {
        let strict = !no_strict && strict_mode;
        let base = tool_wire_schema(&tool.parameters);
        let sanitized = sanitize_responses_node(&base);
        let provider_params = if reject_root_union {
            flatten_exclusive_required_root_union(&sanitized)
        } else {
            sanitized
        };
        let response_params = if view.compat_str("toolSchemaFlavor") == Some("moonshot-mfjs") {
            normalize_schema_for_moonshot(&provider_params)
        } else {
            provider_params
        };
        let (parameters, effective_strict) = adapt_schema_for_strict(&response_params, strict);
        if find_strict_violation(&parameters, "#", reject_root_union).is_some() {
            continue;
        }
        let mut item = Map::new();
        item.insert("type".into(), json!("function"));
        item.insert("name".into(), json!(tool.name));
        item.insert("description".into(), json!(tool.description));
        item.insert("parameters".into(), parameters);
        if effective_strict {
            item.insert("strict".into(), json!(true));
        }
        out.push(Value::Object(item));
        surviving.push(tool);
    }
    (out, surviving)
}

fn map_responses_tool_choice(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => Some(json!("auto")),
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Required => Some(json!("required")),
        ToolChoice::Function { name } if !name.is_empty() => {
            Some(json!({ "type": "function", "name": name }))
        }
        ToolChoice::Function { .. } => None,
    }
}

// ---------------------------------------------------------------- params

fn should_send_service_tier(provider: &str, tier: ServiceTier) -> bool {
    if tier == ServiceTier::Auto {
        return false;
    }
    if provider == "openai" || provider == "openai-codex" {
        return true;
    }
    provider == "openrouter"
        && matches!(
            tier,
            ServiceTier::Flex | ServiceTier::Scale | ServiceTier::Priority
        )
}

fn clamp_output_to_model_max(view: &ModelView, model: &Model) -> Option<u64> {
    view.is("clampOutputToModelMax").then(|| {
        if model.max_tokens == 0 {
            OPENAI_MAX_OUTPUT_TOKENS
        } else {
            model.max_tokens
        }
    })
}

fn applies_summary(view: &ModelView, o: &ResponsesOptions) -> Summary {
    if view.is("supportsReasoningSummary") {
        return o.reasoning_summary;
    }
    if o.reasoning.is_none() {
        Summary::Unset
    } else {
        Summary::Null
    }
}

/// TS `applyResponsesCompatPolicy` (without `forceReasoningOff`).
fn apply_responses_compat_policy(
    params: &mut Map<String, Value>,
    policy: &CompatPolicy,
    view: &ModelView,
    summary: &Summary,
    map_effort: &dyn Fn(&str) -> String,
) {
    let r = &policy.reasoning;
    if !r.model_supported {
        return;
    }
    if view.is("includeEncryptedReasoning") {
        let mut include: Vec<Value> = params
            .get("include")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !include.iter().any(|v| v == "reasoning.encrypted_content") {
            include.push(json!("reasoning.encrypted_content"));
        }
        params.insert("include".into(), Value::Array(include));
    }
    if r.disabled {
        if r.disable_mode == "openrouter-enabled-false" {
            params.insert("reasoning".into(), json!({ "enabled": false }));
            return;
        }
        if matches!(r.disable_mode.as_str(), "lowest-effort" | "none-effort")
            && r.wire_effort.is_some()
            && !r.omit_reasoning_effort
        {
            params.insert("reasoning".into(), json!({ "effort": r.wire_effort }));
        }
        return;
    }
    if r.requested_effort.is_some() || *summary != Summary::Unset {
        if r.omit_reasoning_effort {
            if let Summary::Mode(m) = summary {
                params.insert("reasoning".into(), json!({ "summary": m.as_str() }));
            }
            return;
        }
        let requested = r
            .requested_effort
            .clone()
            .unwrap_or_else(|| "medium".into());
        let wire = r
            .wire_effort
            .clone()
            .unwrap_or_else(|| map_effort(&requested));
        let mut reasoning = Map::new();
        reasoning.insert("effort".into(), json!(wire));
        if *summary != Summary::Null {
            let s = match summary {
                Summary::Mode(m) => m.as_str(),
                _ => "auto",
            };
            reasoning.insert("summary".into(), json!(s));
        }
        params.insert("reasoning".into(), Value::Object(reasoning));
    }
}

/// Which Responses flavor builds the body.
pub(crate) enum Flavor {
    /// `openai-responses` and the OpenRouter Responses default.
    OpenAi,
    /// `azure-openai-responses` with its resolved deployment name.
    Azure { deployment: String },
}

/// The request body TS `buildParams` produces for `flavor`.
///
/// # Errors
/// Never in practice; reserved for configuration errors TS throws.
#[allow(clippy::too_many_lines)]
pub(crate) fn build_body(
    model: &Model,
    view: &ModelView,
    context: &Context,
    options: &ResponsesOptions,
    flavor: &Flavor,
) -> Result<Value, String> {
    let compat = compat_of(view);
    let has_tools = false;
    let policy = resolve_policy(
        model,
        &compat,
        options.reasoning.as_deref(),
        options.disable_reasoning,
        options.tool_choice.as_ref(),
        has_tools,
    );
    let map_effort = |level: &str| -> String {
        view.compat
            .get("reasoningEffortMap")
            .and_then(|m| m.get(level))
            .and_then(Value::as_str)
            .or_else(|| view.effort_map(level))
            .unwrap_or(level)
            .to_string()
    };
    let summary = applies_summary(view, options);
    let cache_retention = resolve_retention(options.base.cache_retention);
    let supports_original = view.is("supportsImageDetailOriginal");

    let mut params = Map::new();
    let mut input: Vec<Value>;
    let mut instructions: Option<String> = None;
    match flavor {
        Flavor::OpenAi => {
            let strict_pairing = view.is("strictResponsesPairing");
            let filter_reasoning = view.is("filterReasoningHistory");
            let can_reconstruct =
                !filter_reasoning || view.is("allowsSyntheticReasoningContentForToolCalls");
            let enabled = policy.reasoning.enabled;
            input = build_input(
                model,
                view,
                context,
                &InputOptions {
                    strict_pairing,
                    supports_original,
                    include_thinking_signatures: !filter_reasoning,
                    preserve_message_ids: false,
                    reasoning_for_all_turns: enabled
                        && view.is("requiresReasoningContentForAllAssistantTurns")
                        && can_reconstruct,
                    reasoning_for_tool_calls: enabled
                        && view.is("requiresReasoningContentForToolCalls")
                        && can_reconstruct,
                    system_role: None,
                    developer_string_content: false,
                },
            );
            let prompts: Vec<&String> = context
                .system_prompt
                .iter()
                .filter(|p| !p.trim().is_empty())
                .collect();
            if !prompts.is_empty() {
                let developer = model.reasoning && view.is("supportsDeveloperRole");
                if developer {
                    for (i, p) in prompts.iter().enumerate() {
                        input.insert(i, json!({ "role": "developer", "content": p }));
                    }
                } else {
                    instructions = Some(
                        prompts
                            .iter()
                            .map(|p| p.as_str())
                            .collect::<Vec<_>>()
                            .join("\n\n"),
                    );
                }
            }
            let key = prompt_cache_key(&options.base);
            let model_id = wire_model_id(model, view, options.openrouter_variant.as_deref());
            params.insert("model".into(), json!(model_id));
            params.insert("input".into(), Value::Null);
            if let Some(i) = &instructions {
                params.insert("instructions".into(), json!(i));
            }
            params.insert("stream".into(), json!(true));
            if let Some(k) = &key {
                params.insert("prompt_cache_key".into(), json!(k));
                if cache_retention == CacheRetention::Long
                    && view.is("supportsLongPromptCacheRetention")
                {
                    params.insert("prompt_cache_retention".into(), json!("24h"));
                }
            }
            if view.is("isOpenRouterHost") && cache_retention != CacheRetention::None {
                if let Some(id) = options.base.session_id.as_deref().filter(|s| !s.is_empty()) {
                    if let Some(s) = normalize_stable_id(id, 256, "session_") {
                        params.insert("session_id".into(), json!(s));
                    }
                }
            }
            let stores = view.is("storeResponses")
                && matches!(
                    std::env::var("PI_MUSE_STORE_RESPONSES").as_deref(),
                    Ok("1" | "true")
                );
            params.insert("store".into(), json!(stores));
            if view.is("supportsObfuscationOptOut") {
                params.insert(
                    "stream_options".into(),
                    json!({ "include_obfuscation": false }),
                );
            }
            if cache_retention != CacheRetention::None
                && view.compat_str("cacheControlFormat") == Some("anthropic")
            {
                params.insert(
                    "cache_control".into(),
                    if cache_retention == CacheRetention::Long {
                        json!({ "type": "ephemeral", "ttl": "1h" })
                    } else {
                        json!({ "type": "ephemeral" })
                    },
                );
            }
            // Output tokens.
            let model_max = (model.max_tokens > 0).then_some(model.max_tokens);
            let omit = model.omit_max_output_tokens.unwrap_or(false);
            let always = view.is("alwaysSendMaxTokens");
            let requested = options
                .base
                .max_tokens
                .or_else(|| always.then(|| model_max.unwrap_or(OPENAI_MAX_OUTPUT_TOKENS)));
            let clamp = clamp_output_to_model_max(view, model);
            let output_token = if omit {
                None
            } else {
                requested
                    .filter(|_| {
                        !(view.is("isOpenRouterHost") && !always && !options.max_tokens_explicit)
                    })
                    .and_then(|r| {
                        let v = r
                            .min(model_max.unwrap_or(u64::MAX))
                            .min(clamp.unwrap_or(OPENAI_MAX_OUTPUT_TOKENS));
                        (v > 0).then_some(v)
                    })
            };
            if let Some(max) = output_token.filter(|m| *m > 0).filter(|_| !omit) {
                params.insert(
                    "max_output_tokens".into(),
                    json!(max
                        .min(model_max.unwrap_or(u64::MAX))
                        .min(clamp.unwrap_or(OPENAI_MAX_OUTPUT_TOKENS))),
                );
            }
        }
        Flavor::Azure { deployment } => {
            let system_role = if model.reasoning && view.is("supportsDeveloperRole") {
                "developer"
            } else {
                "system"
            };
            input = build_input(
                model,
                view,
                context,
                &InputOptions {
                    strict_pairing: true,
                    supports_original,
                    include_thinking_signatures: true,
                    preserve_message_ids: true,
                    reasoning_for_all_turns: false,
                    reasoning_for_tool_calls: false,
                    system_role: Some(system_role),
                    developer_string_content: true,
                },
            );
            params.insert("model".into(), json!(deployment));
            params.insert("input".into(), Value::Null);
            params.insert("stream".into(), json!(true));
            if let Some(k) = prompt_cache_key(&options.base) {
                params.insert("prompt_cache_key".into(), json!(k));
            }
            params.insert("store".into(), json!(false));
            let model_max = (model.max_tokens > 0).then_some(model.max_tokens);
            if let Some(max) = options
                .base
                .max_tokens
                .filter(|m| *m > 0)
                .filter(|_| !model.omit_max_output_tokens.unwrap_or(false))
            {
                let clamp = clamp_output_to_model_max(view, model);
                params.insert(
                    "max_output_tokens".into(),
                    json!(max
                        .min(model_max.unwrap_or(u64::MAX))
                        .min(clamp.unwrap_or(OPENAI_MAX_OUTPUT_TOKENS))),
                );
            }
        }
    }
    // Sampling + service tier.
    if view.is("supportsSamplingParams") {
        if let Some(t) = options.base.temperature {
            params.insert("temperature".into(), json!(t));
        }
    }
    if let Some(tier) = options.base.service_tier {
        if should_send_service_tier(&model.provider, tier) {
            params.insert(
                "service_tier".into(),
                serde_json::to_value(tier).unwrap_or(Value::Null),
            );
        }
    }

    // Tools.
    let mut emitted_tools: Option<Vec<Value>> = None;
    let mut tool_choice: Option<Value> = None;
    if let Some(tools) = &context.tools {
        match flavor {
            Flavor::OpenAi => {
                let strict_mode = view.not_false("supportsStrictMode");
                let (wire, surviving) = convert_tools(tools, strict_mode, view);
                let mut wire = wire;
                if let Some(choice) = &options.tool_choice {
                    let mapped: Option<Value> = if !view.is("supportsToolChoice") {
                        None
                    } else if is_forced(Some(choice)) && !view.is("supportsForcedToolChoice") {
                        Some(json!("auto"))
                    } else {
                        match map_responses_tool_choice(choice) {
                            Some(Value::Object(o)) => {
                                let name = o.get("name").and_then(Value::as_str).unwrap_or("");
                                surviving
                                    .iter()
                                    .any(|t| t.name == name)
                                    .then(|| Value::Object(o))
                            }
                            other => other,
                        }
                    };
                    if let Some(tc) = mapped.filter(|_| !wire.is_empty()) {
                        if tc.get("type") == Some(&json!("function"))
                            && !view.is("supportsNamedToolChoice")
                        {
                            let name = tc["name"].as_str().unwrap_or("").to_string();
                            wire.retain(|t| t["type"] == "function" && t["name"] == name.as_str());
                            tool_choice = Some(json!("required"));
                        } else {
                            tool_choice = Some(tc);
                        }
                    }
                }
                emitted_tools = Some(wire);
            }
            Flavor::Azure { .. } => {
                let serialized: Vec<Value> = tools
                    .iter()
                    .map(|t| {
                        json!({
                            "type": "function", "name": t.name, "description": t.description,
                            "parameters": sanitize_responses_node(&tool_wire_schema(&t.parameters)), "strict": false,
                        })
                    })
                    .collect();
                if !serialized.is_empty() {
                    if let Some(choice) = &options.tool_choice {
                        let mapped = map_responses_tool_choice(choice);
                        tool_choice = match mapped {
                            Some(Value::String(s)) => Some(Value::String(s)),
                            Some(Value::Object(o)) => {
                                let name = o.get("name").and_then(Value::as_str).unwrap_or("");
                                serialized
                                    .iter()
                                    .any(|t| t["name"] == name)
                                    .then(|| Value::Object(o))
                            }
                            _ => None,
                        };
                    }
                    emitted_tools = Some(serialized);
                }
            }
        }
    }
    if let Some(tools) = emitted_tools {
        params.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tc) = &tool_choice {
        if matches!(flavor, Flavor::Azure { .. })
            || !(tc == "auto" && drop_auto_tool_choice(model, &compat, options))
        {
            params.insert("tool_choice".into(), tc.clone());
        }
    }

    // Reasoning.
    let reasoning_policy = if matches!(flavor, Flavor::OpenAi) {
        let choice_for_policy: Option<ToolChoice> =
            params.get("tool_choice").and_then(|t| match t {
                Value::String(s) if s == "auto" => Some(ToolChoice::Auto),
                Value::String(s) if s == "none" => Some(ToolChoice::None),
                Value::String(s) if s == "required" => Some(ToolChoice::Required),
                Value::Object(o) => {
                    o.get("name")
                        .and_then(Value::as_str)
                        .map(|n| ToolChoice::Function {
                            name: n.to_string(),
                        })
                }
                _ => None,
            });
        resolve_policy(
            model,
            &compat,
            options.reasoning.as_deref(),
            options.disable_reasoning,
            choice_for_policy.as_ref(),
            false,
        )
    } else {
        resolve_policy(
            model,
            &compat,
            options.reasoning.as_deref(),
            options.disable_reasoning,
            options.tool_choice.as_ref(),
            false,
        )
    };
    apply_responses_compat_policy(&mut params, &reasoning_policy, view, &summary, &map_effort);

    if let Some(provider_routing) = view
        .compat
        .get("openRouterRouting")
        .filter(|_| view.is("isOpenRouterHost") && matches!(flavor, Flavor::OpenAi))
    {
        params.insert("provider".into(), provider_routing.clone());
    }
    params.insert("input".into(), Value::Array(input));
    // `instructions` keeps its slot after `input`.
    Ok(Value::Object(params))
}

fn drop_auto_tool_choice(
    model: &Model,
    compat: &ResolvedCompat,
    options: &ResponsesOptions,
) -> bool {
    should_drop_auto_tool_choice_for_reasoning(
        model,
        compat,
        Some(&ToolChoice::Auto),
        options.reasoning.as_deref(),
        options.disable_reasoning,
    )
}

/// TS `applyWireModelIdTransform` for the Responses transports (`openrouter` routing variant).
fn wire_model_id(model: &Model, view: &ModelView, variant: Option<&str>) -> String {
    let base = model
        .request_model_id
        .clone()
        .unwrap_or_else(|| model.id.clone());
    match view.compat_str("wireModelIdMode") {
        Some("openrouter") => match variant.filter(|v| !v.is_empty()) {
            Some(v) => {
                let last_slash = base.rfind('/');
                let last_colon = base.rfind(':');
                if last_colon > last_slash && last_colon.is_some() {
                    base
                } else {
                    format!("{base}:{v}")
                }
            }
            None => base,
        },
        _ => base,
    }
}
