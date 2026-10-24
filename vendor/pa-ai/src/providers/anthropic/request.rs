//! Anthropic Messages request body (TS `buildParams` + `convertAnthropicMessages` +
//! `convertTools` + prompt-cache placement in `providers/anthropic.ts`), built as an owned JSON
//! tree so the cache-breakpoint passes can rewrite it in place exactly as TS does.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::compat::ModelView;
use crate::providers::anthropic::convert::normalize_tool_call_id;
use crate::providers::anthropic::AnthropicOptions;
use crate::providers::openai_completions::schema::{spill_to_description, tool_wire_schema};
use crate::providers::transform_ts::{transform_messages, SYNTHETIC_USER_KEY};
use crate::types::{
    AssistantContent, CacheRetention, Context, Message, Model, ModelExt, ServiceTier,
    UserMessageContent, UserOrToolContent,
};

pub(crate) const OUTPUT_FALLBACK_BUFFER: u64 = 4000;
const UNKNOWN_MODEL_MAX_OUTPUT_TOKENS: u64 = 64_000;
const CLAUDE_TOOL_PREFIX: &str = "_";
const CLAUDE_CODE_SYSTEM_INSTRUCTION: &str =
    "You are Claude Code, Anthropic's official CLI for Claude.";
const CLAUDE_BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";
const DEFAULT_CLAUDE_CODE_VERSION: &str = "2.1.280";
const NON_VISION_IMAGE_PLACEHOLDER: &str = "[image omitted: model does not support vision]";
const EMPTY_ERROR_TOOL_RESULT_TEXT: &str = "Tool failed with no output.";
const MAX_BREAKPOINTS: usize = 4;
const DECIMATION_INTERVAL: usize = 15;

pub(crate) fn claude_code_version() -> String {
    std::env::var("PI_AI_CLAUDE_CODE_VERSION")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_CLAUDE_CODE_VERSION.to_string())
}

fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// TS `createClaudeBillingHeader`.
fn billing_header(first_user_text: &str) -> String {
    let chars: Vec<char> = first_user_text.chars().collect();
    // JS indexes UTF-16 code units; BMP text is identical.
    let k: String = [4usize, 7, 20]
        .iter()
        .map(|&i| chars.get(i).copied().unwrap_or('0'))
        .collect();
    let version = claude_code_version();
    let suffix = &sha256_hex(format!("59cf53e54c78{k}{version}").as_bytes())[..3];
    format!("{CLAUDE_BILLING_HEADER_PREFIX} cc_version={version}.{suffix}; cc_entrypoint=cli; cch=00000;")
}

/// TS `getInstallId` (`~/.omp/install-id`, a UUID).
fn install_id() -> Option<String> {
    let home = std::env::var_os("HOME")?;
    let path = std::path::Path::new(&home).join(".omp").join("install-id");
    let text = std::fs::read_to_string(path).ok()?;
    let id = text.trim().to_string();
    let ok = id.len() == 36
        && id.chars().enumerate().all(|(i, c)| {
            if matches!(i, 8 | 13 | 18 | 23) {
                c == '-'
            } else {
                c.is_ascii_hexdigit()
            }
        });
    ok.then_some(id)
}

/// TS `deriveClaudeDeviceId`.
fn derive_device_id(install_id: &str, account_id: Option<&str>) -> String {
    let mut h = Sha256::new();
    match account_id.filter(|a| !a.is_empty()) {
        Some(account) => {
            h.update(b"omp-claude-device-id-v2");
            h.update([0u8]);
            h.update(install_id.as_bytes());
            h.update([0u8]);
            h.update(account.as_bytes());
        }
        None => {
            h.update(b"omp-claude-device-id-v1:");
            h.update(install_id.as_bytes());
        }
    }
    hex::encode(h.finalize())
}

fn is_claude_cloaking_user_id(user_id: &str) -> bool {
    let Some(rest) = user_id.strip_prefix("user_") else {
        return false;
    };
    let Some((hash, rest)) = rest.split_once("_account_") else {
        return false;
    };
    let Some((account, session)) = rest.split_once("_session_") else {
        return false;
    };
    let hex_len = |s: &str, n: usize| s.len() == n && s.chars().all(|c| c.is_ascii_hexdigit());
    let uuid = |s: &str| {
        s.len() == 36
            && s.chars().enumerate().all(|(i, c)| {
                if matches!(i, 8 | 13 | 18 | 23) {
                    c == '-'
                } else {
                    c.is_ascii_hexdigit()
                }
            })
    };
    hex_len(hash, 64) && uuid(account) && uuid(session)
}

fn is_claude_json_user_id(user_id: &str) -> bool {
    if !user_id.starts_with('{') {
        return false;
    }
    serde_json::from_str::<Value>(user_id)
        .ok()
        .and_then(|v| {
            v.get("session_id")
                .and_then(Value::as_str)
                .map(|s| !s.is_empty())
        })
        .unwrap_or(false)
}

/// TS `resolveAnthropicMetadataUserId`.
fn resolve_metadata_user_id(
    user_id: Option<&str>,
    is_oauth: bool,
    session_id: Option<&str>,
    account_id: Option<&str>,
) -> Option<String> {
    if let Some(user_id) = user_id {
        if !is_oauth || is_claude_cloaking_user_id(user_id) || is_claude_json_user_id(user_id) {
            return Some(user_id.to_string());
        }
    }
    if !is_oauth {
        return None;
    }
    let mut map = Map::new();
    map.insert(
        "device_id".into(),
        json!(derive_device_id(
            &install_id().unwrap_or_default(),
            account_id
        )),
    );
    map.insert(
        "session_id".into(),
        json!(session_id.map_or_else(|| uuid_v4(), str::to_string)),
    );
    if let Some(account) = account_id.filter(|a| !a.is_empty()) {
        map.insert("account_uuid".into(), json!(account));
    }
    Some(Value::Object(map).to_string())
}

pub(crate) fn uuid_v4() -> String {
    use rand::RngCore;
    let mut b = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut b);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let h = hex::encode(b);
    format!(
        "{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    )
}

// ---------------------------------------------------------------- helpers

fn resolve_retention(explicit: Option<CacheRetention>, fallback: CacheRetention) -> CacheRetention {
    if let Some(retention) = explicit {
        return retention;
    }
    match std::env::var("PI_CACHE_RETENTION").as_deref() {
        Ok("long") => CacheRetention::Long,
        Ok("short") => CacheRetention::Short,
        Ok("none") => CacheRetention::None,
        _ => fallback,
    }
}

/// TS `getCacheControl`: the `cache_control` object, `None` when caching is off.
pub(crate) fn cache_control_value(
    view: &ModelView,
    retention: Option<CacheRetention>,
    is_oauth: bool,
) -> Option<Value> {
    let long_ok = view.is("supportsLongCacheRetention");
    let default = if is_oauth && long_ok {
        CacheRetention::Long
    } else {
        CacheRetention::Short
    };
    let retention = resolve_retention(retention, default);
    if retention == CacheRetention::None {
        return None;
    }
    let mut map = Map::new();
    map.insert("type".into(), json!("ephemeral"));
    if retention == CacheRetention::Long && long_ok {
        map.insert("ttl".into(), json!("1h"));
    }
    Some(Value::Object(map))
}

fn is_builtin_tool_name(name: &str) -> bool {
    matches!(
        name.to_lowercase().as_str(),
        "web_search" | "code_execution" | "text_editor" | "computer"
    )
}

/// TS `encodeAnthropicToolName`.
pub(crate) fn encode_tool_name(name: &str, is_oauth: bool, escape_builtin: bool) -> String {
    if escape_builtin {
        return format!("{CLAUDE_TOOL_PREFIX}{name}");
    }
    if is_oauth && !is_builtin_tool_name(name) {
        return format!("{CLAUDE_TOOL_PREFIX}{name}");
    }
    name.to_string()
}

/// TS `decodeAnthropicToolName`: the wire-only prefix is stripped from OAuth / escaped tool names.
pub(crate) fn decode_tool_name(name: &str, is_oauth: bool, escape_builtin: bool) -> String {
    if (is_oauth || escape_builtin) && name.to_lowercase().starts_with(CLAUDE_TOOL_PREFIX) {
        return name[CLAUDE_TOOL_PREFIX.len()..].to_string();
    }
    name.to_string()
}

fn normalize_system_prompts(prompt: Option<&str>) -> Vec<String> {
    prompt
        .into_iter()
        .filter(|p| !p.trim().is_empty())
        .map(str::to_string)
        .collect()
}

fn text_block(text: &str) -> Value {
    json!({ "type": "text", "text": text })
}

// ---------------------------------------------------------------- content blocks

fn normalize_image_media_type(mime: &str) -> Option<&'static str> {
    match mime.trim().to_lowercase().as_str() {
        "image/jpg" | "image/jpeg" => Some("image/jpeg"),
        "image/png" => Some("image/png"),
        "image/gif" => Some("image/gif"),
        "image/webp" => Some("image/webp"),
        _ => None,
    }
}

/// TS `convertContentBlocks`: a string for text-only models, else the block array.
fn convert_content_blocks(content: &[UserOrToolContent], supports_images: bool) -> Value {
    let mut blocks: Vec<Value> = Vec::new();
    let mut saw_text = false;
    let mut saw_image = false;
    for block in content {
        match block {
            UserOrToolContent::Text(t) => {
                if t.text.trim().is_empty() {
                    continue;
                }
                saw_text = true;
                blocks.push(text_block(&t.text));
            }
            UserOrToolContent::Image(img) => {
                if !supports_images {
                    blocks.push(text_block(NON_VISION_IMAGE_PLACEHOLDER));
                    continue;
                }
                let source = if let Some(url) = img
                    .rest
                    .get("url")
                    .and_then(Value::as_str)
                    .filter(|u| !u.is_empty())
                {
                    json!({ "type": "url", "url": url })
                } else if let Some(media_type) = normalize_image_media_type(&img.mime_type) {
                    json!({ "type": "base64", "media_type": media_type, "data": img.data })
                } else {
                    blocks.push(text_block(&format!(
                        "[unsupported image: {}]",
                        img.mime_type
                    )));
                    continue;
                };
                saw_image = true;
                blocks.push(json!({ "type": "image", "source": source }));
            }
            UserOrToolContent::Raw(v) => {
                let text = v.to_string();
                if text.trim().is_empty() {
                    continue;
                }
                saw_text = true;
                blocks.push(text_block(&text));
            }
        }
    }
    if !supports_images {
        let joined: Vec<&str> = blocks
            .iter()
            .filter(|b| b["type"] == "text")
            .filter_map(|b| b["text"].as_str())
            .collect();
        return Value::String(joined.join("\n"));
    }
    if saw_image && !saw_text {
        blocks.insert(0, text_block("(see attached image)"));
    }
    Value::Array(blocks)
}

fn is_empty_tool_result_content(content: &Value) -> bool {
    match content {
        Value::String(s) => s.trim().is_empty(),
        Value::Array(a) => a.is_empty(),
        _ => false,
    }
}

fn build_tool_result_block(
    view: &ModelView,
    model: &Model,
    msg: &crate::types::ToolResultMessage,
    hoisted: &mut Vec<Value>,
) -> Value {
    let mut content = convert_content_blocks(&msg.content, model.supports_image_input());
    if msg.is_error {
        if let Value::Array(blocks) = &content {
            if blocks.iter().any(|b| b["type"] == "image") {
                let (images, texts): (Vec<Value>, Vec<Value>) =
                    blocks.iter().cloned().partition(|b| b["type"] == "image");
                hoisted.extend(images);
                content = Value::Array(texts);
            }
        }
    }
    if matches!(&content, Value::Array(a) if a.is_empty()) {
        content = Value::String(String::new());
    }
    if msg.is_error && is_empty_tool_result_content(&content) {
        content = if content.is_string() {
            Value::String(EMPTY_ERROR_TOOL_RESULT_TEXT.into())
        } else {
            json!([text_block(EMPTY_ERROR_TOOL_RESULT_TEXT)])
        };
    }
    let mut block = Map::new();
    block.insert("type".into(), json!("tool_result"));
    block.insert("tool_use_id".into(), json!(msg.tool_call_id));
    block.insert("content".into(), content);
    block.insert("is_error".into(), json!(msg.is_error));
    if view.is("requiresToolResultId") {
        block.insert("id".into(), json!(msg.tool_call_id));
    }
    Value::Object(block)
}

/// `renderDemotedThinking`: thinking replayed as plain text for models that cannot take it as a
/// native block. The Anthropic dialect keeps the text verbatim.
fn render_demoted_thinking(model_id: &str, text: &str) -> String {
    let class = crate::compat::taxonomy::classify_model(crate::compat::rules(), "", model_id, true)
        .map(|i| i.class)
        .unwrap_or_default();
    match class.as_str() {
        "anthropic" => text.to_string(),
        _ => format!("<think>\n{text}\n</think>"),
    }
}

/// One wire message plus the TS provenance marker (`kConversationalUser`).
struct Param {
    value: Value,
    conversational: bool,
}

impl Param {
    fn role(&self) -> &str {
        self.value["role"].as_str().unwrap_or("")
    }
}

/// TS `convertAnthropicMessages` for the message shapes this runtime carries.
fn convert_messages(
    view: &ModelView,
    model: &Model,
    transformed: &[Message],
    is_oauth: bool,
) -> Vec<Param> {
    let mut params: Vec<Param> = Vec::new();
    // Indices of params emitted from `developer` messages (upgraded to `system` below).
    let mut developer_params: Vec<usize> = Vec::new();
    let escape = view.is("escapeBuiltinToolNames");
    let supports_images = model.supports_image_input();
    let mut i = 0;
    while i < transformed.len() {
        match &transformed[i] {
            Message::User(user) | Message::Developer(user) => {
                let content: Value = match &user.content {
                    UserMessageContent::Text(text) => {
                        if text.trim().is_empty() {
                            i += 1;
                            continue;
                        }
                        Value::String(text.clone())
                    }
                    UserMessageContent::Blocks(blocks) => {
                        let converted = convert_content_blocks(blocks, supports_images);
                        match &converted {
                            Value::String(s) if s.trim().is_empty() => {
                                i += 1;
                                continue;
                            }
                            Value::Array(a) if a.is_empty() => {
                                i += 1;
                                continue;
                            }
                            _ => converted,
                        }
                    }
                };
                let is_developer = matches!(&transformed[i], Message::Developer(_));
                if is_developer {
                    developer_params.push(params.len());
                }
                // Conversational = a real user turn: not a developer message, not agent-authored
                // (`synthetic`, or `attribution: "agent"`), not a synthetic user note.
                let synthetic = user.rest.get(SYNTHETIC_USER_KEY) == Some(&Value::Bool(true));
                let agent_authored = user.rest.get("synthetic") == Some(&Value::Bool(true))
                    || user.rest.get("attribution").and_then(Value::as_str) == Some("agent");
                params.push(Param {
                    value: json!({ "role": "user", "content": content }),
                    conversational: !is_developer && !agent_authored && !synthetic,
                });
            }
            Message::Assistant(assistant) => {
                let mut blocks: Vec<Value> = Vec::new();
                let has_signed_thinking = assistant.content.iter().any(|b| {
                    matches!(b, AssistantContent::Thinking(t)
                        if !t.redacted.unwrap_or(false)
                            && t.thinking_signature.as_deref().is_some_and(|s| !s.trim().is_empty()))
                });
                for block in &assistant.content {
                    match block {
                        AssistantContent::Text(t) => {
                            if t.text.trim().is_empty() {
                                continue;
                            }
                            blocks.push(text_block(&t.text));
                        }
                        AssistantContent::Thinking(t) if t.redacted.unwrap_or(false) => {
                            let data = t.thinking_signature.as_deref().unwrap_or("");
                            if data.trim().is_empty() {
                                continue;
                            }
                            blocks.push(json!({ "type": "redacted_thinking", "data": data }));
                        }
                        AssistantContent::Thinking(t) => {
                            let sig = t
                                .thinking_signature
                                .as_deref()
                                .filter(|s| !s.trim().is_empty());
                            if has_signed_thinking {
                                match sig {
                                    None => {
                                        if t.thinking.trim().is_empty() {
                                            continue;
                                        }
                                        blocks.push(text_block(&render_demoted_thinking(&model.id, &t.thinking)));
                                    }
                                    Some(sig) => blocks.push(
                                        json!({ "type": "thinking", "thinking": t.thinking, "signature": sig }),
                                    ),
                                }
                                continue;
                            }
                            if t.thinking.trim().is_empty() {
                                continue;
                            }
                            match sig {
                                None => {
                                    if view.is("replayUnsignedThinking") {
                                        blocks.push(json!({ "type": "thinking", "thinking": t.thinking, "signature": "" }));
                                    } else {
                                        blocks.push(text_block(&render_demoted_thinking(&model.id, &t.thinking)));
                                    }
                                }
                                Some(sig) => blocks
                                    .push(json!({ "type": "thinking", "thinking": t.thinking, "signature": sig })),
                            }
                        }
                        AssistantContent::ToolCall(call) => {
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": call.id,
                                "name": encode_tool_name(&call.name, is_oauth, escape),
                                "input": Value::Object(call.arguments.clone()),
                            }));
                        }
                    }
                }
                // tool_use blocks after non-thinking blocks are hoisted behind the text.
                let mut saw_tool_use = false;
                let mut needs_partition = false;
                for b in &blocks {
                    let ty = b["type"].as_str().unwrap_or("");
                    if ty == "tool_use" {
                        saw_tool_use = true;
                    } else if saw_tool_use && ty != "thinking" && ty != "redacted_thinking" {
                        needs_partition = true;
                        break;
                    }
                }
                if needs_partition {
                    let (tool_use, rest): (Vec<Value>, Vec<Value>) =
                        blocks.into_iter().partition(|b| b["type"] == "tool_use");
                    blocks = rest.into_iter().chain(tool_use).collect();
                }
                if blocks.is_empty() {
                    i += 1;
                    continue;
                }
                params.push(Param {
                    value: json!({ "role": "assistant", "content": blocks }),
                    conversational: false,
                });
            }
            Message::ToolResult(first) => {
                let mut results: Vec<Value> = Vec::new();
                let mut hoisted: Vec<Value> = Vec::new();
                results.push(build_tool_result_block(view, model, first, &mut hoisted));
                let mut j = i + 1;
                while let Some(Message::ToolResult(next)) = transformed.get(j) {
                    results.push(build_tool_result_block(view, model, next, &mut hoisted));
                    j += 1;
                }
                i = j - 1;
                if !hoisted.is_empty() {
                    results.push(text_block(
                        "Attached image(s) from the tool result(s) above:",
                    ));
                    results.extend(hoisted);
                }
                params.push(Param {
                    value: json!({ "role": "user", "content": results }),
                    conversational: false,
                });
            }
        }
        i += 1;
    }
    // Upgrade developer-origin params to mid-conversation `system` messages where Anthropic's
    // placement rules allow it: right after a `user` turn, and last or followed by an `assistant`.
    if !developer_params.is_empty() && view.is("supportsMidConversationSystem") {
        for &idx in developer_params.iter().rev() {
            let follows_user = idx > 0 && params[idx - 1].role() == "user";
            let next = idx + 1;
            let last_or_before_assistant =
                next >= params.len() || params[next].role() == "assistant";
            let system_compatible = match &params[idx].value["content"] {
                Value::String(_) => true,
                Value::Array(blocks) => blocks.iter().all(|b| {
                    matches!(
                        b["type"].as_str(),
                        Some("text" | "tool_addition" | "tool_removal")
                    )
                }),
                _ => false,
            };
            if follows_user && last_or_before_assistant && system_compatible {
                params[idx].value["role"] = json!("system");
            }
        }
    }
    // Assistant followed by assistant needs a user turn between them.
    let mut k = params.len();
    while k > 1 {
        k -= 1;
        if params[k].role() == "assistant" && params[k - 1].role() == "assistant" {
            params.insert(
                k,
                Param {
                    value: json!({ "role": "user", "content": "Continue." }),
                    conversational: false,
                },
            );
        }
    }
    if params.last().is_some_and(|p| p.role() == "assistant") {
        params.push(Param {
            value: json!({ "role": "user", "content": "Continue." }),
            conversational: false,
        });
    }
    params
}

// ---------------------------------------------------------------- tools

const SCHEMA_UNIVERSAL_KEEP: &[&str] = &[
    "$ref",
    "$defs",
    "$schema",
    "definitions",
    "type",
    "anyOf",
    "allOf",
    "enum",
    "const",
    "description",
    "title",
    "default",
    "nullable",
];
const SCHEMA_OBJECT_KEEP: &[&str] = &["properties", "required", "additionalProperties"];
const SCHEMA_ARRAY_KEEP: &[&str] = &["items", "prefixItems", "minItems"];
const SCHEMA_STRING_KEEP: &[&str] = &["format"];
const SCHEMA_STRING_FORMATS: &[&str] = &[
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];
const COMBINATOR_KEYS: [&str; 3] = ["anyOf", "allOf", "oneOf"];
const STRICT_TOOL_ALLOWLIST: &[&str] = &["python", "edit", "find"];
const MAX_STRICT_TOOLS: usize = 20;
const MAX_STRICT_OPTIONAL: i64 = 24;
const MAX_STRICT_UNION: i64 = 16;

fn is_array_node(schema: &Map<String, Value>) -> bool {
    match schema.get("type") {
        Some(Value::String(t)) if t == "array" => return true,
        Some(Value::Array(t)) => {
            if t.iter().any(|x| x == "array") && !t.iter().any(|x| x == "object") {
                return true;
            }
        }
        _ => {}
    }
    schema.contains_key("items") || schema.get("prefixItems").is_some_and(Value::is_array)
}

fn is_object_node(schema: &Map<String, Value>) -> bool {
    if is_array_node(schema) {
        return false;
    }
    match schema.get("type") {
        Some(Value::String(t)) if t == "object" => return true,
        Some(Value::Array(t)) if t.iter().any(|x| x == "object") => return true,
        _ => {}
    }
    schema.get("properties").is_some_and(Value::is_object)
}

fn pick_scalar_type(ty: Option<&Value>) -> Option<String> {
    match ty? {
        Value::String(s) => Some(s.clone()),
        Value::Array(a) => a
            .iter()
            .find_map(|e| e.as_str().filter(|s| *s != "null").map(str::to_string)),
        _ => None,
    }
}

fn effective_scalar_type(schema: &Map<String, Value>) -> Option<String> {
    if let Some(t) = pick_scalar_type(schema.get("type")) {
        return Some(t);
    }
    if schema.get("properties").is_some_and(Value::is_object) {
        return Some("object".into());
    }
    if schema.contains_key("items") || schema.get("prefixItems").is_some_and(Value::is_array) {
        return Some("array".into());
    }
    None
}

fn normalize_tool_schema_node(schema: &Value, is_root: bool) -> Value {
    match schema {
        Value::Array(a) => Value::Array(
            a.iter()
                .map(|e| normalize_tool_schema_node(e, false))
                .collect(),
        ),
        Value::Object(obj) => {
            let scalar = effective_scalar_type(obj);
            let per_type: Option<&[&str]> = match scalar.as_deref() {
                Some("object") => Some(SCHEMA_OBJECT_KEEP),
                Some("array") => Some(SCHEMA_ARRAY_KEEP),
                Some("string") => Some(SCHEMA_STRING_KEEP),
                _ => None,
            };
            let mut result = Map::new();
            let mut spill: Vec<(String, Value)> = Vec::new();
            for (key, value) in obj {
                let root_combinator = is_root && COMBINATOR_KEYS.contains(&key.as_str());
                if !root_combinator
                    && (SCHEMA_UNIVERSAL_KEEP.contains(&key.as_str())
                        || per_type.is_some_and(|k| k.contains(&key.as_str())))
                {
                    result.insert(key.clone(), value.clone());
                } else {
                    spill.push((key.clone(), value.clone()));
                }
            }
            if scalar.as_deref() == Some("string") {
                if let Some(Value::String(format)) = result.get("format") {
                    if !SCHEMA_STRING_FORMATS.contains(&format.as_str()) {
                        let f = format.clone();
                        spill.push(("format".into(), Value::String(f)));
                        result.shift_remove("format");
                    }
                }
            }
            if scalar.as_deref() == Some("array") {
                if let Some(min_items) = result.get("minItems").cloned() {
                    let keep = min_items.as_f64().is_some_and(|n| n == 0.0 || n == 1.0);
                    if !keep {
                        spill.push(("minItems".into(), min_items));
                        result.shift_remove("minItems");
                    }
                }
            }
            if scalar.as_deref() == Some("object") && !result.contains_key("additionalProperties") {
                result.insert("additionalProperties".into(), Value::Bool(false));
            }
            if let Some(Value::Object(props)) = result.get("properties").cloned() {
                let mut normalized = Map::new();
                for (name, p) in &props {
                    normalized.insert(name.clone(), normalize_tool_schema_node(p, false));
                }
                result.insert("properties".into(), Value::Object(normalized));
            }
            if let Some(Value::Object(ap)) = result.get("additionalProperties").cloned() {
                let n = normalize_tool_schema_node(&Value::Object(ap), false);
                if n.as_object().is_some_and(Map::is_empty) {
                    result.insert("additionalProperties".into(), Value::Bool(true));
                } else {
                    result.insert("additionalProperties".into(), n);
                }
            }
            match result.get("items").cloned() {
                Some(Value::Array(items)) => {
                    result.insert(
                        "items".into(),
                        Value::Array(
                            items
                                .iter()
                                .map(|i| normalize_tool_schema_node(i, false))
                                .collect(),
                        ),
                    );
                }
                Some(items @ Value::Object(_)) => {
                    result.insert("items".into(), normalize_tool_schema_node(&items, false));
                }
                _ => {}
            }
            if let Some(Value::Array(items)) = result.get("prefixItems").cloned() {
                result.insert(
                    "prefixItems".into(),
                    Value::Array(
                        items
                            .iter()
                            .map(|i| normalize_tool_schema_node(i, false))
                            .collect(),
                    ),
                );
            }
            for key in COMBINATOR_KEYS {
                if let Some(Value::Array(variants)) = result.get(key).cloned() {
                    result.insert(
                        key.into(),
                        Value::Array(
                            variants
                                .iter()
                                .map(|v| normalize_tool_schema_node(v, false))
                                .collect(),
                        ),
                    );
                }
            }
            for defs_key in ["$defs", "definitions"] {
                if let Some(Value::Object(defs)) = result.get(defs_key).cloned() {
                    let mut normalized = Map::new();
                    for (name, d) in &defs {
                        normalized.insert(name.clone(), normalize_tool_schema_node(d, false));
                    }
                    result.insert(defs_key.into(), Value::Object(normalized));
                }
            }
            spill_to_description(&mut result, &spill);
            Value::Object(result)
        }
        other => other.clone(),
    }
}

struct StrictBudget {
    optional_remaining: i64,
    union_remaining: i64,
    optional_count: i64,
    union_count: i64,
}

fn has_union_type(schema: &Map<String, Value>) -> bool {
    schema.get("type").is_some_and(Value::is_array)
        || schema.get("anyOf").is_some_and(Value::is_array)
}

fn has_null_variant(schema: &Map<String, Value>) -> bool {
    if let Some(Value::Array(t)) = schema.get("type") {
        if t.iter().any(|x| x == "null") {
            return true;
        }
    }
    matches!(schema.get("anyOf"), Some(Value::Array(a)) if a.iter().any(|v| v.get("type").is_some_and(|t| t == "null")))
}

fn has_schema_defining_keyword(schema: &Map<String, Value>) -> bool {
    [
        "type",
        "properties",
        "additionalProperties",
        "items",
        "prefixItems",
        "enum",
        "const",
        "$ref",
    ]
    .iter()
    .any(|k| schema.contains_key(*k))
        || COMBINATOR_KEYS.iter().any(|k| schema.contains_key(*k))
        || schema.contains_key("$defs")
        || schema.contains_key("definitions")
}

fn make_nullable(schema: &Value, budget: &mut StrictBudget) -> Option<Value> {
    if let Value::Object(obj) = schema {
        if has_null_variant(obj) {
            return Some(schema.clone());
        }
        if let Some(Value::Array(any_of)) = obj.get("anyOf") {
            let mut next = obj.clone();
            let mut v = any_of.clone();
            v.push(json!({ "type": "null" }));
            next.insert("anyOf".into(), Value::Array(v));
            return Some(Value::Object(next));
        }
        if let Some(Value::Array(ty)) = obj.get("type") {
            let mut next = obj.clone();
            let mut v = ty.clone();
            v.push(json!("null"));
            next.insert("type".into(), Value::Array(v));
            return Some(Value::Object(next));
        }
    }
    if budget.union_remaining <= 0 {
        return None;
    }
    budget.union_remaining -= 1;
    budget.union_count += 1;
    Some(json!({ "anyOf": [schema.clone(), { "type": "null" }] }))
}

fn normalize_strict_node(schema: &Value, budget: &mut StrictBudget) -> Option<Value> {
    match schema {
        Value::Array(a) => {
            let mut out = Vec::with_capacity(a.len());
            for e in a {
                out.push(normalize_strict_node(e, budget)?);
            }
            Some(Value::Array(out))
        }
        Value::Object(obj) => {
            if !has_schema_defining_keyword(obj) {
                return None;
            }
            if is_object_node(obj) && obj.get("additionalProperties") != Some(&Value::Bool(false)) {
                return None;
            }
            let mut result = obj.clone();
            if has_union_type(&result) {
                if budget.union_remaining <= 0 {
                    return None;
                }
                budget.union_remaining -= 1;
                budget.union_count += 1;
            }
            if let Some(Value::Object(props)) = result.get("properties").cloned() {
                let original_required: Vec<String> = result
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                let mut properties = Map::new();
                let mut required: Vec<Value> = Vec::new();
                for (name, prop) in &props {
                    let normalized = normalize_strict_node(prop, budget)?;
                    if original_required.contains(name) {
                        properties.insert(name.clone(), normalized);
                        required.push(json!(name));
                        continue;
                    }
                    if budget.optional_remaining > 0 {
                        budget.optional_remaining -= 1;
                        budget.optional_count += 1;
                        properties.insert(name.clone(), normalized);
                        continue;
                    }
                    let nullable = make_nullable(&normalized, budget)?;
                    properties.insert(name.clone(), nullable);
                    required.push(json!(name));
                }
                result.insert("properties".into(), Value::Object(properties));
                result.insert("required".into(), Value::Array(required));
            }
            match result.get("items").cloned() {
                Some(items @ (Value::Array(_) | Value::Object(_))) => {
                    result.insert("items".into(), normalize_strict_node(&items, budget)?);
                }
                _ => {}
            }
            if let Some(Value::Array(p)) = result.get("prefixItems").cloned() {
                result.insert(
                    "prefixItems".into(),
                    normalize_strict_node(&Value::Array(p), budget)?,
                );
            }
            for key in COMBINATOR_KEYS {
                if let Some(Value::Array(v)) = result.get(key).cloned() {
                    result.insert(key.into(), normalize_strict_node(&Value::Array(v), budget)?);
                }
            }
            for defs_key in ["$defs", "definitions"] {
                if let Some(Value::Object(defs)) = result.get(defs_key).cloned() {
                    let mut out = Map::new();
                    for (name, d) in &defs {
                        out.insert(name.clone(), normalize_strict_node(d, budget)?);
                    }
                    result.insert(defs_key.into(), Value::Object(out));
                }
            }
            Some(Value::Object(result))
        }
        other => Some(other.clone()),
    }
}

fn has_strict_incompatible_keyword(schema: &Value) -> bool {
    match schema {
        Value::Array(a) => a.iter().any(has_strict_incompatible_keyword),
        Value::Object(o) => {
            [
                "oneOf",
                "allOf",
                "$ref",
                "patternProperties",
                "propertyNames",
            ]
            .iter()
            .any(|k| o.contains_key(*k))
                || o.values().any(has_strict_incompatible_keyword)
        }
        _ => false,
    }
}

fn base_tool_input_schema(tool: &crate::types::Tool) -> Value {
    let wire = tool_wire_schema(&tool.parameters);
    let mut obj = wire.as_object().cloned().unwrap_or_default();
    obj.insert("type".into(), json!("object"));
    let props = obj
        .get("properties")
        .filter(|p| p.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    obj.insert("properties".into(), props);
    let required: Vec<Value> = obj
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter(|v| v.is_string()).cloned().collect())
        .unwrap_or_default();
    obj.insert("required".into(), Value::Array(required));
    normalize_tool_schema_node(&Value::Object(obj), true)
}

/// Eager input streaming skips the API's JSON validation. Models then sometimes emit a nested
/// object member unquoted after a long string argument (observed: SoL-Pi `write.then_run` after
/// ~38 KB `content`), the whole call parses to `{}` and the write is re-generated. Tools with
/// object-typed properties stream buffered (validated) instead.
fn has_object_property(schema: &Value) -> bool {
    schema
        .get("properties")
        .and_then(Value::as_object)
        .is_some_and(|props| {
            props.values().any(|p| {
                p.get("type").and_then(Value::as_str) == Some("object")
                    || p.get("properties").is_some()
                    || p.get("anyOf").and_then(Value::as_array).is_some_and(|alts| {
                        alts.iter().any(|a| a.get("type").and_then(Value::as_str) == Some("object"))
                    })
            })
        })
}

/// TS `convertTools`.
fn convert_tools(
    tools: &[crate::types::Tool],
    is_oauth: bool,
    disable_strict: bool,
    eager_streaming: bool,
    escape_builtin: bool,
) -> Vec<Value> {
    let mut plans: Vec<(Value, bool)> = tools
        .iter()
        .map(|t| (base_tool_input_schema(t), false))
        .collect();
    let no_strict = matches!(std::env::var("PI_NO_STRICT").as_deref(), Ok("1" | "true"));
    if !no_strict && !disable_strict {
        let candidates: Vec<usize> = tools
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                STRICT_TOOL_ALLOWLIST.contains(&t.name.as_str())
                    && !has_strict_incompatible_keyword(&tool_wire_schema(&t.parameters))
            })
            .map(|(i, _)| i)
            .collect();
        let (mut strict_tools, mut opt_used, mut union_used) = (0usize, 0i64, 0i64);
        for index in candidates {
            if strict_tools >= MAX_STRICT_TOOLS {
                break;
            }
            let mut budget = StrictBudget {
                optional_remaining: MAX_STRICT_OPTIONAL - opt_used,
                union_remaining: MAX_STRICT_UNION - union_used,
                optional_count: 0,
                union_count: 0,
            };
            let Some(normalized) = normalize_strict_node(&plans[index].0, &mut budget) else {
                continue;
            };
            if !normalized.is_object() {
                continue;
            }
            plans[index] = (normalized, true);
            strict_tools += 1;
            opt_used += budget.optional_count;
            union_used += budget.union_count;
        }
    }
    tools
        .iter()
        .zip(plans)
        .map(|(tool, (schema, strict))| {
            let mut entry = Map::new();
            entry.insert(
                "name".into(),
                json!(encode_tool_name(&tool.name, is_oauth, escape_builtin)),
            );
            entry.insert("description".into(), json!(tool.description));
            let eager = eager_streaming && !has_object_property(&schema);
            entry.insert("input_schema".into(), schema);
            if eager {
                entry.insert("eager_input_streaming".into(), json!(true));
            }
            if strict {
                entry.insert("strict".into(), json!(true));
            }
            Value::Object(entry)
        })
        .collect()
}

// ---------------------------------------------------------------- thinking

pub(crate) fn uses_adaptive_thinking_tag_only(view: &ModelView) -> bool {
    if view.thinking_str("mode") != Some("anthropic-adaptive") {
        return false;
    }
    let efforts = view.efforts();
    if view
        .thinking
        .as_ref()
        .and_then(|t| t.get("effortMap"))
        .is_none()
    {
        return false;
    }
    !efforts.is_empty()
        && efforts
            .iter()
            .all(|e| view.effort_map(e) == Some("adaptive"))
}

pub(crate) fn uses_budget_thinking(view: &ModelView) -> bool {
    view.thinking_str("mode") != Some("anthropic-adaptive") || view.is("disableAdaptiveThinking")
}

pub(crate) fn is_adaptive_only_thinking(view: &ModelView) -> bool {
    view.thinking_str("mode") == Some("anthropic-adaptive")
        && !view.is("disableAdaptiveThinking")
        && !uses_adaptive_thinking_tag_only(view)
}

/// TS `requireSupportedEffort`.
pub(crate) fn require_supported_effort(
    model: &Model,
    view: &ModelView,
    effort: &str,
) -> Result<(), String> {
    if !model.reasoning {
        return Err(format!(
            "Model {}/{} does not support thinking",
            model.provider, model.id
        ));
    }
    let levels = view.efforts();
    if !levels.contains(&effort) {
        return Err(format!(
            "Thinking effort {effort} is not supported by {}/{}. Supported efforts: {}",
            model.provider,
            model.id,
            levels.join(", ")
        ));
    }
    Ok(())
}

/// TS `mapEffortToAnthropicAdaptiveEffort`.
pub(crate) fn map_effort_to_adaptive(
    model: &Model,
    view: &ModelView,
    effort: &str,
) -> Result<String, String> {
    require_supported_effort(model, view, effort)?;
    let mapped = view.effort_map(effort).unwrap_or(effort);
    Ok(if mapped == "minimal" {
        "low".to_string()
    } else {
        mapped.to_string()
    })
}

fn resolve_adaptive_effort(
    model: &Model,
    view: &ModelView,
    options: &AnthropicOptions,
) -> Result<Option<String>, String> {
    if let Some(effort) = options.effort {
        return Ok(Some(if uses_adaptive_thinking_tag_only(view) {
            "adaptive".into()
        } else {
            effort.as_str().into()
        }));
    }
    let Some(requested) = options.reasoning else {
        return Ok(None);
    };
    map_effort_to_adaptive(model, view, requested.wire_name()).map(Some)
}

fn budget_thinking_output(max_tokens: Option<u64>, budget: u64, max_allowed: u64) -> (u64, i64) {
    let current = max_tokens.unwrap_or(max_allowed).min(max_allowed);
    let raised = current
        .max(budget + OUTPUT_FALLBACK_BUFFER)
        .min(max_allowed);
    let new_budget = (budget as i64).min(raised as i64 - OUTPUT_FALLBACK_BUFFER as i64);
    (raised, new_budget)
}

// ---------------------------------------------------------------- caching

fn apply_cache_control_to_last_block(blocks: &mut [Value], cache_control: &Value) -> bool {
    for block in blocks.iter_mut().rev() {
        let ty = block["type"].as_str().unwrap_or("");
        if matches!(
            ty,
            "thinking" | "redacted_thinking" | "fallback" | "tool_addition" | "tool_removal"
        ) {
            continue;
        }
        if block.get("cache_control").is_some_and(|c| !c.is_null()) {
            return false;
        }
        block["cache_control"] = cache_control.clone();
        return true;
    }
    false
}

fn apply_cache_control_to_message(message: &mut Value, cache_control: &Value) -> bool {
    match message.get_mut("content") {
        Some(Value::String(text)) => {
            let t = std::mem::take(text);
            message["content"] =
                json!([{ "type": "text", "text": t, "cache_control": cache_control }]);
            true
        }
        Some(Value::Array(blocks)) => apply_cache_control_to_last_block(blocks, cache_control),
        _ => false,
    }
}

/// First bytes of the context-strategy slots message (`omp-cli/src/context.rs`).
const SLOTS_MARKER: &str = "<context-slots>";

fn is_slots_message(p: &Param) -> bool {
    if p.role() != "user" {
        return false;
    }
    match &p.value["content"] {
        Value::String(t) => t.starts_with(SLOTS_MARKER),
        Value::Array(b) => b
            .first()
            .and_then(|b| b["text"].as_str())
            .is_some_and(|t| t.starts_with(SLOTS_MARKER)),
        _ => false,
    }
}

fn apply_prompt_caching(messages: &mut [Param], head_breakpoints: usize, cache_control: &Value) {
    let budget = MAX_BREAKPOINTS.saturating_sub(head_breakpoints);
    if budget == 0 || messages.is_empty() {
        return;
    }
    let trailing_index = messages.len() - 1;
    let trailing = &messages[trailing_index];
    let has_pad = trailing.role() == "user"
        && trailing.value["content"] == "Continue."
        && !trailing.conversational
        && trailing_index > 0
        && messages[trailing_index - 1].role() == "assistant";
    let message_end = if has_pad {
        trailing_index - 1
    } else {
        trailing_index
    };

    let mut stable_user_turns = 0usize;
    let mut trailing_candidates: Vec<usize> = Vec::new();
    for index in (0..=message_end).rev() {
        if trailing_candidates.len() < 2 {
            trailing_candidates.push(index);
        }
        if messages[index].conversational {
            stable_user_turns += 1;
        }
    }
    let stable_message_end = message_end as i64;
    let mut applied = 0usize;
    if let Some(&first) = trailing_candidates.first() {
        if applied < budget
            && apply_cache_control_to_message(&mut messages[first].value, cache_control)
        {
            applied += 1;
        }
    }
    // CLM slot region (rs-only; the marker never occurs on TS wire bodies): keep a breakpoint on
    // the slots message so prefix + unchanged slots stay cached while the tail churns.
    if let Some(slots) = messages[..=message_end].iter().rposition(is_slots_message) {
        if applied < budget
            && Some(slots) != trailing_candidates.first().copied()
            && apply_cache_control_to_message(&mut messages[slots].value, cache_control)
        {
            applied += 1;
        }
    }
    if stable_user_turns >= DECIMATION_INTERVAL {
        let mut since = stable_user_turns % DECIMATION_INTERVAL;
        let mut index = stable_message_end;
        while index >= 0 && applied < budget {
            let i = index as usize;
            index -= 1;
            if !messages[i].conversational {
                continue;
            }
            if since > 0 {
                since -= 1;
                continue;
            }
            since = DECIMATION_INTERVAL - 1;
            if apply_cache_control_to_message(&mut messages[i].value, cache_control) {
                applied += 1;
            }
        }
    }
    for index in trailing_candidates {
        if applied >= budget {
            break;
        }
        if apply_cache_control_to_message(&mut messages[index].value, cache_control) {
            applied += 1;
        }
    }
}

const VOLATILE_SYSTEM_MARKERS: [&str; 2] = ["<memories>", "<project-context>"];

fn apply_head_caching(
    system: &mut Option<Vec<Value>>,
    tools: &mut Option<Vec<Value>>,
    cache_control: &Value,
) {
    if let Some(tools) = tools {
        if !tools.is_empty()
            && !tools
                .iter()
                .any(|t| t.get("cache_control").is_some_and(|c| !c.is_null()))
        {
            for tool in tools.iter_mut().rev() {
                if tool.get("defer_loading").is_some_and(|d| d == true) {
                    continue;
                }
                tool["cache_control"] = cache_control.clone();
                break;
            }
        }
    }
    if let Some(system) = system {
        if system.is_empty() {
            return;
        }
        let suffix_start = system
            .iter()
            .position(|b| {
                let text = b["text"].as_str().unwrap_or("");
                VOLATILE_SYSTEM_MARKERS.iter().any(|m| text.starts_with(m))
            })
            .unwrap_or(system.len());
        let anchor = if suffix_start == 0 {
            system.len() - 1
        } else {
            suffix_start - 1
        };
        if system[anchor]
            .get("cache_control")
            .map_or(true, Value::is_null)
        {
            for block in system.iter_mut() {
                if let Some(o) = block.as_object_mut() {
                    o.shift_remove("cache_control");
                }
            }
            system[anchor]["cache_control"] = cache_control.clone();
        }
    }
}

// ---------------------------------------------------------------- body

/// Inputs of [`build_body`] beyond the model and context.
pub(crate) struct BodyInput<'a> {
    pub model: &'a Model,
    pub view: &'a ModelView,
    pub context: &'a Context,
    pub is_oauth: bool,
    pub options: &'a AnthropicOptions,
}

/// The request body TS `buildParams` produces (after `onPayload`-free post-processing).
#[allow(clippy::too_many_lines)]
pub(crate) fn build_body(input: &BodyInput<'_>) -> Result<Value, String> {
    let BodyInput {
        model,
        view,
        context,
        is_oauth,
        options,
    } = *input;
    let base = &options.base;
    let cache_control = cache_control_value(view, base.cache_retention, is_oauth);

    // System blocks.
    let inject_cc =
        is_oauth && view.compat.get("injectClaudeCodeInstruction") != Some(&Value::Bool(false));
    let prompts = normalize_system_prompts(context.system_prompt.as_deref());
    let has_billing_header = prompts
        .iter()
        .any(|p| p.starts_with(CLAUDE_BILLING_HEADER_PREFIX));
    let mut system: Option<Vec<Value>> = if inject_cc && !has_billing_header {
        let first_user = first_user_message_text(&context.messages);
        let mut identity = text_block(CLAUDE_CODE_SYSTEM_INSTRUCTION);
        identity["cache_control"] = cache_control
            .clone()
            .unwrap_or_else(|| json!({ "type": "ephemeral" }));
        let mut blocks = vec![text_block(&billing_header(&first_user)), identity];
        blocks.extend(prompts.iter().map(|p| text_block(p)));
        Some(blocks)
    } else if prompts.is_empty() {
        None
    } else {
        Some(prompts.iter().map(|p| text_block(p)).collect())
    };

    // Tools.
    let mut tools: Option<Vec<Value>> = match &context.tools {
        Some(t) => Some(convert_tools(
            t,
            is_oauth,
            view.is("disableStrictTools"),
            view.is("supportsEagerToolInputStreaming"),
            view.is("escapeBuiltinToolNames"),
        )),
        None if is_oauth => Some(Vec::new()),
        None => None,
    };

    // Metadata.
    let metadata_str = |key: &str| -> Option<String> {
        base.metadata
            .as_ref()?
            .get(key)?
            .as_str()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let account_id = metadata_str("account_uuid")
        .or_else(|| metadata_str("accountId"))
        .or_else(|| metadata_str("account_id"));
    let user_id_source = metadata_str("user_id").or_else(|| {
        (model.provider == "kimi-code")
            .then(|| prompt_cache_key(base.session_id.as_deref(), base.cache_retention))
            .flatten()
    });
    let metadata_user_id = resolve_metadata_user_id(
        user_id_source.as_deref(),
        is_oauth,
        base.session_id.as_deref(),
        account_id.as_deref(),
    );

    // Thinking + output_config effort.
    let mut thinking: Option<Value> = None;
    let mut output_effort: Option<String> = None;
    if model.reasoning {
        let disabled_thinking = view.compat_str("disabledThinking");
        if options.thinking_enabled == Some(false) && disabled_thinking.is_some() {
            let dt = disabled_thinking.unwrap_or("");
            if dt == "between-tools" {
                thinking = Some(json!({ "type": "between_tools" }));
                output_effort = view.compat_str("betweenToolsEffort").map(str::to_string);
            } else if dt != "omit" {
                thinking = Some(if dt == "adaptive" {
                    let mut t = json!({ "type": "adaptive" });
                    if view.thinking_flag("supportsDisplay") == Some(true) {
                        t["display"] = json!(options
                            .thinking_display
                            .map_or("summarized", |d| d.as_str()));
                    }
                    t
                } else {
                    json!({ "type": "disabled" })
                });
            }
        } else if options.thinking_enabled == Some(true) || view.is("requiresThinkingEnabled") {
            let mode = view.thinking_str("mode");
            let effort = resolve_adaptive_effort(model, view, options)?;
            if !uses_budget_thinking(view) {
                let mut t = json!({ "type": "adaptive" });
                if view.thinking_flag("supportsDisplay") == Some(true) {
                    t["display"] = json!(options
                        .thinking_display
                        .map_or("summarized", |d| d.as_str()));
                }
                thinking = Some(t);
                if let Some(e) = effort.filter(|e| e != "adaptive") {
                    output_effort = Some(e);
                }
            } else {
                let mut t = json!({
                    "type": "enabled",
                    "budget_tokens": options.thinking_budget_tokens.filter(|b| *b > 0).unwrap_or(1024),
                });
                if view.thinking_flag("supportsDisplay") != Some(false) {
                    t["display"] = json!(options
                        .thinking_display
                        .map_or("summarized", |d| d.as_str()));
                }
                thinking = Some(t);
                if mode == Some("anthropic-budget-effort") {
                    if let Some(e) = effort.filter(|e| e != "adaptive") {
                        output_effort = Some(e);
                    }
                }
            }
        } else if options.thinking_enabled == Some(false) {
            if view.is("supportsBetweenToolsThinking") {
                thinking = Some(json!({ "type": "between_tools" }));
                output_effort = view.compat_str("betweenToolsEffort").map(str::to_string);
            } else if is_adaptive_only_thinking(view) {
                output_effort = Some("low".into());
            } else {
                thinking = Some(json!({ "type": "disabled" }));
            }
        }
    }

    let prefix_mismatch = (view.thinking_flag("prefixBinding") == Some(true)
        && view.is("supportsThinkingBindingControls"))
    .then_some("drop_block");
    if let Some(behavior) = prefix_mismatch {
        if thinking.is_none()
            && view.thinking_str("mode") == Some("anthropic-adaptive")
            && !(options.thinking_enabled == Some(false)
                && view.compat_str("disabledThinking") == Some("omit"))
        {
            thinking = Some(json!({ "type": "adaptive" }));
        }
        if let Some(t) = thinking.as_mut() {
            if matches!(t["type"].as_str(), Some("adaptive" | "enabled")) {
                t["block_binding"] = json!({ "prefix_mismatch_behavior": behavior });
            }
        }
    }

    // `between_tools` rejects xhigh/max effort.
    if thinking
        .as_ref()
        .is_some_and(|t| t["type"] == "between_tools")
        && matches!(output_effort.as_deref(), Some("xhigh" | "max"))
    {
        thinking = None;
    }

    // Context management.
    let thinking_type = thinking
        .as_ref()
        .and_then(|t| t["type"].as_str())
        .map(str::to_string);
    let keep_thinking_context = view.not_false("supportsContextManagement")
        && !is_vertex_raw_predict(&model.base_url)
        && matches!(thinking_type.as_deref(), Some("adaptive" | "enabled"));
    let context_management = keep_thinking_context
        .then(|| json!({ "edits": [{ "type": "clear_thinking_20251015", "keep": "all" }] }));

    // Messages.
    let transformed = transform_messages(&context.messages, model, view, &|id, _, _| {
        Some(normalize_tool_call_id(id))
    });
    let mut wire_messages = convert_messages(view, model, &transformed, is_oauth);

    if let Some(cc) = &cache_control {
        apply_head_caching(&mut system, &mut tools, cc);
    }

    let max_output = if model.max_tokens == 0 {
        UNKNOWN_MODEL_MAX_OUTPUT_TOKENS
    } else {
        model.max_tokens
    };
    let mut output_config = Map::new();
    if let Some(effort) = &output_effort {
        if view.is("supportsOutputEffort") {
            output_config.insert("effort".into(), json!(effort));
        }
    }

    let mut params = Map::new();
    params.insert(
        "model".into(),
        json!(options
            .request_model_id
            .clone()
            .or_else(|| model.request_model_id.clone())
            .unwrap_or_else(|| model.id.clone())),
    );
    let head_breakpoints = {
        let sys = system.as_ref().map_or(0, |s| {
            s.iter()
                .filter(|b| b.get("cache_control").is_some_and(|c| !c.is_null()))
                .count()
        });
        let tl = tools.as_ref().map_or(0, |t| {
            t.iter()
                .filter(|b| b.get("cache_control").is_some_and(|c| !c.is_null()))
                .count()
        });
        sys + tl
    };
    // `messages` is finalized after the tail-caching pass below; reserve its slot now.
    params.insert("messages".into(), Value::Null);
    if let Some(system) = system {
        params.insert("system".into(), Value::Array(system));
    }
    if let Some(tools) = tools {
        params.insert("tools".into(), Value::Array(tools));
    }
    if let Some(user_id) = metadata_user_id {
        params.insert("metadata".into(), json!({ "user_id": user_id }));
    }
    params.insert(
        "max_tokens".into(),
        json!(max_output.min(base.max_tokens.unwrap_or(max_output))),
    );
    if let Some(t) = &thinking {
        params.insert("thinking".into(), t.clone());
    }
    if let Some(cm) = context_management {
        params.insert("context_management".into(), cm);
    }
    if !output_config.is_empty() {
        params.insert("output_config".into(), Value::Object(output_config));
    }
    params.insert("stream".into(), json!(true));

    let allow_sampling = view.is("supportsSamplingParams")
        && matches!(thinking_type.as_deref(), None | Some("disabled"));
    if allow_sampling {
        if let Some(t) = base.temperature {
            params.insert("temperature".into(), json!(t));
        }
    }
    let fast = view.is("fastMode")
        || (model.provider == "anthropic" && base.service_tier == Some(ServiceTier::Priority));
    if fast {
        params.insert("speed".into(), json!("fast"));
    }

    if let Some(choice) = &options.tool_choice {
        use crate::providers::anthropic::AnthropicToolChoice as C;
        let mut tc = match choice {
            C::Auto => Some(json!({ "type": "auto" })),
            C::Any => Some(json!({ "type": "any" })),
            C::None => Some(json!({ "type": "none" })),
            C::Tool { name } if !name.is_empty() => Some(json!({
                "type": "tool",
                "name": encode_tool_name(name, is_oauth, view.is("escapeBuiltinToolNames")),
            })),
            C::Tool { .. } => None,
        };
        if let Some(t) = &tc {
            if matches!(t["type"].as_str(), Some("any" | "tool"))
                && !view.is("supportsForcedToolChoice")
            {
                tc = Some(json!({ "type": "auto" }));
            }
        }
        if let Some(tc) = tc {
            params.insert("tool_choice".into(), tc);
        }
    }

    // disableThinkingIfToolChoiceForced
    if params
        .get("tool_choice")
        .is_some_and(|t| matches!(t["type"].as_str(), Some("any" | "tool")))
    {
        params.shift_remove("thinking");
        params.shift_remove("context_management");
        if is_adaptive_only_thinking(view) && view.is("supportsOutputEffort") {
            let mut oc = params
                .get("output_config")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            oc.insert("effort".into(), json!("low"));
            params.insert("output_config".into(), Value::Object(oc));
        } else if let Some(Value::Object(oc)) = params.get_mut("output_config") {
            oc.shift_remove("effort");
            if oc.is_empty() {
                params.shift_remove("output_config");
            }
        }
    }

    // ensureMaxTokensForThinking
    if let Some(Value::Object(t)) = params.get_mut("thinking") {
        if t.get("type") == Some(&json!("enabled")) {
            let budget = t.get("budget_tokens").and_then(Value::as_u64).unwrap_or(0);
            if budget > 0 {
                let current = params["max_tokens"].as_u64();
                let (max_tokens, new_budget) = budget_thinking_output(current, budget, max_output);
                params.insert("max_tokens".into(), json!(max_tokens));
                if new_budget != budget as i64 {
                    if new_budget <= 0 {
                        return Err(format!(
                            "Anthropic thinking budget requires max_tokens greater than {OUTPUT_FALLBACK_BUFFER}; got {max_tokens}"
                        ));
                    }
                    if let Some(Value::Object(t)) = params.get_mut("thinking") {
                        t.insert("budget_tokens".into(), json!(new_budget));
                    }
                }
            }
        }
    }

    if let Some(cc) = &cache_control {
        apply_prompt_caching(&mut wire_messages, head_breakpoints, cc);
    }
    params.insert(
        "messages".into(),
        Value::Array(wire_messages.into_iter().map(|p| p.value).collect()),
    );
    Ok(Value::Object(params))
}

fn first_user_message_text(messages: &[Message]) -> String {
    for message in messages {
        let (Message::User(user) | Message::Developer(user)) = message else {
            continue;
        };
        match &user.content {
            UserMessageContent::Text(t) => {
                if !t.trim().is_empty() {
                    return t.clone();
                }
            }
            UserMessageContent::Blocks(blocks) => {
                for b in blocks {
                    if let UserOrToolContent::Text(t) = b {
                        if !t.text.trim().is_empty() {
                            return t.text.clone();
                        }
                    }
                }
                if blocks
                    .iter()
                    .any(|b| matches!(b, UserOrToolContent::Image(_)))
                {
                    return String::new();
                }
            }
        }
    }
    String::new()
}

/// TS `isVertexRawPredictUrl`.
fn is_vertex_raw_predict(url: &str) -> bool {
    url.contains("aiplatform.googleapis.com") && url.contains(":rawPredict")
}

/// TS `getOpenAIPromptCacheKey` (cache identity → 64-char wire limit).
pub(crate) fn prompt_cache_key(
    session_id: Option<&str>,
    retention: Option<CacheRetention>,
) -> Option<String> {
    if resolve_retention(retention, CacheRetention::Short) == CacheRetention::None {
        return None;
    }
    let value = session_id.filter(|v| !v.is_empty())?;
    crate::providers::openai_completions::normalize_stable_id(value, 64, "pc_")
}

#[cfg(test)]
mod eager_streaming_tests {
    use super::*;

    fn tool(name: &str, parameters: Value) -> crate::types::Tool {
        serde_json::from_value(json!({"name": name, "description": "d", "parameters": parameters}))
            .expect("tool")
    }

    #[test]
    fn nested_object_tools_are_not_eager() {
        let flat = tool("write", json!({"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}));
        let fused = tool("write", json!({"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
            "then_run": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}, "required": ["path", "content"]}));
        let out = convert_tools(&[flat, fused], false, true, true, false);
        assert_eq!(out[0].get("eager_input_streaming"), Some(&json!(true)));
        assert!(out[1].get("eager_input_streaming").is_none());
    }
}

#[cfg(test)]
mod slots_cache_tests {
    use super::*;

    fn p(role: &str, text: &str, conversational: bool) -> Param {
        Param {
            value: json!({ "role": role, "content": text }),
            conversational,
        }
    }

    fn marked(m: &[Param]) -> Vec<usize> {
        m.iter()
            .enumerate()
            .filter(|(_, p)| p.value.to_string().contains("cache_control"))
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn slots_message_gets_a_breakpoint_in_addition_to_the_tail() {
        let cc = json!({ "type": "ephemeral" });
        let mut m = vec![
            p("user", "task", true),
            p("user", "<context-slots>\nslot a\n</context-slots>", false),
            p("assistant", "a1", false),
            p("user", "u1", true),
            p("assistant", "a2", false),
            p("user", "u2", true),
        ];
        apply_prompt_caching(&mut m, 2, &cc);
        assert_eq!(marked(&m), vec![1, 5], "slots + last message (budget 2)");
    }

    #[test]
    fn without_marker_layout_is_unchanged() {
        let cc = json!({ "type": "ephemeral" });
        let mut m = vec![
            p("user", "task", true),
            p("assistant", "a1", false),
            p("user", "u1", true),
        ];
        apply_prompt_caching(&mut m, 2, &cc);
        assert_eq!(
            marked(&m),
            vec![1, 2],
            "the two trailing messages, as in TS"
        );
    }
}
