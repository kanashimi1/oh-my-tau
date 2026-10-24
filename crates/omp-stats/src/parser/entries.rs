//! Entry guards and stat extractors from `parser.ts`. Session JSONL on disk
//! is not guaranteed to match the current shapes (crash-truncated turns,
//! older versions, foreign producers), so every field is validated at read
//! time and malformed entries are coerced or skipped — see the TS comments.

use anyhow::Result;
use serde_json::Value;

use crate::taxonomy::{classify_model, ClassifyOptions, ModelIdentity};
use crate::types::{
    AgentType, MessageStatsInput, ServiceTierByFamily, StopReason, ToolCallStats, ToolResultLink,
    UsageInput, UsageInputCost, UserMessageStats,
};
use crate::user_metrics::{compute_user_message_metrics, judge_prose};

use super::tier::{
    get_premium_service_tier_requests, parse_service_tier, resolve_model_service_tier, TierModel,
};
use super::util::{date_parse, js_len};
use super::wyhash::wyhash_hex;

/// Characters a persisted tool name may consist of without sanitization
/// (JS `\w` is ASCII).
const TOOL_NAME_ALLOWED: fn(char) -> bool =
    |c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == ':' || c == '-';

fn is_type(v: &Value, ty: &str) -> bool {
    v.get("type").and_then(Value::as_str) == Some(ty)
}

fn id_nonempty(v: &Value) -> bool {
    v.get("id")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
}

fn msg_role(v: &Value) -> Option<&str> {
    v.get("message")?.get("role")?.as_str()
}

/// Legacy sessions (pre-id tracking) recorded message entries without an
/// `id`. They're not linkable and would violate the messages.entry_id NOT
/// NULL constraint, so skip them at the parser boundary.
pub(crate) fn is_assistant_message(v: &Value) -> bool {
    is_type(v, "message") && id_nonempty(v) && msg_role(v) == Some("assistant")
}

pub(crate) fn is_model_usage(v: &Value) -> bool {
    is_type(v, "model_usage") && id_nonempty(v)
}

/// Check if an entry is a user message (non-toolResult).
pub(crate) fn is_user_message(v: &Value) -> bool {
    is_type(v, "message") && id_nonempty(v) && msg_role(v) == Some("user")
}

/// Check if an entry is a service-tier change.
pub(crate) fn is_service_tier_change(v: &Value) -> bool {
    is_type(v, "service_tier_change")
}

/// Check if an entry is a tool-result message.
pub(crate) fn is_tool_result_message(v: &Value) -> bool {
    is_type(v, "message") && msg_role(v) == Some("toolResult")
}

/// Extract plain text from a user message content payload.
fn extract_user_text(content: &Value) -> String {
    if let Some(s) = content.as_str() {
        return s.to_owned();
    }
    let Some(blocks) = content.as_array() else {
        return String::new();
    };
    let mut parts = Vec::new();
    for block in blocks {
        if block.get("type").and_then(Value::as_str) == Some("text") {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                parts.push(text);
            }
        }
    }
    parts.join("")
}

/// Build user-message stats from an entry. Returns `None` for
/// empty/synthetic content.
pub(crate) fn extract_user_stats(
    session_file: &str,
    folder: &str,
    entry: &Value,
) -> Option<UserMessageStats> {
    let msg = entry.get("message")?;
    if msg_role(entry) != Some("user")
        || msg
            .get("synthetic")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        return None;
    }
    let text = extract_user_text(msg.get("content").unwrap_or(&Value::Null));
    if text.trim().is_empty() {
        return None;
    }
    let metrics = compute_user_message_metrics(&text);
    let prose = judge_prose(&text);
    let timestamp = entry
        .get("timestamp")
        .and_then(Value::as_str)
        .and_then(date_parse)
        .unwrap_or(0);
    Some(UserMessageStats {
        id: None,
        session_file: session_file.to_owned(),
        entry_id: entry
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        folder: folder.to_owned(),
        timestamp,
        model: None,
        provider: None,
        chars: metrics.chars,
        words: metrics.words,
        yelling: metrics.yelling,
        profanity: metrics.profanity,
        anguish: metrics.anguish,
        negation: metrics.negation,
        repetition: metrics.repetition,
        blame: metrics.blame,
        prose: prose.clone(),
        prose_hash: if prose.is_empty() {
            String::new()
        } else {
            wyhash_hex(prose.as_bytes())
        },
    })
}

/// A persisted token counter must be a finite JSON number; anything else
/// counts as absent (`isFiniteCount` / `finiteTokenCount`).
fn finite_count(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64)
}

fn finite_token_count(v: Option<&Value>) -> f64 {
    finite_count(v).unwrap_or(0.0)
}

/// Token-bucket view for total derivation over a raw persisted usage payload.
#[derive(Debug, Clone, Copy)]
pub struct UsageBucketView<'a>(pub &'a Value);

impl UsageBucketView<'_> {
    /// See [`resolve_usage_total`].
    #[must_use]
    pub fn total(&self) -> f64 {
        resolve_usage_total(Some(self.0))
    }
}

/// Total tokens for one usage payload: the conversation buckets plus
/// provider-reported orchestration tokens. A present finite provider total
/// stays authoritative; a missing or malformed one is derived from the
/// buckets, which would otherwise persist a zero total next to real counts.
#[must_use]
pub fn resolve_usage_total(usage: Option<&Value>) -> f64 {
    let usage = match usage {
        Some(u) => u,
        None => return 0.0,
    };
    if let Some(total) = finite_count(usage.get("totalTokens")) {
        return total;
    }
    if !usage.is_object() {
        return 0.0;
    }
    let orchestration = usage.get("orchestration").filter(|v| v.is_object());
    let orch = |key: &str| orchestration.and_then(|o| o.get(key));
    finite_token_count(usage.get("input"))
        + finite_token_count(usage.get("output"))
        + finite_token_count(usage.get("cacheRead"))
        + finite_token_count(usage.get("cacheWrite"))
        + finite_token_count(orch("input"))
        + finite_token_count(orch("output"))
        + finite_token_count(orch("cacheRead"))
}

fn parse_stop_reason(msg: &Value) -> StopReason {
    match msg.get("stopReason") {
        Some(v) if !v.is_null() => {
            match serde_json::from_value::<StopReason>(v.clone()) {
                // Unknown stopReason string: fall back like TS does for an
                // unterminated message (ingest keeps only the typed reasons).
                Ok(StopReason::Other(_)) | Err(_) => fallback_stop_reason(msg),
                Ok(reason) => reason,
            }
        }
        _ => fallback_stop_reason(msg),
    }
}

/// A message persisted without a terminal stop reason never completed
/// normally: classify by whether it carried an error.
fn fallback_stop_reason(msg: &Value) -> StopReason {
    if msg.get("errorMessage").is_some_and(Value::is_string) {
        StopReason::Error
    } else {
        StopReason::Aborted
    }
}

/// Message timestamp, falling back to the entry's ISO timestamp, then 0.
/// A stored zero is the "no timestamp" sentinel, not 1970.
fn coerce_entry_timestamp(msg_timestamp: Option<&Value>, entry: &Value) -> i64 {
    if let Some(ts) = finite_count(msg_timestamp) {
        if ts > 0.0 {
            return ts as i64;
        }
    }
    entry
        .get("timestamp")
        .and_then(Value::as_str)
        .and_then(date_parse)
        .unwrap_or(0)
}

fn usage_cost(v: Option<&Value>) -> Option<UsageInputCost> {
    serde_json::from_value(v?.clone()).ok()
}

/// Extract stats from an assistant message entry. Malformed entries are
/// coerced (missing `stopReason`, token counts, `timestamp`) or skipped
/// (missing `model`/`provider`/`api`/`usage`) instead of crashing the sync.
pub(crate) fn extract_stats(
    session_file: &str,
    folder: &str,
    entry: &Value,
    current_service_tier: Option<&ServiceTierByFamily>,
    agent_type: AgentType,
) -> Option<MessageStatsInput> {
    let msg = entry.get("message")?;
    if msg_role(entry) != Some("assistant") {
        return None;
    }
    let model = msg.get("model")?.as_str()?;
    let provider = msg.get("provider")?.as_str()?;
    let api = msg.get("api")?.as_str()?;
    let raw_usage = msg.get("usage")?;
    if !raw_usage.is_object() {
        return None;
    }

    // Backfill: derive premium-request count when the recorded tier is
    // premium but the usage payload predates `premiumRequests`. Trust any
    // non-zero value already there (Copilot multipliers are fractional).
    let recorded = finite_count(raw_usage.get("premiumRequests")).unwrap_or(0.0);
    let identity: ModelIdentity = classify_model(
        provider,
        model,
        Some(ClassifyOptions {
            lenient: true,
            ..ClassifyOptions::default()
        }),
    )
    .unwrap_or_default();
    let tier_model = TierModel {
        provider,
        api,
        identity_class: &identity.class,
    };
    let served_tier = msg.get("serviceTier").and_then(parse_service_tier);
    let tier =
        served_tier.or_else(|| resolve_model_service_tier(current_service_tier, &tier_model));
    let derived = if recorded > 0.0 {
        recorded
    } else {
        get_premium_service_tier_requests(tier, &tier_model, served_tier.is_some())
    };

    let num = |key: &str| raw_usage.get(key).and_then(Value::as_f64);
    let tok = |key: &str| num(key).unwrap_or(0.0) as i64;
    let well_formed = finite_count(raw_usage.get("input")).is_some()
        && finite_count(raw_usage.get("output")).is_some()
        && finite_count(raw_usage.get("cacheRead")).is_some()
        && finite_count(raw_usage.get("cacheWrite")).is_some()
        && finite_count(raw_usage.get("totalTokens")).is_some();

    let cost = usage_cost(raw_usage.get("cost"));
    // TS keeps the raw usage object, so orchestration/cttl reach pricing.
    let orchestration = raw_usage
        .get("orchestration")
        .filter(|v| v.is_object())
        .and_then(|v| {
            serde_json::from_value::<crate::types::UsageOrchestration>(lenient_counts(v)).ok()
        });
    let cttl = raw_usage
        .get("cttl")
        .filter(|v| v.is_object())
        .and_then(|v| serde_json::from_value::<crate::types::UsageCttl>(lenient_counts(v)).ok());
    // An omitted `cost` must stay omitted: absence reads as "no recorded
    // price" (estimated), a zero reads as an explicitly free request.
    let usage = if well_formed && derived == recorded {
        UsageInput {
            input: tok("input"),
            output: tok("output"),
            cache_read: tok("cacheRead"),
            cache_write: tok("cacheWrite"),
            total_tokens: tok("totalTokens"),
            premium_requests: finite_count(raw_usage.get("premiumRequests")),
            cost,
            orchestration,
            cttl,
        }
    } else {
        UsageInput {
            input: tok("input"),
            output: tok("output"),
            cache_read: tok("cacheRead"),
            cache_write: tok("cacheWrite"),
            // A present finite provider total stays authoritative; a missing
            // or malformed one is derived from the buckets.
            total_tokens: resolve_usage_total(Some(raw_usage)) as i64,
            premium_requests: Some(derived),
            cost,
            orchestration,
            cttl,
        }
    };

    Some(MessageStatsInput {
        id: None,
        session_file: session_file.to_owned(),
        entry_id: entry
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        folder: folder.to_owned(),
        model: model.to_owned(),
        provider: provider.to_owned(),
        api: api.to_owned(),
        timestamp: coerce_entry_timestamp(msg.get("timestamp"), entry),
        duration: msg.get("duration").and_then(Value::as_f64),
        ttft: msg.get("ttft").and_then(Value::as_f64),
        stop_reason: parse_stop_reason(msg),
        error_message: msg
            .get("errorMessage")
            .and_then(Value::as_str)
            .map(str::to_owned),
        usage,
        agent_type,
        service_tier: served_tier,
    })
}

/// Legacy `model_usage` entries are reshaped into a slim assistant message
/// and run through [`extract_stats`].
pub(crate) fn extract_model_usage_stats(
    session_file: &str,
    folder: &str,
    entry: &Value,
    agent_type: AgentType,
) -> Option<MessageStatsInput> {
    let timestamp = entry
        .get("timestamp")
        .and_then(Value::as_str)
        .and_then(date_parse)
        .unwrap_or(0);
    let mut message = serde_json::Map::new();
    message.insert("role".into(), Value::from("assistant"));
    message.insert("content".into(), Value::Array(vec![]));
    if let Some(v) = entry.get("api") {
        message.insert("api".into(), v.clone());
    }
    if let Some(v) = entry.get("provider") {
        message.insert("provider".into(), v.clone());
    }
    if let Some(v) = entry.get("model") {
        message.insert("model".into(), v.clone());
    }
    if let Some(v) = entry.get("usage") {
        message.insert("usage".into(), v.clone());
    }
    match entry.get("stopReason") {
        Some(v) if !v.is_null() => message.insert("stopReason".into(), v.clone()),
        _ => message.insert("stopReason".into(), Value::from("stop")),
    };
    if let Some(v) = entry.get("errorMessage") {
        message.insert("errorMessage".into(), v.clone());
    }
    message.insert("timestamp".into(), Value::from(timestamp));

    let synthetic = serde_json::json!({
        "type": "message",
        "id": entry.get("id").cloned().unwrap_or(Value::Null),
        "timestamp": entry.get("timestamp").cloned().unwrap_or(Value::Null),
        "message": Value::Object(message),
    });
    extract_stats(session_file, folder, &synthetic, None, agent_type)
}

/// Extract one [`ToolCallStats`] per `toolCall` content block of an assistant
/// message. Returns an empty vector for turns without tool calls.
pub(crate) fn extract_tool_calls(
    session_file: &str,
    folder: &str,
    entry: &Value,
    agent_type: AgentType,
) -> Vec<ToolCallStats> {
    let Some(content) = entry
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
    else {
        return vec![];
    };
    let msg = entry.get("message").expect("checked above");
    // `tool_calls` columns are NOT NULL: skip turns that can't be attributed.
    let (Some(model), Some(provider)) = (
        msg.get("model").and_then(Value::as_str),
        msg.get("provider").and_then(Value::as_str),
    ) else {
        return vec![];
    };

    let blocks: Vec<&Value> = content
        .iter()
        .filter(|block| {
            block.is_object()
                && block.get("type").and_then(Value::as_str) == Some("toolCall")
                && block.get("id").is_some_and(Value::is_string)
                && block.get("name").is_some_and(Value::is_string)
        })
        .collect();
    if blocks.is_empty() {
        return vec![];
    }

    let mut calls = Vec::new();
    for block in &blocks {
        // Names reduced to nothing by sanitization carry no tool identity:
        // skip them rather than attributing usage to garbage. callsInTurn
        // still counts the raw block total.
        let name = block
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let Some(tool_name) = sanitize_tool_name(name) else {
            continue;
        };
        // TS `block.arguments ?? {}`: a null/absent payload sizes as "{}".
        let empty = Value::Object(serde_json::Map::new());
        let arguments = block
            .get("arguments")
            .filter(|a| !a.is_null())
            .unwrap_or(&empty);
        let args_chars = js_len(&serde_json::to_string(arguments).unwrap_or_else(|_| "{}".into()));
        calls.push(ToolCallStats {
            session_file: session_file.to_owned(),
            entry_id: entry
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            tool_call_id: block
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            folder: folder.to_owned(),
            tool_name,
            model: model.to_owned(),
            provider: provider.to_owned(),
            timestamp: coerce_entry_timestamp(msg.get("timestamp"), entry),
            agent_type,
            calls_in_turn: blocks.len() as i64,
            args_chars,
        });
    }
    calls
}

/// Tool names as persisted can be polluted by provider-side parse garbage.
/// Reduce such names to their leading identifier token; names that yield no
/// identifier at all carry no tool identity and return `None` so the row is
/// skipped.
#[must_use]
pub fn sanitize_tool_name(name: &str) -> Option<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.chars().all(TOOL_NAME_ALLOWED) {
        return Some(trimmed.to_owned());
    }
    let candidate = trimmed
        .split(|c| !TOOL_NAME_ALLOWED(c))
        .next()
        .unwrap_or("");
    if candidate.is_empty() {
        None
    } else {
        Some(candidate.to_owned())
    }
}

/// Build the result linkage for a `toolResult` entry: text characters fed
/// back into context plus the error flag, keyed to the originating call.
pub(crate) fn extract_tool_result_link(
    session_file: &str,
    entry: &Value,
) -> Option<ToolResultLink> {
    let msg = entry.get("message")?;
    if msg_role(entry) != Some("toolResult") {
        return None;
    }
    let tool_call_id = msg.get("toolCallId").and_then(Value::as_str)?;
    if tool_call_id.is_empty() {
        return None;
    }
    let mut result_chars = 0i64;
    if let Some(content) = msg.get("content").and_then(Value::as_array) {
        for block in content {
            if block.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    result_chars += js_len(text);
                }
            }
        }
    }
    Some(ToolResultLink {
        session_file: session_file.to_owned(),
        tool_call_id: tool_call_id.to_owned(),
        result_chars,
        is_error: msg.get("isError").and_then(Value::as_bool) == Some(true),
    })
}

/// Convenience used by tests: parse one JSONL line.
#[allow(dead_code)]
pub(crate) fn parse_line(line: &str) -> Result<Value> {
    Ok(serde_json::from_str(line)?)
}

/// Keep only finite numeric members (as integers) of a usage sub-object, so a
/// malformed member counts as absent instead of failing the whole object
/// (TS `finiteTokenCount` semantics).
fn lenient_counts(v: &Value) -> Value {
    let map = v
        .as_object()
        .map(|o| {
            o.iter()
                .filter_map(|(k, val)| val.as_f64().map(|n| (k.clone(), Value::from(n as i64))))
                .collect::<serde_json::Map<_, _>>()
        })
        .unwrap_or_default();
    Value::Object(map)
}
