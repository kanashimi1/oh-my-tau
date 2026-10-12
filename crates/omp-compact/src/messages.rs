//! Agent-message constructors and entry → message mapping (pi-agent-core `compaction/messages.ts`,
//! `compaction.ts#getMessageFromEntry`).

use serde_json::{Map, Value};

use crate::json::{entry_type, ms_value, parse_iso_ms, put_opt, str_field};
use crate::prompts;

/// `DEFAULT_CUSTOM_MESSAGE_TYPE`.
const DEFAULT_CUSTOM_MESSAGE_TYPE: &str = "custom-message";

/// `renderCompactionSummaryContext`.
#[must_use]
pub fn render_compaction_summary_context(summary: &str) -> String {
    prompts::render(prompts::COMPACTION_SUMMARY_CONTEXT, &[("summary", summary)])
}

/// `renderBranchSummaryContext`.
#[must_use]
pub fn render_branch_summary_context(summary: &str) -> String {
    prompts::render(prompts::BRANCH_SUMMARY_CONTEXT, &[("summary", summary)])
}

/// `renderHandoffSummaryContext`.
#[must_use]
pub fn render_handoff_summary_context(summary: &str) -> String {
    prompts::render(prompts::HANDOFF_SUMMARY_CONTEXT, &[("summary", summary)])
}

/// `createBranchSummaryMessage`.
#[must_use]
pub fn create_branch_summary_message(summary: &str, from_id: &str, timestamp_iso: &str) -> Value {
    let mut m = Map::new();
    m.insert("role".into(), "branchSummary".into());
    m.insert("summary".into(), summary.into());
    m.insert("fromId".into(), from_id.into());
    m.insert("timestamp".into(), ms_value(parse_iso_ms(timestamp_iso)));
    Value::Object(m)
}

/// `createCompactionSummaryMessage` for a `compaction` entry (local path: no snapcompact blocks,
/// no provider payload).
#[must_use]
pub fn create_compaction_summary_message(entry: &Value) -> Value {
    create_compaction_summary_message_ext(entry, None, None, None)
}

/// `createCompactionSummaryMessage(summary, tokensBefore, timestamp, options)` with the replay
/// options a provider-native / snapcompact compaction attaches: `providerPayload`
/// (`openaiResponsesHistory` / `anthropicCompaction`), snapcompact history `blocks` (their image
/// blocks are mirrored onto `images`), and the Anthropic `historyRewriteAt` marker.
#[must_use]
pub fn create_compaction_summary_message_ext(
    entry: &Value,
    provider_payload: Option<Value>,
    blocks: Option<Vec<Value>>,
    history_rewrite_at: Option<Value>,
) -> Value {
    let blocks = blocks.filter(|b| !b.is_empty());
    let image_blocks: Vec<Value> = blocks
        .iter()
        .flatten()
        .filter(|b| str_field(b, "type") == Some("image"))
        .cloned()
        .collect();
    let mut m = Map::new();
    m.insert("role".into(), "compactionSummary".into());
    m.insert(
        "summary".into(),
        entry.get("summary").cloned().unwrap_or(Value::Null),
    );
    put_opt(&mut m, "shortSummary", entry.get("shortSummary").cloned());
    m.insert(
        "tokensBefore".into(),
        entry.get("tokensBefore").cloned().unwrap_or(Value::Null),
    );
    put_opt(&mut m, "tokensAfter", entry.get("tokensAfter").cloned());
    put_opt(&mut m, "method", entry.get("method").cloned());
    put_opt(
        &mut m,
        "providerPayload",
        provider_payload.filter(|p| !p.is_null()),
    );
    if let Some(blocks) = blocks {
        m.insert("blocks".into(), Value::Array(blocks));
    }
    if !image_blocks.is_empty() {
        m.insert("images".into(), Value::Array(image_blocks));
    }
    put_opt(&mut m, "warning", entry.get("warning").cloned());
    put_opt(
        &mut m,
        "historyRewriteAt",
        history_rewrite_at.filter(|v| !v.is_null()),
    );
    m.insert(
        "timestamp".into(),
        ms_value(str_field(entry, "timestamp").and_then(parse_iso_ms)),
    );
    Value::Object(m)
}

/// `createCustomMessage` + `normalizeCustomMessagePayload` over a `custom_message` entry
/// (`customMessageEntryMessage`): `None` when the content is neither a string nor an array.
#[must_use]
pub fn custom_message_from_entry(entry: &Value) -> Option<Value> {
    let content = entry.get("content")?;
    if !(content.is_string() || content.is_array()) {
        return None;
    }
    let custom_type = str_field(entry, "customType")
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_CUSTOM_MESSAGE_TYPE);
    let mut m = Map::new();
    m.insert("role".into(), "custom".into());
    m.insert("customType".into(), custom_type.into());
    m.insert("content".into(), content.clone());
    m.insert(
        "display".into(),
        entry
            .get("display")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            .into(),
    );
    put_opt(
        &mut m,
        "details",
        entry.get("details").filter(|d| !d.is_null()).cloned(),
    );
    if entry.get("attribution").is_some_and(|a| !a.is_null()) {
        let attribution = if str_field(entry, "attribution") == Some("user") {
            "user"
        } else {
            "agent"
        };
        m.insert("attribution".into(), attribution.into());
    }
    m.insert(
        "timestamp".into(),
        ms_value(str_field(entry, "timestamp").and_then(parse_iso_ms)),
    );
    Some(Value::Object(m))
}

/// `getMessageFromEntry`: the message an entry contributes to LLM context, if any.
#[must_use]
pub fn entry_message(entry: &Value) -> Option<Value> {
    match entry_type(entry) {
        "message" => entry.get("message").cloned(),
        "custom_message" => custom_message_from_entry(entry),
        "branch_summary" => Some(create_branch_summary_message(
            str_field(entry, "summary").unwrap_or(""),
            str_field(entry, "fromId").unwrap_or(""),
            str_field(entry, "timestamp").unwrap_or(""),
        )),
        _ => None,
    }
}
