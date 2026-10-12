//! Context rebuild: which messages the model sees for a session branch, with compactions applied
//! (coding-agent `session/session-context.ts#buildSessionContext`, model-context mode).
//!
//! Layout after a compaction: the summary message first, then the kept tail (entries from
//! `firstKeptEntryId` up to the compaction entry), then everything appended after the compaction.
//! Ported: local compactions, provider-native lanes (`OpenAI` remote replacement history,
//! Anthropic native compaction with `historyRewriteAt`, snapcompact frame-archive blocks),
//! `branch_summary`, `custom_message`, `reset_boundary`, mode / model / thinking-level state,
//! context notes, dangling-tool-call stripping, error-turn dropping. Compactions with
//! `experimental-context-rollover` details are rejected with [`ContextError::Unsupported`]
//! rather than rebuilt incorrectly (rollover request recovery is not ported). The
//! transcript/display variants of the TS function are not ported.

use std::collections::{BTreeSet, HashMap};

use serde_json::{Map, Value};

use crate::json::{entry_id, entry_type, message_role, ms_value, parse_iso_ms, str_field, truthy};
use crate::messages::{create_branch_summary_message, create_compaction_summary_message_ext, custom_message_from_entry};
use crate::native;
use crate::prompts;

const PREWALK_PLAN_MESSAGE_TYPE: &str = "prewalk-plan";
const VIBE_MODE_CONTEXT_MESSAGE_TYPE: &str = "vibe-mode-context";
const INTERRUPTED_THINKING_MESSAGE_TYPE: &str = "interrupted-thinking";
/// `CONTEXT_NOTES_ENTRY_TYPE`.
pub const CONTEXT_NOTES_ENTRY_TYPE: &str = "experimental_context_notes";
const MAX_CONTEXT_NOTES_BYTES: usize = 16_384;

/// Why a context could not be rebuilt faithfully.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    /// The latest compaction depends on a payload this port does not replay.
    Unsupported(&'static str),
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "unsupported compaction payload: {what}"),
        }
    }
}

impl std::error::Error for ContextError {}

/// Which entry the branch ends at (`leafId`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Leaf<'a> {
    /// The last entry (`leafId` undefined).
    #[default]
    Last,
    /// A specific entry; unknown ids fall back to the last entry like TS.
    Id(&'a str),
    /// Before the first entry (`leafId === null`): empty context.
    Before,
}

/// `SessionContext`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SessionContext {
    pub messages: Vec<Value>,
    pub thinking_level: Option<String>,
    pub configured_thinking_level: Option<String>,
    pub service_tier: Option<Value>,
    /// Model roles: `{ default: "provider/modelId", ... }`.
    pub models: Map<String, Value>,
    pub injected_ttsr_rules: Vec<String>,
    pub mode: String,
    pub mode_data: Option<Value>,
}

/// `getLatestCompactionEntry`.
#[must_use]
pub fn latest_compaction(entries: &[Value]) -> Option<&Value> {
    entries.iter().rev().find(|e| entry_type(e) == "compaction")
}

/// Root-to-leaf path for `leaf` (parent cycles stop at the first repeat).
#[must_use]
pub fn branch_path<'a>(entries: &'a [Value], leaf: Leaf<'_>) -> Vec<&'a Value> {
    let by_id: HashMap<&str, &Value> = entries.iter().map(|e| (entry_id(e), e)).collect();
    let start = match leaf {
        Leaf::Before => return Vec::new(),
        Leaf::Id(id) => by_id.get(id).copied().or(entries.last()),
        Leaf::Last => entries.last(),
    };
    let mut path = Vec::new();
    let mut seen = BTreeSet::new();
    let mut current = start;
    while let Some(entry) = current {
        if !seen.insert(entry_id(entry)) {
            break;
        }
        path.push(entry);
        current = str_field(entry, "parentId").and_then(|p| by_id.get(p).copied());
    }
    path.reverse();
    path
}

fn is_empty_error_turn(message: &Value) -> bool {
    if str_field(message, "stopReason") != Some("error") {
        return false;
    }
    !message.get("content").and_then(Value::as_array).is_some_and(|blocks| {
        blocks.iter().any(|block| match str_field(block, "type") {
            Some("text") => str_field(block, "text").is_some_and(|t| !t.trim().is_empty()),
            Some("thinking") => {
                str_field(block, "thinking").is_some_and(|t| !t.trim().is_empty())
                    || str_field(block, "thinkingSignature").is_some_and(|t| !t.trim().is_empty())
            }
            Some("redactedThinking") => str_field(block, "data").is_some_and(|t| !t.trim().is_empty()),
            Some("fallback") => false,
            _ => true,
        })
    })
}

/// `appendMessage` (model-context mode): the message an entry adds to the context, if any.
fn context_message(entry: &Value) -> Option<Value> {
    match entry_type(entry) {
        "message" => {
            let message = entry.get("message")?;
            if message_role(message) == "assistant" && (truthy(message.get("retryRecovery")) || is_empty_error_turn(message)) {
                return None;
            }
            Some(message.clone())
        }
        "custom_message" => {
            let custom_type = str_field(entry, "customType");
            if custom_type == Some(PREWALK_PLAN_MESSAGE_TYPE) || custom_type == Some(VIBE_MODE_CONTEXT_MESSAGE_TYPE) {
                return None;
            }
            custom_message_from_entry(entry)
        }
        "branch_summary" if truthy(entry.get("summary")) => Some(create_branch_summary_message(
            str_field(entry, "summary").unwrap_or(""),
            str_field(entry, "fromId").unwrap_or(""),
            str_field(entry, "timestamp").unwrap_or(""),
        )),
        _ => None,
    }
}

/// `getContextNotes`: latest valid notebook revision after the active reset boundary.
fn context_notes<'a>(path: &[&'a Value]) -> Option<(&'a str, &'a Value)> {
    for entry in path.iter().rev() {
        if entry_type(entry) == "reset_boundary" {
            return None;
        }
        if entry_type(entry) != "custom" || str_field(entry, "customType") != Some(CONTEXT_NOTES_ENTRY_TYPE) {
            continue;
        }
        let Some(data) = entry.get("data").and_then(Value::as_object) else { continue };
        let version_ok = data.get("version").and_then(Value::as_f64) == Some(1.0);
        let text = data.get("text").and_then(Value::as_str);
        if data.len() == 2 && version_ok {
            if let Some(text) = text.filter(|t| t.len() <= MAX_CONTEXT_NOTES_BYTES) {
                return Some((text, entry));
            }
        }
    }
    None
}

/// Port of `buildSessionContext(entries, leafId)` (model-context mode).
///
/// `entries` are the session entries in file order, header and `title` slot excluded.
///
/// # Errors
/// [`ContextError::Unsupported`] when the latest compaction is an `experimental-context-rollover`.
// TS `buildSessionContext` is one function; the compaction emission block is kept inline for parity.
#[allow(clippy::too_many_lines)]
pub fn build_session_context(entries: &[Value], leaf: Leaf<'_>) -> Result<SessionContext, ContextError> {
    let path = branch_path(entries, leaf);
    let mut ctx = SessionContext { mode: "none".to_owned(), thinking_level: Some("off".to_owned()), ..SessionContext::default() };
    if path.is_empty() {
        return Ok(ctx);
    }

    let mut has_explicit_default_model = false;
    let mut compaction: Option<&Value> = None;
    let mut ttsr: Vec<String> = Vec::new();
    for entry in &path {
        match entry_type(entry) {
            "thinking_level_change" => {
                let level = entry.get("thinkingLevel").and_then(Value::as_str);
                ctx.thinking_level = Some(level.unwrap_or("off").to_owned());
                ctx.configured_thinking_level =
                    entry.get("configured").and_then(Value::as_str).or(level).map(str::to_owned);
            }
            "model_change" => {
                if let Some(model) = str_field(entry, "model").filter(|m| !m.is_empty()) {
                    let role = str_field(entry, "role").unwrap_or("default");
                    ctx.models.insert(role.to_owned(), model.into());
                    if role == "default" {
                        has_explicit_default_model = true;
                    }
                }
            }
            "service_tier_change" => ctx.service_tier = entry.get("serviceTier").filter(|v| !v.is_null()).cloned(),
            "message" => {
                if let Some(m) = entry.get("message").filter(|m| message_role(m) == "assistant") {
                    if !has_explicit_default_model {
                        let provider = str_field(m, "provider").unwrap_or("undefined");
                        let model = str_field(m, "model").unwrap_or("undefined");
                        ctx.models.insert("default".to_owned(), format!("{provider}/{model}").into());
                    }
                }
            }
            "compaction" => compaction = Some(entry),
            "ttsr_injection" => {
                for rule in entry.get("injectedRules").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
                    if !ttsr.iter().any(|r| r == rule) {
                        ttsr.push(rule.to_owned());
                    }
                }
            }
            "mode_change" => {
                str_field(entry, "mode").unwrap_or("none").clone_into(&mut ctx.mode);
                ctx.mode_data = entry.get("data").filter(|d| !d.is_null()).cloned();
            }
            _ => {}
        }
    }
    ctx.injected_ttsr_rules = ttsr;

    let reset_idx = path.iter().rposition(|e| entry_type(e) == "reset_boundary");
    let compaction_idx = compaction.and_then(|c| path.iter().position(|e| entry_type(e) == "compaction" && entry_id(e) == entry_id(c)));
    let mut messages: Vec<Value> = Vec::new();

    let after_reset = match (reset_idx, compaction_idx) {
        (Some(r), Some(c)) => r > c,
        (Some(_), None) => true,
        _ => false,
    };
    if after_reset {
        let r = reset_idx.unwrap_or(0);
        messages.extend(path[r + 1..].iter().filter_map(|e| context_message(e)));
    } else if let (Some(compaction), Some(compaction_idx)) = (compaction, compaction_idx) {
        // Rollover recovery (re-asking the discarded turn prefix from the
        // branch journal) is not ported.
        if compaction.get("details").and_then(|d| str_field(d, "kind")) == Some("experimental-context-rollover") {
            return Err(ContextError::Unsupported("experimental-context-rollover"));
        }
        let remote = native::openai_remote_payload(compaction);
        let anthropic = native::anthropic_compaction_payload(compaction);
        let provider_payload = remote.clone().or_else(|| anthropic.clone());
        let remote_replacement_history = remote.as_ref().and_then(|p| p.get("items")).filter(|i| i.is_array());

        // A natively replayed summary must not invalidate the retained tail's
        // bound thinking: predate the rewrite marker to just before the first
        // retained entry instead of the commit timestamp.
        let history_rewrite_at = if anthropic.is_some() {
            let first_kept_idx = path
                .iter()
                .position(|e| entry_id(e) == str_field(compaction, "firstKeptEntryId").unwrap_or(""));
            let snapshot_idx = if str_field(compaction, "firstKeptEntryId") == Some("") {
                str_field(compaction, "providerReplayThroughEntryId")
                    .and_then(|id| path.iter().position(|e| entry_id(e) == id))
            } else {
                None
            };
            let first_retained = (first_kept_idx.filter(|&i| i < compaction_idx).map(|i| path[i]))
                .or_else(|| snapshot_idx.filter(|&i| i + 1 < compaction_idx).map(|i| path[i + 1]))
                .or(path.get(compaction_idx + 1).copied());
            first_retained
                .and_then(|e| str_field(e, "timestamp"))
                .and_then(parse_iso_ms)
                .map(|ms| ms_value(Some(ms - 1)))
        } else {
            None
        };

        // Re-attach any archived snapcompact frames so the model can keep
        // reading the archived history after every context rebuild.
        let blocks = native::get_preserved_archive(compaction.get("preserveData"))
            .map(|archive| native::history_blocks_for_context(&archive))
            .filter(|b| !b.is_empty());
        messages.push(create_compaction_summary_message_ext(compaction, provider_payload, blocks, history_rewrite_at));

        // The remote replacement payload (OpenAI remote compaction) carries
        // the kept turns for the LLM context only; without it, replay the kept
        // tail from the entries (Anthropic with an empty `firstKeptEntryId`
        // falls back to the provider snapshot entry).
        if remote_replacement_history.is_none() {
            let first_kept_idx = path.iter().enumerate().position(|(i, e)| {
                i < compaction_idx && entry_id(e) == str_field(compaction, "firstKeptEntryId").unwrap_or("")
            });
            let snapshot_idx = if anthropic.is_some() && str_field(compaction, "firstKeptEntryId") == Some("") {
                str_field(compaction, "providerReplayThroughEntryId")
                    .and_then(|id| path.iter().position(|e| entry_id(e) == id))
            } else {
                None
            };
            let retained_start = first_kept_idx.or_else(|| snapshot_idx.map(|i| i + 1));
            if let Some(start) = retained_start.filter(|&s| s < compaction_idx) {
                messages.extend(path[start..compaction_idx].iter().filter_map(|e| context_message(e)));
            }
        } else if let Some(replay_through) = str_field(compaction, "providerReplayThroughEntryId")
            .and_then(|id| path.iter().position(|e| entry_id(e) == id))
        {
            if replay_through < compaction_idx {
                messages.extend(path[replay_through + 1..compaction_idx].iter().filter_map(|e| context_message(e)));
            }
        }
        messages.extend(path[compaction_idx + 1..].iter().filter_map(|e| context_message(e)));
    } else {
        messages.extend(path.iter().filter_map(|e| context_message(e)));
    }

    if let Some((text, source)) = context_notes(&path) {
        let rendered = prompts::render(prompts::CONTEXT_NOTES, &[("notes", text)]).trim().to_owned();
        if !text.is_empty() && !rendered.is_empty() {
            let mut m = Map::new();
            m.insert("role".into(), "custom".into());
            m.insert("customType".into(), CONTEXT_NOTES_ENTRY_TYPE.into());
            m.insert("content".into(), rendered.into());
            m.insert("display".into(), false.into());
            m.insert("timestamp".into(), ms_value(str_field(source, "timestamp").and_then(parse_iso_ms)));
            messages.insert(0, Value::Object(m));
        }
    }

    strip_dangling_tool_calls(&mut messages);
    drop_failed_assistant_turns(&mut messages);
    ctx.messages = messages;
    Ok(ctx)
}

/// Strip `toolCall` blocks with no matching `toolResult` anywhere in the context, together with
/// `redactedThinking` and thinking signatures of the rewritten turn; drop turns left empty.
fn strip_dangling_tool_calls(messages: &mut Vec<Value>) {
    let paired: BTreeSet<String> = messages
        .iter()
        .filter(|m| message_role(m) == "toolResult")
        .filter_map(|m| str_field(m, "toolCallId").map(str::to_owned))
        .collect();
    let is_dangling = |block: &Value| str_field(block, "type") == Some("toolCall") && !str_field(block, "id").is_some_and(|id| paired.contains(id));
    for i in (0..messages.len()).rev() {
        if message_role(&messages[i]) != "assistant" {
            continue;
        }
        let blocks = messages[i].get("content").and_then(Value::as_array).cloned().unwrap_or_default();
        if !blocks.iter().any(is_dangling) {
            continue;
        }
        let normalized: Vec<Value> = blocks
            .into_iter()
            .filter(|b| !is_dangling(b) && str_field(b, "type") != Some("redactedThinking"))
            .map(|mut b| {
                if str_field(&b, "type") == Some("thinking") && truthy(b.get("thinkingSignature")) {
                    if let Some(o) = b.as_object_mut() {
                        o.remove("thinkingSignature");
                    }
                }
                b
            })
            .collect();
        if normalized.is_empty() {
            messages.remove(i);
        } else if let Some(o) = messages[i].as_object_mut() {
            o.insert("content".into(), Value::Array(normalized));
        }
    }
}

/// Aborted / errored assistant turns are transcript events, not replayable context; their paired
/// tool results go with them (except the interrupted-thinking continuity pair).
fn drop_failed_assistant_turns(messages: &mut Vec<Value>) {
    for i in (0..messages.len()).rev() {
        if i >= messages.len() || message_role(&messages[i]) != "assistant" {
            continue;
        }
        if !matches!(str_field(&messages[i], "stopReason"), Some("aborted" | "error")) {
            continue;
        }
        if let Some(next) = messages.get(i + 1) {
            if message_role(next) == "custom" && str_field(next, "customType") == Some(INTERRUPTED_THINKING_MESSAGE_TYPE) {
                continue;
            }
        }
        let dropped: BTreeSet<String> = messages[i]
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|b| str_field(b, "type") == Some("toolCall"))
            .filter_map(|b| str_field(b, "id").map(str::to_owned))
            .collect();
        messages.remove(i);
        if !dropped.is_empty() {
            let mut j = messages.len();
            while j > i {
                j -= 1;
                if message_role(&messages[j]) == "toolResult" && str_field(&messages[j], "toolCallId").is_some_and(|id| dropped.contains(id)) {
                    messages.remove(j);
                }
            }
        }
    }
}
