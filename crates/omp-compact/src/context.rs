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
use crate::messages::{
    create_branch_summary_message, create_compaction_summary_message_ext, custom_message_from_entry,
};
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
    !message
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|blocks| {
            blocks.iter().any(|block| match str_field(block, "type") {
                Some("text") => str_field(block, "text").is_some_and(|t| !t.trim().is_empty()),
                Some("thinking") => {
                    str_field(block, "thinking").is_some_and(|t| !t.trim().is_empty())
                        || str_field(block, "thinkingSignature")
                            .is_some_and(|t| !t.trim().is_empty())
                }
                Some("redactedThinking") => {
                    str_field(block, "data").is_some_and(|t| !t.trim().is_empty())
                }
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
            if message_role(message) == "assistant"
                && (truthy(message.get("retryRecovery")) || is_empty_error_turn(message))
            {
                return None;
            }
            Some(message.clone())
        }
        "custom_message" => {
            let custom_type = str_field(entry, "customType");
            if custom_type == Some(PREWALK_PLAN_MESSAGE_TYPE)
                || custom_type == Some(VIBE_MODE_CONTEXT_MESSAGE_TYPE)
            {
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
        if entry_type(entry) != "custom"
            || str_field(entry, "customType") != Some(CONTEXT_NOTES_ENTRY_TYPE)
        {
            continue;
        }
        let Some(data) = entry.get("data").and_then(Value::as_object) else {
            continue;
        };
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
pub fn build_session_context(
    entries: &[Value],
    leaf: Leaf<'_>,
) -> Result<SessionContext, ContextError> {
    let path = branch_path(entries, leaf);
    build_session_context_path(&path)
}

/// [`build_session_context`] over an already-resolved leaf branch (cheap
/// `&Value` refs — no entry DOM is cloned for the walk). Compaction planning
/// and the post-commit token projection use this so a compaction pass never
/// duplicates the whole entry list.
///
/// # Errors
/// [`ContextError::Unsupported`] when the latest compaction is an `experimental-context-rollover`.
#[allow(clippy::too_many_lines)] // one sequential context-derivation pass (was build_session_context)
pub fn build_session_context_path(path: &[&Value]) -> Result<SessionContext, ContextError> {
    let mut ctx = SessionContext {
        mode: "none".to_owned(),
        thinking_level: Some("off".to_owned()),
        ..SessionContext::default()
    };
    if path.is_empty() {
        return Ok(ctx);
    }

    let mut has_explicit_default_model = false;
    let mut compaction: Option<&Value> = None;
    let mut ttsr: Vec<String> = Vec::new();
    for entry in path {
        match entry_type(entry) {
            "thinking_level_change" => {
                let level = entry.get("thinkingLevel").and_then(Value::as_str);
                ctx.thinking_level = Some(level.unwrap_or("off").to_owned());
                ctx.configured_thinking_level = entry
                    .get("configured")
                    .and_then(Value::as_str)
                    .or(level)
                    .map(str::to_owned);
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
            "service_tier_change" => {
                ctx.service_tier = entry.get("serviceTier").filter(|v| !v.is_null()).cloned();
            }
            "message" => {
                if let Some(m) = entry
                    .get("message")
                    .filter(|m| message_role(m) == "assistant")
                {
                    if !has_explicit_default_model {
                        let provider = str_field(m, "provider").unwrap_or("undefined");
                        let model = str_field(m, "model").unwrap_or("undefined");
                        ctx.models
                            .insert("default".to_owned(), format!("{provider}/{model}").into());
                    }
                }
            }
            "compaction" => compaction = Some(entry),
            "ttsr_injection" => {
                for rule in entry
                    .get("injectedRules")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                {
                    if !ttsr.iter().any(|r| r == rule) {
                        ttsr.push(rule.to_owned());
                    }
                }
            }
            "mode_change" => {
                str_field(entry, "mode")
                    .unwrap_or("none")
                    .clone_into(&mut ctx.mode);
                ctx.mode_data = entry.get("data").filter(|d| !d.is_null()).cloned();
            }
            _ => {}
        }
    }
    ctx.injected_ttsr_rules = ttsr;

    let reset_idx = path.iter().rposition(|e| entry_type(e) == "reset_boundary");
    let compaction_idx = compaction.and_then(|c| {
        path.iter()
            .position(|e| entry_type(e) == "compaction" && entry_id(e) == entry_id(c))
    });
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
        if compaction.get("details").and_then(|d| str_field(d, "kind"))
            == Some("experimental-context-rollover")
        {
            return Err(ContextError::Unsupported("experimental-context-rollover"));
        }
        let remote = native::openai_remote_payload(compaction);
        let anthropic = native::anthropic_compaction_payload(compaction);
        let provider_payload = remote.clone().or_else(|| anthropic.clone());
        let remote_replacement_history = remote
            .as_ref()
            .and_then(|p| p.get("items"))
            .filter(|i| i.is_array());

        // A natively replayed summary must not invalidate the retained tail's
        // bound thinking: predate the rewrite marker to just before the first
        // retained entry instead of the commit timestamp.
        let history_rewrite_at = if anthropic.is_some() {
            let first_kept_idx = path.iter().position(|e| {
                entry_id(e) == str_field(compaction, "firstKeptEntryId").unwrap_or("")
            });
            let snapshot_idx = if str_field(compaction, "firstKeptEntryId") == Some("") {
                str_field(compaction, "providerReplayThroughEntryId")
                    .and_then(|id| path.iter().position(|e| entry_id(e) == id))
            } else {
                None
            };
            let first_retained = (first_kept_idx
                .filter(|&i| i < compaction_idx)
                .map(|i| path[i]))
            .or_else(|| {
                snapshot_idx
                    .filter(|&i| i + 1 < compaction_idx)
                    .map(|i| path[i + 1])
            })
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
        messages.push(create_compaction_summary_message_ext(
            compaction,
            provider_payload,
            blocks,
            history_rewrite_at,
        ));

        // The remote replacement payload (OpenAI remote compaction) carries
        // the kept turns for the LLM context only; without it, replay the kept
        // tail from the entries (Anthropic with an empty `firstKeptEntryId`
        // falls back to the provider snapshot entry).
        if remote_replacement_history.is_none() {
            let first_kept_idx = path.iter().enumerate().position(|(i, e)| {
                i < compaction_idx
                    && entry_id(e) == str_field(compaction, "firstKeptEntryId").unwrap_or("")
            });
            let snapshot_idx =
                if anthropic.is_some() && str_field(compaction, "firstKeptEntryId") == Some("") {
                    str_field(compaction, "providerReplayThroughEntryId")
                        .and_then(|id| path.iter().position(|e| entry_id(e) == id))
                } else {
                    None
                };
            let retained_start = first_kept_idx.or_else(|| snapshot_idx.map(|i| i + 1));
            if let Some(start) = retained_start.filter(|&s| s < compaction_idx) {
                messages.extend(
                    path[start..compaction_idx]
                        .iter()
                        .filter_map(|e| context_message(e)),
                );
            }
        } else if let Some(replay_through) = str_field(compaction, "providerReplayThroughEntryId")
            .and_then(|id| path.iter().position(|e| entry_id(e) == id))
        {
            if replay_through < compaction_idx {
                messages.extend(
                    path[replay_through + 1..compaction_idx]
                        .iter()
                        .filter_map(|e| context_message(e)),
                );
            }
        }
        messages.extend(
            path[compaction_idx + 1..]
                .iter()
                .filter_map(|e| context_message(e)),
        );
    } else {
        messages.extend(path.iter().filter_map(|e| context_message(e)));
    }

    if let Some((text, source)) = context_notes(path) {
        let rendered = prompts::render(prompts::CONTEXT_NOTES, &[("notes", text)])
            .trim()
            .to_owned();
        if !text.is_empty() && !rendered.is_empty() {
            let mut m = Map::new();
            m.insert("role".into(), "custom".into());
            m.insert("customType".into(), CONTEXT_NOTES_ENTRY_TYPE.into());
            m.insert("content".into(), rendered.into());
            m.insert("display".into(), false.into());
            m.insert(
                "timestamp".into(),
                ms_value(str_field(source, "timestamp").and_then(parse_iso_ms)),
            );
            messages.insert(0, Value::Object(m));
        }
    }

    strip_dangling_tool_calls(&mut messages);
    drop_failed_assistant_turns(&mut messages);
    ctx.messages = messages;
    Ok(ctx)
}

/// Owned variant of [`build_session_context`]: consumes the branch and *moves*
/// message payloads into the context instead of cloning them. Semantics are
/// identical (see the parity test in `tests/units.rs`); resume uses this so the
/// entry DOM and the rebuilt context never coexist.
///
/// # Errors
/// [`ContextError::Unsupported`] when the latest compaction is an `experimental-context-rollover`.
#[allow(clippy::too_many_lines)]
pub fn build_session_context_owned(
    entries: Vec<Value>,
    leaf: Leaf<'_>,
) -> Result<SessionContext, ContextError> {
    let mut path = branch_path_owned(entries, leaf);
    let mut ctx = SessionContext {
        mode: "none".to_owned(),
        thinking_level: Some("off".to_owned()),
        ..SessionContext::default()
    };
    if path.is_empty() {
        return Ok(ctx);
    }

    let mut has_explicit_default_model = false;
    let mut compaction_idx: Option<usize> = None;
    let mut ttsr: Vec<String> = Vec::new();
    for (i, entry) in path.iter().enumerate() {
        match entry_type(entry) {
            "thinking_level_change" => {
                let level = entry.get("thinkingLevel").and_then(Value::as_str);
                ctx.thinking_level = Some(level.unwrap_or("off").to_owned());
                ctx.configured_thinking_level = entry
                    .get("configured")
                    .and_then(Value::as_str)
                    .or(level)
                    .map(str::to_owned);
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
            "service_tier_change" => {
                ctx.service_tier = entry.get("serviceTier").filter(|v| !v.is_null()).cloned();
            }
            "message" => {
                if let Some(m) = entry
                    .get("message")
                    .filter(|m| message_role(m) == "assistant")
                {
                    if !has_explicit_default_model {
                        let provider = str_field(m, "provider").unwrap_or("undefined");
                        let model = str_field(m, "model").unwrap_or("undefined");
                        ctx.models
                            .insert("default".to_owned(), format!("{provider}/{model}").into());
                    }
                }
            }
            "compaction" => compaction_idx = Some(i),
            "ttsr_injection" => {
                for rule in entry
                    .get("injectedRules")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                {
                    if !ttsr.iter().any(|r| r == rule) {
                        ttsr.push(rule.to_owned());
                    }
                }
            }
            "mode_change" => {
                str_field(entry, "mode")
                    .unwrap_or("none")
                    .clone_into(&mut ctx.mode);
                ctx.mode_data = entry.get("data").filter(|d| !d.is_null()).cloned();
            }
            _ => {}
        }
    }
    ctx.injected_ttsr_rules = ttsr;

    let reset_idx = path.iter().rposition(|e| entry_type(e) == "reset_boundary");
    let mut messages: Vec<Value> = Vec::new();

    let after_reset = match (reset_idx, compaction_idx) {
        (Some(r), Some(c)) => r > c,
        (Some(_), None) => true,
        _ => false,
    };
    if after_reset {
        let r = reset_idx.unwrap_or(0);
        for e in path.iter_mut().skip(r + 1) {
            take_context_message(e, &mut messages);
        }
    } else if let Some(ci) = compaction_idx {
        // Rollover recovery (re-asking the discarded turn prefix from the
        // branch journal) is not ported.
        if path[ci].get("details").and_then(|d| str_field(d, "kind"))
            == Some("experimental-context-rollover")
        {
            return Err(ContextError::Unsupported("experimental-context-rollover"));
        }
        let remote = native::openai_remote_payload(&path[ci]);
        let anthropic = native::anthropic_compaction_payload(&path[ci]);
        let provider_payload = remote.clone().or_else(|| anthropic.clone());
        let remote_replacement_history = remote
            .as_ref()
            .and_then(|p| p.get("items"))
            .filter(|i| i.is_array());

        // A natively replayed summary must not invalidate the retained tail's
        // bound thinking: predate the rewrite marker to just before the first
        // retained entry instead of the commit timestamp.
        let history_rewrite_at = if anthropic.is_some() {
            let first_kept_idx = path.iter().position(|e| {
                entry_id(e) == str_field(&path[ci], "firstKeptEntryId").unwrap_or("")
            });
            let snapshot_idx = if str_field(&path[ci], "firstKeptEntryId") == Some("") {
                str_field(&path[ci], "providerReplayThroughEntryId")
                    .and_then(|id| path.iter().position(|e| entry_id(e) == id))
            } else {
                None
            };
            let first_retained = (first_kept_idx.filter(|&i| i < ci).map(|i| &path[i]))
                .or_else(|| snapshot_idx.filter(|&i| i + 1 < ci).map(|i| &path[i + 1]))
                .or(path.get(ci + 1));
            first_retained
                .and_then(|e| str_field(e, "timestamp"))
                .and_then(parse_iso_ms)
                .map(|ms| ms_value(Some(ms - 1)))
        } else {
            None
        };

        // Re-attach any archived snapcompact frames so the model can keep
        // reading the archived history after every context rebuild.
        let blocks = native::get_preserved_archive(path[ci].get("preserveData"))
            .map(|archive| native::history_blocks_for_context(&archive))
            .filter(|b| !b.is_empty());
        messages.push(create_compaction_summary_message_ext(
            &path[ci],
            provider_payload,
            blocks,
            history_rewrite_at,
        ));

        // The remote replacement payload (OpenAI remote compaction) carries
        // the kept turns for the LLM context only; without it, replay the kept
        // tail from the entries (Anthropic with an empty `firstKeptEntryId`
        // falls back to the provider snapshot entry).
        if remote_replacement_history.is_none() {
            let first_kept_idx = path.iter().enumerate().position(|(i, e)| {
                i < ci && entry_id(e) == str_field(&path[ci], "firstKeptEntryId").unwrap_or("")
            });
            let snapshot_idx =
                if anthropic.is_some() && str_field(&path[ci], "firstKeptEntryId") == Some("") {
                    str_field(&path[ci], "providerReplayThroughEntryId")
                        .and_then(|id| path.iter().position(|e| entry_id(e) == id))
                } else {
                    None
                };
            let retained_start = first_kept_idx.or_else(|| snapshot_idx.map(|i| i + 1));
            if let Some(start) = retained_start.filter(|&s| s < ci) {
                for e in path.iter_mut().take(ci).skip(start) {
                    take_context_message(e, &mut messages);
                }
            }
        } else if let Some(replay_through) = str_field(&path[ci], "providerReplayThroughEntryId")
            .and_then(|id| path.iter().position(|e| entry_id(e) == id))
        {
            if replay_through < ci {
                for e in path.iter_mut().take(ci).skip(replay_through + 1) {
                    take_context_message(e, &mut messages);
                }
            }
        }
        for e in path.iter_mut().skip(ci + 1) {
            take_context_message(e, &mut messages);
        }
    } else {
        for e in &mut path {
            take_context_message(e, &mut messages);
        }
    }

    if let Some((text, source)) = context_notes(&path.iter().collect::<Vec<_>>()) {
        let rendered = prompts::render(prompts::CONTEXT_NOTES, &[("notes", text)])
            .trim()
            .to_owned();
        if !text.is_empty() && !rendered.is_empty() {
            let mut m = Map::new();
            m.insert("role".into(), "custom".into());
            m.insert("customType".into(), CONTEXT_NOTES_ENTRY_TYPE.into());
            m.insert("content".into(), rendered.into());
            m.insert("display".into(), false.into());
            m.insert(
                "timestamp".into(),
                ms_value(str_field(source, "timestamp").and_then(parse_iso_ms)),
            );
            messages.insert(0, Value::Object(m));
        }
    }

    strip_dangling_tool_calls(&mut messages);
    drop_failed_assistant_turns(&mut messages);
    ctx.messages = messages;
    Ok(ctx)
}

/// Root→leaf path for the owned builder: same walk as [`branch_path`], but the
/// entries are moved out of the input instead of referenced.
fn branch_path_owned(entries: Vec<Value>, leaf: Leaf<'_>) -> Vec<Value> {
    if entries.is_empty() {
        return Vec::new();
    }
    let mut by_id: HashMap<String, usize> = HashMap::with_capacity(entries.len());
    for (i, e) in entries.iter().enumerate() {
        by_id.insert(entry_id(e).to_owned(), i);
    }
    let start = match leaf {
        Leaf::Before => return Vec::new(),
        Leaf::Id(id) => by_id.get(id).copied().unwrap_or(entries.len() - 1),
        Leaf::Last => entries.len() - 1,
    };
    let mut idx_path = Vec::new();
    let mut seen = BTreeSet::new();
    let mut cur = Some(start);
    while let Some(i) = cur {
        if !seen.insert(i) {
            break;
        }
        idx_path.push(i);
        let parent = entries[i]
            .get("parentId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        cur = if parent.is_empty() {
            None
        } else {
            by_id.get(&parent).copied()
        };
    }
    let mut slots: Vec<Option<Value>> = entries.into_iter().map(Some).collect();
    idx_path
        .iter()
        .rev()
        .map(|&i| slots[i].take().expect("path indices are unique"))
        .collect()
}

/// Move a branch entry's context message out, leaving non-emitting entries
/// (custom, `reset_boundary`, bookkeeping, shims) intact for the later scans.
fn take_context_message(entry: &mut Value, messages: &mut Vec<Value>) {
    if matches!(
        entry_type(entry),
        "message" | "custom_message" | "branch_summary"
    ) {
        if let Some(m) = context_message_owned(std::mem::take(entry)) {
            messages.push(m);
        }
    }
}

/// Owned `appendMessage`: moves the payload out of the entry instead of cloning
/// (`message.message`, `custom_message.content`, ...).
fn context_message_owned(entry: Value) -> Option<Value> {
    match entry_type(&entry) {
        "message" => {
            let keep = entry.get("message").is_some_and(|message| {
                !(message_role(message) == "assistant"
                    && (truthy(message.get("retryRecovery")) || is_empty_error_turn(message)))
            });
            if !keep {
                return None;
            }
            let Value::Object(mut map) = entry else {
                return None;
            };
            map.remove("message")
        }
        "custom_message" => {
            let custom_type = str_field(&entry, "customType");
            if custom_type == Some(PREWALK_PLAN_MESSAGE_TYPE)
                || custom_type == Some(VIBE_MODE_CONTEXT_MESSAGE_TYPE)
            {
                return None;
            }
            custom_message_from_entry_owned(entry)
        }
        "branch_summary" if truthy(entry.get("summary")) => Some(create_branch_summary_message(
            str_field(&entry, "summary").unwrap_or(""),
            str_field(&entry, "fromId").unwrap_or(""),
            str_field(&entry, "timestamp").unwrap_or(""),
        )),
        _ => None,
    }
}

/// `custom_message_from_entry` with the content/details moved out of the entry.
fn custom_message_from_entry_owned(entry: Value) -> Option<Value> {
    const DEFAULT_CUSTOM_MESSAGE_TYPE: &str = "custom-message";
    let Value::Object(mut map) = entry else {
        return None;
    };
    let content = map.get("content")?;
    if !(content.is_string() || content.is_array()) {
        return None;
    }
    let custom_type = map
        .get("customType")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_CUSTOM_MESSAGE_TYPE)
        .to_owned();
    let mut m = Map::new();
    m.insert("role".into(), "custom".into());
    m.insert("customType".into(), custom_type.into());
    m.insert("content".into(), map.remove("content")?);
    m.insert(
        "display".into(),
        map.get("display")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            .into(),
    );
    if let Some(details) = map.remove("details").filter(|d| !d.is_null()) {
        m.insert("details".into(), details);
    }
    if map.get("attribution").is_some_and(|a| !a.is_null()) {
        let attribution = if map.get("attribution").and_then(Value::as_str) == Some("user") {
            "user"
        } else {
            "agent"
        };
        m.insert("attribution".into(), attribution.into());
    }
    m.insert(
        "timestamp".into(),
        ms_value(
            map.get("timestamp")
                .and_then(Value::as_str)
                .and_then(parse_iso_ms),
        ),
    );
    Some(Value::Object(m))
}

/// Strip `toolCall` blocks with no matching `toolResult` anywhere in the context, together with
/// `redactedThinking` and thinking signatures of the rewritten turn; drop turns left empty.
fn strip_dangling_tool_calls(messages: &mut Vec<Value>) {
    let paired: BTreeSet<String> = messages
        .iter()
        .filter(|m| message_role(m) == "toolResult")
        .filter_map(|m| str_field(m, "toolCallId").map(str::to_owned))
        .collect();
    let is_dangling = |block: &Value| {
        str_field(block, "type") == Some("toolCall")
            && !str_field(block, "id").is_some_and(|id| paired.contains(id))
    };
    messages.retain_mut(|m| {
        if message_role(m) != "assistant" {
            return true;
        }
        let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) else {
            return true;
        };
        if !blocks.iter().any(is_dangling) {
            return true;
        }
        blocks.retain(|b| !is_dangling(b) && str_field(b, "type") != Some("redactedThinking"));
        for b in blocks.iter_mut() {
            if str_field(b, "type") == Some("thinking") && truthy(b.get("thinkingSignature")) {
                if let Some(o) = b.as_object_mut() {
                    o.remove("thinkingSignature");
                }
            }
        }
        !blocks.is_empty()
    });
}

/// Aborted / errored assistant turns are transcript events, not replayable context; their paired
/// tool results go with them (except the interrupted-thinking continuity pair).
///
/// One pass, O(n·α): walks from the end like the reference (each turn sees the first message
/// still alive after it), with a path-compressed "next alive index" and a toolCallId → result
/// index map instead of `Vec::remove` per dropped turn/result.
fn drop_failed_assistant_turns(messages: &mut Vec<Value>) {
    let n = messages.len();
    let is_failed = |m: &Value| {
        message_role(m) == "assistant"
            && matches!(str_field(m, "stopReason"), Some("aborted" | "error"))
    };
    if !messages.iter().any(is_failed) {
        return;
    }
    let mut results_by_call: HashMap<&str, Vec<usize>> = HashMap::new();
    for (j, m) in messages.iter().enumerate() {
        if message_role(m) == "toolResult" {
            if let Some(id) = str_field(m, "toolCallId") {
                results_by_call.entry(id).or_default().push(j);
            }
        }
    }
    let mut alive = vec![true; n];
    // next[x] -> smallest alive index >= x (n = none); path-compressed on lookup.
    let mut next: Vec<usize> = (0..=n).collect();
    fn find(next: &mut [usize], mut x: usize) -> usize {
        let mut root = x;
        while next[root] != root {
            root = next[root];
        }
        while next[x] != root {
            let up = next[x];
            next[x] = root;
            x = up;
        }
        root
    }
    fn kill(alive: &mut [bool], next: &mut [usize], j: usize) {
        alive[j] = false;
        next[j] = j + 1;
    }
    for i in (0..n).rev() {
        if !alive[i] || !is_failed(&messages[i]) {
            continue;
        }
        let nx = find(&mut next, i + 1);
        if nx < n {
            let m = &messages[nx];
            if message_role(m) == "custom"
                && str_field(m, "customType") == Some(INTERRUPTED_THINKING_MESSAGE_TYPE)
            {
                continue;
            }
        }
        kill(&mut alive, &mut next, i);
        for block in messages[i]
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if str_field(block, "type") != Some("toolCall") {
                continue;
            }
            let Some(js) = str_field(block, "id").and_then(|id| results_by_call.get(id)) else {
                continue;
            };
            for &j in js.iter().rev() {
                if j <= i {
                    break;
                }
                if alive[j] {
                    kill(&mut alive, &mut next, j);
                }
            }
        }
    }
    let mut k = 0;
    messages.retain(|_| {
        k += 1;
        alive[k - 1]
    });
}

#[cfg(test)]
mod drop_tests {
    use super::*;
    use serde_json::json;

    /// The previous O(k·n) implementation, kept verbatim as the equivalence oracle.
    fn reference(messages: &mut Vec<Value>) {
        for i in (0..messages.len()).rev() {
            if i >= messages.len() || message_role(&messages[i]) != "assistant" {
                continue;
            }
            if !matches!(
                str_field(&messages[i], "stopReason"),
                Some("aborted" | "error")
            ) {
                continue;
            }
            if let Some(next) = messages.get(i + 1) {
                if message_role(next) == "custom"
                    && str_field(next, "customType") == Some(INTERRUPTED_THINKING_MESSAGE_TYPE)
                {
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
                    if message_role(&messages[j]) == "toolResult"
                        && str_field(&messages[j], "toolCallId")
                            .is_some_and(|id| dropped.contains(id))
                    {
                        messages.remove(j);
                    }
                }
            }
        }
    }

    /// Deterministic xorshift so the test needs no extra dev-dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn random_transcript(r: &mut Rng, n: usize) -> Vec<Value> {
        let ids = ["a", "b", "c", "d"];
        (0..n)
            .map(|k| match r.below(6) {
                0 | 1 => {
                    let stop = ["stop", "toolUse", "aborted", "error"][r.below(4) as usize];
                    let calls: Vec<Value> = (0..r.below(3))
                        .map(|_| json!({"type": "toolCall", "id": ids[r.below(4) as usize]}))
                        .collect();
                    json!({"role": "assistant", "stopReason": stop, "content": calls, "k": k})
                }
                2 | 3 => json!({"role": "toolResult", "toolCallId": ids[r.below(4) as usize], "k": k}),
                4 => json!({"role": "custom", "customType": INTERRUPTED_THINKING_MESSAGE_TYPE, "k": k}),
                _ => json!({"role": "user", "k": k}),
            })
            .collect()
    }

    #[test]
    fn single_pass_matches_reference() {
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        for case in 0..20_000 {
            let n = (case % 40) as usize;
            let msgs = random_transcript(&mut r, n);
            let (mut a, mut b) = (msgs.clone(), msgs);
            drop_failed_assistant_turns(&mut a);
            reference(&mut b);
            assert_eq!(a, b, "case {case}");
        }
    }
}
