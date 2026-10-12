//! Turning a [`CompactionResult`] into the durable `compaction` entry (`appendCompaction`) and
//! projecting the post-compaction context size (`tokensAfter`).

use anyhow::Result;
use omp_session::entries::Compaction;
use omp_session::SessionWriter;
use serde_json::{Map, Number, Value};

use crate::context::{build_session_context, ContextError, Leaf};
use crate::convert::convert_to_llm;
use crate::json::entry_id;
use crate::summary::CompactionResult;
use crate::tokens::{count_messages, CountOptions, Tokenizer};

/// Extra facts the harness records beside the summary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApplyOptions {
    /// `method` (`soft`, `handoff`, `remote`, ...).
    pub method: Option<String>,
    /// Projected context size after the commit, see [`project_tokens_after`].
    pub tokens_after: Option<u64>,
    /// `fromExtension` (the harness writes `false` for its own compactions).
    pub from_extension: Option<bool>,
}

/// Build the typed `compaction` entry for a result. Field order on the wire is
/// `summary, shortSummary, firstKeptEntryId, tokensBefore, tokensAfter, method, details,
/// fromExtension, preserveData` (omp-session's [`Compaction`] serializes in that order).
#[must_use]
pub fn compaction_entry(
    result: &CompactionResult,
    id: &str,
    parent_id: Option<&str>,
    timestamp_iso: &str,
    options: &ApplyOptions,
) -> Compaction {
    Compaction {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        timestamp: timestamp_iso.to_owned(),
        summary: result.summary.clone(),
        short_summary: Some(result.short_summary.clone()),
        first_kept_entry_id: result.first_kept_entry_id.clone(),
        tokens_before: Number::from(result.tokens_before),
        tokens_after: options.tokens_after.map(Number::from),
        method: options.method.clone(),
        provider_replay_through_entry_id: None,
        details: Some(result.details.clone()),
        from_extension: options.from_extension,
        preserve_data: result.preserve_data.clone(),
        warning: None,
        rest: Map::new(),
    }
}

/// Append the compaction to a session file; the writer chains `id`/`parentId`. Returns the new id.
///
/// # Errors
/// The payload does not fit the entry schema or the write failed.
pub fn append_compaction(
    writer: &mut SessionWriter,
    result: &CompactionResult,
    timestamp_iso: &str,
    options: &ApplyOptions,
) -> Result<String> {
    let mut payload = Map::new();
    payload.insert("summary".into(), result.summary.clone().into());
    payload.insert("shortSummary".into(), result.short_summary.clone().into());
    payload.insert(
        "firstKeptEntryId".into(),
        result.first_kept_entry_id.clone().into(),
    );
    payload.insert("tokensBefore".into(), result.tokens_before.into());
    if let Some(after) = options.tokens_after {
        payload.insert("tokensAfter".into(), after.into());
    }
    if let Some(method) = &options.method {
        payload.insert("method".into(), method.clone().into());
    }
    payload.insert("details".into(), result.details.clone());
    if let Some(from_extension) = options.from_extension {
        payload.insert("fromExtension".into(), from_extension.into());
    }
    if let Some(preserve) = &result.preserve_data {
        payload.insert("preserveData".into(), preserve.clone());
    }
    writer.append_entry("compaction", Value::Object(payload), timestamp_iso)
}

/// `#projectCompactionContextTokens` (local path): estimated context tokens right after the
/// commit. A synthetic compaction entry is appended to `branch`, the context is rebuilt and
/// converted exactly like the post-commit rebuild, and counted with encrypted reasoning excluded;
/// `non_message_tokens` is the system prompt + tool schema overhead the caller measures.
///
/// # Errors
/// The rebuilt context hit an unsupported payload.
pub fn project_tokens_after(
    branch: &[Value],
    result: &CompactionResult,
    method: Option<&str>,
    timestamp_iso: &str,
    non_message_tokens: u64,
    tokenizer: &dyn Tokenizer,
) -> Result<u64, ContextError> {
    let leaf_id = branch.last().map_or("", entry_id).to_owned();
    let mut pending = Map::new();
    pending.insert("type".into(), "compaction".into());
    pending.insert(
        "id".into(),
        format!("{leaf_id}:compaction-tokens-projection").into(),
    );
    pending.insert(
        "parentId".into(),
        if leaf_id.is_empty() {
            Value::Null
        } else {
            leaf_id.into()
        },
    );
    pending.insert("timestamp".into(), timestamp_iso.into());
    pending.insert("summary".into(), result.summary.clone().into());
    pending.insert("shortSummary".into(), result.short_summary.clone().into());
    pending.insert(
        "firstKeptEntryId".into(),
        result.first_kept_entry_id.clone().into(),
    );
    pending.insert("tokensBefore".into(), result.tokens_before.into());
    if let Some(m) = method {
        pending.insert("method".into(), m.into());
    }
    pending.insert("details".into(), result.details.clone());
    if let Some(p) = &result.preserve_data {
        pending.insert("preserveData".into(), p.clone());
    }
    let mut entries = branch.to_vec();
    entries.push(Value::Object(pending));
    let rebuilt = build_session_context(&entries, Leaf::Last)?;
    let llm = convert_to_llm(&rebuilt.messages);
    Ok(non_message_tokens
        + count_messages(
            tokenizer,
            &llm,
            CountOptions {
                exclude_encrypted_reasoning: true,
            },
        ))
}
