//! Handoff document generation and commit (`SessionHandoff.generateDocument` +
//! `session-maintenance.ts#handoffSummaryFromDocument` / `appendCompaction`).
//!
//! The request is a normal oneshot LLM call over the caller-built context (same
//! system prompt and transformed history as a live turn, handoff prompt appended
//! as the last user message); the transport is injected via [`Completer`]. The
//! document commits as a plain `compaction` entry with `method: "handoff"`: the
//! cumulative `<files>` block is upserted onto the document and the entry
//! `details` carry the derived file lists (no `shortSummary`).

use anyhow::Result;
use serde_json::{Map, Value};

use crate::files::{compute_file_lists, upsert_file_operations, FileOperations};
use crate::json::str_field;
use crate::summary::{completion_text, Completer, SummaryKind, SummaryRequest};
use crate::{files, prompts, summary::CompactionResult};

/// `renderHandoffPrompt`: the handoff-document instructions, optionally with an
/// `<instruction>` block carrying `additionalFocus`.
#[must_use]
pub fn render_handoff_prompt(custom_instructions: Option<&str>) -> String {
    let template = prompts::HANDOFF_DOCUMENT;
    match custom_instructions.filter(|s| !s.is_empty()) {
        // `prompt.render(handoffDocumentPrompt)` with an empty context: handlebars
        // drops the standalone `{{#if additionalFocus}}...{{/if}}` section (the
        // marker lines removed whole, block content included only when truthy).
        None => {
            let mut lines = template.lines();
            let mut text = String::with_capacity(template.len());
            while let Some(line) = lines.next() {
                if line == "{{#if additionalFocus}}" {
                    for line in lines.by_ref() {
                        if line == "{{/if}}" {
                            break;
                        }
                    }
                    continue;
                }
                text.push_str(line);
                text.push('\n');
            }
            prompts::format(&text)
        }
        Some(focus) => {
            let text = template
                .replace("{{#if additionalFocus}}\n", "")
                .replace("\n{{/if}}", "")
                .replace("{{additionalFocus}}", focus);
            prompts::format(&text)
        }
    }
}

/// `buildHandoffRequest`: the caller's context (history + handoff prompt already
/// appended) sent as one oneshot with a `Handoff` oneshot kind.
#[must_use]
pub fn build_handoff_request(context: pa_types::ai::Context, max_tokens: u64) -> SummaryRequest {
    SummaryRequest {
        kind: SummaryKind::Handoff,
        context,
        max_tokens,
    }
}

/// `generateHandoffFromContext`: run the handoff oneshot and extract the
/// document text.
///
/// # Errors
/// The completer failed or replied with `stopReason: "error"`.
pub async fn generate_handoff_from_context(
    completer: &impl Completer,
    request: SummaryRequest,
) -> Result<String> {
    let reply = completer.complete(request).await?;
    completion_text(&reply, "Handoff generation failed")
}

/// `handoffSummaryFromDocument`: wrap the document as the compaction summary
/// (cumulative `<files>` block upserted) and derive the entry details.
#[must_use]
pub fn handoff_summary_from_document(document: &str, file_ops: &FileOperations) -> (String, Value) {
    let (read_files, modified_files) = compute_file_lists(file_ops);
    let summary =
        upsert_file_operations(document, &read_files, &modified_files, Some(&file_ops.read));
    let details = serde_json::json!({ "readFiles": read_files, "modifiedFiles": modified_files });
    (summary, details)
}

/// Everything [`handoff_compaction_entry`] / [`append_handoff`] need besides the
/// generated document.
#[derive(Debug, Clone)]
pub struct HandoffCommit<'a> {
    pub first_kept_entry_id: &'a str,
    pub tokens_before: u64,
    pub tokens_after: Option<u64>,
    pub file_ops: &'a FileOperations,
}

/// `handoffSummaryFromDocument` + the `appendCompaction` payload for a handoff.
/// Field order on the wire matches TS `appendCompaction` (`summary`, `shortSummary` omitted,
/// `firstKeptEntryId`, `tokensBefore`, `tokensAfter`, `method`, `details`, `fromExtension`).
#[must_use]
pub fn handoff_compaction_entry(
    id: &str,
    parent_id: Option<&str>,
    timestamp_iso: &str,
    document: &str,
    commit: &HandoffCommit<'_>,
) -> omp_session::entries::Compaction {
    let (summary, details) = handoff_summary_from_document(document, commit.file_ops);
    omp_session::entries::Compaction {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        timestamp: timestamp_iso.to_owned(),
        summary,
        short_summary: None,
        first_kept_entry_id: commit.first_kept_entry_id.to_owned(),
        tokens_before: serde_json::Number::from(commit.tokens_before),
        tokens_after: commit.tokens_after.map(serde_json::Number::from),
        method: Some("handoff".to_owned()),
        provider_replay_through_entry_id: None,
        details: Some(details),
        from_extension: Some(false),
        preserve_data: None,
        warning: None,
        rest: Map::new(),
    }
}

/// Append a handoff compaction to a session file; returns the new entry id.
///
/// # Errors
/// The write failed.
pub fn append_handoff(
    writer: &mut omp_session::SessionWriter,
    document: &str,
    timestamp_iso: &str,
    commit: &HandoffCommit<'_>,
) -> Result<String> {
    let (summary, details) = handoff_summary_from_document(document, commit.file_ops);
    let mut payload = Map::new();
    payload.insert("summary".into(), summary.into());
    payload.insert("firstKeptEntryId".into(), commit.first_kept_entry_id.into());
    payload.insert("tokensBefore".into(), commit.tokens_before.into());
    if let Some(after) = commit.tokens_after {
        payload.insert("tokensAfter".into(), after.into());
    }
    payload.insert("method".into(), "handoff".into());
    payload.insert("details".into(), details);
    payload.insert("fromExtension".into(), false.into());
    writer.append_entry("compaction", Value::Object(payload), timestamp_iso)
}

/// Rebuild a [`CompactionResult`]-shaped view of a committed handoff document so
/// [`apply::project_tokens_after`] can project `tokensAfter` (the summary message
/// renders through the handoff context prompt because `method` is `handoff`).
#[must_use]
pub fn handoff_result(
    document: &str,
    first_kept_entry_id: &str,
    tokens_before: u64,
    file_ops: &FileOperations,
) -> CompactionResult {
    let (summary, _) = handoff_summary_from_document(document, file_ops);
    CompactionResult {
        summary,
        short_summary: String::new(),
        first_kept_entry_id: first_kept_entry_id.to_owned(),
        tokens_before,
        details: serde_json::json!({}),
        preserve_data: None,
    }
}

/// Re-export so callers can build the file ops without reaching into [`files`].
pub use files::FileOperations as HandoffFileOperations;

/// Entry-level accessor used by tests: the summary of a compaction entry.
#[must_use]
pub fn entry_summary(entry: &Value) -> &str {
    str_field(entry, "summary").unwrap_or("")
}
