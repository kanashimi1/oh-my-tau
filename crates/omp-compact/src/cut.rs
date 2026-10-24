//! Cut-point selection and compaction preparation (pi-agent-core `compaction.ts`:
//! `findValidCutPoints`, `findCutPoint`, `prepareCompaction`).

use serde_json::Value;

use crate::files::{extract_file_ops_from_message, FileOperations};
use crate::json::{entry_id, entry_type, message_role, str_field};
use crate::messages::entry_message;
use crate::settings::{calculate_context_tokens, calculate_prompt_tokens, CompactionSettings};
use crate::tokens::{count_message, CountOptions, Tokenizer};

/// `CutPointResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutPoint {
    /// Index of the first entry to keep (within the entry slice that was searched).
    pub first_kept_entry_index: usize,
    /// Index of the user message that starts the turn being split; `None` when not splitting.
    pub turn_start_index: Option<usize>,
    /// The cut falls inside a turn (the cut entry is not a turn start).
    pub is_split_turn: bool,
}

/// `isTurnStartEntry`: a user request, bash-execution card, branch summary or custom message.
#[must_use]
pub fn is_turn_start_entry(entry: &Value) -> bool {
    match entry_type(entry) {
        "branch_summary" | "custom_message" => true,
        "message" => entry
            .get("message")
            .is_some_and(|m| matches!(message_role(m), "user" | "bashExecution")),
        _ => false,
    }
}

/// `findTurnStartIndex`: the turn start at or before `entry_index`, not before `start_index`.
#[must_use]
pub fn find_turn_start_index(
    entries: &[&Value],
    entry_index: usize,
    start_index: usize,
) -> Option<usize> {
    (start_index..=entry_index)
        .rev()
        .find(|&i| is_turn_start_entry(entries[i]))
}

/// `findValidCutPoints`: user/assistant-like messages, branch summaries and custom messages;
/// never a tool result.
fn find_valid_cut_points(entries: &[&Value], start: usize, end: usize) -> Vec<usize> {
    (start..end)
        .filter(|&i| {
            let entry = entries[i];
            match entry_type(entry) {
                "message" => entry.get("message").is_some_and(|m| {
                    matches!(
                        message_role(m),
                        "bashExecution"
                            | "hookMessage"
                            | "branchSummary"
                            | "compactionSummary"
                            | "user"
                            | "assistant"
                    )
                }),
                "branch_summary" | "custom_message" => true,
                _ => false,
            }
        })
        .collect()
}

/// `findCutPoint`: the oldest complete recent-history suffix that fits `keep_recent_tokens`
/// (the newest group is always kept, even when it alone exceeds the budget).
#[must_use]
pub fn find_cut_point(
    entries: &[&Value],
    tokenizer: &dyn Tokenizer,
    start: usize,
    end: usize,
    keep_recent_tokens: u64,
) -> CutPoint {
    let cut_points = find_valid_cut_points(entries, start, end);
    let Some(&newest) = cut_points.last() else {
        return CutPoint {
            first_kept_entry_index: start,
            turn_start_index: None,
            is_split_turn: false,
        };
    };
    let mut accumulated = 0u64;
    let mut cp_idx = cut_points.len() as isize - 1;
    let mut cut_index = newest;
    for i in (start..end).rev() {
        if let Some(message) = entry_message(entries[i]) {
            accumulated += count_message(tokenizer, &message, CountOptions::default());
        }
        let target = usize::try_from(cp_idx).ok().and_then(|c| cut_points.get(c));
        if target != Some(&i) {
            continue;
        }
        if accumulated > keep_recent_tokens {
            break;
        }
        cut_index = i;
        cp_idx -= 1;
    }

    let is_turn_start = is_turn_start_entry(entries[cut_index]);
    let turn_start_index = if is_turn_start {
        None
    } else {
        find_turn_start_index(entries, cut_index, start)
    };

    // Pull in non-message entries (settings changes, labels) directly before the cut.
    while cut_index > start {
        let prev = entries[cut_index - 1];
        if matches!(entry_type(prev), "compaction" | "reset_boundary")
            || entry_message(prev).is_some()
        {
            break;
        }
        cut_index -= 1;
    }
    CutPoint {
        first_kept_entry_index: cut_index,
        turn_start_index,
        is_split_turn: !is_turn_start && turn_start_index.is_some(),
    }
}

/// `CompactionPreparation`: what to summarize, what to keep, and the carried state.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPreparation {
    pub cut: CutPoint,
    /// Id of the first kept entry (`firstKeptEntryId`).
    pub first_kept_entry_id: String,
    /// Messages that will be summarized and discarded.
    pub messages_to_summarize: Vec<Value>,
    /// Messages of the split turn's prefix (summarized separately when `is_split_turn`).
    pub turn_prefix_messages: Vec<Value>,
    /// Messages kept verbatim after compaction.
    pub recent_messages: Vec<Value>,
    /// Entry ids parallel to `recent_messages`.
    pub recent_entry_ids: Vec<String>,
    pub is_split_turn: bool,
    pub tokens_before: u64,
    pub previous_summary: Option<String>,
    pub previous_summary_timestamp: Option<String>,
    pub previous_preserve_data: Option<Value>,
    pub file_ops: FileOperations,
    pub settings: CompactionSettings,
}

/// Why [`plan_compaction`] could not prepare a compaction (currently never: every lane present
/// in real sessions is handled; provider-native remote boundaries re-expand like TS).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {}

impl std::fmt::Display for PlanError {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl std::error::Error for PlanError {}

fn extract_file_operations(messages: &[Value], previous: Option<&Value>) -> FileOperations {
    let mut ops = FileOperations::default();
    if let Some(prev) = previous {
        if prev.get("fromExtension").and_then(Value::as_bool) != Some(true) {
            if let Some(details) = prev.get("details") {
                for f in details
                    .get("readFiles")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                {
                    ops.read.insert(crate::files::strip_read_selector(f));
                }
                for f in details
                    .get("modifiedFiles")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                {
                    ops.edited.insert(f.to_owned());
                }
            }
        }
    }
    for m in messages {
        extract_file_ops_from_message(m, &mut ops);
    }
    ops
}

/// Structural, content-free view of one journal entry: everything the
/// compaction boundary decision needs, scannable without retaining message
/// bodies (a session can hold tens of thousands of entries).
#[derive(Debug, Clone, Default)]
pub struct BoundaryEntry {
    /// Entry `type` (`""` when missing).
    pub entry_type: String,
    /// Entry id (`""` when missing).
    pub id: String,
    /// `providerReplayThroughEntryId` (compaction entries).
    pub replay_through: Option<String>,
    /// `firstKeptEntryId` (compaction entries).
    pub first_kept: Option<String>,
    /// `preserveData` (compaction entries; carried into the summary request).
    pub preserve_data: Option<Value>,
    /// The previous compaction is a provider-native remote payload.
    pub remote_replacement: bool,
    /// `summary` text (compaction entries).
    pub summary: Option<String>,
    /// Entry timestamp.
    pub timestamp: Option<String>,
    /// `message.role` (message entries).
    pub message_role: Option<String>,
    /// `message.stopReason` (message entries).
    pub stop_reason: Option<String>,
    /// `message.usage` (message entries; small numbers object).
    pub usage: Option<Value>,
}

impl BoundaryEntry {
    /// Extract the boundary-relevant fields from a parsed entry.
    #[must_use]
    pub fn from_value(entry: &Value) -> Self {
        let ty = entry_type(entry);
        let message = (ty == "message").then(|| entry.get("message")).flatten();
        Self {
            entry_type: ty.to_owned(),
            id: entry_id(entry).to_owned(),
            replay_through: str_field(entry, "providerReplayThroughEntryId").map(str::to_owned),
            first_kept: str_field(entry, "firstKeptEntryId").map(str::to_owned),
            preserve_data: entry.get("preserveData").filter(|d| !d.is_null()).cloned(),
            remote_replacement: entry
                .get("preserveData")
                .and_then(|p| p.get("openaiRemoteCompaction"))
                .is_some_and(|c| c.get("replacementHistory").is_some_and(Value::is_array)),
            summary: str_field(entry, "summary").map(str::to_owned),
            timestamp: str_field(entry, "timestamp").map(str::to_owned),
            message_role: message
                .and_then(|m| str_field(m, "role"))
                .map(str::to_owned),
            stop_reason: message
                .and_then(|m| str_field(m, "stopReason"))
                .map(str::to_owned),
            usage: message
                .and_then(|m| m.get("usage"))
                .filter(|u| crate::json::truthy(Some(u)))
                .cloned(),
        }
    }
}

/// Where the compactable region starts on the branch, plus the small facts the
/// region planning and the summary generation carry forward.
#[derive(Debug)]
pub struct CompactionBoundary {
    /// Branch index of the first entry in the compactable region.
    pub boundary_start: usize,
    /// The branch ends in a compaction (an immediate second compaction is a no-op).
    pub last_is_compaction: bool,
    /// Branch index of the previous compaction entry, when one applies.
    pub prev_compaction_idx: Option<usize>,
    /// `tokensBefore`: context tokens billed for the newest successful assistant turn.
    pub tokens_before: u64,
    /// Usage of the newest successful assistant message (whole branch).
    pub usage: Option<Value>,
    pub previous_summary: Option<String>,
    pub previous_summary_timestamp: Option<String>,
    pub previous_preserve_data: Option<Value>,
}

/// The boundary half of `prepareCompaction` over content-free headers, so the
/// caller only fully parses the compactable region itself. Returns `None` when
/// the branch already ends in a compaction.
#[must_use]
pub fn plan_boundary(
    path: &[BoundaryEntry],
    _settings: &CompactionSettings,
) -> Option<CompactionBoundary> {
    let last_is_compaction = path
        .last()
        .is_some_and(|e| e.entry_type == "compaction" && e.replay_through.is_none());
    if last_is_compaction {
        return None;
    }
    let prev_compaction_idx = path.iter().rposition(|e| e.entry_type == "compaction");
    let reset_idx = path.iter().rposition(|e| e.entry_type == "reset_boundary");
    // A newer reset clears the previous summary.
    let prev_compaction_idx = match (prev_compaction_idx, reset_idx) {
        (Some(c), Some(r)) if r > c => None,
        (c, _) => c,
    };
    let mut boundary_start = prev_compaction_idx.max(reset_idx).map_or(0, |i| i + 1);
    if let Some(prev_idx) = prev_compaction_idx {
        let prev = &path[prev_idx];
        let replay_through = prev.replay_through.as_deref().unwrap_or("");
        if prev.remote_replacement {
            // A remote payload stores an opaque replay and a placeholder summary;
            // re-expand past it. Native replay covers the snapshot, not messages
            // appended while the request ran — include that interval.
            if !replay_through.is_empty() {
                if let Some(rt) = (0..prev_idx).find(|&i| path[i].id == replay_through) {
                    boundary_start = rt.max(reset_idx.unwrap_or(0)) + 1;
                }
            }
        } else {
            // Local and Anthropic summaries exclude the retained tail, whose
            // original entries precede the record.
            let kept = prev.first_kept.as_deref().unwrap_or("");
            let from = reset_idx.map_or(0, |r| r + 1);
            if let Some(i) = (from..prev_idx).find(|&i| path[i].id == kept) {
                boundary_start = i;
            }
            if kept.is_empty() && !replay_through.is_empty() {
                // An empty snapshot tail can still have turns appended during
                // background compaction; those start after the snapshot.
                if let Some(s) = (0..prev_idx).find(|&i| path[i].id == replay_through) {
                    boundary_start = s + 1;
                }
            }
        }
    }
    let usage = path.iter().rev().find_map(|e| {
        if e.entry_type != "message" || e.message_role.as_deref() != Some("assistant") {
            return None;
        }
        if matches!(e.stop_reason.as_deref(), Some("aborted" | "error")) {
            return None;
        }
        e.usage.as_ref()
    });
    let prev = prev_compaction_idx.map(|i| &path[i]);
    Some(CompactionBoundary {
        boundary_start,
        last_is_compaction,
        prev_compaction_idx,
        tokens_before: usage.as_ref().map_or(0, |u| calculate_context_tokens(u)),
        usage: usage.cloned(),
        previous_summary: prev.and_then(|p| p.summary.clone()),
        previous_summary_timestamp: prev.and_then(|p| p.timestamp.clone()),
        previous_preserve_data: prev.and_then(|p| p.preserve_data.clone()),
    })
}

/// The region half of `prepareCompaction`: the cut point over the compactable
/// region (`branch[boundary_start..]`) and everything the summarizer needs.
///
/// Alongside the preparation, returns the kept-tail *entries* (moved out of the
/// owned region — the discarded history is dropped with the rest) so the caller
/// can rebuild the post-compaction context without re-reading the journal.
///
/// # Errors
/// Never currently (`PlanError` is reserved for future boundary failures).
#[allow(clippy::too_many_lines)] // the cut pass: budget walk + seam selection in one place
pub fn plan_region(
    mut region: Vec<Value>,
    boundary: &CompactionBoundary,
    previous: Option<&Value>,
    settings: &CompactionSettings,
    tokenizer: &dyn Tokenizer,
) -> Result<(Option<CompactionPreparation>, Vec<Value>), PlanError> {
    let tokens_before = boundary.tokens_before;
    let mut keep_recent = settings.keep_recent_tokens;
    if let Some(usage) = boundary.usage.as_ref() {
        // Transient per-entry counts for the provider-ratio adjustment; no
        // message list is retained.
        let estimated: u64 = region
            .iter()
            .map(|e| {
                entry_message(e)
                    .map_or(0, |m| count_message(tokenizer, &m, CountOptions::default()))
            })
            .sum();
        let prompt = calculate_prompt_tokens(usage);
        let ratio = if estimated > 0 {
            prompt as f64 / estimated as f64
        } else {
            0.0
        };
        if ratio.is_finite() && ratio > 1.0 {
            keep_recent = ((keep_recent as f64 / ratio).floor() as u64).max(1);
        }
    }

    let compaction_entries: Vec<&Value> = region
        .iter()
        .filter(|e| entry_message(e).is_some())
        .collect();
    let cut = find_cut_point(
        &compaction_entries,
        tokenizer,
        0,
        compaction_entries.len(),
        keep_recent,
    );
    let Some(first_kept) = compaction_entries.get(cut.first_kept_entry_index) else {
        return Ok((None, Vec::new()));
    };
    let first_kept_entry_id = entry_id(first_kept);
    if first_kept_entry_id.is_empty() {
        return Ok((None, Vec::new()));
    }
    let first_kept_entry_id = first_kept_entry_id.to_owned();
    let recent_entry_ids = compaction_entries[cut.first_kept_entry_index..]
        .iter()
        .map(|e| entry_id(e).to_owned())
        .collect();

    // Split the owned region at the first kept entry: the kept tail moves out
    // verbatim (it is small — the recent budget — so the prep's `recent_messages`
    // clone is cheap); the discarded history's payloads are moved into the
    // summary lists, so no second copy of the region DOM ever exists.
    let msg_pos: Vec<usize> = region
        .iter()
        .enumerate()
        .filter(|(_, e)| entry_message(e).is_some())
        .map(|(i, _)| i)
        .collect();
    let kept_start = msg_pos
        .get(cut.first_kept_entry_index)
        .copied()
        .unwrap_or(region.len());
    let kept_entries = region.split_off(kept_start);
    let recent_messages: Vec<Value> = kept_entries.iter().filter_map(entry_message).collect();

    let mut messages: Vec<Value> = Vec::with_capacity(kept_start);
    for mut e in region {
        match entry_type(&e) {
            "message" => {
                if let Some(m) = e.as_object_mut().and_then(|o| o.remove("message")) {
                    messages.push(m);
                }
            }
            _ => {
                if let Some(m) = entry_message(&e) {
                    messages.push(m);
                }
            }
        }
    }

    let mut messages_to_summarize = messages;
    let turn_prefix_messages = match (cut.is_split_turn, cut.turn_start_index) {
        (true, Some(ts)) => messages_to_summarize.split_off(ts),
        _ => Vec::new(),
    };
    if messages_to_summarize.is_empty() && turn_prefix_messages.is_empty() {
        return Ok((None, kept_entries));
    }

    let mut file_ops = extract_file_operations(&messages_to_summarize, previous);
    if cut.is_split_turn {
        for m in &turn_prefix_messages {
            extract_file_ops_from_message(m, &mut file_ops);
        }
    }

    let prep = CompactionPreparation {
        cut,
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        recent_messages,
        recent_entry_ids,
        is_split_turn: cut.is_split_turn,
        tokens_before,
        previous_summary: boundary.previous_summary.clone(),
        previous_summary_timestamp: boundary.previous_summary_timestamp.clone(),
        previous_preserve_data: boundary.previous_preserve_data.clone(),
        file_ops,
        settings: settings.clone(),
    };
    Ok((Some(prep), kept_entries))
}

/// Port of `prepareCompaction` for the local (summary) path.
///
/// `path_entries` is the active branch in order (see [`crate::context::branch_path`]). Returns
/// `Ok(None)` when compaction would be a no-op: the branch already ends in a compaction, the kept
/// entry has no id, or there is nothing to summarize. In-memory variant of the
/// [`plan_boundary`] + [`plan_region`] pair, which stream the journal instead.
///
/// # Errors
/// Never currently (`PlanError` is reserved for future boundary failures).
pub fn plan_compaction(
    path_entries: &[&Value],
    settings: &CompactionSettings,
    tokenizer: &dyn Tokenizer,
) -> Result<Option<CompactionPreparation>, PlanError> {
    let heads: Vec<BoundaryEntry> = path_entries
        .iter()
        .map(|e| BoundaryEntry::from_value(e))
        .collect();
    let Some(boundary) = plan_boundary(&heads, settings) else {
        return Ok(None);
    };
    let previous = boundary.prev_compaction_idx.map(|i| path_entries[i]);
    plan_region(
        path_entries[boundary.boundary_start..]
            .iter()
            .map(|e| (*e).clone())
            .collect(),
        &boundary,
        previous,
        settings,
        tokenizer,
    )
    .map(|(prep, _)| prep)
}
