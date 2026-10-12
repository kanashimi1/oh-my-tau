//! Cut-point selection and compaction preparation (pi-agent-core `compaction.ts`:
//! `findValidCutPoints`, `findCutPoint`, `prepareCompaction`).

use serde_json::Value;

use crate::files::{extract_file_ops_from_message, FileOperations};
use crate::json::{entry_id, entry_type, message_role, str_field};
use crate::messages::entry_message;
use crate::settings::{
    calculate_context_tokens, calculate_prompt_tokens, last_assistant_usage, CompactionSettings,
};
use crate::tokens::{count_message, count_messages, CountOptions, Tokenizer};

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

/// Port of `prepareCompaction` for the local (summary) path.
///
/// `path_entries` is the active branch in order (see [`crate::context::branch_path`]). Returns
/// `Ok(None)` when compaction would be a no-op: the branch already ends in a compaction, the kept
/// entry has no id, or there is nothing to summarize.
///
/// # Errors
/// Never currently (`PlanError` is reserved for future boundary failures).
#[allow(clippy::too_many_lines)] // the cut pass: budget walk + seam selection in one place
pub fn plan_compaction(
    path_entries: &[Value],
    settings: &CompactionSettings,
    tokenizer: &dyn Tokenizer,
) -> Result<Option<CompactionPreparation>, PlanError> {
    if path_entries.last().is_some_and(|e| {
        entry_type(e) == "compaction"
            && e.get("providerReplayThroughEntryId")
                .is_none_or(Value::is_null)
    }) {
        return Ok(None);
    }
    let prev_compaction_idx = path_entries
        .iter()
        .rposition(|e| entry_type(e) == "compaction");
    let reset_idx = path_entries
        .iter()
        .rposition(|e| entry_type(e) == "reset_boundary");
    // A newer reset clears the previous summary.
    let prev_compaction_idx = match (prev_compaction_idx, reset_idx) {
        (Some(c), Some(r)) if r > c => None,
        (c, _) => c,
    };
    let previous = prev_compaction_idx.map(|i| &path_entries[i]);
    let mut boundary_start = prev_compaction_idx.max(reset_idx).map_or(0, |i| i + 1);
    if let (Some(prev), Some(prev_idx)) = (previous, prev_compaction_idx) {
        let replay_through = str_field(prev, "providerReplayThroughEntryId").unwrap_or("");
        let remote = prev
            .get("preserveData")
            .and_then(|p| p.get("openaiRemoteCompaction"))
            .is_some_and(|c| c.get("replacementHistory").is_some_and(Value::is_array));
        if remote {
            // A remote payload stores an opaque replay and a placeholder summary;
            // re-expand past it. Native replay covers the snapshot, not messages
            // appended while the request ran — include that interval.
            if !replay_through.is_empty() {
                if let Some(rt) =
                    (0..prev_idx).find(|&i| entry_id(&path_entries[i]) == replay_through)
                {
                    boundary_start = rt.max(reset_idx.unwrap_or(0)) + 1;
                }
            }
        } else {
            // Local and Anthropic summaries exclude the retained tail, whose
            // original entries precede the record.
            let kept = str_field(prev, "firstKeptEntryId").unwrap_or("");
            let from = reset_idx.map_or(0, |r| r + 1);
            if let Some(i) = (from..prev_idx).find(|&i| entry_id(&path_entries[i]) == kept) {
                boundary_start = i;
            }
            if kept.is_empty() && !replay_through.is_empty() {
                // An empty snapshot tail can still have turns appended during
                // background compaction; those start after the snapshot.
                if let Some(s) =
                    (0..prev_idx).find(|&i| entry_id(&path_entries[i]) == replay_through)
                {
                    boundary_start = s + 1;
                }
            }
        }
    }

    let mut compaction_entries: Vec<&Value> = Vec::new();
    let mut compaction_messages: Vec<Value> = Vec::new();
    for entry in &path_entries[boundary_start..] {
        if let Some(message) = entry_message(entry) {
            compaction_entries.push(entry);
            compaction_messages.push(message);
        }
    }

    let last_usage = last_assistant_usage(path_entries);
    let tokens_before = last_usage.map_or(0, calculate_context_tokens);
    let mut keep_recent = settings.keep_recent_tokens;
    if let Some(usage) = last_usage {
        let estimated = count_messages(tokenizer, &compaction_messages, CountOptions::default());
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

    let cut = find_cut_point(
        &compaction_entries,
        tokenizer,
        0,
        compaction_entries.len(),
        keep_recent,
    );
    let Some(first_kept) = compaction_entries.get(cut.first_kept_entry_index) else {
        return Ok(None);
    };
    let first_kept_entry_id = entry_id(first_kept);
    if first_kept_entry_id.is_empty() {
        return Ok(None);
    }
    let first_kept_entry_id = first_kept_entry_id.to_owned();

    let history_end = if cut.is_split_turn {
        cut.turn_start_index.unwrap_or(cut.first_kept_entry_index)
    } else {
        cut.first_kept_entry_index
    };
    let messages_to_summarize = compaction_messages[..history_end].to_vec();
    let turn_prefix_messages = match (cut.is_split_turn, cut.turn_start_index) {
        (true, Some(ts)) => compaction_messages[ts..cut.first_kept_entry_index].to_vec(),
        _ => Vec::new(),
    };
    let recent_messages = compaction_messages[cut.first_kept_entry_index..].to_vec();
    if messages_to_summarize.is_empty() && turn_prefix_messages.is_empty() {
        return Ok(None);
    }
    let recent_entry_ids = compaction_entries[cut.first_kept_entry_index..]
        .iter()
        .map(|e| entry_id(e).to_owned())
        .collect();

    let mut file_ops = extract_file_operations(&messages_to_summarize, previous);
    if cut.is_split_turn {
        for m in &turn_prefix_messages {
            extract_file_ops_from_message(m, &mut file_ops);
        }
    }

    Ok(Some(CompactionPreparation {
        cut,
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        recent_messages,
        recent_entry_ids,
        is_split_turn: cut.is_split_turn,
        tokens_before,
        previous_summary: previous
            .and_then(|p| str_field(p, "summary"))
            .map(str::to_owned),
        previous_summary_timestamp: previous
            .and_then(|p| str_field(p, "timestamp"))
            .map(str::to_owned),
        previous_preserve_data: previous
            .and_then(|p| p.get("preserveData"))
            .filter(|d| !d.is_null())
            .cloned(),
        file_ops,
        settings: settings.clone(),
    }))
}
