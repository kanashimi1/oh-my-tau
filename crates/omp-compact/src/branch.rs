//! Branch summarization for tree navigation (`compaction/branch-summarization.ts`): what to
//! summarize when leaving a branch, the request, and the `branch_summary` entry.

use std::collections::BTreeSet;

use omp_session::entries::BranchSummary;
use pa_types::ai::AssistantMessage;
use serde_json::{json, Map, Value};

use crate::convert::convert_to_llm;
use crate::files::{
    compute_file_lists, extract_file_ops_from_message, strip_read_selector, upsert_file_operations,
    FileOperations,
};
use crate::json::{entry_id, entry_type, message_role, str_field};
use crate::messages::{
    create_branch_summary_message, create_compaction_summary_message, custom_message_from_entry,
};
use crate::prompts;
use crate::serialize::truncate_tool_result_for_summary;
use crate::summary::{build_context, completion_text, SummaryKind, SummaryRequest};
use crate::tokens::{count_message, CountOptions, Tokenizer};

/// Output budget of a branch summary.
pub const BRANCH_SUMMARY_MAX_TOKENS: u64 = 2048;

/// `collectEntriesForBranchSummary`: entries from the old leaf back to (excluding) the deepest
/// common ancestor with `target_id`'s branch, chronological; plus that ancestor's id.
#[must_use]
pub fn collect_entries_for_branch_summary<'a>(
    entries: &'a [Value],
    old_leaf_id: Option<&str>,
    target_id: &str,
) -> (Vec<&'a Value>, Option<String>) {
    let Some(old_leaf) = old_leaf_id.filter(|l| !l.is_empty()) else {
        return (Vec::new(), None);
    };
    let old_path: BTreeSet<&str> =
        crate::context::branch_path(entries, crate::context::Leaf::Id(old_leaf))
            .iter()
            .map(|e| entry_id(e))
            .collect();
    let target_path = crate::context::branch_path(entries, crate::context::Leaf::Id(target_id));
    let common = target_path
        .iter()
        .rev()
        .map(|e| entry_id(e))
        .find(|id| old_path.contains(id))
        .map(str::to_owned);
    let by_id: std::collections::HashMap<&str, &Value> =
        entries.iter().map(|e| (entry_id(e), e)).collect();
    let mut out = Vec::new();
    let mut current = Some(old_leaf.to_owned());
    while let Some(id) = current.filter(|c| Some(c) != common.as_ref()) {
        let Some(entry) = by_id.get(id.as_str()) else {
            break;
        };
        out.push(*entry);
        current = str_field(entry, "parentId").map(str::to_owned);
    }
    out.reverse();
    (out, common)
}

/// `getMessageFromEntry` of branch-summarization (also maps `compaction` entries, and skips
/// useless non-error tool results).
fn branch_entry_message(entry: &Value) -> Option<Value> {
    match entry_type(entry) {
        "message" => {
            let m = entry.get("message")?;
            if message_role(m) == "toolResult"
                && m.get("useless") == Some(&Value::Bool(true))
                && m.get("isError") != Some(&Value::Bool(true))
            {
                return None;
            }
            Some(m.clone())
        }
        "custom_message" => custom_message_from_entry(entry),
        "branch_summary" => Some(create_branch_summary_message(
            str_field(entry, "summary").unwrap_or(""),
            str_field(entry, "fromId").unwrap_or(""),
            str_field(entry, "timestamp").unwrap_or(""),
        )),
        "compaction" => Some(create_compaction_summary_message(entry)),
        _ => None,
    }
}

/// Token estimate where tool results are priced at their summary-truncated size.
fn estimate_branch_summary_tokens(message: &Value, tokenizer: &dyn Tokenizer) -> u64 {
    if message_role(message) != "toolResult" {
        return count_message(tokenizer, message, CountOptions::default());
    }
    let text: String = message
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|b| str_field(b, "type") == Some("text"))
        .filter_map(|b| str_field(b, "text"))
        .collect();
    if text.is_empty() {
        return 0;
    }
    let mut truncated = message.clone();
    if let Some(o) = truncated.as_object_mut() {
        o.insert(
            "content".into(),
            json!([{ "type": "text", "text": truncate_tool_result_for_summary(&text) }]),
        );
    }
    count_message(tokenizer, &truncated, CountOptions::default())
}

/// `BranchPreparation`.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchPreparation {
    pub messages: Vec<Value>,
    pub file_ops: FileOperations,
    pub total_tokens: u64,
}

/// `prepareBranchEntries`: walk newest to oldest adding messages until `token_budget` (0 = no
/// limit); summaries are squeezed in while under 90% of the budget. File operations come from
/// every entry (nested branch-summary details included).
#[must_use]
pub fn prepare_branch_entries(
    entries: &[&Value],
    tokenizer: &dyn Tokenizer,
    token_budget: u64,
) -> BranchPreparation {
    let mut file_ops = FileOperations::default();
    for entry in entries {
        if entry_type(entry) == "branch_summary"
            && entry.get("fromExtension").and_then(Value::as_bool) != Some(true)
        {
            if let Some(details) = entry.get("details").filter(|d| !d.is_null()) {
                for f in details
                    .get("readFiles")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                {
                    file_ops.read.insert(strip_read_selector(f));
                }
                for f in details
                    .get("modifiedFiles")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                {
                    file_ops.edited.insert(f.to_owned());
                }
            }
        }
    }
    let mut messages: Vec<Value> = Vec::new();
    let mut total = 0u64;
    for entry in entries.iter().rev() {
        let Some(message) = branch_entry_message(entry) else {
            continue;
        };
        extract_file_ops_from_message(&message, &mut file_ops);
        let tokens = estimate_branch_summary_tokens(&message, tokenizer);
        if token_budget > 0 && total + tokens > token_budget {
            if matches!(entry_type(entry), "compaction" | "branch_summary")
                && (total as f64) < token_budget as f64 * 0.9
            {
                messages.insert(0, message);
                total += tokens;
            }
            break;
        }
        messages.insert(0, message);
        total += tokens;
    }
    BranchPreparation {
        messages,
        file_ops,
        total_tokens: total,
    }
}

/// The branch-summary request (`generateBranchSummary`): the conversation serialized with the
/// dialect given, then `customInstructions` or the default structured prompt.
#[must_use]
pub fn build_branch_summary_request(
    messages: &[Value],
    custom_instructions: Option<&str>,
    now_ms: u64,
    dialect: crate::dialect::Dialect,
) -> SummaryRequest {
    let conversation =
        crate::dialect::serialize_conversation_for_summary(&convert_to_llm(messages), dialect);
    let instructions = custom_instructions.filter(|c| !c.is_empty()).map_or_else(
        || prompts::render_static(prompts::BRANCH_SUMMARY),
        str::to_owned,
    );
    let text = format!("<conversation>\n{conversation}\n</conversation>\n\n{instructions}");
    SummaryRequest {
        kind: SummaryKind::BranchSummary,
        context: build_context(&text, now_ms),
        max_tokens: BRANCH_SUMMARY_MAX_TOKENS,
    }
}

/// Final branch-summary text and file lists: reply text, the exploration preamble prepended, and
/// the `<files>` block upserted. `"No summary generated"` when everything is empty.
///
/// # Errors
/// The reply ended with `stopReason: "error"`.
pub fn finish_branch_summary(
    reply: &AssistantMessage,
    file_ops: &FileOperations,
) -> anyhow::Result<(String, Vec<String>, Vec<String>)> {
    let text = completion_text(reply, "Summarization failed")?;
    let with_preamble = format!(
        "{}{text}",
        prompts::render_static(prompts::BRANCH_SUMMARY_PREAMBLE)
    );
    let (read_files, modified_files) = compute_file_lists(file_ops);
    let summary = upsert_file_operations(
        &with_preamble,
        &read_files,
        &modified_files,
        Some(&file_ops.read),
    );
    let summary = if summary.is_empty() {
        "No summary generated".to_owned()
    } else {
        summary
    };
    Ok((summary, read_files, modified_files))
}

/// The `branch_summary` entry for `branchWithSummary` (`fromId` is `"root"` when branching from
/// the start).
#[must_use]
pub fn branch_summary_entry(
    id: &str,
    branch_from_id: Option<&str>,
    timestamp_iso: &str,
    summary: &str,
    read_files: &[String],
    modified_files: &[String],
    from_extension: Option<bool>,
) -> BranchSummary {
    BranchSummary {
        id: id.to_owned(),
        parent_id: branch_from_id.map(str::to_owned),
        timestamp: timestamp_iso.to_owned(),
        from_id: branch_from_id.unwrap_or("root").to_owned(),
        summary: summary.to_owned(),
        details: Some(json!({ "readFiles": read_files, "modifiedFiles": modified_files })),
        from_extension,
        rest: Map::new(),
    }
}
