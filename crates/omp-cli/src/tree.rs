//! Session tree navigation behind the rpc commands and the `navigate_tree` host request:
//! `/tree` leaf moves, `branch`/`fork` into a new session file, and `branch_summary` markers.
//!
//! Ports of coding-agent `AgentSession.branch` / `AgentSession.fork` /
//! `AgentSession.navigateTree` and `SessionManager.createBranchedSession`, adapted to
//! omp-rs's append-only session writer: a branch or fork materializes as a new session
//! file holding the root-to-leaf path (which `load_session` then binds), and a leaf move
//! (`navigate_tree`) retargets the writer's append point so the next entry chains to the
//! selected node — exactly how omp's loader reconstructs the active branch from the last
//! physical journal entry once anything is appended there.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use omp_ext::lifecycle::BranchReason;

use crate::persist;
use crate::runtime::{CmdError, CmdResult, Runtime};

fn err(message: impl Into<String>) -> CmdError {
    CmdError::new(message.into())
}

/// Session entries (header excluded), raw values in file order.
fn entries(rt: &Runtime) -> Result<Vec<Value>, CmdError> {
    rt.entries()
        .map_err(|e| err(format!("cannot read session entries: {e:#}")))
}

/// `session_before_branch` (TS `#branchIntoNewSession`): `true` when an extension cancelled.
async fn branch_cancelled(rt: &Runtime, reason: BranchReason, entry_id: &str) -> bool {
    let Some(host) = rt
        .ext_host()
        .filter(|h| h.has_handlers("session_before_branch"))
    else {
        return false;
    };
    match host
        .emit_session_before_branch(reason, entry_id.to_owned(), None)
        .await
    {
        Ok(outcome) => outcome.cancel,
        Err(e) => {
            eprintln!("omt: extension session_before_branch: {e:#}");
            false
        }
    }
}

/// `session_branch`, after the new file is bound.
async fn announce_branch(rt: &Runtime, reason: BranchReason, previous: Option<String>) {
    if let Some(host) = rt.ext_host() {
        if let Err(e) = host.emit_session_branch(reason, previous, None).await {
            eprintln!("omt: extension session_branch: {e:#}");
        }
    }
}

fn previous_file(rt: &Runtime) -> Option<String> {
    rt.persist_file().map(|p| p.to_string_lossy().into_owned())
}

fn find_entry<'a>(entries: &'a [Value], id: &str) -> Option<&'a Value> {
    entries
        .iter()
        .find(|e| e.get("id").and_then(Value::as_str) == Some(id))
}

/// The abandoned branch rendered as one text block for the summarizer
/// (TS `serializeConversationForSummary`).
#[allow(dead_code)] // summarizer input for the navigate_tree host request
fn abandoned_text(abandoned: &[&Value]) -> String {
    let mut lines = Vec::new();
    for entry in abandoned {
        match entry.get("type").and_then(Value::as_str) {
            Some("message") => {
                if let Some(m) = entry.get("message") {
                    lines.push(m.to_string());
                }
            }
            Some("custom_message") => lines.push(
                json!({
                    "role": "user",
                    "content": entry.get("content").cloned().unwrap_or(Value::Null),
                    "timestamp": entry.get("timestamp").cloned().unwrap_or(Value::Null),
                })
                .to_string(),
            ),
            Some("branch_summary") => lines.push(
                json!({
                    "role": "branchSummary",
                    "summary": entry.get("summary").cloned().unwrap_or(Value::Null),
                    "fromId": entry.get("fromId").cloned().unwrap_or(Value::Null),
                    "timestamp": entry.get("timestamp").cloned().unwrap_or(Value::Null),
                })
                .to_string(),
            ),
            _ => {}
        }
    }
    lines.join("\n")
}

/// Port of `SessionManager.createBranchedSession`: a new session file in the current
/// session directory containing the v3 header plus the root-to-`leaf` path (`leaf = None`:
/// header only, like `newSession`). Label entries are dropped from the path and re-created
/// at the end from the labels in effect, as omp does. Original ids and parent links are
/// kept verbatim, so the copied path branches exactly like the source.
///
/// # Errors
/// The leaf is unknown or the file cannot be written.
pub fn write_branched_session(rt: &Runtime, leaf: Option<&str>) -> Result<PathBuf, CmdError> {
    let entries = entries(rt)?;
    let path = match leaf {
        None => Vec::new(),
        Some(id) => {
            let path = omp_session::tree::branch_path(&entries, id);
            if path.is_empty() {
                return Err(err(format!("Entry {id} not found")));
            }
            path.into_iter().cloned().collect::<Vec<_>>()
        }
    };

    let kept: Vec<&Value> = path
        .iter()
        .filter(|e| e.get("type").and_then(Value::as_str) != Some("label"))
        .collect();
    let kept_ids: std::collections::HashSet<&str> = kept
        .iter()
        .filter_map(|e| e.get("id").and_then(Value::as_str))
        .collect();
    let labels = omp_session::tree::labels_in_effect(&entries)
        .into_iter()
        .filter(|(target, _)| kept_ids.contains(target.as_str()))
        .collect::<Vec<_>>();

    let timestamp = persist::now_iso();
    let session_id = persist::uuid_v7();
    let dir = rt
        .session_file()
        .parent()
        .map_or_else(|| rt.cwd.clone(), PathBuf::from);
    let file_name = format!(
        "{}_{}.jsonl",
        timestamp.replace([':', '.'], "-"),
        session_id
    );
    let file = dir.join(file_name);

    let mut out = String::new();
    let header = json!({
        "type": "session",
        "version": omp_session::SESSION_VERSION,
        "id": session_id,
        "timestamp": timestamp,
        "cwd": rt.cwd.display().to_string(),
        "parentSession": rt.session_file().display().to_string(),
    });
    out.push_str(&header.to_string());
    out.push('\n');
    let mut parent = kept
        .last()
        .and_then(|e| e.get("id").and_then(Value::as_str))
        .map(str::to_owned);
    for entry in &kept {
        out.push_str(&entry.to_string());
        out.push('\n');
    }
    for (target, label) in labels {
        let id = persist::uuid_v7().replace('-', "")[..8].to_owned();
        let entry = json!({
            "type": "label",
            "id": id,
            "parentId": parent,
            "timestamp": persist::now_iso(),
            "targetId": target,
            "label": label,
        });
        parent = Some(id);
        out.push_str(&entry.to_string());
        out.push('\n');
    }
    std::fs::write(&file, out)
        .with_context(|| format!("write {}", file.display()))
        .map_err(|e| err(format!("{e:#}")))?;
    Ok(file)
}

/// rpc `branch` (TS `AgentSession.branch`): rewind to just before a user message, keeping
/// the old path only in the source file. The selected message's text is returned so the
/// host can re-prompt; the session continues in a new file holding the root-to-parent path.
pub async fn branch(rt: &Arc<Runtime>, entry_id: &str) -> CmdResult {
    let entries = entries(rt)?;
    let Some(selected) = find_entry(&entries, entry_id) else {
        return Err(err("Invalid entry ID for branching"));
    };
    if !omp_session::tree::is_user_message_entry(selected) {
        return Err(err("Invalid entry ID for branching"));
    }
    let text = omp_session::tree::user_message_text(selected).unwrap_or_default();
    let parent = selected.get("parentId").and_then(Value::as_str);
    if branch_cancelled(rt, BranchReason::Branch, entry_id).await {
        return Ok(Some(json!({"text": text, "cancelled": true})));
    }
    let previous = previous_file(rt);
    let file = write_branched_session(rt, parent)?;
    rt.load_branched_session(&file)
        .await
        .map_err(|e| err(format!("branch: {e:#}")))?;
    announce_branch(rt, BranchReason::Branch, previous).await;
    Ok(Some(json!({"text": text, "cancelled": false})))
}

/// rpc `fork` (TS `AgentSession.fork` with `requireIdle`, checked by the dispatcher):
/// without `entryId` a fresh session for the same cwd; with one, a new session file
/// holding the root-to-`entryId` path.
pub async fn fork(rt: &Arc<Runtime>, entry_id: Option<&str>) -> CmdResult {
    match entry_id {
        None => {
            rt.new_session(None)
                .await
                .map_err(|e| err(format!("fork: {e:#}")))?;
        }
        Some(id) => {
            if branch_cancelled(rt, BranchReason::Fork, id).await {
                return Ok(Some(json!({"cancelled": true})));
            }
            let previous = previous_file(rt);
            let file = write_branched_session(rt, Some(id))?;
            rt.load_branched_session(&file)
                .await
                .map_err(|e| err(format!("fork: {e:#}")))?;
            announce_branch(rt, BranchReason::Fork, previous).await;
        }
    }
    Ok(Some(json!({"cancelled": false})))
}

/// `navigate_tree` host request / `/tree` selection (the non-interactive core of TS
/// `AgentSession.navigateTree`): moves the live leaf onto `target` — or past it, when the
/// target is a user-authored message, so the draft can be re-entered — and rebuilds the
/// agent's context from the new branch. With `options.summarize`, the abandoned path is
/// summarized first and a `branch_summary` entry is recorded at the navigation point,
/// making the move durable in the journal.
///
/// `summarize` is the model call used for that (the runtime's compaction summarizer);
/// required exactly when `options.summarize` is set, like TS requires a model.
#[allow(dead_code)] // host-request entry point (omp parity), wired with the rpc surface
#[allow(clippy::too_many_lines)] // host-request flow: fork + optional branch summary + context rebuild
pub async fn navigate_tree(
    rt: &Arc<Runtime>,
    target: &str,
    options: &Value,
    summarize: Option<crate::compact::SummarizeFn>,
) -> CmdResult {
    let session = entries(rt)?;
    let old_leaf = omp_session::tree::leaf_id(&session).map(str::to_owned);
    let Some(target_entry) = find_entry(&session, target) else {
        return Err(err(format!("Entry {target} not found")));
    };

    let target_is_user = omp_session::tree::is_user_message_entry(target_entry)
        || (target_entry.get("type").and_then(Value::as_str) == Some("custom_message")
            && target_entry.get("customType").and_then(Value::as_str) != Some("skill-prompt"));
    if Some(target) == old_leaf.as_deref() && !target_is_user {
        return Ok(Some(json!({"cancelled": false})));
    }

    let wants_summary = options.get("summarize").and_then(Value::as_bool) == Some(true);

    // Where the leaf lands: user-authored targets rewind PAST themselves (null = root).
    let new_leaf = if target_is_user {
        target_entry
            .get("parentId")
            .and_then(Value::as_str)
            .map(str::to_owned)
    } else {
        Some(target.to_owned())
    };

    let (abandoned, common_ancestor) = match old_leaf.as_deref() {
        Some(old) => omp_session::tree::entries_for_branch_summary(&session, old, target),
        None => (Vec::new(), None),
    };
    // `session_before_tree`: a handler may cancel, or supply the summary itself.
    let mut summary_result: Option<(String, Value)> = None;
    let mut from_extension = false;
    if let Some(host) = rt
        .ext_host()
        .filter(|h| h.has_handlers("session_before_tree"))
    {
        let preparation = json!({
            "targetId": target,
            "oldLeafId": old_leaf,
            "commonAncestorId": common_ancestor,
            "entriesToSummarize": abandoned,
            "userWantsSummary": wants_summary,
        });
        match host.emit_session_before_tree(preparation, None).await {
            Ok(outcome) => {
                if outcome.cancel {
                    return Ok(Some(json!({"cancelled": true})));
                }
                if let (true, Some(text)) = (
                    wants_summary,
                    outcome
                        .result
                        .pointer("/summary/summary")
                        .and_then(Value::as_str),
                ) {
                    let details = outcome
                        .result
                        .pointer("/summary/details")
                        .cloned()
                        .unwrap_or(Value::Null);
                    summary_result = Some((text.to_owned(), details));
                    from_extension = true;
                }
            }
            Err(e) => eprintln!("omt: extension session_before_tree: {e:#}"),
        }
    }
    if wants_summary && summary_result.is_none() && summarize.is_none() {
        return Err(err("No model available for summarization"));
    }
    if wants_summary && summary_result.is_none() {
        if !abandoned.is_empty() {
            let request = omp_compact::SummaryRequest {
                kind: omp_compact::SummaryKind::BranchSummary,
                context: omp_compact::summary::build_context(
                    &abandoned_text(&abandoned),
                    pa_agent::now_ms().cast_unsigned(),
                ),
                max_tokens: 1024,
            };
            let assistant = (summarize.expect("checked above"))(request)
                .await
                .map_err(|e| err(format!("branch summary failed: {e:#}")))?;
            let text = assistant
                .content
                .iter()
                .filter_map(|c| match c {
                    pa_types::ai::AssistantContentBlock::Text(t) => Some(t.text.clone()),
                    _ => None,
                })
                .collect::<String>();
            summary_result = Some((text, json!({"readFiles": [], "modifiedFiles": []})));
        }
    }

    // Commit: either a durable branch_summary entry at the target, or an in-memory leaf
    // move (TS `branch()`; the next append chains onto the target, which is how the loader
    // reconstructs the branch after a restart too).
    let summary_entry = if let Some((text, details)) = &summary_result {
        let id = rt.append_branch_summary(
            new_leaf.as_deref(),
            text,
            Some(details.clone()),
            None,
        )?;
        Some(json!({
            "type": "branch_summary",
            "id": id,
            "parentId": new_leaf,
            "fromId": new_leaf.as_deref().unwrap_or("root"),
            "summary": text,
        }))
    } else {
        rt.branch_to(new_leaf.clone())?;
        None
    };

    // Rebuild the conversation the model sees from the new leaf.
    let fresh = entries(rt)?;
    let ctx = omp_compact::build_session_context(
        &fresh,
        match new_leaf.as_deref() {
            None => omp_compact::Leaf::Before,
            Some(id) => omp_compact::Leaf::Id(id),
        },
    )
    .map_err(|e| err(format!("navigate_tree: cannot rebuild context: {e}")))?;
    let messages = ctx
        .messages
        .iter()
        .filter_map(|m| serde_json::from_value::<pa_agent::types::AgentMessage>(m.clone()).ok())
        .collect::<Vec<_>>();
    rt.agent.abort();
    rt.agent.wait_for_idle().await;
    rt.agent.set_messages(messages).await;

    if let Some(host) = rt.ext_host().filter(|h| h.has_handlers("session_tree")) {
        let new_leaf_id = summary_entry
            .as_ref()
            .and_then(|e| e.get("id").and_then(Value::as_str).map(str::to_owned))
            .or_else(|| new_leaf.clone());
        if let Err(e) = host
            .emit_session_tree(
                new_leaf_id,
                old_leaf.clone(),
                summary_entry.clone(),
                summary_entry.is_some().then_some(from_extension),
                None,
            )
            .await
        {
            eprintln!("omt: extension session_tree: {e:#}");
        }
    }

    let mut data = json!({"cancelled": false});
    if target_is_user {
        data["editorText"] =
            omp_session::tree::user_message_text(target_entry).map_or(Value::Null, Value::String);
    }
    if let Some(entry) = summary_entry {
        data["summaryEntry"] = entry;
    }
    Ok(Some(data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::read_entries;

    /// The writer lands the summary parented at the navigation target and the leaf moves,
    /// so the next append chains onto the target instead of the old tail.
    #[test]
    fn branched_append_forms_sibling_branch() {
        let dir = tempfile::tempdir().unwrap();
        let stamp = "2026-10-12T00:00:00.000Z";
        let mut writer =
            omp_session::SessionWriter::create(dir.path(), "/tmp", "sess1", stamp).unwrap();
        let root = writer.append_entry("custom", json!({}), stamp).unwrap();
        let first = writer.append_entry("custom", json!({}), stamp).unwrap();
        let tail = writer.append_entry("custom", json!({}), stamp).unwrap();
        // /tree: rewind to the root, answer again.
        writer.branch_to(Some(root.clone())).unwrap();
        let forked = writer.append_entry("custom", json!({}), stamp).unwrap();
        let entries: Vec<Value> = read_entries(writer.path()).unwrap();
        let tree = omp_session::tree::build_tree(&entries);
        assert_eq!(tree.len(), 1);
        assert_eq!(
            tree[0].children.len(),
            2,
            "first and forked are siblings under the root"
        );
        assert_eq!(tree[0].children[1].entry["id"], json!(forked));
        let forked_entry = entries.iter().find(|e| e["id"] == json!(forked)).unwrap();
        assert_eq!(forked_entry["parentId"], json!(root));
        assert!(
            tree[0].children[0].children[0].entry["id"] == json!(tail)
                || tree[0].children[1].children[0].entry["id"] == json!(tail)
        );
        let _ = (first, tail);
    }

    #[test]
    fn branch_to_unknown_id_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = omp_session::SessionWriter::create(
            dir.path(),
            "/tmp",
            "sess1",
            "2026-10-12T00:00:00.000Z",
        )
        .unwrap();
        assert!(w.branch_to(Some("nope".to_owned())).is_err());
        // New root: parentId null.
        w.branch_to(None).unwrap();
        let id = w
            .append_entry("custom", json!({}), "2026-10-12T00:00:01.000Z")
            .unwrap();
        let entries = read_entries(w.path()).unwrap();
        let e = entries.iter().find(|e| e["id"] == json!(id)).unwrap();
        assert_eq!(e["parentId"], Value::Null);
    }

    /// TS `textOf`: plain string content, or the text parts joined by newlines, sliced to 80.
    fn digest_text(m: &Value) -> String {
        let text = match &m["content"] {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| {
                    (p.get("type")? == "text")
                        .then(|| p.get("text")?.as_str().map(str::to_owned))?
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        text.chars().take(80).collect()
    }

    /// `buildSessionContext` over the redacted branched fixture must match the TS digest
    /// (golden produced by running omp's TS under bun; fixture shared with omp-session).
    #[test]
    fn context_matches_ts_build_session_context() {
        let fixture = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../omp-session/tests/fixtures/branched-session.jsonl"
        );
        let golden: Value = serde_json::from_str(include_str!(
            "../../omp-session/tests/fixtures/branched-session.expectation.json"
        ))
        .unwrap();
        let entries: Vec<Value> = std::fs::read_to_string(fixture)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|e| {
                !matches!(
                    e.get("type").and_then(Value::as_str),
                    Some("title" | "session")
                )
            })
            .collect();
        let leaf = golden["leafId"].as_str().unwrap();
        let ctx =
            omp_compact::build_session_context(&entries, omp_compact::Leaf::Id(leaf)).unwrap();
        assert_eq!(ctx.mode, golden["mode"], "mode differs");
        assert_eq!(
            ctx.thinking_level.unwrap_or_default(),
            golden["thinkingLevel"],
            "thinking level differs"
        );
        assert_eq!(
            ctx.messages.len() as u64,
            golden["messageCount"].as_u64().unwrap(),
            "message count differs"
        );
        let digest: Vec<Value> = ctx
            .messages
            .iter()
            .map(|m| {
                if m.get("role").and_then(Value::as_str) == Some("branchSummary") {
                    json!({"role": "branchSummary", "summary": m.get("summary").cloned().unwrap_or(Value::Null)})
                } else {
                    json!({"role": m.get("role").cloned().unwrap_or(Value::Null), "text": digest_text(m)})
                }
            })
            .collect();
        assert_eq!(
            json!(digest),
            golden["messages"],
            "context differs from TS buildSessionContext()"
        );
    }

    #[test]
    fn branch_summary_appends_at_target() {
        let dir = tempfile::tempdir().unwrap();
        let stamp = "2026-10-12T00:00:00.000Z";
        let mut w = omp_session::SessionWriter::create(dir.path(), "/tmp", "sess1", stamp).unwrap();
        let a = w.append_entry("custom", json!({}), stamp).unwrap();
        let _b = w.append_entry("custom", json!({}), stamp).unwrap();
        let s = w
            .append_branch_summary(Some(&a), "the other branch explored X", None, None, stamp)
            .unwrap();
        let entries = read_entries(w.path()).unwrap();
        let s_entry = entries.iter().find(|e| e["id"] == json!(s)).unwrap();
        assert_eq!(s_entry["parentId"], json!(a));
        assert_eq!(s_entry["type"], json!("branch_summary"));
        assert_eq!(s_entry["fromId"], json!(a));
        // Leaf advanced onto the summary entry.
        let d = w.append_entry("custom", json!({}), stamp).unwrap();
        let entries = read_entries(w.path()).unwrap();
        let d_entry = entries.iter().find(|e| e["id"] == json!(d)).unwrap();
        assert_eq!(d_entry["parentId"], json!(s));
    }
}
