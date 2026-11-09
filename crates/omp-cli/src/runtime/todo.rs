//! Todo tool bridge and the reminder that nudges unfinished phases.

use anyhow::Result;
use pa_agent::types::AgentMessage;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

use super::prompt::is_awaiting_user_answer;
use super::session::seed_resumed_message;
use super::{lock, message_text, Runtime};

impl Runtime {
    /// TS `TodoTracker` state leg: a successful state-changing `todo` call is
    /// the canonical snapshot (`details.phases`). Track it for the settle-time
    /// completion reminder and emit `todo_auto_clear` when nothing is left open
    /// (the TUI drops the plan on that event).
    pub(super) fn on_todo_settled(&self, tr: &pa_agent::types::ToolResultMessage) {
        let Some(phases) = tr
            .details
            .as_ref()
            .and_then(|d| d.get("phases"))
            .filter(|p| p.is_array())
        else {
            return;
        };
        lock(&self.settings).todo_phases = phases.clone();
        // Any progress clears the reminder-awaiting flag (TS todo touch).
        lock(&self.prompts).todo_awaiting_progress = false;
        // A read-only `view` echoes the current state: refresh the plan, but
        // only a state-changing call that leaves nothing open auto-clears it.
        let read_only = tr
            .details
            .as_ref()
            .and_then(|d| d.get("op"))
            .and_then(Value::as_str)
            .is_some_and(|op| op == "view");
        if !read_only && !has_incomplete_tasks(phases) {
            self.emit_event(crate::events::todo_auto_clear_frame());
        }
    }

    /// TS `TodoTracker.checkCompletion`: a successful terminal turn that stops
    /// with incomplete todos emits `todo_reminder`, appends the reminder as a
    /// hidden developer message, and continues the run — the `agent_end` goes
    /// out non-terminal (`isTerminal: false`, `yielded: false`, a plain
    /// `willContinue` settle, not `awaitingAsyncWork`). Returns `true` when the
    /// reminder took over the settle.
    pub(super) fn maybe_todo_reminder(
        self: &Arc<Self>,
        messages: &[AgentMessage],
        mut frame: Value,
    ) -> bool {
        let (enabled, max_attempts, phases) = {
            let st = lock(&self.settings);
            (
                st.todo_reminders_enabled,
                st.todo_reminders_max,
                st.todo_phases.clone(),
            )
        };
        let mut p = lock(&self.prompts);
        if p.todo_awaiting_progress {
            return false;
        }
        if !enabled {
            p.todo_reminders = 0;
            return false;
        }
        if p.todo_reminders >= max_attempts {
            return false;
        }
        let incomplete = incomplete_todos(&phases);
        if incomplete.as_array().is_none_or(Vec::is_empty) {
            p.todo_reminders = 0;
            p.todo_awaiting_progress = false;
            return false;
        }
        let awaiting_user = messages
            .iter()
            .rev()
            .find(|m| m.role() == "assistant")
            .is_some_and(|m| is_awaiting_user_answer(&message_text(m)));
        if awaiting_user {
            return false;
        }
        p.todo_reminders += 1;
        p.todo_awaiting_progress = true;
        let attempt = p.todo_reminders;
        drop(p);
        self.emit_event(crate::events::todo_reminder_frame(
            &incomplete,
            attempt,
            max_attempts,
        ));
        let list = todo_reminder_list(&phases);
        let count = incomplete.as_array().map_or(0, Vec::len);
        let reminder = format!(
            "<system-reminder>\nYou stopped with {count} incomplete todo item(s):\n{list}\n\nPlease continue working on these tasks or mark them complete if finished.\n(Reminder {attempt}/{max_attempts})\n</system-reminder>"
        );
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
        let reminder = AgentMessage::Custom(pa_agent::types::CustomAgentMessage {
            role: "developer".to_string(),
            payload: json!({
                "customType": "todo-reminder",
                "content": [{"type": "text", "text": reminder}],
                "display": false,
                "attribution": "agent",
                "timestamp": timestamp,
            }),
        });
        frame["isTerminal"] = json!(false);
        frame["yielded"] = json!(false);
        self.emit_event(frame.clone());
        let rt = self.clone();
        let messages = messages.to_vec();
        tokio::spawn(async move {
            // TS `agent.appendMessage` + `sessionManager.appendMessage`, before
            // the scheduled continue (`scheduleAgentContinue`).
            rt.agent.append_messages(vec![reminder.clone()]).await;
            // `json_round_trip` to pa_types returns None for custom roles like
            // "developer"; `seed_resumed_message` persists the verbatim shape.
            seed_resumed_message(&rt.persist, &reminder, None);
            if let Err(e) = rt.continue_after_retry().await {
                eprintln!("omt: todo reminder continuation failed: {e:#}");
                rt.settle_terminal(frame, &messages, None);
            }
        });
        true
    }

    /// Rebuild the `todo` tool from session entries (`&[]` = fresh session),
    /// matching omp's resume rehydration.
    pub async fn rehydrate_todo_tool(&self, entries: &[Value]) {
        let session_file = self.persist.file().map(|p| p.display().to_string());
        self.swap_todo_tool(Arc::new(crate::tools::TodoTool::with_session(
            crate::tools::phases_from_entries(entries),
            session_file.as_deref(),
        )))
        .await;
    }

    /// Install `todo` as the agent's todo tool (no-op when the tool list has none).
    pub(super) async fn swap_todo_tool(&self, todo: Arc<dyn pa_agent::types::AgentTool>) {
        let mut tools = self.agent.tools().await;
        if let Some(slot) = tools.iter().position(|t| t.name() == "todo") {
            tools[slot] = todo;
            self.agent.set_tools(tools).await;
        }
    }
}

/// Latest todo phases: scan the branch newest-first, parsing lines only until
/// the latest state-changing todo entry — typically the tail, so almost nothing.
pub(super) fn latest_todo_phases(
    index: &omp_session::index::SessionIndex,
    path: &Path,
    branch: &[u32],
) -> Result<Vec<crate::tools::TodoPhase>> {
    let mut phases: Option<Vec<crate::tools::TodoPhase>> = None;
    // Only message/custom lines can carry todo state: other kinds are never read.
    let rev: Vec<u32> = branch
        .iter()
        .rev()
        .copied()
        .filter(|&i| {
            matches!(
                index.refs[i as usize].kind,
                omp_session::index::EntryKind::Message | omp_session::index::EntryKind::Custom
            )
        })
        .collect();
    index.scan_lines(path, &rev, |_r, raw| {
        // Cheap pre-filter: a state-changing todo entry mentions the todo tool
        // or the edit custom type; everything else is skipped without a DOM.
        if raw.contains("\"toolName\":\"todo\"")
            || raw.contains("\"customType\":\"user_todo_edit\"")
        {
            if let Ok(v) = serde_json::from_str::<Value>(raw) {
                let p = crate::tools::phases_from_entries(std::slice::from_ref(&v));
                if !p.is_empty() {
                    phases = Some(p);
                    // Newest state wins: nothing older can change the answer.
                    return Ok(std::ops::ControlFlow::Break(()));
                }
            }
        }
        Ok(std::ops::ControlFlow::Continue(()))
    })?;
    Ok(phases.unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Todo completion reminder helpers (TS `todo-tracker.ts`).
// ---------------------------------------------------------------------------

/// `todo.remindersMax` as a `u32` (TS default 3); a negative or non-finite
/// configured value falls back to the default.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // range-checked: `0.0 <= n < u32::MAX` before the cast.
pub(super) fn reminders_max(cfg: &omp_settings::Settings) -> u32 {
    match cfg.get_number("todo.remindersMax") {
        Some(n) if n.is_finite() && n >= 0.0 => {
            let rounded = n.round();
            if rounded >= f64::from(u32::MAX) {
                u32::MAX
            } else {
                rounded as u32
            }
        }
        _ => 3,
    }
}

/// The `- Phase\n  - task` block inside the reminder text (TS `checkCompletion`'s
/// `todoList`): one heading per phase that still has open tasks.
fn todo_reminder_list(phases: &Value) -> String {
    phases
        .as_array()
        .map(|ps| {
            ps.iter()
                .filter_map(|phase| {
                    let name = phase.get("name").and_then(Value::as_str)?;
                    let open: Vec<String> = phase
                        .get("tasks")
                        .and_then(Value::as_array)
                        .map(|tasks| {
                            tasks
                                .iter()
                                .filter(|t| {
                                    t.get("status")
                                        .and_then(Value::as_str)
                                        .is_some_and(|s| s == "pending" || s == "in_progress")
                                })
                                .filter_map(|t| {
                                    t.get("content").and_then(Value::as_str).map(str::to_owned)
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    (!open.is_empty()).then(|| {
                        let tasks = open
                            .iter()
                            .map(|c| format!("  - {c}"))
                            .collect::<Vec<_>>()
                            .join("\n");
                        format!("- {name}\n{tasks}")
                    })
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// Open tasks (`status` `pending`/`in_progress`) across all todo phases, as the
/// flat `[{content, status}]` list the `todo_reminder` frame carries (TS
/// `checkCompletion`'s `incomplete`).
fn incomplete_todos(phases: &Value) -> Value {
    let todos: Vec<Value> = phases
        .as_array()
        .map(|ps| {
            ps.iter()
                .filter_map(|p| p.get("tasks"))
                .filter_map(Value::as_array)
                .flatten()
                .filter(|t| {
                    t.get("status")
                        .and_then(Value::as_str)
                        .is_some_and(|s| s == "pending" || s == "in_progress")
                })
                .map(|t| {
                    json!({
                        "content": t.get("content").cloned().unwrap_or(Value::Null),
                        "status": t.get("status").cloned().unwrap_or(Value::Null),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    json!(todos)
}

/// Whether any phase has an open task (`pending`/`in_progress`).
fn has_incomplete_tasks(phases: &Value) -> bool {
    phases.as_array().is_some_and(|ps| {
        ps.iter()
            .filter_map(|p| p.get("tasks"))
            .filter_map(Value::as_array)
            .flatten()
            .any(|t| {
                t.get("status")
                    .and_then(Value::as_str)
                    .is_some_and(|s| s == "pending" || s == "in_progress")
            })
    })
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod todo_reminder_tests {
    use super::*;
    use serde_json::json;

    fn phases(open: &[(&str, &str)], done: &[(&str, &str)]) -> Value {
        let task = |(content, status): &(&str, &str)| json!({"content": content, "status": status});
        json!([{
            "name": "Tasks",
            "tasks": open.iter().map(task).chain(done.iter().map(task)).collect::<Vec<_>>(),
        }])
    }

    #[test]
    fn incomplete_todos_flattens_open_tasks() {
        let v = phases(
            &[("a", "pending"), ("b", "in_progress")],
            &[("c", "completed")],
        );
        let inc = incomplete_todos(&v);
        assert_eq!(
            inc,
            json!([
                {"content": "a", "status": "pending"},
                {"content": "b", "status": "in_progress"},
            ])
        );
        assert!(has_incomplete_tasks(&v));
        let all_done = phases(&[], &[("a", "completed")]);
        assert_eq!(incomplete_todos(&all_done), json!([]));
        assert!(!has_incomplete_tasks(&all_done));
        assert!(!has_incomplete_tasks(&json!([])));
    }

    #[test]
    fn awaiting_user_answer_question_lines() {
        assert!(is_awaiting_user_answer("Which approach should I take?"));
        assert!(is_awaiting_user_answer(
            "Done so far.\nDo you want me to continue?"
        ));
        assert!(!is_awaiting_user_answer("All tasks are complete."));
        assert!(!is_awaiting_user_answer(
            "Refactored the parser and updated tests."
        ));
    }

    #[test]
    fn awaiting_user_answer_response_cues_and_labels() {
        assert!(is_awaiting_user_answer("Please confirm the approach."));
        assert!(is_awaiting_user_answer("Let me know if that works."));
        assert!(is_awaiting_user_answer("Q: what is your pick?"));
        // TS `isQuestionPromptLine`: any trailing question starting with an
        // interrogative counts, even a rhetorical one about the code.
        assert!(is_awaiting_user_answer(
            "Why does the parser reject this input?"
        ));
    }

    #[test]
    fn notice_and_stream_frames_match_ts_shape() {
        let n =
            crate::events::notice_frame("warning", "disk almost full", Some("session-persistence"));
        assert_eq!(n["type"], json!("notice"));
        assert_eq!(n["level"], json!("warning"));
        assert_eq!(n["source"], json!("session-persistence"));
        let n = crate::events::notice_frame("info", "ok", None);
        assert!(n.get("source").is_none());

        let t = crate::events::tool_stream_update_frame("call-1", "edit", &json!({"kind": "diff"}));
        assert_eq!(t["type"], json!("tool_stream_update"));
        assert_eq!(t["toolCallId"], json!("call-1"));
        assert_eq!(t["toolName"], json!("edit"));
        assert_eq!(t["update"]["kind"], json!("diff"));

        let r = crate::events::todo_reminder_frame(
            &json!([{"content": "a", "status": "pending"}]),
            1,
            3,
        );
        assert_eq!(r["type"], json!("todo_reminder"));
        assert_eq!(r["attempt"], json!(1));
        assert_eq!(r["maxAttempts"], json!(3));
        assert_eq!(r["todos"][0]["content"], json!("a"));
        assert_eq!(
            crate::events::todo_auto_clear_frame()["type"],
            json!("todo_auto_clear")
        );
    }
}
