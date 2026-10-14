//! Typed view of the omp-cli `Runtime` event stream.
//!
//! The runtime forwards frames as JSON values shaped like
//! `{"type": "...", ...}` (see `crates/omp-cli/src/events.rs` `Framer` and
//! `runtime.rs` `on_event`). [`parse_frame`] classifies a frame into
//! [`AppEvent`]; payloads the UI must inspect (messages, tool calls) stay as
//! `serde_json::Value` in this crate so `omp-tui` does not depend on
//! `pa-agent` types. Widgets that need structured access parse the payload
//! fields documented per variant.
//!
//! TS references: `pi-coding-agent/src/modes/interactive-mode.ts` consumes
//! the same stream from `AgentSession`; `pi-tui/src/tui.ts` `NativeUiEvent`.

use serde_json::Value;

/// A classified runtime frame.
#[derive(Debug, Clone, PartialEq)]
pub enum AppEvent {
    /// `{"type":"agent_start"}` — a turn began.
    AgentStart,
    /// `{"type":"agent_end", ...}` — a turn ended (may carry `willContinue`).
    AgentEnd(Value),
    /// `{"type":"turn_start"}` — first turn of a prompt batch.
    TurnStart,
    /// `{"type":"turn_end"}` — all turns of the batch settled.
    TurnEnd(Value),
    /// `{"type":"message_start", "partial": ...}` — a message began streaming.
    MessageStart(Value),
    /// `{"type":"message_update", "message": <assistant partial>, ...}` —
    /// streaming delta; `message.content[].text` holds the accumulated text.
    MessageUpdate(Value),
    /// `{"type":"message_end", "message": {...}}` — a message settled.
    MessageEnd(Value),
    /// `{"type":"tool_execution_start", ...}`.
    ToolExecutionStart(Value),
    /// `{"type":"tool_execution_update", ...}`.
    ToolExecutionUpdate(Value),
    /// `{"type":"tool_execution_end", ...}`.
    ToolExecutionEnd(Value),
    /// `{"type":"prompt_result", "success": bool, ...}` — outcome of a
    /// submitted prompt ticket.
    PromptResult {
        /// Echoed submit id, if any.
        id: Option<String>,
        /// Whether the turn succeeded.
        success: bool,
        /// Raw frame payload.
        frame: Value,
    },
    /// `{"type":"session_settled"}` — queue drained, runtime idle.
    SessionSettled,
    /// `{"type":"queue_update", "steering": [...], "followUp": [...]}`.
    QueueUpdate {
        /// Steering queue previews.
        steering: Vec<String>,
        /// Follow-up queue previews.
        follow_up: Vec<String>,
    },
    /// `{"type":"model_changed", ...}`.
    ModelChanged(Value),
    /// `{"type":"thinking_level_changed", ...}`.
    ThinkingLevelChanged(Value),
    /// `{"type":"auto_compaction_start"|"auto_compaction_end", ...}`.
    AutoCompaction(Value),
    /// `{"type":"auto_retry_start"|"auto_retry_end"|"retry_fallback_applied"|"retry_fallback_succeeded", ...}`.
    Retry(Value),
    /// `{"type":"available_commands_update", "commands": [...]}`.
    AvailableCommands(Value),
    /// `{"type":"command_output", "text": "..."}` — slash-command output
    /// (omp-cli `slash::output`), shown as a system entry.
    CommandOutput(String),
    /// `{"type":"extension_ui_request", "id", "method", ...}` — an extension
    /// `ask` dialog the UI must answer with `Action::ExtensionUiResponse`.
    ExtensionUiRequest(Value),
    /// `{"type":"session_name_changed", "name": "..."}` — the session was
    /// renamed (user `/rename` or auto-title). Emitted by the omp-cli TUI
    /// bridge via the runtime's name listener; also appears as `sessionName`
    /// inside periodic `state` frames.
    SessionNameChanged(String),
    /// `{"type":"extension_ui_response", "id", ...}` — the answer to a
    /// pending extension dialog (echoed back through the runtime stream so
    /// the UI can retire its queue entry).
    ExtensionUiResponse(Value),
    /// Any frame without a dedicated variant; kept verbatim.
    Unknown(Value),
}

/// Classify one raw runtime frame.
#[must_use]
pub fn parse_frame(frame: &Value) -> AppEvent {
    let kind = frame.get("type").and_then(Value::as_str).unwrap_or("");
    match kind {
        "agent_start" => AppEvent::AgentStart,
        "agent_end" => AppEvent::AgentEnd(frame.clone()),
        "turn_start" => AppEvent::TurnStart,
        "turn_end" => AppEvent::TurnEnd(frame.clone()),
        "message_start" => AppEvent::MessageStart(frame.clone()),
        "message_update" => AppEvent::MessageUpdate(frame.clone()),
        "message_end" => AppEvent::MessageEnd(frame.clone()),
        "tool_execution_start" => AppEvent::ToolExecutionStart(frame.clone()),
        "tool_execution_update" => AppEvent::ToolExecutionUpdate(frame.clone()),
        "tool_execution_end" => AppEvent::ToolExecutionEnd(frame.clone()),
        "prompt_result" => AppEvent::PromptResult {
            id: frame
                .get("id")
                .and_then(Value::as_str)
                .map(std::string::ToString::to_string),
            success: frame
                .get("success")
                .is_some_and(|v| v == &Value::from(true)),
            frame: frame.clone(),
        },
        "session_settled" => AppEvent::SessionSettled,
        "queue_update" => AppEvent::QueueUpdate {
            steering: string_list(frame, "steering"),
            follow_up: string_list(frame, "followUp"),
        },
        "model_changed" | "model_change" => AppEvent::ModelChanged(frame.clone()),
        "thinking_level_changed" | "thinking_level_change" => {
            AppEvent::ThinkingLevelChanged(frame.clone())
        }
        "auto_compaction_start" | "auto_compaction_end" => AppEvent::AutoCompaction(frame.clone()),
        "auto_retry_start"
        | "auto_retry_end"
        | "retry_fallback_applied"
        | "retry_fallback_succeeded" => AppEvent::Retry(frame.clone()),
        "available_commands_update" => AppEvent::AvailableCommands(frame.clone()),
        "command_output" => AppEvent::CommandOutput(
            frame
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        ),
        "extension_ui_request" => AppEvent::ExtensionUiRequest(frame.clone()),
        "extension_ui_response" => AppEvent::ExtensionUiResponse(frame.clone()),
        "session_name_changed" => AppEvent::SessionNameChanged(
            frame
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        ),
        _ => AppEvent::Unknown(frame.clone()),
    }
}

/// Extract a `Vec<String>` field, tolerating missing/non-string entries.
fn string_list(frame: &Value, key: &str) -> Vec<String> {
    frame
        .get(key)
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(std::string::ToString::to_string))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classifies_core_frames() {
        assert_eq!(
            parse_frame(&json!({"type":"agent_start"})),
            AppEvent::AgentStart
        );
        assert_eq!(
            parse_frame(&json!({"type":"session_settled"})),
            AppEvent::SessionSettled
        );
        match parse_frame(&json!({"type":"prompt_result","id":"t1","success":true})) {
            AppEvent::PromptResult { id, success, .. } => {
                assert_eq!(id.as_deref(), Some("t1"));
                assert!(success);
            }
            other => panic!("unexpected: {other:?}"),
        }
        match parse_frame(&json!({"type":"queue_update","steering":["a"],"followUp":["b","c"]})) {
            AppEvent::QueueUpdate {
                steering,
                follow_up,
            } => {
                assert_eq!(steering, ["a"]);
                assert_eq!(follow_up, ["b", "c"]);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn unknown_frames_are_kept_verbatim() {
        let frame = json!({"type":"mystery","x":1});
        match parse_frame(&frame) {
            AppEvent::Unknown(v) => assert_eq!(v, frame),
            other => panic!("unexpected: {other:?}"),
        }
    }
}
