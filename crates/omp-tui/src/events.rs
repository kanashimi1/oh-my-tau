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
    /// `{"type":"prompt_result", "status": "completed"|"aborted"|"error",
    /// "error"?: {"message", ...}, ...}` (TS `RpcPromptResultFrame`) — outcome
    /// of a submitted prompt ticket.
    PromptResult {
        /// Echoed submit id, if any.
        id: Option<String>,
        /// `status != "error"` (an abort is not a failure).
        success: bool,
        /// `error.message` when the prompt failed.
        error: Option<String>,
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
    /// `{"type":"session_loaded", "messages": [...], "name": <string|null>}`
    /// — a session was resumed (startup `--resume` or `/resume`): the
    /// transcript is replaced by the loaded history and, when present, the
    /// title label becomes the restored session name. Emitted by the omp-cli
    /// TUI bridge.
    SessionLoaded {
        messages: Vec<Value>,
        name: Option<String>,
    },
    /// `{"type":"session_reset"}` — `/new` / `/clear` started a fresh session:
    /// the transcript and the terminal scrollback are dropped (TS
    /// `resetTranscript()` + `requestRender(true, { clearScrollback: true })`).
    SessionReset,
    /// `{"type":"context_usage", "model", "modelName", "tokens", "percent",
    /// "contextWindow", "thresholdPercent"?, "speculationPercent"?}` — the
    /// status line's model identity and context gauge inputs (omp-cli
    /// `Runtime::context_usage_frame`).
    ContextUsage(Value),
    /// `{"type":"models_changed","models":[…]}` — the `/model` picker
    /// catalogue changed (F5 online refresh updated `models.db`); the
    /// picker list is replaced so new models are visible immediately.
    ModelsChanged {
        models: Vec<crate::overlays::models::ModelEntry>,
    },
    /// `{"type":"session_info", "title"?, "lines": [...]}` — the `/session`
    /// report (omp-cli `session_info_frame`) or the `/usage` report
    /// (`title: "Usage"`), shown in the session info overlay.
    SessionInfo {
        /// Panel title; `None` = "Session Info".
        title: Option<String>,
        /// Report lines.
        lines: Vec<crate::overlays::session_info::SessionInfoLine>,
    },
    /// `{"type":"notice","level":"info"|"warning"|"error","message":...}` —
    /// a runtime notice (e.g. an ultra mode switching the context strategy).
    Notice {
        /// `info` / `warning` / `error`.
        level: String,
        /// Notice text.
        message: String,
    },
    /// `{"type":"clipboard_copy","text":...}` — `/copy` picked `text` for the
    /// system clipboard (omp-cli `slash_extra::dialog::copy_to_clipboard`).
    ClipboardCopy(String),
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
            success: frame.get("status").and_then(Value::as_str) != Some("error"),
            error: frame
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            frame: frame.clone(),
        },
        "session_settled" => AppEvent::SessionSettled,
        "queue_update" => AppEvent::QueueUpdate {
            steering: string_list(frame, "steering"),
            follow_up: string_list(frame, "followUp"),
        },
        "model_changed" | "model_change" => AppEvent::ModelChanged(frame.clone()),
        "context_usage" => AppEvent::ContextUsage(frame.clone()),
        "thinking_level_changed" | "thinking_level_change" => {
            AppEvent::ThinkingLevelChanged(frame.clone())
        }
        "auto_compaction_start" | "auto_compaction_end" => AppEvent::AutoCompaction(frame.clone()),
        "auto_retry_start"
        | "auto_retry_end"
        | "retry_fallback_applied"
        | "retry_fallback_succeeded" => AppEvent::Retry(frame.clone()),
        "available_commands_update" => AppEvent::AvailableCommands(frame.clone()),
        "notice" => AppEvent::Notice {
            level: frame
                .get("level")
                .and_then(Value::as_str)
                .unwrap_or("info")
                .to_owned(),
            message: frame
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        },
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
        "session_loaded" => AppEvent::SessionLoaded {
            messages: frame
                .get("messages")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            name: session_loaded_name(frame),
        },
        "session_reset" => AppEvent::SessionReset,
        "clipboard_copy" => AppEvent::ClipboardCopy(
            frame
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        ),
        "session_info" => AppEvent::SessionInfo {
            title: frame
                .get("title")
                .and_then(Value::as_str)
                .map(str::to_owned),
            lines: crate::overlays::session_info::SessionInfoLine::parse_frame(frame),
        },
        "models_changed" => AppEvent::ModelsChanged {
            models: frame
                .get("models")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(parse_model_entry).collect())
                .unwrap_or_default(),
        },
        _ => AppEvent::Unknown(frame.clone()),
    }
}

/// Like [`parse_frame`] but consumes the frame so a `session_loaded` history
/// (potentially the whole transcript) is moved into the event, not cloned.
/// Every other frame type parses exactly as [`parse_frame`] does.
#[must_use]
pub fn parse_frame_owned(mut frame: Value) -> AppEvent {
    if frame.get("type").and_then(Value::as_str) == Some("session_loaded") {
        let name = session_loaded_name(&frame);
        let messages = match frame.get_mut("messages").map(Value::take) {
            Some(Value::Array(items)) => items,
            _ => Vec::new(),
        };
        return AppEvent::SessionLoaded { messages, name };
    }
    parse_frame(&frame)
}

/// Optional restored session name of a `session_loaded` frame.
fn session_loaded_name(frame: &Value) -> Option<String> {
    frame
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .map(str::to_owned)
}

/// One `models_changed` entry → a picker row (inverse of the JSON the
/// omp-cli port emits after a refresh).
fn parse_model_entry(entry: &Value) -> Option<crate::overlays::models::ModelEntry> {
    use crate::overlays::models::ModelEntry;
    let mut model = ModelEntry::new(
        entry.get("provider").and_then(Value::as_str)?,
        entry.get("id").and_then(Value::as_str)?,
        entry.get("contextWindow").and_then(Value::as_u64),
    );
    model.name = entry
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    model.max_tokens = entry.get("maxTokens").and_then(Value::as_u64);
    model.cost_input = entry
        .get("costInput")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    model.cost_output = entry
        .get("costOutput")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    model.cost_cache_read = entry
        .get("costCacheRead")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    model.reasoning = entry
        .get("reasoning")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    model.vision = entry
        .get("vision")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    model.available = entry
        .get("available")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    model.kind = entry
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    model.priority = entry.get("priority").and_then(Value::as_i64);
    Some(model)
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
    fn notice_frames_become_notices() {
        assert_eq!(
            parse_frame(&json!({"type":"notice","level":"warning","message":"cache resets"})),
            AppEvent::Notice {
                level: "warning".into(),
                message: "cache resets".into()
            }
        );
    }

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
        match parse_frame(
            &json!({"type":"prompt_result","id":"t1","status":"completed","agentInvoked":true,"sessionSettled":true}),
        ) {
            AppEvent::PromptResult {
                id, success, error, ..
            } => {
                assert_eq!(id.as_deref(), Some("t1"));
                assert!(success);
                assert!(error.is_none());
            }
            other => panic!("unexpected: {other:?}"),
        }
        // TS wire shape: failure is `status: "error"` + `error.message`.
        match parse_frame(
            &json!({"type":"prompt_result","status":"error","agentInvoked":true,
                                  "error":{"message":"429 rate limited","retryable":true}}),
        ) {
            AppEvent::PromptResult { success, error, .. } => {
                assert!(!success);
                assert_eq!(error.as_deref(), Some("429 rate limited"));
            }
            other => panic!("unexpected: {other:?}"),
        }
        // An abort is not a failure.
        match parse_frame(&json!({"type":"prompt_result","status":"aborted","agentInvoked":true})) {
            AppEvent::PromptResult { success, .. } => assert!(success),
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
