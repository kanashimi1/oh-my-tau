//! Anthropic Messages history normalization and stop-reason mapping (the body itself is built in
//! [`super::request`]).

use crate::types::StopReason;

pub(crate) fn normalize_tool_call_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

pub(crate) fn map_stop_reason(reason: &str) -> Result<StopReason, String> {
    match reason {
        "end_turn" | "pause_turn" | "stop_sequence" => Ok(StopReason::Stop),
        "max_tokens" => Ok(StopReason::Length),
        "tool_use" => Ok(StopReason::ToolUse),
        "refusal" | "sensitive" => Ok(StopReason::Error),
        other => Err(format!("Unhandled stop reason: {other}")),
    }
}
