//! Extension UI surface — ports of the extension-host UI hooks in
//! `pi-coding-agent/src/modes/interactive-mode.ts` (`showExtensionUI`,
//! `select`, ask dialogs) consuming `extension_ui_request` frames.
//!
//! Stub: tracks pending requests and answers them via
//! [`Action::ExtensionUiResponse`](crate::action::Action::ExtensionUiResponse).
//! Planned: modal rendering of extension-provided dialogs (TS renders them
//! through the same overlay chrome; the interactive mode registers an
//! `ExtensionUIContext`).

use serde_json::Value;

/// One pending extension dialog.
#[derive(Debug, Clone)]
pub struct PendingRequest {
    /// Frame id to answer with.
    pub id: String,
    /// Dialog method (`ask`, `select`, `editor`, ...).
    pub method: String,
    /// Raw request payload.
    pub frame: Value,
}

/// Pending extension dialogs, in arrival order.
#[derive(Debug, Clone, Default)]
pub struct ExtensionUi {
    pending: Vec<PendingRequest>,
}

impl ExtensionUi {
    /// An empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a request from an `extension_ui_request` frame.
    pub fn push(&mut self, frame: &Value) {
        let id = frame.get("id").and_then(Value::as_str).unwrap_or("").to_owned();
        let method = frame
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        self.pending.push(PendingRequest {
            id,
            method,
            frame: frame.clone(),
        });
    }

    /// The oldest unanswered request, if any.
    #[must_use]
    pub fn head(&self) -> Option<&PendingRequest> {
        self.pending.first()
    }

    /// Remove the request with `id` (after answering it).
    pub fn complete(&mut self, id: &str) {
        self.pending.retain(|r| r.id != id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn queue_is_fifo() {
        let mut ui = ExtensionUi::new();
        ui.push(&json!({"id":"1","method":"ask"}));
        ui.push(&json!({"id":"2","method":"select"}));
        assert_eq!(ui.head().map(|r| r.id.as_str()), Some("1"));
        ui.complete("1");
        assert_eq!(ui.head().map(|r| r.id.as_str()), Some("2"));
    }
}
