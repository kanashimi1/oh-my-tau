//! The chat message model — consumes runtime frames and keeps render-ready
//! entries. Port of `pi-tui/src/chat/chat-messages.ts`.
//!
//! MVP: entries are (role, text) pairs; streaming assistant text deltas
//! append to the newest entry. Tool calls, subagent trees, rewinds and
//! compaction markers arrive as dedicated `ChatEntry` kinds once the tool
//! renderer modules land.

use serde_json::Value;

use crate::events::AppEvent;

/// Who produced an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    /// A user prompt.
    User,
    /// The assistant.
    Assistant,
    /// A tool call or result summary.
    Tool,
    /// A system/runtime notice (compaction, retry, errors).
    System,
}

/// One logical chat entry.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatEntry {
    /// Producer.
    pub role: ChatRole,
    /// Accumulated display text.
    pub text: String,
}

impl ChatEntry {
    /// Append streamed text to this entry.
    pub fn push_str(&mut self, s: &str) {
        self.text.push_str(s);
    }
}

/// The ordered chat model.
#[derive(Debug, Clone, Default)]
pub struct Messages {
    entries: Vec<ChatEntry>,
    /// Whether the newest entry is still receiving deltas.
    streaming: bool,
}

impl Messages {
    /// An empty chat.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// All entries (render only a window of these).
    #[must_use]
    pub fn entries(&self) -> &[ChatEntry] {
        &self.entries
    }

    /// True while a `message_update` stream is open.
    #[must_use]
    pub fn is_streaming(&self) -> bool {
        self.streaming
    }

    /// Apply one runtime event.
    pub fn apply(&mut self, event: &AppEvent) {
        match event {
            AppEvent::AgentStart | AppEvent::TurnStart => self.streaming = true,
            AppEvent::AgentEnd(_) | AppEvent::TurnEnd(_) | AppEvent::SessionSettled => {
                self.streaming = false;
            }
            AppEvent::MessageUpdate(frame) => self.apply_message_update(frame),
            AppEvent::MessageEnd(frame) => {
                self.apply_message_update(frame);
                self.streaming = false;
            }
            AppEvent::PromptResult { success, frame, .. } => {
                if !success {
                    let msg = frame
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("prompt failed");
                    self.push(ChatRole::System, format!("error: {msg}"));
                }
            }
            AppEvent::Retry(frame) => {
                let kind = frame.get("type").and_then(Value::as_str).unwrap_or("retry");
                self.push(ChatRole::System, format!("[{kind}]"));
            }
            AppEvent::AutoCompaction(frame) => {
                let kind = frame.get("type").and_then(Value::as_str).unwrap_or("compaction");
                self.push(ChatRole::System, format!("[{kind}]"));
            }
            _ => {}
        }
    }

    /// Record a locally-submitted user prompt (echoed immediately).
    pub fn push_user(&mut self, text: &str) {
        self.push(ChatRole::User, text.to_owned());
    }

    fn push(&mut self, role: ChatRole, text: String) {
        self.entries.push(ChatEntry { role, text });
    }

    /// Merge an assistant message payload (TS wire shape: `content[]` with
    /// `text` parts) into the newest assistant entry, or start one.
    fn apply_message_update(&mut self, frame: &Value) {
        let Some(message) = frame.get("message") else {
            return;
        };
        let text: String = message
            .get("content")
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();
        if text.is_empty() {
            return;
        }
        match self.entries.last_mut() {
            Some(last) if last.role == ChatRole::Assistant && self.streaming => {
                last.text = text;
            }
            _ => self.push(ChatRole::Assistant, text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn streams_assistant_text_into_one_entry() {
        let mut m = Messages::new();
        m.apply(&AppEvent::AgentStart);
        let delta = |t: &str| {
            AppEvent::MessageUpdate(json!({"type":"message_update","message":{"role":"assistant","content":[{"type":"text","text":t}]}}))
        };
        m.apply(&delta("hel"));
        m.apply(&delta("hello"));
        assert_eq!(m.entries().len(), 1);
        assert_eq!(m.entries()[0].text, "hello");
        m.apply(&AppEvent::MessageEnd(json!({"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"hello"}]}})));
        assert!(!m.is_streaming());
    }

    #[test]
    fn user_prompt_echoes_as_own_entry() {
        let mut m = Messages::new();
        m.push_user("do a thing");
        m.apply(&AppEvent::AgentStart);
        assert_eq!(m.entries().len(), 1);
        assert_eq!(m.entries()[0].role, ChatRole::User);
    }
}
