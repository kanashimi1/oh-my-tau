//! Chat area: the scrollback of user/assistant/tool messages — ports of
//! `pi-tui/src/chat/` (`chat-container.ts`, `chat-messages.ts` and friends).
//!
//! [`Messages`] is the session-backed model: it consumes [`AppEvent`]s and
//! keeps one [`ChatEntry`] per logical message. Rendering turns entries into
//! blocks via the submodules: [`markdown`] for assistant text,
//! [`thinking`] for reasoning blocks, [`code`] for fenced code highlighting.

pub mod blocks;
pub mod code;
pub mod local_time;
pub mod markdown;
pub mod messages;
pub mod rail;
pub mod reveal;
pub mod thinking;

pub use messages::{
    AssistantMsg, AssistantSegment, ChatEntry, ChatKind, ChatRole, Compaction, Custom,
    DisplayPrefs, ExecStatus, Execution, Messages, ReadGroup, ReadItem, ToolCard, ToolStatus,
    Usage,
};
