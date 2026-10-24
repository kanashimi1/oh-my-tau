//! The chat transcript model — consumes runtime frames and keeps render-ready
//! entries. Port of the data half of `pi-tui/src/chat/chat-transcript-builder.ts`
//! (rendering lives in `markdown`/`thinking` and `crate::tools`).
//!
//! Responsibilities (spec `chat.md` §1–2, §9):
//! - every message role becomes an ordered [`ChatEntry`]: user bubble,
//!   assistant segments, tool cards, read groups, bash/python execution,
//!   compaction/branch dividers, custom messages, usage rows, notices;
//! - pending tool calls are placed in stream order and matched to their
//!   results by `toolCallId` (`pending` map, builder §2);
//! - consecutive `read` calls fold into one [`ReadGroup`] (§7.1);
//! - [`Messages::finished_len`] decides the committed (scrollback) prefix vs
//!   the still-live tail — only leading finished entries may retire;
//! - ctrl+O transcript-wide expansion ([`Messages::toggle_expanded`], §9).

use std::cell::RefCell;
use std::collections::HashMap;

use ratatui::text::Line;
use serde_json::Value;

use crate::chat::blocks;
use crate::chat::reveal::{RevealBlock, StreamingRevealController};
use crate::chat::thinking::{format_thinking_for_display, ThinkingDisplay};
use crate::events::AppEvent;
use crate::theme::Theme;

/// Who produced an entry (coarse bucket; renderers take details from
/// [`ChatEntry::kind`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    /// A user prompt.
    User,
    /// The assistant.
    Assistant,
    /// A tool card, read group or bash/python execution.
    Tool,
    /// A system/runtime notice (compaction, branch, custom, usage, errors).
    System,
}

/// One block of an assistant message, in stream order.
#[derive(Debug, Clone, PartialEq)]
pub enum AssistantSegment {
    /// Markdown text.
    Text {
        /// Accumulated text.
        text: String,
    },
    /// Reasoning text (rendered collapsed by default, §5).
    Thinking {
        /// Accumulated thinking.
        text: String,
    },
}

impl AssistantSegment {
    /// The segment's text.
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Text { text } | Self::Thinking { text } => text,
        }
    }
}

/// Terminal state of a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStatus {
    /// Card created, no result yet.
    Pending,
    /// Streaming/partial result received.
    Running,
    /// Finished successfully.
    Ok,
    /// Finished with an error.
    Error,
}

impl ToolStatus {
    /// Whether the call has settled (result or error received).
    #[must_use]
    pub fn settled(self) -> bool {
        matches!(self, Self::Ok | Self::Error)
    }
}

/// A pending tool card (builder `#pendingTools`, §2). Results land here.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCard {
    /// `toolCallId`.
    pub call_id: String,
    /// Tool name.
    pub name: String,
    /// Accumulated arguments object (may be partial while streaming).
    pub args: Value,
    /// Settlement state.
    pub status: ToolStatus,
    /// Result text (partial or final).
    pub result_text: Option<String>,
    /// Settled result `details` (`Null` until the result arrived).
    pub details: Value,
    /// True until the final result arrived.
    pub partial: bool,
}

/// One read inside a [`ReadGroup`].
#[derive(Debug, Clone, PartialEq)]
pub struct ReadItem {
    /// `toolCallId`.
    pub call_id: String,
    /// Canonical path (`path`, legacy alias `file_path`).
    pub path: String,
    /// Settlement state.
    pub status: ToolStatus,
    /// Optional content preview.
    pub preview: Option<String>,
    /// Tool name: `read`, or a search tool folded in by the clean style
    /// (`grep`/`search`/`glob`/`find`; `path` then holds its pattern).
    pub tool: String,
    /// Call arguments (search tools: re-rendered for the row summary).
    pub args: Value,
    /// Result `details` once settled.
    pub details: Value,
}

/// Search tools the clean style folds into the exploration group.
pub(crate) fn is_explore_tool(name: &str) -> bool {
    matches!(name, "grep" | "search" | "glob" | "find")
}

/// Consecutive filesystem reads folded into one compact group (§7.1).
#[derive(Debug, Clone, PartialEq)]
pub struct ReadGroup {
    /// Reads in call order.
    pub reads: Vec<ReadItem>,
    /// Per-request usage rows attached beneath the group (TS
    /// `ReadToolGroupComponent.attachUsage`): `(anchor toolCallId, usage)`,
    /// the anchor being the request's last read in this group.
    pub usages: Vec<(String, Usage)>,
    /// False once a non-read entry displaced the group (no more folding).
    pub open: bool,
    /// `read.toolResultPreview` at creation: content previews instead of
    /// the compact summary.
    pub preview: bool,
}

/// State of a `!`/`$` execution block (§7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecStatus {
    /// Still running.
    Running,
    /// Exited 0.
    Complete,
    /// Cancelled by the user.
    Cancelled,
    /// Non-zero exit.
    Error,
}

/// A bash (`!`) or python (`$`) execution entry.
#[derive(Debug, Clone, PartialEq)]
pub struct Execution {
    /// `toolCallId` (empty for persisted transcript entries).
    pub call_id: String,
    /// The command/source line.
    pub command: String,
    /// Accumulated output.
    pub output: String,
    /// Settlement state.
    pub status: ExecStatus,
    /// Exit code when finished.
    pub exit_code: Option<i64>,
    /// `!!`/`$$` — excluded from context (renders dim).
    pub excluded: bool,
    /// Settled tool result `details` (bash: `wallTimeMs`, `timeoutSeconds`, …);
    /// `Null` for user `!` executions and streaming entries.
    pub details: Value,
}

/// Token usage tallies (assistant message `usage` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    /// Input tokens.
    pub input: u64,
    /// Output tokens.
    pub output: u64,
    /// Cache-read tokens.
    pub cache_read: u64,
    /// Cache-write tokens.
    pub cache_write: u64,
    /// Provider-reported request duration (ms) — drives the tok/s rate.
    pub duration_ms: Option<u64>,
    /// Time to first token (ms), when the provider reports it.
    pub ttft_ms: Option<u64>,
    /// The assistant message's `timestamp` (epoch ms): the row's leading
    /// local wall-clock stamp.
    pub timestamp_ms: Option<i64>,
    /// Prompt→yield wall time (`display.showTurnTime`, TS `turnElapsedMs`).
    pub turn_elapsed_ms: Option<u64>,
}

/// Transcript display settings the model honours (TS `display.*` /
/// `read.*` settings read by the event controller).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DisplayPrefs {
    /// `display.showTokenUsage` (default false): a usage row per billed
    /// assistant message.
    pub show_token_usage: bool,
    /// `display.showTurnTime` (default false): `Δ <elapsed>` in usage rows.
    pub show_turn_time: bool,
    /// `read.toolResultPreview` (default false): read groups show content
    /// previews instead of the compact `Read <path>` summary.
    pub read_preview: bool,
}

/// A settled message's usage waiting for its tool cards to exist (the
/// runtime announces cards with `tool_execution_start` after `message_end`;
/// TS places the row below the message's tool blocks).
#[derive(Debug, Clone, PartialEq)]
struct PendingUsage {
    usage: Usage,
    /// Tool calls of the message whose cards have not been created yet.
    waiting: Vec<String>,
    /// TS `groupedReadUsageCallIds`: set when every call is a groupable read.
    read_ids: Option<Vec<String>>,
}

/// A compaction/handoff divider (§10).
#[derive(Debug, Clone, PartialEq)]
pub struct Compaction {
    /// Method: `remote|soft|handoff|snapcompact|shake|…`.
    pub method: String,
    /// Context size before compaction.
    pub tokens_before: u64,
    /// Context size after (absent for legacy methods).
    pub tokens_after: Option<u64>,
    /// Summary text (markdown).
    pub summary: String,
    /// Optional warning.
    pub warning: Option<String>,
}

impl Compaction {
    /// Divider label (`remote-compacted|soft-compacted|handed-off|
    /// snap-compacted|shaken|compacted`, §10).
    #[must_use]
    pub fn label(&self) -> &str {
        match self.method.as_str() {
            "remote" => "remote-compacted",
            "soft" => "soft-compacted",
            "handoff" => "handed-off",
            "snapcompact" => "snap-compacted",
            "shake" => "shaken",
            _ => "compacted",
        }
    }
}

/// A `custom`/`custom_message` transcript entry (§11).
#[derive(Debug, Clone, PartialEq)]
pub struct Custom {
    /// `customType` (or the message role for unknown custom roles).
    pub custom_type: String,
    /// Flattened text content.
    pub content_text: String,
    /// `display === false` — render nothing.
    pub display: bool,
    /// `attribution`: `user|agent`.
    pub attribution: String,
}

/// The typed payload of one transcript entry.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatKind {
    /// User/developer prompt bubble (§4).
    User {
        /// Flattened text.
        text: String,
        /// Synthetic (advisor dump etc.) — collapsed by default.
        synthetic: bool,
        /// Live-steered marker (`*` badge).
        live_steered: bool,
    },
    /// Assistant message (§3).
    Assistant(AssistantMsg),
    /// One tool card (§7).
    Tool(ToolCard),
    /// Folded consecutive reads (§7.1).
    ReadGroup(ReadGroup),
    /// Bash `!` execution (§7.2).
    Bash(Execution),
    /// Python `$` execution (§7.2).
    Python(Execution),
    /// Compaction/handoff divider (§10).
    Compaction(Compaction),
    /// Branch summary divider (§10).
    Branch {
        /// Summary text.
        summary: String,
    },
    /// Custom message box (§11).
    Custom(Custom),
    /// Standalone deferred usage row (§2 usage flush).
    Usage(Usage),
    /// Retry/compaction/runtime notice.
    Notice(String),
}

/// Assistant message payload (§3).
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantMsg {
    /// Text/thinking blocks in stream order.
    pub segments: Vec<AssistantSegment>,
    /// `stopReason` once settled.
    pub stop_reason: Option<String>,
    /// `errorMessage` when the turn failed.
    pub error_message: Option<String>,
    /// The turn-ending error is shown in the pinned banner above the editor,
    /// so the inline block is suppressed (TS `setErrorPinned`).
    pub error_pinned: bool,
    /// True while the message is still streaming.
    pub live: bool,
}

impl AssistantMsg {
    /// Joined markdown text (thinking excluded — renderers walk `segments`).
    #[must_use]
    pub fn text(&self) -> String {
        self.segments
            .iter()
            .filter_map(|s| match s {
                AssistantSegment::Text { text } => Some(text.as_str()),
                AssistantSegment::Thinking { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

/// One logical chat entry: a coarse [`ChatRole`] bucket plus the typed
/// [`ChatKind`] payload and a `finished` flag (scrollback-commit gate).
#[derive(Debug, Clone, PartialEq)]
pub struct ChatEntry {
    /// Coarse producer bucket (kept as a field for the naive app renderer).
    pub role: ChatRole,
    /// One-line display summary (kept as a field for the naive renderer).
    pub text: String,
    /// Typed payload for real renderers.
    pub kind: ChatKind,
    finished: bool,
}

impl ChatEntry {
    /// Whether this entry may retire into committed scrollback: entries only
    /// ever transition live → finished, so the committed prefix is the
    /// leading run of finished entries (`Messages::finished_len`).
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Whether ctrl+O expansion applies (tool cards, read groups,
    /// executions, dividers, synthetic inputs — §9).
    #[must_use]
    pub fn is_expandable(&self) -> bool {
        matches!(
            self.kind,
            ChatKind::Tool(_)
                | ChatKind::ReadGroup(_)
                | ChatKind::Bash(_)
                | ChatKind::Python(_)
                | ChatKind::Compaction(_)
                | ChatKind::Branch { .. }
                | ChatKind::User {
                    synthetic: true,
                    ..
                }
        )
    }

    fn new(role: ChatRole, text: String, kind: ChatKind, finished: bool) -> Self {
        Self {
            role,
            text,
            kind,
            finished,
        }
    }

    /// Test constructor over the typed payload (block-renderer unit tests).
    #[cfg(test)]
    pub fn new_for_test(role: ChatRole, kind: ChatKind) -> Self {
        let text = match &kind {
            ChatKind::Assistant(a) => a.text(),
            ChatKind::Tool(card) => format!("{} {}", card.name, card.call_id),
            ChatKind::User { text, .. } => text.clone(),
            ChatKind::Bash(e) | ChatKind::Python(e) => e.command.clone(),
            ChatKind::Notice(t) => t.clone(),
            ChatKind::Branch { summary } => summary.clone(),
            ChatKind::Compaction(c) => c.label().to_owned(),
            ChatKind::Custom(c) => c.content_text.clone(),
            ChatKind::ReadGroup(g) => format!("{} reads", g.reads.len()),
            ChatKind::Usage(_) => String::new(),
        };
        Self::new(role, text, kind, true)
    }
}

/// The ordered chat transcript model.
#[derive(Debug, Clone)]
pub struct Messages {
    entries: Vec<ChatEntry>,
    /// `toolCallId` → entry index of its card / read group.
    pending: HashMap<String, usize>,
    /// Entry index of the assistant message currently streaming.
    live_assistant: Option<usize>,
    /// The read group still accepting consecutive reads (`None` once any
    /// other entry displaced it).
    open_group: Option<usize>,
    /// A locally echoed user prompt still awaits its runtime echo (dedupe
    /// against `message_*` frames).
    user_echo_pending: bool,
    /// Usage captured at `message_end`, placed once the message's tool
    /// cards exist (or at the next message / turn end).
    pending_usage: Option<PendingUsage>,
    /// The latest user prompt's timestamp (epoch ms): the turn-time anchor.
    turn_started_at: Option<i64>,
    /// Entry index + message of the error shown in the pinned banner.
    pinned_error: Option<(usize, String)>,
    /// Display settings (usage rows, read previews).
    prefs: DisplayPrefs,
    /// Number of leading assistant content parts already frozen into
    /// earlier (pre-tool) entries; later updates skip them.
    split_skip: usize,
    /// Total text/thinking parts in the last processed assistant message.
    seen_parts: usize,
    streaming: bool,
    expanded: bool,
    /// Entries [`..committed`](Self::committed) have already been retired
    /// into the terminal scrollback by the shell renderer.
    committed: usize,
    /// Theme the block renderer resolves tokens through.
    theme: Theme,
    /// Smooth streaming reveal (chat.md §8); ticked by the draw loop.
    reveal: StreamingRevealController,
    /// Last revealed snapshot (live tail renders this instead of the raw
    /// segments while smooth reveal is catching up).
    reveal_display: Option<Vec<RevealBlock>>,
    /// Incremental layout of the live assistant entry (drawn every frame
    /// while it streams).
    live_cache: RefCell<blocks::LiveCache>,
}

impl Default for Messages {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            pending: HashMap::new(),
            live_assistant: None,
            open_group: None,
            user_echo_pending: false,
            pending_usage: None,
            turn_started_at: None,
            pinned_error: None,
            prefs: DisplayPrefs::default(),
            split_skip: 0,
            seen_parts: 0,
            streaming: false,
            expanded: false,
            committed: 0,
            theme: Theme::default(),
            reveal: StreamingRevealController::new(true),
            reveal_display: None,
            live_cache: RefCell::default(),
        }
    }
}

impl Messages {
    /// An empty chat.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// All entries, oldest first.
    #[must_use]
    pub fn entries(&self) -> &[ChatEntry] {
        &self.entries
    }

    /// True while a `message_update` stream is open.
    #[must_use]
    pub fn is_streaming(&self) -> bool {
        self.streaming
    }

    /// Transcript-wide expansion state (ctrl+O, action `app.tools.expand`).
    #[must_use]
    pub fn expanded(&self) -> bool {
        self.expanded
    }

    /// Set transcript-wide expansion; applies to every expandable entry (§9).
    pub fn set_expanded(&mut self, expanded: bool) {
        self.expanded = expanded;
    }

    /// Toggle transcript-wide expansion (ctrl+O).
    pub fn toggle_expanded(&mut self) {
        self.expanded = !self.expanded;
    }

    /// Length of the leading run of finished entries — those may be
    /// committed to scrollback; everything from this index on is live.
    #[must_use]
    pub fn finished_len(&self) -> usize {
        self.entries
            .iter()
            .position(|e| !e.is_finished())
            .unwrap_or(self.entries.len())
    }

    /// Number of entries already retired into terminal scrollback (the
    /// commit watermark). Entries `[committed_len()..]` must be the only
    /// ones the live region redraws.
    #[must_use]
    pub fn committed_len(&self) -> usize {
        self.committed
    }

    /// The live region: entries at/after the commit watermark (`ShellRender`
    /// invariant — committed entries are never re-drawn in the live region).
    #[must_use]
    pub fn live_tail(&self) -> &[ChatEntry] {
        &self.entries[self.committed..]
    }

    /// Rows of newly finished entries for scrollback commit (`ShellRender`).
    /// Advances the commit watermark; later calls only return entries that
    /// finished since the last call. Rows go through the chat block
    /// renderer: user bubbles, markdown bodies, tool family cards.
    pub fn take_committed_rows(&mut self, width: u16) -> Vec<Line<'static>> {
        let end = self.finished_len();
        let theme = &self.theme;
        let expanded = self.expanded;
        let mut rows = Vec::new();
        for entry in &self.entries[self.committed..end] {
            rows.extend(blocks::entry_lines(
                entry,
                expanded,
                usize::from(width).max(8),
                theme,
            ));
        }
        self.committed = end;
        rows
    }

    /// Rows for the whole transcript (resize rebuild inside tmux/herdr).
    /// Resets the commit watermark to the finished prefix: rows past it are
    /// the live tail the live region keeps redrawing.
    pub fn all_rows(&mut self, width: u16) -> Vec<Line<'static>> {
        let theme = &self.theme;
        let expanded = self.expanded;
        let mut rows = Vec::new();
        for entry in &self.entries {
            rows.extend(blocks::entry_lines(
                entry,
                expanded,
                usize::from(width).max(8),
                theme,
            ));
        }
        self.committed = self.finished_len();
        rows
    }

    /// Theme the block renderer resolves tokens through.
    #[must_use]
    pub fn theme(&self) -> &Theme {
        &self.theme
    }

    /// Replace the renderer theme (user theme load, tests).
    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
        *self.live_cache.get_mut() = blocks::LiveCache::default();
    }

    /// Display settings in effect.
    #[must_use]
    pub fn display_prefs(&self) -> DisplayPrefs {
        self.prefs
    }

    /// Apply display settings (startup, `/settings` edits). Affects entries
    /// settled from now on (TS reads the settings at `message_end`).
    pub fn set_display_prefs(&mut self, prefs: DisplayPrefs) {
        self.prefs = prefs;
    }

    /// The turn-ending provider error pinned above the editor (TS
    /// `ErrorBannerComponent`); cleared when the next run starts.
    #[must_use]
    pub fn pinned_error(&self) -> Option<&str> {
        self.pinned_error
            .as_ref()
            .map(|(_, message)| message.as_str())
    }

    /// Drop the pinned banner and restore its error inline in the
    /// transcript (TS `clearPinnedError` + `setErrorPinned(false)`). The
    /// entry was held live while pinned, so it commits with the error shown.
    fn clear_pinned_error(&mut self) {
        let Some((idx, _)) = self.pinned_error.take() else {
            return;
        };
        if let Some(entry) = self.entries.get_mut(idx) {
            if let ChatKind::Assistant(a) = &mut entry.kind {
                a.error_pinned = false;
            }
            entry.finished = true;
        }
    }

    /// One draw-loop reveal tick (chat.md §8, 30 fps cadence). Call once per
    /// frame from the draw loop; the live tail renders the returned
    /// snapshot until the next tick.
    pub fn tick_reveal(&mut self) {
        if let Some(blocks) = self.reveal.tick() {
            self.reveal_display = Some(blocks);
        }
    }

    /// Whether the streaming reveal still has frames to produce.
    #[must_use]
    pub fn reveal_animating(&self) -> bool {
        self.reveal.is_animating()
    }

    /// Rows for the live-tail entry at `idx` (transcript index). A live
    /// assistant entry renders the reveal snapshot when one exists; every
    /// other entry renders through [`blocks::entry_lines`].
    #[must_use]
    pub fn entry_lines_at(&self, idx: usize, width: usize) -> Vec<Line<'static>> {
        self.entry_rows_at(idx, width, usize::MAX)
    }

    /// The last `max_rows` rows of the uncommitted tail (entries at/after
    /// the commit watermark) — what the live region can show. The streaming
    /// assistant entry is laid out incrementally and only its visible rows
    /// are copied, so a frame costs O(new text + screen), not O(answer).
    #[must_use]
    pub fn tail_rows(&self, width: usize, max_rows: usize) -> Vec<Line<'static>> {
        let mut picked: Vec<Vec<Line<'static>>> = Vec::new();
        let mut left = max_rows;
        for idx in (self.committed..self.entries.len()).rev() {
            if left == 0 {
                break;
            }
            let rows = self.entry_rows_at(idx, width, left);
            left -= rows.len();
            picked.push(rows);
        }
        picked.into_iter().rev().flatten().collect()
    }

    /// [`Self::entry_lines_at`], keeping only the last `max_rows` rows.
    fn entry_rows_at(&self, idx: usize, width: usize, max_rows: usize) -> Vec<Line<'static>> {
        let entry = &self.entries[idx];
        let width = width.max(8);
        if let ChatKind::Assistant(a) = &entry.kind {
            if a.live {
                let mut cache = self.live_cache.borrow_mut();
                return match &self.reveal_display {
                    Some(snapshot) => blocks::revealed_rows(
                        snapshot,
                        width,
                        &self.theme,
                        Some(&mut *cache),
                        max_rows,
                    ),
                    None => blocks::assistant_rows(
                        a,
                        self.expanded,
                        width,
                        &self.theme,
                        Some(&mut *cache),
                        max_rows,
                    ),
                };
            }
        }
        let mut rows = blocks::entry_lines(entry, self.expanded, width, &self.theme);
        rows.drain(..rows.len().saturating_sub(max_rows));
        rows
    }

    /// Apply one runtime event (live path).
    pub fn apply(&mut self, event: &AppEvent) {
        self.apply_event(event);
        self.close_displaced_groups();
    }

    fn apply_event(&mut self, event: &AppEvent) {
        match event {
            AppEvent::AgentStart => {
                self.streaming = true;
                self.clear_pinned_error();
            }
            AppEvent::TurnStart => self.streaming = true,
            AppEvent::AgentEnd(_) | AppEvent::TurnEnd(_) | AppEvent::SessionSettled => {
                self.streaming = false;
                self.seal_turn(matches!(event, AppEvent::TurnEnd(_)));
            }
            AppEvent::MessageStart(frame) => {
                if let Some(message) = frame.get("message") {
                    self.begin_message(message);
                }
            }
            AppEvent::MessageUpdate(frame) => {
                if let Some(message) = frame.get("message") {
                    self.update_message(message, true);
                }
            }
            AppEvent::MessageEnd(frame) => {
                if let Some(message) = frame.get("message") {
                    self.update_message(message, false);
                }
            }
            AppEvent::ToolExecutionStart(frame) => self.tool_start(frame),
            AppEvent::ToolExecutionUpdate(frame) => self.tool_update(frame),
            AppEvent::ToolExecutionEnd(frame) => self.tool_end(frame),
            AppEvent::PromptResult {
                success,
                error,
                frame,
                ..
            } => {
                // A failed agent run already shows its error on the assistant
                // block (TS event-controller: errorMessage on message_end); a
                // second row would repeat it. Only failures that never reached
                // the agent (unknown command, resume/rewind failure) need one.
                let agent_invoked = frame
                    .get("agentInvoked")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if !success && !agent_invoked {
                    let msg = error.as_deref().unwrap_or("prompt failed");
                    self.push(ChatEntry::new(
                        ChatRole::System,
                        format!("error: {msg}"),
                        ChatKind::Notice(format!("error: {msg}")),
                        true,
                    ));
                }
            }
            AppEvent::Retry(frame) => {
                let kind = frame.get("type").and_then(Value::as_str).unwrap_or("retry");
                self.push(ChatEntry::new(
                    ChatRole::System,
                    format!("[{kind}]"),
                    ChatKind::Notice(format!("[{kind}]")),
                    true,
                ));
            }
            AppEvent::AutoCompaction(frame) => {
                let kind = frame
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("compaction");
                self.push(ChatEntry::new(
                    ChatRole::System,
                    format!("[{kind}]"),
                    ChatKind::Notice(format!("[{kind}]")),
                    true,
                ));
            }
            _ => {}
        }
    }

    /// Reset and rebuild from a persisted message array (`agent_end`'s
    /// `messages`, a resumed session's entries). Everything is finished.
    pub fn rebuild(&mut self, messages: &[Value]) {
        let theme = std::mem::take(&mut self.theme);
        let prefs = self.prefs;
        *self = Self::new();
        self.theme = theme;
        self.prefs = prefs;
        for message in messages {
            self.append_message(message, true);
        }
        // Calls whose result never arrived (aborted session) are settled.
        for entry in &mut self.entries {
            if let ChatKind::Bash(e) | ChatKind::Python(e) = &mut entry.kind {
                if e.status == ExecStatus::Running {
                    e.status = ExecStatus::Cancelled;
                    entry.text = exec_summary(e);
                    entry.finished = true;
                }
            }
        }
        self.seal_turn(false);
    }

    /// Record a purely local UI notice (app-level echo, not a runtime
    /// frame): "retry aborted", "unknown command /x", hints, etc.
    pub fn push_system(&mut self, text: &str) {
        self.push(ChatEntry::new(
            ChatRole::System,
            text.to_owned(),
            ChatKind::Notice(text.to_owned()),
            true,
        ));
    }

    /// Record a locally-submitted user prompt (echoed immediately).
    pub fn push_user(&mut self, text: &str) {
        self.user_echo_pending = true;
        let entry = ChatEntry::new(
            ChatRole::User,
            text.to_owned(),
            ChatKind::User {
                text: text.to_owned(),
                synthetic: false,
                live_steered: self.streaming,
            },
            true,
        );
        self.push(entry);
    }

    // ---- internal helpers -------------------------------------------------

    fn push(&mut self, entry: ChatEntry) -> usize {
        if !matches!(entry.kind, ChatKind::ReadGroup(_)) {
            // Any non-read entry displaces the open read group.
            self.open_group = None;
        }
        self.entries.push(entry);
        self.entries.len() - 1
    }

    fn begin_message(&mut self, message: &Value) {
        if role_of(message) == "assistant" && self.live_assistant.is_none() {
            self.flush_usage();
            let idx = self.push(assistant_entry(Vec::new(), true));
            self.live_assistant = Some(idx);
            self.streaming = true;
        }
    }

    /// Merge an accumulated message payload into the transcript.
    fn update_message(&mut self, message: &Value, live: bool) {
        match role_of(message) {
            "assistant" => {
                let (segments, total) = assistant_segments(message, self.split_skip);
                self.seen_parts = total;
                if live {
                    self.reveal
                        .set_target(reveal_blocks(&segments, &ThinkingDisplay::default()));
                }
                match self.live_assistant {
                    Some(idx) => {
                        let entry = &mut self.entries[idx];
                        if let ChatKind::Assistant(a) = &mut entry.kind {
                            a.segments = segments;
                            a.live = live;
                            entry.text = a.text();
                        }
                    }
                    None if !segments.is_empty() => {
                        let idx = self.push(assistant_entry(segments, live));
                        self.live_assistant = Some(idx);
                        if let Some(entry) = self.entries.last_mut() {
                            if let ChatKind::Assistant(a) = &mut entry.kind {
                                entry.text = a.text();
                            }
                        }
                    }
                    None => {}
                }
                if !live {
                    // Settle: capture stop reason / error / deferred usage.
                    self.streaming = false;
                    self.reveal_display = Some(self.reveal.finish());
                    let idx = match self.live_assistant.take() {
                        Some(idx) => Some(idx),
                        // An error that streamed nothing still needs its own
                        // block (never an earlier turn's answer).
                        None if str_field(message, "errorMessage").is_some() => {
                            Some(self.push(assistant_entry(Vec::new(), false)))
                        }
                        None => self.last_assistant_index(),
                    };
                    let pinned = pinned_error_message(message);
                    if let Some(idx) = idx {
                        let entry = &mut self.entries[idx];
                        if let ChatKind::Assistant(a) = &mut entry.kind {
                            a.live = false;
                            a.stop_reason = str_field(message, "stopReason").map(String::from);
                            a.error_message = str_field(message, "errorMessage").map(String::from);
                            a.error_pinned = pinned.is_some();
                        }
                        // A pinned error's entry stays live until the banner
                        // clears: it then commits with the error inline.
                        entry.finished = pinned.is_none();
                        if let Some(message) = pinned {
                            self.clear_pinned_error();
                            self.pinned_error = Some((idx, message));
                        }
                    }
                    self.settle_usage(message, true);
                    self.split_skip = 0;
                }
            }
            _ => self.append_message(message, !live),
        }
    }

    /// Append a persisted assistant message: text/thinking chunks become
    /// settled assistant entries, each `toolCall` block a pending tool entry
    /// (completed by the following `toolResult`), in content order.
    fn append_assistant(&mut self, message: &Value) {
        let parts: &[Value] = message
            .get("content")
            .and_then(Value::as_array)
            .map_or(&[], Vec::as_slice);
        let mut chunk: Vec<Value> = Vec::new();
        let mut last_assistant: Option<usize> = None;
        let mut had_tool = false;
        for part in parts {
            if part.get("type").and_then(Value::as_str) == Some("toolCall") {
                last_assistant = self.flush_assistant_chunk(&mut chunk).or(last_assistant);
                had_tool = true;
                let call_id = str_field(part, "id").unwrap_or("").to_owned();
                let name = str_field(part, "name").unwrap_or("tool").to_owned();
                let args = part.get("arguments").cloned().unwrap_or(Value::Null);
                self.start_tool_entry(call_id, name, args);
            } else {
                chunk.push(part.clone());
            }
        }
        last_assistant = self.flush_assistant_chunk(&mut chunk).or(last_assistant);
        let error_message = str_field(message, "errorMessage").map(String::from);
        if last_assistant.is_none() && (!had_tool || error_message.is_some()) {
            // No visible content (e.g. an aborted/errored turn): keep the
            // entry so the stop reason / error still shows.
            last_assistant = Some(self.push(assistant_entry(Vec::new(), false)));
        }
        if let Some(idx) = last_assistant {
            if let ChatKind::Assistant(a) = &mut self.entries[idx].kind {
                a.stop_reason = str_field(message, "stopReason").map(String::from);
                a.error_message = error_message;
            }
        }
        // Rebuild: the message's tool entries already exist (TS rebuild
        // `flushPendingUsage` once the turn's tool blocks materialized).
        self.settle_usage(message, false);
    }

    fn flush_assistant_chunk(&mut self, chunk: &mut Vec<Value>) -> Option<usize> {
        let content = Value::Array(std::mem::take(chunk));
        let (segments, _) = assistant_segments(&serde_json::json!({ "content": content }), 0);
        if segments.is_empty() {
            return None;
        }
        Some(self.push(assistant_entry(segments, false)))
    }

    /// Append a full (persisted-shaped) message by role (§1 dispatch table).
    #[allow(clippy::too_many_lines)] // one arm per wire role; splitting buys nothing
    fn append_message(&mut self, message: &Value, finished: bool) {
        match role_of(message) {
            "user" | "developer" => {
                // Only user-attributed prompts anchor the turn time (TS
                // `#turnStartedAt`; an agent-attributed redirect must not).
                if str_field(message, "attribution") != Some("agent") {
                    if let Some(ts) = message.get("timestamp").and_then(Value::as_i64) {
                        self.turn_started_at = Some(ts);
                    }
                }
                let text = flatten_content(message.get("content"));
                // Dedupe the runtime echo of a locally submitted prompt.
                if self.user_echo_pending
                    && self
                        .entries
                        .last()
                        .is_some_and(|e| e.role == ChatRole::User && e.text == text)
                {
                    self.user_echo_pending = false;
                    return;
                }
                self.user_echo_pending = false;
                let synthetic = message
                    .get("synthetic")
                    .is_some_and(|v| v == &Value::Bool(true));
                let live_steered = message
                    .get("liveSteered")
                    .is_some_and(|v| v == &Value::Bool(true));
                self.push(ChatEntry::new(
                    ChatRole::User,
                    text.clone(),
                    ChatKind::User {
                        text,
                        synthetic,
                        live_steered,
                    },
                    finished,
                ));
            }
            "toolResult" => {
                let call_id = str_field(message, "toolCallId").unwrap_or("");
                let status = if message
                    .get("isError")
                    .is_some_and(|v| v == &Value::Bool(true))
                {
                    ToolStatus::Error
                } else {
                    ToolStatus::Ok
                };
                let text = flatten_content(message.get("content"));
                let details = message.get("details").cloned().unwrap_or(Value::Null);
                if self.pending.contains_key(call_id) {
                    self.apply_tool_result(call_id, status, text, details);
                } else {
                    let name = str_field(message, "toolName").unwrap_or("tool");
                    self.push_orphan_result(call_id, name, status, text, details);
                }
            }
            "assistant" => self.append_assistant(message),
            "bashExecution" | "pythonExecution" | "jsExecution" => {
                let role = role_of(message);
                let command = str_field(message, "command")
                    .or_else(|| str_field(message, "code"))
                    .unwrap_or("")
                    .to_owned();
                let output = str_field(message, "output").unwrap_or("").to_owned();
                let exit_code = message.get("exitCode").and_then(Value::as_i64);
                let cancelled = message
                    .get("cancelled")
                    .is_some_and(|v| v == &Value::Bool(true));
                let status = if cancelled {
                    ExecStatus::Cancelled
                } else {
                    match exit_code {
                        Some(code) if code != 0 => ExecStatus::Error,
                        _ => ExecStatus::Complete,
                    }
                };
                let excluded = message
                    .get("excludeFromContext")
                    .is_some_and(|v| v == &Value::Bool(true));
                let exec = Execution {
                    call_id: String::new(),
                    command,
                    output,
                    status,
                    exit_code,
                    excluded,
                    details: Value::Null,
                };
                let text = exec_summary(&exec);
                let kind = if role == "bashExecution" {
                    ChatKind::Bash(exec)
                } else {
                    ChatKind::Python(exec)
                };
                self.push(ChatEntry::new(ChatRole::Tool, text, kind, finished));
            }
            "compactionSummary" | "compaction" => {
                let method = str_field(message, "method").unwrap_or("").to_owned();
                let tokens_before = message
                    .get("tokensBefore")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let tokens_after = message.get("tokensAfter").and_then(Value::as_u64);
                let summary = str_field(message, "summary").unwrap_or("").to_owned();
                let warning = str_field(message, "warning").map(String::from);
                let text = compaction_text(&method, tokens_before, tokens_after);
                self.push(ChatEntry::new(
                    ChatRole::System,
                    text,
                    ChatKind::Compaction(Compaction {
                        method,
                        tokens_before,
                        tokens_after,
                        summary,
                        warning,
                    }),
                    finished,
                ));
            }
            "branchSummary" | "branch_summary" => {
                let summary = str_field(message, "summary").unwrap_or("").to_owned();
                let text = format!("branch: {}", first_line(&summary));
                self.push(ChatEntry::new(
                    ChatRole::System,
                    text,
                    ChatKind::Branch { summary },
                    finished,
                ));
            }
            other => {
                // custom / custom_message / hookMessage / unknown roles.
                let custom_type = str_field(message, "customType").unwrap_or(other).to_owned();
                let content_text = match message.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    other_content => flatten_content(other_content),
                };
                let display = message
                    .get("display")
                    .is_none_or(|v| v != &Value::Bool(false));
                let attribution = str_field(message, "attribution")
                    .unwrap_or("agent")
                    .to_owned();
                self.push(ChatEntry::new(
                    ChatRole::System,
                    format!("[{custom_type}] {}", first_line(&content_text)),
                    ChatKind::Custom(Custom {
                        custom_type,
                        content_text,
                        display,
                        attribution,
                    }),
                    finished,
                ));
            }
        }
    }

    // ---- tools ------------------------------------------------------------

    fn tool_start(&mut self, frame: &Value) {
        // A tool call finishes the assistant's leading segment immediately
        // (TS snap on `hasToolCalls`): freeze it; text arriving after the
        // tool card lands in a fresh assistant entry after the card.
        self.reveal.snap_to_tool_call();
        self.reveal_display = None;
        if let Some(idx) = self.live_assistant.take() {
            self.split_skip = self.seen_parts;
            if let Some(entry) = self.entries.get_mut(idx) {
                if let ChatKind::Assistant(a) = &mut entry.kind {
                    a.live = false;
                }
                entry.finished = true;
            }
        }
        let call_id = str_field(frame, "toolCallId").unwrap_or("").to_owned();
        let name = str_field(frame, "toolName").unwrap_or("tool").to_owned();
        let args = frame.get("args").cloned().unwrap_or(Value::Null);
        self.start_tool_entry(call_id.clone(), name, args);
        self.note_tool_started(&call_id);
    }

    // ---- usage rows (TS event-controller `message_end` usage block) --------

    /// A settled assistant message's usage (`display.showTokenUsage` only,
    /// billed messages only). Placed once the message's tool cards exist:
    /// `wait_for_tools` is false on rebuild, where they already do.
    fn settle_usage(&mut self, message: &Value, wait_for_tools: bool) {
        self.flush_usage();
        if !self.prefs.show_token_usage {
            return;
        }
        let Some(mut usage) = parse_usage(message) else {
            return;
        };
        if self.prefs.show_turn_time {
            usage.turn_elapsed_ms = turn_elapsed_ms(self.turn_started_at, message);
        }
        let waiting = if wait_for_tools {
            tool_call_ids(message)
                .into_iter()
                .filter(|id| !self.pending.contains_key(id))
                .collect()
        } else {
            Vec::new()
        };
        self.pending_usage = Some(PendingUsage {
            usage,
            waiting,
            read_ids: grouped_read_usage_call_ids(message),
        });
        if self
            .pending_usage
            .as_ref()
            .is_some_and(|p| p.waiting.is_empty())
        {
            self.flush_usage();
        }
    }

    /// A tool card was created: place the pending usage row once every call
    /// of its message has a card.
    fn note_tool_started(&mut self, call_id: &str) {
        let Some(pending) = &mut self.pending_usage else {
            return;
        };
        pending.waiting.retain(|id| id != call_id);
        if pending.waiting.is_empty() {
            self.flush_usage();
        }
    }

    /// Place the pending usage: nested under the open read group when the
    /// message only made groupable reads (TS `attachUsage`), else a
    /// standalone row that also ends the read group.
    fn flush_usage(&mut self) {
        let Some(pending) = self.pending_usage.take() else {
            return;
        };
        if let (Some(ids), Some(idx)) = (&pending.read_ids, self.open_group) {
            if let ChatKind::ReadGroup(g) = &mut self.entries[idx].kind {
                let anchor = ids
                    .iter()
                    .rev()
                    .find(|id| g.reads.iter().any(|r| &r.call_id == *id));
                if let (true, Some(anchor)) = (g.open, anchor) {
                    g.usages.push((anchor.clone(), pending.usage));
                    return;
                }
            }
        }
        self.push(ChatEntry::new(
            ChatRole::System,
            usage_text(pending.usage),
            ChatKind::Usage(pending.usage),
            true,
        ));
    }

    /// Create the pending entry for a tool call (shared by the live
    /// `tool_execution_start` path and the persisted-history rebuild).
    fn start_tool_entry(&mut self, call_id: String, name: String, args: Value) {
        let explore = self.theme.clean() && !self.prefs.read_preview && is_explore_tool(&name);
        if name == "read" || explore {
            let path = item_label(&name, &args).unwrap_or("").to_owned();
            let item = ReadItem {
                call_id: call_id.clone(),
                path,
                status: ToolStatus::Pending,
                preview: None,
                tool: name.clone(),
                args: args.clone(),
                details: Value::Null,
            };
            let open = self.open_group.or_else(|| self.reusable_group());
            let idx = match open {
                Some(idx) => match &mut self.entries[idx].kind {
                    ChatKind::ReadGroup(g) if g.open => {
                        g.reads.push(item);
                        idx
                    }
                    _ => self.push_read_group(item),
                },
                None => self.push_read_group(item),
            };
            self.open_group = Some(idx);
            self.pending.insert(call_id, idx);
            return;
        }
        if name == "bash" || name == "eval" {
            let command = arg_str(&args, "command")
                .or_else(|| arg_str(&args, "code"))
                .unwrap_or("")
                .to_owned();
            let excluded = arg_flag(&args, "excludeFromContext");
            let exec = Execution {
                call_id: call_id.clone(),
                command,
                output: String::new(),
                status: ExecStatus::Running,
                exit_code: None,
                excluded,
                details: Value::Null,
            };
            let text = exec_summary(&exec);
            let kind = if name == "bash" {
                ChatKind::Bash(exec)
            } else {
                ChatKind::Python(exec)
            };
            let idx = self.push(ChatEntry::new(ChatRole::Tool, text, kind, false));
            self.pending.insert(call_id, idx);
            return;
        }
        let text = tool_summary(&name, &args);
        let idx = self.push(ChatEntry::new(
            ChatRole::Tool,
            text,
            ChatKind::Tool(ToolCard {
                call_id: call_id.clone(),
                name,
                args,
                status: ToolStatus::Pending,
                result_text: None,
                details: Value::Null,
                partial: true,
            }),
            false,
        ));
        self.pending.insert(call_id, idx);
    }

    /// Clean style: the still-open exploration group behind entries that
    /// draw nothing (an assistant message that only carried tool calls), so
    /// tool calls spread over several messages fold into one row.
    fn reusable_group(&self) -> Option<usize> {
        if !self.theme.clean() {
            return None;
        }
        for (idx, entry) in self.entries.iter().enumerate().rev() {
            match &entry.kind {
                ChatKind::ReadGroup(g) if g.open && !g.preview => return Some(idx),
                _ if !entry_visible(entry) => {}
                _ => return None,
            }
        }
        None
    }

    fn push_read_group(&mut self, item: ReadItem) -> usize {
        let text = format!("read {}", item.path);
        self.push(ChatEntry::new(
            ChatRole::Tool,
            text,
            ChatKind::ReadGroup(ReadGroup {
                reads: vec![item],
                usages: Vec::new(),
                open: true,
                preview: self.prefs.read_preview,
            }),
            false,
        ))
    }

    fn tool_update(&mut self, frame: &Value) {
        let call_id = str_field(frame, "toolCallId").unwrap_or("");
        let Some(&idx) = self.pending.get(call_id) else {
            return;
        };
        if let Some(args) = frame.get("args") {
            self.with_tool_card(idx, |card| card.args = args.clone());
            if let Some(ChatKind::ReadGroup(g)) = self.entries.get_mut(idx).map(|e| &mut e.kind) {
                if let Some(item) = g.reads.iter_mut().find(|r| r.call_id == call_id) {
                    if let Some(p) = item_label(&item.tool, args) {
                        item.path = p.to_owned();
                    }
                    item.args = args.clone();
                }
            }
        }
        if let Some(partial) = frame.get("partialResult") {
            let text = result_text(partial);
            self.with_tool_card(idx, |card| {
                card.result_text = Some(text.clone());
                if !card.status.settled() {
                    card.status = ToolStatus::Running;
                }
            });
            if let Some(entry) = self.entries.get_mut(idx) {
                match &mut entry.kind {
                    ChatKind::ReadGroup(g) => {
                        if let Some(item) = g.reads.iter_mut().find(|r| r.call_id == call_id) {
                            item.status = ToolStatus::Running;
                        }
                    }
                    ChatKind::Bash(e) | ChatKind::Python(e) => {
                        e.output = text;
                        entry.text = exec_summary(e);
                    }
                    _ => {}
                }
            }
        }
    }

    fn tool_end(&mut self, frame: &Value) {
        let call_id = str_field(frame, "toolCallId").unwrap_or("");
        let is_error = frame
            .get("isError")
            .is_some_and(|v| v == &Value::Bool(true));
        let text = frame.get("result").map_or_else(String::new, result_text);
        let details = frame
            .get("result")
            .and_then(|r| r.get("details"))
            .cloned()
            .unwrap_or(Value::Null);
        let status = if is_error {
            ToolStatus::Error
        } else {
            ToolStatus::Ok
        };
        if !self.pending.contains_key(call_id) {
            let name = str_field(frame, "toolName").unwrap_or("tool");
            self.push_orphan_result(call_id, name, status, text, details);
            return;
        }
        self.apply_tool_result(call_id, status, text, details);
    }

    /// Result without a card: create a finished one (builder
    /// `#appendToolResult`). Registered in `pending` so a later duplicate
    /// result for the same call attaches instead of creating a second row.
    fn push_orphan_result(
        &mut self,
        call_id: &str,
        name: &str,
        status: ToolStatus,
        text: String,
        details: Value,
    ) {
        let idx = self.push(ChatEntry::new(
            ChatRole::Tool,
            format!("tool result: {}", first_line(&text)),
            ChatKind::Tool(ToolCard {
                call_id: call_id.to_owned(),
                name: name.to_owned(),
                args: Value::Null,
                status,
                result_text: Some(text),
                details,
                partial: false,
            }),
            true,
        ));
        if !call_id.is_empty() {
            self.pending.insert(call_id.to_owned(), idx);
        }
    }

    /// Route a settled result into the card / read item / execution.
    fn apply_tool_result(
        &mut self,
        call_id: &str,
        status: ToolStatus,
        text: String,
        details: Value,
    ) {
        let Some(&idx) = self.pending.get(call_id) else {
            return;
        };
        self.with_tool_card(idx, |card| {
            card.status = status;
            card.result_text = Some(text.clone());
            card.details = details.clone();
            card.partial = false;
        });
        if let Some(entry) = self.entries.get_mut(idx) {
            match &mut entry.kind {
                ChatKind::ReadGroup(g) => {
                    if let Some(item) = g.reads.iter_mut().find(|r| r.call_id == call_id) {
                        item.status = status;
                        item.preview = Some(text.clone());
                        item.details = details.clone();
                    }
                    let summary = if g.reads.len() == 1 {
                        format!("read {}", g.reads[0].path)
                    } else {
                        format!("read {} files", g.reads.len())
                    };
                    entry.text = summary;
                    if !g.open && g.reads.iter().all(|r| r.status.settled()) {
                        entry.finished = true;
                    }
                }
                ChatKind::Bash(e) | ChatKind::Python(e) => {
                    e.output = text;
                    e.details = details;
                    e.status = if status.settled() && status == ToolStatus::Error {
                        ExecStatus::Error
                    } else if e.status == ExecStatus::Running {
                        ExecStatus::Complete
                    } else {
                        e.status
                    };
                    entry.text = exec_summary(e);
                    entry.finished = true;
                }
                ChatKind::Tool(_) => {
                    entry.finished = true;
                }
                _ => {}
            }
        }
    }

    fn last_assistant_index(&self) -> Option<usize> {
        self.entries
            .iter()
            .rposition(|e| matches!(e.kind, ChatKind::Assistant(_)))
    }

    fn with_tool_card(&mut self, idx: usize, f: impl FnOnce(&mut ToolCard)) {
        if let Some(entry) = self.entries.get_mut(idx) {
            if let ChatKind::Tool(card) = &mut entry.kind {
                f(card);
            }
        }
    }

    /// Turn settled: close the live assistant, seal abandoned pending tools
    /// and flush the deferred usage row (§2 `#flushPendingUsage`).
    /// `soft` (a turn inside a running agent loop): the clean style keeps a
    /// settled exploration group open so the next turn's reads/searches fold
    /// into it; [`Self::close_displaced_groups`] closes it once anything
    /// visible lands after it.
    fn seal_turn(&mut self, soft: bool) {
        let keep_open = soft && self.theme.clean();
        // Before the read groups close: a reads-only message's row nests.
        self.flush_usage();
        self.split_skip = 0;
        if let Some(idx) = self.live_assistant.take() {
            if let Some(entry) = self.entries.get_mut(idx) {
                if let ChatKind::Assistant(a) = &mut entry.kind {
                    a.live = false;
                }
                entry.finished = true;
            }
        }
        for entry in &mut self.entries {
            match &mut entry.kind {
                ChatKind::Tool(card) if card.partial => {
                    // Turn abandoned the call without a result (`seal()`).
                    card.partial = false;
                    entry.finished = true;
                }
                ChatKind::ReadGroup(g) if g.open => {
                    if keep_open && !g.preview {
                        continue;
                    }
                    g.open = false;
                    entry.finished = true;
                }
                _ => {}
            }
        }
    }

    /// Clean style: close exploration groups that something visible now
    /// follows (they can no longer fold, and must not hold back the
    /// scrollback commit).
    fn close_displaced_groups(&mut self) {
        if !self.theme.clean() {
            return;
        }
        let mut visible_after = false;
        for entry in self.entries.iter_mut().rev() {
            match &mut entry.kind {
                ChatKind::ReadGroup(g) if g.open => {
                    if visible_after {
                        g.open = false;
                        if g.reads.iter().all(|r| r.status.settled()) {
                            entry.finished = true;
                        }
                    }
                    visible_after = true;
                }
                _ => {
                    if entry_visible(entry) {
                        visible_after = true;
                    }
                }
            }
        }
    }
}

// ---- payload helpers (TS wire shapes, spec section 1) -----------------------

/// Row label of a grouped call: the read path, or a search's pattern
/// (`glob` takes its pattern as `path`).
fn item_label<'v>(tool: &str, args: &'v Value) -> Option<&'v str> {
    if tool == "read" {
        arg_str(args, "path").or_else(|| arg_str(args, "file_path"))
    } else {
        arg_str(args, "pattern")
            .or_else(|| arg_str(args, "query"))
            .or_else(|| arg_str(args, "path"))
    }
}

/// Whether an entry draws anything (an assistant message that only carried
/// tool calls draws nothing; usage rows are chrome).
fn entry_visible(entry: &ChatEntry) -> bool {
    match &entry.kind {
        ChatKind::Assistant(a) => {
            a.error_message.is_some()
                || a.segments.iter().any(|s| match s {
                    AssistantSegment::Text { text } | AssistantSegment::Thinking { text } => {
                        !text.trim().is_empty()
                    }
                    #[allow(unreachable_patterns)]
                    _ => true,
                })
        }
        ChatKind::Usage(_) => false,
        ChatKind::ReadGroup(_) => true,
        _ => true,
    }
}

fn str_field<'v>(value: &'v Value, key: &str) -> Option<&'v str> {
    value.get(key).and_then(Value::as_str)
}

fn role_of(message: &Value) -> &str {
    str_field(message, "role").unwrap_or("")
}

fn arg_str<'v>(args: &'v Value, key: &str) -> Option<&'v str> {
    args.get(key).and_then(Value::as_str)
}

fn arg_flag(args: &Value, key: &str) -> bool {
    args.get(key).is_some_and(|v| v == &Value::Bool(true))
}

/// Flatten `content` (`string | [{text}] | {text}`) to plain text.
fn flatten_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        Some(Value::Object(_)) => str_field(content.unwrap_or(&Value::Null), "text")
            .unwrap_or("")
            .to_owned(),
        _ => String::new(),
    }
}

/// Build the ordered text/thinking segments of an assistant message,
/// skipping the first `skip` text/thinking content parts (already frozen
/// into pre-tool entries). Returns the segments and the total part count.
fn assistant_segments(message: &Value, skip: usize) -> (Vec<AssistantSegment>, usize) {
    let mut out: Vec<AssistantSegment> = Vec::new();
    let Some(parts) = message.get("content").and_then(Value::as_array) else {
        return (out, 0);
    };
    let mut seen = 0usize;
    for part in parts {
        let kind = part.get("type").and_then(Value::as_str);
        if !matches!(kind, Some("text" | "thinking")) {
            continue;
        }
        if seen < skip {
            seen += 1;
            continue;
        }
        seen += 1;
        match kind {
            Some("text") => {
                let text = part.get("text").and_then(Value::as_str).unwrap_or("");
                match out.last_mut() {
                    Some(AssistantSegment::Text { text: t }) => t.push_str(text),
                    _ if !text.is_empty() => {
                        out.push(AssistantSegment::Text { text: text.into() });
                    }
                    _ => {}
                }
            }
            Some("thinking") => {
                let text = part
                    .get("thinking")
                    .or_else(|| part.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                out.push(AssistantSegment::Thinking { text });
            }
            _ => {}
        }
    }
    (out, seen)
}

fn assistant_entry(segments: Vec<AssistantSegment>, live: bool) -> ChatEntry {
    let text = segments
        .iter()
        .filter_map(|s| match s {
            AssistantSegment::Text { text } => Some(text.as_str()),
            AssistantSegment::Thinking { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("");
    ChatEntry::new(
        ChatRole::Assistant,
        text,
        ChatKind::Assistant(AssistantMsg {
            segments,
            stop_reason: None,
            error_message: None,
            error_pinned: false,
            live,
        }),
        !live,
    )
}

/// A message's billed usage plus its request timing (`duration`/`ttft`/
/// `timestamp` ride on the message, not inside `usage`). `None` unless
/// billed (TS `assistantUsageIsBilled`).
fn parse_usage(message: &Value) -> Option<Usage> {
    let usage = message.get("usage")?;
    let ms = |key: &str| -> Option<u64> {
        let v = message.get(key)?;
        v.as_u64()
            .or_else(|| v.as_f64().map(|f| f.round().max(0.0) as u64))
    };
    let count = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
    let parsed = Usage {
        input: count("input"),
        output: count("output"),
        cache_read: count("cacheRead"),
        cache_write: count("cacheWrite"),
        duration_ms: ms("duration"),
        ttft_ms: ms("ttft"),
        timestamp_ms: message
            .get("timestamp")
            .and_then(Value::as_i64)
            .filter(|t| *t > 0),
        turn_elapsed_ms: None,
    };
    let premium = usage
        .get("premiumRequests")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    (parsed.input != 0
        || parsed.output != 0
        || parsed.cache_read != 0
        || parsed.cache_write != 0
        || premium > 0.0)
        .then_some(parsed)
}

/// TS `turnElapsedMs`: prompt timestamp → the response's `completedAt`.
fn turn_elapsed_ms(turn_started_at: Option<i64>, message: &Value) -> Option<u64> {
    let completed = message.get("completedAt").and_then(Value::as_i64)?;
    let elapsed = completed - turn_started_at?;
    u64::try_from(elapsed).ok().filter(|e| *e > 0)
}

/// `toolCall` ids of an assistant message, in content order.
fn tool_call_ids(message: &Value) -> Vec<String> {
    message
        .get("content")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .filter(|part| str_field(part, "type") == Some("toolCall"))
        .filter_map(|part| str_field(part, "id").map(String::from))
        .collect()
}

/// TS `groupedReadUsageCallIds`: the message's call ids when every call is a
/// read that folds into a group and nothing visible follows the first call.
fn grouped_read_usage_call_ids(message: &Value) -> Option<Vec<String>> {
    let parts = message.get("content").and_then(Value::as_array)?;
    let mut ids = Vec::new();
    for part in parts {
        match str_field(part, "type") {
            Some("toolCall") => {
                let args = part.get("arguments").unwrap_or(&Value::Null);
                let has_target = arg_str(args, "path")
                    .or_else(|| arg_str(args, "file_path"))
                    .is_some_and(|p| !p.trim().is_empty());
                if str_field(part, "name") != Some("read") || !has_target {
                    return None;
                }
                ids.push(str_field(part, "id").unwrap_or("").to_owned());
            }
            Some("image") if !ids.is_empty() => return None,
            Some("text") if !ids.is_empty() && !part_text(part, "text").trim().is_empty() => {
                return None;
            }
            Some("thinking")
                if !ids.is_empty() && !part_text(part, "thinking").trim().is_empty() =>
            {
                return None;
            }
            _ => {}
        }
    }
    (!ids.is_empty()).then_some(ids)
}

fn part_text<'v>(part: &'v Value, key: &str) -> &'v str {
    str_field(part, key).unwrap_or("")
}

/// The error to pin above the editor for a settled assistant message (TS
/// `message_end`: `stopReason === "error"` with a message, not a silent
/// abort).
fn pinned_error_message(message: &Value) -> Option<String> {
    if str_field(message, "stopReason") != Some("error") {
        return None;
    }
    str_field(message, "errorMessage")
        .filter(|m| !m.trim().is_empty())
        .map(String::from)
}

fn usage_text(usage: Usage) -> String {
    format!(
        "{} in / {} out",
        compact_number(usage.input),
        compact_number(usage.output)
    )
}

/// `256K`-style compact token amounts (section 10 `formatNumber`).
#[allow(clippy::cast_precision_loss)] // display-only rounding of token counts
fn compact_number(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn compaction_text(method: &str, before: u64, after: Option<u64>) -> String {
    let mut text = Compaction {
        method: method.to_owned(),
        tokens_before: before,
        tokens_after: after,
        summary: String::new(),
        warning: None,
    }
    .label()
    .to_owned();
    if let Some(after) = after {
        use std::fmt::Write as _;
        let _ = write!(
            text,
            " · {}→{}",
            compact_number(before),
            compact_number(after)
        );
    }
    text
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or("")
}

/// Short inline args preview for a tool card's summary line.
fn tool_summary(name: &str, args: &Value) -> String {
    let detail = match args {
        Value::Object(map) => map
            .iter()
            .find(|(k, _)| *k != "excludeFromContext")
            .and_then(|(_, v)| match v {
                Value::String(s) => Some(first_line(s).to_owned()),
                Value::Number(n) => Some(n.to_string()),
                _ => None,
            })
            .unwrap_or_default(),
        _ => String::new(),
    };
    if detail.is_empty() {
        name.to_owned()
    } else {
        format!("{name} {detail}")
    }
}

fn exec_summary(exec: &Execution) -> String {
    let marker = if exec.excluded { "!" } else { "$" };
    let state = match exec.status {
        ExecStatus::Running => "…".to_owned(),
        ExecStatus::Complete => String::new(),
        ExecStatus::Cancelled => " (cancelled)".to_owned(),
        ExecStatus::Error => format!(" (exit {})", exec.exit_code.unwrap_or(-1)),
    };
    format!("{marker} {}{state}", first_line(&exec.command))
}

/// Extract display text from a tool `result` / `partialResult` payload.
fn result_text(result: &Value) -> String {
    match result {
        Value::String(s) => s.clone(),
        Value::Object(_) => {
            let content = result.get("content").map_or_else(
                || {
                    str_field(result, "output")
                        .or_else(|| str_field(result, "text"))
                        .unwrap_or("")
                        .to_owned()
                },
                |c| flatten_content(Some(c)),
            );
            if content.is_empty() {
                // Details-only result (diffs, suffix resolutions…): keep the
                // card's text empty rather than dumping raw JSON.
                String::new()
            } else {
                content
            }
        }
        _ => String::new(),
    }
}

/// Build reveal blocks from assistant segments (chat.md §8): thinking text
/// is kept in display form, mirroring TS `rawThinking`.
fn reveal_blocks(segments: &[AssistantSegment], settings: &ThinkingDisplay) -> Vec<RevealBlock> {
    segments
        .iter()
        .map(|s| match s {
            AssistantSegment::Text { text } => RevealBlock::Text(text.clone()),
            AssistantSegment::Thinking { text } => RevealBlock::Thinking {
                display: format_thinking_for_display(text, settings.prose_only),
                raw: text.clone(),
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msg_update(text: &str) -> AppEvent {
        AppEvent::MessageUpdate(json!({
            "type": "message_update",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}
        }))
    }

    fn msg_end(text: &str) -> AppEvent {
        AppEvent::MessageEnd(json!({
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}
        }))
    }

    #[test]
    fn streams_assistant_text_into_one_entry() {
        let mut m = Messages::new();
        m.apply(&AppEvent::AgentStart);
        m.apply(&msg_update("hel"));
        m.apply(&msg_update("hello"));
        assert_eq!(m.entries().len(), 1);
        assert_eq!(m.entries()[0].text, "hello");
        assert_eq!(m.finished_len(), 0); // still live
        m.apply(&msg_end("hello"));
        assert!(!m.is_streaming());
        assert_eq!(m.finished_len(), 1);
    }

    #[test]
    fn assistant_keeps_thinking_segments_in_order() {
        let mut m = Messages::new();
        m.apply(&AppEvent::AgentStart);
        m.apply(&AppEvent::MessageUpdate(json!({
            "type": "message_update",
            "message": {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "answer"}
            ]}
        })));
        let ChatKind::Assistant(a) = &m.entries()[0].kind else {
            panic!("expected assistant");
        };
        assert_eq!(a.segments.len(), 2);
        assert!(matches!(&a.segments[0], AssistantSegment::Thinking { text } if text == "hmm"));
        assert!(matches!(&a.segments[1], AssistantSegment::Text { text } if text == "answer"));
    }

    #[test]
    fn user_prompt_echoes_and_dedupes_runtime_echo() {
        let mut m = Messages::new();
        m.push_user("do a thing");
        m.apply(&AppEvent::MessageStart(json!({
            "type": "message_start",
            "message": {"role": "user", "content": [{"type": "text", "text": "do a thing"}]}
        })));
        assert_eq!(m.entries().len(), 1);
        assert_eq!(m.entries()[0].role, ChatRole::User);
    }

    #[test]
    fn pending_tool_placed_and_settled_by_call_id() {
        let mut m = Messages::new();
        m.apply(&AppEvent::AgentStart);
        m.apply(&msg_update("let me look"));
        m.apply(&AppEvent::ToolExecutionStart(json!({
            "type": "tool_execution_start",
            "toolCallId": "c1",
            "toolName": "edit",
            "args": {"path": "src/lib.rs"}
        })));
        assert_eq!(m.entries().len(), 2);
        assert_eq!(m.finished_len(), 1); // assistant not settled, tool live
        m.apply(&AppEvent::ToolExecutionEnd(json!({
            "type": "tool_execution_end",
            "toolCallId": "c1",
            "toolName": "edit",
            "result": {"content": [{"type": "text", "text": "done"}]},
            "isError": false
        })));
        let ChatKind::Tool(card) = &m.entries()[1].kind else {
            panic!("expected tool card");
        };
        assert_eq!(card.status, ToolStatus::Ok);
        assert!(!card.partial);
        assert!(m.entries()[1].is_finished());
        m.apply(&msg_end("let me look"));
        assert_eq!(m.finished_len(), 2);
    }

    #[test]
    fn bash_result_details_reach_the_card_as_wall_and_timeout_line() {
        let mut m = Messages::new();
        m.apply(&AppEvent::AgentStart);
        m.apply(&AppEvent::ToolExecutionStart(json!({
            "type": "tool_execution_start",
            "toolCallId": "b1",
            "toolName": "bash",
            "args": {"command": "echo hi"}
        })));
        m.apply(&AppEvent::ToolExecutionEnd(json!({
            "type": "tool_execution_end",
            "toolCallId": "b1",
            "toolName": "bash",
            "result": {
                "content": [{"type": "text", "text": "hi\n\nWall time: 0.45 seconds"}],
                "details": {"wallTimeMs": 450, "timeoutSeconds": 300}
            },
            "isError": false
        })));
        let ChatKind::Bash(exec) = &m.entries()[0].kind else {
            panic!("expected bash execution");
        };
        assert_eq!(exec.details["wallTimeMs"], 450);
        let text: Vec<String> = m
            .entry_lines_at(0, 60)
            .iter()
            .map(|l| l.to_string())
            .collect();
        assert!(
            text.iter()
                .any(|r| r.contains("⟦Wall: 0.45s | Timeout: 300s⟧")),
            "{text:?}"
        );
        assert!(!text.iter().any(|r| r.contains("Wall time:")), "{text:?}");
    }

    #[test]
    fn consecutive_reads_fold_into_one_group() {
        let mut m = Messages::new();
        m.apply(&AppEvent::AgentStart);
        for (id, path) in [("r1", "a.rs"), ("r2", "b.rs")] {
            m.apply(&AppEvent::ToolExecutionStart(json!({
                "type": "tool_execution_start",
                "toolCallId": id,
                "toolName": "read",
                "args": {"path": path}
            })));
        }
        assert_eq!(m.entries().len(), 1);
        let ChatKind::ReadGroup(g) = &m.entries()[0].kind else {
            panic!("expected read group");
        };
        assert_eq!(g.reads.len(), 2);
        assert!(g.open);
        // A non-read tool displaces the group.
        m.apply(&AppEvent::ToolExecutionStart(json!({
            "type": "tool_execution_start",
            "toolCallId": "e1",
            "toolName": "edit",
            "args": {}
        })));
        assert_eq!(m.entries().len(), 2);
        // The displaced group no longer accepts reads.
        m.apply(&AppEvent::ToolExecutionStart(json!({
            "type": "tool_execution_start",
            "toolCallId": "r3",
            "toolName": "read",
            "args": {"path": "c.rs"}
        })));
        assert_eq!(m.entries().len(), 3);
    }

    #[test]
    fn clean_style_folds_searches_across_turns_into_one_group() {
        let mut m = Messages::new();
        let mut theme = Theme::default();
        theme.ui = crate::theme::UiStyle::Clean;
        m.set_theme(theme);
        m.apply(&AppEvent::AgentStart);
        let call = |m: &mut Messages, id: &str, name: &str, args: Value| {
            m.apply(&AppEvent::ToolExecutionStart(json!({
                "type": "tool_execution_start", "toolCallId": id, "toolName": name, "args": args
            })));
            m.apply(&AppEvent::ToolExecutionEnd(json!({
                "type": "tool_execution_end", "toolCallId": id, "toolName": name,
                "result": {"content": [{"type": "text", "text": "ok"}], "details": {"matchCount": 1, "fileCount": 1}}
            })));
        };
        call(&mut m, "g1", "grep", json!({"pattern": "beta"}));
        m.apply(&AppEvent::TurnEnd(json!({})));
        // The next turn's tool-call-only assistant message draws nothing.
        m.apply(&AppEvent::MessageStart(
            json!({"message": {"role": "assistant", "content": []}}),
        ));
        call(&mut m, "g2", "glob", json!({"pattern": "*.txt"}));
        call(&mut m, "r1", "read", json!({"path": "a.rs"}));
        let groups: Vec<&ReadGroup> = m
            .entries()
            .iter()
            .filter_map(|e| match &e.kind {
                ChatKind::ReadGroup(g) => Some(g),
                _ => None,
            })
            .collect();
        assert_eq!(
            groups.len(),
            1,
            "{:?}",
            m.entries().iter().map(|e| &e.text).collect::<Vec<_>>()
        );
        let tools: Vec<&str> = groups[0].reads.iter().map(|r| r.tool.as_str()).collect();
        assert_eq!(tools, ["grep", "glob", "read"]);
        assert_eq!(groups[0].reads[1].path, "*.txt");
        assert!(groups[0].open, "kept open across the soft turn end");
        // Visible text after the group closes it (and lets it commit).
        m.apply(&AppEvent::MessageUpdate(json!({"message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}})));
        let group = m
            .entries()
            .iter()
            .find(|e| matches!(e.kind, ChatKind::ReadGroup(_)))
            .unwrap();
        assert!(group.is_finished());
        // Collapsed render: one rail row naming every item.
        let rows: Vec<String> = blocks::entry_lines(group, false, 80, m.theme())
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert_eq!(rows, ["", "● Explored grep beta, glob *.txt, read a.rs"]);
        let rows = blocks::entry_lines(group, true, 80, m.theme());
        assert_eq!(rows.len(), 5, "header + one row per item");
    }

    #[test]
    fn read_group_collects_usage_on_flush() {
        let mut m = Messages::new();
        m.apply(&AppEvent::AgentStart);
        m.apply(&msg_update("reading"));
        m.apply(&AppEvent::ToolExecutionStart(json!({
            "type": "tool_execution_start", "toolCallId": "r1",
            "toolName": "read", "args": {"path": "a.rs"}
        })));
        m.apply(&msg_end("reading"));
        m.apply(&AppEvent::TurnEnd(json!({"type": "turn_end"})));
        let last = m.entries().last().unwrap();
        let ChatKind::ReadGroup(g) = &last.kind else {
            panic!("usage should attach under the read group, got {last:?}");
        };
        assert!(g.usages.is_empty()); // no usage in the message; group still last
    }

    fn usage_on() -> Messages {
        let mut m = Messages::new();
        m.set_display_prefs(DisplayPrefs {
            show_token_usage: true,
            ..DisplayPrefs::default()
        });
        m
    }

    fn billed_end(content: &Value, stop: &str) -> AppEvent {
        AppEvent::MessageEnd(json!({
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": content,
                "usage": {"input": 100, "output": 7, "cacheRead": 0, "cacheWrite": 0},
                "stopReason": stop,
                "timestamp": 1_792_111_302_143_i64,
                "duration": 41.2,
                "ttft": 40.7,
                "completedAt": 1_792_111_302_223_i64
            }
        }))
    }

    #[test]
    fn usage_rows_off_by_default() {
        let mut m = Messages::new();
        m.apply(&AppEvent::AgentStart);
        m.apply(&billed_end(
            &json!([{"type": "text", "text": "hi"}]),
            "stop",
        ));
        m.apply(&AppEvent::TurnEnd(json!({"type": "turn_end"})));
        assert!(!m
            .entries()
            .iter()
            .any(|e| matches!(e.kind, ChatKind::Usage(_))));
    }

    #[test]
    fn usage_row_follows_each_billed_message() {
        let mut m = usage_on();
        m.apply(&AppEvent::AgentStart);
        m.apply(&billed_end(
            &json!([{"type": "text", "text": "hi"}]),
            "stop",
        ));
        // Placed at message_end (TS), not deferred to the turn end.
        let last = m.entries().last().unwrap();
        let ChatKind::Usage(u) = &last.kind else {
            panic!("expected usage row, got {last:?}");
        };
        assert_eq!(
            (u.input, u.output, u.ttft_ms, u.duration_ms),
            (100, 7, Some(41), Some(41))
        );
        assert_eq!(u.timestamp_ms, Some(1_792_111_302_143));
        assert_eq!(m.finished_len(), m.entries().len());
    }

    /// TS tool turn: the reads-only message's row nests under the read
    /// group once its card exists; the final answer gets a standalone row.
    #[test]
    fn read_turn_usage_nests_under_group_then_answer_row() {
        let mut m = usage_on();
        m.apply(&AppEvent::AgentStart);
        let call = json!([{"type": "toolCall", "id": "r1", "name": "read", "arguments": {"path": "a.rs"}}]);
        m.apply(&billed_end(&call, "toolUse"));
        m.apply(&AppEvent::ToolExecutionStart(json!({
            "type": "tool_execution_start", "toolCallId": "r1",
            "toolName": "read", "args": {"path": "a.rs"}
        })));
        m.apply(&AppEvent::ToolExecutionEnd(json!({
            "type": "tool_execution_end", "toolCallId": "r1", "toolName": "read",
            "result": {"content": [{"type": "text", "text": "x"}]}, "isError": false
        })));
        m.apply(&billed_end(
            &json!([{"type": "text", "text": "done"}]),
            "stop",
        ));
        m.apply(&AppEvent::TurnEnd(json!({"type": "turn_end"})));
        let kinds: Vec<&ChatKind> = m.entries().iter().map(|e| &e.kind).collect();
        let ChatKind::ReadGroup(g) = kinds[0] else {
            panic!("{kinds:?}");
        };
        assert_eq!(g.usages.len(), 1);
        assert_eq!(g.usages[0].0, "r1");
        assert!(matches!(kinds[1], ChatKind::Assistant(_)));
        assert!(matches!(kinds[2], ChatKind::Usage(_)));
        assert_eq!(kinds.len(), 3);
    }

    /// A non-read tool keeps the standalone row, below the tool card.
    #[test]
    fn tool_turn_usage_row_goes_below_its_card() {
        let mut m = usage_on();
        m.apply(&AppEvent::AgentStart);
        let call = json!([{"type": "toolCall", "id": "b1", "name": "bash", "arguments": {"command": "ls"}}]);
        m.apply(&billed_end(&call, "toolUse"));
        assert!(!m
            .entries()
            .iter()
            .any(|e| matches!(e.kind, ChatKind::Usage(_))));
        m.apply(&AppEvent::ToolExecutionStart(json!({
            "type": "tool_execution_start", "toolCallId": "b1",
            "toolName": "bash", "args": {"command": "ls"}
        })));
        let kinds: Vec<&ChatKind> = m.entries().iter().map(|e| &e.kind).collect();
        assert!(
            matches!(kinds[kinds.len() - 2], ChatKind::Bash(_)),
            "{kinds:?}"
        );
        assert!(
            matches!(kinds[kinds.len() - 1], ChatKind::Usage(_)),
            "{kinds:?}"
        );
    }

    #[test]
    fn turn_time_measures_prompt_to_completion() {
        let mut m = Messages::new();
        m.set_display_prefs(DisplayPrefs {
            show_token_usage: true,
            show_turn_time: true,
            read_preview: false,
        });
        m.apply(&AppEvent::MessageEnd(json!({
            "type": "message_end",
            "message": {"role": "user", "content": "go", "timestamp": 1_792_111_300_000_i64}
        })));
        m.apply(&AppEvent::AgentStart);
        m.apply(&billed_end(
            &json!([{"type": "text", "text": "hi"}]),
            "stop",
        ));
        let ChatKind::Usage(u) = &m.entries().last().unwrap().kind else {
            panic!("expected usage row");
        };
        assert_eq!(u.turn_elapsed_ms, Some(2_223));
    }

    /// TS pinned banner: a turn-ending provider error is pinned (inline
    /// block suppressed, entry held live) until the next run starts; then
    /// it is restored inline and commits.
    #[test]
    fn provider_error_pins_until_next_run() {
        let mut m = Messages::new();
        m.apply(&AppEvent::AgentStart);
        m.apply(&AppEvent::MessageEnd(json!({
            "type": "message_end",
            "message": {
                "role": "assistant", "content": [],
                "stopReason": "error", "errorMessage": "400 E2E-PROVIDER-FAILURE"
            }
        })));
        m.apply(&AppEvent::AgentEnd(json!({"type": "agent_end"})));
        assert_eq!(m.pinned_error(), Some("400 E2E-PROVIDER-FAILURE"));
        let idx = m.entries().len() - 1;
        assert!(
            m.entry_lines_at(idx, 80).is_empty(),
            "inline error suppressed while pinned"
        );
        assert_eq!(m.finished_len(), idx, "pinned entry stays live");
        m.push_user("after failure");
        m.apply(&AppEvent::AgentStart);
        assert_eq!(m.pinned_error(), None);
        let rows: Vec<String> = m
            .entry_lines_at(idx, 80)
            .iter()
            .map(ToString::to_string)
            .collect();
        assert!(
            rows.iter().any(|r| r == " Error: 400 E2E-PROVIDER-FAILURE"),
            "{rows:?}"
        );
        assert_eq!(m.finished_len(), m.entries().len());
    }

    /// Regression: a successful turn printed `⏱ … this turn` twice and then
    /// `✗ Error: prompt failed` (the TUI read a `success` field the runtime
    /// never sends; the TS frame carries `status`).
    #[test]
    fn successful_turn_shows_one_usage_row_and_no_error() {
        let mut m = usage_on();
        m.apply(&AppEvent::AgentStart);
        m.apply(&AppEvent::MessageEnd(json!({
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Привет!"}],
                "usage": {"input": 2, "output": 15, "cacheRead": 0, "cacheWrite": 16_541},
                "stopReason": "stop"
            }
        })));
        m.apply(&AppEvent::TurnEnd(json!({"type": "turn_end"})));
        m.apply(&AppEvent::AgentEnd(json!({"type": "agent_end"})));
        m.apply(&crate::events::parse_frame(&json!({
            "type": "prompt_result", "agentInvoked": true,
            "status": "completed", "sessionSettled": true
        })));
        m.apply(&AppEvent::SessionSettled);
        let text: String = (0..m.entries().len())
            .flat_map(|i| m.entry_lines_at(i, 100))
            .map(|l| l.to_string() + "\n")
            .collect();
        assert_eq!(text.matches("⤵").count(), 1, "{text}");
        assert!(!text.contains("Error"), "{text}");

        // An agent-run failure is already on the assistant block: no extra row.
        let before = m.entries().len();
        m.apply(&crate::events::parse_frame(&json!({
            "type": "prompt_result", "agentInvoked": true, "status": "error",
            "error": {"message": "429 rate limited", "retryable": true}, "sessionSettled": true
        })));
        assert_eq!(m.entries().len(), before);
        // A failure that never reached the agent gets one row with its text.
        m.apply(&crate::events::parse_frame(&json!({
            "type": "prompt_result", "agentInvoked": false, "status": "error",
            "error": {"message": "Unknown command: /zz", "retryable": false}, "sessionSettled": true
        })));
        let last = m.entries().len() - 1;
        let row: String = m
            .entry_lines_at(last, 100)
            .iter()
            .map(ToString::to_string)
            .collect();
        assert!(row.contains("Unknown command: /zz"), "{row}");
    }

    #[test]
    fn bash_execution_tracks_output_and_exit() {
        let mut m = Messages::new();
        m.apply(&AppEvent::ToolExecutionStart(json!({
            "type": "tool_execution_start",
            "toolCallId": "b1",
            "toolName": "bash",
            "args": {"command": "cargo test", "excludeFromContext": true}
        })));
        let ChatKind::Bash(e) = &m.entries()[0].kind else {
            panic!("expected bash entry");
        };
        assert_eq!(e.status, ExecStatus::Running);
        assert!(e.excluded);
        m.apply(&AppEvent::ToolExecutionEnd(json!({
            "type": "tool_execution_end",
            "toolCallId": "b1",
            "toolName": "bash",
            "result": {"content": [{"type": "text", "text": "test result: ok"}]},
            "isError": false
        })));
        let ChatKind::Bash(e) = &m.entries()[0].kind else {
            panic!("expected bash entry");
        };
        assert_eq!(e.status, ExecStatus::Complete);
        assert_eq!(e.output, "test result: ok");
        assert!(m.entries()[0].is_finished());
    }

    #[test]
    fn rebuild_dispatches_all_message_roles() {
        let mut m = Messages::new();
        m.rebuild(&[
            json!({"role": "user", "content": [{"type": "text", "text": "hi"}]}),
            json!({"role": "assistant", "content": [{"type": "text", "text": "hello"}],
                   "usage": {"input": 10, "output": 5}, "stopReason": "stop"}),
            json!({"role": "bashExecution", "command": "ls", "output": "a\nb", "exitCode": 0}),
            json!({"role": "compactionSummary", "method": "soft",
                    "tokensBefore": 256_000, "tokensAfter": 20_000, "summary": "we talked"}),
            json!({"role": "branchSummary", "summary": "side quest"}),
            json!({"role": "custom", "customType": "advisor", "content": "note",
                    "display": true, "attribution": "agent"}),
        ]);
        assert_eq!(m.entries().len(), 6);
        assert!(matches!(m.entries()[2].kind, ChatKind::Bash(_)));
        let ChatKind::Compaction(c) = &m.entries()[3].kind else {
            panic!("expected compaction");
        };
        assert_eq!(c.label(), "soft-compacted");
        assert!(matches!(m.entries()[4].kind, ChatKind::Branch { .. }));
        let ChatKind::Custom(c) = &m.entries()[5].kind else {
            panic!("expected custom");
        };
        assert_eq!(c.custom_type, "advisor");
        assert!(c.display);
        assert_eq!(m.finished_len(), 6); // everything finished after rebuild
    }

    fn resume_history(result_error: bool) -> Vec<Value> {
        vec![
            json!({"role": "user", "content": [{"type": "text", "text": "run it"}]}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "toolCall", "id": "c1", "name": "bash",
                 "arguments": {"command": "echo hi"}}
            ], "stopReason": "toolUse"}),
            json!({"role": "toolResult", "toolCallId": "c1", "toolName": "bash",
                   "content": [{"type": "text", "text": "hi-output"}], "isError": result_error}),
            json!({"role": "assistant", "content": [{"type": "text", "text": "done"}],
                   "stopReason": "stop"}),
        ]
    }

    #[test]
    fn rebuild_attaches_tool_result_to_call() {
        let mut m = Messages::new();
        m.rebuild(&resume_history(false));
        let kinds: Vec<_> = m.entries().iter().map(|e| e.role).collect();
        assert_eq!(
            kinds,
            [
                ChatRole::User,
                ChatRole::Assistant,
                ChatRole::Tool,
                ChatRole::Assistant
            ]
        );
        // thinking renders as the settled assistant segment, no bare row
        let ChatKind::Assistant(a) = &m.entries()[1].kind else {
            panic!("expected assistant");
        };
        assert!(matches!(a.segments[0], AssistantSegment::Thinking { .. }));
        let ChatKind::Bash(e) = &m.entries()[2].kind else {
            panic!("expected bash");
        };
        assert_eq!(e.command, "echo hi");
        assert_eq!(e.status, ExecStatus::Complete);
        assert!(e.output.contains("hi-output"));
        assert!(m.entries()[2].is_finished());
        assert_eq!(m.entries()[3].text, "done");
        assert!(m.entries().iter().all(|e| e.text != "[assistant]"));
        assert_eq!(m.finished_len(), 4);
    }

    #[test]
    fn rebuild_error_result_marks_error_and_generic_tool_card() {
        let mut m = Messages::new();
        m.rebuild(&resume_history(true));
        let ChatKind::Bash(e) = &m.entries()[2].kind else {
            panic!("expected bash");
        };
        assert_eq!(e.status, ExecStatus::Error);

        let mut m = Messages::new();
        m.rebuild(&[
            json!({"role": "assistant", "content": [
                {"type": "toolCall", "id": "w1", "name": "web_search", "arguments": {"query": "x"}}
            ]}),
            json!({"role": "toolResult", "toolCallId": "w1", "toolName": "web_search",
                   "content": [{"type": "text", "text": "boom"}], "isError": true}),
        ]);
        assert_eq!(m.entries().len(), 1); // toolCall-only assistant: no bare row
        let ChatKind::Tool(card) = &m.entries()[0].kind else {
            panic!("expected tool card");
        };
        assert_eq!(card.status, ToolStatus::Error);
        assert_eq!(card.result_text.as_deref(), Some("boom"));
        assert!(m.entries()[0].is_finished());
    }

    #[test]
    fn rebuild_orphan_tool_result_stays_visible() {
        let mut m = Messages::new();
        m.rebuild(&[
            json!({"role": "toolResult", "toolCallId": "zz", "toolName": "grep",
                           "content": [{"type": "text", "text": "lonely"}]}),
        ]);
        assert_eq!(m.entries().len(), 1);
        let ChatKind::Tool(card) = &m.entries()[0].kind else {
            panic!("expected tool card");
        };
        assert_eq!(card.status, ToolStatus::Ok);
        assert_eq!(card.result_text.as_deref(), Some("lonely"));
    }

    #[test]
    fn compaction_label_mapping() {
        assert_eq!(
            Compaction {
                method: "remote".into(),
                tokens_before: 0,
                tokens_after: None,
                summary: String::new(),
                warning: None
            }
            .label(),
            "remote-compacted"
        );
        assert_eq!(
            Compaction {
                method: "handoff".into(),
                tokens_before: 0,
                tokens_after: None,
                summary: String::new(),
                warning: None
            }
            .label(),
            "handed-off"
        );
        assert_eq!(
            Compaction {
                method: "legacy".into(),
                tokens_before: 0,
                tokens_after: None,
                summary: String::new(),
                warning: None
            }
            .label(),
            "compacted"
        );
    }

    #[test]
    fn committed_rows_advance_watermark() {
        let mut m = Messages::new();
        m.push_user("hi");
        let rows = m.take_committed_rows(80);
        assert!(!rows.is_empty());
        assert!(rows.iter().any(|r| r.to_string().contains("hi")));
        // Nothing new: no rows.
        assert!(m.take_committed_rows(80).is_empty());
        // Streaming entry is not committed until settled.
        m.apply(&AppEvent::AgentStart);
        m.apply(&msg_update("working"));
        assert!(m.take_committed_rows(80).is_empty());
        m.apply(&msg_end("working"));
        assert!(!m.take_committed_rows(80).is_empty());
    }

    #[test]
    fn all_rows_covers_everything_and_resets_watermark() {
        let mut m = Messages::new();
        m.push_user("one");
        m.apply(&AppEvent::AgentStart);
        m.apply(&msg_update("two"));
        let rows = m.all_rows(80);
        assert!(rows.iter().any(|r| r.to_string().contains("one")));
        assert!(rows.iter().any(|r| r.to_string().contains("two")));
        // Live tail not committed; finished prefix is.
        assert_eq!(m.committed, m.finished_len());
        assert!(m.take_committed_rows(80).is_empty());
    }

    #[test]
    fn committed_rows_render_through_block_renderer() {
        // Error notices wrap at the block width through format_error_block.
        let rows = blocks::entry_lines(
            &ChatEntry::new(
                ChatRole::System,
                "error: alpha beta gamma delta epsilon zeta".to_owned(),
                ChatKind::Notice("error: alpha beta gamma delta epsilon zeta".to_owned()),
                true,
            ),
            false,
            20,
            &crate::theme::Theme::default(),
        );
        assert!(rows.len() > 1);
        assert!(rows[0].to_string().contains("Error:"));
        // Continuation rows hang under the prefix within the width budget.
        assert!(rows[1..].iter().all(|r| r.to_string().len() <= 20));
    }

    #[test]
    fn push_system_adds_finished_notice() {
        let mut m = Messages::new();
        m.push_system("unknown command /x");
        assert_eq!(m.entries().len(), 1);
        assert!(matches!(m.entries()[0].kind, ChatKind::Notice(_)));
        assert_eq!(m.finished_len(), 1);
    }

    #[test]
    fn ctrl_o_toggles_transcript_expansion() {
        let mut m = Messages::new();
        assert!(!m.expanded());
        m.toggle_expanded();
        assert!(m.expanded());
        m.set_expanded(false);
        assert!(!m.expanded());
    }

    #[test]
    fn settled_bash_tool_renders_family_card_rows() {
        let mut m = Messages::new();
        m.apply(&AppEvent::AgentStart);
        m.apply(&AppEvent::ToolExecutionStart(json!({
            "type": "tool_execution_start", "toolCallId": "b1",
            "toolName": "bash", "args": {"command": "cargo test -p omp-tui"}
        })));
        m.apply(&AppEvent::ToolExecutionEnd(json!({
            "type": "tool_execution_end", "toolCallId": "b1",
            "toolName": "bash",
            "result": {"text": "test result: ok. 7 passed"}
        })));
        m.apply(&AppEvent::AgentEnd(json!({"type": "agent_end"})));
        let rows = m.take_committed_rows(80);
        assert!(rows.len() > 1, "bash card must render multiple rows");
        let joined = rows
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("cargo test -p omp-tui"), "{joined}");
        assert!(joined.contains("7 passed"), "{joined}");
    }

    #[test]
    fn aborted_tool_is_sealed_at_turn_end() {
        let mut m = Messages::new();
        m.apply(&AppEvent::AgentStart);
        m.apply(&AppEvent::ToolExecutionStart(json!({
            "type": "tool_execution_start", "toolCallId": "t1",
            "toolName": "web_search", "args": {"query": "x"}
        })));
        assert_eq!(m.finished_len(), 0);
        m.apply(&AppEvent::AgentEnd(json!({"type": "agent_end"})));
        assert_eq!(m.finished_len(), 1);
        let ChatKind::Tool(card) = &m.entries()[0].kind else {
            panic!("expected tool card");
        };
        assert!(!card.partial);
    }
}
