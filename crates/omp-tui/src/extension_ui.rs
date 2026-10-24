//! Extension UI surface — ports of the extension-host UI hooks in
//! `pi-coding-agent/src/modes/controllers/extension-ui-controller.ts`
//! (`confirm`, `select`, `input`, `notify`, ask dialogs) consuming
//! `extension_ui_request` frames.
//!
//! Two frame shapes arrive (see `crates/omp-cli/src/rpc.rs` `RpcAskUi` and
//! `runtime.rs` `send_extension_frame`):
//!
//! - **Dialogs** (`{"type":"extension_ui_request","id":"ui-…","method":…}`):
//!   the ask-tool RPC sink (`method: "ask"`), plus `select`/`confirm`/
//!   `input`/`editor`. Exactly one dialog is presented at a time (TS
//!   `#dialogActive` + `#dialogQueue`); the rest queue in arrival order.
//!   Every dialog is abortable (Esc → `{"cancelled": true}`), and a `timeout`
//!   (ms) auto-cancels with `timedOut` like TS `CountdownTimer`.
//! - **Side effects** (no `id`, fields under `params`): `ui_notify`,
//!   `setStatus`, `setWorkingMessage`, `setWidget` — stored here and read by
//!   the status line / chrome owners via the getters below.
//!
//! Answers go back through [`RuntimePort::extension_ui_response`] as the
//! response *object* (`answers` / `value` / `cancelled`); the embedding
//! binary merges it into the `extension_ui_response` envelope (omp-cli
//! `tui.rs`), which is what `tools/ask.rs` validates.

use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::port::RuntimePort;
use crate::theme::Theme;
use ratatui::text::{Line, Span};

/// One selectable option (`omp-cli/src/tools/ask.rs` `SelectOption`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DialogOption {
    /// Primary label (answer value).
    pub label: String,
    /// Dim secondary line, if any.
    pub description: Option<String>,
}

impl DialogOption {
    fn from_value(v: &Value) -> Option<Self> {
        if let Some(label) = v.as_str() {
            return Some(Self {
                label: label.to_owned(),
                description: None,
            });
        }
        let label = v.get("label").and_then(Value::as_str)?.to_owned();
        let description = v
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned);
        Some(Self { label, description })
    }
}

/// One question of an `ask` dialog (`tools/ask.rs` `QuestionItem`).
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    /// Stable id echoed back in the answer.
    pub id: String,
    /// Prompt text.
    pub text: String,
    /// Optional heading above the prompt.
    pub header: Option<String>,
    /// Selectable options.
    pub options: Vec<DialogOption>,
    /// Multi-select (space toggles, a done row commits).
    pub multi: bool,
    /// Recommended option index (label gains the omp suffix).
    pub recommended: Option<usize>,
}

impl Question {
    // `recommended` is a small option index validated by the bounds filter
    // below (omp clamps it onto the option list in ask.rs).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn from_value(v: &Value) -> Option<Self> {
        let id = v.get("id").and_then(Value::as_str)?.to_owned();
        let text = v
            .get("question")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let header = v.get("header").and_then(Value::as_str).map(str::to_owned);
        let options: Vec<DialogOption> = v
            .get("options")
            .and_then(Value::as_array)
            .map_or_else(Vec::new, |a| {
                a.iter().filter_map(DialogOption::from_value).collect()
            });
        let multi = v.get("multi").and_then(Value::as_bool).unwrap_or(false);
        let recommended = v
            .get("recommended")
            .and_then(Value::as_f64)
            .map(|f| f as usize)
            .filter(|i| *i < options.len());
        Some(Self {
            id,
            text,
            header,
            options,
            multi,
            recommended,
        })
    }

    /// Option labels with omp's " (Recommended)" suffix applied.
    fn display_labels(&self) -> Vec<String> {
        self.options
            .iter()
            .enumerate()
            .map(|(i, o)| {
                if self.recommended == Some(i) {
                    format!("{}{RECOMMENDED_SUFFIX}", o.label)
                } else {
                    o.label.clone()
                }
            })
            .collect()
    }
}

/// omp's recommended marker (`tools/ask.rs` `RECOMMENDED_SUFFIX`).
const RECOMMENDED_SUFFIX: &str = " (Recommended)";
/// omp's custom-answer row (`tools/ask.rs` `OTHER_OPTION_LABEL`).
const OTHER_OPTION_LABEL: &str = "Other (type your own)";
/// omp's multi-select commit row (`tools/ask.rs` `DONE_SELECTING_LABEL`).
const DONE_SELECTING_LABEL: &str = "✓ Done selecting";
/// Per-widget line cap (`extension-ui-controller.ts` `MAX_WIDGET_LINES`).
const MAX_WIDGET_LINES: usize = 10;

/// A persistent hook widget (`setWidget`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookWidget {
    /// Widget key (deduped by the runtime/extension name).
    pub key: String,
    /// Rendered content lines.
    pub lines: Vec<String>,
    /// Whether the widget was truncated to [`MAX_WIDGET_LINES`].
    pub truncated: bool,
}

/// A queued dialog request (`extension_ui_request` frame).
#[derive(Debug, Clone, PartialEq)]
pub struct PendingRequest {
    /// Frame id to answer with.
    pub id: String,
    /// Dialog method (`ask`, `select`, `confirm`, `input`, `editor`).
    pub method: String,
    /// Raw request payload (params or top-level fields).
    pub params: Value,
    /// Optional answer deadline (frame `timeout`, ms).
    pub timeout: Option<Duration>,
}

/// The dialog currently owning the screen.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveDialog {
    id: String,
    method: String,
    kind: DialogKind,
    deadline: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq)]
enum DialogKind {
    /// Multi-question ask form: questions answered in order, then submitted.
    Ask {
        questions: Vec<Question>,
        answers: Vec<Option<AskAnswer>>,
        /// Which question is on screen.
        current: usize,
        /// Cursor within the flattened option rows of `current`.
        cursor: usize,
        /// Checked option indices when the question is multi-select.
        checked: Vec<usize>,
        /// Buffer while typing a custom ("Other") answer.
        custom: Option<String>,
    },
    /// Single-option-picker (`select`, and degraded ask flows).
    Select {
        title: String,
        options: Vec<DialogOption>,
        /// Row within [`select_matches`] of `query` (the filtered view).
        cursor: usize,
        /// Type-to-search filter (only typed into when `searchable`).
        query: String,
        /// Printable keys edit `query` instead of `j`/`k`/digit navigation.
        searchable: bool,
        /// Drawn over the whole alternate screen (frame `fullscreen: true`,
        /// e.g. `/login`'s provider picker) instead of a centered popup.
        fullscreen: bool,
    },
    /// Yes/no modal.
    Confirm { title: String, message: String },
    /// Single-line text prompt.
    Input {
        title: String,
        placeholder: String,
        buffer: String,
    },
    /// Multi-line text prompt (Enter inserts a newline).
    Editor { title: String, buffer: String },
}

/// One answered question (validated by `tools/ask.rs` `parse_answers`).
#[derive(Debug, Clone, PartialEq)]
struct AskAnswer {
    id: String,
    selected_options: Vec<String>,
    custom_input: Option<String>,
}

impl AskAnswer {
    fn to_json(&self) -> Value {
        let mut m = serde_json::Map::new();
        m.insert("id".into(), Value::from(self.id.clone()));
        m.insert(
            "selectedOptions".into(),
            Value::Array(
                self.selected_options
                    .iter()
                    .map(|s| Value::from(s.as_str()))
                    .collect(),
            ),
        );
        if let Some(custom) = &self.custom_input {
            m.insert("customInput".into(), Value::from(custom.clone()));
        }
        Value::Object(m)
    }
}

/// `select` options: labels may be plain strings with descriptions in a
/// parallel `optionDetails` array (`rpc.rs` `AskUi::select`).
fn select_options(params: &Value) -> Vec<DialogOption> {
    let Some(options) = params.get("options").and_then(Value::as_array) else {
        return Vec::new();
    };
    let details = params
        .get("optionDetails")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    options
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut option = DialogOption::from_value(v).unwrap_or_else(|| DialogOption {
                label: String::new(),
                description: None,
            });
            if option.description.is_none() {
                option.description = details
                    .get(i)
                    .and_then(|d| d.get("description"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
            }
            option
        })
        .collect()
}

/// Indices of the `options` a type-to-search `query` keeps: every whitespace-separated term
/// must occur (case-insensitively) in the label or description. An empty query keeps all.
fn select_matches(options: &[DialogOption], query: &str) -> Vec<usize> {
    let terms: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
    options
        .iter()
        .enumerate()
        .filter(|(_, o)| {
            let hay = format!(
                "{} {}",
                o.label,
                o.description.as_deref().unwrap_or_default()
            )
            .to_lowercase();
            terms.iter().all(|t| hay.contains(t.as_str()))
        })
        .map(|(i, _)| i)
        .collect()
}

/// The slice of option rows that fits `avail` screen rows with the cursor row visible, roughly
/// centred (`heights[i]` = rows option `i` occupies). Returns `start..end` over `heights`.
fn select_window(heights: &[usize], cursor: usize, avail: usize) -> std::ops::Range<usize> {
    if heights.is_empty() {
        return 0..0;
    }
    let cursor = cursor.min(heights.len() - 1);
    let avail = avail.max(heights[cursor]);
    let (mut start, mut end, mut used) = (cursor, cursor + 1, heights[cursor]);
    // Up to half the space above the cursor, then fill below, then give any rest to the top.
    while start > 0 && used + heights[start - 1] <= avail / 2 + 1 {
        start -= 1;
        used += heights[start];
    }
    while end < heights.len() && used + heights[end] <= avail {
        used += heights[end];
        end += 1;
    }
    while start > 0 && used + heights[start - 1] <= avail {
        start -= 1;
        used += heights[start];
    }
    start..end
}

/// PageUp/PageDown step in a select list.
const SELECT_PAGE: usize = 10;

impl ActiveDialog {
    fn new(req: &PendingRequest) -> Self {
        let params = &req.params;
        let kind = match req.method.as_str() {
            "ask" => {
                let questions = params
                    .get("questions")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Question::from_value).collect())
                    .unwrap_or_default();
                DialogKind::Ask {
                    questions,
                    answers: Vec::new(),
                    current: 0,
                    cursor: 0,
                    checked: Vec::new(),
                    custom: None,
                }
            }
            "select" => {
                let flag = |k: &str| params.get(k).and_then(Value::as_bool) == Some(true);
                let fullscreen = flag("fullscreen");
                DialogKind::Select {
                    title: params
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    options: select_options(params),
                    cursor: params
                        .get("initialIndex")
                        .and_then(Value::as_u64)
                        .and_then(|i| usize::try_from(i).ok())
                        .unwrap_or(0),
                    query: String::new(),
                    searchable: fullscreen || flag("searchable"),
                    fullscreen,
                }
            }
            "confirm" => DialogKind::Confirm {
                title: params
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                message: params
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            },
            "editor" => DialogKind::Editor {
                title: params
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                // TS rpc `editor {prefill}`: the buffer starts with the text to edit.
                buffer: params
                    .get("prefill")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            },
            // "input" and unknown methods degrade to the single-line prompt.
            _ => DialogKind::Input {
                title: params
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                placeholder: params
                    .get("placeholder")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                buffer: String::new(),
            },
        };
        Self {
            id: req.id.clone(),
            method: req.method.clone(),
            kind,
            deadline: req.timeout.map(|d| Instant::now() + d),
        }
    }

    /// Remaining seconds until the deadline (for the title countdown).
    fn remaining_secs(&self) -> Option<u64> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()).as_secs())
    }
}

/// Pending extension dialogs (one active at a time) plus extension-owned
/// chrome (status segment, working message, widgets, last notification).
#[derive(Debug, Clone, Default)]
pub struct ExtensionUi {
    queue: VecDeque<PendingRequest>,
    active: Option<ActiveDialog>,
    /// The request the active dialog was built from (for `head()`).
    active_req: Option<PendingRequest>,
    /// `setStatus` segments keyed by extension key.
    status: BTreeMap<String, String>,
    /// `setWorkingMessage` loader text, if set.
    working_message: Option<String>,
    /// `setWidget` widgets shown above the editor.
    widgets_above: Vec<HookWidget>,
    /// `setWidget` widgets shown below the editor.
    widgets_below: Vec<HookWidget>,
    /// Most recent `ui_notify` toast, consumed once by the embedding loop.
    notification: Option<(String, String)>,
}

impl ExtensionUi {
    /// An empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Route one `extension_ui_request` frame: dialogs (frames with an `id`)
    /// queue; side-effect notifications apply immediately
    /// (TS `ExtensionUIContext.setStatus` and friends are synchronous).
    pub fn push(&mut self, frame: &Value) {
        let method = frame.get("method").and_then(Value::as_str).unwrap_or("");
        let id = frame.get("id").and_then(Value::as_str).unwrap_or("");
        if id.is_empty() {
            self.apply_side_effect(method, &payload_of(frame));
            return;
        }
        let timeout = frame
            .get("timeout")
            .and_then(Value::as_u64)
            .map(Duration::from_millis);
        self.queue.push_back(PendingRequest {
            id: id.to_owned(),
            method: method.to_owned(),
            params: payload_of(frame),
            timeout,
        });
        if self.active.is_none() {
            self.activate_next();
        }
    }

    /// The oldest unanswered request, if any (the active dialog first).
    #[must_use]
    pub fn head(&self) -> Option<&PendingRequest> {
        self.active_req.as_ref().or_else(|| self.queue.front())
    }

    /// Remove the request with `id` (after answering it) and promote the
    /// next queued dialog.
    pub fn complete(&mut self, id: &str) {
        if self.active.as_ref().is_some_and(|a| a.id == id) {
            self.active = None;
            self.active_req = None;
            self.activate_next();
        }
        self.queue.retain(|r| r.id != id);
    }

    /// Whether a dialog currently owns the screen and keyboard.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    /// Whether the active dialog is a full-screen picker (frame `fullscreen: true`): the shell
    /// holds the alternate screen for it like a full-screen overlay.
    #[must_use]
    pub fn wants_fullscreen(&self) -> bool {
        matches!(
            self.active.as_ref().map(|d| &d.kind),
            Some(DialogKind::Select {
                fullscreen: true,
                ..
            })
        )
    }

    /// Number of unanswered dialogs (active + queued).
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.queue.len() + usize::from(self.active.is_some())
    }

    /// `setStatus` segments as `key=text` pairs, in key order.
    #[must_use]
    pub fn status_segments(&self) -> Vec<(String, String)> {
        self.status
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// `setWorkingMessage` loader text, if any.
    #[must_use]
    pub fn working_message(&self) -> Option<&str> {
        self.working_message.as_deref()
    }

    /// Widgets destined for the editor's top gap.
    #[must_use]
    pub fn widgets_above(&self) -> &[HookWidget] {
        &self.widgets_above
    }

    /// Widgets destined for below the editor.
    #[must_use]
    pub fn widgets_below(&self) -> &[HookWidget] {
        &self.widgets_below
    }

    /// Take the most recent `ui_notify` toast (`(type, message)`); consumed
    /// once by the embedding loop so it is not shown repeatedly.
    pub fn take_notification(&mut self) -> Option<(String, String)> {
        self.notification.take()
    }

    /// Collect timed-out dialogs: each expired active dialog is removed
    /// (the queue advances) and returned as `(id, {"cancelled": true,
    /// "timedOut": true})` (TS `CountdownTimer` → onTimeout). The caller
    /// forwards each pair through `RuntimePort::extension_ui_response`.
    /// Call once per tick from the controller loop.
    pub fn poll_deadlines(&mut self, now: Instant) -> Vec<(String, Value)> {
        let due = self
            .active
            .as_ref()
            .and_then(|a| a.deadline)
            .is_some_and(|d| d <= now);
        // Only a due dialog leaves: taking first would drop every dialog that has no
        // deadline (or is not yet due) on the next tick, leaving it drawn but deaf to keys.
        if !due {
            return Vec::new();
        }
        let Some(dialog) = self.active.take() else {
            return Vec::new();
        };
        self.active_req = None;
        self.activate_next();
        vec![(dialog.id, cancelled_value(true))]
    }

    /// Promote the oldest queued request to the active dialog.
    fn activate_next(&mut self) {
        self.active_req = self.queue.pop_front();
        self.active = self.active_req.as_ref().map(ActiveDialog::new);
    }

    /// Apply a side-effect notification (frames without an `id`).
    fn apply_side_effect(&mut self, method: &str, params: &Value) {
        match method {
            // The host withdrew a dialog it no longer waits for (e.g. `/login`'s paste prompt
            // once the browser callback won): drop it unanswered.
            "cancelDialog" | "cancel_dialog" => {
                if let Some(id) = params.get("id").and_then(Value::as_str) {
                    let id = id.to_owned();
                    self.complete(&id);
                }
            }
            "ui_notify" | "notify" => {
                let kind = params
                    .get("type")
                    .or_else(|| params.get("level"))
                    .and_then(Value::as_str)
                    .unwrap_or("info")
                    .to_owned();
                let message = params
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                self.notification = Some((kind, message));
            }
            "setStatus" | "set_status" => {
                let key = params
                    .get("key")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                if let Some(text) = params.get("text").and_then(Value::as_str) {
                    self.status.insert(key, text.to_owned());
                }
            }
            "setWorkingMessage" | "set_working_message" => {
                self.working_message = params
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
            }
            "setWidget" | "set_widget" => {
                let key = params
                    .get("key")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let content = params
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let above = params
                    .get("position")
                    .and_then(Value::as_str)
                    .is_none_or(|p| p != "below");
                let mut lines: Vec<String> = content.lines().map(str::to_owned).collect();
                let truncated = lines.len() > MAX_WIDGET_LINES;
                lines.truncate(MAX_WIDGET_LINES);
                let widget = HookWidget {
                    key,
                    lines,
                    truncated,
                };
                let slot = if above {
                    &mut self.widgets_above
                } else {
                    &mut self.widgets_below
                };
                if let Some(existing) = slot.iter_mut().find(|w| w.key == widget.key) {
                    *existing = widget;
                } else {
                    slot.push(widget);
                }
            }
            _ => {} // unknown side effects are ignored, never fatal
        }
    }
}

/// The dialog payload: `params` when present, else the frame minus the
/// envelope keys (`RpcAskUi` puts `questions`/`timeout` at the top level).
const ENVELOPE_KEYS: [&str; 4] = ["type", "id", "method", "extension"];

fn payload_of(frame: &Value) -> Value {
    if let Some(params) = frame.get("params").filter(|p| p.is_object()) {
        return params.clone();
    }
    match frame {
        Value::Object(m) => Value::Object(
            m.iter()
                .filter(|(k, _)| !ENVELOPE_KEYS.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `{"cancelled": true}` (with `timedOut` when the deadline fired).
fn cancelled_value(timed_out: bool) -> Value {
    serde_json::json!({"cancelled": true, "timedOut": timed_out})
}

/// Interactive dialog handling plus modal rendering.
impl ExtensionUi {
    /// Whether a dialog is open and should own keyboard focus before the
    /// editor (TS `#dialogActive` gates editor input).
    #[must_use]
    pub fn wants_focus(&self) -> bool {
        self.active.is_some()
    }

    /// Handle a key event while a dialog is active. Answers are sent through
    /// `port.extension_ui_response`; `Esc` aborts the active dialog
    /// (`{"cancelled": true}`) and promotes the next queued one.
    pub fn handle_key(&mut self, key: KeyEvent, port: &impl RuntimePort) -> bool {
        let Some(dialog) = self.active.as_mut() else {
            return false;
        };
        let id = dialog.id.clone();
        let outcome = match &mut dialog.kind {
            DialogKind::Confirm { .. } => Self::key_confirm(&mut dialog.kind, key),
            DialogKind::Input { .. } | DialogKind::Editor { .. } => Self::key_text(dialog, key),
            DialogKind::Select { .. } => Self::key_select(dialog, key),
            DialogKind::Ask { .. } => Self::key_ask(dialog, key),
        };
        match outcome {
            KeyResult::Consumed => true,
            KeyResult::Answer(value) => {
                port.extension_ui_response(id.clone(), value);
                self.complete(&id);
                true
            }
            KeyResult::Cancel => {
                port.extension_ui_response(id.clone(), cancelled_value(false));
                self.complete(&id);
                true
            }
        }
    }

    fn key_confirm(kind: &mut DialogKind, key: KeyEvent) -> KeyResult {
        let DialogKind::Confirm { .. } = kind else {
            return KeyResult::Consumed;
        };
        match key.code {
            KeyCode::Enter | KeyCode::Char('y' | 'Y') => {
                KeyResult::Answer(serde_json::json!({"value": true}))
            }
            KeyCode::Esc | KeyCode::Char('n' | 'N') => {
                KeyResult::Answer(serde_json::json!({"value": false}))
            }
            _ => KeyResult::Consumed, // swallow keys while the modal is open
        }
    }

    fn key_text(dialog: &mut ActiveDialog, key: KeyEvent) -> KeyResult {
        let multiline = matches!(dialog.kind, DialogKind::Editor { .. });
        let (DialogKind::Input { buffer, .. } | DialogKind::Editor { buffer, .. }) =
            &mut dialog.kind
        else {
            return KeyResult::Consumed;
        };
        match key.code {
            KeyCode::Esc => KeyResult::Cancel,
            KeyCode::Enter if multiline && !key.modifiers.contains(KeyModifiers::CONTROL) => {
                buffer.push('\n');
                KeyResult::Consumed
            }
            KeyCode::Enter => KeyResult::Answer(serde_json::json!({"value": buffer.clone()})),
            KeyCode::Backspace => {
                buffer.pop();
                KeyResult::Consumed
            }
            KeyCode::Char(c) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+Enter submits the editor dialog; other ctrl chords
                // are swallowed so they cannot leak to the editor below.
                if c == '\r' || c == '\n' || c == 'j' {
                    return KeyResult::Answer(serde_json::json!({"value": buffer.clone()}));
                }
                KeyResult::Consumed
            }
            KeyCode::Char(c) => {
                buffer.push(c);
                KeyResult::Consumed
            }
            _ => KeyResult::Consumed,
        }
    }

    fn key_select(dialog: &mut ActiveDialog, key: KeyEvent) -> KeyResult {
        let DialogKind::Select {
            options,
            cursor,
            query,
            searchable,
            ..
        } = &mut dialog.kind
        else {
            return KeyResult::Consumed;
        };
        let matches = select_matches(options, query);
        let last = matches.len().saturating_sub(1);
        let typing = *searchable
            && !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
        match key.code {
            KeyCode::Esc => return KeyResult::Cancel,
            KeyCode::Enter => {
                return matches.get(*cursor).map_or(KeyResult::Consumed, |&i| {
                    KeyResult::Answer(serde_json::json!({"value": options[i].label.clone()}))
                });
            }
            KeyCode::Up => *cursor = cursor.saturating_sub(1).min(last),
            KeyCode::Down => *cursor = (*cursor + 1).min(last),
            KeyCode::PageUp => *cursor = cursor.saturating_sub(SELECT_PAGE).min(last),
            KeyCode::PageDown => *cursor = (*cursor + SELECT_PAGE).min(last),
            KeyCode::Home => *cursor = 0,
            KeyCode::End => *cursor = last,
            KeyCode::Backspace if *searchable => {
                query.pop();
                *cursor = 0;
            }
            KeyCode::Char(c) if typing => {
                query.push(c);
                *cursor = 0;
            }
            KeyCode::Char('k') => *cursor = cursor.saturating_sub(1).min(last),
            KeyCode::Char('j') => *cursor = (*cursor + 1).min(last),
            KeyCode::Char(c) if c.is_ascii_digit() => {
                let i = c.to_digit(10).expect("ascii digit") as usize;
                if (1..=matches.len()).contains(&i) {
                    *cursor = i - 1;
                }
            }
            _ => {}
        }
        KeyResult::Consumed
    }

    fn key_ask(dialog: &mut ActiveDialog, key: KeyEvent) -> KeyResult {
        let DialogKind::Ask {
            questions,
            answers,
            current,
            cursor,
            checked,
            custom,
        } = &mut dialog.kind
        else {
            return KeyResult::Consumed;
        };
        let Some(question) = questions.get(*current).cloned() else {
            return KeyResult::Cancel; // malformed ask: release the caller
        };

        // Custom-answer entry mode ("Other (type your own)").
        if let Some(buffer) = custom {
            return match key.code {
                KeyCode::Esc => {
                    *custom = None;
                    KeyResult::Consumed
                }
                KeyCode::Enter => {
                    let answer = AskAnswer {
                        id: question.id.clone(),
                        selected_options: Vec::new(),
                        custom_input: Some(buffer.trim().to_owned()),
                    };
                    Self::ask_advance(
                        &question, answer, questions, answers, current, cursor, checked, custom,
                    )
                }
                KeyCode::Backspace => {
                    buffer.pop();
                    KeyResult::Consumed
                }
                KeyCode::Char(c) => {
                    buffer.push(c);
                    KeyResult::Consumed
                }
                _ => KeyResult::Consumed,
            };
        }

        let labels = question.display_labels();
        let rows = labels.len() + 1;
        match key.code {
            KeyCode::Esc => KeyResult::Cancel,
            KeyCode::Up | KeyCode::Char('k') => {
                *cursor = cursor.saturating_sub(1).min(rows - 1);
                KeyResult::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                *cursor = (*cursor + 1).min(rows - 1);
                KeyResult::Consumed
            }
            KeyCode::Char(' ') if question.multi => {
                if *cursor < labels.len() {
                    if let Some(pos) = checked.iter().position(|i| *i == *cursor) {
                        checked.remove(pos);
                    } else {
                        checked.push(*cursor);
                    }
                }
                KeyResult::Consumed
            }
            KeyCode::Enter => {
                if question.multi {
                    if *cursor == labels.len() {
                        // "✓ Done selecting": commit the checked options.
                        let selected = checked
                            .iter()
                            .filter_map(|i| labels.get(*i).cloned())
                            .collect();
                        let answer = AskAnswer {
                            id: question.id.clone(),
                            selected_options: selected,
                            custom_input: None,
                        };
                        Self::ask_advance(
                            &question, answer, questions, answers, current, cursor, checked, custom,
                        )
                    } else {
                        KeyResult::Consumed // in multi, only space toggles
                    }
                } else if *cursor == labels.len() {
                    *custom = Some(String::new());
                    KeyResult::Consumed
                } else {
                    let answer = AskAnswer {
                        id: question.id.clone(),
                        selected_options: vec![labels[*cursor].clone()],
                        custom_input: None,
                    };
                    Self::ask_advance(
                        &question, answer, questions, answers, current, cursor, checked, custom,
                    )
                }
            }
            _ => KeyResult::Consumed,
        }
    }

    /// Record one answer and either advance to the next question or submit
    /// the full `answers` array (validated by `tools/ask.rs` `parse_answers`).
    #[allow(clippy::too_many_arguments)]
    fn ask_advance(
        question: &Question,
        answer: AskAnswer,
        questions: &[Question],
        answers: &mut Vec<Option<AskAnswer>>,
        current: &mut usize,
        cursor: &mut usize,
        checked: &mut Vec<usize>,
        custom: &mut Option<String>,
    ) -> KeyResult {
        debug_assert_eq!(question.id, answer.id);
        answers.push(Some(answer));
        *checked = Vec::new();
        *custom = None;
        if *current + 1 < questions.len() {
            *current += 1;
            *cursor = 0;
            KeyResult::Consumed
        } else {
            let payload: Vec<Value> = answers.iter().flatten().map(AskAnswer::to_json).collect();
            KeyResult::Answer(serde_json::json!({"answers": payload}))
        }
    }

    /// Render the active dialog as a centered bordered modal (TS
    /// `OverlayPanel` / `overlayCard`), leaving `area` untouched when no
    /// dialog is open.
    pub fn render(
        &self,
        frame: &mut ratatui::Frame<'_>,
        area: ratatui::layout::Rect,
        theme: &Theme,
    ) {
        use ratatui::widgets::{Block, Borders, Clear, Paragraph};
        let Some(dialog) = &self.active else {
            return;
        };
        let fullscreen = self.wants_fullscreen();
        let popup = if fullscreen {
            area
        } else {
            crate::overlays::centered(area, 70, 60)
        };
        frame.render_widget(Clear, popup);
        let title = match dialog.remaining_secs() {
            Some(secs) => format!("{} ({secs}s)", self.dialog_title()),
            None => self.dialog_title(),
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(if fullscreen {
                ratatui::widgets::BorderType::Rounded
            } else {
                ratatui::widgets::BorderType::Plain
            })
            .border_style(ratatui::style::Style::default().fg(theme.color("border")))
            .title(Span::styled(
                if fullscreen {
                    format!(" {title} ")
                } else {
                    title
                },
                ratatui::style::Style::default().fg(theme.color("accent")),
            ));
        let inner = block.inner(popup);
        frame.render_widget(block, popup);
        // The last inner row is the hint.
        let lines = self.dialog_lines(theme, usize::from(inner.height.saturating_sub(1)));
        frame.render_widget(Paragraph::new(lines), inner);
        let hint = Paragraph::new(Span::styled(
            self.dialog_hint(),
            ratatui::style::Style::default().fg(theme.color("text_muted")),
        ));
        let hint_area = ratatui::layout::Rect {
            y: inner.y + inner.height.saturating_sub(1),
            height: 1,
            ..inner
        };
        frame.render_widget(hint, hint_area);
    }

    /// The modal title (first line only, like TS `title.split(/\r?\n/, 1)`).
    fn dialog_title(&self) -> String {
        let Some(dialog) = &self.active else {
            return String::new();
        };
        let raw = match &dialog.kind {
            DialogKind::Ask {
                questions, current, ..
            } => questions
                .get(*current)
                .map_or_else(|| "Question".to_string(), |q| q.text.clone()),
            DialogKind::Select { title, .. }
            | DialogKind::Confirm { title, .. }
            | DialogKind::Input { title, .. }
            | DialogKind::Editor { title, .. } => title.clone(),
        };
        raw.lines().next().unwrap_or_default().to_string()
    }

    /// Footer hint row (TS `actionHint` rows).
    fn dialog_hint(&self) -> &'static str {
        match self.active.as_ref().map(|d| &d.kind) {
            Some(DialogKind::Confirm { .. }) => "enter yes  ·  esc no",
            Some(DialogKind::Input { .. }) => "enter submit  ·  esc cancel",
            Some(DialogKind::Editor { .. }) => "ctrl+enter submit  ·  enter newline  ·  esc cancel",
            Some(DialogKind::Select {
                searchable: true, ..
            }) => "type to search  ·  ↑/↓ navigate  ·  enter select  ·  esc cancel",
            Some(DialogKind::Select { .. }) => "↑/↓ navigate  ·  enter select  ·  esc cancel",
            Some(DialogKind::Ask {
                questions,
                current,
                custom,
                ..
            }) => match (
                custom.is_some(),
                questions.get(*current).is_some_and(|q| q.multi),
            ) {
                (true, _) => "type your answer  ·  enter submit  ·  esc back",
                (_, true) => "space toggle  ·  enter done  ·  esc cancel",
                _ => "↑/↓ navigate  ·  enter select  ·  esc cancel",
            },
            None => "",
        }
    }

    /// Body lines of the active dialog.
    #[allow(clippy::too_many_lines)] // one style branch per dialog kind
    fn dialog_lines(&self, theme: &Theme, body_rows: usize) -> Vec<Line<'static>> {
        use ratatui::style::Style;
        let Some(dialog) = &self.active else {
            return Vec::new();
        };
        let accent = Style::default().fg(theme.color("accent"));
        let muted = Style::default().fg(theme.color("text_muted"));
        let text = Style::default().fg(theme.color("text"));
        let mut lines: Vec<Line<'static>> = Vec::new();
        match &dialog.kind {
            DialogKind::Confirm { message, .. } => {
                for line in message.lines() {
                    lines.push(Line::styled(line.to_string(), text));
                }
            }
            DialogKind::Input {
                placeholder,
                buffer,
                ..
            } => {
                let shown = if buffer.is_empty() {
                    format!("{placeholder}▏")
                } else {
                    format!("{buffer}▏")
                };
                lines.push(Line::styled(shown, text));
            }
            DialogKind::Editor { buffer, .. } => {
                let shown = format!("{buffer}▏");
                for line in shown.split('\n') {
                    lines.push(Line::styled(line.to_owned(), text));
                }
            }
            DialogKind::Select {
                title,
                options,
                cursor,
                query,
                searchable,
                ..
            } => {
                // The header shows the first title line; the rest (e.g. the tool approval
                // prompt's `Command: ...` details) is body text above the options.
                let detail: Vec<&str> = title.lines().skip(1).collect();
                if !detail.is_empty() {
                    lines.push(Line::default());
                    for line in detail {
                        lines.push(Line::styled(line.to_string(), text));
                    }
                    lines.push(Line::default());
                }
                if *searchable {
                    lines.push(if query.is_empty() {
                        Line::styled("Type to search", muted)
                    } else {
                        Line::styled(format!("Search: {query}▏"), text)
                    });
                }
                let matches = select_matches(options, query);
                if matches.is_empty() {
                    lines.push(Line::styled("  No matches", muted));
                    return lines;
                }
                let heights: Vec<usize> = matches
                    .iter()
                    .map(|&i| 1 + usize::from(options[i].description.is_some()))
                    .collect();
                let rows = body_rows.saturating_sub(lines.len());
                let mut window = select_window(&heights, *cursor, rows);
                if window.len() < matches.len() {
                    // Room for the `(n/N)` position row under a scrolled list.
                    window = select_window(&heights, *cursor, rows.saturating_sub(1));
                }
                for row in window.clone() {
                    let option = &options[matches[row]];
                    let marker = if row == *cursor {
                        theme.glyph("prompt")
                    } else {
                        " "
                    };
                    let style = if row == *cursor { accent } else { text };
                    lines.push(Line::styled(format!("{marker} {}", option.label), style));
                    if let Some(description) = &option.description {
                        lines.push(Line::styled(format!("  {description}"), muted));
                    }
                }
                if window.len() < matches.len() {
                    lines.push(Line::styled(
                        format!(
                            "  ({}/{})",
                            (*cursor).min(matches.len() - 1) + 1,
                            matches.len()
                        ),
                        muted,
                    ));
                }
            }
            DialogKind::Ask {
                questions,
                current,
                cursor,
                checked,
                custom,
                ..
            } => {
                if let Some(question) = questions.get(*current) {
                    if let Some(header) = &question.header {
                        lines.push(Line::styled(header.clone(), accent));
                    }
                    for (i, label) in question.display_labels().iter().enumerate() {
                        let marker = if i == *cursor {
                            theme.glyph("prompt")
                        } else {
                            " "
                        };
                        let mark = if question.multi && checked.contains(&i) {
                            "[x]"
                        } else {
                            ""
                        };
                        let style = if i == *cursor { accent } else { text };
                        lines.push(Line::styled(format!("{marker} {mark}{label}"), style));
                        if let Some(description) =
                            question.options.get(i).and_then(|o| o.description.as_ref())
                        {
                            lines.push(Line::styled(format!("    {description}"), muted));
                        }
                    }
                    let extra = if question.multi {
                        DONE_SELECTING_LABEL
                    } else {
                        OTHER_OPTION_LABEL
                    };
                    let extra_row = question.options.len();
                    let marker = if *cursor == extra_row {
                        theme.glyph("prompt")
                    } else {
                        " "
                    };
                    let style = if *cursor == extra_row { accent } else { muted };
                    lines.push(Line::styled(format!("{marker} {extra}"), style));
                    if let Some(buffer) = custom {
                        lines.push(Line::styled(format!("▸ {buffer}▏"), text));
                    }
                    if questions.len() > 1 {
                        lines.push(Line::styled(
                            format!("question {}/{}", *current + 1, questions.len()),
                            muted,
                        ));
                    }
                }
            }
        }
        lines
    }
}

/// What a keypress did inside a dialog.
enum KeyResult {
    /// Absorbed; the dialog stays open.
    Consumed,
    /// The dialog resolved with this answer value.
    Answer(Value),
    /// Esc: abort with `{"cancelled": true}`.
    Cancel,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port::FakePort;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn snapshot(ui: &ExtensionUi) -> Vec<String> {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| ui.render(f, f.area(), &Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..24)
            .map(|y| (0..80).map(|x| buf[(x, y)].symbol()).collect())
            .collect()
    }

    #[test]
    fn queue_serializes_one_dialog_at_a_time() {
        let mut ui = ExtensionUi::new();
        ui.push(&json!({"type":"extension_ui_request","id":"1","method":"confirm","title":"A","message":"a?"}));
        ui.push(&json!({"type":"extension_ui_request","id":"2","method":"confirm","title":"B","message":"b?"}));
        assert!(ui.is_active());
        assert_eq!(ui.pending_len(), 2);
        assert_eq!(ui.head().map(|r| r.id.as_str()), Some("1"));
        let port = FakePort::default();
        ui.handle_key(key(KeyCode::Enter), &port); // yes to A
        assert_eq!(ui.head().map(|r| r.id.as_str()), Some("2"));
        assert!(ui.is_active());
        ui.handle_key(key(KeyCode::Enter), &port); // yes to B
        assert!(!ui.is_active());
        assert_eq!(ui.pending_len(), 0);
        assert_eq!(ui.take_notification(), None);
    }

    #[test]
    fn select_enter_sends_value_and_esc_sends_cancelled() {
        let mut ui = ExtensionUi::new();
        ui.push(
            &json!({"type":"extension_ui_request","id":"sel-1","method":"select",
            "title":"Pick","options":["Basic","Pro"]}),
        );
        let port = FakePort::default();
        ui.handle_key(key(KeyCode::Down), &port);
        ui.handle_key(key(KeyCode::Enter), &port);
        let log = port.log.lock();
        assert_eq!(log.len(), 1);
        assert!(log[0].starts_with("extension_ui_response(sel-1, "));
        assert!(log[0].contains(r#""value":"Pro""#));
        drop(log);

        ui.push(
            &json!({"type":"extension_ui_request","id":"sel-2","method":"select",
            "title":"Pick","options":["Basic"]}),
        );
        let port = FakePort::default();
        ui.handle_key(key(KeyCode::Esc), &port);
        let log = port.log.lock();
        assert!(log[0].starts_with("extension_ui_response(sel-2, "));
        assert!(log[0].contains(r#""cancelled":true"#));
        assert!(!log[0].contains(r#""timedOut":true"#));
    }

    #[test]
    fn fullscreen_select_searches_and_answers_the_filtered_label() {
        let mut ui = ExtensionUi::new();
        ui.push(&json!({"type":"extension_ui_request","id":"login","method":"select",
            "title":"Select provider to login","fullscreen":true,
            "options":["ChatGPT Plus/Pro","Anthropic (Claude Pro/Max)","Kimi Code","Moonshot (Kimi API)"]}));
        assert!(ui.wants_fullscreen());
        let port = FakePort::default();
        for c in "kimi api".chars() {
            ui.handle_key(key(KeyCode::Char(c)), &port); // j/k/digits type, not navigate
        }
        let screen = snapshot(&ui).join("\n");
        assert!(screen.contains("Search: kimi api"), "{screen}");
        assert!(
            screen.contains("Moonshot (Kimi API)") && !screen.contains("Kimi Code"),
            "{screen}"
        );
        ui.handle_key(key(KeyCode::Enter), &port);
        assert!(port.log.lock()[0].contains(r#""value":"Moonshot (Kimi API)""#));
        assert!(!ui.wants_fullscreen());
    }

    #[test]
    fn long_select_scrolls_to_keep_the_cursor_visible() {
        let options: Vec<String> = (1..=80).map(|i| format!("provider-{i:02}")).collect();
        let mut ui = ExtensionUi::new();
        ui.push(&json!({"type":"extension_ui_request","id":"s","method":"select","title":"Pick","options":options}));
        let port = FakePort::default();
        for _ in 0..59 {
            ui.handle_key(key(KeyCode::Down), &port);
        }
        let screen = snapshot(&ui).join("\n");
        assert!(
            screen.contains("provider-60") && !screen.contains("provider-01"),
            "{screen}"
        );
        assert!(screen.contains("(60/80)"), "{screen}");
    }

    #[test]
    fn cancel_dialog_withdraws_it_unanswered() {
        let mut ui = ExtensionUi::new();
        ui.push(
            &json!({"type":"extension_ui_request","id":"paste","method":"input","title":"Paste"}),
        );
        ui.push(&json!({"type":"extension_ui_request","id":"next","method":"confirm","title":"B","message":"b?"}));
        ui.push(
            &json!({"type":"extension_ui_request","method":"cancelDialog","params":{"id":"paste"}}),
        );
        assert_eq!(ui.head().map(|r| r.id.as_str()), Some("next"));
        assert_eq!(ui.pending_len(), 1);
    }

    #[test]
    fn select_number_keys_jump_to_option() {
        let mut ui = ExtensionUi::new();
        ui.push(
            &json!({"type":"extension_ui_request","id":"sel","method":"select",
            "title":"Pick","options":["One","Two","Three"]}),
        );
        let port = FakePort::default();
        ui.handle_key(key(KeyCode::Char('3')), &port);
        ui.handle_key(key(KeyCode::Enter), &port);
        assert!(port.log.lock()[0].contains(r#""value":"Three""#));
    }

    #[test]
    fn select_renders_modal_with_title_options_and_hint() {
        let mut ui = ExtensionUi::new();
        ui.push(
            &json!({"type":"extension_ui_request","id":"sel","method":"select",
            "title":"Pick a plan",
            "options":[{"label":"Basic","description":"cheap"},{"label":"Pro"}]}),
        );
        let rows = snapshot(&ui);
        assert!(rows.iter().any(|r| r.contains("Pick a plan")));
        assert!(rows.iter().any(|r| r.contains("❯ Basic")));
        assert!(rows.iter().any(|r| r.contains("cheap")));
        assert!(rows
            .iter()
            .any(|r| r.contains("enter select") && r.contains("esc cancel")));
        let closed = ExtensionUi::new();
        assert!(snapshot(&closed).iter().all(|r| !r.contains("Pick a plan")));
    }

    #[test]
    fn ask_two_questions_select_then_custom_other() {
        let mut ui = ExtensionUi::new();
        ui.push(
            &json!({"type":"extension_ui_request","id":"ask-1","method":"ask",
            "questions":[
                {"id":"q1","question":"Storage?","header":"Config",
                 "options":[{"label":"SQLite"},{"label":"Postgres"}]},
                {"id":"q2","question":"Name?","options":[{"label":"auto"}]
                }]}),
        );
        let port = FakePort::default();
        ui.handle_key(key(KeyCode::Enter), &port); // q1: SQLite
                                                   // q2: down to "Other (type your own)", type, submit
        ui.handle_key(key(KeyCode::Down), &port);
        ui.handle_key(key(KeyCode::Enter), &port);
        for c in "mine".chars() {
            ui.handle_key(key(KeyCode::Char(c)), &port);
        }
        ui.handle_key(key(KeyCode::Enter), &port);
        let log = port.log.lock();
        assert_eq!(log.len(), 1);
        assert!(log[0].starts_with("extension_ui_response(ask-1, "));
        assert!(log[0].contains(r#""id":"q1""#));
        assert!(log[0].contains(r#""selectedOptions":["SQLite"]"#));
        assert!(log[0].contains(r#""customInput":"mine""#));
        assert!(!ui.is_active());
    }

    #[test]
    fn ask_multi_space_toggle_done_commits() {
        let mut ui = ExtensionUi::new();
        ui.push(
            &json!({"type":"extension_ui_request","id":"ask-m","method":"ask",
            "questions":[{"id":"m","question":"Pick tags","multi":true,
                "options":[{"label":"a"},{"label":"b"},{"label":"c"}]}]}),
        );
        let rows = snapshot(&ui);
        assert!(rows.iter().any(|r| r.contains("✓ Done selecting")));
        let port = FakePort::default();
        ui.handle_key(key(KeyCode::Char(' ')), &port); // toggle a
        ui.handle_key(key(KeyCode::Down), &port);
        ui.handle_key(key(KeyCode::Char(' ')), &port); // toggle b
        ui.handle_key(key(KeyCode::Down), &port);
        ui.handle_key(key(KeyCode::Down), &port); // cursor on Done row
        ui.handle_key(key(KeyCode::Enter), &port);
        let log = port.log.lock();
        assert!(log[0].contains(r#""selectedOptions":["a","b"]"#));
    }

    #[test]
    fn ask_esc_cancels_whole_dialog() {
        let mut ui = ExtensionUi::new();
        ui.push(
            &json!({"type":"extension_ui_request","id":"ask-c","method":"ask",
            "questions":[{"id":"q1","question":"Q?","options":[{"label":"x"}]}]}),
        );
        let port = FakePort::default();
        ui.handle_key(key(KeyCode::Esc), &port);
        let log = port.log.lock();
        assert!(log[0].contains(r#""cancelled":true"#));
        assert!(!ui.is_active());
    }

    #[test]
    fn timeout_answers_cancelled_timed_out_and_advances() {
        let mut ui = ExtensionUi::new();
        ui.push(
            &json!({"type":"extension_ui_request","id":"t1","method":"select",
            "title":"Pick","options":["a"],"timeout":50}),
        );
        ui.push(
            &json!({"type":"extension_ui_request","id":"t2","method":"select",
            "title":"Pick","options":["b"],"timeout":50}),
        );
        let expired = ui.poll_deadlines(Instant::now() + Duration::from_secs(1));
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].0, "t1");
        assert!(expired[0].1.get("cancelled").and_then(Value::as_bool) == Some(true));
        assert!(expired[0].1.get("timedOut").and_then(Value::as_bool) == Some(true));
        // The next queued dialog is now active.
        assert_eq!(ui.head().map(|r| r.id.as_str()), Some("t2"));
    }

    #[test]
    fn a_dialog_without_deadline_survives_deadline_polls_and_keeps_taking_keys() {
        let mut ui = ExtensionUi::new();
        ui.push(&json!({"type":"extension_ui_request","id":"i1","method":"input","title":"Key"}));
        assert!(ui
            .poll_deadlines(Instant::now() + Duration::from_secs(60))
            .is_empty());
        assert!(
            ui.is_active(),
            "an unexpired poll must not drop the active dialog"
        );
        let port = FakePort::default();
        assert!(ui.handle_key(key(KeyCode::Char('x')), &port));
        ui.handle_key(key(KeyCode::Enter), &port);
        assert!(port.log.lock()[0].contains(r#""value":"x""#));
    }

    #[test]
    fn confirm_yes_no_and_input_typing() {
        let mut ui = ExtensionUi::new();
        ui.push(
            &json!({"type":"extension_ui_request","id":"c","method":"confirm",
            "title":"Switch plan?","message":"Reserve reached"}),
        );
        let port = FakePort::default();
        ui.handle_key(key(KeyCode::Char('n')), &port);
        assert!(port.log.lock()[0].contains(r#""value":false"#));

        ui.push(
            &json!({"type":"extension_ui_request","id":"i","method":"input",
            "title":"Label","placeholder":"type here"}),
        );
        let port = FakePort::default();
        let rows = snapshot(&ui);
        assert!(rows.iter().any(|r| r.contains("type here")));
        for c in "hello".chars() {
            ui.handle_key(key(KeyCode::Char(c)), &port);
        }
        ui.handle_key(key(KeyCode::Backspace), &port);
        ui.handle_key(key(KeyCode::Enter), &port);
        assert!(port.log.lock()[0].contains(r#""value":"hell""#));
    }

    #[test]
    fn side_effect_frames_apply_without_dialog() {
        let mut ui = ExtensionUi::new();
        ui.push(&json!({"type":"extension_ui_request","method":"ui_notify",
            "params":{"type":"warning","message":"Nightly available"}}));
        assert!(!ui.is_active());
        assert_eq!(
            ui.take_notification(),
            Some(("warning".to_string(), "Nightly available".to_string()))
        );
        assert_eq!(ui.take_notification(), None); // consumed once

        ui.push(&json!({"type":"extension_ui_request","method":"setStatus",
            "params":{"key":"gates","text":"3 gates active"}}));
        ui.push(
            &json!({"type":"extension_ui_request","method":"setWorkingMessage",
            "params":{"message":"indexing…"}}),
        );
        let many: Vec<String> = (0..15).map(|i| format!("line{i}")).collect();
        ui.push(&json!({"type":"extension_ui_request","method":"setWidget",
            "params":{"key":"w","position":"above","content":many.join("\n")}}));
        ui.push(&json!({"type":"extension_ui_request","method":"setWidget",
            "params":{"key":"w2","position":"below","content":"tail"}}));

        assert_eq!(
            ui.status_segments(),
            vec![("gates".to_string(), "3 gates active".to_string())]
        );
        assert_eq!(ui.working_message(), Some("indexing…"));
        let above = ui.widgets_above();
        assert_eq!(above.len(), 1);
        assert_eq!(above[0].lines.len(), MAX_WIDGET_LINES);
        assert!(above[0].truncated);
        assert_eq!(ui.widgets_below()[0].key, "w2");
    }
}
