//! Runtime-driven dialogs for interactive slash commands (TS `hasUI` + `ctx.ui.select` /
//! `confirm` / `input` / `editor`).
//!
//! A dialog is an `extension_ui_request` frame with an `id` (TS rpc wire shapes:
//! `select {title, options: string[], optionDetails?}`, `confirm {title, message}`,
//! `input {title, placeholder?}`, `editor {title, prefill?}`); the answer is the matching
//! `extension_ui_response` (`value` / `confirmed` / `cancelled`), routed to the waiter by
//! [`crate::rpc::resolve_pending_ui_request`] — from the TUI bridge (`tui.rs`) or the RPC read
//! loop (`rpc.rs`, which keeps reading while a command awaits its dialog). A dialog dropped
//! unanswered is withdrawn with a `cancelDialog` side-effect frame.
//!
//! [`has_ui`] gates the dialogs like TS `hasUI`: set by the TUI host and by `--mode rpc`
//! (unless `--no-ui`); print/json modes keep the commands' text fallbacks.
//!
//! Fire-and-forget host effects: [`set_editor_text`] (TS rpc `set_editor_text`) and
//! [`copy_to_clipboard`] (`{"type":"clipboard_copy","text":…}`, see the `rpc` module doc).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use serde_json::{json, Value};

use crate::runtime::Runtime;

static INTERACTIVE_UI: AtomicBool = AtomicBool::new(false);
static DIALOG_SEQ: AtomicU64 = AtomicU64::new(0);

/// Mark this process as one whose host answers dialogs (TUI, or RPC without `--no-ui`).
pub fn mark_interactive_ui() {
    INTERACTIVE_UI.store(true, Ordering::SeqCst);
}

/// TS `hasUI`: whether `extension_ui_request` dialogs get answered in this process.
pub fn has_ui() -> bool {
    INTERACTIVE_UI.load(Ordering::SeqCst)
}

/// Withdraws a dialog whose round trip was dropped before the answer came: the host closes it
/// (`cancelDialog`) instead of leaving a prompt nobody reads.
struct Withdraw<'a> {
    rt: &'a Runtime,
    id: String,
    answered: bool,
}

impl Drop for Withdraw<'_> {
    fn drop(&mut self) {
        if !self.answered {
            crate::rpc::forget_pending_ui_request(&self.id);
            self.rt.send(json!({
                "type": "extension_ui_request",
                "method": "cancelDialog",
                "params": {"id": self.id},
            }));
        }
    }
}

/// One dialog round trip. `None` = cancelled (Esc) or the answer never came.
pub async fn request(rt: &Runtime, method: &str, fields: Value) -> Option<Value> {
    let id = format!("ui-dlg-{}", DIALOG_SEQ.fetch_add(1, Ordering::SeqCst));
    let rx = crate::rpc::register_pending_ui_request(&id);
    let mut frame = fields;
    frame["type"] = json!("extension_ui_request");
    frame["id"] = json!(id);
    frame["method"] = json!(method);
    rt.send(frame);
    let mut guard = Withdraw {
        rt,
        id,
        answered: false,
    };
    let answer = rx.await.ok();
    guard.answered = true;
    let answer = answer?;
    if answer.get("cancelled").and_then(Value::as_bool) == Some(true) {
        return None;
    }
    Some(answer)
}

/// One `select` row: the answer value is `label`; `description` is the dim detail line.
#[derive(Debug, Clone)]
pub struct Choice {
    pub label: String,
    pub description: Option<String>,
}

impl Choice {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: None,
        }
    }

    pub fn with(label: impl Into<String>, description: impl Into<String>) -> Self {
        let description = description.into();
        Self {
            label: label.into(),
            description: (!description.is_empty()).then_some(description),
        }
    }
}

/// `select` presentation flags.
#[derive(Debug, Clone, Copy, Default)]
pub struct SelectStyle {
    /// Full-screen picker with type-to-search (TS full-screen selectors).
    pub fullscreen: bool,
    /// Type-to-search in the inline list.
    pub searchable: bool,
    /// Initially highlighted row.
    pub initial: usize,
}

/// Labels made unique (the answer is matched back to its row by label).
fn unique_labels(choices: &[Choice]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(choices.len());
    for c in choices {
        let mut label = c.label.clone();
        let mut n = 2;
        while out.contains(&label) {
            label = format!("{} ({n})", c.label);
            n += 1;
        }
        out.push(label);
    }
    out
}

/// A `select` dialog over `choices`; the chosen index.
pub async fn select_with(
    rt: &Runtime,
    title: &str,
    choices: &[Choice],
    style: SelectStyle,
) -> Option<usize> {
    if choices.is_empty() {
        return None;
    }
    let labels = unique_labels(choices);
    let details: Vec<Value> = choices
        .iter()
        .map(|c| {
            c.description
                .as_ref()
                .map_or_else(|| json!({}), |d| json!({"description": d}))
        })
        .collect();
    let mut fields = json!({"title": title, "options": labels, "optionDetails": details});
    if style.fullscreen {
        fields["fullscreen"] = json!(true);
    }
    if style.searchable {
        fields["searchable"] = json!(true);
    }
    if style.initial > 0 {
        fields["initialIndex"] = json!(style.initial);
    }
    let answer = request(rt, "select", fields).await?;
    let value = answer.get("value")?.as_str()?;
    labels.iter().position(|l| l == value)
}

/// A plain `select` over string labels; the chosen index.
pub async fn select(rt: &Runtime, title: &str, labels: &[String]) -> Option<usize> {
    let choices: Vec<Choice> = labels.iter().map(Choice::new).collect();
    select_with(rt, title, &choices, SelectStyle::default()).await
}

/// A full-screen, type-to-search `select` over string labels (TS full-screen selectors).
pub async fn select_fullscreen(rt: &Runtime, title: &str, labels: &[String]) -> Option<usize> {
    let choices: Vec<Choice> = labels.iter().map(Choice::new).collect();
    select_with(
        rt,
        title,
        &choices,
        SelectStyle {
            fullscreen: true,
            ..SelectStyle::default()
        },
    )
    .await
}

/// A yes/no `confirm` dialog; `None` when cancelled. Accepts both the TS rpc answer
/// (`confirmed`) and the TUI's (`value`).
pub async fn confirm(rt: &Runtime, title: &str, message: &str) -> Option<bool> {
    let answer = request(rt, "confirm", json!({"title": title, "message": message})).await?;
    answer
        .get("confirmed")
        .or_else(|| answer.get("value"))
        .and_then(Value::as_bool)
}

/// A single-line `input` dialog; `None` when cancelled.
pub async fn input(rt: &Runtime, title: &str, placeholder: Option<&str>) -> Option<String> {
    let mut fields = json!({"title": title});
    if let Some(p) = placeholder {
        fields["placeholder"] = json!(p);
    }
    let answer = request(rt, "input", fields).await?;
    Some(
        answer
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    )
}

/// A multi-line `editor` dialog pre-filled with `prefill`; `None` when cancelled.
pub async fn editor(rt: &Runtime, title: &str, prefill: &str) -> Option<String> {
    let answer = request(rt, "editor", json!({"title": title, "prefill": prefill})).await?;
    Some(
        answer
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    )
}

/// TS `ctx.ui.setEditorText` / rpc `set_editor_text`: replace the host's prompt draft.
pub fn set_editor_text(rt: &Runtime, text: &str) {
    let id = format!("ui-dlg-{}", DIALOG_SEQ.fetch_add(1, Ordering::SeqCst));
    rt.send(json!({"type": "extension_ui_request", "id": id, "method": "set_editor_text", "text": text}));
}

/// Ask the host to put `text` on the system clipboard (TS `copyToClipboard`): the TUI copies
/// it (OSC 52 + platform tools); RPC clients receive the `clipboard_copy` frame.
pub fn copy_to_clipboard(rt: &Runtime, text: &str) {
    rt.send(json!({"type": "clipboard_copy", "text": text}));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_labels_are_disambiguated() {
        let labels = unique_labels(&[Choice::new("a"), Choice::new("a"), Choice::new("b")]);
        assert_eq!(labels, vec!["a", "a (2)", "b"]);
    }
}
