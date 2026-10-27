//! Outbound notices, tool-stream updates, herdr status, extension frames, and commands.

use serde_json::{json, Value};
use std::sync::Arc;

use super::{lock, Runtime};

impl Runtime {
    /// Emits a transient `notice` frame (TS `AgentSession.emitNotice`): the TUI
    /// renders info/warning/error rows; RPC clients receive the same shape.
    // Not wired yet: session-persistence failure notices live in persist.rs
    // (another owner); the TUI asks for this exact emission point.
    #[allow(dead_code)]
    pub fn emit_notice(&self, level: &str, message: &str, source: Option<&str>) {
        self.emit_event(crate::events::notice_frame(level, message, source));
    }

    /// Emits a `tool_stream_update` frame for a streaming tool's incremental
    /// preview (TS `tool_stream_update`, e.g. edit-tool diff hunks). pa-agent
    /// tools do not stream yet; this is the emission point for when they do —
    /// the TUI already renders the shape.
    #[allow(dead_code)] // awaits a streaming pa-agent tool; the TUI consumes the shape already.
    pub fn emit_tool_stream_update(&self, tool_call_id: &str, tool_name: &str, update: &Value) {
        self.emit_event(crate::events::tool_stream_update_frame(
            tool_call_id,
            tool_name,
            update,
        ));
    }

    /// Report `working`, unless an `ask` dialog still holds the pane Blocked.
    pub(super) fn herdr_working(&self) {
        if let Some(h) = &self.herdr {
            if self.ask_blocked.load(std::sync::atomic::Ordering::SeqCst) == 0 {
                h.report(crate::herdr::State::Working, None);
            }
        }
    }

    pub(super) fn herdr_idle(&self) {
        if let Some(h) = &self.herdr {
            h.report(crate::herdr::State::Idle, None);
        }
    }

    /// Session switch: rebuild the resume command and report the new identity.
    pub(super) fn herdr_session(&self, start_source: &str) {
        if let Some(h) = &self.herdr {
            let model_ref = self.live_selector_with_level(None);
            h.set_session(
                self.persist.file().as_deref(),
                Some(&model_ref),
                start_source,
            );
        }
    }

    /// Replace the extension/custom/file part of the slash-command catalog; announced (`available_commands_update`) once the
    /// session is live, stored silently before that (`rpc::run_with`'s start-up hook).
    #[allow(dead_code)]
    pub fn set_available_commands(&self, commands: Vec<Value>) {
        *lock(&self.commands) = commands;
        if self.announced.load(std::sync::atomic::Ordering::SeqCst) {
            self.announce_commands();
        }
    }

    /// Emit `available_commands_update` with the current catalog.
    pub fn announce_commands(&self) {
        self.announced
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let commands = self.available_commands();
        self.send(json!({"type": "available_commands_update", "commands": commands}));
    }

    /// omp's builtin slash commands followed by the extension/custom/file commands supplied
    /// through [`Self::set_available_commands`] (`getAvailableCommands` order).
    #[must_use]
    pub fn available_commands(&self) -> Vec<Value> {
        let mut all = builtin_commands();
        all.extend(crate::slash_extra::extra_catalog());
        all.extend(lock(&self.commands).iter().cloned());
        all
    }

    /// Queue a user message from an extension (`send_message` notification).
    pub fn extension_message(self: &Arc<Self>, text: &str) {
        // `prompt` steers into a live run and admits otherwise, matching omp's
        // queueing of extension-originated messages.
        let rt = self.clone();
        let text = text.to_string();
        tokio::spawn(async move {
            if let Err(e) = rt.prompt(None, &text, Some("steer"), Vec::new()).await {
                eprintln!("omt: extension send_message: {}", e.message);
            }
        });
    }

    /// Forward an extension ui notification as an `extension_ui_request` frame
    /// (rpc mode golden frames; print mode drops it — no consumer).
    pub fn send_extension_frame(&self, extension: Option<&str>, method: &str, params: &Value) {
        self.send(json!({
            "type": "extension_ui_request",
            "extension": extension,
            "method": method,
            "params": params,
        }));
    }
}

/// omp's builtin slash-command descriptors (`source: "builtin"`), as `get_available_commands` lists them.
fn builtin_commands() -> Vec<Value> {
    static BUILTIN: std::sync::OnceLock<Vec<Value>> = std::sync::OnceLock::new();
    BUILTIN
        .get_or_init(|| {
            serde_json::from_str(include_str!("../builtin_commands.json")).unwrap_or_default()
        })
        .clone()
}

// -------------------------------------------------- extension hook points
// (CliExt2: new functions only; used by `ext.rs` for the extension host.)
