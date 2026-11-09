//! Runtime settings: the `set` gateway and the `SettingsView` toggles.

use pa_agent::abort::AbortController;
use serde_json::Value;
use std::collections::HashSet;

use super::{lock, Runtime, Settings};

impl Runtime {
    pub fn set<R>(&self, f: impl FnOnce(&mut SettingsView<'_>) -> R) -> R {
        let mut st = lock(&self.settings);
        f(&mut SettingsView { st: &mut st })
    }
}

/// Handle to the mutable per-session settings (no lock leaks out of `set`).
pub struct SettingsView<'a> {
    st: &'a mut Settings,
}

impl SettingsView<'_> {
    pub fn auto_compaction(&mut self, v: bool) {
        self.st.auto_compaction = v;
    }
    pub fn interrupt_mode(&mut self, v: &str) {
        self.st.interrupt_mode = v.to_string();
    }
    pub fn cache_warming(&mut self, v: &str) {
        self.st.cache_warming = v.to_string();
    }
    pub fn ask_dialog(&mut self, v: bool) -> bool {
        self.st.ask_dialog = v;
        v
    }
    pub fn subagent_level(&mut self, v: &str) {
        self.st.subagent_level = v.to_string();
    }
    pub fn todo_phases(&mut self, v: Value) {
        self.st.todo_phases = v;
    }
    pub fn event_filter(&mut self, filter: Option<HashSet<String>>, delta: bool) {
        self.st.event_filter = filter;
        self.st.delta_updates = delta;
    }
    pub fn running_bash(&mut self, v: Option<AbortController>) -> Option<AbortController> {
        std::mem::replace(&mut self.st.running_bash, v)
    }
    pub fn abort_bash(&mut self) {
        if let Some(c) = self.st.running_bash.take() {
            c.abort();
        }
    }
    #[must_use]
    pub fn todo_phases_value(&self) -> Value {
        self.st.todo_phases.clone()
    }
}
