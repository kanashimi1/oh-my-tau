//! Session naming and background auto-title generation.

use anyhow::Result;
use std::sync::Arc;

use super::model::NO_SESSION_PRINT;
use super::{lock, NameListener, Runtime};

/// One title-model candidate: model plus the credentials to call it with.
type TitleCandidate = (
    pa_types::ai::Model,
    Option<String>,
    Option<std::collections::BTreeMap<String, String>>,
);

impl Runtime {
    /// A resumed session keeps its title: seed the name/source from the file's
    /// title slot (TS `SessionManager` restore) so auto-titling does not rename it.
    pub(super) fn restore_title_from(&self, index: &omp_session::index::SessionIndex) {
        if let Some(slot) = &index.title_slot {
            if !slot.title.is_empty() {
                let mut st = lock(&self.settings);
                st.session_name = Some(slot.title.clone());
                st.session_name_source.clone_from(&slot.source);
            }
        }
    }

    /// `SessionManager.setSessionName`: persist the `title_change` entry, patch the
    /// fixed-width `title` slot at line 1, update state, the terminal title, and the
    /// ext bridge. `source` is `"user"` (`/rename`, rpc `set_session_name`) or
    /// `"auto"` (title generator). A user-set name is never replaced by an auto
    /// title, and an auto title never replaces an existing one: both are no-ops.
    ///
    /// # Errors
    /// The cleaned name is empty, or a persistence write failed.
    pub fn set_session_name(&self, name: &str, source: &str) -> Result<()> {
        let title = crate::title::clean_title(name);
        anyhow::ensure!(!title.is_empty(), "Session name cannot be empty");
        let previous = {
            let mut st = lock(&self.settings);
            if source == "auto"
                && (st.session_name.is_some() || st.session_name_source.as_deref() == Some("user"))
            {
                return Ok(());
            }
            let previous = st.session_name.replace(title.clone());
            st.session_name_source = Some(source.to_string());
            previous
        };
        let ts = crate::persist::now_iso();
        self.persist
            .append_title_change(&title, source, previous.as_deref())?;
        let _ = self
            .persist
            .with_writer(|w| w.update_title_slot(&title, Some(source), &ts));
        crate::title::set_terminal_title(&crate::title::format_session_terminal_title(
            Some(&title),
            Some(&self.cwd),
        ));
        if let Some(f) = lock(&self.on_session_name).clone() {
            f(&title);
        }
        Ok(())
    }

    /// The current (restored or set) session name.
    pub fn session_name(&self) -> Option<String> {
        lock(&self.settings).session_name.clone()
    }

    /// Sink applied session-name changes (the ext bridge's `set_session_name`).
    pub fn set_name_listener(&self, f: Option<NameListener>) {
        *lock(&self.on_session_name) = f;
    }

    /// Start one background auto-title for the first non-low-signal user message
    /// of an unnamed session (TS `#startAutoTitle`); no-op when a name exists,
    /// titling is disabled, or the input is a greeting/filler.
    pub(super) fn spawn_auto_title(self: &Arc<Self>, message: &str) {
        let trace = std::env::var_os("OMP_RS_TITLE_TRACE").is_some();
        if std::env::var_os("OMP_RS_NO_TITLE").is_some() {
            return;
        }
        // Print mode without a session file: the title has no reader (no
        // session to name, no UI to show it), so the side request is skipped.
        if NO_SESSION_PRINT.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        if lock(&self.settings).session_name.is_some() {
            return;
        }
        if crate::title::is_low_signal_title_input(message) {
            return;
        }
        if trace {
            eprintln!("[title] auto-title spawned");
        }
        let rt = self.clone();
        let message = message.to_string();
        let handle = tokio::spawn(async move {
            let Some(title) = rt.generate_auto_title(&message).await else {
                return;
            };
            let _ = rt.set_session_name(&title, "auto");
        });
        *lock(&self.title_task) = Some(handle);
    }

    /// Wait for a pending automatic title generation (print mode exits only
    /// after the turn pipeline drains; titling is part of that drain in TS).
    pub async fn wait_for_title(&self) {
        // Take the handle out first: the guard must not live across the await.
        let pending = lock(&self.title_task).take();
        if let Some(h) = pending {
            let _ = h.await;
        }
    }

    /// Title models: the configured `smol` role first (`config.yml` modelRoles),
    /// then the session's current model as fallback (TS `getTitleModels`).
    async fn generate_auto_title(&self, first_message: &str) -> Option<String> {
        let mut candidates: Vec<TitleCandidate> = Vec::new();
        if std::env::var_os("OMP_RS_BASE_URL").is_none() {
            match crate::model::select(Some("smol")).await {
                Ok(sel) => candidates.push((sel.model, sel.api_key, sel.headers)),
                Err(e) => {
                    if std::env::var_os("OMP_RS_TITLE_TRACE").is_some() {
                        eprintln!("[title] smol role resolve failed: {e:#}");
                    }
                }
            }
        }
        if std::env::var_os("OMP_RS_TITLE_TRACE").is_some() {
            eprintln!("[title] candidates: {}", candidates.len());
        }
        if let Ok(live) = crate::compact::LiveCompleter::from_target(&self.target) {
            let key = format!("{}/{}", live.model.provider, live.model.id);
            if candidates
                .iter()
                .all(|(m, _, _)| format!("{}/{}", m.provider, m.id) != key)
            {
                candidates.push((live.model, live.api_key, live.headers));
            }
        }
        let trace = std::env::var_os("OMP_RS_TITLE_TRACE").is_some();
        for (model, api_key, headers) in candidates {
            if trace {
                eprintln!("[title] candidate {}/{}", model.provider, model.id);
            }
            match crate::title::generate_title_with_model(first_message, &model, api_key, headers)
                .await
            {
                Some(title) => return Some(title),
                None if trace => eprintln!("[title] candidate produced no title"),
                None => {}
            }
        }
        if trace {
            eprintln!("[title] no candidate produced a title");
        }
        None
    }
}
