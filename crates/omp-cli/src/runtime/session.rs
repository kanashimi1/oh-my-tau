//! Session switching: new/load/fork, branch-seeded resume, and switch hooks.

use super::recovery::{RecoveryConfig, EPHEMERAL_ROLE};
use anyhow::Context;
use anyhow::Result;
use pa_agent::types::{AgentMessage, ThinkingLevel};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::model::{MODEL_PINNED, THINKING_PINNED};
use super::session_state::{
    custom_type, is_bookkeeping, link_shim, message_head_shim, session_model_state,
    MessageEnvelope, SessionModelState,
};
use super::todo::latest_todo_phases;
use super::{
    level_name, lock, parse_level, parse_ultra_name, BeforeStartHook, CompactionHooks, Runtime,
    SwitchCancelled, SwitchGate, SwitchListener,
};

impl Runtime {
    /// Clear the conversation and start a fresh session file.
    pub async fn new_session(&self, dir: Option<&Path>) -> Result<()> {
        self.check_switch("new", None).await?;
        let previous = self.persist.file();
        // An in-flight compaction must not write into the old file (or swap the
        // old context over the fresh one) after the switch (bug-hunt #2).
        self.cancel_compaction_and_wait().await;
        self.cancel_retry_and_wait().await;
        self.agent.abort();
        self.agent.wait_for_idle().await;
        self.agent.reset().await;
        self.persist.start_new(&self.cwd, dir)?;
        self.bump_generation();
        self.publish_session_file();
        self.herdr_session("new");
        self.rehydrate_todo_tool(&[]).await;
        {
            let mut st = lock(&self.settings);
            st.session_name = None;
            st.session_name_source = None;
            st.last_queue = (Vec::new(), Vec::new());
        }
        lock(&self.prompts).open.clear();
        self.record_model_state().await;
        // Interactive TUI: drop the rendered transcript and the terminal
        // scrollback (TS `#runNewSessionFlow`: `resetTranscript()` +
        // `requestRender(true, { clearScrollback: true })`). RPC clients never
        // see this frame.
        if self
            .ui_context_frames
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            self.send(json!({"type": "session_reset"}));
        }
        self.notify_session_switch("new", previous);
        Ok(())
    }

    /// Resume `path` in place (TS `SessionManager.open`): the agent context is rebuilt from the
    /// journal and new entries are appended to the SAME file, chained to the active branch tip.
    /// `dir` is unused (the file keeps its own directory).
    pub async fn load_session(&self, path: &Path, dir: Option<&Path>) -> Result<()> {
        self.load_session_as(path, dir, false, false).await
    }

    /// Bind a freshly written branch file (TS `#branchIntoNewSession`): no `session_before_switch`
    /// gate and no `session_switch` event — the caller emits `session_before_branch` /
    /// `session_branch` itself; the listener only refreshes the extension ctx (`"branch"`).
    pub async fn load_branched_session(&self, path: &Path) -> Result<()> {
        self.load_session_as(path, None, false, true).await
    }

    /// Fork `path` (TS `SessionManager.forkFrom`): the continuation is written to a fresh
    /// session file in `dir` seeded with the loaded history.
    pub async fn fork_session(&self, path: &Path, dir: Option<&Path>) -> Result<()> {
        self.load_session_as(path, dir, true, false).await
    }

    /// Shared body of [`Self::load_session`] / [`Self::fork_session`].
    ///
    /// Resume is lazy like omp's: the journal is scanned once into a link index
    /// ([`omp_session::index::SessionIndex`]) and only the active branch — TS
    /// `SessionEntryIndex.pathTo(leaf)`, root → leaf — is materialized, parsed
    /// on demand from the raw lines. Non-compacted sessions never build the
    /// entry DOM at all: messages deserialize straight from the line text.
    #[allow(clippy::too_many_lines)] // resume loader: read lines, rebuild the branch, seed messages; steps share the parse
    async fn load_session_as(
        &self,
        path: &Path,
        dir: Option<&Path>,
        fork: bool,
        branched: bool,
    ) -> Result<()> {
        if !branched {
            self.check_switch(
                if fork { "fork" } else { "resume" },
                Some(path.to_path_buf()),
            )
            .await?;
        }
        let previous = self.persist.file();
        // An in-flight compaction must not write into the old session or swap
        // its context over the resumed one after the switch (bug-hunt #2).
        self.cancel_compaction_and_wait().await;
        self.cancel_retry_and_wait().await;
        // `--no-session` must never write into the resumed file: seed a scratch copy instead.
        let fork = fork || self.persist.file().is_none();
        let trace_t0 = std::time::Instant::now();
        macro_rules! rss_trace {
            ($label:expr) => {
                if std::env::var_os("OMP_RS_RESUME_TRACE").is_some() {
                    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                        let rss = status
                            .lines()
                            .find_map(|l| l.strip_prefix("VmRSS:"))
                            .unwrap_or("?");
                        let hwm = status
                            .lines()
                            .find_map(|l| l.strip_prefix("VmHWM:"))
                            .unwrap_or("?");
                        eprintln!(
                            "[resume-trace] {} t={}ms rss={} hwm={}",
                            $label,
                            trace_t0.elapsed().as_millis(),
                            rss.trim(),
                            hwm.trim()
                        );
                    }
                }
            };
        }
        rss_trace!("start");
        let index = omp_session::index::SessionIndex::load(path)
            .with_context(|| format!("cannot read session {}", path.display()))?;
        rss_trace!("indexed");
        // A compacted session resumes with the rebuilt context (summary + kept tail),
        // not the raw log; checked over the whole journal like the old loader.
        let compacted = index.refs.iter().any(|r| {
            matches!(
                r.kind,
                omp_session::index::EntryKind::Compaction
                    | omp_session::index::EntryKind::ResetBoundary
                    | omp_session::index::EntryKind::BranchSummary
            )
        });
        let branch = index.branch();
        let brefs: Vec<&omp_session::index::EntryRef> =
            branch.iter().map(|&i| &index.refs[i as usize]).collect();
        let mut messages = Vec::new();
        // Original `message` JSON of custom-role messages (parallel to
        // `messages`); seeded verbatim when pa_types cannot model the shape.
        let mut raw_messages: Vec<Option<Map<String, Value>>> = Vec::new();
        // Reduced branch for the context rebuild: entries outside the emitted
        // ranges become link-only shims, so the DOM the rebuild works on stays
        // the size of the context, not of the journal.
        if compacted {
            let (full, notes_after, head_until) =
                reduced_branch_flags(&index, path, &branch, &brefs)?;
            let entries =
                materialize_reduced_branch(&index, path, &branch, &full, notes_after, head_until)?;
            rss_trace!("materialized");
            // Rebuild first, then release the entry DOM before converting:
            // the context messages and the typed messages never coexist with
            // the full reduced tree.
            // Owned rebuild: message payloads move out of the reduced tree into
            // the context, so the two never coexist (resume stays low-RSS).
            let ctx = omp_compact::build_session_context_owned(entries, omp_compact::Leaf::Last)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            rss_trace!("rebuilt");
            messages = crate::compact::context_messages(&ctx);
        } else {
            // No compaction on the journal: messages deserialize straight from
            // the raw lines, one pass, no entry DOM at all.
            index.for_each_line(path, &branch, |r, raw| {
                if r.kind == omp_session::index::EntryKind::Message {
                    if let Ok(envelope) = serde_json::from_str::<MessageEnvelope>(raw) {
                        let m = envelope.message;
                        // Custom roles pa_types cannot model (e.g. "developer")
                        // keep their original JSON shape so resume can re-persist
                        // them byte-identically instead of dropping them.
                        if matches!(m, AgentMessage::Custom(_)) {
                            let shape = serde_json::from_str::<Value>(raw)
                                .ok()
                                .and_then(|v| v.get("message").cloned())
                                .and_then(|v| v.as_object().cloned());
                            raw_messages.push(shape);
                        } else {
                            raw_messages.push(None);
                        }
                        messages.push(m);
                    }
                }
                Ok(())
            })?;
        }
        rss_trace!("messages");
        // Todo phases (TS `getLatestTodoPhasesFromEntries(getBranch())`).
        let phases = latest_todo_phases(&index, path, &branch)?;
        rss_trace!("loaded");
        self.agent.abort();
        self.agent.wait_for_idle().await;
        self.agent.reset().await;
        if fork {
            // TS fork: a NEW file seeded with the branch; the startup file never held anything.
            if self.persist.path() != path {
                self.persist.discard_pristine();
            }
            self.persist.start_new(&self.cwd, dir)?;
            self.publish_session_file();
            crate::tools::eval_journal::carry_over(path);
            self.record_model_state().await;
            if compacted {
                crate::compact::seed_branch(&self.persist, &index, path)?;
            } else {
                for (m, raw) in messages.iter().zip(&raw_messages) {
                    seed_resumed_message(&self.persist, m, raw.as_ref());
                }
            }
        } else {
            // TS `SessionManager.open`: keep appending to the same file (header untouched,
            // nothing re-seeded), chained to the active branch tip. A compacted journal is
            // only read to rebuild the context above, never rewritten.
            let header_id = index
                .refs
                .iter()
                .find(|r| r.kind == omp_session::index::EntryKind::Header)
                .map(|r| &*r.id);
            let tip = branch.last().map(|&i| &*index.refs[i as usize].id);
            self.persist
                .bind_existing(path, header_id, tip.filter(|t| !t.is_empty()))?;
            self.publish_session_file();
        }
        // New session identity: in-flight compaction guards captured before the
        // switch are stale from here on (bug-hunt #2).
        self.bump_generation();
        rss_trace!("seeded");
        rss_trace!("pre-set");
        if std::env::var_os("OMP_RS_RESUME_TRACE").is_some() {
            let json = serde_json::to_string(&messages).unwrap_or_default();
            eprintln!(
                "[resume-trace] messages: n={} json={}B sizeof={}B",
                messages.len(),
                json.len(),
                std::mem::size_of_val(&messages)
            );
        }
        rss_trace!("swapped");
        let session_file = self.persist.file().map(|p| p.display().to_string());
        self.swap_todo_tool(Arc::new(crate::tools::TodoTool::with_session(
            phases,
            session_file.as_deref(),
        )))
        .await;
        self.agent.set_messages(messages).await;
        rss_trace!("messages-set");
        // TS restores the session's model and thinking selector on resume and
        // on every session switch; explicit `--model` / `--thinking` win only
        // for the startup load.
        let saved = session_model_state(&index, path, &branch)?;
        if !MODEL_PINNED.swap(false, std::sync::atomic::Ordering::Relaxed) {
            self.restore_session_model(&saved).await;
        }
        if !THINKING_PINNED.swap(false, std::sync::atomic::Ordering::Relaxed) {
            self.restore_session_thinking(&saved).await;
        }
        self.herdr_session("resume");
        lock(&self.prompts).open.clear();
        self.restore_title_from(&index);
        self.notify_session_switch(
            if branched {
                "branch"
            } else if fork {
                "fork"
            } else {
                "resume"
            },
            previous,
        );
        Ok(())
    }

    /// TS `getRestorableSessionModels` + registry lookup: the last-used role's
    /// model, then the session default; the first one that resolves with
    /// credentials becomes live (no `model_change` is written, as in TS).
    async fn restore_session_model(&self, saved: &SessionModelState) {
        let mut candidates: Vec<&str> = Vec::new();
        let default = saved.models.get("default").map(String::as_str);
        match saved.last_role.as_deref() {
            None | Some("default" | EPHEMERAL_ROLE) => candidates.extend(default),
            Some(role) => match saved.models.get(role).map(String::as_str) {
                None => candidates.extend(default),
                Some(r) => {
                    candidates.push(r);
                    if let Some(d) = default.filter(|d| *d != r) {
                        candidates.push(d);
                    }
                }
            },
        }
        for reference in candidates {
            let Some((provider, id)) = reference.split_once('/') else {
                continue;
            };
            if self
                .live_provider_id()
                .is_some_and(|(p, i)| p == provider && i == id)
            {
                return;
            }
            let Ok(resolved) = (self.resolver)(reference.to_owned()).await else {
                continue;
            };
            if resolved.provider != provider || resolved.id != id || resolved.api_key.is_none() {
                continue;
            }
            lock(&self.recovery).reconfigure(RecoveryConfig::from_resolved(&resolved));
            self.swap_model(&resolved, None, false, false).await;
            return;
        }
    }

    /// TS `restoreThinkingLevel`: the branch's last `thinking_level_change`
    /// (its configured selector, so `auto` stays auto), clamped to the live
    /// model; not journaled.
    async fn restore_session_thinking(&self, saved: &SessionModelState) {
        let Some((level, configured)) = saved.thinking.as_ref() else {
            // No thinking record on this branch: an ultra mode does not leak in.
            self.leave_ultra().await;
            return;
        };
        if let Some(mode) = saved.ultra.as_deref().and_then(parse_ultra_name) {
            // The mode comes back with its effects (max thinking, CLM), not journaled.
            let prev = lock(&self.ultra).replace(mode);
            self.auto_thinking
                .store(false, std::sync::atomic::Ordering::Relaxed);
            let effective = self.live_model().map_or(ThinkingLevel::Max, |m| {
                crate::model::clamp_level(&m, ThinkingLevel::Max)
            });
            if self.agent.thinking_level().await != effective {
                self.agent.set_thinking_level(effective).await;
            }
            let name = level_name(effective);
            self.emit_event(json!({
                "type": "thinking_level_changed",
                "thinkingLevel": name,
                "configured": name,
                "ultra": mode.name(),
            }));
            self.sync_clm(prev, Some(mode)).await;
            return;
        }
        self.leave_ultra().await;
        let auto = configured.as_deref() == Some("auto");
        let Some(requested) = parse_level(level) else {
            return;
        };
        let model = self.target.read().as_ref().map(|t| t.model.clone());
        let effective = model.map_or(requested, |m| crate::model::clamp_level(&m, requested));
        self.auto_thinking
            .store(auto, std::sync::atomic::Ordering::Relaxed);
        if self.agent.thinking_level().await != effective {
            self.agent.set_thinking_level(effective).await;
        }
        let name = level_name(effective);
        self.emit_event(json!({
            "type": "thinking_level_changed",
            "thinkingLevel": name,
            "configured": if auto { "auto" } else { name },
            "ultra": null,
        }));
    }

    async fn check_switch(&self, reason: &'static str, target: Option<PathBuf>) -> Result<()> {
        let gate = lock(&self.switch_gate).clone();
        if let Some(g) = gate {
            if g(reason, target).await {
                return Err(SwitchCancelled.into());
            }
        }
        Ok(())
    }

    /// The extension host, once attached: `session_before_branch` / `session_before_tree` and the
    /// other session events the runtime forwards itself.
    #[must_use]
    pub fn ext_host(&self) -> Option<Arc<omp_ext::ExtHost>> {
        lock(&self.ext_host).clone()
    }

    /// Attach the extension host (see [`Self::ext_host`]).
    pub fn set_ext_host(&self, host: Option<Arc<omp_ext::ExtHost>>) {
        let feed = host.clone().map(|host| {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
            tokio::spawn(async move {
                while let Some(frame) = rx.recv().await {
                    let Some(ty) = frame["type"].as_str().map(str::to_owned) else {
                        continue;
                    };
                    if !host.has_handlers(&ty) {
                        continue;
                    }
                    if let Err(e) = host.emit(frame, None).await {
                        eprintln!("omt: extension {ty}: {e:#}");
                    }
                }
            });
            tx
        });
        *lock(&self.ext_frames) = feed;
        *lock(&self.ext_host) = host;
    }

    /// Install the `session_before_switch` gate.
    pub fn set_switch_gate(&self, g: Option<SwitchGate>) {
        *lock(&self.switch_gate) = g;
    }

    fn notify_session_switch(&self, reason: &'static str, previous: Option<PathBuf>) {
        if let Some(f) = lock(&self.on_session_switch).clone() {
            f(reason, previous);
        }
    }

    /// Install the extension compaction bridge (see [`CompactionHooks`]).
    pub fn set_compaction_hooks(&self, h: Option<Arc<dyn CompactionHooks>>) {
        *lock(&self.compaction_hooks) = h;
    }

    /// Install the extension `before_agent_start` hook (see [`BeforeStartHook`]).
    pub fn set_before_start_hook(&self, f: Option<BeforeStartHook>) {
        *lock(&self.before_start) = f;
    }

    /// Sink completed session switches (the ext bridge's `session_switch`).
    pub fn set_switch_listener(&self, f: Option<SwitchListener>) {
        *lock(&self.on_session_switch) = f;
    }
}

/// Re-persist one resumed context message into the fresh session file (TS
/// `SessionManager` keeps the same file; the port resumes into a new file, so
/// every message in the context is seeded). Messages `pa_types` cannot model
/// (custom roles like `"developer"`) are written with their original JSON
/// shape verbatim — the raw journal map when captured, otherwise the
/// re-serialized message — so nothing in the context is silently dropped.
pub(super) fn seed_resumed_message(
    persist: &crate::persist::Persistence,
    m: &AgentMessage,
    raw: Option<&Map<String, Value>>,
) {
    let converted =
        serde_json::to_value(m).and_then(serde_json::from_value::<pa_types::session::AgentMessage>);
    if let Ok(c) = converted {
        let _ = persist.append_message(&c, m.timestamp());
        return;
    }
    let map = raw.cloned().or_else(|| {
        serde_json::to_value(m)
            .ok()
            .and_then(|v| v.as_object().cloned())
    });
    if let Some(map) = map {
        let ts = map
            .get("timestamp")
            .and_then(Value::as_i64)
            .unwrap_or_else(|| m.timestamp());
        let _ = persist.append_raw_message(map, ts);
    }
}

/// Compute the reduced-branch `full` flags for a compacted journal (TS
/// `buildReplanRebuildView`-style range selection): the rebuild reads the latest
/// compaction, the bookkeeping entries (state accumulation walks the whole
/// branch), and the emitted ranges (kept tail + post-compaction, or post-reset) —
/// but never the pre-compaction history again. Also returns the context-notes
/// start and the head-only message cut.
///
/// # Errors
/// The latest compaction entry cannot be read.
// invariant: after_reset is only true when reset_pos is Some.
#[allow(clippy::expect_used, clippy::unwrap_used)]
fn reduced_branch_flags(
    index: &omp_session::index::SessionIndex,
    path: &Path,
    branch: &[u32],
    brefs: &[&omp_session::index::EntryRef],
) -> Result<(Vec<bool>, usize, usize)> {
    let reset_pos = brefs
        .iter()
        .rposition(|r| r.kind == omp_session::index::EntryKind::ResetBoundary);
    let compaction_pos = brefs
        .iter()
        .rposition(|r| r.kind == omp_session::index::EntryKind::Compaction);
    let after_reset = match (reset_pos, compaction_pos) {
        (Some(r), Some(c)) => r > c,
        (Some(_), None) => true,
        _ => false,
    };
    let mut full = vec![false; branch.len()];
    if after_reset {
        for f in &mut full[reset_pos.expect("after_reset implies Some") + 1..] {
            *f = true;
        }
    } else if let Some(cp) = compaction_pos {
        // The compaction entry alone decides the kept range.
        let cv = index
            .load_branch(path, &branch[cp..=cp], true)?
            .into_iter()
            .next()
            .and_then(|l| l.value)
            .ok_or_else(|| anyhow::anyhow!("compaction entry vanished"))?;
        let remote = omp_compact::openai_remote_payload(&cv);
        let anthropic = omp_compact::anthropic_compaction_payload(&cv);
        let has_replacement = remote
            .as_ref()
            .and_then(|p| p.get("items"))
            .is_some_and(Value::is_array);
        let id_pos = |id: &str| brefs.iter().position(|r| &*r.id == id);
        let first_kept = cv
            .get("firstKeptEntryId")
            .and_then(Value::as_str)
            .unwrap_or("");
        let kept_start = if has_replacement {
            None
        } else if anthropic.is_some() && first_kept.is_empty() {
            cv.get("providerReplayThroughEntryId")
                .and_then(Value::as_str)
                .and_then(|id| id_pos(id).map(|i| i + 1))
        } else {
            id_pos(first_kept)
        };
        if let Some(start) = kept_start.filter(|&s| s < cp) {
            for f in &mut full[start..cp] {
                *f = true;
            }
        }
        for f in &mut full[cp..] {
            *f = true;
        }
    } else {
        // The compaction marker lives on another branch: the rebuild replays the
        // whole branch through `context_message` — keep it verbatim rather than
        // shim what may be emitted.
        full.fill(true);
    }
    // Bookkeeping entries feed the state accumulation from anywhere on the
    // branch; `context_notes` scans the custom entries after the last reset
    // boundary.
    let notes_after = reset_pos.map_or(0, |r| r + 1);
    for (i, r) in brefs.iter().enumerate() {
        if is_bookkeeping(r.kind) {
            full[i] = true;
        }
    }
    // Message entries feed the default-model inference only until the first
    // model_change; earlier ones get a head-only shim.
    let head_until = brefs
        .iter()
        .position(|r| r.kind == omp_session::index::EntryKind::ModelChange)
        .unwrap_or(branch.len());
    Ok((full, notes_after, head_until))
}

/// Materialize the reduced branch for the context rebuild: `full` entries and
/// context-note custom entries parse for real, pre-head message entries become
/// head-only shims, everything else a link-only shim — so the DOM the rebuild
/// works on stays the size of the context, not of the journal.
fn materialize_reduced_branch(
    index: &omp_session::index::SessionIndex,
    path: &Path,
    branch: &[u32],
    full: &[bool],
    notes_after: usize,
    head_until: usize,
) -> Result<Vec<Value>> {
    let mut entries: Vec<Value> = Vec::with_capacity(branch.len());
    index.for_each_line(path, branch, |r, raw| {
        let pos = entries.len();
        let value = if full[pos] {
            serde_json::from_str::<Value>(raw)
                .with_context(|| format!("{}:{}: entry parse", path.display(), r.line_no))?
        } else if r.kind == omp_session::index::EntryKind::Message && pos < head_until {
            message_head_shim(r, raw)
        } else if r.kind == omp_session::index::EntryKind::Custom
            && pos >= notes_after
            && custom_type(raw).as_deref() == Some(omp_compact::context::CONTEXT_NOTES_ENTRY_TYPE)
        {
            serde_json::from_str::<Value>(raw)
                .with_context(|| format!("{}:{}: entry parse", path.display(), r.line_no))?
        } else {
            link_shim(r)
        };
        entries.push(value);
        Ok(())
    })?;
    Ok(entries)
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod resume_seed_tests {
    use super::*;
    use serde_json::json;

    /// Shape of a real `developer` custom message (see ~/.omp journals): keys
    /// beyond `role` are an arbitrary payload `pa_types` cannot model.
    fn developer_map() -> Map<String, Value> {
        serde_json::from_value(json!({
            "role": "developer",
            "customType": "todo-reminder",
            "content": [{"type": "text", "text": "<system-reminder>keep going</system-reminder>"}],
            "display": false,
            "attribution": "agent",
            "timestamp": 1_757_890_000_000_i64,
            "synthetic": true,
        }))
        .expect("developer map")
    }

    /// The non-compacted `load_session` path: context messages plus the raw
    /// `message` JSON of custom-role messages, straight from the journal lines.
    fn load_like_resume(
        path: &std::path::Path,
    ) -> (Vec<AgentMessage>, Vec<Option<Map<String, Value>>>) {
        let index = omp_session::index::SessionIndex::load(path).expect("index");
        let branch = index.branch();
        let mut messages = Vec::new();
        let mut raws = Vec::new();
        index
            .for_each_line(path, &branch, |r, raw| {
                if r.kind == omp_session::index::EntryKind::Message {
                    if let Ok(envelope) = serde_json::from_str::<MessageEnvelope>(raw) {
                        let m = envelope.message;
                        if matches!(m, AgentMessage::Custom(_)) {
                            raws.push(
                                serde_json::from_str::<Value>(raw)
                                    .ok()
                                    .and_then(|v| v.get("message").cloned())
                                    .and_then(|v| v.as_object().cloned()),
                            );
                        } else {
                            raws.push(None);
                        }
                        messages.push(m);
                    }
                }
                Ok(())
            })
            .expect("for_each_line");
        (messages, raws)
    }

    /// `message` payloads of every `message` entry in a session file, in order.
    fn message_payloads(path: &std::path::Path) -> Vec<Value> {
        omp_session::read_values(path)
            .expect("read_values")
            .into_iter()
            .filter(|v| v.get("type").and_then(Value::as_str) == Some("message"))
            .map(|v| v["message"].clone())
            .collect()
    }

    #[test]
    fn developer_custom_message_survives_double_resume_verbatim() {
        let base = std::env::temp_dir().join(format!(
            "omp-rs-resume-seed-{}-{}",
            std::process::id(),
            crate::persist::uuid_v7()
        ));
        std::fs::create_dir_all(&base).unwrap();
        let dir_a = base.join("a");
        std::fs::create_dir_all(&dir_a).unwrap();

        // Original journal: a user turn and a developer custom message.
        let mut w = omp_session::SessionWriter::create(
            &dir_a,
            "/proj",
            "sess-a",
            "2026-10-10T00-00-00-000Z",
        )
        .unwrap();
        w.append_omp_message(
            omp_session::OmpMessage::Agent(developer_map()),
            "2026-10-10T00:00:01.000Z",
        )
        .unwrap();
        let original = w.path().to_path_buf();
        drop(w);
        let original_payloads = message_payloads(&original);
        assert_eq!(original_payloads.len(), 1);
        // The developer entry must be a context message (Custom), confirming
        // the resume-bug premise: it loads into memory...
        let (messages, raws) = load_like_resume(&original);
        assert_eq!(messages.len(), 1);
        assert!(
            matches!(messages[0], AgentMessage::Custom(_)),
            "developer message must load as a Custom context message"
        );
        // ...but never converts to pa_types, so the old seeding loop dropped it.
        assert!(
            serde_json::from_value::<pa_types::session::AgentMessage>(
                serde_json::to_value(&messages[0]).unwrap()
            )
            .is_err(),
            "pa_types cannot model role=developer"
        );

        // Resume #1: seed into a new session file.
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_b).unwrap();
        let persist_b = crate::persist::Persistence::for_test(&dir_b).unwrap();
        for (m, raw) in messages.iter().zip(&raws) {
            seed_resumed_message(&persist_b, m, raw.as_ref());
        }
        let file_b = persist_b.file().unwrap();
        let payloads_b = message_payloads(&file_b);
        assert_eq!(payloads_b.len(), 1, "every context message is re-persisted");
        assert_eq!(
            payloads_b[0], original_payloads[0],
            "developer message keeps its exact original shape"
        );
        assert_eq!(
            payloads_b[0].to_string(),
            original_payloads[0].to_string(),
            "developer message keeps its original key order (byte-identical)"
        );

        // Resume #2 from the new file: same context, same shape again.
        let (messages2, raws2) = load_like_resume(&file_b);
        assert_eq!(messages2.len(), 1);
        assert!(matches!(messages2[0], AgentMessage::Custom(_)));
        let dir_c = base.join("c");
        std::fs::create_dir_all(&dir_c).unwrap();
        let persist_c = crate::persist::Persistence::for_test(&dir_c).unwrap();
        for (m, raw) in messages2.iter().zip(&raws2) {
            seed_resumed_message(&persist_c, m, raw.as_ref());
        }
        let payloads_c = message_payloads(&persist_c.file().unwrap());
        assert_eq!(payloads_c, payloads_b, "double resume is stable");

        let _ = std::fs::remove_dir_all(&base);
    }

    /// The compacted `load_session` path end to end (what `load_session` does
    /// when the journal has a `compaction` entry): reduced-branch rebuild →
    /// context messages, plus `seed_branch` copying the raw branch lines into
    /// the fresh session file. A custom-role (`developer`) message in the kept
    /// range must survive in both the context and the seeded file, byte-identical.
    fn compacted_resume_once(
        path: &std::path::Path,
        dir: &std::path::Path,
    ) -> (Vec<AgentMessage>, std::path::PathBuf) {
        let index = omp_session::index::SessionIndex::load(path).expect("index");
        let branch = index.branch();
        let brefs: Vec<&omp_session::index::EntryRef> =
            branch.iter().map(|&i| &index.refs[i as usize]).collect();
        let (full, notes_after, head_until) =
            reduced_branch_flags(&index, path, &branch, &brefs).expect("flags");
        let entries =
            materialize_reduced_branch(&index, path, &branch, &full, notes_after, head_until)
                .expect("materialize");
        let ctx = omp_compact::build_session_context_owned(entries, omp_compact::Leaf::Last)
            .expect("context");
        let messages = crate::compact::context_messages(&ctx);
        let persist = crate::persist::Persistence::for_test(dir).unwrap();
        crate::compact::seed_branch(&persist, &index, path).expect("seed");
        (messages, persist.file().unwrap())
    }

    /// `type` field counts over a session file.
    fn entry_type_counts(path: &std::path::Path) -> std::collections::BTreeMap<String, usize> {
        let mut counts = std::collections::BTreeMap::new();
        for v in omp_session::read_values(path).expect("read_values") {
            if let Some(t) = v.get("type").and_then(Value::as_str) {
                *counts.entry(t.to_owned()).or_default() += 1;
            }
        }
        counts
    }

    #[test]
    #[allow(clippy::too_many_lines)] // one resume scenario across two compactions; the steps build on each other
    fn developer_custom_message_survives_compacted_double_resume_verbatim() {
        let base = std::env::temp_dir().join(format!(
            "omp-rs-compacted-resume-seed-{}-{}",
            std::process::id(),
            crate::persist::uuid_v7()
        ));
        std::fs::create_dir_all(&base).unwrap();
        let dir_a = base.join("a");
        std::fs::create_dir_all(&dir_a).unwrap();

        // Compacted journal: user turn, developer custom message (kept range),
        // a compaction entry, then a post-compaction user turn.
        let mut w = omp_session::SessionWriter::create(
            &dir_a,
            "/proj",
            "sess-a",
            "2026-10-10T00-00-00-000Z",
        )
        .unwrap();
        let first_kept = w
            .append_omp_message(
                omp_session::OmpMessage::Agent(
                    serde_json::from_value(json!({
                        "role": "user",
                        "content": [{"type": "text", "text": "before compaction"}],
                        "timestamp": 1_757_890_000_000_i64,
                    }))
                    .expect("user map"),
                ),
                "2026-10-10T00:00:01.000Z",
            )
            .unwrap();
        w.append_omp_message(
            omp_session::OmpMessage::Agent(developer_map()),
            "2026-10-10T00:00:02.000Z",
        )
        .unwrap();
        let result = omp_compact::summary::CompactionResult {
            summary: "summary text".to_owned(),
            short_summary: "short".to_owned(),
            first_kept_entry_id: first_kept,
            tokens_before: 1000,
            details: json!({"readFiles": [], "modifiedFiles": []}),
            preserve_data: None,
        };
        omp_compact::apply::append_compaction(
            &mut w,
            &result,
            "2026-10-10T00:00:03.000Z",
            &omp_compact::apply::ApplyOptions::default(),
        )
        .unwrap();
        w.append_omp_message(
            omp_session::OmpMessage::Agent(
                serde_json::from_value(json!({
                    "role": "user",
                    "content": [{"type": "text", "text": "after compaction"}],
                    "timestamp": 1_757_890_001_000_i64,
                }))
                .expect("user map"),
            ),
            "2026-10-10T00:00:04.000Z",
        )
        .unwrap();
        let original = w.path().to_path_buf();
        drop(w);
        let original_payloads = message_payloads(&original);
        assert_eq!(original_payloads.len(), 3);
        let developer_payload = original_payloads[1].clone();
        assert_eq!(developer_payload["role"], json!("developer"));

        // Resume #1 through the compacted path: the developer message is a
        // Custom context message...
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_b).unwrap();
        let (messages_b, file_b) = compacted_resume_once(&original, &dir_b);
        assert!(
            messages_b
                .iter()
                .any(|m| matches!(m, AgentMessage::Custom(_))),
            "compacted context keeps the developer message (n={})",
            messages_b.len()
        );
        // ...and the seeded file carries it byte-identically.
        let counts_b = entry_type_counts(&file_b);
        assert_eq!(
            counts_b.get("compaction").copied(),
            Some(1),
            "compaction entry is re-seeded"
        );
        let payloads_b = message_payloads(&file_b);
        assert_eq!(payloads_b.len(), 3, "every branch message is re-seeded");
        let seeded_developer = payloads_b
            .iter()
            .find(|p| p.get("role").and_then(Value::as_str) == Some("developer"))
            .expect("developer message in seeded file");
        assert_eq!(
            seeded_developer.to_string(),
            developer_payload.to_string(),
            "developer message keeps its exact original shape and key order"
        );

        // Resume #2 from the seeded file: same context, same shape again.
        let dir_c = base.join("c");
        std::fs::create_dir_all(&dir_c).unwrap();
        let (messages_c, file_c) = compacted_resume_once(&file_b, &dir_c);
        assert_eq!(
            messages_c.len(),
            messages_b.len(),
            "double resume keeps the same context"
        );
        assert!(
            messages_c
                .iter()
                .any(|m| matches!(m, AgentMessage::Custom(_))),
            "second compacted context keeps the developer message"
        );
        assert_eq!(
            entry_type_counts(&file_c).get("compaction").copied(),
            Some(1),
            "compaction entry survives the second seed"
        );
        let payloads_c = message_payloads(&file_c);
        assert_eq!(payloads_c, payloads_b, "double compacted resume is stable");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn standard_message_keeps_existing_conversion_path() {
        let dir = std::env::temp_dir().join(format!(
            "omp-rs-resume-seed-std-{}-{}",
            std::process::id(),
            crate::persist::uuid_v7()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let persist = crate::persist::Persistence::for_test(&dir).unwrap();
        let user = AgentMessage::user("hello there");
        seed_resumed_message(&persist, &user, None);
        let payloads = message_payloads(&persist.file().unwrap());
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0]["role"], json!("user"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
