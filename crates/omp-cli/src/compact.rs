//! Context compaction glue around `omp-compact`: settings from `config.yml`, the model-visible
//! context rebuilt from a session file (`rebuild_context`), the summarizer call over `pa-ai`
//! (`LiveCompleter`), the manual/auto compaction commit (`compact_now`) and the post-turn
//! threshold decision (`auto_check`).
//!
//! Only the local summary path (`soft` / `context-full`) is implemented; `snapcompact`, `handoff`,
//! `shake`, remote and speculative/idle compaction are not (a `methodOrder` without `soft` therefore
//! disables automatic compaction).

use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context as _, Result};
use omp_compact::context::branch_path;
use omp_compact::settings::{calculate_context_tokens, compaction_context_tokens};
use omp_compact::tokens::CountOptions;
use omp_compact::{
    build_session_context_path, plan_boundary, plan_region, project_tokens_after, should_compact,
    synthetic_compaction_entry, ApplyOptions, BoundaryEntry, ByteEstimate, CompactionResult,
    CompactionSettings, Completer, Leaf, SummaryParams, SummaryRequest,
};
use pa_agent::types::AgentMessage;
use pa_types::ai::{AssistantMessage, Model};
use serde_json::{json, Value};

use crate::persist::{iso_from_ms, Persistence};

/// `compaction.*` from `config.yml` (flat dotted keys are honoured too), TS defaults otherwise.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn settings_from_config(config: &serde_yaml_ng::Value) -> CompactionSettings {
    let get = |key: &str| {
        config
            .get("compaction")
            .and_then(|c| c.get(key))
            .or_else(|| config.get(format!("compaction.{key}")))
    };
    let num = |key: &str| get(key).and_then(serde_yaml_ng::Value::as_f64);
    let mut s = CompactionSettings::default();
    if let Some(b) = get("enabled").and_then(serde_yaml_ng::Value::as_bool) {
        s.enabled = b;
    }
    // TS uses -1 as the "unset" sentinel (compaction-threshold.ts); only a
    // positive finite value is a real threshold, anything else clamps to -1.
    if let Some(n) = num("thresholdPercent") {
        s.threshold_percent = Some(if n.is_finite() && n > 0.0 { n } else { -1.0 });
    }
    if let Some(n) = num("thresholdTokens") {
        s.threshold_tokens = Some(if n.is_finite() && n > 0.0 { n } else { -1.0 });
    }
    if let Some(n) = num("reserveTokens").filter(|n| *n >= 0.0) {
        s.reserve_tokens = Some(n as u64);
    }
    if let Some(n) = num("keepRecentTokens").filter(|n| *n >= 0.0) {
        s.keep_recent_tokens = n as u64;
    }
    // Only the `soft` method is available here: an explicit order without it leaves nothing to run.
    if let Some(order) = get("methodOrder").and_then(serde_yaml_ng::Value::as_sequence) {
        let has_soft = order
            .iter()
            .filter_map(serde_yaml_ng::Value::as_str)
            .any(|m| matches!(m, "soft" | "context-full"));
        if !has_soft {
            s.strategy = Some("off".to_owned());
        }
    }
    s
}

/// Settings of the user's omp install (`$OMP_AGENT_DIR` or `~/.omp/agent`, `config.yml`).
#[must_use]
pub fn load_settings() -> CompactionSettings {
    let dir = omp_providers::AgentDir::default().0;
    load_settings_from(&dir)
}

/// [`load_settings`] for an explicit agent directory, via `omp-settings`
/// (registry defaults < global `config.yml`; no project layer, matching the
/// historical agent-dir-only read). Flat dotted keys (`compaction.enabled: …`
/// at top level) are a port extension still honoured underneath.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn load_settings_from(dir: &Path) -> CompactionSettings {
    let settings = omp_settings::Settings::load_from(
        Some(&dir.join("config.yml")),
        &dir.join(".no-project-layer"),
    );
    // Schema defaults equal `CompactionSettings::default()` for every key this
    // reads, so an unset (Default-provenance) key leaves the default standing.
    let unset = |id: &str| {
        settings
            .provenance(id)
            .is_none_or(|p| p == omp_settings::Provenance::Default)
    };
    let mut s = CompactionSettings::default();
    if !unset("compaction.enabled") {
        if let Some(b) = settings.get_bool("compaction.enabled") {
            s.enabled = b;
        }
    }
    // TS uses -1 as the "unset" sentinel (compaction-threshold.ts); only a
    // positive finite value is a real threshold, anything else clamps to -1.
    if !unset("compaction.thresholdPercent") {
        s.threshold_percent = Some(
            settings
                .get_number("compaction.thresholdPercent")
                .filter(|n| n.is_finite() && *n > 0.0)
                .unwrap_or(-1.0),
        );
    }
    if !unset("compaction.thresholdTokens") {
        s.threshold_tokens = Some(
            settings
                .get_number("compaction.thresholdTokens")
                .filter(|n| n.is_finite() && *n > 0.0)
                .unwrap_or(-1.0),
        );
    }
    if !unset("compaction.reserveTokens") {
        if let Some(n) = settings
            .get_number("compaction.reserveTokens")
            .filter(|n| *n >= 0.0)
        {
            s.reserve_tokens = Some(n as u64);
        }
    }
    if !unset("compaction.keepRecentTokens") {
        if let Some(n) = settings
            .get_number("compaction.keepRecentTokens")
            .filter(|n| *n >= 0.0)
        {
            s.keep_recent_tokens = n as u64;
        }
    }
    // Only the `soft` method is available here: an explicit order without it leaves nothing to run.
    if !unset("compaction.methodOrder") {
        let order = settings.get_string_array("compaction.methodOrder");
        if !order.iter().any(|m| m == "soft" || m == "context-full") {
            s.strategy = Some("off".to_owned());
        }
    }
    // Flat dotted-key extension underneath the registry values.
    if let Some(config) = std::fs::read_to_string(dir.join("config.yml"))
        .ok()
        .and_then(|text| serde_yaml_ng::from_str::<serde_yaml_ng::Value>(&text).ok())
    {
        let flat = settings_from_config(&config);
        if unset("compaction.enabled") {
            s.enabled = flat.enabled;
        }
        if unset("compaction.thresholdPercent") {
            s.threshold_percent = flat.threshold_percent;
        }
        if unset("compaction.thresholdTokens") {
            s.threshold_tokens = flat.threshold_tokens;
        }
        if unset("compaction.reserveTokens") {
            s.reserve_tokens = flat.reserve_tokens;
        }
        if unset("compaction.keepRecentTokens") {
            s.keep_recent_tokens = flat.keep_recent_tokens;
        }
        if unset("compaction.methodOrder") && flat.strategy == Some("off".to_owned()) {
            s.strategy = flat.strategy;
        }
    }
    s
}

/// The messages the model sees for `entries` (session entries without the header): the latest
/// compaction summary first, then the kept tail and everything after it.
///
/// # Errors
/// The latest compaction carries a payload this port cannot replay (remote/snapcompact/...).
pub fn rebuild_context(entries: &[Value]) -> Result<Vec<AgentMessage>> {
    let path = branch_path(entries, Leaf::Last);
    let ctx = build_session_context_path(&path).map_err(|e| anyhow!("{e}"))?;
    Ok(context_messages(&ctx))
}

/// `SessionContext` → typed agent messages: each context message is converted
/// and deserialized one at a time, so the converted LLM-shape `Value` never
/// coexists with the whole list (resume stays low-RSS).
pub fn context_messages(ctx: &omp_compact::SessionContext) -> Vec<AgentMessage> {
    ctx.messages
        .iter()
        .filter_map(|m| {
            let converted = omp_compact::convert_message(m)?;
            serde_json::from_value::<AgentMessage>(converted).ok()
        })
        .collect()
}

/// Summarizer calls over `pa-ai` against the live model.
pub struct LiveCompleter {
    pub model: Model,
    pub api_key: Option<String>,
    pub headers: Option<std::collections::BTreeMap<String, String>>,
    /// Stored credential row behind `api_key`: re-resolved per summary (expired OAuth refreshes).
    pub credential_id: Option<i64>,
}

impl Completer for LiveCompleter {
    fn complete(
        &self,
        request: SummaryRequest,
    ) -> impl Future<Output = Result<AssistantMessage>> + Send {
        let model = self.model.clone();
        let api_key = self.api_key.clone();
        let credential_id = self.credential_id;
        let headers = self.headers.clone();
        async move {
            let (api_key, _) =
                crate::adapter::fresh_stored_key(&model.provider, credential_id, api_key).await;
            let options = pa_ai::types::SimpleStreamOptions {
                base: pa_ai::types::StreamOptions {
                    max_tokens: Some(request.max_tokens),
                    api_key,
                    headers: headers.map(|h| h.into_iter().collect()),
                    ..pa_ai::types::StreamOptions::default()
                },
                // `resolveCompactionEffort`: High by default, only for reasoning models.
                reasoning: model
                    .reasoning
                    .then_some(pa_types::ai::ModelThinkingLevel::High),
                thinking_budgets: None,
                disable_reasoning: false,
                tool_choice: None,
            };
            pa_ai::complete_simple(&model, &request.context, Some(options))
                .await
                .map_err(|e| anyhow!("{e:?}"))
        }
    }
}

impl LiveCompleter {
    /// The live provider target (`set_model` swaps it, so resolve per compaction).
    ///
    /// # Errors
    /// No provider target is set.
    pub fn from_target(target: &crate::agent::Target) -> Result<Self> {
        let t = target
            .read()
            .map_err(|_| anyhow!("provider target lock poisoned"))?
            .clone()
            .ok_or_else(|| anyhow!("no provider target"))?;
        Ok(Self {
            model: t.model,
            api_key: t.api_key,
            headers: t.headers,
            credential_id: t.credential_id,
        })
    }
}

/// An injected summarizer (tests, hosts that route summaries elsewhere).
pub type SummarizeFn = std::sync::Arc<
    dyn Fn(
            SummaryRequest,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<AssistantMessage>> + Send>>
        + Send
        + Sync,
>;

/// The summarizer a compaction runs with: the live model, or an injected function.
#[derive(Clone)]
pub enum Summarizer {
    Live(std::sync::Arc<LiveCompleter>),
    Injected(SummarizeFn),
}

impl Completer for Summarizer {
    fn complete(
        &self,
        request: SummaryRequest,
    ) -> impl Future<Output = Result<AssistantMessage>> + Send {
        let this = self.clone();
        async move {
            match this {
                Self::Live(live) => live.complete(request).await,
                Self::Injected(f) => f(request).await,
            }
        }
    }
}

/// What a compaction pass needs besides the session file.
pub struct CompactParams<'a> {
    pub settings: &'a CompactionSettings,
    /// Summarizer model id (`preferredDialect`) and window.
    pub model_id: &'a str,
    pub context_window: Option<u64>,
    /// `/compact <instructions>`.
    pub custom_instructions: Option<&'a str>,
    /// System prompt + tool schema tokens, added to the `tokensAfter` projection.
    pub non_message_tokens: u64,
    /// Extension `session_before_compact` gate (see [`CompactHook`]).
    pub hook: Option<&'a dyn CompactHook>,
    /// Session+generation snapshot the commit must still match (see
    /// [`CommitGuard`]); `None` skips the commit gate (tests, offline replays).
    pub guard: Option<&'a CommitGuard>,
}

/// `session_before_compact` bridge: receives the TS-shaped `preparation` (and the journal's
/// branch entries) before the summarizer runs; `true` cancels the compaction.
pub trait CompactHook: Send + Sync {
    fn before<'a>(
        &'a self,
        preparation: Value,
        custom_instructions: Option<&'a str>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>>;
}

fn now_ms() -> u64 {
    u64::try_from(pa_agent::now_ms()).unwrap_or(0)
}

/// The cheap per-line scan for [`plan_boundary`]: top-level links plus the
/// small fields the boundary decision reads; message bodies and payloads are
/// not retained.
#[derive(serde::Deserialize)]
struct PlanHead {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(rename = "firstKeptEntryId", default)]
    first_kept: Option<String>,
    #[serde(rename = "providerReplayThroughEntryId", default)]
    replay_through: Option<String>,
    #[serde(default)]
    preserve_data: Option<PreserveHead>,
    #[serde(default)]
    message: Option<PlanHeadMessage>,
}

#[derive(serde::Deserialize)]
struct PreserveHead {
    #[serde(rename = "openaiRemoteCompaction", default)]
    openai_remote: Option<OpenAiRemoteHead>,
}

#[derive(serde::Deserialize)]
struct OpenAiRemoteHead {
    /// Only the array check is needed; the payload itself can be large and
    /// is never retained here.
    #[serde(rename = "replacementHistory", default)]
    replacement_history: Option<Value>,
}

#[derive(serde::Deserialize)]
struct PlanHeadMessage {
    #[serde(default)]
    role: Option<String>,
    #[serde(rename = "stopReason", default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<Value>,
}

/// Scan the branch's raw lines into content-free [`BoundaryEntry`] headers.
fn scan_boundary_entries(
    path: &Path,
    index: &omp_session::index::SessionIndex,
    branch: &[u32],
) -> Result<Vec<BoundaryEntry>> {
    let mut out = Vec::with_capacity(branch.len());
    index.for_each_line(path, branch, |_, raw| {
        let head: PlanHead = serde_json::from_str(raw)
            .with_context(|| format!("boundary scan: malformed entry line: {raw:.120}"))?;
        out.push(BoundaryEntry {
            entry_type: head.kind.unwrap_or_default(),
            id: head.id.unwrap_or_default(),
            replay_through: head.replay_through,
            first_kept: head.first_kept,
            preserve_data: None, // injected from the fully parsed previous entry
            remote_replacement: head
                .preserve_data
                .and_then(|p| p.openai_remote)
                .and_then(|o| o.replacement_history)
                .is_some_and(|v| v.is_array()),
            summary: head.summary,
            timestamp: head.timestamp,
            message_role: head
                .message
                .as_ref()
                .and_then(|m| m.role.as_deref().map(str::to_owned)),
            stop_reason: head.message.as_ref().and_then(|m| m.stop_reason.clone()),
            usage: head.message.and_then(|m| m.usage),
        });
        Ok(())
    })?;
    Ok(out)
}

/// The session state a compaction captured at its start and re-checked at
/// commit time (TS session-maintenance: `sessionId() !== sourceSessionId ||
/// getBranch().at(-1)?.id !== sourceLeafId` around every await).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionSnapshot {
    /// Live session file identity (`Persistence::session_id`).
    pub session_id: String,
    /// Runtime switch generation (bumped by `new_session` / `load_session*`).
    pub generation: u64,
    /// Pending `/tree` leaf move (`Persistence::leaf_divergence`).
    pub nav_leaf: Option<Option<String>>,
    /// Last journaled entry id (the append point while no move is pending).
    pub file_leaf: Option<String>,
}

/// Session+generation guard around one compaction (bug-hunt #2, TS
/// `#compactionAbortController` + the `sourceSessionId`/`sourceLeafId`
/// re-checks): captured when the compaction starts, consulted right before the
/// `compaction` entry is appended. A `/new`, `/resume`, `/tree` move, a
/// concurrent append or a cancel (Esc, a queued prompt) since capture discards
/// the pass instead of writing the summary into the wrong session/branch.
pub struct CommitGuard {
    snapshot: CompactionSnapshot,
    /// Live switch generation, read fresh at the commit check.
    switch_generation: Arc<std::sync::atomic::AtomicU64>,
    cancel: tokio_util::sync::CancellationToken,
}

impl CommitGuard {
    /// Capture the session state of `persist` plus the runtime's current
    /// switch generation.
    ///
    /// # Errors
    /// The session index cannot be read (the commit check needs the same scan).
    pub fn capture(
        persist: &Persistence,
        switch_generation: Arc<std::sync::atomic::AtomicU64>,
    ) -> Result<Self> {
        let snapshot = CompactionSnapshot::capture(
            persist,
            switch_generation.load(std::sync::atomic::Ordering::SeqCst),
        )?;
        Ok(Self {
            snapshot,
            switch_generation,
            cancel: tokio_util::sync::CancellationToken::new(),
        })
    }

    /// Cancel the pass (TS `abortCompaction`): Esc, a queued prompt, `abort`.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Whether [`Self::cancel`] arrived.
    #[cfg(test)]
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Whether the session identity survived since capture: only the session id
    /// and the switch generation matter after a successful commit (the commit
    /// itself advanced the journal tip and cleared a pending navigation move; a
    /// cancel that lost the race to the commit does not roll the pass back,
    /// matching TS).
    #[must_use]
    pub fn session_alive(&self) -> bool {
        self.snapshot.generation
            == self
                .switch_generation
                .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Resolves when [`Self::cancel`] fires (the select arm around the pass).
    pub async fn cancelled(&self) {
        self.cancel.cancelled().await
    }

    /// Re-derive the live session state and refuse to commit when anything the
    /// plan was computed from moved since capture.
    ///
    /// # Errors
    /// `"Compaction cancelled"` when the guard aborted or the live session no
    /// longer matches the captured snapshot; the index scan failed.
    pub fn check_commit(&self, persist: &Persistence) -> Result<()> {
        if self.cancel.is_cancelled() {
            return Err(anyhow!("Compaction cancelled"));
        }
        let generation = self
            .switch_generation
            .load(std::sync::atomic::Ordering::SeqCst);
        let live = CompactionSnapshot::capture(persist, generation)?;
        if live != self.snapshot {
            return Err(anyhow!("Compaction cancelled"));
        }
        Ok(())
    }
}

impl CompactionSnapshot {
    /// Capture the session state a compaction commit depends on.
    ///
    /// # Errors
    /// The session index cannot be read.
    pub fn capture(persist: &Persistence, generation: u64) -> Result<Self> {
        let index = omp_session::index::SessionIndex::load(&persist.path())?;
        Ok(Self {
            session_id: persist.session_id(),
            generation,
            nav_leaf: persist.leaf_divergence(),
            file_leaf: index
                .branch()
                .last()
                .map(|&i| index.refs[i as usize].id.to_string()),
        })
    }
}

/// The branch a compaction must summarize: the writer's live append point (TS
/// `getBranch()`, leaf-based), not the last journaled line — between a `/tree`
/// move and the next append the two differ, and cutting the file-order branch
/// would summarize the abandoned branch while the `compaction` entry lands on
/// the navigated one (bug-hunt #4).
fn compact_branch(
    index: &omp_session::index::SessionIndex,
    nav_leaf: Option<Option<&str>>,
) -> Vec<u32> {
    match nav_leaf {
        None => index.branch(),
        Some(None) => Vec::new(),
        Some(Some(id)) => branch_from_leaf(index, id).unwrap_or_else(|| index.branch()),
    }
}

/// `SessionEntryIndex.pathTo(leaf)` over the lazy index: walk `parentId` links
/// from `leaf_id` back to the root; indices into `index.refs`. `None` when the
/// id is not an entry of this file (the caller falls back to the file-order
/// branch, the pre-#4 behaviour).
fn branch_from_leaf(index: &omp_session::index::SessionIndex, leaf_id: &str) -> Option<Vec<u32>> {
    let is_structural = |i: usize| {
        let r = &index.refs[i];
        r.kind == omp_session::index::EntryKind::Header
            || (r.kind == omp_session::index::EntryKind::Title && i == 0)
    };
    let mut by_id: std::collections::HashMap<&str, u32> =
        std::collections::HashMap::with_capacity(index.refs.len());
    for (i, r) in index.refs.iter().enumerate() {
        if !is_structural(i) && !r.id.is_empty() {
            by_id.insert(&r.id, i as u32);
        }
    }
    let mut cur = *by_id.get(leaf_id)?;
    let mut path = Vec::new();
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    loop {
        if !seen.insert(cur) {
            break;
        }
        path.push(cur);
        let parent = &index.refs[cur as usize].parent_id;
        if parent.is_empty() {
            break;
        }
        match by_id.get(parent.as_ref()) {
            Some(&p) => cur = p,
            None => break,
        }
    }
    path.reverse();
    Some(path)
}

/// Plan, summarize and commit one compaction of the live session; `Ok` carries the rpc `compact`
/// result (`summary`, `shortSummary`, `firstKeptEntryId`, `tokensBefore`, `details`, `preserveData`)
/// and the post-compaction rebuild view (`compaction` entry + kept tail) for [`rebuild_context`].
///
/// The journal is scanned once for structure (index + headers) and only the
/// compactable region is fully parsed — the pre-boundary history never becomes
/// a `Value` DOM, which keeps a compaction of a large session within budget.
///
/// # Errors
/// Nothing to compact / already compacted (TS messages), summarization failure, unsupported
/// previous compaction payload, or a write failure.
pub async fn compact_now(
    persist: &Persistence,
    path: &Path,
    completer: &impl Completer,
    params: &CompactParams<'_>,
) -> Result<(CompactionResult, Value, Vec<Value>)> {
    let index = omp_session::index::SessionIndex::load(path)
        .with_context(|| format!("index {}", path.display()))?;
    // Summarize the branch the writer would append to (TS `getBranch()`), not
    // the last journaled line: after a `/tree` move the two differ, and the
    // file-order branch is the abandoned one (bug-hunt #4).
    let branch = compact_branch(
        &index,
        persist.leaf_divergence().as_ref().map(|o| o.as_deref()),
    );
    let heads = scan_boundary_entries(path, &index, &branch)?;
    let Some(mut boundary) = plan_boundary(&heads, params.settings) else {
        if heads.last().is_some_and(|h| h.entry_type == "compaction") {
            return Err(anyhow!("Already compacted"));
        }
        return Err(anyhow!("Nothing to compact (session too small)"));
    };
    // Fully parse only the compactable region (and the previous compaction
    // entry, whose preserveData the summary request carries forward).
    let mut region: Vec<Value> = Vec::new();
    index
        .for_each_line(path, &branch[boundary.boundary_start..], |_, raw| {
            region.push(
                serde_json::from_str(raw)
                    .with_context(|| format!("region parse: malformed entry line: {raw:.120}"))?,
            );
            Ok(())
        })
        .with_context(|| format!("read {}", path.display()))?;
    let previous: Option<Value> = boundary
        .prev_compaction_idx
        .map(|i| load_one(path, &index, branch[i]))
        .transpose()?;
    if let (Some(prev), b) = (&previous, &mut boundary) {
        b.previous_preserve_data = prev.get("preserveData").filter(|d| !d.is_null()).cloned();
    }
    let tokenizer = ByteEstimate;
    let (prep, kept_entries) = plan_region(
        region,
        &boundary,
        previous.as_ref(),
        params.settings,
        &tokenizer,
    )
    .map_err(|e| anyhow!("{e}"))?;
    let Some(prep) = prep else {
        return Err(anyhow!("Nothing to compact (session too small)"));
    };
    if let Some(hook) = params.hook {
        let preparation = json!({
            "firstKeptEntryId": prep.first_kept_entry_id,
            "messagesToSummarize": prep.messages_to_summarize,
            "turnPrefixMessages": prep.turn_prefix_messages,
            "recentMessages": prep.recent_messages,
            "isSplitTurn": prep.is_split_turn,
            "tokensBefore": prep.tokens_before,
            "previousSummary": prep.previous_summary,
            "previousPreserveData": prep.previous_preserve_data,
            "fileOps": {
                "read": prep.file_ops.read,
                "written": prep.file_ops.written,
                "edited": prep.file_ops.edited,
            },
        });
        if hook.before(preparation, params.custom_instructions).await {
            return Err(anyhow!("Compaction cancelled"));
        }
    }
    let mut summary_params = SummaryParams::new(now_ms()).for_model(params.model_id);
    summary_params.reserve_tokens = params
        .settings
        .reserve_tokens
        .unwrap_or(omp_compact::settings::DEFAULT_RESERVE_TOKENS);
    summary_params.context_window = params.context_window;
    summary_params.custom_instructions = params.custom_instructions;
    let result = omp_compact::compact(completer, &prep, &summary_params).await?;
    // Commit gate (TS `sessionId() !== sourceSessionId || getBranch().at(-1)?.id
    // !== sourceLeafId`): a session switch, a `/tree` move, a concurrent append
    // or an abort since the plan was captured discards the pass — the summary
    // must not land in the wrong file or on the wrong branch.
    if let Some(guard) = params.guard {
        guard.check_commit(persist)?;
    }
    let leaf_id = heads.last().map_or("", |h| h.id.as_str()).to_owned();
    let (ts, view, tokens_after) = build_rebuild_view(
        &leaf_id,
        &result,
        kept_entries,
        params.non_message_tokens,
        &tokenizer,
    );
    let options = ApplyOptions {
        method: Some("soft".to_owned()),
        tokens_after,
        from_extension: Some(false),
    };
    persist
        .append_compaction(&result, &ts, &options)
        .context("writing the compaction entry")?;
    let mut data = json!({
        "summary": result.summary,
        "shortSummary": result.short_summary,
        "firstKeptEntryId": result.first_kept_entry_id,
        "tokensBefore": result.tokens_before,
        "details": result.details,
    });
    if let Some(p) = &result.preserve_data {
        data["preserveData"] = p.clone();
    }
    Ok((result, data, view))
}

/// Build the synthetic compaction entry plus the post-compaction rebuild view
/// (the kept tail re-parented to it, mirroring the file the append writes) and
/// the projected token count for that view.
fn build_rebuild_view(
    leaf_id: &str,
    result: &omp_compact::summary::CompactionResult,
    kept_entries: Vec<Value>,
    non_message_tokens: u64,
    tokenizer: &dyn omp_compact::Tokenizer,
) -> (String, Vec<Value>, Option<u64>) {
    let ts = iso_from_ms(i64::try_from(now_ms()).unwrap_or(0));
    let rebuild_entry = synthetic_compaction_entry(
        &format!("{leaf_id}:compaction"),
        if leaf_id.is_empty() {
            None
        } else {
            Some(leaf_id)
        },
        &ts,
        result,
        Some("soft"),
    );
    let compaction_id = rebuild_entry
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let mut kept_entries = kept_entries;
    if let (Some(id), Some(first)) = (compaction_id, kept_entries.first_mut()) {
        if let Some(obj) = first.as_object_mut() {
            obj.insert("parentId".into(), id.into());
        }
    }
    let mut view: Vec<Value> = Vec::with_capacity(kept_entries.len() + 1);
    view.push(rebuild_entry);
    view.extend(kept_entries);
    let view_refs: Vec<&Value> = view.iter().collect();
    let tokens_after = project_tokens_after(
        &view_refs,
        result,
        Some("soft"),
        &ts,
        non_message_tokens,
        tokenizer,
    )
    .ok();
    (ts, view, tokens_after)
}

/// Parse one branch entry fully, on demand.
fn load_one(path: &Path, index: &omp_session::index::SessionIndex, at: u32) -> Result<Value> {
    let mut out = None;
    index.for_each_line(path, &[at], |_, raw| {
        out = Some(serde_json::from_str(raw).context("entry parse")?);
        Ok(())
    })?;
    out.ok_or_else(|| anyhow!("entry {at} vanished"))
}

/// The post-turn threshold decision (`checkCompaction`, case "threshold"): the provider-billed
/// context of the newest successful assistant turn after the latest compaction, floored by the
/// local estimate of the rebuilt context, against `resolveThresholdTokens`.
#[must_use]
/// Local token estimate of the live typed context (`rebuild_estimate`
/// equivalent: `count_message` over every context message) — the threshold
/// check's stored-conversation floor without touching the journal.
///
/// An assistant turn that predates the latest compaction carries stale usage;
/// the caller passes the newest usable usage separately.
pub fn estimate_context_tokens(messages: &[AgentMessage]) -> u64 {
    messages
        .iter()
        .map(|m| {
            serde_json::to_value(m).map_or(0, |v| {
                omp_compact::tokens::count_message(&ByteEstimate, &v, CountOptions::default())
            })
        })
        .sum()
}

/// The threshold decision (`checkCompaction`, case "threshold"): the
/// provider-billed context of the newest successful assistant turn, floored by
/// the local estimate of the live context, against `resolveThresholdTokens`.
pub fn auto_check(
    usage: Option<Value>,
    stored_estimate: u64,
    context_window: u64,
    settings: &CompactionSettings,
) -> bool {
    if !settings.enabled || context_window == 0 {
        return false;
    }
    let billed = usage.map_or(0, |v| calculate_context_tokens(&v));
    should_compact(
        compaction_context_tokens(billed, stored_estimate),
        context_window,
        settings,
    )
}

/// Copy the active branch of the session at `path` into the (fresh) live session file so
/// its `compaction` entries keep working: fresh ids chained to the writer's leaf,
/// `firstKeptEntryId` / `providerReplayThroughEntryId` remapped to the copied ids. The
/// seed works on the raw journal lines — payload bytes stay verbatim, no `Value` DOM is
/// built for the branch — which keeps resuming a large session cheap; lines the rewriter
/// cannot patch fall back to the DOM path per entry.
///
/// # Errors
/// An entry line cannot be read, rewritten or written.
pub fn seed_branch(
    persist: &Persistence,
    index: &omp_session::index::SessionIndex,
    path: &std::path::Path,
) -> Result<()> {
    let branch = index.branch();
    let mut remap = std::collections::HashMap::<String, String>::new();
    index.for_each_line(path, &branch, |r, raw| {
        let old_id = r.id.to_string();
        let new_id = persist
            .append_seeded_line(raw, &remap)
            .or_else(|_| seed_entry_dom(persist, raw, &remap))?;
        remap.insert(old_id, new_id);
        Ok(())
    })
}

/// DOM fallback for one seeded entry (rare: only lines the raw rewriter rejects).
/// Mirrors the previous seeding behaviour: fields copied from the parsed entry,
/// link fields regenerated by the writer, remap keys rewritten through `remap`.
fn seed_entry_dom(
    persist: &Persistence,
    raw: &str,
    remap: &std::collections::HashMap<String, String>,
) -> Result<String> {
    let entry: Value = serde_json::from_str(raw)?;
    let obj = entry
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("entry is not an object"))?;
    let kind = obj
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("entry has no type"))?;
    let mut fields = serde_json::Map::with_capacity(obj.len());
    for (k, v) in obj {
        if matches!(k.as_str(), "type" | "id" | "parentId" | "timestamp") {
            continue;
        }
        fields.insert(k.clone(), v.clone());
    }
    for key in ["firstKeptEntryId", "providerReplayThroughEntryId"] {
        if let Some(mapped) = fields
            .get(key)
            .and_then(Value::as_str)
            .and_then(|id| remap.get(id))
        {
            fields.insert(key.to_owned(), Value::String(mapped.clone()));
        }
    }
    let ts = obj
        .get("timestamp")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    // Rare DOM fallback (the raw rewriter rejected the line): the raw writer
    // append is invisible to the navigation mirror, so note it like every
    // other append.
    persist
        .with_writer(|w| w.append_entry(kind, Value::Object(fields), &ts))
        .inspect(|_| persist.note_append())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
    use pa_agent::scripted::ScriptedProvider;
    use tokio::sync::mpsc::UnboundedReceiver;

    use super::*;
    use crate::adapter::{json_round_trip, ProviderTarget};
    use crate::runtime::Runtime;

    fn pa_model() -> Model {
        serde_json::from_value(json!({
            "id": "a", "name": "a", "api": "openai-completions", "provider": "p",
            "baseUrl": "http://127.0.0.1:9", "reasoning": false, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 100_000, "maxTokens": 4096
        }))
        .expect("model json")
    }

    fn reply(text: &str) -> AssistantMessage {
        serde_json::from_value(json!({
            "content": [{"type": "text", "text": text}], "api": "openai-completions", "provider": "p", "model": "a",
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
            "stopReason": "stop", "timestamp": 1
        }))
        .expect("reply json")
    }

    /// An injected summarizer answering every request with `text`, counting the calls.
    fn fake_summarizer(text: &'static str) -> (SummarizeFn, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let f: SummarizeFn = Arc::new(move |_req| {
            seen.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Ok(reply(text)) })
        });
        (f, calls)
    }

    struct Harness {
        rt: Arc<Runtime>,
        rx: UnboundedReceiver<Value>,
    }

    async fn harness(turns: usize) -> Harness {
        harness_in(turns, None).await
    }

    /// `dir = Some(..)`: a real (non-ephemeral) session file inside `dir`.
    async fn harness_in(turns: usize, dir: Option<&std::path::Path>) -> Harness {
        let model = pa_model();
        let provider = Arc::new(ScriptedProvider::new(
            json_round_trip(&model).expect("agent model"),
        ));
        for i in 0..turns {
            provider.push_text_turn(&format!("answer {i} {}", "y".repeat(400)));
        }
        let agent = Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                system_prompt: Some("test".into()),
                model: json_round_trip(&model),
                thinking_level: Some(pa_agent::types::ThinkingLevel::Off),
                tools: Some(Vec::new()),
                messages: None,
            },
            stream_fn: Some(provider.stream_fn()),
            ..AgentOptions::default()
        });
        let cwd = std::env::temp_dir();
        let persist = match dir {
            Some(d) => {
                crate::persist::attach_in(&agent, &cwd, d, None, crate::persist::uuid_v7()).await
            }
            None => crate::persist::attach_with(&agent, &cwd, true).await,
        }
        .expect("session");
        let target = Arc::new(std::sync::RwLock::new(Some(ProviderTarget {
            api_key: Some("k".into()),
            model,
            service_tier: None,
            headers: None,
            credential_id: None,
        })));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let rt = Runtime::start(Arc::new(agent), target, cwd, persist, tx).await;
        // keep the provider alive for the runtime's lifetime
        std::mem::forget(provider);
        Harness { rt, rx }
    }

    async fn until_settled(rx: &mut UnboundedReceiver<Value>) -> Vec<Value> {
        let mut out = Vec::new();
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), rx.recv())
                .await
                .expect("frame in time")
                .expect("open");
            let done = f["type"] == "session_settled";
            out.push(f);
            if done {
                return out;
            }
        }
    }

    fn small_keep() -> CompactionSettings {
        CompactionSettings {
            keep_recent_tokens: 100,
            ..CompactionSettings::default()
        }
    }

    async fn talk(h: &mut Harness, prompts: usize) {
        for i in 0..prompts {
            h.rt.prompt(
                None,
                &format!("question {i} {}", "x".repeat(400)),
                None,
                Vec::new(),
            )
            .await
            .expect("prompt");
            until_settled(&mut h.rx).await;
        }
    }

    fn journal(path: &std::path::Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn jsonl_count(dir: &std::path::Path) -> usize {
        std::fs::read_dir(dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|x| x == "jsonl")
            })
            .count()
    }

    /// TS `SessionManager.open`: resume appends to the SAME file (header untouched, new entries
    /// chained to the previous tip) and leaves no orphan startup file behind.
    #[tokio::test]
    async fn resume_continues_same_session_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut first = harness_in(2, Some(dir.path())).await;
        talk(&mut first, 1).await;
        let file = first.rt.session_file();
        let before = journal(&file);
        let tip = before.last().unwrap()["id"].as_str().unwrap().to_owned();
        let header = before[0].clone();
        assert_eq!(jsonl_count(dir.path()), 1);

        // A second process: its startup file exists until the resume replaces it.
        let mut second = harness_in(2, Some(dir.path())).await;
        assert_eq!(jsonl_count(dir.path()), 2);
        second.rt.load_session(&file, None).await.expect("resume");
        assert_eq!(second.rt.session_file(), file, "same file");
        assert_eq!(second.rt.session_id(), header["id"].as_str().unwrap());
        assert_eq!(
            jsonl_count(dir.path()),
            1,
            "startup file dropped, none forked"
        );
        assert_eq!(
            journal(&file).len(),
            before.len(),
            "resume itself appends nothing"
        );

        talk(&mut second, 1).await;
        let after = journal(&file);
        assert!(
            after.len() > before.len(),
            "new turn landed in the same file"
        );
        assert_eq!(after[0], header, "header untouched");
        assert_eq!(&after[..before.len()], &before[..], "old entries untouched");
        assert_eq!(
            after[before.len()]["parentId"],
            tip,
            "first new entry chains to the old tip"
        );
        for pair in after[before.len()..].windows(2) {
            assert_eq!(pair[1]["parentId"], pair[0]["id"], "linear chain");
        }
        assert_eq!(jsonl_count(dir.path()), 1);
    }

    /// `--fork` keeps TS fork semantics: a new file, the source untouched.
    #[tokio::test]
    async fn fork_still_creates_new_session_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut first = harness_in(1, Some(dir.path())).await;
        talk(&mut first, 1).await;
        let file = first.rt.session_file();
        let before = journal(&file);

        let second = harness_in(1, Some(dir.path())).await;
        second
            .rt
            .fork_session(&file, Some(dir.path()))
            .await
            .expect("fork");
        assert_ne!(second.rt.session_file(), file, "fork writes a new file");
        assert_eq!(journal(&file), before, "source untouched");
        assert_eq!(jsonl_count(dir.path()), 2, "source + fork, no orphan");
    }

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../omp-compact/tests/fixtures/compacted-session.jsonl")
    }

    fn user_text(m: &AgentMessage) -> String {
        serde_json::to_value(m).unwrap()["content"].to_string()
    }

    #[test]
    fn config_yaml_maps_to_settings() {
        let cfg: serde_yaml_ng::Value = serde_yaml_ng::from_str("compaction:\n  enabled: false\n  thresholdPercent: 70\n  reserveTokens: 5000\n  keepRecentTokens: 123\n  methodOrder: [snapcompact, shake]\n").unwrap();
        let s = settings_from_config(&cfg);
        assert!(!s.enabled);
        assert_eq!(
            (s.threshold_percent, s.reserve_tokens, s.keep_recent_tokens),
            (Some(70.0), Some(5000), 123)
        );
        assert_eq!(
            s.strategy.as_deref(),
            Some("off"),
            "no soft method configured"
        );
        let defaults = settings_from_config(
            &serde_yaml_ng::from_str("compaction:\n  methodOrder: [snapcompact, soft]\n").unwrap(),
        );
        assert!(defaults.enabled && defaults.strategy.as_deref() == Some("context-full"));
    }

    /// Negative/zero/non-finite thresholds clamp to the TS `-1` "unset"
    /// sentinel instead of being accepted verbatim (`ReviewWave3`).
    #[test]
    fn config_negative_thresholds_clamp_to_unset() {
        let cfg: serde_yaml_ng::Value = serde_yaml_ng::from_str(
            "compaction:\n  thresholdPercent: -20\n  thresholdTokens: -1\n",
        )
        .unwrap();
        let s = settings_from_config(&cfg);
        assert_eq!(s.threshold_percent, Some(-1.0));
        assert_eq!(s.threshold_tokens, Some(-1.0));
        let zero: serde_yaml_ng::Value =
            serde_yaml_ng::from_str("compaction:\n  thresholdPercent: 0\n").unwrap();
        assert_eq!(settings_from_config(&zero).threshold_percent, Some(-1.0));
        let ok: serde_yaml_ng::Value = serde_yaml_ng::from_str(
            "compaction:\n  thresholdPercent: 70\n  thresholdTokens: 90000\n",
        )
        .unwrap();
        let s = settings_from_config(&ok);
        assert_eq!(s.threshold_percent, Some(70.0));
        assert_eq!(s.threshold_tokens, Some(90_000.0));
    }

    #[tokio::test]
    async fn resume_of_compacted_session_rebuilds_summary_plus_tail() {
        let h = harness(0).await;
        let raw = crate::runtime::read_entries(&fixture()).expect("fixture");
        let expected = rebuild_context(&raw).expect("rebuild");
        assert!(
            user_text(&expected[0]).contains("<summary>"),
            "summary comes first: {}",
            user_text(&expected[0])
        );
        assert!(expected.len() >= 2);

        h.rt.fork_session(&fixture(), None).await.expect("load");
        let live = h.rt.messages().await;
        assert_eq!(
            live.len(),
            expected.len(),
            "model sees the rebuilt context, not the raw log"
        );
        assert!(live[0].to_string().contains("<summary>"));

        // The seeded file keeps the compaction usable: same context when rebuilt from it, and
        // `firstKeptEntryId` points at a copied entry.
        let seeded = h.rt.entries().expect("entries");
        let compaction = seeded
            .iter()
            .find(|e| e["type"] == "compaction")
            .expect("compaction copied");
        let kept = compaction["firstKeptEntryId"].as_str().unwrap();
        assert!(
            seeded.iter().any(|e| e["id"] == kept),
            "firstKeptEntryId remapped to the new file"
        );
        assert_eq!(rebuild_context(&seeded).unwrap().len(), expected.len());
    }

    /// s5 regression (39.5 MB real session): resuming a session whose latest
    /// compaction carries a provider-native (`openaiRemoteCompaction`)
    /// `preserveData`, then compacting locally, must not re-stamp the stale
    /// remote payload onto the new entry — the context rebuild trusts that
    /// payload for the kept tail and replays nothing, which surfaced as
    /// `Error: no assistant response` on resume.
    #[tokio::test]
    async fn resume_over_remote_compaction_then_local_compact_keeps_tail() {
        let dir = std::env::temp_dir().join(format!("omp-rs-s5-regression-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("s5-redacted.jsonl");
        let msg = |id: &str, parent: &str, ts: &str, role: &str, text: &str| {
            format!(
                r#"{{"type":"message","id":"{id}","parentId":"{parent}","timestamp":"{ts}","message":{{"role":"{role}","content":[{{"type":"text","text":"{text}"}}],"stopReason":"stop","timestamp":1}}}}"#
            )
        };
        let body = [
            r#"{"type":"session","version":3,"id":"s5redacted","timestamp":"2026-09-11T00:31:09.193Z","cwd":"/tmp"}"#.to_string(),
            msg("u1", "s5redacted", "2026-09-11T00:33:01.083Z", "user", "first request"),
            msg("a1", "u1", "2026-09-11T00:33:05.083Z", "assistant", "first reply"),
            // Previous provider-native (remote) compaction boundary, as written
            // by omp for the real s5 session (payload redacted).
            r#"{"type":"compaction","id":"c9","parentId":"a1","timestamp":"2026-09-11T01:30:12.437Z","summary":"Remote compaction preserved provider-native history for this session.","shortSummary":"remote","firstKeptEntryId":"u1","tokensBefore":"90000","tokensAfter":"9000","method":"remote","preserveData":{"openaiRemoteCompaction":{"version":"v2","provider":"openai","replacementHistory":[{"id":"cmp_x"}]},"snapcompact":{"historyBlocks":[{"id":"f1"}],"frameDataBytes":3}}}"#.to_string(),
            msg("u2", "c9", "2026-09-11T02:00:00.000Z", "user", &format!("after-boundary request {}", "x".repeat(200))),
            msg("a2", "u2", "2026-09-11T02:00:05.000Z", "assistant", &format!("after-boundary reply {}", "y".repeat(200))),
        ]
        .join("\n");
        std::fs::write(&file, body).unwrap();

        let h = harness(0).await;
        let (fake, _) = fake_summarizer("LOCAL SUMMARY");
        h.rt.set_summarizer(Some(fake));
        h.rt.set_compaction_settings(CompactionSettings {
            enabled: false,
            ..small_keep()
        });

        // Resume: the rebuilt context is the remote summary plus the tail.
        h.rt.load_session(&file, None)
            .await
            .expect("load s5-shaped session");
        let live = h.rt.messages().await;
        assert!(
            !live.is_empty(),
            "resume of the s5-shaped session yields a context"
        );
        assert!(
            live.iter()
                .any(|m| m.to_string().contains("after-boundary request")),
            "kept tail is part of the resumed context: {live:?}"
        );

        // A local compaction of the resumed session must not carry the stale
        // remote payload forward; the rebuilt context keeps the tail.
        h.rt.compact(None).await.expect("local compaction");
        let entries = h.rt.entries().expect("entries");
        let new_compaction = entries
            .iter()
            .rev()
            .find(|e| e["type"] == "compaction")
            .expect("new compaction entry");
        let preserve = &new_compaction["preserveData"];
        assert!(
            preserve.get("openaiRemoteCompaction").is_none() || preserve.is_null(),
            "stale remote payload must not be re-stamped: {preserve}"
        );
        let rebuilt = rebuild_context(&entries).expect("rebuild after local compaction");
        assert!(
            rebuilt.iter().any(|m| serde_json::to_string(m)
                .unwrap_or_default()
                .contains("after-boundary reply")),
            "kept tail survives the local compaction: {rebuilt:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The lazy resume shims the pre-compaction history (link-only stand-ins)
    /// but must rebuild the exact context the full-parse rebuild produces.
    #[tokio::test]
    async fn resume_with_lazy_shims_matches_full_parse_rebuild() {
        let dir = std::env::temp_dir().join(format!("omp-rs-lazy-shim-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("lazy-shim.jsonl");
        let msg = |id: &str, parent: &str, ts: &str, role: &str, text: &str| {
            format!(
                r#"{{"type":"message","id":"{id}","parentId":"{parent}","timestamp":"{ts}","message":{{"role":"{role}","provider":"p","model":"a","content":[{{"type":"text","text":"{text}"}}],"stopReason":"stop","timestamp":1}}}}"#
            )
        };
        let mut body = vec![
            r#"{"type":"session","version":3,"id":"hdr","timestamp":"2026-10-01T00:00:00.000Z","cwd":"/tmp"}"#.to_string(),
        ];
        // Pre-compaction history: long-abandoned turns, including an assistant
        // turn that seeded the default model inference.
        let mut prev = "hdr".to_string();
        for i in 0..30 {
            let id = format!("pre{i:02}");
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            body.push(msg(
                &id,
                &prev,
                "2026-10-01T00:01:00.000Z",
                role,
                &format!("history turn {i} {}", "h".repeat(200)),
            ));
            prev = id;
        }
        body.push(msg(
            "kept1",
            &prev,
            "2026-10-01T00:02:00.000Z",
            "user",
            "kept request",
        ));
        body.push(msg(
            "kept2",
            "kept1",
            "2026-10-01T00:02:05.000Z",
            "assistant",
            "kept reply",
        ));
        body.push(
            r#"{"type":"compaction","id":"cmp1","parentId":"kept2","timestamp":"2026-10-01T00:03:00.000Z","summary":"SHIM SUMMARY","shortSummary":"shim","firstKeptEntryId":"kept1","tokensBefore":"9000","tokensAfter":"900","method":"soft","details":{"kind":"local"}}"#
                .to_string(),
        );
        body.push(msg(
            "post1",
            "cmp1",
            "2026-10-01T00:04:00.000Z",
            "user",
            "post request",
        ));
        body.push(msg(
            "post2",
            "post1",
            "2026-10-01T00:04:05.000Z",
            "assistant",
            "post reply",
        ));
        std::fs::write(&file, body.join("\n")).unwrap();

        let expected = rebuild_context(&crate::runtime::read_entries(&file).unwrap()).unwrap();
        let h = harness(0).await;
        h.rt.load_session(&file, None).await.expect("lazy resume");
        let live = h.rt.messages().await;
        assert_eq!(
            live.len(),
            expected.len(),
            "shimmed rebuild == full-parse rebuild"
        );
        assert!(
            live.iter().any(|m| m.to_string().contains("SHIM SUMMARY")),
            "summary first: {live:?}"
        );
        assert!(
            live.iter().any(|m| m.to_string().contains("post request")),
            "post-compaction tail kept: {live:?}"
        );
        assert!(
            live.iter().all(|m| !m.to_string().contains("history turn")),
            "pre-compaction history stays summarized: {live:?}"
        );
        // The seeded continuation file must rebuild to the same context.
        let seeded = h.rt.entries().expect("seeded entries");
        assert_eq!(rebuild_context(&seeded).unwrap().len(), expected.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn forced_compaction_with_fake_summarizer_commits_entry_and_swaps_context() {
        let mut h = harness(3).await;
        h.rt.set_compaction_settings(CompactionSettings {
            enabled: false,
            ..small_keep()
        });
        let (fake, calls) = fake_summarizer("FAKE SUMMARY");
        h.rt.set_summarizer(Some(fake));
        talk(&mut h, 3).await;
        let before = h.rt.messages().await.len();
        assert_eq!(before, 6);

        let result = h.rt.compact(None).await.expect("compact");
        assert!(
            result["summary"].as_str().unwrap().contains("FAKE SUMMARY"),
            "{result}"
        );
        assert!(calls.load(Ordering::SeqCst) >= 1);
        let entries = h.rt.entries().unwrap();
        let comp = entries
            .iter()
            .find(|e| e["type"] == "compaction")
            .expect("compaction entry");
        assert_eq!(comp["method"], "soft");
        assert_eq!(comp["firstKeptEntryId"], result["firstKeptEntryId"]);
        assert!(
            entries.iter().any(|e| e["id"] == comp["firstKeptEntryId"]),
            "kept entry exists"
        );
        assert!(comp["tokensAfter"].as_u64().is_some());

        let after = h.rt.messages().await;
        assert!(
            after.len() < before + 1 && after[0].to_string().contains("FAKE SUMMARY"),
            "{after:?}"
        );
        // a second manual compaction right away is a no-op
        let again = h.rt.compact(None).await.unwrap_err();
        assert!(
            again.message.contains("Already compacted")
                || again.message.contains("Nothing to compact"),
            "{}",
            again.message
        );
    }

    #[tokio::test]
    async fn auto_compaction_fires_after_a_turn_over_threshold_and_emits_events() {
        let mut h = harness(4).await;
        let (fake, _) = fake_summarizer("AUTO SUMMARY");
        h.rt.set_summarizer(Some(fake));
        h.rt.set_compaction_settings(CompactionSettings {
            enabled: false,
            ..small_keep()
        });
        talk(&mut h, 3).await;
        assert!(
            h.rt.entries()
                .unwrap()
                .iter()
                .all(|e| e["type"] != "compaction"),
            "disabled: no compaction yet"
        );

        h.rt.set_compaction_settings(CompactionSettings {
            threshold_tokens: Some(200.0),
            ..small_keep()
        });
        h.rt.prompt(
            Some(json!("p4")),
            &format!("question 4 {}", "x".repeat(400)),
            None,
            Vec::new(),
        )
        .await
        .expect("prompt");
        let frames = until_settled(&mut h.rx).await;
        let kinds: Vec<&str> = frames
            .iter()
            .map(|f| f["type"].as_str().unwrap_or("?"))
            .filter(|k| {
                k.starts_with("auto_")
                    || matches!(*k, "agent_end" | "prompt_result" | "session_settled")
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "agent_end",
                "auto_compaction_start",
                "auto_compaction_end",
                "prompt_result",
                "session_settled"
            ],
            "{kinds:?}"
        );
        let start = frames
            .iter()
            .find(|f| f["type"] == "auto_compaction_start")
            .unwrap();
        assert_eq!(
            (start["reason"].clone(), start["action"].clone()),
            (json!("threshold"), json!("context-full"))
        );
        let end = frames
            .iter()
            .find(|f| f["type"] == "auto_compaction_end")
            .unwrap();
        assert_eq!(
            (end["aborted"].clone(), end["willRetry"].clone()),
            (json!(false), json!(false))
        );
        assert!(
            end["result"]["summary"]
                .as_str()
                .unwrap()
                .contains("AUTO SUMMARY"),
            "{end}"
        );
        assert!(h
            .rt
            .entries()
            .unwrap()
            .iter()
            .any(|e| e["type"] == "compaction"));
        assert!(h.rt.messages().await[0]
            .to_string()
            .contains("AUTO SUMMARY"));
    }

    // ------------------------------------------------------- bug-hunt #4

    /// A capturing summarizer: records every request context (conversation
    /// text included) and answers with `text`.
    fn capturing_summarizer(
        text: &'static str,
    ) -> (SummarizeFn, Arc<std::sync::Mutex<Vec<String>>>) {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = seen.clone();
        let f: SummarizeFn = Arc::new(move |req: SummaryRequest| {
            let sink = sink.clone();
            Box::pin(async move {
                sink.lock()
                    .unwrap()
                    .push(serde_json::to_string(&req.context).unwrap_or_default());
                Ok(reply(text))
            })
        });
        (f, seen)
    }

    /// `/compact` right after a `/tree` navigation must summarize the branch
    /// the writer would append to (the selected one), not the file-order tail
    /// (the abandoned one), and the `compaction` entry must chain to the
    /// navigated leaf (bug-hunt #4).
    #[tokio::test]
    async fn compact_after_tree_navigation_compacts_the_selected_branch() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("branched.jsonl");
        let msg = |id: &str, parent: &str, role: &str, text: &str| {
            format!(
                r#"{{"type":"message","id":"{id}","parentId":"{parent}","timestamp":"2026-09-11T00:33:0{id}.000Z","message":{{"role":"{role}","content":[{{"type":"text","text":"{text}"}}],"stopReason":"stop","timestamp":1}}}}"#
            )
        };
        let long_x = "x".repeat(400);
        let long_y = "y".repeat(400);
        let body = [
            r#"{"type":"session","version":3,"id":"branched","timestamp":"2026-09-11T00:31:09.193Z","cwd":"/tmp"}"#.to_string(),
            msg("u1", "branched", "user", &format!("first request {long_x}")),
            msg("a1", "u1", "assistant", &format!("first reply {long_y}")),
            msg("u2", "a1", "user", &format!("second request {long_x}")),
            msg("a2", "u2", "assistant", &format!("second reply {long_y}")),
            msg("u3", "a2", "user", &format!("third request {long_x}")),
            msg("a3", "u3", "assistant", &format!("third reply {long_y}")),
        ]
        .join("\n");
        std::fs::write(&file, format!("{body}\n")).unwrap();

        let h = harness_in(0, Some(dir.path())).await;
        h.rt.load_session(&file, None).await.expect("resume");
        // `/tree` navigate to a1 without a summary: an in-memory leaf move,
        // nothing journaled — the file still ends on a3.
        h.rt.branch_to(Some("a1".to_owned())).expect("branch_to a1");

        let (fake, seen) = capturing_summarizer("SELECTED SUMMARY");
        h.rt.set_summarizer(Some(fake));
        h.rt.set_compaction_settings(small_keep());

        h.rt.compact(None)
            .await
            .expect("compact the selected branch");

        let entries = h.rt.entries().expect("entries");
        let comp = entries
            .iter()
            .rev()
            .find(|e| e["type"] == "compaction")
            .expect("compaction entry");
        assert_eq!(
            comp["parentId"],
            json!("a1"),
            "entry chains to the navigated leaf: {comp}"
        );

        let requests = seen.lock().unwrap().join("\n");
        assert!(
            requests.contains("first request"),
            "the selected branch is summarized: {requests:.600}"
        );
        assert!(
            !requests.contains("second request") && !requests.contains("third request"),
            "the abandoned branch is not summarized: {requests:.600}"
        );
    }

    /// `compact_branch` targets the writer's navigation state (TS `getBranch()`
    /// is leaf-based): a pending move selects the moved-to branch, a pending
    /// root reset selects nothing, and an unknown id falls back to the
    /// file-order branch.
    #[test]
    fn compact_branch_follows_the_writer_navigation_state() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("b.jsonl");
        let entry = |id: &str, parent: &str| {
            format!(
                r#"{{"type":"message","id":"{id}","parentId":"{parent}","timestamp":"t","message":{{"role":"user","content":[{{"type":"text","text":"{id}"}}],"stopReason":"stop","timestamp":1}}}}"#
            )
        };
        let body = [
            r#"{"type":"session","version":3,"id":"s","timestamp":"t","cwd":"/tmp"}"#,
            &entry("u1", "s"),
            &entry("a1", "u1"),
            &entry("u2", "a1"),
            &entry("a2", "u2"),
        ]
        .join("\n")
            + "\n";
        std::fs::write(&file, body).unwrap();
        let index = omp_session::index::SessionIndex::load(&file).unwrap();
        assert_eq!(
            index.branch(),
            vec![1, 2, 3, 4],
            "file order: u1 a1 u2 a2 (refs[0] is the header)"
        );

        // No pending move: the file-order branch.
        assert_eq!(compact_branch(&index, None), vec![1, 2, 3, 4]);
        // Pending move to a1: the selected branch only.
        assert_eq!(compact_branch(&index, Some(Some("a1"))), vec![1, 2]);
        // Pending root reset: nothing to compact (TS getBranch() on a
        // before-root leaf is empty).
        assert_eq!(compact_branch(&index, Some(None)), Vec::<u32>::new());
        // Unknown id: fall back to the pre-#4 behaviour instead of guessing.
        assert_eq!(compact_branch(&index, Some(Some("nope"))), vec![1, 2, 3, 4]);
    }

    /// The commit guard: a fresh snapshot commits; a `/tree` move, any append,
    /// a session switch (generation bump) or a cancel discards the pass.
    #[test]
    fn commit_guard_discards_stale_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let persist = Persistence::for_test(dir.path()).expect("persist");
        let gen = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let e1 = persist
            .with_writer(|w| w.append_entry("custom", json!({}), "t"))
            .expect("entry");

        let guard = CommitGuard::capture(&persist, Arc::clone(&gen)).expect("capture");
        assert!(
            guard.check_commit(&persist).is_ok(),
            "fresh snapshot commits"
        );
        assert!(guard.session_alive());

        // A pending `/tree` move moves the append point: stale.
        persist.branch_to(Some(e1.clone())).expect("branch_to");
        assert!(guard.check_commit(&persist).is_err(), "navigation discards");

        // Any append ends the pending move AND advances the tip: stale either way.
        persist.append_entry("custom", json!({})).expect("append");
        assert!(guard.check_commit(&persist).is_err(), "append discards");

        // A session switch (generation bump) discards — and the post-commit
        // apply check reports the session as gone.
        let guard2 = CommitGuard::capture(&persist, Arc::clone(&gen)).expect("capture2");
        assert!(guard2.check_commit(&persist).is_ok());
        gen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        assert!(
            guard2.check_commit(&persist).is_err(),
            "generation discards"
        );
        assert!(!guard2.session_alive(), "switched session is not alive");

        // An explicit cancel (Esc / queued prompt) discards.
        let guard3 = CommitGuard::capture(&persist, Arc::clone(&gen)).expect("capture3");
        guard3.cancel();
        assert!(guard3.is_cancelled());
        assert!(guard3.check_commit(&persist).is_err(), "cancel discards");
    }
}
