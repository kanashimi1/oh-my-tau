//! Manual and automatic compaction driving the session summary.

use anyhow::Result;
use pa_agent::types::{AgentMessage, Message, StopReason};
use serde_json::{json, Value};
use std::sync::Arc;

use super::prompt::AUTO_CONTINUE_PROMPT;
use super::session::seed_resumed_message;
use super::usage::prompt_overhead_chars;
use super::{lock, rss_kb, CmdError, CompactionHooks, Runtime};

/// Adapts [`CompactionHooks`] to the per-compaction [`crate::compact::CompactHook`] (adds the branch).
struct HookWithBranch<'a> {
    hooks: &'a dyn CompactionHooks,
    branch: Vec<Value>,
}

impl crate::compact::CompactHook for HookWithBranch<'_> {
    fn before<'a>(
        &'a self,
        preparation: Value,
        custom_instructions: Option<&'a str>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>> {
        self.hooks
            .before(preparation, self.branch.clone(), custom_instructions)
    }
}

impl Runtime {
    /// Replace the `compaction.*` policy (default: `config.yml`).
    #[allow(dead_code)]
    pub fn set_compaction_settings(&self, settings: omp_compact::CompactionSettings) {
        lock(&self.settings).auto_compaction = settings.enabled;
        *lock(&self.compaction) = settings;
    }

    /// Route compaction summaries through `f` instead of the live model.
    #[allow(dead_code)]
    pub fn set_summarizer(&self, f: Option<crate::compact::SummarizeFn>) {
        *lock(&self.summarizer) = f;
    }

    async fn summarizer(&self) -> Result<crate::compact::Summarizer> {
        if let Some(f) = lock(&self.summarizer).clone() {
            return Ok(crate::compact::Summarizer::Injected(f));
        }
        // Cheap per-compaction model (`OMP_COMPACTION_MODEL` overrides the default);
        // fall back to the live target when it cannot be resolved.
        if let Ok(selected) =
            crate::model::select(Some(&crate::compact::compaction_model_ref())).await
        {
            let mut completer = crate::compact::LiveCompleter::from_selected(selected);
            if completer.model.provider == "opencode-go" {
                let session = format!("{}-compaction", self.persist.session_id());
                completer
                    .headers
                    .get_or_insert_with(std::collections::BTreeMap::new)
                    .insert("x-opencode-session".to_owned(), session);
            }
            return Ok(crate::compact::Summarizer::Live(Arc::new(completer)));
        }
        Ok(crate::compact::Summarizer::Live(Arc::new(
            crate::compact::LiveCompleter::from_target(&self.target)?,
        )))
    }

    fn live_window(&self) -> u64 {
        self.target
            .read()
            .as_ref()
            .map_or(0, |t| t.model.context_window)
    }

    /// rpc `compact`: summarize the older history into a `compaction` entry and swap the agent's
    /// messages for the rebuilt context. Returns the `CompactionResult` the response carries.
    ///
    /// A live turn is aborted first and, once the summary commits, resumed (TS
    /// `SessionMaintenance.compact` → `#scheduleInterruptedTurnResume`): queued
    /// messages drive the resume, otherwise the hidden auto-continue prompt does.
    /// Without it an extension's mid-run `ctx.compact()` (compact-gate step-gate)
    /// ended the run as "Request was aborted" and left the agent idle.
    pub async fn compact(
        self: &Arc<Self>,
        custom_instructions: Option<&str>,
    ) -> Result<Value, CmdError> {
        use std::sync::atomic::Ordering::SeqCst;
        if self.compacting.swap(true, SeqCst) {
            return Err(CmdError::new("Compaction already in progress"));
        }
        let interrupted = self.agent.is_streaming().await;
        if interrupted {
            self.compaction_abort.store(true, SeqCst);
            self.persist.drop_empty_aborted(true);
        }
        self.agent.abort();
        self.agent.wait_for_idle().await;
        if interrupted {
            self.persist.drop_empty_aborted(false);
            self.compaction_abort.store(false, SeqCst);
            self.compaction_stub_start.store(false, SeqCst);
            // The stub never reached the journal; keep the live context in step.
            self.agent
                .mutate_messages(|m| {
                    if m.last().is_some_and(crate::persist::is_empty_aborted) {
                        m.pop();
                    }
                })
                .await;
        }
        let out = self.compact_inner(custom_instructions).await;
        self.compacting.store(false, SeqCst);
        // Release prompts parked in `cancel_compaction_and_wait`.
        self.compaction_done.notify_waiters();
        // No-op rejections (too small / already compacted) change no history, so
        // the cut turn resumes on them too; a hook cancel or a summarizer failure
        // does not (TS `rejectedAsNoOp`).
        let committed_or_noop = match &out {
            Ok(_) => true,
            Err(e) => {
                e.message.starts_with("Nothing to compact") || e.message == "Already compacted"
            }
        };
        if interrupted
            && committed_or_noop
            && !self.user_interrupt.load(SeqCst)
            && omp_settings::Settings::load(&self.cwd)
                .get("compaction.autoContinue")
                .as_bool()
                != Some(false)
        {
            self.resume_after_compaction();
        }
        out
    }

    /// TS `#scheduleCompactionContinuation` for a turn a manual compaction cut
    /// mid-run: a queued steer/follow-up continues the run, otherwise the
    /// hidden `auto-continue` developer prompt does.
    fn resume_after_compaction(self: &Arc<Self>) {
        let rt = self.clone();
        tokio::spawn(async move {
            rt.agent.wait_for_idle().await;
            if rt.agent.is_streaming().await {
                return; // a prompt parked on the compaction already took the session
            }
            if rt.agent.has_queued_messages() {
                let _ = rt.agent.continue_run_until_accepted().await;
                return;
            }
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
            let nudge = AgentMessage::Custom(pa_agent::types::CustomAgentMessage {
                role: "developer".to_string(),
                payload: json!({
                    "customType": "auto-continue",
                    "content": [{"type": "text", "text": AUTO_CONTINUE_PROMPT}],
                    "display": false,
                    "attribution": "agent",
                    "synthetic": true,
                    "timestamp": timestamp,
                }),
            });
            rt.agent.append_messages(vec![nudge.clone()]).await;
            seed_resumed_message(&rt.persist, &nudge, None);
            if let Err(e) = rt.continue_after_retry().await {
                eprintln!("omt: post-compaction resume failed: {e:#}");
            }
        });
    }

    /// One compaction pass with its session+generation guard: captured at
    /// start, the commit proceeds only while the session, the switch
    /// generation, the navigation state and the journal tip are unchanged;
    /// Esc / a queued prompt cancel it outright (bug-hunt #2).
    async fn compact_inner(&self, custom_instructions: Option<&str>) -> Result<Value, CmdError> {
        let guard = Arc::new(crate::compact::CommitGuard::capture(
            &self.persist,
            Arc::clone(&self.switch_generation),
        )?);
        *lock(&self.compaction_guard) = Some(Arc::clone(&guard));
        let out = self
            .compact_inner_guarded(custom_instructions, &guard)
            .await;
        *lock(&self.compaction_guard) = None;
        out
    }

    async fn compact_inner_guarded(
        &self,
        custom_instructions: Option<&str>,
        guard: &crate::compact::CommitGuard,
    ) -> Result<Value, CmdError> {
        let trace = std::env::var_os("OMP_RS_TURN_TRACE").is_some();
        turn_trace!("[turn-trace] compact-begin rss={}kB", rss_kb());
        let summarizer = self.summarizer().await?;
        // Only the prompt overhead and the model id are needed from the state:
        // release the snapshot (a full copy of the live context) before the
        // summarization, which is the compaction's memory peak.
        let (non_message_tokens, model_id) = {
            let (st, _) = self.agent.status().await;
            (
                u64::try_from(prompt_overhead_chars(&st.system_prompt, &st.tools).div_ceil(4))
                    .unwrap_or(0),
                st.model.id,
            )
        };
        let settings = lock(&self.compaction).clone();
        let hooks = lock(&self.compaction_hooks).clone();
        // The branch extensions see matches what the pass will summarize: the
        // writer's live append point, not the file-order tail (bug-hunt #4).
        let nav_leaf = self.persist.leaf_divergence();
        let bridge = hooks.as_deref().map(|h| HookWithBranch {
            hooks: h,
            branch: self
                .entries()
                .ok()
                .map(|entries| {
                    let leaf = match nav_leaf.as_ref() {
                        Some(Some(id)) => Some(id.clone()),
                        Some(None) => None,
                        None => omp_session::tree::leaf_id(&entries).map(str::to_owned),
                    };
                    leaf.map(|leaf| {
                        omp_session::tree::branch_path(&entries, &leaf)
                            .into_iter()
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default()
                })
                .unwrap_or_default(),
        });
        let params = crate::compact::CompactParams {
            settings: &settings,
            model_id: &model_id,
            context_window: Some(self.live_window()).filter(|w| *w > 0),
            custom_instructions,
            non_message_tokens,
            hook: bridge
                .as_ref()
                .map(|b| b as &dyn crate::compact::CompactHook),
            guard: Some(guard),
        };
        if trace {
            turn_trace!("[turn-trace] compact-planned rss={}kB", rss_kb());
        }
        // The guard's token cancels the pass outright (Esc / a queued prompt):
        // dropping `compact_now` aborts the summarizer before anything is
        // written; a session that moved under the plan is caught by the
        // guard's own commit check inside `compact_now`.
        let session_path = self.persist.path();
        let (_result, data, view) = tokio::select! {
            out = crate::compact::compact_now(
                &self.persist,
                &session_path,
                &summarizer,
                &params,
            ) => out?,
            () = guard.cancelled() => {
                return Err(CmdError::new("Compaction cancelled"));
            }
        };
        if trace {
            turn_trace!("[turn-trace] compact-summarized rss={}kB", rss_kb());
        }
        let rebuilt = crate::compact::rebuild_context(&view)?;
        if trace {
            turn_trace!("[turn-trace] compact-rebuilt rss={}kB", rss_kb());
        }
        // The commit landed (in the file it was planned for). Apply the rebuilt
        // context only while the session that started the pass is still live:
        // after `/new` or `/resume` the agent was reset, and a late
        // `set_messages` would clobber the fresh context (bug-hunt #2).
        if !guard.session_alive() {
            return Ok(data);
        }
        self.agent.set_messages(rebuilt).await;
        if trace {
            turn_trace!("[turn-trace] compact-set rss={}kB", rss_kb());
        }
        self.agent
            .set_compaction_epoch(self.agent.compaction_epoch() + 1);
        if let Some(h) = hooks.as_deref() {
            // TS `session_compact`: the committed `compaction` entry (last journal entry).
            if let Some(entry) = self.entries().ok().and_then(|e| {
                e.into_iter()
                    .rev()
                    .find(|e| e.get("type").and_then(Value::as_str) == Some("compaction"))
            }) {
                h.after(entry).await;
            }
        }
        Ok(data)
    }

    /// Post-turn threshold check (`checkCompaction`): auto-compaction is on, the turn succeeded and
    /// the provider-billed / stored context crossed `resolveThresholdTokens`. The provider side
    /// comes from the newest settled assistant message, the stored floor from a token estimate
    /// over the live typed context — no journal parse.
    pub(super) async fn auto_compaction_due(&self) -> bool {
        let mut settings = lock(&self.compaction).clone();
        settings.enabled = true;
        let window = self.live_window();
        if !settings.enabled || window == 0 {
            return false;
        }
        // Read the live context in place (no snapshot copy of the messages).
        let mut usage = None;
        let mut stored = 0;
        self.agent
            .mutate_messages(|messages| {
                usage = messages.iter().rev().find_map(|m| match m {
                    AgentMessage::Standard(Message::Assistant(a))
                        if !matches!(a.stop_reason, StopReason::Error | StopReason::Aborted) =>
                    {
                        serde_json::to_value(&a.usage).ok()
                    }
                    _ => None,
                });
                stored = crate::compact::estimate_context_tokens(messages);
            })
            .await;
        crate::compact::auto_check(usage, stored, window, &settings)
    }

    /// The threshold compaction pass with its `auto_compaction_start` / `auto_compaction_end` events.
    pub(super) async fn run_auto_compaction(&self) {
        const ACTION: &str = "context-full";
        // One pass at a time (the `on_agent_end` pre-check races two near-
        // simultaneous turn ends); the loser is a silent no-op — its guard
        // would discard it at the commit check anyway.
        if self
            .compacting
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        self.emit_event(
            json!({"type": "auto_compaction_start", "reason": "threshold", "action": ACTION}),
        );
        let end = match self.compact_inner(None).await {
            Ok(result) => {
                json!({"type": "auto_compaction_end", "action": ACTION, "result": result, "aborted": false, "willRetry": false})
            }
            // An extension `session_before_compact` cancel is an abort, not a failure.
            Err(e) if e.message == "Compaction cancelled" => json!({
                "type": "auto_compaction_end", "action": ACTION, "aborted": true, "willRetry": false,
            }),
            Err(e) => json!({
                "type": "auto_compaction_end", "action": ACTION, "aborted": false, "willRetry": false,
                "errorMessage": format!("Auto-compaction failed: {}", e.message),
            }),
        };
        self.compacting
            .store(false, std::sync::atomic::Ordering::SeqCst);
        // Release prompts parked in `cancel_compaction_and_wait`.
        self.compaction_done.notify_waiters();
        self.emit_event(end);
    }
}

/// Compaction concurrency guard (bug-hunt #2): a prompt, `/new` or Esc during
/// in-flight compaction must cancel/serialize the pass — the concurrent turn
/// is never silently replaced and the summary never lands in the wrong file.
#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod compaction_guard_tests {
    use std::sync::Arc;
    use std::time::Duration;

    use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
    use pa_agent::scripted::ScriptedProvider;
    use serde_json::{json, Value};
    use tokio::sync::mpsc::UnboundedReceiver;

    use super::*;
    use crate::adapter::{json_round_trip, ProviderTarget};
    use crate::compact::SummarizeFn;
    use omp_compact::CompactionSettings;

    fn model() -> pa_types::ai::Model {
        serde_json::from_value(json!({
            "id": "a", "name": "a", "api": "openai-completions", "provider": "p",
            "baseUrl": "http://127.0.0.1:9", "reasoning": false, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 100_000, "maxTokens": 4096
        }))
        .expect("model json")
    }

    fn reply(text: &str) -> pa_types::ai::AssistantMessage {
        serde_json::from_value(json!({
            "content": [{"type": "text", "text": text}], "api": "openai-completions", "provider": "p", "model": "a",
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
            "stopReason": "stop", "timestamp": 1
        }))
        .expect("reply json")
    }

    /// A summarizer that parks until the test releases it, so the test can
    /// interleave a prompt / session switch / Esc while the pass runs.
    struct Gate {
        entered: tokio::sync::watch::Sender<bool>,
        /// Dropped (or set `true`) by the test to release the summarizer.
        proceed: parking_lot::Mutex<tokio::sync::watch::Receiver<bool>>,
    }

    impl Gate {
        fn new() -> (
            Arc<Self>,
            tokio::sync::watch::Receiver<bool>,
            tokio::sync::watch::Sender<bool>,
        ) {
            let (entered_tx, entered_rx) = tokio::sync::watch::channel(false);
            let (proceed_tx, proceed_rx) = tokio::sync::watch::channel(false);
            (
                Arc::new(Self {
                    entered: entered_tx,
                    proceed: parking_lot::Mutex::new(proceed_rx),
                }),
                entered_rx,
                proceed_tx,
            )
        }
    }

    fn gated_summarizer(text: &'static str, gate: &Arc<Gate>) -> SummarizeFn {
        let gate = Arc::clone(gate);
        Arc::new(move |_req| {
            let gate = Arc::clone(&gate);
            Box::pin(async move {
                gate.entered.send_replace(true);
                loop {
                    let mut rx = gate.proceed.lock().clone();
                    if *rx.borrow_and_update() {
                        break;
                    }
                    if rx.changed().await.is_err() {
                        break;
                    }
                }
                Ok(reply(text))
            })
        })
    }

    struct Harness {
        rt: Arc<Runtime>,
        rx: UnboundedReceiver<Value>,
        _dir: tempfile::TempDir,
    }

    /// A runtime over a scripted provider with a REAL session file, so journal
    /// assertions see what a compaction would have written.
    async fn harness(turns: usize) -> Harness {
        harness_scripted(|p| {
            for i in 0..turns {
                p.push_text_turn(&format!("answer {i} {}", "y".repeat(400)));
            }
        })
        .await
    }

    async fn harness_scripted(script: impl FnOnce(&ScriptedProvider)) -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let model = model();
        let provider = Arc::new(ScriptedProvider::new(
            json_round_trip(&model).expect("agent model"),
        ));
        script(&provider);
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
        let persist = crate::persist::attach_in(
            &agent,
            dir.path(),
            dir.path(),
            None,
            crate::persist::uuid_v7(),
        )
        .await
        .expect("session");
        let target = Arc::new(parking_lot::RwLock::new(Some(ProviderTarget {
            api_key: Some("k".into()),
            model,
            service_tier: None,
            headers: None,
            credential_id: None,
        })));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let rt = Runtime::start(
            Arc::new(agent),
            target,
            dir.path().to_path_buf(),
            persist,
            tx,
        )
        .await;
        std::mem::forget(provider);
        Harness { rt, rx, _dir: dir }
    }

    async fn until_settled(rx: &mut UnboundedReceiver<Value>) {
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), rx.recv())
                .await
                .expect("frame in time")
                .expect("open");
            if f["type"] == "session_settled" {
                return;
            }
        }
    }

    fn journal(path: &std::path::Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
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

    /// A prompt admitted while a compaction is in flight cancels the pass and
    /// waits for it, then runs on the settled context: no compaction entry is
    /// written and the turn lands in the journal.
    #[tokio::test]
    async fn prompt_during_in_flight_compaction_cancels_it_and_runs() {
        let mut h = harness(6).await;
        talk(&mut h, 1).await;
        let file = h.rt.session_file();
        let before = journal(&file);

        // `_proceed` keeps the gate closed until the test ends: the pass is cancelled, never released.
        let (gate, mut entered, _proceed) = Gate::new();
        h.rt.set_summarizer(Some(gated_summarizer("TOO LATE SUMMARY", &gate)));
        h.rt.set_compaction_settings(small_keep());

        let rt2 = Arc::clone(&h.rt);
        let compact = tokio::spawn(async move { rt2.compact(None).await });
        tokio::time::timeout(Duration::from_secs(30), async {
            while !*entered.borrow_and_update() {
                entered.changed().await.expect("gate");
            }
        })
        .await
        .expect("summarizer entered");

        // The user submits a prompt while the summary runs.
        h.rt.prompt(
            Some(json!("p2")),
            &format!("question 2 {}", "x".repeat(400)),
            None,
            Vec::new(),
        )
        .await
        .expect("prompt admitted behind the compaction");
        until_settled(&mut h.rx).await;

        let out = compact
            .await
            .expect("compact task")
            .expect_err("compaction cancelled");
        assert!(
            out.message.contains("Compaction cancelled"),
            "{}",
            out.message
        );

        let after = journal(&file);
        assert_eq!(
            after.iter().filter(|e| e["type"] == "compaction").count(),
            0,
            "no summary lands behind the prompt: {after:?}"
        );
        assert!(
            after.len() > before.len(),
            "the concurrent turn still lands"
        );
        assert!(
            after.last().unwrap().to_string().contains("answer 1"),
            "the prompt turn completed: {:?}",
            after.last().unwrap()
        );
    }

    /// Regression ("Request was aborted" after compact-gate's step-gate): an
    /// extension's `ctx.compact()` lands mid-run. The live turn is aborted, the
    /// summary commits, and the run resumes on the auto-continue prompt — no
    /// aborted stub is rendered or journaled.
    #[tokio::test]
    async fn compaction_mid_run_resumes_the_interrupted_turn() {
        let mut h = harness_scripted(|p| {
            for i in 0..2 {
                p.push_text_turn(&format!("answer {i} {}", "y".repeat(400)));
            }
            p.push_stalled_turn("");
            p.push_text_turn("resumed after summary");
        })
        .await;
        talk(&mut h, 2).await;
        h.rt.set_summarizer(Some(Arc::new(|_req| {
            Box::pin(async { Ok(reply("THE SUMMARY")) })
        })));
        h.rt.set_compaction_settings(small_keep());

        h.rt.prompt(None, "third question", None, Vec::new())
            .await
            .expect("prompt");
        tokio::time::timeout(Duration::from_secs(30), async {
            while !h.rt.agent.is_streaming().await {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("turn in flight");

        h.rt.compact(None).await.expect("compaction commits");

        let mut frames = Vec::new();
        let resumed = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let f = h.rx.recv().await.expect("open");
                let done =
                    f.to_string().contains("resumed after summary") && f["type"] == "message_end";
                frames.push(f);
                if done {
                    return;
                }
            }
        })
        .await;
        assert!(resumed.is_ok(), "the interrupted run resumes: {frames:?}");
        assert!(
            !frames
                .iter()
                .any(|f| f.to_string().contains("Request was aborted")
                    && matches!(f["type"].as_str(), Some("message_start" | "message_end"))),
            "no aborted stub is rendered"
        );
        let entries = journal(&h.rt.session_file());
        assert_eq!(
            entries.iter().filter(|e| e["type"] == "compaction").count(),
            1
        );
        assert!(
            !entries
                .iter()
                .any(|e| e.to_string().contains("Request was aborted")),
            "no aborted stub is journaled: {entries:?}"
        );
        let after = entries
            .iter()
            .position(|e| e["type"] == "compaction")
            .unwrap();
        assert!(
            entries[after..]
                .iter()
                .any(|e| e.to_string().contains("Resume the user's latest intent")),
            "auto-continue prompt journaled after the summary"
        );
    }

    /// A no-op rejection ("Nothing to compact") changes no history: the turn the
    /// pass cut still resumes (TS `rejectedAsNoOp`).
    #[tokio::test]
    async fn noop_compaction_mid_run_still_resumes() {
        let mut h = harness_scripted(|p| {
            p.push_stalled_turn("");
            p.push_text_turn("resumed after no-op");
        })
        .await;
        h.rt.prompt(None, "first question", None, Vec::new())
            .await
            .expect("prompt");
        tokio::time::timeout(Duration::from_secs(30), async {
            while !h.rt.agent.is_streaming().await {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("turn in flight");
        let err = h.rt.compact(None).await.expect_err("too small");
        assert!(
            err.message.starts_with("Nothing to compact"),
            "{}",
            err.message
        );
        let resumed = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let f = h.rx.recv().await.expect("open");
                if f["type"] == "message_end" && f.to_string().contains("resumed after no-op") {
                    return;
                }
            }
        })
        .await;
        assert!(resumed.is_ok(), "the interrupted run resumes");
    }

    /// Esc (`abort_compaction`) cancels the in-flight auto-compaction: the end
    /// event reports `aborted` and nothing is committed.
    #[tokio::test]
    async fn esc_cancels_in_flight_auto_compaction() {
        let mut h = harness(6).await;
        talk(&mut h, 1).await;
        let file = h.rt.session_file();

        let (gate, mut entered, proceed) = Gate::new();
        h.rt.set_summarizer(Some(gated_summarizer("NEVER COMMITTED", &gate)));
        h.rt.set_compaction_settings(small_keep());

        let rt2 = Arc::clone(&h.rt);
        let pass = tokio::spawn(async move { rt2.run_auto_compaction().await });
        tokio::time::timeout(Duration::from_secs(30), async {
            while !*entered.borrow_and_update() {
                entered.changed().await.expect("gate");
            }
        })
        .await
        .expect("summarizer entered");

        h.rt.abort_compaction(); // the Esc ladder's `abort_compaction`
        drop(proceed);
        pass.await.expect("auto-compaction task");

        let mut end = None;
        while let Ok(f) = h.rx.try_recv() {
            if f["type"] == "auto_compaction_end" {
                end = Some(f);
            }
        }
        let end = end.expect("auto_compaction_end frame");
        assert_eq!(end["aborted"], json!(true), "{end}");
        assert!(
            journal(&file).iter().all(|e| e["type"] != "compaction"),
            "nothing committed after Esc"
        );
    }

    /// `/new` during in-flight compaction discards the pass: the summary never
    /// lands in the old file and the fresh session is untouched.
    #[tokio::test]
    async fn new_session_discards_in_flight_compaction() {
        let mut h = harness(6).await;
        talk(&mut h, 1).await;
        let file = h.rt.session_file();
        let before = journal(&file);

        // `_proceed` keeps the gate closed until the test ends: the pass is cancelled, never released.
        let (gate, mut entered, _proceed) = Gate::new();
        h.rt.set_summarizer(Some(gated_summarizer("WRONG FILE SUMMARY", &gate)));
        h.rt.set_compaction_settings(small_keep());

        let rt2 = Arc::clone(&h.rt);
        let compact = tokio::spawn(async move { rt2.compact(None).await });
        tokio::time::timeout(Duration::from_secs(30), async {
            while !*entered.borrow_and_update() {
                entered.changed().await.expect("gate");
            }
        })
        .await
        .expect("summarizer entered");

        h.rt.new_session(None).await.expect("new session");
        let out = compact
            .await
            .expect("compact task")
            .expect_err("compaction cancelled");
        assert!(
            out.message.contains("Compaction cancelled"),
            "{}",
            out.message
        );

        assert_eq!(
            journal(&file)
                .iter()
                .filter(|e| e["type"] == "compaction")
                .count(),
            0,
            "the old file is untouched"
        );
        assert_eq!(
            journal(&file).len(),
            before.len(),
            "no late entries in the old file"
        );
        assert_ne!(h.rt.session_file(), file, "the fresh session file is live");

        // The fresh session still accepts turns.
        h.rt.prompt(
            None,
            &format!("fresh {}", "x".repeat(400)),
            None,
            Vec::new(),
        )
        .await
        .expect("prompt after switch");
        until_settled(&mut h.rx).await;
        let fresh = journal(&h.rt.session_file());
        assert!(
            fresh.last().unwrap().to_string().contains("answer 1"),
            "the turn landed in the new session: {:?}",
            fresh.last().unwrap()
        );
    }
}
