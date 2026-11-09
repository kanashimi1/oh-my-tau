//! Agent-event handling: turns `AgentEvent`s into frames and settles each turn.

use super::recovery::FailedTurn;
use pa_agent::types::{AgentEvent, AgentMessage, Message, StopReason};
use serde_json::{json, Value};
use std::sync::Arc;

use super::prompt::run_outcome;
use super::{last_assistant_msg, lock, rss_kb, Runtime};

impl Runtime {
    pub fn send(&self, frame: Value) {
        if matches!(
            frame["type"].as_str(),
            Some(
                "auto_compaction_start"
                    | "auto_compaction_end"
                    | "auto_retry_start"
                    | "auto_retry_end"
            )
        ) {
            if let Some(tx) = lock(&self.ext_frames).as_ref() {
                let _ = tx.send(frame.clone());
            }
        }
        let _ = self.out.send(frame);
    }

    /// Forward a session event subject to `set_event_filter`.
    pub(super) fn emit_event(&self, mut frame: Value) {
        let st = lock(&self.settings);
        let kind = frame
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if st.event_filter.as_ref().is_some_and(|f| !f.contains(&kind)) {
            return;
        }
        if st.delta_updates && kind == "message_update" {
            if let Some(obj) = frame.as_object_mut() {
                obj.remove("message");
                if let Some(ev) = obj
                    .get_mut("assistantMessageEvent")
                    .and_then(Value::as_object_mut)
                {
                    ev.remove("partial");
                }
            }
        }
        drop(st);
        self.send(frame);
    }

    #[allow(clippy::too_many_lines)] // one match over every agent event; each arm updates one piece of state
    pub(super) fn on_event(self: &Arc<Self>, event: &AgentEvent) {
        let ev_kind = match event {
            AgentEvent::MessageUpdate { .. } => "message_update",
            AgentEvent::MessageEnd { message } => match message.role() {
                "user" => "message_end:user",
                "assistant" => "message_end:assistant",
                _ => "message_end:toolResult",
            },
            AgentEvent::AgentEnd { .. } => "agent_end",
            AgentEvent::TurnEnd { .. } => "turn_end",
            AgentEvent::AgentStart => "agent_start",
            AgentEvent::TurnStart => "turn_start",
            AgentEvent::MessageStart { .. } => "message_start",
            AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
            AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
            AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
        };
        let trace = std::env::var_os("OMP_RS_TURN_TRACE").is_some();
        if trace {
            turn_trace!("[turn-trace] event={ev_kind} rss={}kB", rss_kb());
        }
        let frame = lock(&self.framer).frame(event);
        if trace {
            let n = serde_json::to_string(&frame).map_or(0, |s| s.len());
            turn_trace!(
                "[turn-trace] framed={ev_kind} frame_json={n}B rss={}kB",
                rss_kb()
            );
        }
        // context-usage frames are TUI chrome; rpc/print clients never read
        // them and each publish re-serializes the whole context. Skip in
        // non-interactive sessions (stdout is what rpc/print write to).
        let interactive = std::io::IsTerminal::is_terminal(&std::io::stdout());
        if interactive
            && matches!(
                event,
                AgentEvent::AgentEnd { .. }
                    | AgentEvent::MessageEnd {
                        message: AgentMessage::Standard(
                            Message::Assistant(_) | Message::ToolResult(_)
                        )
                    }
            )
        {
            self.publish_context_usage();
        }
        // Free pages on the thread that produced the per-update message values.
        // Every full-partial `message_update` frame clones the whole partial
        // assistant message on this thread; the RPC writer frees those clones on
        // another thread, and without a producer-side collect mimalloc keeps the
        // worker's pages committed for the process lifetime. One 64 KiB reply in
        // 4,096 deltas settled at 158.8 MiB anonymous RSS without this and
        // 18.5 MiB with it. Collect per model turn (not only at run end): the
        // 18 GB leak session ran 1,620 model turns inside few prompts, so
        // run-end collection alone would still accumulate within a run. The
        // interactive TUI renders on this thread and collects at exit.
        if !interactive
            && matches!(
                event,
                AgentEvent::AgentEnd { .. } | AgentEvent::TurnEnd { .. }
            )
        {
            crate::collect_retained_memory();
        }
        if let AgentEvent::AgentEnd { messages } = event {
            self.on_agent_end(frame, messages);
            return;
        }
        let mut frame = frame;
        if self
            .compaction_abort
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            use std::sync::atomic::Ordering::SeqCst;
            match event {
                // Failure path: the stub's own start/end pair — render neither.
                AgentEvent::MessageStart { message }
                    if crate::persist::is_empty_aborted(message) =>
                {
                    self.compaction_stub_start.store(true, SeqCst);
                    return;
                }
                AgentEvent::MessageEnd { message } if crate::persist::is_empty_aborted(message) => {
                    if self.compaction_stub_start.swap(false, SeqCst) {
                        return;
                    }
                    // A streamed-but-empty turn: settle its open block without the error row.
                    if let Some(m) = frame.get_mut("message").and_then(Value::as_object_mut) {
                        m.remove("errorMessage");
                    }
                }
                _ => {}
            }
        }
        self.emit_event(frame);
        match event {
            AgentEvent::AgentStart => {
                let mut p = lock(&self.prompts);
                p.agent_starts += 1;
                p.active = true;
                drop(p);
                self.herdr_working();
            }
            AgentEvent::ToolExecutionStart {
                tool_name, args, ..
            } if tool_name == "ask" => {
                // Herdr: an ask dialog blocks on the user.
                self.ask_blocked
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let message = args
                    .get("questions")
                    .and_then(Value::as_array)
                    .and_then(|qs| {
                        qs.iter()
                            .find_map(|q| q.get("question").and_then(Value::as_str))
                    })
                    .map(str::to_string);
                if let (Some(h), Some(m)) = (&self.herdr, message) {
                    h.report(crate::herdr::State::Blocked, Some(&m));
                } else if let Some(h) = &self.herdr {
                    h.report(crate::herdr::State::Blocked, Some("waiting for user input"));
                }
            }
            AgentEvent::ToolExecutionEnd { tool_name, .. } if tool_name == "ask" => {
                let prev = self
                    .ask_blocked
                    .fetch_update(
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                        |n| Some(n.saturating_sub(1)),
                    )
                    .unwrap_or(0);
                if prev == 1 {
                    self.herdr_working();
                }
            }
            AgentEvent::MessageEnd { message } if message.role() == "user" => {
                self.emit_queue_update();
            }
            AgentEvent::MessageEnd {
                message: AgentMessage::Standard(Message::Assistant(a)),
            } => {
                self.on_assistant_settled(a);
            }
            // Checkpoint runtime hooks (reports/Checkpoint.md): a successful
            // `checkpoint` tool result arms the shared checkpoint state with the
            // post-append message count; a successful `rewind` result stashes
            // its report for the TurnEnd truncation.
            AgentEvent::MessageEnd {
                message: AgentMessage::Standard(Message::ToolResult(tr)),
            } => {
                self.on_tool_result_settled(tr);
            }
            AgentEvent::TurnEnd { .. } => {
                self.maybe_apply_rewind();
            }
            _ => {}
        }
    }

    /// TS `AgentSession`'s checkpoint toolResult leg of `#checkpointActiveReminderFor`:
    /// arm the shared checkpoint state with the message count AFTER the
    /// checkpoint tool result is appended (async: the listener runs sync while
    /// the agent holds its state lock, so the count is read on a task).
    fn on_tool_result_settled(&self, tr: &pa_agent::types::ToolResultMessage) {
        if tr.is_error {
            return;
        }
        if tr.tool_name == "checkpoint" {
            let started_at = tr
                .details
                .as_ref()
                .and_then(|d| d.get("startedAt"))
                .and_then(Value::as_str)
                .map_or_else(crate::persist::now_iso, str::to_owned);
            let agent = self.agent.clone();
            tokio::spawn(async move {
                let count = agent.message_count().await;
                crate::tools::checkpoint::shared().set_checkpoint_state(
                    crate::tools::checkpoint::CheckpointState {
                        checkpoint_message_count: count,
                        checkpoint_entry_id: None,
                        started_at,
                    },
                );
            });
        } else if tr.tool_name == "rewind" {
            let report = tr
                .details
                .as_ref()
                .and_then(|d| d.get("report"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|r| !r.is_empty())
                .map(str::to_owned)
                .or_else(|| {
                    tr.content.iter().find_map(|part| match part {
                        pa_agent::types::ToolResultContent::Text(t) => {
                            let text = t.text.trim();
                            (!text.is_empty()).then(|| text.to_owned())
                        }
                        pa_agent::types::ToolResultContent::Image(_) => None,
                    })
                });
            if let Some(report) = report {
                *lock(&self.pending_rewind_report) = Some(report);
            }
        } else if tr.tool_name == "todo" {
            self.on_todo_settled(tr);
        }
    }

    /// TS `setOnTurnEnd`'s rewind leg (`#applyRewind`): when this turn carried a
    /// successful `rewind` result, cut the context back to the checkpoint in a
    /// spawned task — the listener runs under the agent's state lock, so the
    /// `set_messages` surgery happens after the run went idle.
    fn maybe_apply_rewind(self: &Arc<Self>) {
        let Some(report) = lock(&self.pending_rewind_report).take() else {
            return;
        };
        if crate::tools::checkpoint::shared()
            .checkpoint_state()
            .is_none()
        {
            return;
        }
        let rt = self.clone();
        tokio::spawn(async move {
            rt.agent.wait_for_idle().await;
            let Some(state) = crate::tools::checkpoint::shared().checkpoint_state() else {
                return;
            };
            let count = state.checkpoint_message_count;
            // Only the kept prefix is copied (not the whole transcript, twice).
            let prefix = rt
                .agent
                .with_messages(|m| {
                    (m.len() > count).then(|| m.iter().take(count).cloned().collect::<Vec<_>>())
                })
                .await;
            if let Some(mut messages) = prefix {
                // Carry the report into the cut context like the TS branch view's
                // rewind-report custom message.
                messages.push(AgentMessage::Custom(pa_agent::types::CustomAgentMessage {
                    role: "agent".to_string(),
                    payload: serde_json::json!({
                        "customType": "rewind-report",
                        "content": report,
                        "display": false,
                        "details": {
                            "report": report,
                            "startedAt": state.started_at,
                            "rewoundAt": crate::persist::now_iso(),
                        },
                    }),
                }));
                rt.agent.set_messages(messages).await;
            }
            // Persist the `rewind-report` entry (TS `appendCustomMessageEntry`).
            let details = serde_json::json!({
                "report": report,
                "startedAt": state.started_at,
                "rewoundAt": crate::persist::now_iso(),
            });
            let _ = rt.persist.append_entry(
                "custom_message",
                serde_json::json!({
                    "customType": "rewind-report",
                    "content": report,
                    "display": false,
                    "details": details,
                    "attribution": "agent",
                }),
            );
            crate::tools::checkpoint::shared().set_last_completed_rewind(
                crate::tools::checkpoint::CompletedRewindState {
                    report,
                    started_at: state.started_at,
                    rewound_at: crate::persist::now_iso(),
                },
            );
            crate::tools::checkpoint::shared().clear_checkpoint_state();
        });
    }

    /// `onAssistantSettledSuccessfully`: a turn that produced output settled on the live model.
    fn on_assistant_settled(&self, a: &pa_agent::types::AssistantMessage) {
        if matches!(a.stop_reason, StopReason::Error | StopReason::Aborted) || a.content.is_empty()
        {
            return;
        }
        let selector = self.live_selector_with_level(None);
        let frames = lock(&self.recovery).on_settled_ok(&selector);
        if !frames.is_empty() {
            // The failed turn was superseded: it never reaches the session file.
            self.persist.drop_deferred_error();
        }
        for f in frames {
            self.emit_event(f);
        }
    }

    /// Laya shadow judge (default OFF, `OMP_LAYA_JUDGE=1`): hand the finished run to the
    /// background worker. Never blocks or fails the turn: nothing is built when disabled, the
    /// queue is bounded and an overflowing turn is dropped (counted).
    fn shadow_judge_turn(&self, messages: &[AgentMessage]) {
        let Some(worker) = omp_judge::shadow::global() else {
            return;
        };
        let entries =
            omp_judge::shadow::message_entries(messages.iter().map(crate::events::message_value));
        let turn_index = self
            .judge_turns
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Best-effort shadow judging: a full queue drops this job by design.
        let _ = worker.submit(omp_judge::shadow::Job {
            session_id: self.persist.session_id(),
            turn_index,
            ts: crate::persist::now_iso(),
            entries,
            is_sub: false,
        });
    }

    /// Terminal `agent_end`, or the retry saga when the run failed transiently.
    fn on_agent_end(self: &Arc<Self>, frame: Value, messages: &[AgentMessage]) {
        // Budget gate: after each assistant turn, stop once the limit is spent.
        // Spawned so the check runs after the agent's state lock is released.
        if crate::features::budget::is_armed() {
            let rt = self.clone();
            tokio::spawn(async move {
                crate::features::budget::check_after_turn(&rt).await;
            });
        }
        self.emit_queue_update();
        self.shadow_judge_turn(messages);
        let failed = last_assistant_msg(messages).and_then(FailedTurn::from_message);
        // Cheap pre-check only; the threshold decision itself (`checkCompaction`)
        // runs on the live typed context inside the task, off the agent's state
        // lock — no journal DOM is built for a turn that will not compact.
        let auto_maybe = failed.is_none()
            && !self.compacting.load(std::sync::atomic::Ordering::SeqCst)
            && lock(&self.settings).auto_compaction
            && last_assistant_msg(messages)
                .is_some_and(|a| !matches!(a.stop_reason, StopReason::Error | StopReason::Aborted));
        if let Some(f) = failed.filter(|f| lock(&self.recovery).is_retryable(f)) {
            // Herdr: the retry saga holds the pane Blocked until the backoff
            // elapses and the run continues (AgentStart -> working).
            if let Some(h) = &self.herdr {
                let msg = f.message.chars().take(200).collect::<String>();
                h.report(crate::herdr::State::Blocked, Some(&msg));
            }
            let rt = self.clone();
            let messages = messages.to_vec();
            // The listener runs while the agent holds its state lock: recovery (model swap,
            // context surgery, continuing the run) happens after the run went idle.
            let handle = tokio::spawn(async move { rt.recover(frame, messages, f).await });
            *lock(&self.retry_task) = Some(handle);
            return;
        }
        if auto_maybe {
            let rt = self.clone();
            let messages = messages.to_vec();
            // Like `recover`: the listener runs under the agent's state lock.
            let handle = tokio::spawn(async move {
                if rt.auto_compaction_due().await {
                    // With async `task` children the terminal frame waits for
                    // the delivery park (awaitingAsyncWork), like
                    // `settle_terminal`.
                    let park =
                        crate::tools::task::async_children_active_for(&Some(rt.a2a_owner.clone()));
                    if !park {
                        rt.emit_event(frame.clone());
                    }
                    rt.run_auto_compaction().await;
                    if !rt.maybe_deliver_task_results(frame.clone(), &messages) {
                        rt.report_prompts(&run_outcome(&messages));
                        rt.drain_stranded_queue();
                    }
                } else {
                    rt.settle_terminal(frame, &messages, None);
                }
            });
            // A prompt / session switch behind this task cancels + awaits it
            // (bug-hunt #2: the pass used to run untracked, fire-and-forget).
            *lock(&self.post_turn_task) = Some(handle);
            return;
        }
        self.settle_terminal(frame, messages, None);
    }

    /// Final `agent_end` of a prompt: persist/close failure bookkeeping, then `prompt_result`s.
    pub(super) fn settle_terminal(
        self: &Arc<Self>,
        mut frame: Value,
        messages: &[AgentMessage],
        error_override: Option<&str>,
    ) {
        // Herdr: the turn is over — the pane waits for input.
        self.herdr_idle();
        let mut messages = messages.to_vec();
        if let Some(AgentMessage::Standard(Message::Assistant(a))) =
            messages.iter_mut().rev().find(|m| m.role() == "assistant")
        {
            if a.stop_reason == StopReason::Error {
                if let Some(e) = error_override {
                    a.error_message = Some(e.to_string());
                    if let Some(last) = frame
                        .get_mut("messages")
                        .and_then(Value::as_array_mut)
                        .and_then(|v| {
                            v.iter_mut()
                                .rev()
                                .find(|m| m.get("role") == Some(&json!("assistant")))
                        })
                    {
                        last["errorMessage"] = json!(e);
                    }
                }
                self.persist.flush_deferred_error(error_override);
                let text = a.error_message.clone().unwrap_or_default();
                // Never emit while holding the recovery lock (`emit_event` takes `settings`,
                // `state()` takes `settings` then `recovery`).
                let end = lock(&self.recovery).on_error_settled(&text);
                if let Some(end) = end {
                    self.emit_event(end);
                }
            }
        }
        // TS `TodoTracker.checkCompletion`: a successful terminal turn with
        // incomplete todos nudges the agent to continue instead of settling.
        // Not for failed/aborted turns, and never over a user interrupt (the
        // reminder would resurrect the run the user just stopped).
        let failed_turn = messages
            .iter()
            .rev()
            .find(|m| m.role() == "assistant")
            .and_then(|m| match m {
                AgentMessage::Standard(Message::Assistant(a)) => Some(a),
                _ => None,
            })
            .is_some_and(|a| matches!(a.stop_reason, StopReason::Error | StopReason::Aborted));
        let interrupted = self
            .user_interrupt
            .load(std::sync::atomic::Ordering::SeqCst);
        if error_override.is_none()
            && !failed_turn
            && !interrupted
            && self.maybe_todo_reminder(&messages, frame.clone())
        {
            return;
        }
        // A2A: with `task` children running (or their results still queued),
        // the session must not settle yet — park, then start a turn with the
        // next settled batch (print/rpc sessions stay alive until every
        // child settled). Never auto-resumes over a user interrupt; the next
        // explicit prompt picks the queue up.
        if !interrupted && self.maybe_deliver_task_results(frame.clone(), &messages) {
            return;
        }
        let outcome = run_outcome(&messages);
        self.emit_event(frame);
        self.report_prompts(&outcome);
        self.drain_stranded_queue();
    }
}
