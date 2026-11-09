//! Task-result delivery (a2a/hub wake-ups) that starts follow-up turns.

use pa_agent::types::AgentMessage;
use serde_json::{json, Value};
use std::sync::Arc;

use super::session::seed_resumed_message;
use super::Runtime;

impl Runtime {
    /// A2A settle leg (async `task`): when `task` children are still running
    /// or their results are waiting for a turn boundary, take over the settle
    /// like the todo reminder does — the `agent_end` goes out non-terminal
    /// with `awaitingAsyncWork` (the session parks instead of settling), and
    /// the next settled batch starts a fresh turn carrying the delivery.
    /// Returns `true` when the park took over the settle.
    pub(super) fn maybe_deliver_task_results(
        self: &Arc<Self>,
        mut frame: Value,
        messages: &[AgentMessage],
    ) -> bool {
        let owner = Some(self.a2a_owner.clone());
        if !crate::tools::task::async_children_active_for(&owner) {
            return false;
        }
        let terminal_frame = frame.clone();
        frame["isTerminal"] = json!(false);
        frame["yielded"] = json!(false);
        frame["awaitingAsyncWork"] = json!(true);
        self.emit_event(frame);
        let rt = self.clone();
        let messages = messages.to_vec();
        tokio::spawn(async move {
            if !rt.start_delivery_turn().await {
                // Race: the batch was consumed elsewhere and no child is
                // left — close the settle out normally.
                rt.settle_terminal(terminal_frame, &messages, None);
            }
        });
        true
    }

    /// Append the current delivery batch as a hidden message and continue the
    /// run (TS `scheduleAgentContinue`). Returns `false` only when the race
    /// "queue drained elsewhere and no child is left" wins.
    async fn start_delivery_turn(self: &Arc<Self>) -> bool {
        let batch = crate::tools::task::next_delivery_batch(&Some(self.a2a_owner.clone())).await;
        if batch.is_empty() {
            return false;
        }
        let text = crate::tools::task::render_delivery_batch(&batch);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
        let message = AgentMessage::Custom(pa_agent::types::CustomAgentMessage {
            role: "developer".to_string(),
            payload: json!({
                "customType": "task-delivery",
                "content": [{"type": "text", "text": text}],
                "display": false,
                "attribution": "agent",
                "timestamp": timestamp,
            }),
        });
        self.agent.append_messages(vec![message.clone()]).await;
        // `json_round_trip` to pa_types returns None for custom roles like
        // "developer"; `seed_resumed_message` persists the verbatim shape.
        seed_resumed_message(&self.persist, &message, None);
        let rt = self.clone();
        tokio::spawn(async move {
            if let Err(e) = rt.continue_after_retry().await {
                eprintln!("omt: task-delivery continuation failed: {e:#}");
            }
        });
        true
    }

    /// A2A idle wake (registered on the bus in `Runtime::start`): a child
    /// message or result arrived after the session settled. With the session
    /// idle — no streaming run, no user interrupt — start a turn with the
    /// delivery; while a turn runs, its settle delivers; after an interrupt
    /// the next explicit prompt picks the queue up.
    pub(super) async fn on_delivery_wake(self: Arc<Self>) {
        if self
            .user_interrupt
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        // Before the first prompt is admitted the session has no turn to
        // continue: starting a delivery turn now would put the hub message
        // ahead of the user's first prompt (and make that prompt fail with
        // "Agent is already processing a prompt"). The delivery stays queued
        // for the first settle's delivery leg instead.
        if !self
            .first_prompt_admitted
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        if self.agent.is_streaming().await {
            return;
        }
        if !crate::tools::task::async_children_active_for(&Some(self.a2a_owner.clone())) {
            return;
        }
        let _ = self.start_delivery_turn().await;
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod a2a_delivery_tests {
    use super::*;
    use crate::adapter::json_round_trip;
    use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
    use pa_agent::scripted::ScriptedProvider;
    use pa_agent::types::ThinkingLevel;
    use tokio::sync::mpsc::UnboundedReceiver;

    use crate::adapter::ProviderTarget;

    struct Named(&'static str);

    impl pa_agent::types::AgentTool for Named {
        fn name(&self) -> &'static str {
            self.0
        }
        fn description(&self) -> &'static str {
            "test tool"
        }
        fn parameters(&self) -> &Value {
            static SCHEMA: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
            SCHEMA.get_or_init(|| json!({"type": "object", "properties": {}}))
        }
        fn execute(
            self: Arc<Self>,
            _id: String,
            _params: Value,
            _signal: pa_agent::abort::AbortSignal,
            _on_update: pa_agent::types::AgentToolUpdateCallback,
        ) -> pa_agent::BoxFut<'static, anyhow::Result<pa_agent::types::AgentToolResult>> {
            Box::pin(async { Ok(pa_agent::types::AgentToolResult::text("ok")) })
        }
    }

    fn model() -> pa_types::ai::Model {
        serde_json::from_value(json!({
            "id": "a", "name": "a", "api": "openai-completions", "provider": "p",
            "baseUrl": "http://127.0.0.1:9", "reasoning": true, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 100_000, "maxTokens": 4096
        }))
        .expect("model json")
    }

    struct Harness {
        rt: Arc<Runtime>,
        rx: UnboundedReceiver<Value>,
        provider: Arc<ScriptedProvider>,
        _cwd: tempfile::TempDir,
    }

    async fn harness() -> Harness {
        let model = model();
        let provider = Arc::new(ScriptedProvider::new(
            json_round_trip(&model).expect("agent model"),
        ));
        for i in 0..4 {
            provider.push_text_turn(&format!("answer {i}"));
        }
        let agent = Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                system_prompt: Some("test".into()),
                model: json_round_trip(&model),
                thinking_level: Some(ThinkingLevel::Medium),
                tools: Some(vec![
                    Arc::new(Named("task")) as Arc<dyn pa_agent::types::AgentTool>
                ]),
                messages: None,
            },
            convert_to_llm: Some(crate::agent::convert_to_llm()),
            stream_fn: Some(provider.stream_fn()),
            ..AgentOptions::default()
        });
        let cwd = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(cwd.path().join(".omp")).unwrap();
        let persist = crate::persist::attach_with(&agent, cwd.path(), true)
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
            cwd.path().to_path_buf(),
            persist,
            tx,
        )
        .await;
        Harness {
            rt,
            rx,
            provider,
            _cwd: cwd,
        }
    }

    /// Flatten one recorded wire request into searchable text.
    fn wire_text(ctx: &pa_agent::stream::LlmContext) -> String {
        ctx.messages
            .iter()
            .map(|m| {
                let v = serde_json::to_value(m).unwrap();
                match &v["content"] {
                    Value::String(s) => s.clone(),
                    Value::Array(parts) => parts
                        .iter()
                        .filter_map(|p| p["text"].as_str())
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Regression (merge 4c7f819): a hub `deliver` push replayed during
    /// startup — before the session's first prompt is admitted — must stay
    /// queued for the first turn boundary. Starting it as its own turn puts
    /// the delivery ahead of the user's first prompt, whose admission then
    /// fails with "Agent is already processing a prompt" (the smoke's
    /// session-B exit-1).
    #[tokio::test]
    async fn hub_delivery_before_first_prompt_waits_for_turn_boundary() {
        // Cleanup guard: the delivery queue is process-global; never leak this
        // test's envelope into other tests, even on panic.
        struct Guard {
            owner: crate::tools::a2a::Owner,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = crate::tools::a2a::take_deliveries_for(&self.owner);
            }
        }
        let mut h = harness().await;
        let _guard = Guard {
            owner: Some(h.rt.a2a_owner.clone()),
        };
        crate::tools::a2a::queue_delivery(crate::tools::a2a::PendingDelivery {
            owner: Some(h.rt.a2a_owner.clone()),
            from: "session:A".into(),
            kind: crate::tools::a2a::kind::MESSAGE.to_string(),
            text: "SMOKE-QUESTION: what is 6*7?".into(),
        });
        // Give the idle-wake hook a chance to run: it must not start a turn.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(
            !h.rt.agent.is_streaming().await,
            "delivery turn must not start before the first prompt is admitted"
        );
        // Admission succeeds only when the delivery did not steal the run.
        h.rt.prompt(None, "first question", None, Vec::new())
            .await
            .expect("first prompt admitted");
        loop {
            let f = tokio::time::timeout(std::time::Duration::from_secs(30), h.rx.recv())
                .await
                .expect("frame in time")
                .expect("open");
            if f["type"] == "session_settled" {
                break;
            }
        }
        // Ordering proof from the wire: the user turn runs first; the hub
        // message lands as the follow-up delivery turn.
        let calls = h.provider.calls();
        assert!(
            calls.len() >= 2,
            "expected the user turn plus the delivery turn, got {}",
            calls.len()
        );
        let first = wire_text(&calls[0]);
        let second = wire_text(&calls[1]);
        assert!(
            first.contains("first question"),
            "first provider call is the user turn: {first:?}"
        );
        assert!(
            !first.contains("SMOKE-QUESTION"),
            "the hub delivery must not precede the user turn: {first:?}"
        );
        assert!(
            second.contains("SMOKE-QUESTION"),
            "the delivery lands at the first turn boundary: {second:?}"
        );
    }
}
