//! Regression: sequential runs must not wedge when an event listener calls
//! back into the agent (state reads, etc.) while a previous turn's events are
//! still being dispatched.
//!
//! Pre-fix, `AgentInner::process_events` held the `shared` state mutex across
//! `listener(...).await`, so any listener that re-entered the agent (like the
//! extension listener chain, which forwards events to the ext host and can
//! trigger state reads / message queues) deadlocked against the guard owned
//! by its own await chain. With extensions loaded this surfaced as: turn N+1's
//! spawned run task wedges acquiring `shared` after 1-13 sequential prompts.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use pa_agent::agent::{Agent, AgentInitialState, AgentOptions, AgentPromptInput};
use pa_agent::scripted::ScriptedProvider;
use pa_agent::types::{Model, UsageCost};

fn test_model() -> Model {
    Model {
        id: "m".into(),
        name: "m".into(),
        api: "test".into(),
        provider: "test".into(),
        base_url: String::new(),
        reasoning: false,
        cost: UsageCost::default(),
        context_window: 100_000,
        max_tokens: 4096,
        max_tokens_explicit: false,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sequential_prompts_with_reentrant_listener_complete() {
    const TURNS: usize = 12;
    let model = test_model();
    let provider = Arc::new(ScriptedProvider::new(model.clone()));
    for n in 0..TURNS {
        provider.push_text_turn(&format!("turn-{n}"));
    }

    let agent = Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            model: Some(model),
            system_prompt: Some("test".into()),
            ..Default::default()
        },
        stream_fn: Some(provider.stream_fn()),
        ..Default::default()
    });

    // Extension-style listener: forwards every event to a "host" that calls
    // back into the agent (state read) across real await points. Pre-fix this
    // deadlocked against the `shared` guard held by `process_events`.
    let observed = Arc::new(AtomicUsize::new(0));
    let listener_agent = agent.clone();
    let listener_observed = Arc::clone(&observed);
    agent
        .subscribe(move |_event, _signal| {
            let agent = listener_agent.clone();
            let observed = Arc::clone(&listener_observed);
            Box::pin(async move {
                // Yield and wait, mimicking an RPC round-trip to the ext host.
                tokio::task::yield_now().await;
                tokio::time::sleep(Duration::from_millis(2)).await;
                // Host handler calls back into the agent: state read, plus a
                // peek at the queued steering surface.
                let _ = agent.state().await;
                let _ = agent.tools().await;
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        })
        .await;

    let drive = async {
        for n in 0..TURNS {
            agent
                .prompt(AgentPromptInput::text(format!("prompt-{n}")))
                .await
                .expect("prompt must be accepted");
            agent.wait_for_idle().await;
            let state = agent.state().await;
            let last_text = state
                .messages
                .last()
                .map(|m| format!("{m:?}"))
                .unwrap_or_default();
            assert!(
                last_text.contains(&format!("turn-{n}")),
                "turn {n} reply missing: {last_text}"
            );
        }
    };
    tokio::time::timeout(Duration::from_secs(30), drive)
        .await
        .expect("sequential prompts wedged (listener ran under the shared lock)");

    assert!(
        observed.load(Ordering::SeqCst) > 0,
        "listener must have observed events"
    );
    assert_eq!(provider.calls().len(), TURNS);
}
