#![allow(clippy::disallowed_methods)] // vendored third-party tests: the sync-bridge rule applies to omp-rs first-party crates only
//! Early read-only tool execution: read-only calls (host-declared via
//! `AgentLoopConfig::is_tool_read_only`) must start preparing and executing at
//! their streamed `ToolCallEnd`, before the assistant stream ends; hooks stay
//! sequential/stateful while executions overlap; mutating tools never run
//! early; aborted or failed turns never commit early tool results; no call
//! executes twice.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pa_agent::abort::{AbortController, AbortSignal};
use pa_agent::agent_loop::{run_agent_loop, AgentEventSink, AgentLoopConfig};
use pa_agent::scripted::{ScriptStep, ScriptedProvider, ScriptedTurn};
use pa_agent::stream::AssistantMessageEvent;
use pa_agent::types::{
    AgentContext, AgentEvent, AgentMessage, AgentTool, AgentToolResult, AgentToolUpdateCallback,
    BeforeToolCallContext, BeforeToolCallResult, Message, Model, StopReason, TextContent,
    ToolResultContent, ToolResultMessage,
};

fn test_model() -> Model {
    Model {
        id: "test-model".into(),
        name: "Test Model".into(),
        api: "test".into(),
        provider: "test".into(),
        base_url: String::new(),
        reasoning: false,
        cost: pa_agent::types::UsageCost::default(),
        context_window: 100_000,
        max_tokens: 4_096,
        max_tokens_explicit: false,
    }
}

/// Records execution start/end instants relative to a shared run start.
struct Recorder {
    name: &'static str,
    delay_ms: u64,
    run_start: Instant,
    starts: Mutex<Vec<(String, Instant)>>,
    ends: Mutex<Vec<(String, Instant)>>,
    calls: AtomicUsize,
}

impl Recorder {
    fn new(name: &'static str, delay_ms: u64, run_start: Instant) -> Arc<Self> {
        Arc::new(Recorder {
            name,
            delay_ms,
            run_start,
            starts: Mutex::new(Vec::new()),
            ends: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
        })
    }

    fn start_offset(&self, id: &str) -> Duration {
        self.starts
            .lock()
            .unwrap()
            .iter()
            .find(|(call, _)| call == id)
            .map_or_else(
                || panic!("{id} was never started"),
                |(_, at)| at.duration_since(self.run_start),
            )
    }

    fn bounds(&self, id: &str) -> (Duration, Duration) {
        let start = self.start_offset(id);
        let end = self
            .ends
            .lock()
            .unwrap()
            .iter()
            .find(|(call, _)| call == id)
            .map_or_else(
                || panic!("{id} never finished"),
                |(_, at)| at.duration_since(self.run_start),
            );
        (start, end)
    }
}

impl AgentTool for Recorder {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &'static str {
        "Records execution timing and echoes text."
    }

    fn parameters(&self) -> &serde_json::Value {
        static SCHEMA: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        SCHEMA.get_or_init(|| {
            serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
            })
        })
    }

    fn execute(
        self: Arc<Self>,
        tool_call_id: String,
        params: serde_json::Value,
        _signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> pa_agent::BoxFut<'static, anyhow::Result<AgentToolResult>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let now = Instant::now();
            self.starts
                .lock()
                .unwrap()
                .push((tool_call_id.clone(), now));
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            self.ends
                .lock()
                .unwrap()
                .push((tool_call_id, Instant::now()));
            let text = params
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            Ok(AgentToolResult::text(format!("echo:{text}")))
        })
    }
}

fn event_log() -> (AgentEventSink, Arc<Mutex<Vec<&'static str>>>) {
    let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let sink: AgentEventSink = {
        let log = Arc::clone(&log);
        Arc::new(move |event| {
            let log = Arc::clone(&log);
            Box::pin(async move {
                log.lock().unwrap().push(match event {
                    AgentEvent::AgentStart => "agent_start",
                    AgentEvent::AgentEnd { .. } => "agent_end",
                    AgentEvent::TurnStart => "turn_start",
                    AgentEvent::TurnEnd { .. } => "turn_end",
                    AgentEvent::MessageStart { .. } => "message_start",
                    AgentEvent::MessageUpdate { .. } => "message_update",
                    AgentEvent::MessageEnd { .. } => "message_end",
                    AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
                    AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
                    AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
                });
                Ok(())
            })
        })
    };
    (sink, log)
}

fn test_context(tools: Vec<Arc<dyn AgentTool>>) -> AgentContext {
    AgentContext {
        system_prompt: String::new(),
        tools,
        messages: vec![AgentMessage::user("go")].into(),
    }
}

/// A tool-call turn with a sleep inserted before the terminal `Done`,
/// stretching the window between a call's `ToolCallEnd` and the stream end.
fn tool_turn_with_tail_delay(
    provider: &ScriptedProvider,
    tool_calls: Vec<(&str, &str, serde_json::Value)>,
    tail_delay_ms: u64,
) {
    let mut steps = pa_agent::scripted::tool_call_turn_steps(&test_model(), None, tool_calls);
    let done = steps.pop().expect("tool turn ends with Done");
    steps.push(ScriptStep::SleepMs(tail_delay_ms));
    steps.push(done);
    provider.push_turn(ScriptedTurn::Events(steps));
}

fn tool_result_messages(messages: &[AgentMessage]) -> Vec<&ToolResultMessage> {
    messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Standard(Message::ToolResult(result)) => Some(result),
            _ => None,
        })
        .collect()
}

fn single_text(content: &[ToolResultContent]) -> &str {
    match content {
        [ToolResultContent::Text(TextContent { text, .. })] => text,
        other => panic!("expected single text block, got {other:?}"),
    }
}

#[tokio::test]
async fn read_only_call_executes_before_assistant_stream_ends() {
    let provider = Arc::new(ScriptedProvider::new(test_model()));
    let run_start = Instant::now();
    let reader = Recorder::new("reader", 50, run_start);
    tool_turn_with_tail_delay(
        &provider,
        vec![("call-1", "reader", serde_json::json!({ "text": "hi" }))],
        300,
    );
    provider.push_text_turn("done");

    let mut config = AgentLoopConfig::new(test_model(), AgentLoopConfig::default_convert_to_llm());
    config.is_tool_read_only = Some(Arc::new(|name| name == "reader"));

    let (sink, log) = event_log();
    let messages = run_agent_loop(
        vec![AgentMessage::user("go")],
        test_context(vec![reader.clone()]),
        &config,
        sink,
        None,
        Some(&provider.stream_fn()),
    )
    .await
    .unwrap();

    // Exactly one execution, committed exactly once.
    assert_eq!(reader.calls.load(Ordering::SeqCst), 1);
    let results = tool_result_messages(&messages);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].tool_call_id, "call-1");
    assert_eq!(single_text(&results[0].content), "echo:hi");

    // The execution began while the stream was still in its 300ms tail.
    assert!(
        reader.start_offset("call-1") < Duration::from_millis(280),
        "read-only execution must start before the stream tail ends, started after {:?}",
        reader.start_offset("call-1")
    );

    // Event-level proof: `tool_execution_start` precedes the assistant
    // `message_end`; for a normally-dispatched call it follows it.
    let log = log.lock().unwrap().clone();
    let end = log
        .iter()
        .position(|e| *e == "tool_execution_end")
        .expect("tool_execution_end emitted");
    let assistant_end = log[..end]
        .iter()
        .rposition(|e| *e == "message_end")
        .expect("assistant message_end before tool_execution_end");
    assert!(
        log[..assistant_end].contains(&"tool_execution_start"),
        "tool_execution_start must precede the assistant message_end: {log:?}"
    );
}

#[tokio::test]
async fn mutating_call_waits_for_stream_end_and_runs_once() {
    let provider = Arc::new(ScriptedProvider::new(test_model()));
    let run_start = Instant::now();
    let writer = Recorder::new("writer", 0, run_start);
    tool_turn_with_tail_delay(
        &provider,
        vec![("call-1", "writer", serde_json::json!({ "text": "hi" }))],
        300,
    );
    provider.push_text_turn("done");

    let mut config = AgentLoopConfig::new(test_model(), AgentLoopConfig::default_convert_to_llm());
    config.is_tool_read_only = Some(Arc::new(|name| name == "reader"));

    let (sink, log) = event_log();
    let messages = run_agent_loop(
        vec![AgentMessage::user("go")],
        test_context(vec![writer.clone()]),
        &config,
        sink,
        None,
        Some(&provider.stream_fn()),
    )
    .await
    .unwrap();

    assert_eq!(writer.calls.load(Ordering::SeqCst), 1);
    assert_eq!(tool_result_messages(&messages).len(), 1);
    // The mutating call only started after the 300ms stream tail.
    assert!(
        writer.start_offset("call-1") >= Duration::from_millis(280),
        "mutating call must not execute before the stream ends, started after {:?}",
        writer.start_offset("call-1")
    );

    let log = log.lock().unwrap().clone();
    let start = log
        .iter()
        .position(|e| *e == "tool_execution_start")
        .unwrap();
    let end = log.iter().position(|e| *e == "tool_execution_end").unwrap();
    let assistant_end = log[..end]
        .iter()
        .rposition(|e| *e == "message_end")
        .unwrap();
    assert!(
        start > assistant_end,
        "mutating tool_execution_start must follow the assistant message_end: {log:?}"
    );
    assert!(start < end);
}

#[tokio::test]
async fn early_hooks_run_sequentially_while_executions_overlap() {
    let provider = Arc::new(ScriptedProvider::new(test_model()));
    let run_start = Instant::now();
    // call-a's execution outlives call-b's hook: the hook of call-b must
    // start while call-a is still executing, and hooks must observe calls in
    // source order.
    let reader = Recorder::new("reader", 150, run_start);
    provider.push_turn(ScriptedTurn::Events(
        pa_agent::scripted::tool_call_turn_steps(
            &test_model(),
            None,
            vec![
                ("call-a", "reader", serde_json::json!({ "text": "a" })),
                ("call-b", "reader", serde_json::json!({ "text": "b" })),
            ],
        ),
    ));
    provider.push_text_turn("done");

    let hook_log: Arc<Mutex<Vec<(String, Instant)>>> = Arc::new(Mutex::new(Vec::new()));
    let hook_sleep = Duration::from_millis(150);
    let mut config = AgentLoopConfig::new(test_model(), AgentLoopConfig::default_convert_to_llm());
    config.is_tool_read_only = Some(Arc::new(|name| name == "reader"));
    let hook_log_for_config = Arc::clone(&hook_log);
    config.before_tool_call = Some(Arc::new(move |ctx: BeforeToolCallContext, _signal| {
        let hook_log = Arc::clone(&hook_log_for_config);
        Box::pin(async move {
            hook_log
                .lock()
                .unwrap()
                .push((ctx.tool_call.id.clone(), Instant::now()));
            tokio::time::sleep(hook_sleep).await;
            Ok(None)
        })
    }));

    let (sink, _log) = event_log();
    let messages = run_agent_loop(
        vec![AgentMessage::user("go")],
        test_context(vec![reader.clone()]),
        &config,
        sink,
        None,
        Some(&provider.stream_fn()),
    )
    .await
    .unwrap();

    // One execution per call, results in assistant source order.
    assert_eq!(reader.calls.load(Ordering::SeqCst), 2);
    let results = tool_result_messages(&messages);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].tool_call_id, "call-a");
    assert_eq!(results[1].tool_call_id, "call-b");

    // Each hook ran exactly once, in source order (stateful hook semantics).
    let hooks = hook_log.lock().unwrap().clone();
    assert_eq!(
        hooks.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>(),
        vec!["call-a".to_string(), "call-b".to_string()]
    );

    // Preparation of call-b overlapped the execution of call-a: hook-b starts
    // after exec-a started but before exec-a finished. (Serial preparation
    // and execution would give hook-b.start >= exec-a.end.)
    let (a_start, a_end) = reader.bounds("call-a");
    let b_hook = hooks[1].1.duration_since(run_start);
    assert!(
        a_start <= b_hook + Duration::from_millis(20),
        "pipeline must not delay exec-a behind call-b's hook: {a_start:?} vs {b_hook:?}"
    );
    assert!(
        b_hook < a_end,
        "call-b's hook must overlap call-a's execution: hook at {b_hook:?}, exec-a [{a_start:?}, {a_end:?}]"
    );
}

#[tokio::test]
#[ignore = "local wall-clock A/B; timing values are reported, not asserted"]
async fn early_read_only_wall_clock_ab() {
    async fn run(early: bool) -> Duration {
        let provider = Arc::new(ScriptedProvider::new(test_model()));
        let started = Instant::now();
        let reader = Recorder::new("reader", 250, started);
        tool_turn_with_tail_delay(
            &provider,
            vec![("call-1", "reader", serde_json::json!({ "text": "hi" }))],
            250,
        );
        provider.push_text_turn("done");
        let mut config =
            AgentLoopConfig::new(test_model(), AgentLoopConfig::default_convert_to_llm());
        if early {
            config.is_tool_read_only = Some(Arc::new(|name| name == "reader"));
        }
        let (sink, _) = event_log();
        let messages = run_agent_loop(
            vec![AgentMessage::user("go")],
            test_context(vec![reader.clone()]),
            &config,
            sink,
            None,
            Some(&provider.stream_fn()),
        )
        .await
        .unwrap();
        assert_eq!(reader.calls.load(Ordering::SeqCst), 1);
        assert_eq!(tool_result_messages(&messages).len(), 1);
        started.elapsed()
    }

    // Alternate order to reduce warm-up/order bias. The no-early control has
    // the same hooks, tools, events and result path; only dispatch timing differs.
    let control = run(false).await;
    let early = run(true).await;
    println!("early_tool_ab control={control:?} early={early:?}");
}

#[tokio::test]
async fn aborted_turn_discards_early_tool_results() {
    let provider = Arc::new(ScriptedProvider::new(test_model()));
    let run_start = Instant::now();
    let reader = Recorder::new("reader", 0, run_start);

    // Stream the tool call's arguments, then stall until the abort lands.
    let mut steps = pa_agent::scripted::tool_call_turn_steps(
        &test_model(),
        None,
        vec![("call-1", "reader", serde_json::json!({ "text": "hi" }))],
    );
    steps.pop(); // drop the terminal Done
    provider.push_turn(ScriptedTurn::Stalled { prelude: steps });

    let mut config = AgentLoopConfig::new(test_model(), AgentLoopConfig::default_convert_to_llm());
    config.is_tool_read_only = Some(Arc::new(|name| name == "reader"));

    let controller = AbortController::new();
    let (sink, _log) = event_log();
    let run = tokio::spawn({
        let signal = controller.signal();
        let stream_fn = provider.stream_fn();
        let context = test_context(vec![reader.clone()]);
        let config = config.clone();
        Box::pin(async move {
            run_agent_loop(
                vec![AgentMessage::user("go")],
                context,
                &config,
                sink,
                Some(&signal),
                Some(&stream_fn),
            )
            .await
        })
    });
    tokio::time::sleep(Duration::from_millis(150)).await;
    controller.abort();
    let messages = run.await.unwrap().unwrap();

    // The turn ended as aborted with no tool result committed anywhere.
    let assistant = messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Standard(Message::Assistant(a)) => Some(a),
            _ => None,
        })
        .next_back()
        .expect("aborted assistant message");
    assert_eq!(assistant.stop_reason, StopReason::Aborted);
    assert!(
        tool_result_messages(&messages).is_empty(),
        "aborted turns must never commit tool results"
    );
}

#[tokio::test]
async fn failed_tail_discards_early_tool_results() {
    let provider = Arc::new(ScriptedProvider::new(test_model()));
    let run_start = Instant::now();
    let reader = Recorder::new("reader", 80, run_start);

    // ToolCallEnd, a grace window so the early execution is in flight, then a
    // mid-stream provider failure instead of Done.
    let mut steps = pa_agent::scripted::tool_call_turn_steps(
        &test_model(),
        None,
        vec![("call-1", "reader", serde_json::json!({ "text": "hi" }))],
    );
    steps.pop(); // drop Done
    steps.push(ScriptStep::SleepMs(20));
    let partial = steps
        .iter()
        .rev()
        .find_map(|step| match step {
            ScriptStep::Event(event) => match &**event {
                AssistantMessageEvent::ToolCallEnd { partial, .. } => Some(partial.clone()),
                _ => None,
            },
            ScriptStep::SleepMs(_) => None,
        })
        .expect("tool turn has a ToolCallEnd partial");
    let mut error_message_partial = partial;
    error_message_partial.stop_reason = StopReason::Error;
    error_message_partial.error_message = Some("connection reset by peer".to_string());
    steps.push(ScriptStep::Event(Box::new(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: error_message_partial,
    })));
    provider.push_turn(ScriptedTurn::Events(steps));

    let mut config = AgentLoopConfig::new(test_model(), AgentLoopConfig::default_convert_to_llm());
    config.is_tool_read_only = Some(Arc::new(|name| name == "reader"));

    let (sink, _log) = event_log();
    let messages = run_agent_loop(
        vec![AgentMessage::user("go")],
        test_context(vec![reader.clone()]),
        &config,
        sink,
        None,
        Some(&provider.stream_fn()),
    )
    .await
    .unwrap();

    let assistant = messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Standard(Message::Assistant(a)) => Some(a),
            _ => None,
        })
        .next_back()
        .expect("error assistant message");
    assert_eq!(assistant.stop_reason, StopReason::Error);
    assert!(
        tool_result_messages(&messages).is_empty(),
        "failed turns must never commit tool results"
    );
    assert_eq!(
        reader.calls.load(Ordering::SeqCst),
        1,
        "the early read-only execution happens at most once and is never retried inline"
    );
}

/// An early-started read-only call that its `beforeToolCall` hook blocks
/// still gets exactly one `tool_execution_start` and one end: the batch must
/// not re-announce it (a second start left consumers with a phantom,
/// forever-pending card).
#[tokio::test]
async fn blocked_early_call_is_announced_once() {
    let provider = Arc::new(ScriptedProvider::new(test_model()));
    let run_start = Instant::now();
    let reader = Recorder::new("reader", 0, run_start);
    tool_turn_with_tail_delay(
        &provider,
        vec![
            ("call-a", "reader", serde_json::json!({ "text": "a" })),
            ("call-b", "reader", serde_json::json!({ "text": "b" })),
        ],
        100,
    );
    provider.push_text_turn("done");

    let mut config = AgentLoopConfig::new(test_model(), AgentLoopConfig::default_convert_to_llm());
    config.is_tool_read_only = Some(Arc::new(|name| name == "reader"));
    config.before_tool_call = Some(Arc::new(|ctx: BeforeToolCallContext, _signal| {
        Box::pin(async move {
            Ok(
                (ctx.tool_call.id == "call-a").then(|| BeforeToolCallResult {
                    block: true,
                    reason: Some("denied".into()),
                    args: None,
                    additional_context: None,
                }),
            )
        })
    }));

    let (sink, log) = event_log();
    let messages = run_agent_loop(
        vec![AgentMessage::user("go")],
        test_context(vec![reader.clone()]),
        &config,
        sink,
        None,
        Some(&provider.stream_fn()),
    )
    .await
    .unwrap();

    assert_eq!(reader.calls.load(Ordering::SeqCst), 1, "only call-b runs");
    let results = tool_result_messages(&messages);
    assert_eq!(results.len(), 2);
    assert!(results[0].is_error, "blocked call reports an error result");
    let log = log.lock().unwrap().clone();
    let count = |name: &str| log.iter().filter(|e| **e == name).count();
    assert_eq!(count("tool_execution_start"), 2, "{log:?}");
    assert_eq!(count("tool_execution_end"), 2, "{log:?}");
}
