#![allow(clippy::disallowed_methods)] // vendored third-party tests: the sync-bridge rule applies to omp-rs first-party crates only
//! A panicking tool becomes an error tool result for that call; the turn and
//! the agent loop keep running (parallel, sequential and early read-only paths).
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly

use std::sync::Arc;

use pa_agent::abort::AbortSignal;
use pa_agent::agent_loop::{run_agent_loop, AgentEventSink, AgentLoopConfig};
use pa_agent::scripted::ScriptedProvider;
use pa_agent::types::{
    AgentContext, AgentMessage, AgentTool, AgentToolResult, AgentToolUpdateCallback, Message,
    Model, TextContent, ToolExecutionMode, ToolResultContent, ToolResultMessage,
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

/// Panics inside its future (`in_future`) or while building it.
struct Panicker {
    in_future: bool,
    mode: Option<ToolExecutionMode>,
}

impl AgentTool for Panicker {
    fn name(&self) -> &'static str {
        "panicker"
    }

    fn description(&self) -> &'static str {
        "Always panics."
    }

    fn parameters(&self) -> &serde_json::Value {
        static SCHEMA: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        SCHEMA.get_or_init(|| serde_json::json!({"type": "object", "properties": {}}))
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        self.mode
    }

    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        _params: serde_json::Value,
        _signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> pa_agent::BoxFut<'static, anyhow::Result<AgentToolResult>> {
        assert!(self.in_future, "tool exploded before returning its future");
        Box::pin(async move { panic!("tool exploded mid-flight") })
    }
}

fn quiet_sink() -> AgentEventSink {
    Arc::new(|_event| Box::pin(async { Ok(()) }))
}

async fn run(tool: Panicker, read_only_early: bool) -> Vec<AgentMessage> {
    let provider = Arc::new(ScriptedProvider::new(test_model()));
    provider.push_tool_call_turn(None, vec![("call-1", "panicker", serde_json::json!({}))]);
    provider.push_text_turn("done");
    let mut config = AgentLoopConfig::new(test_model(), AgentLoopConfig::default_convert_to_llm());
    if read_only_early {
        config.is_tool_read_only = Some(Arc::new(|name| name == "panicker"));
    }
    let tool: Arc<dyn AgentTool> = Arc::new(tool);
    run_agent_loop(
        vec![AgentMessage::user("go")],
        AgentContext {
            system_prompt: String::new(),
            tools: vec![tool],
            messages: vec![AgentMessage::user("go")].into(),
        },
        &config,
        quiet_sink(),
        None,
        Some(&provider.stream_fn()),
    )
    .await
    .expect("a tool panic must not fail the agent loop")
}

fn assert_panic_result(messages: &[AgentMessage], expected: &str) {
    let results: Vec<&ToolResultMessage> = messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Standard(Message::ToolResult(result)) => Some(result),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 1);
    assert!(results[0].is_error, "panic must be an error result");
    let [ToolResultContent::Text(TextContent { text, .. })] = results[0].content.as_slice() else {
        panic!("expected one text block, got {:?}", results[0].content);
    };
    assert!(
        text.contains("panicked") && text.contains(expected),
        "{text}"
    );
    // The loop went on to the next turn after the failed call.
    assert!(
        matches!(
            messages.last(),
            Some(AgentMessage::Standard(Message::Assistant(_)))
        ),
        "the turn after the tool result must still run"
    );
}

#[tokio::test]
async fn parallel_tool_panic_becomes_error_result() {
    let messages = run(
        Panicker {
            in_future: true,
            mode: None,
        },
        false,
    )
    .await;
    assert_panic_result(&messages, "tool exploded mid-flight");
}

#[tokio::test]
async fn sequential_tool_panic_becomes_error_result() {
    let tool = Panicker {
        in_future: true,
        mode: Some(ToolExecutionMode::Sequential),
    };
    let messages = run(tool, false).await;
    assert_panic_result(&messages, "tool exploded mid-flight");
}

#[tokio::test]
async fn panic_while_building_the_future_becomes_error_result() {
    let tool = Panicker {
        in_future: false,
        mode: Some(ToolExecutionMode::Sequential),
    };
    let messages = run(tool, false).await;
    assert_panic_result(&messages, "before returning its future");
}

#[tokio::test]
async fn early_read_only_tool_panic_becomes_error_result() {
    let messages = run(
        Panicker {
            in_future: true,
            mode: None,
        },
        true,
    )
    .await;
    assert_panic_result(&messages, "tool exploded mid-flight");
}
