use std::path::PathBuf;
use std::time::Duration;

use omp_ext::hooks::{self, ExtHooks};
use omp_ext::{ExtHost, HostConfig, HostMessage};
use serde_json::{json, Value};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn config() -> HostConfig {
    let mut cfg =
        HostConfig::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ext-host/host.ts"));
    cfg.handler_timeout_ms = Some(400);
    apply_shared_mode(&mut cfg);
    cfg
}

/// `OMP_EXT_TEST_SHARED=1` runs the suite against the shared daemon (one bun
/// process, one Worker per session) instead of a per-session stdio host.
fn apply_shared_mode(cfg: &mut HostConfig) {
    if std::env::var_os("OMP_EXT_TEST_SHARED").is_some_and(|v| v != "0") {
        let dir = std::env::temp_dir().join(format!("omp-ext-shared-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        cfg.shared_socket = Some(dir.join("ext.sock"));
    }
}

async fn loaded() -> (
    std::sync::Arc<ExtHost>,
    tokio::sync::mpsc::UnboundedReceiver<HostMessage>,
) {
    let host = ExtHost::spawn(&config()).await.unwrap();
    let rx = host.take_incoming().unwrap();
    let r = host
        .load(
            Some(&[
                fixture("fixture.ts"),
                fixture("broken.ts"),
                fixture("nofactory.ts"),
            ]),
            None,
            "/tmp",
            Some(json!({"sessionId": "s1"})),
        )
        .await
        .unwrap();
    assert_eq!(r.loaded.len(), 1, "{r:?}");
    assert_eq!(r.errors.len(), 2, "{r:?}");
    assert!(r
        .errors
        .iter()
        .any(|e| e.path.ends_with("nofactory.ts") && e.error.contains("factory")));
    (host, rx)
}

#[tokio::test]
async fn gates_rewrites_and_host_survives_failing_handlers() {
    let (host, mut rx) = loaded().await;

    // block wins
    let o = hooks::tool_call(&host, "bash", "1", &json!({"command":"rm -rf /"}), None)
        .await
        .unwrap();
    assert!(o.block);
    assert_eq!(o.reason.as_deref(), Some("no rm"));

    // input last-wins + additionalContext joined across handlers in registration order
    let o = hooks::tool_call(&host, "bash", "2", &json!({"command":"echo hi"}), None)
        .await
        .unwrap();
    assert!(!o.block);
    assert_eq!(o.input, Some(json!({"command":"echo rewritten"})));
    assert_eq!(o.additional_context.as_deref(), Some("ctx-a\n\nctx-b"));

    // throwing tool_call handler fails closed, host keeps serving
    let o = hooks::tool_call(&host, "boom", "3", &json!({}), None)
        .await
        .unwrap();
    assert!(o.block);
    assert!(o.reason.unwrap().contains("handler exploded"));
    let o = hooks::tool_call(&host, "read", "4", &json!({}), None)
        .await
        .unwrap();
    assert!(!o.block);

    // tool_result rewrite
    let r = host
        .emit(json!({"type":"tool_result","toolName":"bash","toolCallId":"2","input":{},"content":[{"type":"text","text":"hello"}],"isError":false}), None)
        .await
        .unwrap();
    assert_eq!(r.result["content"][0]["text"], "[redacted] 5");

    // hung handler is cut off by the budget; non-tool_call errors are only reported
    let t = std::time::Instant::now();
    let r = host.emit(json!({"type":"hang"}), None).await.unwrap();
    assert!(t.elapsed() < Duration::from_secs(5));
    assert_eq!(r.errors.len(), 1);

    // context transform
    let r = host
        .emit(
            json!({"type":"context","messages":[{"role":"user","content":"a"},{"role":"drop"}]}),
            None,
        )
        .await
        .unwrap();
    assert_eq!(r.result["messages"].as_array().unwrap().len(), 1);

    // before_agent_start aggregation
    let r = host
        .emit(
            json!({"type":"before_agent_start","prompt":"p","systemPrompt":["sys"]}),
            None,
        )
        .await
        .unwrap();
    assert_eq!(r.result["systemPrompt"], json!(["sys", "extra"]));
    assert_eq!(r.result["messages"][0]["content"], "prompt:p");

    // session_start -> append_entry notification using ctx state
    host.emit(json!({"type":"session_start"}), None)
        .await
        .unwrap();
    let mut seen = None;
    while let Ok(Some(m)) = tokio::time::timeout(Duration::from_secs(2), rx.recv()).await {
        if let HostMessage::Notification { method, params } = m {
            if method == "append_entry" {
                seen = Some(params);
                break;
            }
        }
    }
    let p = seen.expect("append_entry");
    assert_eq!(p["customType"], "fixture-started");
    assert_eq!(p["data"]["sid"], "s1");
    assert_eq!(p["data"]["cwd"], "/tmp");

    host.shutdown().await.unwrap();
}

/// Two concurrent sessions on ONE shared daemon keep fully separate extension
/// state: `fx_echo` counts calls per session, so the count observed in session
/// B starts at 1 no matter how many calls session A made (`OMP_EXT_TEST_SHARED`
/// is irrelevant here — this test always uses the shared daemon).
#[tokio::test]
async fn shared_daemon_sessions_are_isolated() {
    let dir = std::env::temp_dir().join(format!("omp-ext-shared-iso-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let sock = dir.join("ext.sock");
    let mut cfg = config();
    cfg.shared_socket = Some(sock.clone());
    let paths = [fixture("fixture.ts")];

    let spawn_session = |label: &'static str| {
        let cfg = cfg.clone();
        let paths = paths.clone();
        tokio::spawn(async move {
            let host = ExtHost::spawn(&cfg).await.unwrap();
            host.load(Some(&paths), None, "/tmp", Some(json!({"sessionId": label})))
                .await
                .unwrap();
            host
        })
    };
    let (ta, tb) = (spawn_session("a"), spawn_session("b"));
    let (a, b) = (ta.await.unwrap(), tb.await.unwrap());

    let n = |v: &Value| v["details"]["n"].as_u64().unwrap();
    let a1 = a.call_tool("fx_echo", "a1", json!({"text":"x"}), None).await.unwrap();
    let a2 = a.call_tool("fx_echo", "a2", json!({"text":"x"}), None).await.unwrap();
    assert_eq!(n(&a1), 1);
    assert_eq!(n(&a2), 2, "session A counts its own calls");
    let b1 = b.call_tool("fx_echo", "b1", json!({"text":"x"}), None).await.unwrap();
    assert_eq!(n(&b1), 1, "session B state is not affected by session A");

    // `list_flags`/`list_tools` are per session too.
    assert_eq!(a.list_tools().await.unwrap().len(), 1);
    assert_eq!(b.list_tools().await.unwrap().len(), 1);

    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
    // After the last session disconnects the daemon exits on its own and
    // removes the socket (2 s grace by default).
    let gone = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if !sock.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await;
    assert!(gone.is_ok(), "shared daemon did not exit after last session");
}

#[tokio::test]
async fn tools_and_commands() {
    let (host, mut rx) = loaded().await;
    let tools = host.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "fx_echo");
    assert_eq!(
        tools[0].parameters["type"], "object",
        "typebox schema from api.typebox"
    );
    let r = host
        .call_tool("fx_echo", "c1", json!({"text":"x"}), None)
        .await
        .unwrap();
    assert_eq!(r["content"][0]["text"], "echo:x");
    assert!(host.call_tool("nope", "c2", json!({}), None).await.is_err());
    let cmds = host.list_commands().await.unwrap();
    assert_eq!(cmds[0].name, "fx-cmd");
    assert_eq!(
        host.call_command("fx-cmd", "a b").await.unwrap(),
        Value::String("cmd:a b".into())
    );
    let mut got = false;
    while let Ok(Some(m)) = tokio::time::timeout(Duration::from_secs(2), rx.recv()).await {
        if matches!(m, HostMessage::Notification { ref method, .. } if method == "send_message") {
            got = true;
            break;
        }
    }
    assert!(got);
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn hook_points_map_to_pa_agent() {
    use pa_agent::abort::AbortSignal;
    let (host, _rx) = loaded().await;
    let hooks = ExtHooks::new(host.clone());
    let msgs = vec![]; // empty list round-trips through the typed hook
    let out = (hooks.transform_context())(msgs, AbortSignal::never())
        .await
        .unwrap();
    assert!(out.is_empty());
    host.shutdown().await.unwrap();
}

struct BashTool;

impl pa_agent::types::AgentTool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn description(&self) -> &'static str {
        "runs a command"
    }

    fn parameters(&self) -> &Value {
        static SCHEMA: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
        SCHEMA.get_or_init(|| {
            json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]})
        })
    }

    fn execute(
        self: std::sync::Arc<Self>,
        _id: String,
        params: Value,
        _signal: pa_agent::abort::AbortSignal,
        _on_update: pa_agent::types::AgentToolUpdateCallback,
    ) -> pa_agent::BoxFut<'static, anyhow::Result<pa_agent::types::AgentToolResult>> {
        Box::pin(async move {
            let cmd = params["command"].as_str().unwrap_or_default();
            Ok(pa_agent::types::AgentToolResult::text(format!("ran:{cmd}")))
        })
    }
}

#[tokio::test]
async fn ext_hooks_rewrite_input_and_deliver_additional_context_through_the_loop() {
    use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
    use pa_agent::scripted::ScriptedProvider;
    use pa_agent::types::{AgentMessage, Message, ToolResultContent, UsageCost};

    let (host, _rx) = loaded().await;
    let hooks = ExtHooks::new(host.clone());
    let model = pa_agent::types::Model {
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
    };
    let provider = std::sync::Arc::new(ScriptedProvider::new(model.clone()));
    let agent = Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            tools: Some(vec![std::sync::Arc::new(BashTool)]),
            ..Default::default()
        },
        stream_fn: Some(provider.stream_fn()),
        before_tool_call: Some(hooks.before_tool_call()),
        after_tool_call: Some(hooks.after_tool_call()),
        ..Default::default()
    });
    agent.set_model(model).await;
    provider.push_tool_call_turn(None, vec![("c1", "bash", json!({"command":"echo hi"}))]);
    provider.push_text_turn("done");

    agent.prompt("go").await.unwrap();
    agent.wait_for_idle().await;

    let state = agent.state().await;
    // user, assistant, toolResult, developer(additionalContext), assistant
    assert_eq!(state.messages.len(), 5, "{:?}", state.messages);
    let AgentMessage::Standard(Message::ToolResult(result)) = &state.messages[2] else {
        panic!("expected tool result");
    };
    // The tool ran with the rewritten input ("ran:echo rewritten" = 18 chars; the fixture's
    // tool_result handler then redacts the content to its length).
    let [ToolResultContent::Text(text)] = &result.content[..] else {
        panic!("expected single text block");
    };
    assert_eq!(text.text, "[redacted] 18");
    let AgentMessage::Custom(dev) = &state.messages[3] else {
        panic!("expected developer message");
    };
    assert_eq!(dev.role, "developer");
    assert_eq!(dev.payload["content"][0]["text"], "ctx-a\n\nctx-b");
    // the recorded args the second model call saw carry the rewrite
    let calls = provider.calls();
    let rewritten = calls[1].messages.iter().any(|m| match m {
        Message::Assistant(a) => a.content.iter().any(|c| {
            matches!(c, pa_agent::types::AssistantContent::ToolCall(t) if t.arguments["command"] == "echo rewritten")
        }),
        _ => false,
    });
    assert!(rewritten, "assistant tool call must carry the revised arguments");
    host.shutdown().await.unwrap();
}
