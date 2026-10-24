use std::path::PathBuf;
use std::time::Duration;

use omp_ext::hooks::{self, ExtHooks};
use omp_ext::{ExtHost, HostConfig, HostMessage};
use serde_json::{json, Value};

mod common;

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
    if common::bun_missing("gates_rewrites_and_host_survives_failing_handlers") {
        return;
    }
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
    if common::bun_missing("shared_daemon_sessions_are_isolated") {
        return;
    }
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
            host.load(
                Some(&paths),
                None,
                "/tmp",
                Some(json!({"sessionId": label})),
            )
            .await
            .unwrap();
            host
        })
    };
    let (ta, tb) = (spawn_session("a"), spawn_session("b"));
    let (a, b) = (ta.await.unwrap(), tb.await.unwrap());

    assert_eq!(
        daemon_pids(&sock).len(),
        1,
        "two sessions share exactly one bun daemon"
    );

    let n = |v: &Value| v["details"]["n"].as_u64().unwrap();
    let a1 = a
        .call_tool("fx_echo", "a1", json!({"text":"x"}), None)
        .await
        .unwrap();
    let a2 = a
        .call_tool("fx_echo", "a2", json!({"text":"x"}), None)
        .await
        .unwrap();
    assert_eq!(n(&a1), 1);
    assert_eq!(n(&a2), 2, "session A counts its own calls");
    let b1 = b
        .call_tool("fx_echo", "b1", json!({"text":"x"}), None)
        .await
        .unwrap();
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
    assert!(
        gone.is_ok(),
        "shared daemon did not exit after last session"
    );
}

/// Find the shared-daemon pid by its `OMP_RS_EXT_SOCKET` env var (the daemon's
/// cmdline is just `bun .../shard.ts`, shared with any other test daemon).
fn daemon_pid(sock: &std::path::Path) -> Option<u32> {
    daemon_pids(sock).into_iter().next()
}

/// All live daemon pids serving `sock`.
fn daemon_pids(sock: &std::path::Path) -> Vec<u32> {
    let mut out = Vec::new();
    let needle = format!("OMP_RS_EXT_SOCKET={}", sock.display());
    let Ok(rd) = std::fs::read_dir("/proc") else {
        return out;
    };
    for entry in rd {
        let Ok(entry) = entry else { continue };
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        let Ok(env) = std::fs::read(format!("/proc/{pid}/environ")) else {
            continue;
        };
        if env.windows(needle.len()).any(|w| w == needle.as_bytes()) {
            out.push(pid);
        }
    }
    out
}

/// `kill -9` on the shared daemon mid-session: in-flight RPCs fail cleanly,
/// the session transparently reconnects to a freshly started shared daemon
/// (generation bump, new pid; private host only if that fails),
/// extensions re-load, and emits/hooks fire again — the next prompt works.
#[tokio::test]
async fn shared_daemon_death_falls_back_to_private_host() {
    if common::bun_missing("shared_daemon_death_falls_back_to_private_host") {
        return;
    }
    let dir = std::env::temp_dir().join(format!("omp-ext-shared-death-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let sock = dir.join("ext.sock");
    let mut cfg = config();
    cfg.shared_socket = Some(sock.clone());
    // On-the-fly extension with a tool that never answers, so an RPC can be
    // genuinely in flight when the daemon dies.
    let slow = dir.join("slow.ts");
    std::fs::write(
        &slow,
        r#"export default function (pi: any): void {
  pi.registerTool({
    name: "fx_slow", label: "Slow", description: "never returns",
    parameters: {},
    async execute() { await new Promise(() => {}); return { content: [] }; },
  });
}
"#,
    )
    .unwrap();
    let paths = [fixture("fixture.ts"), slow];

    let host = ExtHost::spawn(&cfg).await.unwrap();
    let _rx = host.take_incoming().unwrap();
    host.load(
        Some(&paths),
        None,
        "/tmp",
        Some(json!({"sessionId": "s1", "sessionName": "old title"})),
    )
    .await
    .unwrap();
    let r = host
        .call_tool("fx_echo", "e1", json!({"text":"x"}), None)
        .await
        .unwrap();
    assert_eq!(r["content"][0]["text"], "echo:x");
    let pid = daemon_pid(&sock).expect("shared daemon running");

    // Subscribe before the kill: `watch::subscribe` does not replay a bump
    // that already happened.
    let mut generations = host.reconnections();

    // Kill the daemon while an RPC is in flight: it must error, not hang.
    let in_flight = {
        let host = host.clone();
        tokio::spawn(async move { host.call_tool("fx_slow", "h", json!({}), None).await })
    };
    tokio::time::sleep(Duration::from_millis(300)).await;
    let killed = std::process::Command::new("kill")
        .args(["-9"])
        .arg(pid.to_string())
        .status()
        .unwrap();
    assert!(killed.success());
    let res = tokio::time::timeout(Duration::from_secs(10), in_flight)
        .await
        .expect("in-flight RPC must not hang")
        .unwrap();
    assert!(res.is_err(), "in-flight call must fail cleanly: {res:?}");

    // Transparent fallback: a reconnect generation is published.
    let bumped = tokio::time::timeout(Duration::from_secs(15), generations.changed())
        .await
        .expect("private-host fallback after daemon death");
    assert!(bumped.is_ok(), "generation watch not dropped");
    assert!(host.may_reconnect());

    // Like the omp-cli supervisor: re-load extensions on the fresh host.
    let mut rx = host.take_incoming().expect("fresh incoming channel");
    host.load(
        Some(&paths),
        None,
        "/tmp",
        Some(json!({"sessionId": "s1", "sessionName": "old title"})),
    )
    .await
    .unwrap();

    // Hooks fire again: session_start -> append_entry from the reloaded ext.
    host.emit(json!({"type":"session_start"}), None)
        .await
        .unwrap();
    let mut seen = None;
    while let Ok(Some(m)) = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
        if let HostMessage::Notification { method, params } = m {
            if method == "append_entry" {
                seen = Some(params);
                break;
            }
        }
    }
    let p = seen.expect("append_entry after recovery");
    assert_eq!(p["customType"], "fixture-started");
    assert_eq!(p["data"]["sid"], "s1");
    assert_eq!(p["data"]["cwd"], "/tmp");

    // The next prompt works: tool calls on the private fallback host. Fresh
    // module state means the per-session counter restarted at 1.
    let r = host
        .call_tool("fx_echo", "e2", json!({"text":"z"}), None)
        .await
        .unwrap();
    assert_eq!(r["content"][0]["text"], "echo:z");
    assert_eq!(
        r["details"]["n"], 1,
        "fallback host has fresh extension state"
    );
    let new_pid = daemon_pid(&sock).expect("shared daemon restarted");
    assert_ne!(
        new_pid, pid,
        "reconnected to a new shared daemon, not a private host"
    );

    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn tools_and_commands() {
    if common::bun_missing("tools_and_commands") {
        return;
    }
    let (host, mut rx) = loaded().await;
    let tools = host.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "fx_echo");
    // `api.typebox` comes from the installed omp TS package; without it the
    // fixture registers the tool with a plain (schema-less) parameters object.
    if !common::omp_pkg_missing("tools_and_commands") {
        assert_eq!(
            tools[0].parameters["type"], "object",
            "typebox schema from api.typebox"
        );
    }
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

/// `skipExtensions` leaves the named extension's handlers out of that one
/// emit; the others still run, and the next emit without it runs everything.
#[tokio::test]
async fn context_emit_skips_listed_extensions() {
    if common::bun_missing("context_emit_skips_listed_extensions") {
        return;
    }
    use pa_agent::types::AgentMessage;
    let host = ExtHost::spawn(&config()).await.unwrap();
    let r = host
        .load(
            Some(&[fixture("fixture.ts"), fixture("context-slim.ts")]),
            None,
            "/tmp",
            Some(json!({"sessionId": "s1"})),
        )
        .await
        .unwrap();
    assert_eq!(r.loaded.len(), 2, "{r:?}");
    let msgs: Vec<AgentMessage> = vec![AgentMessage::user("hi")];
    let texts = |m: &[AgentMessage]| {
        m.iter()
            .map(|m| serde_json::to_value(m).unwrap()["content"].clone())
            .collect::<Vec<_>>()
    };
    let slimmed = hooks::context_owned(&host, msgs.clone(), None)
        .await
        .unwrap();
    assert_eq!(slimmed.len(), 2, "context-slim ran: {:?}", texts(&slimmed));
    let skipped =
        hooks::context_owned_skipping(&host, msgs.clone(), None, &["context-slim".to_owned()])
            .await
            .unwrap();
    assert_eq!(
        skipped.len(),
        1,
        "context-slim skipped: {:?}",
        texts(&skipped)
    );
    // Per emit only: the next plain emit runs it again.
    assert_eq!(
        hooks::context_owned(&host, msgs, None).await.unwrap().len(),
        2
    );
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn hook_points_map_to_pa_agent() {
    if common::bun_missing("hook_points_map_to_pa_agent") {
        return;
    }
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
    if common::bun_missing("ext_hooks_rewrite_input_and_deliver_additional_context_through_the_loop") {
        return;
    }
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
    assert!(
        rewritten,
        "assistant tool call must carry the revised arguments"
    );
    host.shutdown().await.unwrap();
}

/// A `message_update` handler that mutates `event.message` must not leak its
/// edits into the next update or break the delta chain: every rebuilt update
/// equals the message a full frame would carry.
#[tokio::test]
async fn message_update_deltas_are_isolated_from_handler_mutations() {
    if common::bun_missing("message_update_deltas_are_isolated_from_handler_mutations") {
        return;
    }
    let host = ExtHost::spawn(&config()).await.unwrap();
    let r = host
        .load(Some(&[fixture("mutating.ts")]), None, "/tmp", None)
        .await
        .unwrap();
    assert_eq!(r.loaded.len(), 1, "{r:?}");
    let header =
        json!({"role": "assistant", "api": "a", "usage": {"input": 1, "cost": {"total": 0}}});
    let message = |text: &str, x: &str| {
        let mut m = header.clone();
        m["content"] = json!([
            {"type": "text", "text": text},
            {"type": "toolCall", "id": "t", "name": "n", "arguments": {"x": x}}
        ]);
        m
    };
    let seen = |r: &omp_ext::EmitResult| -> Value {
        assert_ne!(r.result, json!({"resync": true}), "delta rejected: {r:?}");
        serde_json::from_str(r.errors[0]["error"].as_str().unwrap()).unwrap()
    };
    let frames = [
        (
            json!({"type": "message_update", "message": message("a", "1"),
                   "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": "a"},
                   "delta": {"id": 7, "seq": 0}}),
            message("a", "1"),
        ),
        (
            json!({"type": "message_update",
                   "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": "b"},
                   "delta": {"id": 7, "seq": 1, "n": 2, "b": [
                       {"i": 0, "v": "a"}, {"i": 1, "v": "p", "x": {"o": {"x": {"+": "2"}}}}]}}),
            message("ab", "12"),
        ),
        (
            json!({"type": "message_update",
                   "assistantMessageEvent": {"type": "toolcall_delta", "contentIndex": 1, "delta": ""},
                   "delta": {"id": 7, "seq": 2, "n": 2, "b": [{"i": 1, "v": "k"}]}}),
            message("ab", "12"),
        ),
    ];
    for (frame, want) in frames {
        let r = host.emit(frame, None).await.unwrap();
        assert_eq!(seen(&r), want);
    }
}

/// Two awaited emits parked behind a fire-and-forget emit whose handler then
/// blocks on a core request: both must stop waiting. (`ReviewExtDelta`: the
/// single wake slot woke only the last parked emit; the other waited for the
/// core's answer, which only comes once both returned -- a deadlock.)
#[tokio::test]
async fn blocked_nowait_handler_wakes_every_parked_emit() {
    if common::bun_missing("blocked_nowait_handler_wakes_every_parked_emit") {
        return;
    }
    let mut cfg = config();
    cfg.handler_timeout_ms = Some(30_000);
    let host = ExtHost::spawn(&cfg).await.unwrap();
    let mut rx = host.take_incoming().unwrap();
    let r = host
        .load(Some(&[fixture("nowait_blocking.ts")]), None, "/tmp", None)
        .await
        .unwrap();
    assert_eq!(r.loaded.len(), 1, "{r:?}");
    host.emit_nowait(&json!({"type": "agent_start"}), None, |_| {})
        .unwrap();
    let parked = |n: u64| {
        let host = host.clone();
        tokio::spawn(async move {
            host.emit(json!({"type": "turn_start", "turnIndex": n}), None)
                .await
        })
    };
    let (a, b) = (parked(1), parked(2));
    let (a, b) = tokio::time::timeout(Duration::from_secs(5), async { (a.await, b.await) })
        .await
        .expect("a parked emit stayed blocked behind the blocked nowait handler");
    assert!(a.unwrap().is_ok() && b.unwrap().is_ok());

    // Answer the handler's compact request so the nowait emit can finish.
    loop {
        let m = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("compact request")
            .expect("incoming open");
        if let HostMessage::Request { id, method, .. } = m {
            assert_eq!(method, "compact");
            host.respond(id, Ok(Value::Null)).await.unwrap();
            break;
        }
    }
    host.shutdown().await.unwrap();
}

/// Many clients racing to be first start exactly one daemon (start lock).
#[tokio::test]
async fn concurrent_first_clients_start_one_daemon() {
    if common::bun_missing("concurrent_first_clients_start_one_daemon") {
        return;
    }
    let dir = std::env::temp_dir().join(format!("omp-ext-shared-race-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let sock = dir.join("ext.sock");
    let mut cfg = config();
    cfg.shared_socket = Some(sock.clone());
    let tasks: Vec<_> = (0..4)
        .map(|_| {
            let cfg = cfg.clone();
            tokio::spawn(async move { ExtHost::spawn(&cfg).await.unwrap() })
        })
        .collect();
    let mut hosts = Vec::new();
    for t in tasks {
        hosts.push(t.await.unwrap());
    }
    assert_eq!(
        daemon_pids(&sock).len(),
        1,
        "racing first clients must share one daemon"
    );
    for h in hosts {
        h.shutdown().await.unwrap();
    }
}

/// Each session's Worker sees ITS client's environment (`process.env`,
/// `os.homedir()`), not the env of the process that happened to start the
/// shared daemon.
#[tokio::test]
async fn shared_daemon_sessions_get_their_own_env() {
    if common::bun_missing("shared_daemon_sessions_get_their_own_env") {
        return;
    }
    let dir = std::env::temp_dir().join(format!("omp-ext-shared-env-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let sock = dir.join("ext.sock");
    let ext = dir.join("envext.ts");
    std::fs::write(
        &ext,
        r#"import { homedir } from "node:os";
export default function (pi: any): void {
  pi.registerTool({
    name: "fx_env", label: "Env", description: "env",
    parameters: {},
    async execute() {
      return { content: [{ type: "text", text: `${process.env.FX_MARK}|${homedir()}|${process.env.HOOK_TRACE}` }] };
    },
  });
}
"#,
    )
    .unwrap();
    let paths = [ext];
    let mut hosts = Vec::new();
    for (mark, home) in [("a", "/tmp/fx-home-a"), ("b", "/tmp/fx-home-b")] {
        let mut cfg = config();
        cfg.shared_socket = Some(sock.clone());
        cfg.env = vec![
            ("FX_MARK".into(), mark.into()),
            ("HOME".into(), home.into()),
        ];
        let host = ExtHost::spawn(&cfg).await.unwrap();
        host.load(Some(&paths), None, "/tmp", None).await.unwrap();
        hosts.push((mark, home, host));
    }
    assert_eq!(daemon_pids(&sock).len(), 1);
    for (mark, home, host) in &hosts {
        let r = host
            .call_tool("fx_env", "e", json!({}), None)
            .await
            .unwrap();
        assert_eq!(r["content"][0]["text"], format!("{mark}|{home}|0"));
    }
    for (_, _, h) in hosts {
        h.shutdown().await.unwrap();
    }
}

/// Shared daemon regressions behind hung omp-rs panes:
/// - a frame larger than the socket buffer (here a 3 MB tool result; live it
///   was a `context` event of a long session) used to be truncated by
///   `shard.ts` (Bun `socket.write` writes a prefix under backpressure) and
///   the client waited forever for the rest of the line;
/// - child processes spawned by an extension ran with the daemon's EMPTY env
///   and the daemon's cwd instead of the session's.
#[tokio::test]
async fn shared_daemon_delivers_big_frames_and_spawns_with_session_env() {
    if common::bun_missing("shared_daemon_delivers_big_frames_and_spawns_with_session_env") {
        return;
    }
    let dir = std::env::temp_dir().join(format!("omp-ext-shared-big-{}", std::process::id()));
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let sock = dir.join("ext.sock");
    let ext = dir.join("bigext.ts");
    std::fs::write(
        &ext,
        r#"import { execSync } from "node:child_process";
export default function (pi: any): void {
  pi.registerTool({
    name: "fx_big", label: "Big", description: "big",
    parameters: {},
    async execute() {
      return { content: [{ type: "text", text: "x".repeat(3_000_000) }] };
    },
  });
  pi.registerTool({
    name: "fx_child", label: "Child", description: "child env",
    parameters: {},
    async execute() {
      const out = execSync('printf "%s|%s|" "$FX_MARK" "$HOME"; pwd').toString().trim();
      return { content: [{ type: "text", text: out }] };
    },
  });
}
"#,
    )
    .unwrap();
    let mut cfg = config();
    cfg.shared_socket = Some(sock.clone());
    cfg.env = vec![
        ("FX_MARK".into(), "m".into()),
        ("HOME".into(), "/tmp/fx-home-big".into()),
    ];
    let host = ExtHost::spawn(&cfg).await.unwrap();
    host.load(Some(&[ext]), None, work.to_str().unwrap(), None)
        .await
        .unwrap();

    let big = tokio::time::timeout(
        Duration::from_secs(20),
        host.call_tool("fx_big", "b", json!({}), None),
    )
    .await
    .expect("3 MB frame must arrive whole, not hang on a truncated line")
    .unwrap();
    assert_eq!(
        big["content"][0]["text"].as_str().map(str::len),
        Some(3_000_000)
    );

    let child = host
        .call_tool("fx_child", "c", json!({}), None)
        .await
        .unwrap();
    assert_eq!(
        child["content"][0]["text"],
        format!("m|/tmp/fx-home-big|{}", work.display()),
        "extension child processes inherit the session env and cwd"
    );
    host.shutdown().await.unwrap();
}
