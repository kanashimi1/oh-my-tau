use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use omp_ext::lifecycle::{
    BoxFuture, BranchReason, HeadlessResponder, HostResponder, InputSource, LifecycleEvent,
    SwitchReason,
};
use omp_ext::{ExtHost, HostConfig, HostMessage};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedReceiver;

mod common;

fn config() -> HostConfig {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut cfg = HostConfig::new(root.join("../../ext-host/host.ts"));
    cfg.handler_timeout_ms = Some(2000);
    if std::env::var_os("OMP_EXT_TEST_SHARED").is_some_and(|v| v != "0") {
        let dir = std::env::temp_dir().join(format!("omp-ext-shared-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        cfg.shared_socket = Some(dir.join("ext.sock"));
    }
    cfg
}

async fn loaded() -> Arc<ExtHost> {
    let host = ExtHost::spawn(&config()).await.unwrap();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lifecycle.ts");
    let r = host
        .load(
            Some(&[fixture]),
            None,
            "/tmp",
            Some(json!({"sessionId": "s1"})),
        )
        .await
        .unwrap();
    assert!(r.errors.is_empty(), "{r:?}");
    assert_eq!(r.loaded.len(), 1);
    host
}

async fn entry(rx: &mut UnboundedReceiver<HostMessage>, custom_type: &str) -> Value {
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .unwrap_or_else(|_| panic!("no {custom_type} entry"))
            .expect("host closed");
        if let HostMessage::Notification { method, params } = msg {
            if method == "append_entry" && params["customType"] == custom_type {
                return params["data"].clone();
            }
        }
    }
}

/// The fixture re-serializes the event it received; it must equal our serialization byte for byte.
async fn assert_wire(
    host: &ExtHost,
    rx: &mut UnboundedReceiver<HostMessage>,
    ev: LifecycleEvent,
    expected: &str,
) {
    assert_eq!(serde_json::to_string(&ev).unwrap(), expected);
    let ty = ev.to_value()["type"].as_str().unwrap().to_owned();
    let r = host.emit_event(&ev, None).await.unwrap();
    assert!(r.errors.is_empty(), "{ty}: {:?}", r.errors);
    assert_eq!(
        entry(rx, &format!("seen:{ty}")).await,
        json!(expected),
        "{ty}"
    );
}

#[tokio::test]
async fn observed_events_reach_extensions_with_omp_payloads() {
    if common::bun_missing("observed_events_reach_extensions_with_omp_payloads") {
        return;
    }
    let host = loaded().await;
    let mut rx = host.take_incoming().unwrap();
    let msg = json!({"role":"user","content":"hi","timestamp":1});
    let a = json!({"role":"assistant","content":[{"type":"text","text":"yo"}]});
    let tr = json!({"role":"toolResult","toolCallId":"c1","toolName":"bash","content":[],"isError":false});

    let cases: Vec<(LifecycleEvent, String)> = vec![
        (LifecycleEvent::SessionStart, r#"{"type":"session_start"}"#.into()),
        (LifecycleEvent::AgentStart, r#"{"type":"agent_start"}"#.into()),
        (
            LifecycleEvent::AgentEnd { messages: vec![msg.clone()], will_continue: None },
            format!(r#"{{"type":"agent_end","messages":[{msg}]}}"#),
        ),
        (
            LifecycleEvent::AgentEnd { messages: vec![], will_continue: Some(true) },
            r#"{"type":"agent_end","messages":[],"willContinue":true}"#.into(),
        ),
        (
            LifecycleEvent::TurnStart { turn_index: 2, timestamp: 1700 },
            r#"{"type":"turn_start","turnIndex":2,"timestamp":1700}"#.into(),
        ),
        (
            LifecycleEvent::TurnEnd { turn_index: 2, message: a.clone(), tool_results: vec![tr.clone()] },
            format!(r#"{{"type":"turn_end","turnIndex":2,"message":{a},"toolResults":[{tr}]}}"#),
        ),
        (
            LifecycleEvent::MessageStart { message: msg.clone() },
            format!(r#"{{"type":"message_start","message":{msg}}}"#),
        ),
        (
            LifecycleEvent::MessageUpdate { message: a.clone(), assistant_message_event: json!({"type":"text_delta","delta":"y"}) },
            format!(r#"{{"type":"message_update","message":{a},"assistantMessageEvent":{{"type":"text_delta","delta":"y"}}}}"#),
        ),
        (
            LifecycleEvent::MessageEnd { message: a.clone() },
            format!(r#"{{"type":"message_end","message":{a}}}"#),
        ),
        (
            LifecycleEvent::ToolExecutionStart { tool_call_id: "c1".into(), tool_name: "bash".into(), args: json!({"command":"ls"}), intent: Some("list".into()) },
            r#"{"type":"tool_execution_start","toolCallId":"c1","toolName":"bash","args":{"command":"ls"},"intent":"list"}"#.into(),
        ),
        (
            LifecycleEvent::ToolExecutionStart { tool_call_id: "c1".into(), tool_name: "bash".into(), args: json!({}), intent: None },
            r#"{"type":"tool_execution_start","toolCallId":"c1","toolName":"bash","args":{}}"#.into(),
        ),
        (
            LifecycleEvent::ToolExecutionUpdate { tool_call_id: "c1".into(), tool_name: "bash".into(), args: json!({}), partial_result: json!({"content":[]}) },
            r#"{"type":"tool_execution_update","toolCallId":"c1","toolName":"bash","args":{},"partialResult":{"content":[]}}"#.into(),
        ),
        (
            LifecycleEvent::ToolExecutionEnd { tool_call_id: "c1".into(), tool_name: "bash".into(), result: json!({"content":[]}), is_error: false },
            r#"{"type":"tool_execution_end","toolCallId":"c1","toolName":"bash","result":{"content":[]},"isError":false}"#.into(),
        ),
        (
            LifecycleEvent::SessionCompact { compaction_entry: json!({"type":"compaction","id":"e1"}), from_extension: true },
            r#"{"type":"session_compact","compactionEntry":{"type":"compaction","id":"e1"},"fromExtension":true}"#.into(),
        ),
        (
            LifecycleEvent::SessionSwitch { reason: SwitchReason::Resume, previous_session_file: Some("/a.jsonl".into()) },
            r#"{"type":"session_switch","reason":"resume","previousSessionFile":"/a.jsonl"}"#.into(),
        ),
        (
            LifecycleEvent::SessionSwitch { reason: SwitchReason::New, previous_session_file: None },
            r#"{"type":"session_switch","reason":"new"}"#.into(),
        ),
        (
            LifecycleEvent::SessionBranch { reason: BranchReason::Btw, previous_session_file: None },
            r#"{"type":"session_branch","reason":"btw"}"#.into(),
        ),
        (
            LifecycleEvent::SessionTree { new_leaf_id: Some("n".into()), old_leaf_id: None, summary_entry: None, from_extension: Some(false) },
            r#"{"type":"session_tree","newLeafId":"n","oldLeafId":null,"fromExtension":false}"#.into(),
        ),
        (LifecycleEvent::SessionShutdown, r#"{"type":"session_shutdown"}"#.into()),
    ];
    for (ev, expected) in cases {
        assert_wire(&host, &mut rx, ev, &expected).await;
    }

    host.shutdown().await.unwrap();
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn events_with_results_are_aggregated_and_shaped() {
    if common::bun_missing("events_with_results_are_aggregated_and_shaped") {
        return;
    }
    let host = loaded().await;
    let mut rx = host.take_incoming().unwrap();

    let out = host
        .emit_before_agent_start(
            "do it".into(),
            Some(json!([{"type":"image"}])),
            vec!["base".into()],
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        out.messages,
        vec![json!({"customType":"fx","content":"hello","display":false})]
    );
    assert_eq!(
        out.system_prompt,
        Some(vec!["sys-a".into(), "sys-b".into()])
    );
    assert_eq!(
        entry(&mut rx, "seen:before_agent_start").await,
        json!(
            r#"{"type":"before_agent_start","prompt":"do it","images":[{"type":"image"}],"systemPrompt":["base"]}"#
        )
    );

    let out = host
        .emit_input("raw".into(), None, InputSource::Rpc, None)
        .await
        .unwrap();
    assert_eq!(
        (out.handled, out.text.as_deref(), out.images),
        (false, Some("rewritten"), Some(json!([])))
    );
    assert_eq!(
        entry(&mut rx, "seen:input").await,
        json!(r#"{"type":"input","text":"raw","source":"rpc"}"#)
    );

    let out = host
        .emit_session_before_compact(
            json!({"firstKeptEntryId":"e2"}),
            vec![json!({"id":"e1"})],
            Some("be brief".into()),
            None,
        )
        .await
        .unwrap();
    assert!(!out.cancel);
    assert_eq!(out.result["compaction"], json!({"summary":"ext summary"}));
    assert_eq!(
        entry(&mut rx, "seen:session_before_compact").await,
        json!(
            r#"{"type":"session_before_compact","preparation":{"firstKeptEntryId":"e2"},"branchEntries":[{"id":"e1"}],"customInstructions":"be brief"}"#
        )
    );

    assert!(
        host.emit_session_before_switch(SwitchReason::Fork, None, None)
            .await
            .unwrap()
            .cancel
    );
    assert_eq!(
        entry(&mut rx, "seen:session_before_switch").await,
        json!(r#"{"type":"session_before_switch","reason":"fork"}"#)
    );
    let out = host
        .emit_session_before_switch(SwitchReason::Resume, Some("/t.jsonl".into()), None)
        .await
        .unwrap();
    assert!(out.cancel);
    assert_eq!(
        entry(&mut rx, "seen:session_before_switch").await,
        json!(
            r#"{"type":"session_before_switch","reason":"resume","targetSessionFile":"/t.jsonl"}"#
        )
    );

    assert!(
        !host
            .emit_session_before_branch(BranchReason::Branch, "e9".into(), None)
            .await
            .unwrap()
            .cancel
    );
    assert_eq!(
        entry(&mut rx, "seen:session_before_branch").await,
        json!(r#"{"type":"session_before_branch","reason":"branch","entryId":"e9"}"#)
    );
    assert!(
        host.emit_session_before_tree(json!({"targetId":"t"}), None)
            .await
            .unwrap()
            .cancel
    );
    assert_eq!(
        entry(&mut rx, "seen:session_before_tree").await,
        json!(r#"{"type":"session_before_tree","preparation":{"targetId":"t"}}"#)
    );

    let stop = LifecycleEvent::SessionStop {
        messages: vec![],
        turn_id: 3,
        last_assistant_message: None,
        session_id: "s1".into(),
        session_file: Some("/s.jsonl".into()),
        stop_hook_active: false,
    };
    let out = host.emit_session_stop(stop, None).await.unwrap();
    assert!(out.wants_continuation() && out.continue_turn && !out.block);
    assert_eq!(out.additional_context.as_deref(), Some("keep going"));
    assert_eq!(
        entry(&mut rx, "seen:session_stop").await,
        json!(
            r#"{"type":"session_stop","messages":[],"turn_id":3,"session_id":"s1","session_file":"/s.jsonl","stop_hook_active":false}"#
        )
    );
    assert!(host
        .emit_session_stop(LifecycleEvent::AgentStart, None)
        .await
        .is_err());

    host.shutdown().await.unwrap();
}

struct Scripted;
impl HostResponder for Scripted {
    fn ui_confirm<'a>(&'a self, title: &'a str, message: &'a str) -> BoxFuture<'a, bool> {
        Box::pin(async move { title == "title" && message == "message" })
    }
    fn navigate_tree<'a>(
        &'a self,
        target: &'a str,
        options: Value,
    ) -> BoxFuture<'a, Result<Value, String>> {
        Box::pin(
            async move { Ok(json!({"cancelled": true, "target": target, "options": options})) },
        )
    }
    fn set_model(&self, model: Value) -> BoxFuture<'_, bool> {
        Box::pin(async move { model["id"] == "m" })
    }
}

async fn probe(responder: Arc<dyn HostResponder>) -> Value {
    let host = loaded().await;
    let incoming = host.take_incoming().unwrap();
    let mut notes = omp_ext::lifecycle::serve(&host, incoming, responder);
    let r = host
        .emit(json!({"type":"probe_requests"}), None)
        .await
        .unwrap();
    assert!(r.errors.is_empty(), "{:?}", r.errors);
    let data = entry(&mut notes, "probe").await;
    host.shutdown().await.unwrap();
    serde_json::from_str(data.as_str().unwrap()).unwrap()
}

#[tokio::test]
async fn host_requests_default_to_headless_answers() {
    if common::bun_missing("host_requests_default_to_headless_answers") {
        return;
    }
    let got = probe(Arc::new(HeadlessResponder)).await;
    assert_eq!(
        got,
        json!({"confirm": false, "tree": {"cancelled": false}, "model": false})
    );
}

#[tokio::test]
async fn host_requests_use_a_pluggable_responder() {
    if common::bun_missing("host_requests_use_a_pluggable_responder") {
        return;
    }
    let got = probe(Arc::new(Scripted)).await;
    assert_eq!(
        got,
        json!({"confirm": true, "tree": {"cancelled": true, "target": "entry-1", "options": {"summarize": true}}, "model": true})
    );
}
