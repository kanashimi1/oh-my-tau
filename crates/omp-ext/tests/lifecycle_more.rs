//! Payload shapes for the retry/compaction/goal/approval/user-shell events and the
//! `assistant_message` / `user_bash` / `user_python` result aggregation (same fixture as `lifecycle.rs`).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use omp_ext::lifecycle::{ApprovalMode, CompactionAction, CompactionReason, LifecycleEvent};
use omp_ext::{ExtHost, HostConfig, HostMessage};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedReceiver;

mod common;

async fn loaded() -> Arc<ExtHost> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut cfg = HostConfig::new(root.join("../../ext-host/host.ts"));
    cfg.handler_timeout_ms = Some(2000);
    if std::env::var_os("OMP_EXT_TEST_SHARED").is_some_and(|v| v != "0") {
        let dir = std::env::temp_dir().join(format!("omp-ext-shared-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        cfg.shared_socket = Some(dir.join("ext.sock"));
    }
    let host = ExtHost::spawn(&cfg).await.unwrap();
    let r = host
        .load(
            Some(&[root.join("tests/fixtures/lifecycle.ts")]),
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

#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn retry_compaction_goal_approval_and_user_shell_payloads() {
    if common::bun_missing("retry_compaction_goal_approval_and_user_shell_payloads") {
        return;
    }
    let host = loaded().await;
    let mut rx = host.take_incoming().unwrap();
    let s = String::from;
    let cases: Vec<(LifecycleEvent, &str)> = vec![
        (
            LifecycleEvent::AutoCompactionStart {
                reason: CompactionReason::Threshold,
                action: CompactionAction::ContextFull,
            },
            r#"{"type":"auto_compaction_start","reason":"threshold","action":"context-full"}"#,
        ),
        (
            LifecycleEvent::AutoCompactionEnd {
                action: CompactionAction::Remote,
                result: Some(json!({"summary":"s"})),
                aborted: false,
                will_retry: true,
                error_message: None,
                skipped: Some(false),
            },
            r#"{"type":"auto_compaction_end","action":"remote","result":{"summary":"s"},"aborted":false,"willRetry":true,"skipped":false}"#,
        ),
        (
            LifecycleEvent::AutoCompactionEnd {
                action: CompactionAction::Shake,
                result: None,
                aborted: true,
                will_retry: false,
                error_message: Some(s("boom")),
                skipped: None,
            },
            r#"{"type":"auto_compaction_end","action":"shake","aborted":true,"willRetry":false,"errorMessage":"boom"}"#,
        ),
        (
            LifecycleEvent::AutoRetryStart {
                attempt: 1,
                max_attempts: 3,
                delay_ms: 2000,
                error_message: s("429"),
                error_id: Some(7),
            },
            r#"{"type":"auto_retry_start","attempt":1,"maxAttempts":3,"delayMs":2000,"errorMessage":"429","errorId":7}"#,
        ),
        (
            LifecycleEvent::AutoRetryEnd {
                success: false,
                attempt: 3,
                final_error: Some(s("x")),
                retry_errors: Some(vec![json!({"entryId":"e","note":"n"})]),
            },
            r#"{"type":"auto_retry_end","success":false,"attempt":3,"finalError":"x","retryErrors":[{"entryId":"e","note":"n"}]}"#,
        ),
        (
            LifecycleEvent::AutoRetryEnd {
                success: true,
                attempt: 2,
                final_error: None,
                retry_errors: None,
            },
            r#"{"type":"auto_retry_end","success":true,"attempt":2}"#,
        ),
        (
            LifecycleEvent::RetryFallbackApplied {
                from: s("a/b"),
                to: s("c/d"),
                role: s("default"),
                reason: Some(s("quota")),
            },
            r#"{"type":"retry_fallback_applied","from":"a/b","to":"c/d","role":"default","reason":"quota"}"#,
        ),
        (
            LifecycleEvent::RetryFallbackSucceeded {
                model: s("c/d"),
                role: s("default"),
            },
            r#"{"type":"retry_fallback_succeeded","model":"c/d","role":"default"}"#,
        ),
        (
            LifecycleEvent::TtsrTriggered {
                rules: vec![json!({"name":"r"})],
            },
            r#"{"type":"ttsr_triggered","rules":[{"name":"r"}]}"#,
        ),
        (
            LifecycleEvent::TodoReminder {
                todos: vec![json!({"content":"t","status":"pending"})],
                attempt: 1,
                max_attempts: 3,
            },
            r#"{"type":"todo_reminder","todos":[{"content":"t","status":"pending"}],"attempt":1,"maxAttempts":3}"#,
        ),
        (
            LifecycleEvent::GoalUpdated {
                goal: None,
                state: None,
            },
            r#"{"type":"goal_updated","goal":null}"#,
        ),
        (
            LifecycleEvent::GoalUpdated {
                goal: Some(json!({"id":"g"})),
                state: Some(json!({"mode":"on"})),
            },
            r#"{"type":"goal_updated","goal":{"id":"g"},"state":{"mode":"on"}}"#,
        ),
        (
            LifecycleEvent::ToolApprovalRequested {
                session_id: s("s1"),
                tool_name: s("bash"),
                tool_call_id: s("c1"),
                reason: Some(s("exec")),
                approval_mode: ApprovalMode::AlwaysAsk,
            },
            r#"{"type":"tool_approval_requested","sessionId":"s1","toolName":"bash","toolCallId":"c1","reason":"exec","approvalMode":"always-ask"}"#,
        ),
        (
            LifecycleEvent::ToolApprovalRequested {
                session_id: s("s1"),
                tool_name: s("bash"),
                tool_call_id: s("c1"),
                reason: None,
                approval_mode: ApprovalMode::Yolo,
            },
            r#"{"type":"tool_approval_requested","sessionId":"s1","toolName":"bash","toolCallId":"c1","approvalMode":"yolo"}"#,
        ),
        (
            LifecycleEvent::ToolApprovalResolved {
                session_id: s("s1"),
                tool_name: s("bash"),
                tool_call_id: s("c1"),
                approved: false,
                reason: Some(s("no UI")),
            },
            r#"{"type":"tool_approval_resolved","sessionId":"s1","toolName":"bash","toolCallId":"c1","approved":false,"reason":"no UI"}"#,
        ),
        (
            LifecycleEvent::UserBash {
                command: s("ls"),
                exclude_from_context: true,
                cwd: s("/tmp"),
            },
            r#"{"type":"user_bash","command":"ls","excludeFromContext":true,"cwd":"/tmp"}"#,
        ),
        (
            LifecycleEvent::UserPython {
                code: s("1+1"),
                exclude_from_context: false,
                cwd: s("/tmp"),
            },
            r#"{"type":"user_python","code":"1+1","excludeFromContext":false,"cwd":"/tmp"}"#,
        ),
        (
            LifecycleEvent::AssistantMessage {
                message: json!({"role":"assistant","content":[{"type":"text","text":"a"}]}),
            },
            r#"{"type":"assistant_message","message":{"role":"assistant","content":[{"type":"text","text":"a"}]}}"#,
        ),
    ];
    for (ev, expected) in cases {
        assert_wire(&host, &mut rx, ev, expected).await;
    }
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn assistant_message_rewrite_and_user_shell_results_aggregate() {
    if common::bun_missing("assistant_message_rewrite_and_user_shell_results_aggregate") {
        return;
    }
    let host = loaded().await;
    let message = json!({"role":"assistant","content":[
        {"type":"thinking","thinking":"hm"},
        {"type":"text","text":"hello","textSignature":"sig"},
    ]});
    let out = host.emit_assistant_message(message, None).await.unwrap();
    assert_eq!(
        out.content,
        Some(vec![
            json!({"type":"thinking","thinking":"hm"}),
            json!({"type":"text","text":"HELLO"})
        ])
    );
    let none = host
        .emit_assistant_message(
            json!({"role":"assistant","content":[{"type":"text","text":"ABC"}]}),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        none.content, None,
        "uppercase text is unchanged, so nothing is rewritten"
    );

    let bash = host
        .emit_user_bash("ls".into(), false, "/tmp".into(), None)
        .await
        .unwrap();
    assert_eq!(bash.result, Some(json!({"output":"handled","exitCode":0})));
    let py = host
        .emit_user_python("1".into(), false, "/tmp".into(), None)
        .await
        .unwrap();
    assert_eq!(py.result, Some(json!({"output":"py","exitCode":0})));
    host.shutdown().await.unwrap();
}
