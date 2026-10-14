//! Unit tests for the eval tool loop: the pre-port parity tests plus the
//! new loop behavior (default timeout, cancel, is_error, per-language reset,
//! output cap, kernel-loss notice, bridge-backed helpers).

use super::*;

/// Unique cwd per test: parallel tests must not share a kernel slot (a
/// JS timeout kill is global to the slot's kernel).
fn test_env(tag: &str) -> Arc<Env> {
    let dir = std::env::temp_dir().join(format!("omp-rs-eval-test-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Arc::new(Env::new(&dir, false))
}

fn ctx() -> CallCtx {
    CallCtx { tool_call_id: "t1".into(), signal: pa_agent::abort::AbortSignal::never(), on_update: None }
}

async fn run(env: &Arc<Env>, args: Value) -> AgentToolResult {
    eval(env.clone(), args, ctx()).await.unwrap()
}

fn text(r: &AgentToolResult) -> String {
    let pa_agent::types::ToolResultContent::Text(content) = &r.content[0] else {
        panic!("unexpected content");
    };
    content.text.clone()
}

#[test]
fn schema_matches_ts_reference() {
    let s = schema();
    let props = s["properties"].as_object().unwrap();
    assert_eq!(props["language"]["enum"], json!(["py", "js"]));
    assert_eq!(s["required"], json!(["language", "code"]));
    assert_eq!(props.len(), 5);
}

#[test]
fn timeout_defaults_to_30s_zero_disables_and_clamps() {
    assert_eq!(resolve_timeout(&json!({})), Some(Duration::from_secs(30)));
    assert_eq!(resolve_timeout(&json!({"timeout": 0})), None);
    assert_eq!(resolve_timeout(&json!({"timeout": 0.2})), Some(Duration::from_secs(1)));
    assert_eq!(resolve_timeout(&json!({"timeout": 1e9})), Some(Duration::from_secs(3600)));
}

#[test]
fn small_output_is_not_capped() {
    assert_eq!(cap_output("hello\n"), "hello\n");
}

#[tokio::test]
async fn missing_language_is_rejected_like_ts_validation() {
    let env = test_env("lang");
    let err = eval(env.clone(), json!({"code": "1"}), ctx()).await.unwrap_err();
    assert!(err.to_string().contains("language"), "got: {err}");
}

#[tokio::test]
async fn python_state_persists_and_errors_are_flagged() {
    let env = test_env("py");
    run(&env, json!({"language": "py", "code": "x = 40 + 2"})).await;
    let r = run(&env, json!({"language": "py", "code": "x"})).await;
    assert!(text(&r).contains("42"), "got: {}", text(&r));
    assert_eq!(r.is_error, None);

    let r = run(&env, json!({"language": "py", "code": "raise ValueError('bad')"})).await;
    assert_eq!(r.is_error, Some(true), "raised cell must be is_error");
    assert!(text(&r).contains("ValueError: bad"), "got: {}", text(&r));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn python_failed_cell_rolls_back_new_names() {
    let env = test_env("txn");
    let r = run(&env, json!({"language": "py", "code": "leaked = 1\nraise ValueError('boom')"})).await;
    assert_eq!(r.is_error, Some(true));
    assert!(text(&r).contains("[omp-txn] rolled back: leaked"), "got: {}", text(&r));
    let r = run(&env, json!({"language": "py", "code": "'leaked' in globals()"})).await;
    assert!(text(&r).contains("False"), "name must be rolled back, got: {}", text(&r));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn reset_wipes_only_the_requested_language() {
    let env = test_env("reset");
    run(&env, json!({"language": "py", "code": "p = 1"})).await;
    run(&env, json!({"language": "js", "code": "globalThis.j = 2"})).await;
    // Python reset: the fresh kernel has no `p` (and reports the state loss) ...
    let r = run(&env, json!({"language": "py", "code": "'p' in globals()", "reset": true})).await;
    assert!(text(&r).contains("False"), "got: {}", text(&r));
    // ... while the JS kernel keeps its state.
    let r = run(&env, json!({"language": "js", "code": "j"})).await;
    assert!(text(&r).contains('2'), "js state survives a py reset: {}", text(&r));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn fresh_kernel_generation_gets_a_kernel_loss_notice() {
    let env = test_env("notice");
    let first = run(&env, json!({"language": "py", "code": "1"})).await;
    assert!(!text(&first).contains("Kernel state was lost"), "first kernel: {}", text(&first));
    let second = run(&env, json!({"language": "py", "code": "2", "reset": true})).await;
    assert!(text(&second).contains("Kernel state was lost"), "second generation: {}", text(&second));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn abort_cancels_a_running_cell() {
    let env = test_env("abort");
    let controller = pa_agent::abort::AbortController::new();
    let signal = controller.signal();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(500)).await;
        controller.abort();
    });
    let started = std::time::Instant::now();
    let r = eval(
        env.clone(),
        json!({"language": "py", "code": "import time\ntime.sleep(20)", "timeout": 0}),
        CallCtx { tool_call_id: "t2".into(), signal, on_update: None },
    )
    .await
    .unwrap();
    assert!(started.elapsed() < Duration::from_secs(10), "cancel must not wait for the cell");
    assert_eq!(r.is_error, Some(true));
    // The Python kernel survives an interrupt.
    let r = run(&env, json!({"language": "py", "code": "'alive'"})).await;
    assert!(text(&r).contains("alive"), "got: {}", text(&r));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn python_prelude_tool_bridge_reaches_session_tools() {
    let env = test_env("bridge");
    std::fs::write(env.cwd.join("note.txt"), "bridged\n").unwrap();
    // `tool.read` goes cell → host frame → EvalBridge → the read tool.
    let r = run(
        &env,
        json!({"language": "py", "code": "r = await tool.read({'path': 'note.txt'})\nprint(r)"}),
    )
    .await;
    assert!(text(&r).contains("bridged"), "got: {}", text(&r));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn js_state_persists_and_top_level_await_works() {
    let env = test_env("js-state");
    run(&env, json!({"language": "js", "code": "let acc = await Promise.resolve(40)"})).await;
    let r = run(&env, json!({"language": "js", "code": "acc + 2"})).await;
    assert!(text(&r).contains("42"), "got: {}", text(&r));
    run(&env, json!({"language": "js", "code": "acc", "reset": true})).await;
    let r = run(&env, json!({"language": "js", "code": "typeof acc"})).await;
    assert!(text(&r).contains("\"undefined\""), "state wiped: {}", text(&r));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn js_error_and_timeout_annotations_match_omp() {
    let env = test_env("js-timeout");
    let r = run(&env, json!({"language": "js", "code": "throw new Error('boom')"})).await;
    assert!(text(&r).contains("Error: boom"), "got: {}", text(&r));
    assert_eq!(r.is_error, Some(true));

    let r = run(&env, json!({"language": "js", "code": "while (true) {}", "timeout": 1})).await;
    assert!(
        text(&r).contains("The JS worker was force-killed and its VM state was reset"),
        "got: {}",
        text(&r)
    );
    let r = run(&env, json!({"language": "js", "code": "'alive'"})).await;
    assert!(text(&r).contains("alive"), "got: {}", text(&r));
    shutdown_slot(&env.cwd);
}
