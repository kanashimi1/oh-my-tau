//! Integration tests for the controllable-runner review findings
//! (ReviewEvalAgents 1-4 + the process-group security finding) with fake
//! `--mode rpc` / `-p` shell children.

use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use omp_task::{ChildMode, RunPhase, RunStatus, TaskRunner, TaskSpec};

/// The Finishing-deadline knob is process-global env; every RPC test in this
/// file takes the lock and sets an explicit value so they cannot interfere.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn set_finishing_deadline(ms: &str) {
    std::env::set_var("OMP_TASK_TEST_FINISHING_DEADLINE_MS", ms);
}

fn fake_binary(dir: &Path, body: &str) -> PathBuf {
    let path = dir.join("fake-omp-rs");
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "#!/bin/sh\n{body}").unwrap();
    drop(f);
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path
}

fn spec(prompt: &str, cwd: &Path) -> TaskSpec {
    TaskSpec {
        id: Some("it-1".to_string()),
        agent: "task".to_string(),
        prompt: prompt.to_string(),
        cwd: cwd.to_path_buf(),
        output_schema: None,
        schema_mode: omp_task::SchemaMode::Permissive,
        model: None,
        tools: None,
        depth: 0,
        extra_args: Vec::new(),
    }
}

async fn wait_phase(control: &omp_task::RunControl, phase: RunPhase) {
    for _ in 0..500 {
        if control.phase() == phase {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("run never reached phase {phase:?}");
}

/// Finding 1 (cancel vs completion race): a cancel that wins the killable
/// transition *after* the drive loop collected the child's completed output
/// must still settle the run Cancelled (receipt and outcome consistent); a
/// run that settles first makes a late `cancel()` return `false`.
#[tokio::test]
async fn cancel_in_the_post_loop_window_is_consistent() {
    let _env = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_finishing_deadline("15000");
    let dir = tempfile::tempdir().unwrap();
    // Answers everything and exits, but a detached grandchild holds stderr
    // open ~4s: the drive loop spends that window in the stderr join, which
    // is exactly the pre-fix race window where `cancel()` returned true for
    // a run that then settled Completed.
    let bin = fake_binary(
        dir.path(),
        r#"read -r prompt
echo '{"type":"response","command":"prompt","success":true,"id":"omp-task-prompt"}'
echo '{"type":"prompt_result","status":"completed","sessionSettled":true}'
read -r q1
read -r q2
echo '{"type":"response","command":"get_last_assistant_text","success":true,"data":{"text":"answer"}}'
echo '{"type":"response","command":"get_session_stats","success":true,"data":{"tokens":{"output":5}}}'
setsid sleep 4 >&2 &
exit 0"#,
    );
    let runner = TaskRunner::new(&bin, 1);
    let (control, join) = runner.start(spec("go", dir.path()), ChildMode::Rpc);
    // Under heavy CPU load the run can pass through Finishing between two
    // polls; the invariant below (receipt matches outcome) holds either way.
    for _ in 0..500 {
        if matches!(control.phase(), RunPhase::Finishing | RunPhase::Settled) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    let cancelled = control.cancel();
    let out = tokio::time::timeout(Duration::from_secs(20), join)
        .await
        .expect("run must settle even with a detached stderr holder");
    if cancelled {
        assert_eq!(
            out.status,
            RunStatus::Cancelled,
            "cancel() won the killable transition: the run must settle Cancelled, got {:?}",
            out
        );
    } else {
        assert_eq!(
            out.status,
            RunStatus::Completed,
            "the run finished first: it must settle Completed, got {:?}",
            out
        );
    }
    assert_eq!(runner.available_permits(), 1, "permit released");
    assert!(!control.cancel(), "cancel after settle is a no-op");
}

/// The other side of finding 1: no cancel at all — the run settles
/// Completed even though a detached grandchild holds stderr (bounded join).
#[tokio::test]
async fn completed_run_with_detached_stderr_holder_settles() {
    let _env = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_finishing_deadline("15000");
    let dir = tempfile::tempdir().unwrap();
    let bin = fake_binary(
        dir.path(),
        r#"read -r prompt
echo '{"type":"response","command":"prompt","success":true,"id":"omp-task-prompt"}'
echo '{"type":"prompt_result","status":"completed","sessionSettled":true}'
read -r q1
read -r q2
echo '{"type":"response","command":"get_last_assistant_text","success":true,"data":{"text":"answer"}}'
echo '{"type":"response","command":"get_session_stats","success":true,"data":{"tokens":{"output":5}}}'
setsid sleep 4 >&2 &
exit 0"#,
    );
    let runner = TaskRunner::new(&bin, 1);
    let (control, join) = runner.start(spec("go", dir.path()), ChildMode::Rpc);
    let started = Instant::now();
    let out = tokio::time::timeout(Duration::from_secs(15), join)
        .await
        .expect("bounded join: never blocks on the detached holder");
    assert_eq!(out.status, RunStatus::Completed, "{out:?}");
    assert_eq!(out.output, "answer");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "pipe joins are bounded, took {:?}",
        started.elapsed()
    );
    assert!(!control.cancel());
}

/// Finding 2 (Finishing-phase deadline): a child whose prompt settles but
/// that never answers the final-text/stats requests must be killed and the
/// run must settle failed — no hang, permit released.
#[tokio::test]
async fn finishing_phase_deadline_kills_hanging_child() {
    let _env = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_finishing_deadline("400");
    let dir = tempfile::tempdir().unwrap();
    let bin = fake_binary(
        dir.path(),
        r#"read -r prompt
echo '{"type":"response","command":"prompt","success":true,"id":"omp-task-prompt"}'
echo '{"type":"prompt_result","status":"completed","sessionSettled":true}'
sleep 30"#,
    );
    let runner = TaskRunner::new(&bin, 1);
    let (control, join) = runner.start(spec("go", dir.path()), ChildMode::Rpc);
    let started = Instant::now();
    let out = tokio::time::timeout(Duration::from_secs(15), join)
        .await
        .expect("run must not hang when the stats reply never comes");
    assert_eq!(out.status, RunStatus::Failed, "{out:?}");
    assert!(
        out.stderr.contains("timed out in the Finishing phase"),
        "stderr: {}",
        out.stderr
    );
    assert!(started.elapsed() < Duration::from_secs(10));
    assert_eq!(runner.available_permits(), 1, "permit released");
    assert_eq!(control.phase(), RunPhase::Settled);
}

/// Finding 4 (steer receipt): a steer sent while the prompt is still running
/// is acknowledged once written; once the prompt settled (Finishing), the
/// receipt reports failure instead of a silently dropped message.
#[tokio::test]
async fn steer_receipt_reports_finishing_failure() {
    let _env = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_finishing_deadline("15000");
    let dir = tempfile::tempdir().unwrap();
    let bin = fake_binary(
        dir.path(),
        r#"read -r prompt
echo '{"type":"response","command":"prompt","success":true,"id":"omp-task-prompt"}'
read -r first
echo '{"type":"prompt_result","status":"completed","sessionSettled":true}'
sleep 1
read -r q1
read -r q2
echo '{"type":"response","command":"get_last_assistant_text","success":true,"data":{"text":"done"}}'
echo '{"type":"response","command":"get_session_stats","success":true,"data":{"tokens":{"output":3}}}'
read -r eof || exit 0"#,
    );
    let runner = TaskRunner::new(&bin, 1);
    let (control, join) = runner.start(spec("go", dir.path()), ChildMode::Rpc);
    wait_phase(&control, RunPhase::Running).await;
    control
        .steer("first")
        .await
        .expect("running steer is delivered");
    wait_phase(&control, RunPhase::Finishing).await;
    let receipt = control.steer("second").await;
    assert_eq!(
        receipt.unwrap_err().to_string(),
        "the child is not running",
        "a steer that races the Finishing transition must fail explicitly"
    );
    let out = join.await;
    assert_eq!(out.status, RunStatus::Completed, "{out:?}");
    assert_eq!(out.output, "done");
}

/// Finding 3 (process groups): after a *normal* exit, surviving group
/// members are reaped and can neither linger nor hold the pipes open.
#[tokio::test]
async fn normal_exit_reaps_group_members_and_bounded_pipes() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("survived");
    let bin = fake_binary(
        dir.path(),
        &format!(
            "(sleep 2; touch {}) &\n( exec sleep 30 ) >&1 &\nexit 0",
            marker.display()
        ),
    );
    let runner = TaskRunner::new(&bin, 1);
    let (_control, join) = runner.start(spec("go", dir.path()), ChildMode::Print);
    let started = Instant::now();
    let out = tokio::time::timeout(Duration::from_secs(15), join)
        .await
        .expect("pipe held by a group member must not block the settle");
    assert_eq!(out.status, RunStatus::Completed, "{out:?}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "settle waited on the pipe-holding survivor ({:?})",
        started.elapsed()
    );
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert!(!marker.exists(), "group member survived a normal exit");
}

/// Recursion-depth stamp and policy passthrough reach the child argv/env.
#[tokio::test]
async fn child_gets_depth_env_and_extra_args() {
    let dir = tempfile::tempdir().unwrap();
    let bin = fake_binary(dir.path(), r#"echo "depth=$PI_TASK_DEPTH args=$*""#);
    let runner = TaskRunner::new(&bin, 1);
    let mut spec = spec("hello", dir.path());
    spec.depth = 3;
    spec.extra_args = vec![
        "--no-extensions".to_string(),
        "--extension".to_string(),
        "/tmp/x".to_string(),
    ];
    let (_control, join) = runner.start(spec, ChildMode::Print);
    let out = join.await;
    assert_eq!(out.status, RunStatus::Completed, "{out:?}");
    assert!(out.output.contains("depth=3"), "got {:?}", out.output);
    assert!(
        out.output.contains("--no-extensions --extension /tmp/x"),
        "got {:?}",
        out.output
    );
}
