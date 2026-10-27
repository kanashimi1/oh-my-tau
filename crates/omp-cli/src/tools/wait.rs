//! The `wait` tool (omp `WaitTool`): block on background jobs and services the
//! calling agent started (`tools/wait.ts`). The IRC-message legs of omp's
//! implementation have no counterpart in this port (no peer bus), so a wait
//! wakes on an owned job settling, an owned service reaching a terminal state,
//! the 30-minute safety cap, or tool-call abort.
//!
//! Like omp: work this agent did not start never sustains a wait, and with
//! nothing owned running the call errors instead of parking.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use pa_agent::BoxFut;
use serde_json::{json, Value};

use super::jobs::{JobStatus, ServiceState};
use super::Env;

const DESCRIPTION: &str = "Wait only when blocked with nothing else to do.\nBlocks on background jobs/services and running task children you started; returns on the first result or message; a safety cap returns a still-running snapshot.\nNothing you started running? Errors; NEVER wait on other agents.\nResults and messages auto-deliver. NEVER poll while work remains.";

/// omp `WAIT_MAX_MS`.
const WAIT_MAX: Duration = Duration::from_mins(30);
/// Poll cadence for job/service completion (omp races job promises; the rust
/// job registry is a plain snapshot store, so `wait` observes it on a tick).
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Parameter schema, byte-equal (as JSON) to omp's empty `waitSchema`.
#[must_use]
pub fn schema() -> Value {
    json!({"type": "object", "properties": {}})
}

/// The `wait` tool.
pub struct WaitTool {
    schema: Value,
    env: Arc<Env>,
}

/// The visible `wait` tool (omp `BUILTIN_TOOLS.wait`).
#[must_use]
pub fn tool(env: Arc<Env>) -> Arc<dyn AgentTool> {
    Arc::new(WaitTool {
        schema: schema(),
        env,
    })
}

/// One owned background item the wait watches (omp job/service legs unified).
struct Watched {
    id: String,
    finished: Option<String>,
}

fn watch_snapshot(env: &Env) -> Vec<Watched> {
    let mut out = Vec::new();
    for (id, job) in env.jobs.sorted() {
        let finished = match job.status {
            JobStatus::Running => None,
            JobStatus::Completed => job.result.clone(),
            JobStatus::Failed => job.error.clone(),
        };
        out.push(Watched { id, finished });
    }
    for (name, service) in env.jobs.services_sorted() {
        let service = service.lock();
        let finished = match service.state {
            ServiceState::Exited => Some("exited".to_string()),
            ServiceState::Failed => Some("failed".to_string()),
            _ => None,
        };
        out.push(Watched { id: name, finished });
    }
    out
}

impl AgentTool for WaitTool {
    fn name(&self) -> &'static str {
        "wait"
    }
    fn description(&self) -> &str {
        DESCRIPTION
    }
    fn parameters(&self) -> &Value {
        &self.schema
    }
    fn label(&self) -> &'static str {
        "Wait"
    }
    fn set_a2a_owner(&self, owner: &Arc<str>) {
        self.env.set_a2a_owner(owner.clone());
    }
    fn a2a_owner(&self) -> Option<Arc<str>> {
        self.env.a2a_owner()
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        _params: Value,
        signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move { self.run(signal).await })
    }
}

impl WaitTool {
    async fn run(&self, signal: AbortSignal) -> Result<AgentToolResult> {
        // Deliveries belong to this Env's session ([`super::a2a::Owner`],
        // stamped by the runtime at start); `None` (no stamped runtime, e.g.
        // bare unit tests) matches unowned deliveries.
        let owner = self.env.a2a_owner();
        // A delivery that already landed is the immediate answer — except
        // results of a still-running spawn batch: those are held so the whole
        // batch delivers in one combined result (messages/steering still
        // return immediately).
        if super::a2a::has_pending_for(&owner) {
            let drained = super::a2a::take_deliveries_for(&owner);
            let (ready, held) = super::task::split_batch_ready(drained);
            super::task::requeue_deliveries(held);
            if !ready.is_empty() {
                return Ok(AgentToolResult::text(super::task::render_delivery_batch(
                    &ready,
                )));
            }
        }
        // omp snapshot: jobs owned by the caller, the service surface, and
        // this session's running `task` children. Held batch results waiting
        // for stragglers count as something to wait for, like live children.
        let watching_children = super::task::live_children_for(&owner) > 0;
        if self.env.jobs.sorted().is_empty()
            && !has_live_service(&self.env)
            && !watching_children
            && !super::a2a::has_pending_for(&owner)
        {
            bail!(
                "Nothing to wait for: no background job, service, or task child you started is running. \
Other agents' results and messages arrive on their own."
            );
        }
        let deadline = tokio::time::Instant::now() + WAIT_MAX;
        loop {
            // Task children: a settled child (or a child message to
            // `agent://parent`) queues a combined delivery batch. Results of
            // a still-running spawn batch stay held (re-queued) so the whole
            // batch delivers together; messages/steering/questions always
            // return immediately.
            if super::a2a::has_pending_for(&owner) {
                let drained = super::a2a::take_deliveries_for(&owner);
                let (ready, held) = super::task::split_batch_ready(drained);
                super::task::requeue_deliveries(held);
                if !ready.is_empty() {
                    return Ok(AgentToolResult::text(super::task::render_delivery_batch(
                        &ready,
                    )));
                }
                // Everything drained belonged to still-running batches: fall
                // through to the deadline/abort checks and keep parking.
            }
            let snapshot = watch_snapshot(&self.env);
            // omp `undeliveredJobs`: an accepted completion not yet in the
            // transcript is exactly what this wait is for — return it now.
            if let Some(watched) = snapshot.iter().find(|w| w.finished.is_some()) {
                return Ok(Self::job_result(&self.env, &watched.id));
            }
            if tokio::time::Instant::now() >= deadline {
                return Ok(AgentToolResult::text(
                    "Wait limit reached; background work may still be running. Read proc:// for status.",
                ));
            }
            if signal.is_aborted() {
                // omp interrupt path: a steering/interrupt wake returns a
                // normal (useless-flagged upstream) result, other aborts stop.
                return Ok(AgentToolResult::text("Wait interrupted by message."));
            }
            // Park on the a2a delivery queue with the job-poll cadence as the
            // fallback tick (the job registry is a snapshot store).
            tokio::select! {
                () = super::a2a::wait_delivery_notify() => {}
                () = tokio::time::sleep(POLL_INTERVAL) => {}
            }
        }
    }

    /// Deliver one settled job's formatted result (omp `buildJobResult` subset:
    /// the bash job manager already formats the result into `Job.result`).
    fn job_result(env: &Env, id: &str) -> AgentToolResult {
        let text = env
            .jobs
            .get(id)
            .and_then(|job| job.result.or(job.error))
            .unwrap_or_else(|| format!("{id} finished."));
        AgentToolResult::text(text)
    }
}

fn has_live_service(env: &Env) -> bool {
    env.jobs.services_sorted().iter().any(|(_, service)| {
        !matches!(
            service.lock().state,
            ServiceState::Exited | ServiceState::Failed
        )
    })
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod tests {
    use super::*;
    use crate::tools::task;

    /// Fake `--mode rpc` child that answers immediately: the settle must
    /// queue the result delivery the `wait` tool returns.
    const FAKE_WAIT_CHILD: &str = r#"#!/usr/bin/env python3
import json, sys
stdin = sys.stdin
stdin.readline()
print(json.dumps({"type": "response", "command": "prompt", "success": True, "id": "omp-task-prompt"}), flush=True)
print(json.dumps({"type": "prompt_result", "status": "completed", "sessionSettled": True}), flush=True)
for _ in range(2):
    q = json.loads(stdin.readline())
    if q["type"] == "get_last_assistant_text":
        print(json.dumps({"type": "response", "command": "get_last_assistant_text", "success": True, "data": {"text": "wait-child-answer"}}), flush=True)
    else:
        print(json.dumps({"type": "response", "command": "get_session_stats", "success": True, "data": {"tokens": {"output": 1}}}), flush=True)
stdin.readline()
"#;

    #[tokio::test]
    async fn wait_blocks_on_task_children_and_returns_the_first_delivery() {
        let _serial = super::super::a2a::tests::LOCK.lock().await;
        super::super::a2a::tests::without_parent_bus(async {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let bin = dir.path().join("fake-wait-child");
            std::fs::write(&bin, FAKE_WAIT_CHILD).unwrap();
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
            // A private owner for this test's Env: its children and
            // deliveries are invisible to a concurrent test runtime's
            // park/wake legs, and vice versa.
            let env = std::sync::Arc::new(Env::new(dir.path(), false));
            env.set_a2a_owner(crate::persist::uuid_v7().into());
            let owner = env.a2a_owner();
            let _ = super::super::a2a::take_deliveries_for(&owner);
            let tool = WaitTool {
                schema: schema(),
                env: env.clone(),
            };
            // Nothing owned running: errors, like jobs/services alone.
            let err = tool
                .run(pa_agent::abort::AbortSignal::never())
                .await
                .unwrap_err();
            assert!(err.to_string().contains("Nothing to wait for"), "{err}");

            // A `task` child: `wait` parks and returns its result delivery.
            let spawned = task::spawn_children(
                &omp_task::TaskRunner::new(&bin, 4),
                vec![omp_task::runner::TaskSpec {
                    id: Some("wait-kid".to_string()),
                    agent: "task".into(),
                    prompt: "go".into(),
                    cwd: dir.path().to_path_buf(),
                    output_schema: None,
                    schema_mode: omp_task::SchemaMode::Permissive,
                    model: None,
                    tools: None,
                    depth: 1,
                    extra_args: Vec::new(),
                    env: Vec::new(),
                    worktree: false,
                }],
                vec![None],
                vec!["go".to_string()],
                &owner,
            )
            .unwrap();
            assert_eq!(spawned.len(), 1);
            let result = tool
                .run(pa_agent::abort::AbortSignal::never())
                .await
                .unwrap();
            let text = result
                .content
                .iter()
                .find_map(|part| match part {
                    pa_agent::types::ToolResultContent::Text(t) => Some(t.text.clone()),
                    pa_agent::types::ToolResultContent::Image(_) => None,
                })
                .expect("text content");
            assert!(text.starts_with("<task-results>"), "{text}");
            assert!(
                text.contains("wait-child-answer"),
                "the child's answer rides the delivery: {text}"
            );
            assert_eq!(task::live_children(), 0);
        })
        .await;
    }

    /// Fake `--mode rpc` child that echoes its prompt in the answer and
    /// stalls when the prompt says "slow": two of these in one spawned batch
    /// settle at clearly different times.
    const FAKE_BATCH_CHILD: &str = r#"#!/usr/bin/env python3
import json, sys, time
stdin = sys.stdin
frame = json.loads(stdin.readline())
message = frame.get("message", "")
if "slow" in message:
    time.sleep(0.8)
print(json.dumps({"type": "response", "command": "prompt", "success": True, "id": "omp-task-prompt"}), flush=True)
print(json.dumps({"type": "prompt_result", "status": "completed", "sessionSettled": True}), flush=True)
for _ in range(2):
    q = json.loads(stdin.readline())
    if q["type"] == "get_last_assistant_text":
        print(json.dumps({"type": "response", "command": "get_last_assistant_text", "success": True, "data": {"text": "batch-answer " + message}}), flush=True)
    else:
        print(json.dumps({"type": "response", "command": "get_session_stats", "success": True, "data": {"tokens": {"output": 1}}}), flush=True)
stdin.readline()
"#;

    fn batch_spec(id: &str, prompt: &str, dir: &std::path::Path) -> omp_task::runner::TaskSpec {
        omp_task::runner::TaskSpec {
            id: Some(id.to_string()),
            agent: "task".into(),
            prompt: prompt.into(),
            cwd: dir.to_path_buf(),
            output_schema: None,
            schema_mode: omp_task::SchemaMode::Permissive,
            model: None,
            tools: None,
            depth: 1,
            extra_args: Vec::new(),
            env: Vec::new(),
            worktree: false,
        }
    }

    fn spawn_batch(
        bin: &std::path::Path,
        dir: &std::path::Path,
        items: &[(&str, &str)],
        owner: &super::super::a2a::Owner,
    ) -> Vec<task::SpawnedChild> {
        let specs: Vec<_> = items
            .iter()
            .map(|(id, prompt)| batch_spec(id, prompt, dir))
            .collect();
        let prompts: Vec<String> = specs.iter().map(|spec| spec.prompt.clone()).collect();
        task::spawn_children(
            &omp_task::TaskRunner::new(bin, 4),
            specs,
            vec![None; items.len()],
            prompts,
            owner,
        )
        .unwrap()
    }

    async fn wait_once(env: std::sync::Arc<Env>) -> String {
        let tool = WaitTool {
            schema: schema(),
            env,
        };
        let result = tool
            .run(pa_agent::abort::AbortSignal::never())
            .await
            .unwrap();
        result
            .content
            .iter()
            .find_map(|part| match part {
                pa_agent::types::ToolResultContent::Text(t) => Some(t.text.clone()),
                pa_agent::types::ToolResultContent::Image(_) => None,
            })
            .expect("text content")
    }

    /// Two children of ONE spawned batch settle ~0.8s apart; a single `wait`
    /// must hold the fast child's result and return both in one combined
    /// delivery (one parent turn, not one per child).
    #[tokio::test]
    async fn one_wait_delivers_a_whole_spawned_batch_together() {
        let _serial = super::super::a2a::tests::LOCK.lock().await;
        super::super::a2a::tests::without_parent_bus(async {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let bin = dir.path().join("fake-batch-child");
            std::fs::write(&bin, FAKE_BATCH_CHILD).unwrap();
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
            let _ = super::super::a2a::take_deliveries();
            let env = std::sync::Arc::new(Env::new(dir.path(), false));
            // A private owner for this Env, so a concurrent test runtime
            // neither consumes this batch nor keeps this wait pending.
            env.set_a2a_owner(crate::persist::uuid_v7().into());
            let owner = env.a2a_owner();

            let spawned = spawn_batch(
                &bin,
                dir.path(),
                &[("batch-fast", "fast work"), ("batch-slow", "slow work")],
                &owner,
            );
            assert_eq!(spawned.len(), 2);

            // One wait, no matter when the fast child settled first.
            let text = wait_once(env).await;
            assert!(text.starts_with("<task-results>"), "{text}");
            assert!(
                text.contains("batch-fast") && text.contains("batch-slow"),
                "both children of the one batch in one delivery: {text}"
            );
            assert!(
                text.contains("batch-answer fast work") && text.contains("batch-answer slow work"),
                "both answers ride the same delivery: {text}"
            );
            // Exactly the two envelopes: the fast child's result was held,
            // not delivered early in a separate wait.
            assert_eq!(task::live_children(), 0);
            assert!(
                !super::super::a2a::has_pending_for(&owner),
                "nothing left held after the batch delivered"
            );
        })
        .await;
    }

    /// The automatic parent-turn delivery path uses the same batch join as
    /// the explicit `wait` tool; it must not leak the fast child early.
    #[tokio::test]
    async fn auto_delivery_joins_the_whole_spawned_batch() {
        let _serial = super::super::a2a::tests::LOCK.lock().await;
        super::super::a2a::tests::without_parent_bus(async {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let bin = dir.path().join("fake-batch-child");
            std::fs::write(&bin, FAKE_BATCH_CHILD).unwrap();
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
            let _ = super::super::a2a::take_deliveries();
            // A private owner for this test's bus traffic.
            let owner: super::super::a2a::Owner = Some(crate::persist::uuid_v7().into());
            spawn_batch(
                &bin,
                dir.path(),
                &[("auto-fast", "fast work"), ("auto-slow", "slow work")],
                &owner,
            );

            let deliveries = tokio::time::timeout(
                std::time::Duration::from_secs(4),
                task::next_delivery_batch(&owner),
            )
            .await
            .expect("joined auto delivery");
            assert_eq!(deliveries.len(), 2, "{deliveries:?}");
            let text = task::render_delivery_batch(&deliveries);
            assert!(
                text.contains("auto-fast") && text.contains("auto-slow"),
                "{text}"
            );
            assert!(!super::super::a2a::has_pending_for(&owner));
            assert_eq!(task::live_children(), 0);
        })
        .await;
    }

    /// A child spawned in a LATER call is an independent batch: it delivers
    /// on its own wait and is not held by, nor merged with, the earlier
    /// batch's deliveries.
    #[tokio::test]
    async fn independent_later_child_is_not_joined_into_an_earlier_batch() {
        let _serial = super::super::a2a::tests::LOCK.lock().await;
        super::super::a2a::tests::without_parent_bus(async {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let bin = dir.path().join("fake-batch-child");
            std::fs::write(&bin, FAKE_BATCH_CHILD).unwrap();
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
            let _ = super::super::a2a::take_deliveries();
            let env = std::sync::Arc::new(Env::new(dir.path(), false));
            // A private owner for this Env, so a concurrent test runtime
            // neither consumes this test's deliveries nor feeds it.
            env.set_a2a_owner(crate::persist::uuid_v7().into());
            let owner = env.a2a_owner();

            // First batch: two children, one combined delivery.
            spawn_batch(
                &bin,
                dir.path(),
                &[("batch-fast", "fast work"), ("batch-slow", "slow work")],
                &owner,
            );
            let first = wait_once(env.clone()).await;
            assert!(
                first.contains("batch-fast") && first.contains("batch-slow"),
                "{first}"
            );

            // Independent later child: one wait, one delivery, only its own
            // result (the earlier batch must not be re-held or re-delivered).
            spawn_batch(
                &bin,
                dir.path(),
                &[("later-child", "slow later work")],
                &owner,
            );
            let second = wait_once(env).await;
            assert!(second.starts_with("<task-results>"), "{second}");
            assert!(
                second.contains("later-child") && second.contains("batch-answer slow later work"),
                "{second}"
            );
            assert!(
                !second.contains("batch-fast") && !second.contains("batch-slow"),
                "no residue of the earlier batch: {second}"
            );
            assert!(
                !super::super::a2a::has_pending_for(&owner),
                "nothing held: {second}"
            );
            assert_eq!(task::live_children(), 0);
        })
        .await;
    }
}
