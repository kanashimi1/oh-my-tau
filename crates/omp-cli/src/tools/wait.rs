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

const DESCRIPTION: &str = "Wait only when blocked with nothing else to do.\nBlocks on background jobs/services you started; returns on the first result, a message sent to you, or a steering interrupt; a safety cap returns a still-running snapshot.\nNothing you started running? Errors; NEVER wait on other agents.\nResults and messages auto-deliver. NEVER poll while work remains.";

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
        let service = service.lock().expect("service poisoned");
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
        // omp snapshot: jobs owned by the caller plus the service surface.
        if self.env.jobs.sorted().is_empty() && !has_live_service(&self.env) {
            bail!(
                "Nothing to wait for: no background job or service you started is running. \
Other agents' results and messages arrive on their own."
            );
        }
        let deadline = tokio::time::Instant::now() + WAIT_MAX;
        loop {
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
            tokio::time::sleep(POLL_INTERVAL).await;
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
            service.lock().expect("service poisoned").state,
            ServiceState::Exited | ServiceState::Failed
        )
    })
}
