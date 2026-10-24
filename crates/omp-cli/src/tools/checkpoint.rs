//! `checkpoint` / `rewind`: omp's context-checkpoint pair, ported from
//! pi-coding-agent's `src/tools/checkpoint.ts`.
//!
//! `checkpoint` marks the current conversation position before exploratory
//! work; `rewind` ends the checkpoint, replacing the intermediate exploration
//! with the caller's report. The tools only validate and report — the session
//! runtime owns the actual state transitions (message counting at checkpoint,
//! context truncation and the persisted `rewind-report` entry on rewind) via
//! the shared [`Shared`] registry, mirroring TS `AgentSession`'s private
//! `#checkpointState` / `#lastCompletedRewind` fields.

use std::sync::{Arc, LazyLock};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback, ToolResultContent};
use pa_agent::BoxFut;
use parking_lot::Mutex;
use serde_json::{json, Value};

// ─── Descriptions (byte-equal to omp's rendered checkpoint.md / rewind.md) ─

/// omp `checkpointDescription` (`prompts/tools/checkpoint.md`, rendered).
#[allow(clippy::needless_raw_string_hashes)] // r#"…"# is matched literally by tests-tools/compare.ts
pub const CHECKPOINT_DESCRIPTION: &str = r#"Context checkpoint: before exploratory work; later `rewind`, retaining only concise report.

Use for investigations with many intermediate tool calls (`read`/`grep`/`glob`/`lsp`/etc.) to minimize subsequent context cost.

Rules:
- MUST `rewind` before yielding after starting a checkpoint.
- NEVER `checkpoint` while another checkpoint active.
- Subagents: disabled by default. Enable: agent-definition `tools:` frontmatter lists `checkpoint` or `rewind`; sister tool auto-included; requires `checkpoint.enabled` setting.

Typical flow:
1. `checkpoint(goal: …)`
2. Exploratory work
3. `rewind(report: …)` with concise findings

After `rewind`: intermediate checkpoint messages removed from active context; replaced by report."#;

/// omp `rewindDescription` (`prompts/tools/rewind.md`, rendered).
#[allow(clippy::needless_raw_string_hashes)] // r#"…"# is matched literally by tests-tools/compare.ts
pub const REWIND_DESCRIPTION: &str = r#"End the active checkpoint; rewind context to it, replacing intermediate exploration with your report."#;

// ─── Parameter schemas (byte-equal, as JSON, to omp's ArkType schemas) ─────

/// omp `checkpointSchema`.
#[must_use]
pub fn schema_checkpoint() -> Value {
    json!({"type":"object","properties":{"goal":{"type":"string","description":"investigation goal"}},"required":["goal"]})
}

/// omp `rewindSchema`.
#[must_use]
pub fn schema_rewind() -> Value {
    json!({"type":"object","properties":{"report":{"type":"string","description":"investigation findings"}},"required":["report"]})
}

// ─── Shared session state (TS: AgentSession `#checkpointState` / `#lastCompletedRewind`) ──

/// Active checkpoint, set by the runtime when a `checkpoint` tool result is
/// appended (AFTER the result is in the message list, per TS).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointState {
    /// Number of in-memory messages at checkpoint (AFTER checkpoint tool result is appended).
    pub checkpoint_message_count: usize,
    /// Session entry ID at checkpoint (for session tree branching); backfilled
    /// by the runtime once the toolResult entry is persisted.
    pub checkpoint_entry_id: Option<String>,
    /// Timestamp (`Date.toISOString` shape, `…T…Z`).
    pub started_at: String,
}

/// Report retained after a successful rewind (TS `CompletedRewindState`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletedRewindState {
    /// Report retained after a successful rewind.
    pub report: String,
    /// Timestamp for the checkpoint that was rewound.
    pub started_at: String,
    /// Timestamp when the rewind completed.
    pub rewound_at: String,
}

#[derive(Debug, Default)]
struct SharedInner {
    checkpoint: Option<CheckpointState>,
    last_completed_rewind: Option<CompletedRewindState>,
}

/// Process-wide checkpoint/rewind state. omp-rs runs one agent session per
/// process, so a global matches the TS per-session fields in practice; the
/// runtime (message counting, truncation) and both tools share it.
pub struct Shared {
    inner: Mutex<SharedInner>,
}

static SHARED: LazyLock<Shared> = LazyLock::new(|| Shared {
    inner: Mutex::new(SharedInner::default()),
});

/// The shared registry used by the tools and the agent runtime.
pub fn shared() -> &'static Shared {
    &SHARED
}

impl Shared {
    /// Active checkpoint state, if any (TS `getCheckpointState`).
    pub fn checkpoint_state(&self) -> Option<CheckpointState> {
        self.inner.lock().checkpoint.clone()
    }

    /// Install a new checkpoint; clears any completed-rewind recovery state,
    /// as TS `setCheckpointState` does when a state is provided.
    pub fn set_checkpoint_state(&self, state: CheckpointState) {
        let mut inner = self.inner.lock();
        inner.checkpoint = Some(state);
        inner.last_completed_rewind = None;
    }

    /// Clear the active checkpoint (TS `setCheckpointState(undefined)`); used
    /// by the runtime rewind hook (`Runtime::maybe_apply_rewind`).
    pub fn clear_checkpoint_state(&self) {
        self.inner.lock().checkpoint = None;
    }

    /// Most recent completed rewind, if this session just rewound (TS
    /// `getLastCompletedRewind`).
    pub fn last_completed_rewind(&self) -> Option<CompletedRewindState> {
        self.inner.lock().last_completed_rewind.clone()
    }

    /// Record a completed rewind.
    pub fn set_last_completed_rewind(&self, state: CompletedRewindState) {
        self.inner.lock().last_completed_rewind = Some(state);
    }
}

// ─── now_iso: `Date.toISOString()` shape (`YYYY-MM-DDTHH:MM:SS.sssZ`) ──────

#[allow(clippy::cast_possible_truncation)] // millis since 1970 fits i64 for the next 290M years
fn now_iso() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64);
    let (year, month, day, hour, minute, second, ms) = civil_from_millis(millis);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{ms:03}Z")
}

/// Days→civil date conversion (Howard Hinnant's algorithm), pure std.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // every cast follows div_euclid/rem_euclid, so the value is inside the target range
fn civil_from_millis(millis: i64) -> (i64, u32, u32, u32, u32, u32, u32) {
    let seconds = millis.div_euclid(1000);
    let ms = (millis.rem_euclid(1000)) as u32;
    let days = seconds.div_euclid(86_400);
    let secs_of_day = seconds.rem_euclid(86_400);
    let hour = (secs_of_day / 3600) as u32;
    let minute = ((secs_of_day % 3600) / 60) as u32;
    let second = (secs_of_day % 60) as u32;
    // 1970-01-01 is day 0; shift the epoch to 0000-03-01 for the algorithm.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let year = if month <= 2 { y + 1 } else { y };
    (year, month, day, hour, minute, second, ms)
}

// ─── Tools ─────────────────────────────────────────────────────────────────

struct CheckpointTool {
    state: &'static Shared,
    schema: Value,
}

struct RewindTool {
    state: &'static Shared,
    schema: Value,
}

fn result_with_text(text: &str, details: Value) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContent::text(text)],
        details,
        terminate: None,
        is_error: None,
    }
}

impl AgentTool for CheckpointTool {
    fn name(&self) -> &'static str {
        "checkpoint"
    }
    fn description(&self) -> &str {
        CHECKPOINT_DESCRIPTION
    }
    fn parameters(&self) -> &Value {
        &self.schema
    }
    fn label(&self) -> &'static str {
        "Checkpoint"
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        params: Value,
        _signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move {
            if self.state.checkpoint_state().is_some() {
                bail!("Checkpoint already active.");
            }
            let goal = params
                .get("goal")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let started_at = now_iso();
            Ok(result_with_text(
                &format!("Checkpoint: {goal}\nFinish exploration and formulate findings."),
                json!({"goal": goal, "startedAt": started_at}),
            ))
        })
    }
}

impl AgentTool for RewindTool {
    fn name(&self) -> &'static str {
        "rewind"
    }
    fn description(&self) -> &str {
        REWIND_DESCRIPTION
    }
    fn parameters(&self) -> &Value {
        &self.schema
    }
    fn label(&self) -> &'static str {
        "Rewind"
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        params: Value,
        _signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move {
            if self.state.checkpoint_state().is_none() {
                if self.state.last_completed_rewind().is_some() {
                    bail!("Checkpoint already completed; continue from the retained rewind report instead of calling rewind again.");
                }
                bail!("No active checkpoint. Create a checkpoint before calling rewind.");
            }
            let report = params
                .get("report")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string();
            if report.is_empty() {
                bail!("Report cannot be empty.");
            }
            Ok(result_with_text(
                "Rewind requested.\nReport captured for context replacement.",
                json!({"report": report, "rewound": true}),
            ))
        })
    }
}

/// Registered `checkpoint` tool, sharing the process-wide [`Shared`] state.
#[must_use]
pub fn checkpoint_tool() -> Arc<dyn AgentTool> {
    Arc::new(CheckpointTool {
        state: shared(),
        schema: schema_checkpoint(),
    })
}

/// Registered `rewind` tool, sharing the process-wide [`Shared`] state.
#[must_use]
pub fn rewind_tool() -> Arc<dyn AgentTool> {
    Arc::new(RewindTool {
        state: shared(),
        schema: schema_rewind(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(state: &'static Shared) -> Arc<CheckpointTool> {
        Arc::new(CheckpointTool {
            state,
            schema: schema_checkpoint(),
        })
    }

    fn rewind(state: &'static Shared) -> Arc<RewindTool> {
        Arc::new(RewindTool {
            state,
            schema: schema_rewind(),
        })
    }

    fn text_of(result: &AgentToolResult) -> &str {
        match &result.content[0] {
            ToolResultContent::Text(t) => &t.text,
            ToolResultContent::Image(_) => "",
        }
    }

    fn run(tool: Arc<dyn AgentTool>, params: Value) -> Result<AgentToolResult> {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(tool.execute(
                "call-1".into(),
                params,
                AbortSignal::never(),
                Arc::new(|_| {}),
            ))
    }

    /// Fresh state per test: build standalone tools over a leaked Shared.
    fn fresh() -> &'static Shared {
        Box::leak(Box::new(Shared {
            inner: Mutex::new(SharedInner::default()),
        }))
    }

    #[test]
    fn checkpoint_result_text_and_details() {
        let state = fresh();
        let result = run(checkpoint(state), json!({"goal": "trace flush path"})).unwrap();
        assert_eq!(
            text_of(&result),
            "Checkpoint: trace flush path\nFinish exploration and formulate findings."
        );
        let details = &result.details;
        assert_eq!(details["goal"], "trace flush path");
        let started = details["startedAt"].as_str().unwrap();
        assert!(
            started.ends_with('Z') && started.contains('T'),
            "iso ts: {started}"
        );
    }

    #[test]
    fn checkpoint_rejects_second_active_checkpoint() {
        let state = fresh();
        state.set_checkpoint_state(CheckpointState {
            checkpoint_message_count: 3,
            checkpoint_entry_id: None,
            started_at: "2026-10-01T00:00:00.000Z".into(),
        });
        let err = run(checkpoint(state), json!({"goal": "again"})).unwrap_err();
        assert_eq!(err.to_string(), "Checkpoint already active.");
    }

    #[test]
    fn rewind_requires_active_checkpoint() {
        let state = fresh();
        let err = run(rewind(state), json!({"report": "findings"})).unwrap_err();
        assert_eq!(
            err.to_string(),
            "No active checkpoint. Create a checkpoint before calling rewind."
        );
    }

    #[test]
    fn rewind_after_completed_checkpoint_gets_recovery_message() {
        let state = fresh();
        state.set_last_completed_rewind(CompletedRewindState {
            report: "r".into(),
            started_at: "2026-10-01T00:00:00.000Z".into(),
            rewound_at: "2026-10-01T00:01:00.000Z".into(),
        });
        let err = run(rewind(state), json!({"report": "again"})).unwrap_err();
        assert!(err
            .to_string()
            .starts_with("Checkpoint already completed; continue from the retained rewind report"));
    }

    #[test]
    fn rewind_rejects_blank_report() {
        let state = fresh();
        state.set_checkpoint_state(CheckpointState {
            checkpoint_message_count: 2,
            checkpoint_entry_id: Some("e1".into()),
            started_at: "2026-10-01T00:00:00.000Z".into(),
        });
        let err = run(rewind(state), json!({"report": "   \n "})).unwrap_err();
        assert_eq!(err.to_string(), "Report cannot be empty.");
    }

    #[test]
    fn rewind_result_text_and_details() {
        let state = fresh();
        state.set_checkpoint_state(CheckpointState {
            checkpoint_message_count: 2,
            checkpoint_entry_id: None,
            started_at: "2026-10-01T00:00:00.000Z".into(),
        });
        let result = run(rewind(state), json!({"report": "  root cause: X  "})).unwrap();
        assert_eq!(
            text_of(&result),
            "Rewind requested.\nReport captured for context replacement."
        );
        assert_eq!(result.details["report"], "root cause: X");
        assert_eq!(result.details["rewound"], true);
    }

    #[test]
    fn iso_format_matches_javascript_toisostring() {
        // 2026-10-12T00:00:00.000Z is 1 761 782 400 000 ms after the epoch.
        let parts = civil_from_millis(1_791_763_200_123);
        assert_eq!(parts, (2026, 10, 12, 0, 0, 0, 123));
        let leap = civil_from_millis(951_782_400_000); // 2000-02-29T00:00:00Z
        assert_eq!(leap, (2000, 2, 29, 0, 0, 0, 0));
    }
}
