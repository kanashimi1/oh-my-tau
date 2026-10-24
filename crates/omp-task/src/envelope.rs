//! `<task-result>` envelope for a settled child run, ported from
//! `task/result-summary.ts::formatTaskResultSummary`.
//!
//! The parent always reads the same shape: status, an inline preview (with a
//! 5000-char budget), and — once the output exceeds the budget — an
//! `agent://<id>` pointer to the full artifact.

use serde::Serialize;

use crate::runner::{RunOutput, RunStatus};

/// Inline preview budget before the envelope points at `agent://<id>`.
/// Mirrors `FULL_OUTPUT_THRESHOLD` in `task/result-summary.ts`.
pub const FULL_OUTPUT_THRESHOLD: usize = 5000;

/// Model-facing result envelope for one settled child run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TaskResultEnvelope {
    pub agent: String,
    pub id: String,
    /// `completed` | `failed (exit N)` | `merge failed` | `cancelled`.
    pub status: String,
    pub duration_ms: u64,
    /// Inline preview of the child output (capped at
    /// [`FULL_OUTPUT_THRESHOLD`] chars, line-boundary aligned when possible).
    pub preview: String,
    /// `agent://<id>` handle for the full artifact.
    pub agent_url: String,
    /// True when the preview was cut and the full output lives at `agent_url`.
    pub truncated: bool,
    /// Failure detail (stderr summary, schema violation, abort reason).
    pub error: Option<String>,
}

/// Render the `<task-result>` envelope for a settled run
/// (mirrors `formatTaskResultSummary`).
#[must_use]
pub fn format_task_result(result: &RunOutput) -> TaskResultEnvelope {
    let status = match result.status {
        RunStatus::Cancelled => "cancelled".to_string(),
        RunStatus::Completed => "completed".to_string(),
        RunStatus::Failed => match result.exit_code {
            Some(code) => format!("failed (exit {code})"),
            None => "failed".to_string(),
        },
    };
    let preview_source = if result.output.trim().is_empty() {
        result.stderr.trim()
    } else {
        result.output.trim()
    };
    let preview = preview_head(preview_source);
    let error = if result.status == RunStatus::Completed {
        None
    } else {
        result.schema.summary().or_else(|| {
            if result.stderr.trim().is_empty() {
                None
            } else {
                Some(result.stderr.trim().to_string())
            }
        })
    };
    TaskResultEnvelope {
        agent: result.agent.clone(),
        id: result.id.clone(),
        status,
        duration_ms: u64::try_from(result.duration.as_millis()).unwrap_or(u64::MAX),
        truncated: result.truncated || preview_source.len() > FULL_OUTPUT_THRESHOLD,
        agent_url: format!("agent://{}", result.id),
        preview,
        error,
    }
}

/// Head of the output that fits the inline budget. Prefers a line boundary
/// so the preview does not end mid-row; hard-cuts otherwise
/// (mirrors `previewHead`).
fn preview_head(output: &str) -> String {
    if output.len() <= FULL_OUTPUT_THRESHOLD {
        return output.to_string();
    }
    let slice = &output[..output.floor_char_boundary(FULL_OUTPUT_THRESHOLD)];
    match slice.rfind('\n') {
        Some(idx) if idx >= FULL_OUTPUT_THRESHOLD / 2 => slice[..idx].to_string(),
        _ => slice.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::RunStatus;
    use crate::schema::SchemaValidation;
    use std::time::Duration;

    fn run(output: &str, status: RunStatus, exit_code: Option<i32>) -> RunOutput {
        RunOutput {
            id: "a1".to_string(),
            agent: "scout-glm".to_string(),
            status,
            exit_code,
            output: output.to_string(),
            stderr: String::new(),
            schema: SchemaValidation::NoSchema,
            duration: Duration::from_millis(42),
            truncated: false,
        }
    }

    #[test]
    fn completed_envelope() {
        let env = format_task_result(&run("done", RunStatus::Completed, Some(0)));
        assert_eq!(env.status, "completed");
        assert_eq!(env.preview, "done");
        assert_eq!(env.agent_url, "agent://a1");
        assert!(!env.truncated);
        assert_eq!(env.duration_ms, 42);
    }

    #[test]
    fn failed_envelope_carries_exit_code() {
        let env = format_task_result(&run("", RunStatus::Failed, Some(3)));
        assert_eq!(env.status, "failed (exit 3)");
    }

    #[test]
    fn long_output_previews_with_agent_url_pointer() {
        let long = format!("{}\nTAIL", "x".repeat(FULL_OUTPUT_THRESHOLD + 100));
        let env = format_task_result(&run(&long, RunStatus::Completed, Some(0)));
        assert!(env.truncated);
        assert!(env.preview.len() <= FULL_OUTPUT_THRESHOLD);
        assert!(!env.preview.contains("TAIL"));
        assert_eq!(env.agent_url, "agent://a1");
    }

    #[test]
    fn preview_prefers_line_boundary() {
        let mut long = format!("{}\nsecond-line\n", "y".repeat(4900));
        long.push_str(&"z".repeat(200));
        let env = format_task_result(&run(&long, RunStatus::Completed, Some(0)));
        // Cut at the newline after the first line, not mid-line.
        assert!(env.preview.ends_with("second-line") || env.preview.ends_with('y'));
        assert!(env.preview.len() <= FULL_OUTPUT_THRESHOLD);
    }

    #[test]
    fn preview_cut_never_splits_a_multibyte_char() {
        // 'п' is 2 bytes; an odd prefix puts byte THRESHOLD mid-char.
        let long = format!("x{}", "п".repeat(FULL_OUTPUT_THRESHOLD));
        let env = format_task_result(&run(&long, RunStatus::Completed, Some(0)));
        assert!(env.truncated);
        assert!(env.preview.len() <= FULL_OUTPUT_THRESHOLD);
        assert!(env.preview.ends_with('п'));
    }

    #[test]
    fn empty_output_falls_back_to_stderr() {
        let mut r = run("", RunStatus::Failed, Some(1));
        r.stderr = "boom".to_string();
        let env = format_task_result(&r);
        assert_eq!(env.preview, "boom");
        assert_eq!(env.error.as_deref(), Some("boom"));
    }
}
