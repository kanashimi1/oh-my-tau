//! Background `bash` jobs (omp's async mode) and the `proc://` read surface.
//!
//! omp's async bash defers a finite command to the session's job manager and
//! returns `formatBackgroundNotice`; results are injected as a follow-up or
//! inspected via `read proc://` / `read proc://<id>` (see
//! `prompts/internal-urls/proc.md`). Services (`name`/`ready`) need omp's
//! daemon broker, which omp-rs does not port; the tool keeps the parameters
//! and returns omp's disabled-mode error text.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use pa_agent::types::AgentToolResult;

/// Live state of one background job.
pub struct Job {
    pub command: String,
    pub started: Instant,
    pub status: JobStatus,
    /// Incremental output tap (bounded), for `proc://<id>` while running.
    pub output: Arc<Mutex<String>>,
    /// Formatted omp bash result text once finished.
    pub result: Option<String>,
    pub error: Option<String>,
    pub ended: Option<Duration>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Running,
    Completed,
    Failed,
}

impl JobStatus {
    fn as_str(self) -> &'static str {
        match self {
            JobStatus::Running => "running",
            JobStatus::Completed => "completed",
            JobStatus::Failed => "failed",
        }
    }
}

#[derive(Default)]
struct JobsInner {
    next_id: usize,
    jobs: HashMap<String, Job>,
}

/// Per-session background job registry (omp `asyncJobManager`, job subset).
#[derive(Clone, Default)]
pub struct JobRegistry {
    inner: Arc<Mutex<JobsInner>>,
}

impl JobRegistry {
    /// Register a job with a fresh `bg_N` id (omp `JobManager.#resolveJobId`).
    pub fn insert(&self, job: Job) -> String {
        let mut inner = self.inner.lock().expect("job registry poisoned");
        if inner.next_id == 0 {
            inner.next_id = 1; // omp `#nextAutoId = 1`: first job is bg_1.
        }
        let id = loop {
            let id = format!("bg_{}", inner.next_id);
            inner.next_id += 1;
            if !inner.jobs.contains_key(&id) {
                break id;
            }
        };
        inner.jobs.insert(id.clone(), job);
        id
    }

    pub fn get(&self, id: &str) -> Option<Job> {
        self.inner.lock().expect("job registry poisoned").jobs.get(id).cloned()
    }

    pub fn update(&self, id: &str, f: impl FnOnce(&mut Job)) {
        let mut inner = self.inner.lock().expect("job registry poisoned");
        if let Some(job) = inner.jobs.get_mut(id) {
            f(job);
        }
    }

    fn sorted(&self) -> Vec<(String, Job)> {
        let inner = self.inner.lock().expect("job registry poisoned");
        let mut v: Vec<(String, Job)> = inner.jobs.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }
}

// Cloning a Job shares the output tap; status/result are snapshotted values,
// which is exactly what the read-only `proc://` surface needs.
impl Clone for Job {
    fn clone(&self) -> Self {
        Self {
            command: self.command.clone(),
            started: self.started,
            status: self.status,
            output: self.output.clone(),
            result: self.result.clone(),
            error: self.error.clone(),
            ended: self.ended,
        }
    }
}

/// omp `formatDuration` (pi-tui render-utils): `m:ss`, `h:mm:ss` beyond an hour.
#[must_use]
pub fn format_duration(ms: u64) -> String {
    let total_secs = ms / 1000;
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3600;
    if hours > 0 {
        format!("{hours}:{mins:02}:{secs:02}")
    } else {
        format!("{mins}:{secs:02}")
    }
}

/// omp `formatBackgroundNotice`.
#[must_use]
pub fn background_notice(job_id: &str, timeout_secs: Option<f64>) -> String {
    let deadline = match timeout_secs {
        None => " (no deadline)".to_string(),
        Some(t) => format!(" (killed once it has run {t}s in total; `timeout: 0` disables the deadline)"),
    };
    format!(
        "Backgrounded as job {job_id}{deadline}; its output is injected into the conversation as a follow-up the moment it finishes. Do NOT poll for it (no `sleep`, `ps`, `pgrep`, `top`, `pidwait`, log tailing): every poll is a wasted turn. Do other work, or end your reply and wait to be woken."
    )
}

/// `read proc://` — the session's background jobs (omp `ProcProtocolHandler.resolve`).
///
/// # Errors
/// Writable-only targets, unknown ids and malformed URLs (omp's error texts).
pub fn read_proc(env: &super::Env, url: &str) -> Result<AgentToolResult> {
    // `url` is everything after `proc://`.
    if url.contains('?') || url.contains('#') || url == "/mode" || url == "/kill" {
        bail!("Invalid proc:// URL: proc://{url}. Use proc://, proc://<id>, proc://<id>/kill, or proc://<id>/mode.");
    }
    let (id, action) = match url.strip_suffix("/mode") {
        Some(rest) => (rest, "mode"),
        None => match url.strip_suffix("/kill") {
            Some(rest) => (rest, "kill"),
            None => (url, "stdin"),
        },
    };
    if id.contains('/') || id == "." || id == ".." {
        bail!("Invalid process id: {id}");
    }
    if action != "stdin" {
        bail!("proc://{id}/{action} is writable only");
    }
    if id.is_empty() {
        let now = Instant::now();
        let rows: Vec<String> = env
            .jobs
            .sorted()
            .into_iter()
            .map(|(job_id, job)| {
                let duration = match job.ended {
                    None => format!("up {}", format_duration(u64::try_from(now.duration_since(job.started).as_millis()).unwrap_or(u64::MAX))),
                    Some(elapsed) => format!("in {}", format_duration(u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))),
                };
                format!("{job_id} [bash] {} {duration} — {}", job.status.as_str(), squash(&job.command))
            })
            .collect();
        let text = if rows.is_empty() {
            "No background jobs or services.".to_string()
        } else {
            rows.join("\n")
        };
        return Ok(AgentToolResult::text(text));
    }
    let Some(job) = env.jobs.get(id) else {
        bail!("Background job or service not found: {id}");
    };
    let mut lines = vec![format!("{id} [bash] — {} — {}", job.status.as_str(), job.command)];
    if let Some(result) = &job.result {
        lines.push(result.clone());
    }
    if let Some(error) = &job.error {
        lines.push(format!("Error: {error}"));
    }
    if job.status == JobStatus::Running {
        let output = job.output.lock().expect("job output poisoned").clone();
        let trimmed = output.trim_end();
        if !trimmed.is_empty() {
            lines.push(trimmed.to_string());
        }
    }
    Ok(AgentToolResult::text(lines.join("\n")))
}

fn squash(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_matches_omp_format() {
        assert_eq!(format_duration(5_000), "0:05");
        assert_eq!(format_duration(65_000), "1:05");
        assert_eq!(format_duration(3_726_000), "1:02:06");
    }

    #[test]
    fn notice_matches_omp_text() {
        let n = background_notice("bg_1", Some(300.0));
        assert!(n.starts_with("Backgrounded as job bg_1 (killed once it has run 300s in total"), "{n}");
        assert!(n.contains("Do NOT poll for it"), "{n}");
        assert!(background_notice("bg_2", None).starts_with("Backgrounded as job bg_2 (no deadline)"));
    }
}
