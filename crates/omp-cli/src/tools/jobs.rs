//! Background `bash` jobs (omp's async mode) and the `proc://` read surface.
//!
//! omp's async bash defers a finite command to the session's job manager and
//! returns `formatBackgroundNotice`; results are injected as a follow-up or
//! inspected via `read proc://` / `read proc://<id>` (see
//! `prompts/internal-urls/proc.md`).
//!
//! Named long-lived services (bash `name`/`ready`) are supervised in-process:
//! the child is spawned piped (omp's shared daemon broker is folded into the
//! session), readiness is observed from a log regex and/or a TCP port probe,
//! the captured log is inspectable via `read proc://<name>`, and the child is
//! killed when the session's registry is dropped (omp stops owned services at
//! session end).

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

/// omp daemon snapshot states (`DaemonSnapshot.state`), service subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceState {
    Starting,
    Ready,
    Running,
    Exited,
    Failed,
}

impl ServiceState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Ready => "ready",
            Self::Running => "running",
            Self::Exited => "exited",
            Self::Failed => "failed",
        }
    }

    #[allow(dead_code)] // proc:// surface state query, wired with the service inspector
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Exited | Self::Failed)
    }
}

/// Live state of one named service (omp broker `ManagedDaemon`, in-proc port).
///
/// The owning child lives in the supervisor task (`bash::start_service`); this
/// record is the shared snapshot the `proc://` surface renders from.
pub struct Service {
    pub name: String,
    #[allow(dead_code)] // rendered by the proc:// surface in the full broker UI
    pub command: String,
    pub started: Instant,
    pub state: ServiceState,
    pub pid: Option<u32>,
    pub ready_at: Option<Instant>,
    pub exited_at: Option<Instant>,
    pub exit_code: Option<i32>,
    pub exit_reason: Option<String>,
    /// Captured service log (bounded), shown via `read proc://<name>`.
    pub output: Arc<Mutex<String>>,
    /// Set by `proc://<name>/kill` and tool-call abort; the supervisor kills
    /// the child on its next tick.
    pub kill_requested: bool,
}

impl Service {
    pub fn is_ready(&self) -> bool {
        self.ready_at.is_some()
    }

    /// The captured log tail, trimmed.
    pub fn log(&self) -> String {
        self.output
            .lock()
            .expect("service output poisoned")
            .trim_end()
            .to_string()
    }
}

#[derive(Default)]
pub(crate) struct JobsInner {
    next_id: usize,
    jobs: HashMap<String, Job>,
    services: HashMap<String, Arc<Mutex<Service>>>,
}

/// Per-session background job registry (omp `asyncJobManager`, job subset).
#[derive(Clone, Default)]
pub struct JobRegistry {
    pub(crate) inner: Arc<Mutex<JobsInner>>,
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
        self.inner
            .lock()
            .expect("job registry poisoned")
            .jobs
            .get(id)
            .cloned()
    }

    pub fn update(&self, id: &str, f: impl FnOnce(&mut Job)) {
        let mut inner = self.inner.lock().expect("job registry poisoned");
        if let Some(job) = inner.jobs.get_mut(id) {
            f(job);
        }
    }

    pub fn sorted(&self) -> Vec<(String, Job)> {
        let inner = self.inner.lock().expect("job registry poisoned");
        let mut v: Vec<(String, Job)> = inner
            .jobs
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// TS `AgentSession.getAsyncJobSnapshot`: running jobs, the newest settled ones and the
    /// (always idle here) delivery queue, as the extension ctx reads it.
    #[must_use]
    pub fn async_snapshot(&self, recent_limit: usize) -> serde_json::Value {
        use serde_json::json;
        let now = i64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis()),
        )
        .unwrap_or(0);
        let start_of = |j: &Job| now - i64::try_from(j.started.elapsed().as_millis()).unwrap_or(0);
        let label = |c: &str| {
            if c.chars().count() > 120 {
                format!("{}...", c.chars().take(117).collect::<String>())
            } else {
                c.to_owned()
            }
        };
        let jobs = self.sorted();
        let running: Vec<_> = jobs
            .iter()
            .filter(|(_, j)| j.status == JobStatus::Running)
            .map(|(id, j)| {
                json!({"id": id, "type": "bash", "status": "running", "label": label(&j.command),
                       "command": j.command, "startTime": start_of(j)})
            })
            .collect();
        let mut settled: Vec<_> = jobs
            .iter()
            .filter(|(_, j)| j.status != JobStatus::Running)
            .collect();
        settled.sort_by_key(|(_, j)| std::cmp::Reverse(start_of(j)));
        let recent: Vec<_> = settled
            .into_iter()
            .take(recent_limit)
            .map(|(id, j)| {
                let start = start_of(j);
                let mut row = json!({"id": id, "type": "bash", "status": j.status.as_str(),
                    "label": label(&j.command), "command": j.command, "startTime": start});
                if let Some(d) = j.ended {
                    row["endTime"] = json!(start + i64::try_from(d.as_millis()).unwrap_or(0));
                }
                row
            })
            .collect();
        json!({"running": running, "recent": recent,
               "delivery": {"queued": 0, "delivering": false, "pendingJobIds": []}})
    }

    /// Register a service under its name (omp start with `replace: true`: a
    /// second start under the same name kills the previous instance).
    pub fn insert_service(&self, service: Service) -> Arc<Mutex<Service>> {
        let arc = Arc::new(Mutex::new(service));
        let mut inner = self.inner.lock().expect("job registry poisoned");
        if let Some(old) = inner.services.insert(
            arc.lock().expect("service poisoned").name.clone(),
            arc.clone(),
        ) {
            old.lock().expect("service poisoned").kill_requested = true;
        }
        arc
    }

    pub fn service(&self, name: &str) -> Option<Arc<Mutex<Service>>> {
        self.inner
            .lock()
            .expect("job registry poisoned")
            .services
            .get(name)
            .cloned()
    }

    pub fn services_sorted(&self) -> Vec<(String, Arc<Mutex<Service>>)> {
        let inner = self.inner.lock().expect("job registry poisoned");
        let mut v: Vec<(String, Arc<Mutex<Service>>)> = inner
            .services
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
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
        Some(t) => {
            format!(" (deadline {t}s)")
        }
    };
    format!(
        "Backgrounded as job {job_id}{deadline}; output auto-delivers on finish; don't poll, keep working or end the turn."
    )
}

/// `read proc://` — the session's background jobs (omp `ProcProtocolHandler.resolve`).
///
/// # Errors
/// Writable-only targets, unknown ids and malformed URLs (omp's error texts).
#[allow(clippy::too_many_lines)] // proc:// render: one read across all service fields, mirrors omp's proc surface
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
        let mut rows: Vec<String> = env
            .jobs
            .sorted()
            .into_iter()
            .map(|(job_id, job)| {
                let duration = match job.ended {
                    None => format!(
                        "up {}",
                        format_duration(
                            u64::try_from(now.duration_since(job.started).as_millis())
                                .unwrap_or(u64::MAX)
                        )
                    ),
                    Some(elapsed) => format!(
                        "in {}",
                        format_duration(u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
                    ),
                };
                format!(
                    "{job_id} [bash] {} {duration} — {}",
                    job.status.as_str(),
                    squash(&job.command)
                )
            })
            .collect();
        // Services ride along (omp `serviceStatus` / ps-data service rows).
        for (name, service) in env.jobs.services_sorted() {
            let service = service.lock().expect("service poisoned");
            let end = service.exited_at.unwrap_or(now);
            let age = format_duration(
                u64::try_from(end.duration_since(service.started).as_millis()).unwrap_or(u64::MAX),
            );
            rows.push(format!(
                "{name} [service] — {} — up {age}{}",
                service.state.as_str(),
                service
                    .pid
                    .map(|p| format!(" — pid {p}"))
                    .unwrap_or_default()
            ));
        }
        let text = if rows.is_empty() {
            "No background jobs or services.".to_string()
        } else {
            rows.join("\n")
        };
        return Ok(AgentToolResult::text(text));
    }
    if let Some(job) = env.jobs.get(id) {
        let mut lines = vec![format!(
            "{id} [bash] — {} — {}",
            job.status.as_str(),
            job.command
        )];
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
        return Ok(AgentToolResult::text(lines.join("\n")));
    }
    let Some(service) = env.jobs.service(id) else {
        bail!("Background job or service not found: {id}");
    };
    {
        let service = service.lock().expect("service poisoned");
        let end = service.exited_at.unwrap_or_else(Instant::now);
        let age = format_duration(
            u64::try_from(end.duration_since(service.started).as_millis()).unwrap_or(u64::MAX),
        );
        let mut lines = vec![format!(
            "{id} [service] — {} — up {age}{}",
            service.state.as_str(),
            service
                .pid
                .map(|p| format!(" — pid {p}"))
                .unwrap_or_default()
        )];
        let log = service.log();
        if !log.is_empty() {
            lines.push(log);
        }
        Ok(AgentToolResult::text(lines.join("\n")))
    }
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
        assert!(
            n.starts_with("Backgrounded as job bg_1 (deadline 300s)"),
            "{n}"
        );
        assert!(
            background_notice("bg_2", None).starts_with("Backgrounded as job bg_2 (no deadline)")
        );
    }
}
