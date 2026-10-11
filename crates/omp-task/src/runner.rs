//! Task spawning: run child agents as subprocesses of an `omp-rs -p`-style
//! binary, with bounded concurrency, result collection, and outputSchema
//! enforcement.
//!
//! Ported from `task/spawn-run.ts` (permit-bounded `SpawnRun`) and
//! `task/parallel.ts` (concurrent dispatch): every spawn holds one permit of
//! the runner's semaphore from launch until the child settles; results are
//! collected per item and returned in input order.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;

use crate::envelope::{format_task_result, TaskResultEnvelope};
use crate::params::{SchemaMode, TaskItem};
use crate::schema::{validate_output, SchemaValidation};

/// Maximum output bytes per agent (mirrors `MAX_OUTPUT_BYTES` in
/// `task/types.ts`).
const MAX_OUTPUT_BYTES: usize = 500_000;
/// Maximum output lines per agent (mirrors `MAX_OUTPUT_LINES`).
const MAX_OUTPUT_LINES: usize = 5_000;

/// Terminal status of one child run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    /// Exit code 0 and (in strict mode) a schema-valid output.
    Completed,
    /// Non-zero exit, spawn failure, or strict-mode schema violation.
    Failed,
    /// Killed by the runner (e.g. runner shutdown) or the caller's budget.
    Cancelled,
}

/// Collected output of one settled child run.
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// Caller-chosen or generated agent id; the `agent://<id>` handle.
    pub id: String,
    pub agent: String,
    pub status: RunStatus,
    pub exit_code: Option<i32>,
    /// Captured stdout (the child's final answer), byte/line-capped.
    pub output: String,
    /// Captured stderr.
    pub stderr: String,
    /// Schema validation outcome for the (capped) output.
    pub schema: SchemaValidation,
    pub duration: Duration,
    /// Whether the output was truncated by the byte/line cap.
    pub truncated: bool,
}

impl RunOutput {
    /// Whether this run failed under its schema mode.
    #[must_use]
    pub fn schema_failed(&self, mode: SchemaMode) -> bool {
        self.schema.is_failure(mode)
    }

    /// Render the model-facing `<task-result>` envelope.
    #[must_use]
    pub fn envelope(&self) -> TaskResultEnvelope {
        format_task_result(self)
    }
}

/// A single spawn request handed to the runner.
#[derive(Debug, Clone)]
pub struct TaskSpec {
    /// Stable id for the run (`agent://<id>`). Generated when absent.
    pub id: Option<String>,
    /// Agent definition name to spawn.
    pub agent: String,
    /// Full prompt for the child (task + context already composed).
    pub prompt: String,
    /// Working directory for the child process.
    pub cwd: PathBuf,
    /// Declared output schema for the child's final output.
    pub output_schema: Option<Value>,
    /// Strict vs permissive schema enforcement.
    pub schema_mode: SchemaMode,
}

impl TaskSpec {
    /// Build a spec from a validated [`TaskItem`] plus shared call context.
    #[must_use]
    pub fn from_item(item: &TaskItem, context: Option<&str>, cwd: impl Into<PathBuf>) -> Self {
        let prompt = match context {
            Some(ctx) if !ctx.trim().is_empty() => format!("{ctx}\n\n{}", item.task),
            _ => item.task.clone(),
        };
        Self {
            id: item.name.clone(),
            agent: item.agent.clone(),
            prompt,
            cwd: cwd.into(),
            output_schema: item.output_schema.clone(),
            schema_mode: item.schema_mode,
        }
    }
}

/// Spawns child agents as subprocesses of an `omp-rs -p`-style CLI binary.
///
/// The binary path is injected at construction so tests can substitute a
/// fake child; the default invocation is `<binary> -p <prompt>` run in the
/// spec's `cwd`.
#[derive(Clone)]
pub struct TaskRunner {
    binary: PathBuf,
    semaphore: Arc<Semaphore>,
    /// Monotonic counter backing generated agent ids.
    next_id: Arc<Mutex<u64>>,
}

impl TaskRunner {
    /// Create a runner that spawns `binary` with at most `max_concurrent`
    /// children running at once. `max_concurrent == 0` means unbounded.
    #[must_use]
    pub fn new(binary: impl Into<PathBuf>, max_concurrent: usize) -> Self {
        let semaphore = if max_concurrent == 0 {
            Semaphore::new(usize::MAX >> 8)
        } else {
            Semaphore::new(max_concurrent)
        };
        Self {
            binary: binary.into(),
            semaphore: Arc::new(semaphore),
            next_id: Arc::new(Mutex::new(0)),
        }
    }

    /// Path of the child binary.
    #[must_use]
    pub fn binary(&self) -> &Path {
        &self.binary
    }

    async fn alloc_id(&self) -> String {
        let mut n = self.next_id.lock().await;
        *n += 1;
        format!("agent-{}", *n)
    }

    /// Spawn one child, blocking until a permit is free
    /// (mirrors `SpawnRun`: permit held from launch until settle).
    ///
    /// The returned [`RunOutput`] carries `Failed`/`Cancelled` statuses and
    /// the stderr detail for child-level failures; acquiring the permit only
    /// fails if the semaphore is closed, which the runner never does.
    ///
    /// # Panics
    /// Panics if the runner's internal semaphore is unexpectedly closed.
    pub async fn spawn(&self, spec: TaskSpec) -> RunOutput {
        let id = match &spec.id {
            Some(id) => id.clone(),
            None => self.alloc_id().await,
        };
        let permit = self.semaphore.clone().acquire_owned().await.expect(
            "task runner semaphore is never closed by the runner itself",
        );
        self.run_child(id, spec, permit).await
    }

    /// Run several specs concurrently, returning results in input order
    /// (mirrors the batch dispatch in `task/parallel.ts`).
    ///
    /// # Panics
    /// Panics only if the tokio runtime shuts down mid-dispatch; treat as
    /// unrecoverable, matching the TS unhandled-rejection posture.
    pub async fn spawn_all(&self, specs: Vec<TaskSpec>) -> Vec<RunOutput> {
        let mut set: JoinSet<(usize, RunOutput)> = JoinSet::new();
        for (index, spec) in specs.into_iter().enumerate() {
            let runner = self.clone();
            set.spawn(async move {
                let out = runner.spawn(spec).await;
                (index, out)
            });
        }
        let mut collected = Vec::new();
        while let Some(result) = set.join_next().await {
            // A join error means the task was cancelled (runtime shutdown);
            // surface it as a cancelled placeholder rather than dropping the
            // batch result silently.
            match result {
                Ok(pair) => collected.push(pair),
                Err(e) => collected.push((
                    usize::MAX,
                    RunOutput {
                        id: format!("agent-lost-{e}"),
                        agent: "task".to_string(),
                        status: RunStatus::Cancelled,
                        exit_code: None,
                        output: String::new(),
                        stderr: format!("spawn task cancelled: {e}"),
                        schema: SchemaValidation::NoSchema,
                        duration: Duration::ZERO,
                        truncated: false,
                    },
                )),
            }
        }
        collected.sort_by_key(|(index, _)| *index);
        collected.into_iter().map(|(_, out)| out).collect()
    }

    /// Child permits currently available (for tests/introspection).
    #[must_use]
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    async fn run_child(
        &self,
        id: String,
        spec: TaskSpec,
        _permit: OwnedSemaphorePermit,
    ) -> RunOutput {
        let start = Instant::now();
        let result = tokio::process::Command::new(&self.binary)
            .arg("-p")
            .arg(&spec.prompt)
            .current_dir(&spec.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        match result {
            Ok(out) => {
                let exit_code = out.status.code();
                let mut output = String::from_utf8_lossy(&out.stdout).into_owned();
                let mut truncated = false;
                if output.len() > MAX_OUTPUT_BYTES {
                    output.truncate(MAX_OUTPUT_BYTES);
                    truncated = true;
                }
                let line_count = output.lines().count();
                let output = if line_count > MAX_OUTPUT_LINES {
                    truncated = true;
                    output.lines().take(MAX_OUTPUT_LINES).collect::<Vec<_>>().join("\n")
                } else {
                    output
                };
                let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                let schema = validate_output(&output, spec.output_schema.as_ref());
                let schema_failed = schema.is_failure(spec.schema_mode);
                let status = if out.status.success() && !schema_failed {
                    RunStatus::Completed
                } else {
                    RunStatus::Failed
                };
                RunOutput {
                    id,
                    agent: spec.agent,
                    status,
                    exit_code,
                    output,
                    stderr,
                    schema,
                    duration: start.elapsed(),
                    truncated,
                }
            }
            Err(e) => RunOutput {
                id,
                agent: spec.agent,
                status: RunStatus::Failed,
                exit_code: None,
                output: String::new(),
                stderr: format!("failed to spawn {}: {e}", self.binary.display()),
                schema: SchemaValidation::NoSchema,
                duration: start.elapsed(),
                truncated: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SchemaMode;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    /// Write an executable shell script that acts as a fake `omp-rs` child:
    /// echoes its args to stdout, optionally sleeps (to test concurrency),
    /// optionally exits non-zero, optionally prints invalid JSON.
    fn fake_binary(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("fake-omp-rs");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "#!/bin/sh\n{body}").unwrap();
        let mut perms = f.metadata().unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    fn spec(agent: &str, prompt: &str, cwd: &Path) -> TaskSpec {
        TaskSpec {
            id: None,
            agent: agent.to_string(),
            prompt: prompt.to_string(),
            cwd: cwd.to_path_buf(),
            output_schema: None,
            schema_mode: SchemaMode::Permissive,
        }
    }

    #[tokio::test]
    async fn runs_fake_child_and_collects_output() {
        let dir = tempfile::tempdir().unwrap();
        let bin = fake_binary(dir.path(), r#"echo "$2""#);
        let runner = TaskRunner::new(&bin, 4);
        let out = runner.spawn(spec("scout-glm", "count files", dir.path())).await;
        assert_eq!(out.status, RunStatus::Completed);
        assert_eq!(out.exit_code, Some(0));
        // -p flag and prompt are passed through to the child ($1=-p, $2=prompt).
        assert!(out.output.contains("count files"), "got: {:?}", out.output);
        assert_eq!(out.agent, "scout-glm");
        assert!(!out.truncated);
        let envelope = out.envelope();
        assert_eq!(envelope.status, "completed");
    }

    #[tokio::test]
    async fn nonzero_exit_is_failed() {
        let dir = tempfile::tempdir().unwrap();
        let bin = fake_binary(dir.path(), "echo oops; echo err >&2; exit 3");
        let runner = TaskRunner::new(&bin, 1);
        let out = runner.spawn(spec("task", "x", dir.path())).await;
        assert_eq!(out.status, RunStatus::Failed);
        assert_eq!(out.exit_code, Some(3));
        assert_eq!(out.stderr.trim(), "err");
        assert_eq!(out.envelope().status, "failed (exit 3)");
    }

    #[tokio::test]
    async fn missing_binary_is_failed_not_panicked() {
        let dir = tempfile::tempdir().unwrap();
        let runner = TaskRunner::new(dir.path().join("does-not-exist"), 1);
        let out = runner.spawn(spec("task", "x", dir.path())).await;
        assert_eq!(out.status, RunStatus::Failed);
        assert!(out.stderr.contains("failed to spawn"));
    }

    #[tokio::test]
    async fn strict_schema_failure_marks_run_failed() {
        let dir = tempfile::tempdir().unwrap();
        let bin = fake_binary(dir.path(), "echo 'not json'");
        let runner = TaskRunner::new(&bin, 1);
        let mut s = spec("task", "x", dir.path());
        s.output_schema = Some(serde_json::json!({"type": "object"}));
        s.schema_mode = SchemaMode::Strict;
        let out = runner.spawn(s).await;
        assert_eq!(out.status, RunStatus::Failed);
        assert!(matches!(out.schema, SchemaValidation::NotJson));
        // Permissive mode would have completed.
        let mut s2 = spec("task", "x", dir.path());
        s2.output_schema = Some(serde_json::json!({"type": "object"}));
        s2.schema_mode = SchemaMode::Permissive;
        let out2 = runner.spawn(s2).await;
        assert_eq!(out2.status, RunStatus::Completed);
    }

    #[tokio::test]
    async fn concurrency_limit_is_respected() {
        let dir = tempfile::tempdir().unwrap();
        // Each child sleeps 0.2s and appends its PID to a shared log.
        let log = dir.path().join("pids");
        let bin = fake_binary(
            dir.path(),
            &format!(
                "echo $$ >> {}; sleep 0.2",
                log.display()
            ),
        );
        let runner = TaskRunner::new(&bin, 2);
        assert_eq!(runner.available_permits(), 2);
        let specs: Vec<TaskSpec> = (0..5)
            .map(|i| TaskSpec {
                id: Some(format!("w{i}")),
                ..spec("task", "x", dir.path())
            })
            .collect();
        let results = runner.spawn_all(specs).await;
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.status == RunStatus::Completed));
        assert_eq!(runner.available_permits(), 2);
        // With a limit of 2 and 0.2s per child, 5 children take >= 0.6s
        // wall-clock; unbounded would finish in ~0.2s.
        let pids = std::fs::read_to_string(&log).unwrap();
        assert_eq!(pids.lines().count(), 5);
    }

    #[tokio::test]
    async fn spawn_all_preserves_input_order() {
        let dir = tempfile::tempdir().unwrap();
        let bin = fake_binary(dir.path(), r#"echo "child:$1""#);
        let runner = TaskRunner::new(&bin, 3);
        let specs: Vec<TaskSpec> = (0..4)
            .map(|i| TaskSpec {
                id: Some(format!("child{i}")),
                ..spec("task", &format!("job{i}"), dir.path())
            })
            .collect();
        let results = runner.spawn_all(specs).await;
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["child0", "child1", "child2", "child3"]);
    }

    #[tokio::test]
    async fn caller_supplied_id_is_the_agent_handle() {
        let dir = tempfile::tempdir().unwrap();
        let bin = fake_binary(dir.path(), "echo hi");
        let runner = TaskRunner::new(&bin, 1);
        let out = runner
            .spawn(TaskSpec {
                id: Some("Parent.Child-1".to_string()),
                ..spec("task", "x", dir.path())
            })
            .await;
        assert_eq!(out.envelope().agent_url, "agent://Parent.Child-1");
    }
}
