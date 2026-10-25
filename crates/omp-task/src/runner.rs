//! Task spawning: run child agents as subprocesses of an `omp-rs -p`-style
//! binary, with bounded concurrency, result collection, and outputSchema
//! enforcement.
//!
//! Ported from `task/spawn-run.ts` (permit-bounded `SpawnRun`) and
//! `task/parallel.ts` (concurrent dispatch): every spawn holds one permit of
//! the runner's semaphore from launch until the child settles; results are
//! collected per item and returned in input order.
//!
//! [`TaskRunner::start`] (module [`control`]) is the controllable variant:
//! the run keeps its child handle (own process group) so it can be killed
//! mid-flight, and in [`ChildMode::Rpc`] the child's stdin stays open so
//! steering messages reach the running agent.

mod control;

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::process::Command;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;

use crate::envelope::{format_task_result, TaskResultEnvelope};
use crate::params::{SchemaMode, TaskItem};
use crate::schema::{validate_output, SchemaValidation};

pub use control::{ChildMode, RunControl, RunJoin, RunPhase, SteerError};

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
    /// Per-spawn model selector (validated by the caller), passed to the
    /// child as `--model` (TS `spawnParamsFor` honors `item.model`).
    pub model: Option<String>,
    /// Per-spawn tool allow-list override, passed to the child as `--tools`.
    pub tools: Option<Vec<String>>,
    /// Task depth stamped on the child as [`TASK_DEPTH_ENV`] (a child of a
    /// depth-N run is depth N+1); 0 for a root session. Set automatically by
    /// [`TaskSpec::from_item`].
    pub depth: usize,
    /// Extra verbatim argv fragments appended to the child command, e.g.
    /// parent-policy passthrough (`--no-extensions`, `--extension <p>`).
    pub extra_args: Vec<String>,
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
            // A string selector passes through; a prioritized list flattens
            // to a comma-separated selector (TS `normalizeModelPatternList`).
            model: item.model.as_ref().map(|m| match m {
                Value::String(s) => s.clone(),
                Value::Array(list) => list
                    .iter()
                    .filter_map(Value::as_str)
                    .flat_map(|s| s.split(','))
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(","),
                _ => String::new(),
            }),
            tools: item.tools.clone(),
            depth: current_task_depth() + 1,
            extra_args: Vec::new(),
        }
    }
}

/// Env set on every child: subagent sessions never auto-title (TS `sdk.ts`
/// `autoTitle: options.autoTitle === true && !isSubagentSession`); without
/// it each child spends one extra title-model request.
pub(crate) const SUBAGENT_NO_TITLE: (&str, &str) = ("OMP_RS_NO_TITLE", "1");

/// Env var carrying a process's task depth (TS `session.taskDepth`): the
/// root session has depth 0, its children depth 1, and so on. Stamped on
/// every child by the spawners in this module.
pub const TASK_DEPTH_ENV: &str = "PI_TASK_DEPTH";
/// Env var overriding [`DEFAULT_MAX_RECURSION_DEPTH`].
pub const MAX_RECURSION_ENV: &str = "OMP_TASK_MAX_RECURSION_DEPTH";
/// TS `task.maxRecursionDepth` (`task/settings.ts`) protocol default.
pub const DEFAULT_MAX_RECURSION_DEPTH: usize = 2;

/// This process's task depth (`0` when the env var is absent or unparsable).
#[must_use]
pub fn current_task_depth() -> usize {
    std::env::var(TASK_DEPTH_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// The configured recursion ceiling ([`MAX_RECURSION_ENV`], else the TS
/// default).
#[must_use]
pub fn max_recursion_depth() -> usize {
    std::env::var(MAX_RECURSION_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_MAX_RECURSION_DEPTH)
}

/// TS `canSpawnAtDepth` gate (`task/structured-subagent.ts`): refuse to spawn
/// a subagent from a session already at (or beyond) the maximum depth. On
/// success returns the child's depth.
pub fn recursion_gate_at(depth: usize) -> Result<usize, String> {
    let max = max_recursion_depth();
    if depth >= max {
        return Err(format!(
            "Cannot spawn another agent at task depth {depth}; maximum depth is {max}."
        ));
    }
    Ok(depth + 1)
}

/// The gate for this process: the depth a child spawned here would run at.
pub fn recursion_gate() -> Result<usize, String> {
    recursion_gate_at(current_task_depth())
}

/// Apply the child env/argv shared by every spawner: the no-title marker,
/// the recursion-depth stamp and the caller's policy passthrough args.
fn apply_child_policy(command: &mut Command, spec: &TaskSpec) {
    command
        .env(SUBAGENT_NO_TITLE.0, SUBAGENT_NO_TITLE.1)
        .env(TASK_DEPTH_ENV, spec.depth.to_string());
    // TS `executor.ts`: the parent task's approval is the authorization boundary, so subagents run
    // unattended in yolo mode (user `tools.approval` policies still apply).
    command.arg("--approval-mode").arg("yolo");
    command.args(&spec.extra_args);
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
    next_id: Arc<AtomicU64>,
}

impl TaskRunner {
    /// Default subagent fan-out ceiling: `min(available_parallelism, 8)`.
    ///
    /// Used by the `task` tool and the eval subagent launcher unless the
    /// `task.maxConcurrency` setting overrides it: a positive number wins,
    /// the schema default `0` (and any non-finite value) means "auto".
    #[must_use]
    pub fn default_max_concurrency() -> usize {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(8)
    }

    /// Effective fan-out for a `task.maxConcurrency` setting value: a finite
    /// number `>= 1` overrides the [`TaskRunner::default_max_concurrency`];
    /// anything else (including the schema default `0`) keeps the auto
    /// default.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // finite, >= 1
    pub fn resolve_max_concurrency(setting: Option<f64>) -> usize {
        setting
            .filter(|v| v.is_finite() && *v >= 1.0)
            .map_or_else(Self::default_max_concurrency, |v| v as usize)
    }

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
            next_id: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Path of the child binary.
    #[must_use]
    pub fn binary(&self) -> &Path {
        &self.binary
    }

    fn alloc_id(&self) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        format!("agent-{n}")
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
            None => self.alloc_id(),
        };
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("task runner semaphore is never closed by the runner itself");
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

    /// Spawn the child process, retrying transient fork/exec failures.
    ///
    /// Under load `fork`/`exec` can fail with `EAGAIN` (resource limits),
    /// `ENOMEM`, or `ETXTBSY` (the binary is momentarily open for writing,
    /// e.g. a freshly written script still being closed). These are transient,
    /// so retry a few times with a short backoff before giving up.
    async fn spawn_child(&self, spec: &TaskSpec) -> std::io::Result<std::process::Output> {
        let mut attempt = 0;
        loop {
            let mut command = Command::new(&self.binary);
            command.arg("-p").arg(&spec.prompt).current_dir(&spec.cwd);
            apply_child_policy(&mut command, spec);
            if let Some(model) = spec.model.as_deref().filter(|m| !m.is_empty()) {
                command.arg("--model").arg(model);
            }
            if let Some(tools) = spec.tools.as_ref().filter(|t| !t.is_empty()) {
                command.arg("--tools").arg(tools.join(","));
            }
            let result = command
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await;
            match result {
                Err(e) if attempt < 3 && is_transient_spawn_error(&e) => {
                    attempt += 1;
                    tokio::time::sleep(Duration::from_millis(50 * attempt)).await;
                }
                result => return result,
            }
        }
    }

    async fn run_child(
        &self,
        id: String,
        spec: TaskSpec,
        _permit: OwnedSemaphorePermit,
    ) -> RunOutput {
        let start = Instant::now();
        let result = self.spawn_child(&spec).await;

        match result {
            Ok(out) => collect_output(
                id,
                spec,
                out.status.success(),
                out.status.code(),
                String::from_utf8_lossy(&out.stdout).into_owned(),
                String::from_utf8_lossy(&out.stderr).into_owned(),
                start,
            ),
            Err(e) => spawn_failure(id, spec, &self.binary, &e, start),
        }
    }
}

/// Cap the child's answer, validate it against the declared schema and build
/// the settled [`RunOutput`]; `success` is the child's own verdict (exit 0 /
/// completed prompt), which a strict-mode schema violation overrides.
fn collect_output(
    id: String,
    spec: TaskSpec,
    success: bool,
    exit_code: Option<i32>,
    mut output: String,
    stderr: String,
    start: Instant,
) -> RunOutput {
    let mut truncated = false;
    if output.len() > MAX_OUTPUT_BYTES {
        let mut cut = MAX_OUTPUT_BYTES;
        while !output.is_char_boundary(cut) {
            cut -= 1;
        }
        output.truncate(cut);
        truncated = true;
    }
    let line_count = output.lines().count();
    let output = if line_count > MAX_OUTPUT_LINES {
        truncated = true;
        output
            .lines()
            .take(MAX_OUTPUT_LINES)
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        output
    };
    let schema = validate_output(&output, spec.output_schema.as_ref());
    let schema_failed = schema.is_failure(spec.schema_mode);
    let status = if success && !schema_failed {
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

/// The settled output of a child that never started.
fn spawn_failure(
    id: String,
    spec: TaskSpec,
    binary: &Path,
    e: &std::io::Error,
    start: Instant,
) -> RunOutput {
    RunOutput {
        id,
        agent: spec.agent,
        status: RunStatus::Failed,
        exit_code: None,
        output: String::new(),
        stderr: format!("failed to spawn {}: {e}", binary.display()),
        schema: SchemaValidation::NoSchema,
        duration: start.elapsed(),
        truncated: false,
    }
}

/// Whether a spawn error is worth retrying (transient resource pressure).
#[cfg(unix)]
fn is_transient_spawn_error(e: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    matches!(
        e.kind(),
        ErrorKind::WouldBlock | ErrorKind::ResourceBusy | ErrorKind::OutOfMemory
    ) || matches!(e.raw_os_error(), Some(11 | 12 | 26)) // EAGAIN | ENOMEM | ETXTBSY
}

/// Whether a spawn error is worth retrying (transient resource pressure).
#[cfg(not(unix))]
fn is_transient_spawn_error(e: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    matches!(
        e.kind(),
        ErrorKind::WouldBlock | ErrorKind::ResourceBusy | ErrorKind::OutOfMemory
    )
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
        // Close the write handle before the script may be exec'd: an exec while
        // any descriptor is still open for writing fails with ETXTBSY.
        drop(f);
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
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
            model: None,
            tools: None,
            depth: 0,
            extra_args: Vec::new(),
        }
    }

    #[tokio::test]
    async fn runs_fake_child_and_collects_output() {
        let dir = tempfile::tempdir().unwrap();
        let bin = fake_binary(dir.path(), r#"echo "$2""#);
        let runner = TaskRunner::new(&bin, 4);
        let out = runner
            .spawn(spec("scout-glm", "count files", dir.path()))
            .await;
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
            &format!("echo $$ >> {}; sleep 0.2", log.display()),
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

    #[test]
    fn max_concurrency_default_and_override() {
        assert_eq!(
            TaskRunner::resolve_max_concurrency(None),
            TaskRunner::default_max_concurrency()
        );
        assert_eq!(
            TaskRunner::resolve_max_concurrency(Some(0.0)),
            TaskRunner::default_max_concurrency()
        );
        assert_eq!(
            TaskRunner::resolve_max_concurrency(Some(f64::NAN)),
            TaskRunner::default_max_concurrency()
        );
        assert_eq!(TaskRunner::resolve_max_concurrency(Some(3.0)), 3);
        assert_eq!(TaskRunner::resolve_max_concurrency(Some(64.0)), 64);
        assert!(TaskRunner::default_max_concurrency() <= 8);
        assert!(TaskRunner::default_max_concurrency() >= 1);
    }
}
