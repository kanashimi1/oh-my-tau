//! `bash`: one shell command (`bash -c`, stderr merged into stdout), omp's
//! result text: output, `Wall time: N.NN seconds`, `Command exited with code N`.
//! Non-persistent shell (omp embeds a brush-based shell; see the port report).
//!
//! omp's `async: true` backgrounds the command in the session job manager and
//! returns `formatBackgroundNotice`; the job is inspected via `read proc://`
//! (`tools/jobs.rs`). Services (`name`/`ready`) need the daemon broker omp-rs
//! does not port: the params stay schema-compatible and the tool returns
//! omp's disabled-mode error text.

use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::types::AgentToolResult;
use serde_json::Value;
use tokio::io::AsyncReadExt;

use super::jobs::{self, JobRegistry, JobStatus};
use super::util::{format_bytes, resolve_to_cwd, str_arg, truncate_head, truncate_tail, DEFAULT_MAX_BYTES};

const DEFAULT_TIMEOUT_SECS: f64 = 300.0;
const MIN_TIMEOUT_SECS: f64 = 1.0;
const MAX_TIMEOUT_SECS: f64 = 3600.0;
const HEAD_BYTES: usize = 20 * 1024;
const TAIL_BYTES: usize = 20 * 1024;
/// Bytes held in memory while the command runs (the rest is dropped from the middle).
const RETAIN_BYTES: usize = 8 * DEFAULT_MAX_BYTES;

/// omp `formatWallTimeNotice`.
#[must_use]
pub fn wall_time_notice(elapsed: Duration) -> String {
    format!("Wall time: {:.2} seconds", elapsed.as_secs_f64())
}

/// Middle elision of over-long output (omp `truncateMiddle` with the default
/// 20KB head + 20KB tail budget); returns the text and, when cut, the notice.
#[must_use]
pub fn elide_middle(output: &str) -> (String, Option<String>) {
    if output.len() <= DEFAULT_MAX_BYTES {
        return (output.to_string(), None);
    }
    let head = truncate_head(output, HEAD_BYTES);
    let tail = truncate_tail(output, TAIL_BYTES);
    let total = head.total_lines;
    let elided_lines = total.saturating_sub(head.output_lines + tail.output_lines);
    let elided_bytes = output.len().saturating_sub(head.content.len() + tail.content.len());
    let marker = if elided_lines <= 1 {
        format!("[…{elided_bytes}B elided…]")
    } else {
        format!("[…{elided_lines}ln elided…]")
    };
    let tail_start = total - tail.output_lines + 1;
    let notice = format!(
        "Showing lines 1-{} and {tail_start}-{total} of {total}; {elided_lines} middle line{} ({}) elided",
        head.output_lines,
        if elided_lines == 1 { "" } else { "s" },
        format_bytes(elided_bytes)
    );
    (format!("{}\n{marker}\n{}", head.content, tail.content), Some(notice))
}

fn timeout_arg(args: &Value) -> Option<f64> {
    match args.get("timeout").and_then(Value::as_f64) {
        None => Some(DEFAULT_TIMEOUT_SECS),
        Some(t) if t == 0.0 => None,
        Some(t) => Some(t.clamp(MIN_TIMEOUT_SECS, MAX_TIMEOUT_SECS)),
    }
}

/// Run a command and return `(output, exit code, wall time, outcome)`.
///
/// `tap`, when given, receives every output chunk as it arrives (the
/// background-job surface shows it via `read proc://<id>` while running).
async fn run(
    cwd: &std::path::Path,
    command: &str,
    timeout: Option<f64>,
    signal: &AbortSignal,
    tap: Option<Arc<Mutex<String>>>,
) -> Result<(String, Option<i32>, Duration, Option<&'static str>)> {
    let script = format!("exec 2>&1\n{command}");
    let started = Instant::now();
    let mut child = tokio::process::Command::new("bash")
        .arg("-c")
        .arg(script)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("failed to spawn bash in {}", cwd.display()))?;
    let mut stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 8192];
    let deadline = async {
        match timeout {
            Some(t) => tokio::time::sleep(Duration::from_secs_f64(t)).await,
            None => std::future::pending::<()>().await,
        }
    };
    tokio::pin!(deadline);
    let mut outcome = None;
    loop {
        tokio::select! {
            n = stdout.read(&mut chunk) => {
                let n = n.context("reading command output")?;
                if n == 0 { break; }
                if let Some(tap) = &tap {
                    let mut t = tap.lock().expect("job output tap poisoned");
                    t.push_str(&String::from_utf8_lossy(&chunk[..n]));
                    // Bound the tap: proc reads only need the recent window.
                    if t.len() > RETAIN_BYTES {
                        let cut = t.len() - RETAIN_BYTES / 2;
                        t.drain(..cut);
                    }
                }
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() > RETAIN_BYTES * 2 {
                    // Keep the head and tail windows; drop the middle.
                    let tail = buf.split_off(buf.len() - RETAIN_BYTES / 2);
                    buf.truncate(RETAIN_BYTES / 2);
                    buf.extend_from_slice(&tail);
                }
            }
            () = &mut deadline => { outcome = Some("timeout"); break; }
            () = signal.aborted() => { outcome = Some("aborted"); break; }
        }
    }
    let code = if outcome.is_some() {
        let _ = child.kill().await;
        None
    } else {
        child.wait().await.context("waiting for command")?.code()
    };
    Ok((String::from_utf8_lossy(&buf).into_owned(), code, started.elapsed(), outcome))
}

/// Outcome of a host-issued shell command (rpc `bash`), as omp's `BashResult`.
pub struct Executed {
    pub output: String,
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub timed_out: bool,
    pub truncated: bool,
    pub total_lines: usize,
    pub total_bytes: usize,
    pub output_lines: usize,
    pub output_bytes: usize,
}

/// Run `command` in `cwd` for the rpc `bash` command.
///
/// # Errors
/// Spawn or read failures.
pub async fn execute(cwd: &std::path::Path, command: &str, signal: &AbortSignal) -> Result<Executed> {
    let (raw, code, _, outcome) = run(cwd, command, Some(DEFAULT_TIMEOUT_SECS), signal, None).await?;
    let total_bytes = raw.len();
    let total_lines = raw.lines().count();
    let t = truncate_tail(&raw, DEFAULT_MAX_BYTES);
    Ok(Executed {
        output_bytes: t.content.len(),
        output_lines: t.content.lines().count(),
        truncated: t.truncated,
        output: t.content,
        exit_code: code,
        cancelled: outcome == Some("aborted"),
        timed_out: outcome == Some("timeout"),
        total_lines,
        total_bytes,
    })
}

/// Format a finished command as omp's bash result text (without error propagation).
fn finished_text(output: &str, elapsed: Duration, outcome: Option<&str>, timeout: Option<f64>, code: Option<i32>) -> String {
    let (shown, elision) = elide_middle(output.trim_end_matches('\n'));
    let body = if shown.is_empty() { "(no output)".to_string() } else { shown };
    let mut text = format!("{body}\n\n{}", wall_time_notice(elapsed));
    if let Some(n) = elision {
        text.push_str(&format!("\n\n[{n}]"));
    }
    match outcome {
        Some("timeout") => {
            let secs = timeout.unwrap_or(0.0);
            text.push_str(&format!("\n\n[Command timed out after {secs} seconds]"));
        }
        Some(_) => text.push_str("\n\n[Command aborted]"),
        None => {}
    }
    if outcome.is_none() {
        if let Some(c) = code {
            if c != 0 {
                text.push_str(&format!("\n\nCommand exited with code {c}"));
            }
        } else {
            text.push_str("\n\nCommand failed: missing exit status");
        }
    }
    text
}

/// Whether `ready` carries any service-only field (omp `normalizeReady`).
fn ready_given(args: &Value) -> bool {
    let r = &args["ready"];
    if !r.is_object() {
        return false;
    }
    ["log", "host"].iter().any(|k| r[k].as_str().is_some_and(|s| !s.trim().is_empty()))
        || ["port", "timeout"].iter().any(|k| r[k].as_f64().is_some())
}

/// Run the `bash` tool.
///
/// # Errors
/// Spawn failure, timeout, abort or a non-zero exit (carrying the output);
/// disabled service mode (omp's error text).
pub async fn bash(env: std::sync::Arc<super::Env>, args: Value, signal: AbortSignal) -> Result<AgentToolResult> {
    let cwd = env.cwd.clone();
    let command = str_arg(&args, "command")?.to_string();
    let dir = args
        .get("cwd")
        .and_then(Value::as_str)
        .map_or_else(|| cwd.clone(), |d| resolve_to_cwd(d, &cwd));
    let name = args.get("name").and_then(Value::as_str).map(str::trim).filter(|n| !n.is_empty());
    let async_requested = args.get("async").and_then(Value::as_bool) == Some(true);
    let timeout = timeout_arg(&args);
    let mut notices: Vec<String> = Vec::new();
    if let Some(name) = name {
        // omp-rs has no daemon broker; keep omp's disabled-mode error text.
        let _ = name;
        bail!("Service launch is disabled in this session.");
    }
    if ready_given(&args) {
        // Nothing can honour ready without a service to attach it to (omp).
        notices.push("Ignored ready: service-only, and no service name was given.".to_string());
    }
    if async_requested {
        return start_background(&env.jobs, dir, command, timeout, notices, &signal);
    }
    let (output, code, elapsed, outcome) = run(&dir, &command, timeout, &signal, None).await?;
    let mut text = finished_text(&output, elapsed, outcome, timeout, code);
    if !notices.is_empty() {
        text = format!("{}\n{text}", notices.join("\n"));
    }
    if outcome.is_some() || code != Some(0) {
        bail!("{text}");
    }
    Ok(AgentToolResult::text(text))
}

/// omp `BashTool` async mode: register a `bg_N` job, run it detached, and
/// return `formatBackgroundNotice` (the result is read later via `proc://`).
fn start_background(
    registry: &JobRegistry,
    dir: std::path::PathBuf,
    command: String,
    timeout: Option<f64>,
    notices: Vec<String>,
    _signal: &AbortSignal,
) -> Result<AgentToolResult> {
    let output: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let job = jobs::Job {
        command: command.clone(),
        started: Instant::now(),
        status: JobStatus::Running,
        output: output.clone(),
        result: None,
        error: None,
        ended: None,
    };
    let job_id = registry.insert(job);
    let registry = registry.clone();
    let task_job_id = job_id.clone();
    tokio::spawn(async move {
        let controller = pa_agent::abort::AbortController::new();
        let raw = run(&dir, &command, timeout, &controller.signal(), Some(output)).await;
        let (status, result, error) = match &raw {
            Ok((output, code, elapsed, outcome)) => {
                let text = finished_text(output, *elapsed, *outcome, timeout, *code);
                let status = if outcome.is_none() && *code == Some(0) { JobStatus::Completed } else { JobStatus::Failed };
                let error = (status == JobStatus::Failed).then(|| {
                    if *outcome == Some("timeout") {
                        format!("Command timed out after {} seconds", timeout.unwrap_or(0.0))
                    } else if *outcome == Some("aborted") {
                        "[Command aborted]".to_string()
                    } else {
                        format!("Command exited with code {}", code.map_or("?".to_string(), |c| c.to_string()))
                    }
                });
                (status, Some(text), error)
            }
            Err(e) => (JobStatus::Failed, None, Some(e.to_string())),
        };
        registry.update(&task_job_id, |job| {
            job.status = status;
            job.result = result;
            job.error = error;
            job.ended = Some(job.started.elapsed());
        });
    });
    let mut lines: Vec<String> = Vec::new();
    lines.extend(notices);
    lines.push(jobs::background_notice(&job_id, timeout));
    Ok(AgentToolResult::text(lines.join("\n")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig() -> AbortSignal {
        pa_agent::abort::AbortController::new().signal()
    }

    fn text_of(r: &AgentToolResult) -> String {
        r.content
            .iter()
            .filter_map(|c| match c {
                pa_agent::types::ToolResultContent::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn test_env() -> std::sync::Arc<super::super::Env> {
        std::sync::Arc::new(super::super::Env::new(&std::env::temp_dir(), false))
    }

    #[tokio::test]
    async fn merges_streams_and_appends_wall_time() {
        let env = test_env();
        let r = bash(
            env.clone(),
            serde_json::json!({"command": "echo out; echo err 1>&2"}),
            sig(),
        )
        .await
        .unwrap();
        let t = text_of(&r);
        assert!(t.starts_with("out\nerr\n\nWall time: "), "{t}");
        assert!(t.ends_with(" seconds"), "{t}");
    }

    #[tokio::test]
    async fn nonzero_exit_is_an_error_with_output_and_code() {
        let env = test_env();
        let e = bash(env.clone(), serde_json::json!({"command": "echo boom; exit 3"}), sig())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("boom\n\nWall time: "), "{e}");
        assert!(e.ends_with("\n\nCommand exited with code 3"), "{e}");
    }

    #[tokio::test]
    async fn empty_output_and_timeout() {
        let env = test_env();
        let t = text_of(&bash(env.clone(), serde_json::json!({"command": "true"}), sig()).await.unwrap());
        assert!(t.starts_with("(no output)\n\nWall time: "), "{t}");
        let e = bash(env.clone(), serde_json::json!({"command": "sleep 5", "timeout": 1}), sig())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.ends_with("[Command timed out after 1 seconds]"), "{e}");
    }

    #[tokio::test]
    async fn async_mode_backgrounds_job_with_proc_retrieval() {
        let env = test_env();
        let r = bash(
            env.clone(),
            serde_json::json!({"command": "echo first; sleep 0.2; echo done-marker", "async": true}),
            sig(),
        )
        .await
        .unwrap();
        let t = text_of(&r);
        assert!(t.starts_with("Backgrounded as job bg_1 (killed once it has run 300s in total"), "{t}");
        assert!(t.contains("Do NOT poll for it"), "{t}");

        // proc:// lists the running job.
        let listing = super::super::jobs::read_proc(&env, "").unwrap();
        let lt = text_of(&listing);
        assert!(lt.contains("bg_1 [bash] running up "), "{lt}");
        assert!(lt.contains("echo first; sleep 0.2; echo done-marker"), "{lt}");

        // Wait for completion, then proc://<id> carries the formatted result.
        for _ in 0..50 {
            let j = env.jobs.get("bg_1").unwrap();
            if j.status != JobStatus::Running {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let detail = super::super::jobs::read_proc(&env, "bg_1").unwrap();
        let dt = text_of(&detail);
        assert!(dt.starts_with("bg_1 [bash] — completed — echo first"), "{dt}");
        assert!(dt.contains("first\ndone-marker\n\nWall time: "), "{dt}");
        assert!(dt.ends_with(" seconds"), "{dt}");

        let after = super::super::jobs::read_proc(&env, "").unwrap();
        assert!(text_of(&after).contains("bg_1 [bash] completed in "), "{}", text_of(&after));
    }

    #[tokio::test]
    async fn service_name_returns_omp_disabled_error() {
        let env = test_env();
        let e = bash(env.clone(), serde_json::json!({"command": "echo x", "name": "svc"}), sig())
            .await
            .unwrap_err()
            .to_string();
        assert_eq!(e, "Service launch is disabled in this session.");
    }

    #[tokio::test]
    async fn ready_without_name_is_ignored_with_notice() {
        let env = test_env();
        let r = bash(
            env.clone(),
            serde_json::json!({"command": "echo hi", "ready": {"port": 8080}}),
            sig(),
        )
        .await
        .unwrap();
        let t = text_of(&r);
        assert!(t.starts_with("Ignored ready: service-only, and no service name was given.\nhi\n\nWall time: "), "{t}");
    }

    #[test]
    fn long_output_is_elided_in_the_middle() {
        let long: String = (1..=5000).map(|i| format!("line number {i}\n")).collect();
        let (shown, notice) = elide_middle(long.trim_end());
        assert!(shown.len() < long.len());
        assert!(shown.starts_with("line number 1\n"));
        assert!(shown.ends_with("line number 5000"));
        assert!(shown.contains("ln elided…]"));
        assert!(notice.unwrap().contains("middle lines"));
    }
}
