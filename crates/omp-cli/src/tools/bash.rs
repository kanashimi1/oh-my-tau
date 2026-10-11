//! `bash`: one shell command (`bash -c`, stderr merged into stdout), omp's
//! result text: output, `Wall time: N.NN seconds`, `Command exited with code N`.
//! Non-persistent shell (omp embeds a brush-based shell; see the port report).

use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::types::AgentToolResult;
use serde_json::Value;
use tokio::io::AsyncReadExt;

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
async fn run(
    cwd: &std::path::Path,
    command: &str,
    timeout: Option<f64>,
    signal: &AbortSignal,
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
    let (raw, code, _, outcome) = run(cwd, command, Some(DEFAULT_TIMEOUT_SECS), signal).await?;
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

/// Run the `bash` tool.
///
/// # Errors
/// Spawn failure, timeout, abort or a non-zero exit (carrying the output).
pub async fn bash(cwd: std::path::PathBuf, args: Value, signal: AbortSignal) -> Result<AgentToolResult> {
    let command = str_arg(&args, "command")?.to_string();
    let dir = args
        .get("cwd")
        .and_then(Value::as_str)
        .map_or_else(|| cwd.clone(), |d| resolve_to_cwd(d, &cwd));
    let timeout = timeout_arg(&args);
    let (output, code, elapsed, outcome) = run(&dir, &command, timeout, &signal).await?;
    let (shown, elision) = elide_middle(output.trim_end_matches('\n'));
    let body = if shown.is_empty() { "(no output)".to_string() } else { shown };
    let mut text = format!("{body}\n\n{}", wall_time_notice(elapsed));
    if let Some(n) = elision {
        text.push_str(&format!("\n\n[{n}]"));
    }
    match outcome {
        Some("timeout") => {
            let secs = timeout.unwrap_or(0.0);
            bail!("{text}\n\n[Command timed out after {secs} seconds]");
        }
        Some(_) => bail!("{text}\n\n[Command aborted]"),
        None => {}
    }
    match code {
        Some(0) => Ok(AgentToolResult::text(text)),
        Some(c) => bail!("{text}\n\nCommand exited with code {c}"),
        None => bail!("{text}\n\nCommand failed: missing exit status"),
    }
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

    #[tokio::test]
    async fn merges_streams_and_appends_wall_time() {
        let r = bash(
            std::env::temp_dir(),
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
        let e = bash(std::env::temp_dir(), serde_json::json!({"command": "echo boom; exit 3"}), sig())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("boom\n\nWall time: "), "{e}");
        assert!(e.ends_with("\n\nCommand exited with code 3"), "{e}");
    }

    #[tokio::test]
    async fn empty_output_and_timeout() {
        let t = text_of(&bash(std::env::temp_dir(), serde_json::json!({"command": "true"}), sig()).await.unwrap());
        assert!(t.starts_with("(no output)\n\nWall time: "), "{t}");
        let e = bash(std::env::temp_dir(), serde_json::json!({"command": "sleep 5", "timeout": 1}), sig())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.ends_with("[Command timed out after 1 seconds]"), "{e}");
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
