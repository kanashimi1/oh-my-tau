//! `bash`: one shell command (`bash -c`, stderr merged into stdout), omp's
//! result text: output, `Wall time: N.NN seconds`, `Command exited with code N`.
//! Non-persistent shell (omp embeds a brush-based shell; see the port report).
//!
//! omp's `async: true` backgrounds the command in the session job manager and
//! returns `formatBackgroundNotice`; the job is inspected via `read proc://`
//! (`tools/jobs.rs`). Services (`name`/`ready`) are supervised in-process
//! (omp's shared daemon broker folded into the session): the child is spawned
//! piped, readiness is observed from a log regex and/or a TCP port probe with
//! a timeout (default 30s), the captured log is inspectable via
//! `read proc://<name>`, and the child is killed when the session's job
//! registry is dropped (omp stops owned services at session end).

use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::types::AgentToolResult;
use serde_json::Value;
use tokio::io::AsyncReadExt;
use tokio::net::TcpStream;
use tokio::sync::oneshot;

use super::jobs::{self, JobRegistry, JobStatus, Service, ServiceState};
use super::util::{
    enforce_inline_byte_cap, notice_with_artifact, resolve_to_cwd, save_artifact, sink_window,
    str_arg, truncate_tail, SpillConfig, DEFAULT_MAX_BYTES,
};

const DEFAULT_TIMEOUT_SECS: f64 = 300.0;
const MIN_TIMEOUT_SECS: f64 = 1.0;
const MAX_TIMEOUT_SECS: f64 = 3600.0;
/// Bytes held in memory while the command runs (the rest is dropped from the middle).
const RETAIN_BYTES: usize = 8 * DEFAULT_MAX_BYTES;

/// omp `formatWallTimeNotice`.
#[must_use]
pub fn wall_time_notice(elapsed: Duration) -> String {
    format!("Wall time: {:.2} seconds", elapsed.as_secs_f64())
}

/// Bound over-long output the way omp's bash does: the `OutputSink` stream
/// window (middle elision past `DEFAULT_MAX_BYTES`, head per
/// `tools.artifactHeadBytes`), then `enforceInlineByteCap` at
/// `tools.artifactSpillThreshold` + slack. Anything cut is saved as an
/// artifact through `save` (returns its `artifact://` URL); returns the
/// inline body and, when only the stream window cut, its notice.
///
/// Past the spill threshold with a non-empty `query` (the request the call
/// serves), the inline body is instead the BM25-selected chunks within
/// `artifactHeadBytes + artifactTailBytes` (see [`super::relevance`]);
/// `OMP_TRUNC_BM25=0` keeps the head/tail path.
fn bound_output(
    output: &str,
    cfg: &SpillConfig,
    query: &str,
    save: impl Fn(&str) -> Option<String>,
) -> (String, Option<String>) {
    if crate::methodology::raw_tool_output_requested() {
        return (output.to_string(), None);
    }
    if output.len() > cfg.threshold && !query.trim().is_empty() && super::relevance::enabled() {
        let budget = match cfg.head_bytes + cfg.tail_bytes {
            0 => cfg.threshold,
            b => b.min(cfg.threshold),
        };
        if let Some(p) = super::relevance::pick(
            output,
            query,
            budget,
            None,
            (cfg.head_bytes, cfg.tail_bytes),
        ) {
            let url = save(output);
            return (
                p.content,
                Some(notice_with_artifact(&p.notice, url.as_deref())),
            );
        }
    }
    let (body, notice, mut artifact) = match sink_window(output, cfg) {
        Some(e) => {
            let url = save(output);
            let notice = notice_with_artifact(&e.notice, url.as_deref());
            (e.content, Some(notice), url)
        }
        None => (output.to_string(), None, None),
    };
    if body.len() <= cfg.inline_cap() {
        return (body, notice);
    }
    // The cap's marker and `[raw output: …]` footer replace the stream
    // notice, whose line ranges no longer describe the shown text.
    let capped = enforce_inline_byte_cap(&body, cfg.inline_cap(), || {
        artifact.take().or_else(|| save(output))
    });
    (capped, None)
}

fn timeout_arg(args: &Value) -> Option<f64> {
    match args.get("timeout").and_then(Value::as_f64) {
        None => Some(DEFAULT_TIMEOUT_SECS),
        Some(0.0) => None,
        Some(t) => Some(t.clamp(MIN_TIMEOUT_SECS, MAX_TIMEOUT_SECS)),
    }
}

/// Run a command and return `(output, exit code, wall time, outcome)`.
///
/// `bash` in its own session (TS: brush runs every external command under
/// `setsid`): the command has no controlling terminal, so nothing it starts
/// can open `/dev/tty` and flip the TUI's raw mode, steal the foreground
/// group, or be stopped by SIGTTIN. The workspace forbids `unsafe`
/// (`pre_exec`), so util-linux `setsid` does it; the spawned child is not a
/// group leader, so `setsid` execs bash in place (same pid, kill works).
fn detached_bash() -> tokio::process::Command {
    static SETSID: std::sync::LazyLock<Option<std::path::PathBuf>> =
        std::sync::LazyLock::new(|| {
            ["/usr/bin/setsid", "/bin/setsid"]
                .into_iter()
                .map(std::path::PathBuf::from)
                .find(|p| p.is_file())
        });
    match SETSID.as_deref() {
        Some(setsid) => {
            let mut cmd = tokio::process::Command::new(setsid);
            cmd.arg("--wait").arg("bash");
            cmd
        }
        None => tokio::process::Command::new("bash"),
    }
}

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
    let mut child = detached_bash()
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
    Ok((
        String::from_utf8_lossy(&buf).into_owned(),
        code,
        started.elapsed(),
        outcome,
    ))
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
pub async fn execute(
    cwd: &std::path::Path,
    command: &str,
    signal: &AbortSignal,
) -> Result<Executed> {
    let (raw, code, _, outcome) =
        run(cwd, command, Some(DEFAULT_TIMEOUT_SECS), signal, None).await?;
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
fn finished_text(
    output: &str,
    elapsed: Duration,
    outcome: Option<&str>,
    timeout: Option<f64>,
    code: Option<i32>,
    cfg: &SpillConfig,
    query: &str,
) -> String {
    use std::fmt::Write as _;
    let (shown, elision) = bound_output(output.trim_end_matches('\n'), cfg, query, save_artifact);
    let body = if shown.is_empty() {
        "(no output)".to_string()
    } else {
        shown
    };
    let mut text = format!("{body}\n\n{}", wall_time_notice(elapsed));
    if let Some(n) = elision {
        let _ = write!(text, "\n\n[{n}]");
    }
    match outcome {
        Some("timeout") => {
            let secs = timeout.unwrap_or(0.0);
            let _ = write!(text, "\n\n[Command timed out after {secs} seconds]");
        }
        Some(_) => text.push_str("\n\n[Command aborted]"),
        None => {}
    }
    if outcome.is_none() {
        if let Some(c) = code {
            if c != 0 {
                let _ = write!(text, "\n\nCommand exited with code {c}");
            }
        } else {
            text.push_str("\n\nCommand failed: missing exit status");
        }
    }
    text
}

/// omp `ServiceReady` after `normalizeReady`: blank strings are dropped and a
/// spec with no surviving condition is `None` entirely.
#[derive(Default, Clone)]
struct ReadySpec {
    log: Option<String>,
    host: Option<String>,
    port: Option<u16>,
    timeout: Option<f64>,
}

fn blank(v: Option<&str>) -> Option<String> {
    v.map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
}

/// Parse `args["ready"]` (omp `normalizeReady`).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // port fraction already validated 1..=65535 before the cast
fn parse_ready(args: &Value) -> Option<ReadySpec> {
    let r = &args["ready"];
    if !r.is_object() {
        return None;
    }
    let raw_port = r["port"].as_f64().filter(|p| p.is_finite());
    let port = raw_port
        .filter(|p| p.fract() == 0.0 && *p >= 1.0 && *p <= 65_535.0)
        .map(|p| p as u16);
    let spec = ReadySpec {
        log: blank(r["log"].as_str()),
        host: blank(r["host"].as_str()),
        port,
        timeout: r["timeout"].as_f64().filter(|t| t.is_finite()),
    };
    if spec.log.is_none() && spec.host.is_none() && spec.port.is_none() && spec.timeout.is_none() {
        return None;
    }
    Some(spec)
}

/// Whether `ready` carries any service-only field (omp `normalizeReady`'s
/// "is this a service request" test, used for the no-name notice).
fn ready_given(args: &Value) -> bool {
    parse_ready(args).is_some()
}

/// omp `startService` validation; `bail!`s with omp's `ToolError` texts.
fn validate_service(
    name: &str,
    ready: Option<&ReadySpec>,
    args: &Value,
    async_requested: bool,
) -> Result<()> {
    if !valid_service_name(name) {
        bail!("Service name must be 1-48 letters, numbers, dots, underscores, or hyphens");
    }
    if async_requested || args.get("timeout").is_some() {
        bail!("Service mode does not accept async or timeout; use ready.timeout for readiness.");
    }
    if let Some(ready) = ready {
        if let Some(port) = args["ready"]["port"].as_f64() {
            if ready.port.is_none() || port.fract() != 0.0 {
                bail!("ready.port must be an integer from 1 to 65535");
            }
        }
        if ready.log.is_none() && ready.port.is_none() {
            bail!("ready requires log or port");
        }
        if let Some(log) = &ready.log {
            if let Err(e) = regex::Regex::new(log) {
                bail!("Invalid readiness regex: {e}");
            }
        }
    }
    Ok(())
}

/// omp's service-name rule: `^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$`.
fn valid_service_name(name: &str) -> bool {
    let b = name.as_bytes();
    b.len() <= 48
        && b.first().is_some_and(u8::is_ascii_alphanumeric)
        && b.iter()
            .all(|c| c.is_ascii_alphanumeric() || matches!(*c, b'.' | b'_' | b'-'))
}

/// Run the `bash` tool.
///
/// # Errors
/// Spawn failure, timeout, abort or a non-zero exit (carrying the output);
/// disabled service mode (omp's error text).
pub async fn bash(
    env: std::sync::Arc<super::Env>,
    args: Value,
    signal: AbortSignal,
) -> Result<AgentToolResult> {
    let cwd = env.cwd.clone();
    let command = str_arg(&args, "command")?.to_string();
    let dir = args
        .get("cwd")
        .and_then(Value::as_str)
        .map_or_else(|| cwd.clone(), |d| resolve_to_cwd(d, &cwd));
    let name = args
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|n| !n.is_empty());
    let async_requested = args.get("async").and_then(Value::as_bool) == Some(true);
    let timeout = timeout_arg(&args);
    let mut notices: Vec<String> = Vec::new();
    if let Some(name) = name {
        let ready = parse_ready(&args);
        validate_service(name, ready.as_ref(), &args, async_requested)?;
        let timeout_secs = ready
            .as_ref()
            .and_then(|r| r.timeout)
            .unwrap_or(30.0)
            .clamp(0.05, 3_600.0);
        let outcome = start_service(
            &env.jobs,
            name,
            &dir,
            &command,
            ready.as_ref(),
            timeout_secs,
            &signal,
        )
        .await?;
        let mut lines = vec![format!(
            "{name}: {}{}{}",
            outcome.state.as_str(),
            outcome.pid.map(|p| format!(" pid={p}")).unwrap_or_default(),
            if outcome.ready { " ready" } else { "" }
        )];
        if outcome.timed_out {
            lines.push(format!(
                "NOT ready — readiness timed out after {timeout_secs}s; process state: {}.",
                outcome.state.as_str()
            ));
        }
        if let Some(reason) = &outcome.exit_reason {
            lines.push(format!("Reason: {reason}"));
        }
        if !outcome.log.is_empty() {
            lines.push(outcome.log);
        }
        return Ok(AgentToolResult::text(lines.join("\n")));
    }
    if ready_given(&args) {
        // Nothing can honour ready without a service to attach it to (omp).
        notices.push("Ignored ready: service-only, and no service name was given.".to_string());
    }
    let spill = SpillConfig::load(&cwd);
    let query = env.query();
    if async_requested {
        return Ok(start_background(
            &env.jobs, dir, command, timeout, notices, spill, query, &signal,
        ));
    }
    let (output, code, elapsed, outcome) = run(&dir, &command, timeout, &signal, None).await?;
    let mut text = finished_text(&output, elapsed, outcome, timeout, code, &spill, &query);
    if !notices.is_empty() {
        text = format!("{}\n{text}", notices.join("\n"));
    }
    if outcome.is_some() || code != Some(0) {
        bail!("{text}");
    }
    let mut result = AgentToolResult::text(text);
    // Transcript card facts (`⟦Wall: … | Timeout: …⟧`); failures keep the
    // error-text path, which carries no details.
    result.details = run_details(timeout, &args, elapsed);
    Ok(result)
}

/// omp `BashToolDetails` for a finished foreground run.
fn run_details(timeout: Option<f64>, args: &Value, elapsed: Duration) -> Value {
    let mut d = serde_json::Map::new();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    // ms fit u64; whole seconds fit u64
    {
        d.insert(
            "wallTimeMs".into(),
            Value::from(elapsed.as_millis().min(u128::from(u64::MAX)) as u64),
        );
        match timeout {
            Some(t) => {
                d.insert("timeoutSeconds".into(), Value::from(t.round() as u64));
            }
            None => {
                d.insert("timeoutDisabled".into(), Value::Bool(true));
            }
        }
        if let (Some(req), Some(t)) = (args.get("timeout").and_then(Value::as_f64), timeout) {
            if req != 0.0 && (req - t).abs() > f64::EPSILON {
                d.insert(
                    "requestedTimeoutSeconds".into(),
                    Value::from(req.round() as u64),
                );
            }
        }
    }
    Value::Object(d)
}

/// What the bash tool reports after a service start attempt.
struct ServiceOutcome {
    state: ServiceState,
    pid: Option<u32>,
    ready: bool,
    timed_out: bool,
    exit_reason: Option<String>,
    log: String,
}

/// Readiness buffer window (omp broker `READINESS_BUFFER_CHARS`).
const READINESS_BUFFER_CHARS: usize = 64 * 1024;
/// Supervisor liveness tick: checks kill requests and registry liveness.
const SUPERVISOR_TICK: Duration = Duration::from_millis(100);

/// omp `startService`, in-proc: spawn the child piped, supervise it in a
/// background task, and wait for readiness (log regex and/or TCP port probe)
/// up to `timeout_secs`. Both conditions must hold when both are given (omp
/// `#markReady`). Returns once ready, once the process settles, or on timeout
/// (the service keeps running; `timed_out` is set, omp's `readyTimedOut`).
///
/// The supervisor owns the child and keeps draining its output after this
/// call returns (a full pipe would otherwise block the service); it kills the
/// child when the job registry is dropped (session end) or `kill_requested`
/// is set (`proc://<name>/kill`, tool-call abort).
///
/// # Errors
/// Spawn failure, or abort while waiting.
#[allow(clippy::too_many_lines)] // one supervisor task: spawn + readiness poll + liveness loop; mirrors omp's service supervisor
async fn start_service(
    registry: &JobRegistry,
    name: &str,
    dir: &std::path::Path,
    command: &str,
    ready: Option<&ReadySpec>,
    timeout_secs: f64,
    signal: &AbortSignal,
) -> Result<ServiceOutcome> {
    let script = format!("exec 2>&1\n{command}");
    let mut child = detached_bash()
        .arg("-c")
        .arg(&script)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("failed to spawn bash in {}", dir.display()))?;
    let pid = child.id();
    let mut stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let pattern = ready
        .and_then(|r| r.log.as_ref())
        .map(|p| regex::Regex::new(p).expect("readiness regex validated"));
    let host = ready
        .and_then(|r| r.host.clone())
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port = ready.and_then(|r| r.port);
    let has_ready = ready.is_some();

    let service = Service {
        name: name.to_string(),
        command: command.to_string(),
        started: Instant::now(),
        state: if has_ready {
            ServiceState::Starting
        } else {
            ServiceState::Running
        },
        pid,
        ready_at: None,
        exited_at: None,
        exit_code: None,
        exit_reason: None,
        output: Arc::new(Mutex::new(String::new())),
        kill_requested: false,
    };
    let service = registry.insert_service(service);
    let inner = registry.inner.clone();
    let registry_liveness: std::sync::Weak<std::sync::Mutex<super::jobs::JobsInner>> =
        Arc::downgrade(&inner);

    // Supervisor: drains output (log capture + readiness buffer), polls the
    // port, watches for kill requests and registry death, settles on exit.
    let (notify_tx, mut notify_rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        let mut child = child;
        let mut notify = Some(notify_tx);
        let mut log_ready = pattern.is_none();
        let mut port_ready = port.is_none();
        let mut readiness_buffer = String::new();
        let mut chunk = [0u8; 8192];
        let mut tick = tokio::time::interval(SUPERVISOR_TICK);
        tick.tick().await; // first tick is immediate; skip it.
        loop {
            tokio::select! {
                n = stdout.read(&mut chunk) => {
                    match n {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            let text = String::from_utf8_lossy(&chunk[..n]).into_owned();
                            {
                                let svc = service.lock().expect("service poisoned");
                                let mut out = svc.output.lock().expect("service output poisoned");
                                out.push_str(&text);
                                if out.len() > RETAIN_BYTES {
                                    let cut = out.len() - RETAIN_BYTES / 2;
                                    out.drain(..cut);
                                }
                            }
                            if !log_ready {
                                readiness_buffer.push_str(&text);
                                if readiness_buffer.len() > READINESS_BUFFER_CHARS {
                                    readiness_buffer.drain(..readiness_buffer.len() - READINESS_BUFFER_CHARS);
                                }
                                if pattern.as_ref().is_some_and(|re| re.is_match(&readiness_buffer)) {
                                    log_ready = true;
                                }
                            }
                        }
                    }
                }
                _ = tick.tick() => {
                    {
                        let svc = service.lock().expect("service poisoned");
                        if svc.kill_requested || registry_liveness.upgrade().is_none() {
                            let _ = child.start_kill();
                        }
                    }
                    if !port_ready {
                        let Some(port) = port else { continue };
                        let addr = format!("{host}:{port}");
                        if TcpStream::connect(&addr).await.is_ok() {
                            port_ready = true;
                        }
                    }
                }
            }
            let ready_now = log_ready && port_ready;
            if ready_now {
                let mut svc = service.lock().expect("service poisoned");
                if svc.state == ServiceState::Starting {
                    svc.state = ServiceState::Ready;
                    svc.ready_at = Some(Instant::now());
                }
                if let Some(tx) = notify.take() {
                    let _ = tx.send(());
                }
            }
        }
        // EOF or a fatal read error: settle from the child's exit status.
        let status = child.wait().await;
        let mut svc = service.lock().expect("service poisoned");
        svc.exited_at = Some(Instant::now());
        svc.pid = None;
        match status {
            Ok(st) => {
                svc.exit_code = st.code();
                svc.state = if st.success() {
                    ServiceState::Exited
                } else {
                    ServiceState::Failed
                };
                if let Some(code) = st.code() {
                    if code != 0 {
                        svc.exit_reason = Some(format!("exited with code {code}"));
                    }
                } else {
                    svc.exit_reason = Some("killed by signal".to_string());
                }
            }
            Err(e) => {
                svc.state = ServiceState::Failed;
                svc.exit_reason = Some(e.to_string());
            }
        }
        if let Some(tx) = notify.take() {
            let _ = tx.send(());
        }
    });

    // Wait for readiness or terminal settlement, the timeout, or abort.
    let outcome = tokio::select! {
        _ = &mut notify_rx => false,
        () = tokio::time::sleep(Duration::from_secs_f64(timeout_secs)) => true,
        () = signal.aborted() => {
            if let Some(svc) = registry.service(name) {
                svc.lock().expect("service poisoned").kill_requested = true;
            }
            bail!("Command aborted");
        }
    };
    let svc_arc = registry.service(name).expect("service registered");
    let svc = svc_arc.lock().expect("service poisoned");
    Ok(ServiceOutcome {
        state: svc.state,
        pid: svc.pid,
        ready: svc.is_ready(),
        timed_out: outcome,
        exit_reason: svc.exit_reason.clone(),
        log: svc.log(),
    })
}

/// omp `BashTool` async mode: register a `bg_N` job, run it detached, and
/// return `formatBackgroundNotice` (the result is read later via `proc://`).
#[allow(clippy::too_many_arguments)] // job plumbing: registry, command, limits, spill policy, request query
fn start_background(
    registry: &JobRegistry,
    dir: std::path::PathBuf,
    command: String,
    timeout: Option<f64>,
    notices: Vec<String>,
    spill: SpillConfig,
    query: String,
    _signal: &AbortSignal,
) -> AgentToolResult {
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
                let text =
                    finished_text(output, *elapsed, *outcome, timeout, *code, &spill, &query);
                let status = if outcome.is_none() && *code == Some(0) {
                    JobStatus::Completed
                } else {
                    JobStatus::Failed
                };
                let error = (status == JobStatus::Failed).then(|| {
                    if *outcome == Some("timeout") {
                        format!("Command timed out after {} seconds", timeout.unwrap_or(0.0))
                    } else if *outcome == Some("aborted") {
                        "[Command aborted]".to_string()
                    } else {
                        format!(
                            "Command exited with code {}",
                            code.map_or("?".to_string(), |c| c.to_string())
                        )
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
    AgentToolResult::text(lines.join("\n"))
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
                pa_agent::types::ToolResultContent::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn test_env() -> std::sync::Arc<super::super::Env> {
        std::sync::Arc::new(super::super::Env::new(&std::env::temp_dir(), false))
    }

    /// Regression (hung TUI menus): a tool command must not reach the TUI's
    /// controlling terminal — it runs in its own session, so `/dev/tty` is
    /// unavailable and `stty`/interop programs cannot drop raw mode.
    #[tokio::test]
    async fn commands_have_no_controlling_terminal() {
        let r = bash(
            test_env(),
            serde_json::json!({"command": "if (: </dev/tty) 2>/dev/null; then echo HAS_TTY; else echo NO_TTY; fi; ps -o sid= -p $$ | tr -d ' '; echo $$"}),
            sig(),
        )
        .await
        .unwrap();
        let t = text_of(&r);
        assert!(t.contains("NO_TTY"), "{t}");
        let mut lines = t.lines().skip(1);
        let (sid, pid) = (
            lines.next().unwrap_or_default(),
            lines.next().unwrap_or_default(),
        );
        assert_eq!(sid, pid, "bash is its own session leader: {t}");
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
    async fn success_carries_wall_and_timeout_details_for_the_card() {
        let env = test_env();
        let r = bash(env.clone(), serde_json::json!({"command": "true"}), sig())
            .await
            .unwrap();
        assert!(r.details["wallTimeMs"].is_u64(), "{:?}", r.details);
        assert_eq!(r.details["timeoutSeconds"], 300);
        let r = bash(
            env,
            serde_json::json!({"command": "true", "timeout": 0}),
            sig(),
        )
        .await
        .unwrap();
        assert_eq!(r.details["timeoutDisabled"], true);
    }

    #[tokio::test]
    async fn nonzero_exit_is_an_error_with_output_and_code() {
        let env = test_env();
        let e = bash(
            env.clone(),
            serde_json::json!({"command": "echo boom; exit 3"}),
            sig(),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(e.starts_with("boom\n\nWall time: "), "{e}");
        assert!(e.ends_with("\n\nCommand exited with code 3"), "{e}");
    }

    #[tokio::test]
    async fn empty_output_and_timeout() {
        let env = test_env();
        let t = text_of(
            &bash(env.clone(), serde_json::json!({"command": "true"}), sig())
                .await
                .unwrap(),
        );
        assert!(t.starts_with("(no output)\n\nWall time: "), "{t}");
        let e = bash(
            env.clone(),
            serde_json::json!({"command": "sleep 5", "timeout": 1}),
            sig(),
        )
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
        assert!(
            t.starts_with("Backgrounded as job bg_1 (deadline 300s)"),
            "{t}"
        );

        // proc:// lists the running job.
        let listing = super::super::jobs::read_proc(&env, "").unwrap();
        let lt = text_of(&listing);
        assert!(lt.contains("bg_1 [bash] running up "), "{lt}");
        assert!(
            lt.contains("echo first; sleep 0.2; echo done-marker"),
            "{lt}"
        );

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
        assert!(
            dt.starts_with("bg_1 [bash] — completed — echo first"),
            "{dt}"
        );
        assert!(dt.contains("first\ndone-marker\n\nWall time: "), "{dt}");
        assert!(dt.ends_with(" seconds"), "{dt}");

        let after = super::super::jobs::read_proc(&env, "").unwrap();
        assert!(
            text_of(&after).contains("bg_1 [bash] completed in "),
            "{}",
            text_of(&after)
        );
    }

    #[tokio::test]
    async fn service_ready_by_log_regex() {
        let env = test_env();
        let r = bash(
            env.clone(),
            serde_json::json!({
                "command": "echo boot-banner; echo service-up-now; sleep 30",
                "name": "svc-log",
                "ready": {"log": "service-up-now"}
            }),
            sig(),
        )
        .await
        .unwrap();
        let t = text_of(&r);
        assert!(t.starts_with("svc-log: ready pid="), "{t}");
        assert!(t.contains(" ready\n"), "{t}");
        assert!(t.contains("service-up-now"), "{t}");
        // proc:// lists the service as ready.
        let listing = super::super::jobs::read_proc(&env, "").unwrap();
        let lt = text_of(&listing);
        assert!(lt.contains("svc-log [service] — ready — up "), "{lt}");
        // proc://<name> shows the captured log.
        let detail = super::super::jobs::read_proc(&env, "svc-log").unwrap();
        assert!(
            text_of(&detail).contains("boot-banner"),
            "{}",
            text_of(&detail)
        );
    }

    #[tokio::test]
    async fn service_ready_by_port() {
        let env = test_env();
        // A listening socket makes the TCP probe succeed (omp `#pollPort`).
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let r = bash(
            env.clone(),
            serde_json::json!({
                "command": "sleep 30",
                "name": "svc-port",
                "ready": {"port": port}
            }),
            sig(),
        )
        .await
        .unwrap();
        let t = text_of(&r);
        assert!(t.starts_with("svc-port: ready pid="), "{t}");
    }

    #[tokio::test]
    async fn service_readiness_timeout_reports_not_ready() {
        let env = test_env();
        let r = bash(
            env.clone(),
            serde_json::json!({
                "command": "sleep 30",
                "name": "svc-slow",
                "ready": {"log": "a-marker-that-never-appears", "timeout": 0.2}
            }),
            sig(),
        )
        .await
        .unwrap();
        let t = text_of(&r);
        assert!(t.starts_with("svc-slow: starting pid="), "{t}");
        assert!(!t.contains(" ready\n"), "{t}");
        assert!(
            t.contains("NOT ready — readiness timed out after 0.2s; process state: starting."),
            "{t}"
        );
        // The service keeps running after the timeout (omp returns readyTimedOut).
        let svc = env.jobs.service("svc-slow").unwrap();
        assert_eq!(
            svc.lock().unwrap().state,
            super::super::jobs::ServiceState::Starting
        );
        assert!(svc.lock().unwrap().pid.is_some());
    }

    #[tokio::test]
    async fn service_validation_matches_omp_errors() {
        let env = test_env();
        let err = bash(
            env.clone(),
            serde_json::json!({"command": "true", "name": "-bad", "ready": {"log": "x"}}),
            sig(),
        )
        .await
        .unwrap_err()
        .to_string();
        assert_eq!(
            err,
            "Service name must be 1-48 letters, numbers, dots, underscores, or hyphens"
        );
        let err = bash(
            env.clone(),
            serde_json::json!({"command": "true", "name": "svc", "timeout": 10}),
            sig(),
        )
        .await
        .unwrap_err()
        .to_string();
        assert_eq!(
            err,
            "Service mode does not accept async or timeout; use ready.timeout for readiness."
        );
        let err = bash(
            env.clone(),
            serde_json::json!({"command": "true", "name": "svc", "ready": {"log": "x", "port": 70000}}),
            sig(),
        )
        .await
        .unwrap_err()
        .to_string();
        assert_eq!(err, "ready.port must be an integer from 1 to 65535");
        let err = bash(
            env.clone(),
            serde_json::json!({"command": "true", "name": "svc", "ready": {"host": "127.0.0.1"}}),
            sig(),
        )
        .await
        .unwrap_err()
        .to_string();
        assert_eq!(err, "ready requires log or port");
        let err = bash(
            env.clone(),
            serde_json::json!({"command": "true", "name": "svc", "ready": {"log": "(unclosed"}}),
            sig(),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.starts_with("Invalid readiness regex: "), "{err}");
    }

    #[tokio::test]
    async fn service_exit_settles_and_captures_state() {
        let env = test_env();
        // Without a ready spec the tool returns immediately in "running" state
        // (omp start does not wait); the supervisor settles the exit afterwards.
        let r = bash(
            env.clone(),
            serde_json::json!({"command": "echo brief; exit 3", "name": "svc-exit"}),
            sig(),
        )
        .await
        .unwrap();
        let t = text_of(&r);
        // The supervisor may settle the fast exit before the tool returns
        // (omp's start RPC has the same race); both snapshots are valid.
        let running = t.starts_with("svc-exit: running pid=");
        let failed = t.starts_with("svc-exit: failed");
        assert!(running || failed, "{t}");
        assert!(!t.contains(" ready\n"), "{t}");
        if failed {
            assert!(t.contains("Reason: exited with code 3"), "{t}");
        }
        assert!(t.contains("brief"), "{t}");
        // proc://<name> settles to "failed" once the child exits.
        let mut settled = String::new();
        for _ in 0..50 {
            let detail = super::super::jobs::read_proc(&env, "svc-exit").unwrap();
            settled = text_of(&detail);
            if settled.contains("svc-exit [service] — failed — up ") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(
            settled.contains("svc-exit [service] — failed — up "),
            "{settled}"
        );
        assert!(settled.contains("brief"), "{settled}");
    }

    #[tokio::test]
    async fn service_ready_wait_observes_pre_ready_exit() {
        // The readiness wait must wake on a terminal state instead of blocking
        // for the full timeout (omp broker wake condition).
        let env = test_env();
        let r = bash(
            env.clone(),
            serde_json::json!({
                "command": "echo partial; exit 1",
                "name": "svc-die",
                "ready": {"log": "never-appears", "timeout": 30}
            }),
            sig(),
        )
        .await
        .unwrap();
        let t = text_of(&r);
        assert!(t.starts_with("svc-die: failed"), "{t}");
        assert!(!t.contains("NOT ready"), "{t}");
        assert!(t.contains("Reason: exited with code 1"), "{t}");
        assert!(t.contains("partial"), "{t}");
    }

    #[tokio::test]
    async fn service_child_is_killed_when_registry_drops() {
        let pid;
        {
            let env = test_env();
            bash(
                env.clone(),
                serde_json::json!({"command": "sleep 60", "name": "svc-drop"}),
                sig(),
            )
            .await
            .unwrap();
            pid = env
                .jobs
                .service("svc-drop")
                .unwrap()
                .lock()
                .unwrap()
                .pid
                .unwrap();
            assert!(std::path::Path::new(&format!("/proc/{pid}")).exists());
        } // registry dropped: the supervisor must kill the child.
        let mut gone = false;
        for _ in 0..100 {
            if !std::path::Path::new(&format!("/proc/{pid}")).exists() {
                gone = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(gone, "service child {pid} survived registry drop");
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
        assert!(
            t.starts_with(
                "Ignored ready: service-only, and no service name was given.\nhi\n\nWall time: "
            ),
            "{t}"
        );
    }

    #[test]
    fn long_output_is_elided_in_the_middle() {
        let mut long = String::new();
        for i in 1..=5000 {
            use std::fmt::Write as _;
            let _ = writeln!(long, "line number {i}");
        }
        let (shown, notice) = bound_output(long.trim_end(), &SpillConfig::default(), "", |_| None);
        assert!(shown.len() < long.len());
        assert!(shown.starts_with("line number 1\n"));
        assert!(shown.ends_with("line number 5000"));
        assert!(shown.contains("ln elided…]"));
        assert!(notice.unwrap().contains("middle lines"));
    }

    /// `tools.artifactSpillThreshold: 8` (project settings layer) caps a
    /// 100 KB command output near 8 KB (head + tail around a marker) and
    /// points at the artifact holding the full output. The artifact sink is
    /// injected, so the test never touches the process-global session file.
    #[test]
    fn lowered_spill_threshold_caps_bash_output_with_artifact_pointer() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".omp")).unwrap();
        // Project layer pins every key so the user's global config cannot leak in.
        std::fs::write(
            dir.path().join(".omp/config.yml"),
            "tools:\n  artifactSpillThreshold: 8\n  artifactHeadBytes: 20\n  artifactTailBytes: 20\n  artifactTailLines: 500\n",
        )
        .unwrap();
        let cfg = SpillConfig::load(dir.path());
        assert_eq!(cfg.threshold, 8 * 1024);
        // 1000 lines of 99 digits + newline = 100 000 bytes.
        let cmd = "for i in $(seq 1 1000); do printf '%099d\\n' \"$i\"; done";
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (output, code, _, outcome) = rt
            .block_on(run(dir.path(), cmd, Some(60.0), &sig(), None))
            .unwrap();
        assert_eq!((code, outcome), (Some(0), None));
        let saved = std::cell::RefCell::new(Vec::<String>::new());
        let save = |text: &str| {
            let mut saved = saved.borrow_mut();
            saved.push(text.to_string());
            Some(format!("artifact://{}", saved.len() - 1))
        };
        let (body, notice) = bound_output(output.trim_end_matches('\n'), &cfg, "", save);
        assert!(notice.is_none(), "{notice:?}");
        assert!(body.len() < 9 * 1024, "inline body is {} bytes", body.len());
        assert!(body.starts_with(&format!("{:099}\n", 1)), "{body}");
        assert!(body.contains("B elided…]\n"), "{body}");
        let shown = body
            .strip_suffix("[raw output: artifact://0]")
            .expect("artifact pointer footer");
        assert!(
            shown.trim_end().ends_with(&format!("{:099}", 1000)),
            "{body}"
        );
        let saved = saved.into_inner();
        assert_eq!(saved.len(), 1, "one artifact for the whole output");
        assert_eq!(
            saved[0].len(),
            99_999,
            "artifact holds the full (trimmed) output"
        );
    }

    /// A 100 KB output whose only relevant line sits in the middle: the
    /// relevance view keeps it when the request names it (and points at the
    /// artifact); plain head/tail (no request) drops it.
    #[test]
    fn relevance_view_keeps_mid_output_line_named_by_the_request() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".omp")).unwrap();
        std::fs::write(
            dir.path().join(".omp/config.yml"),
            "tools:\n  artifactSpillThreshold: 12\n  artifactHeadBytes: 3\n  artifactTailBytes: 6\n  artifactTailLines: 150\n",
        )
        .unwrap();
        let marker = "deploy failed: QuorumLostError on shard_alpha7";
        // 1000 lines of 99 digits + newline = 100 000 bytes, marker at line 500.
        let cmd = format!(
            "for i in $(seq 1 1000); do if [ $i -eq 500 ]; then echo '{marker}'; else printf '%099d\\n' \"$i\"; fi; done"
        );
        let env = std::sync::Arc::new(super::super::Env::new(dir.path(), false));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let run_with = |query: &str| {
            *env.query.lock() = query.to_string();
            text_of(
                &rt.block_on(bash(
                    env.clone(),
                    serde_json::json!({"command": cmd}),
                    sig(),
                ))
                .unwrap(),
            )
        };
        let ranked =
            run_with("why did the deploy hit QuorumLostError?\nChecking the shard_alpha7 log.");
        assert!(ranked.contains(marker), "{ranked}");
        assert!(ranked.contains(" lines omitted] …"), "{ranked}");
        assert!(ranked.len() < 12 * 1024, "inline {} bytes", ranked.len());
        let plain = run_with("");
        assert!(!plain.contains(marker), "head/tail keeps only the edges");
        assert!(plain.contains("B elided…]"), "{plain}");
        // The relevance view always carries the artifact pointer.
        let output = rt
            .block_on(run(dir.path(), &cmd, Some(60.0), &sig(), None))
            .unwrap()
            .0;
        let cfg = SpillConfig::load(dir.path());
        let (body, notice) = bound_output(
            output.trim_end_matches('\n'),
            &cfg,
            "QuorumLostError",
            |_| Some("artifact://4".to_string()),
        );
        assert!(body.contains(marker));
        let notice = notice.unwrap();
        assert!(
            notice.ends_with("Read artifact://4 for full output"),
            "{notice}"
        );
    }
}
