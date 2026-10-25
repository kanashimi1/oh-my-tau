//! omp `resolveConfigValue`: a config value is either `!shell command` (stdout,
//! trimmed, cached on success), the name of an environment variable, or a literal.
//!
//! Command semantics follow `pi-coding-agent/src/config/resolve-config-value.ts`:
//! `/bin/sh -c` in the project directory, 10 s timeout, non-zero exit / timeout /
//! empty output = failure, a failure is not retried for 30 s, a success is cached
//! for the process lifetime.

use std::collections::HashMap;
use std::io::Read;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(10);
const FAILURE_RETRY: Duration = Duration::from_secs(30);
/// Grace for the stdout reader to finish after the process tree is killed.
/// A grandchild that escaped the process group (e.g. `setsid`) can hold the
/// pipe open forever; we abandon the read instead of deadlocking.
const READ_GRACE: Duration = Duration::from_secs(1);

#[derive(Default)]
struct State {
    ok: HashMap<String, String>,
    failed_until: HashMap<String, Instant>,
    /// Per-command in-flight slots (TS `commandInFlight`): concurrent resolves
    /// of the SAME command share one run; different commands run in parallel.
    in_flight: HashMap<String, Arc<InFlight>>,
}

#[derive(Default)]
struct InFlight {
    done: Mutex<bool>,
    cv: Condvar,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(Mutex::default);

/// Whether the raw config value is a `!command`.
#[must_use]
pub fn is_command(raw: &str) -> bool {
    raw.starts_with('!')
}

/// Resolve one config value. `!cmd` runs through the shell (cached); anything
/// else is an env var name when that variable is set and non-empty, else the literal.
/// `None` = empty value, failed command, or non-enterable working directory.
#[must_use]
pub fn resolve_config_value(raw: &str) -> Option<String> {
    if raw.is_empty() {
        return None;
    }
    if !is_command(raw) {
        return match std::env::var(raw) {
            Ok(v) if !v.is_empty() => Some(v),
            _ => Some(raw.to_string()),
        };
    }
    let command = raw[1..].trim().to_string();
    // Fast paths under a brief lock; never held across a run.
    {
        let st = STATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(v) = st.ok.get(&command) {
            return Some(v.clone());
        }
        if st
            .failed_until
            .get(&command)
            .is_some_and(|t| Instant::now() < *t)
        {
            return None;
        }
    }
    // Per-command in-flight dedup (TS `commandInFlight`): atomically claim the
    // slot; the thread whose slot is still registered runs the command, the
    // rest wait for it and read the cache.
    let (slot, is_leader) = {
        let mut st = STATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(v) = st.ok.get(&command) {
            return Some(v.clone());
        }
        match st.in_flight.get(&command) {
            Some(s) => (Arc::clone(s), false),
            None => {
                let s = Arc::new(InFlight::default());
                st.in_flight.insert(command.clone(), Arc::clone(&s));
                (s, true)
            }
        }
    };
    if !is_leader {
        let done = slot
            .done
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut done = done;
        while !*done {
            done = slot
                .cv
                .wait(done)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        let st = STATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        return st.ok.get(&command).cloned();
    }
    let result = run(&command, TIMEOUT);
    let mut st = STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.in_flight.remove(&command);
    if let Some(v) = &result {
        st.failed_until.remove(&command);
        st.ok.insert(command, v.clone());
    } else {
        st.failed_until
            .insert(command, Instant::now() + FAILURE_RETRY);
    }
    {
        let mut done = slot
            .done
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *done = true;
    }
    slot.cv.notify_all();
    result
}

/// [`resolve_config_value`] for async callers: `!command` values shell out for
/// up to 10 s, so they must run on the blocking pool instead of an executor thread.
pub async fn resolve_config_value_async(raw: &str) -> Option<String> {
    if !is_command(raw) {
        // Env var / literal lookup is cheap; no blocking work happens.
        return resolve_config_value(raw);
    }
    let raw = raw.to_string();
    tokio::task::spawn_blocking(move || resolve_config_value(&raw))
        .await
        .unwrap_or_default()
}

/// Drop the cache and failure backoff of one `!command` value (e.g. after a 401).
pub fn invalidate(raw: &str) {
    if !is_command(raw) {
        return;
    }
    let command = raw[1..].trim();
    let mut st = STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.ok.remove(command);
    st.failed_until.remove(command);
}

/// Mirror of TS `getProjectDir()` + `directoryIsEnterable` (resolve-config-value.ts:68-71):
/// run in the current project directory; `None` when it cannot be entered.
fn project_dir() -> Option<std::path::PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    std::fs::metadata(&cwd).ok()?.is_dir().then_some(cwd)
}

fn run(command: &str, timeout: Duration) -> Option<String> {
    let cwd = project_dir()?;
    let mut cmd = Command::new("/bin/sh");
    cmd.arg("-c")
        .arg(command)
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        // POSIX process group so `kill_group` can reach descendants.
        cmd.process_group(0);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        // TS logs a warning on errno failures (no command output leaked).
        Err(e) => {
            tracing::warn!(code = ?e.kind(), os_errno = e.raw_os_error(), "config: !command value resolution failed");
            return None;
        }
    };
    let mut out = child.stdout.take()?;
    // H1: read on a detached thread into a shared buffer. Once the direct
    // child has exited we return whatever was collected (TS `execSync`
    // semantics), so a leftover background descendant holding the pipe open
    // cannot turn a successful command into a cached failure.
    //
    // The pipe is non-blocking on Unix: a `setsid` descendant escapes the
    // process group, so `kill_group` cannot unblock the reader. The stop flag
    // ends the poll loop instead, guaranteeing the thread exits per attempt.
    // On non-Unix platforms the pipe stays blocking (no process-group escape
    // via `setsid` there); the reader thread simply drains until EOF.
    #[cfg(unix)]
    {
        let fd = out.as_raw_fd();
        // SAFETY: `fcntl` on a live fd owned by `out` for the rest of `run`.
        unsafe {
            libc::fcntl(fd, libc::F_SETFL, libc::O_NONBLOCK);
        }
    }
    let buf: Arc<Mutex<Vec<u8>>> = Arc::default();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let buf = Arc::clone(&buf);
        let stop = Arc::clone(&stop);
        std::thread::Builder::new()
            .name("config-cmd-reader".to_string())
            .spawn(move || {
                let mut chunk = [0u8; 8192];
                loop {
                    match out.read(&mut chunk) {
                        Ok(0) => break,
                        Ok(n) => buf
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .extend_from_slice(&chunk[..n]),
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(10));
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                        Err(_) => break,
                    }
                }
                let _ = done_tx.send(());
            })
            .ok()?;
    }
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                // Kill the whole group so descendants cannot keep the stdout pipe open.
                kill_group(&mut child);
                let _ = child.wait();
                break None;
            }
        }
    };
    // The direct child has exited. Give the reader READ_GRACE to see EOF; if a
    // descendant still holds the pipe, kill the group and stop the reader's
    // poll loop so the thread exits instead of leaking per attempt.
    if done_rx.recv_timeout(READ_GRACE).is_err() {
        kill_group(&mut child);
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = done_rx.recv_timeout(READ_GRACE);
    }
    let text = String::from_utf8_lossy(
        &buf.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
    .trim()
    .to_string();
    if !status?.success() || text.is_empty() {
        return None;
    }
    Some(text)
}

/// Kill the child's whole process group (and the child itself).
#[cfg(unix)]
fn kill_group(child: &mut std::process::Child) {
    let _ = Command::new("/bin/sh")
        .args(["-c", &format!("kill -KILL -{}", child.id())])
        .status();
    let _ = child.kill();
}

/// No process groups on this platform: kill the direct child only.
#[cfg(not(unix))]
fn kill_group(child: &mut std::process::Child) {
    let _ = child.kill();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_output_is_trimmed_and_cached() {
        let dir = tempfile::tempdir().unwrap();
        let counter = dir.path().join("n");
        let cmd = format!("!  echo x >> {p}; echo '  sekret  '", p = counter.display());
        assert_eq!(resolve_config_value(&cmd).as_deref(), Some("sekret"));
        assert_eq!(resolve_config_value(&cmd).as_deref(), Some("sekret"));
        assert_eq!(
            std::fs::read_to_string(&counter).unwrap().lines().count(),
            1
        );
        invalidate(&cmd);
        assert_eq!(resolve_config_value(&cmd).as_deref(), Some("sekret"));
        assert_eq!(
            std::fs::read_to_string(&counter).unwrap().lines().count(),
            2
        );
    }

    #[test]
    fn failing_empty_and_timeout_commands_are_none() {
        assert_eq!(resolve_config_value("!exit 3"), None);
        assert_eq!(resolve_config_value("!true"), None);
        assert_eq!(run("sleep 5", Duration::from_millis(100)), None);
    }

    /// A successful command whose background descendant keeps the stdout pipe
    /// open must still return the child's stdout (TS `execSync` semantics):
    /// previously the partial output was discarded, cached as failed for 30 s,
    /// and the blocked reader thread leaked on every attempt.
    #[test]
    fn background_descendant_does_not_turn_success_into_failure() {
        let started = Instant::now();
        // `sleep 30 &` stays in the process group holding the write end of the
        // stdout pipe after the direct child has exited 0.
        let got = run("echo done; sleep 30 &", Duration::from_secs(5));
        assert_eq!(got.as_deref(), Some("done"));
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "took {:?}",
            started.elapsed()
        );
    }

    /// A `setsid` grandchild escapes the process group and holds the stdout
    /// pipe open; the run must still return promptly with the child's output
    /// (the escaped reader is abandoned -- it cannot be killed via the group).
    #[test]
    fn escaped_grandchild_does_not_hang_success() {
        let started = Instant::now();
        assert_eq!(
            run(
                "setsid sleep 100 & echo started",
                Duration::from_millis(300)
            )
            .as_deref(),
            Some("started")
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "took {:?}",
            started.elapsed()
        );
    }

    /// Reader threads must not accumulate across runs whose pipe is held open
    /// by an escaped (`setsid`) descendant: `run` returns only after its
    /// reader thread has exited (ReviewWave3).
    #[test]
    fn escaped_pipe_holder_does_not_leak_reader_threads() {
        fn named_reader_threads() -> usize {
            std::fs::read_dir("/proc/self/task")
                .unwrap()
                .filter_map(|e| std::fs::read_to_string(e.ok()?.path().join("comm")).ok())
                .filter(|name| name.trim() == "config-cmd-reader")
                .count()
        }
        for _ in 0..3 {
            assert_eq!(
                run("setsid sleep 60 & echo started", Duration::from_millis(300)).as_deref(),
                Some("started")
            );
        }
        assert_eq!(named_reader_threads(), 0, "reader threads leaked");
    }

    /// L1: different commands run concurrently -- the global lock is never held
    /// across a run.
    #[test]
    fn different_commands_run_concurrently() {
        let started = Instant::now();
        let h1 = std::thread::spawn(|| resolve_config_value("!sleep 0.4; echo one"));
        let h2 = std::thread::spawn(|| resolve_config_value("!sleep 0.4; echo two"));
        assert_eq!(h1.join().unwrap().as_deref(), Some("one"));
        assert_eq!(h2.join().unwrap().as_deref(), Some("two"));
        assert!(
            started.elapsed() < Duration::from_millis(750),
            "took {:?}",
            started.elapsed()
        );
    }

    /// L1: concurrent resolves of the SAME command share one run (TS
    /// `commandInFlight`).
    #[test]
    fn same_command_concurrent_runs_once() {
        let dir = tempfile::tempdir().unwrap();
        let counter = dir.path().join("n");
        let cmd = format!(
            "! echo x >> {p}; sleep 0.3; echo once",
            p = counter.display()
        );
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let cmd = cmd.clone();
                std::thread::spawn(move || resolve_config_value(&cmd))
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap().as_deref(), Some("once"));
        }
        assert_eq!(
            std::fs::read_to_string(&counter).unwrap().lines().count(),
            1
        );
    }
}
