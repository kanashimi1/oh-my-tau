//! omp `resolveConfigValue`: a config value is either `!shell command` (stdout,
//! trimmed, cached on success), the name of an environment variable, or a literal.
//!
//! Command semantics follow `pi-coding-agent/src/config/resolve-config-value.ts`:
//! `/bin/sh -c` in the project directory, 10 s timeout, non-zero exit / timeout /
//! empty output = failure, a failure is not retried for 30 s, a success is cached
//! for the process lifetime.

use std::collections::HashMap;
use std::io::Read;
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
        let st = STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(v) = st.ok.get(&command) {
            return Some(v.clone());
        }
        if st.failed_until.get(&command).is_some_and(|t| Instant::now() < *t) {
            return None;
        }
    }
    // Per-command in-flight dedup (TS `commandInFlight`): atomically claim the
    // slot; the thread whose slot is still registered runs the command, the
    // rest wait for it and read the cache.
    let (slot, is_leader) = {
        let mut st = STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
        let done = slot.done.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut done = done;
        while !*done {
            done = slot.cv.wait(done).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        let st = STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        return st.ok.get(&command).cloned();
    }
    let result = run(&command, TIMEOUT);
    let mut st = STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    st.in_flight.remove(&command);
    if let Some(v) = &result {
        st.failed_until.remove(&command);
        st.ok.insert(command, v.clone());
    } else {
        st.failed_until.insert(command, Instant::now() + FAILURE_RETRY);
    }
    {
        let mut done = slot.done.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *done = true;
    }
    slot.cv.notify_all();
    result
}

/// Drop the cache and failure backoff of one `!command` value (e.g. after a 401).
pub fn invalidate(raw: &str) {
    if !is_command(raw) {
        return;
    }
    let command = raw[1..].trim();
    let mut st = STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
    let mut child = match Command::new("/bin/sh")
        .arg("-c")
        .arg(command)
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
    {
        Ok(c) => c,
        // TS logs a warning on errno failures (no command output leaked).
        Err(e) => {
            tracing::warn!(code = ?e.kind(), os_errno = e.raw_os_error(), "config: !command value resolution failed");
            return None;
        }
    };
    let mut out = child.stdout.take()?;
    // H1: read on a detached thread and receive with a bound; joining would
    // block forever if an escaped grandchild keeps the stdout pipe open.
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = out.read_to_end(&mut buf);
        let _ = tx.send(buf);
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                // Kill the whole group so descendants cannot keep the stdout pipe open.
                let _ = Command::new("/bin/sh")
                    .args(["-c", &format!("kill -KILL -{}", child.id())])
                    .status();
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };
    let buf = rx.recv_timeout(READ_GRACE).ok()?;
    if !status?.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&buf).trim().to_string();
    (!text.is_empty()).then_some(text)
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
        assert_eq!(std::fs::read_to_string(&counter).unwrap().lines().count(), 1);
        invalidate(&cmd);
        assert_eq!(resolve_config_value(&cmd).as_deref(), Some("sekret"));
        assert_eq!(std::fs::read_to_string(&counter).unwrap().lines().count(), 2);
    }

    #[test]
    fn failing_empty_and_timeout_commands_are_none() {
        assert_eq!(resolve_config_value("!exit 3"), None);
        assert_eq!(resolve_config_value("!true"), None);
        assert_eq!(run("sleep 5", Duration::from_millis(100)), None);
    }

    /// H1: a grandchild that escaped the process group (`setsid`) keeps the
    /// stdout pipe open; the timeout path must still return promptly.
    #[test]
    fn escaped_grandchild_does_not_hang_timeout() {
        let started = Instant::now();
        assert_eq!(run("setsid sleep 100 & echo started", Duration::from_millis(300)), None);
        assert!(started.elapsed() < Duration::from_secs(3), "took {:?}", started.elapsed());
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
        assert!(started.elapsed() < Duration::from_millis(750), "took {:?}", started.elapsed());
    }

    /// L1: concurrent resolves of the SAME command share one run (TS
    /// `commandInFlight`).
    #[test]
    fn same_command_concurrent_runs_once() {
        let dir = tempfile::tempdir().unwrap();
        let counter = dir.path().join("n");
        let cmd = format!("! echo x >> {p}; sleep 0.3; echo once", p = counter.display());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let cmd = cmd.clone();
                std::thread::spawn(move || resolve_config_value(&cmd))
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap().as_deref(), Some("once"));
        }
        assert_eq!(std::fs::read_to_string(&counter).unwrap().lines().count(), 1);
    }
}
