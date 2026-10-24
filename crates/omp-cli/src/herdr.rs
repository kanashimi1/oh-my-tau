//! Native Herdr agent-state reporting (Herdr >= 0.9.2 agent integration,
//! docs: add-herdr-support.mdx). Active only inside Herdr panes
//! (`HERDR_ENV=1`); everywhere else every entry point is a no-op, so wire
//! behavior outside Herdr is byte-identical.
//!
//! Three report kinds, all through `"$HERDR_BIN_PATH"` in the background with
//! a short timeout and ignored failures, per the "keep it out of the way"
//! rules:
//!
//! 1. state (`idle`/`working`/`blocked`) via `pane report-agent`
//! 2. session identity + resume command via `pane report-agent-session`
//! 3. `pane release-agent` when the agent exits
//!
//! `--seq` increases monotonically (millisecond timestamp base + counter, like
//! the TS integration's `Date.now() * 1000`), so stale reports arriving late
//! never overwrite newer state. Only the latest queued state is sent: while a
//! report is in flight, older pending states are dropped.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Stable, unique integration id (never `herdr:*` — that prefix is Herdr's own).
pub const SOURCE: &str = "omp-rs";
/// The label users see in the sidebar / `herdr agent list`.
pub const AGENT: &str = "omt";

/// Per-report wall-clock budget. Herdr's socket answers in milliseconds; if
/// the binary hangs, the child is killed after this and the report is dropped.
const REPORT_TIMEOUT: Duration = Duration::from_millis(2000);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Idle,
    Working,
    Blocked,
}

impl State {
    fn as_str(self) -> &'static str {
        match self {
            State::Idle => "idle",
            State::Working => "working",
            State::Blocked => "blocked",
        }
    }
}

/// One report to deliver; `seq` is fixed at enqueue time so ordering holds
/// even if delivery of a later report overtakes an earlier one.
#[derive(Clone, Debug, PartialEq)]
enum Pending {
    State {
        state: State,
        message: Option<String>,
        seq: u64,
    },
    Session {
        start_source: String,
        seq: u64,
    },
    Release {
        seq: u64,
    },
}

pub struct Herdr {
    bin: PathBuf,
    pane: String,
    seq: AtomicU64,
    /// Resume argv reported after `--`: `omp-rs --resume <session-file> --model <provider/id>`.
    resume: Mutex<Vec<String>>,
    /// Current session file, sent as `--agent-session-path`.
    session_path: Mutex<Option<String>>,
    /// Last reported (state, message) for dedupe.
    last: Mutex<(Option<State>, Option<String>)>,
    /// Queued reports, drained in order by the worker. A new report replaces
    /// only the last queued one of the same kind (so a pending session
    /// identity is never clobbered by a state change, per the docs' "report
    /// it again whenever your agent switches sessions").
    pending: Mutex<Vec<Pending>>,
    cv: Condvar,
}

impl Herdr {
    /// Build a reporter for explicit `bin`/`pane` (unit tests) — no env read.
    pub fn new(bin: PathBuf, pane: String) -> Arc<Self> {
        let base = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let this = Arc::new(Self {
            bin,
            pane,
            seq: AtomicU64::new(base.saturating_mul(1000)),
            resume: Mutex::new(Vec::new()),
            session_path: Mutex::new(None),
            last: Mutex::new((None, None)),
            pending: Mutex::new(Vec::new()),
            cv: Condvar::new(),
        });
        let weak = Arc::downgrade(&this);
        std::thread::Builder::new()
            .name("herdr-report".into())
            .spawn(move || worker(weak))
            .expect("herdr report thread");
        this
    }

    /// Report this integration from the process environment, or `None`
    /// outside Herdr (or when the env contract is incomplete).
    pub fn detect() -> Option<Arc<Herdr>> {
        if std::env::var_os("HERDR_ENV").as_deref() != Some(std::ffi::OsStr::new("1")) {
            return None;
        }
        let bin = PathBuf::from(std::env::var_os("HERDR_BIN_PATH")?);
        let pane = std::env::var("HERDR_PANE_ID")
            .ok()
            .filter(|p| !p.is_empty())?;
        // Nested/test processes inherit HERDR_ENV=1 from a pane owned by the
        // TS `omp` integration (source `herdr:omp`). Reporting from them is
        // rejected server-side (owner conflict; our reports carry resume
        // argv, so every one logs a `resume_not_accepted` error — a storm).
        // If the pane's foreground agent is `omp`, this process is not the
        // pane's authoritative reporter: stay silent.
        if pane_foreground_is_ts_omp(&bin, &pane) {
            return None;
        }
        Some(Self::new(bin, pane))
    }

    /// Point the reporter at a new session: rebuilds the resume command and
    /// queues `pane report-agent-session` so Herdr restores this session in
    /// this pane after a server restart. `model_ref` is `provider/id`.
    pub fn set_session(
        &self,
        session_file: Option<&Path>,
        model_ref: Option<&str>,
        start_source: &str,
    ) {
        let mut resume = lock(&self.resume);
        *resume = Vec::new();
        if let Some(f) = session_file {
            resume.push("omt".to_string());
            resume.push("--resume".to_string());
            resume.push(f.display().to_string());
            if let Some(m) = model_ref {
                resume.push("--model".to_string());
                resume.push(m.to_string());
            }
        }
        drop(resume);
        *lock(&self.session_path) = session_file.map(|f| f.display().to_string());
        self.enqueue(Pending::Session {
            start_source: start_source.to_string(),
            seq: self.next_seq(),
        });
        self.clear_last();
    }

    /// Report agent state; identical consecutive states are dropped.
    /// Never blocks.
    pub fn report(&self, state: State, message: Option<&str>) {
        let message = message.and_then(|m| {
            let t = m.trim();
            (!t.is_empty()).then(|| t.chars().take(240).collect())
        });
        {
            let mut last = lock(&self.last);
            if *last == (Some(state), message.clone()) {
                return;
            }
            *last = (Some(state), message.clone());
        }
        self.enqueue(Pending::State {
            state,
            message,
            seq: self.next_seq(),
        });
    }

    /// Synchronous, still bounded, release for process-exit paths (Drop,
    /// `std::process::exit`): a background thread could be killed before it
    /// delivers, and the release is exactly what restores nothing-on-restart.
    pub fn release_blocking(&self) {
        self.run(&Pending::Release {
            seq: self.next_seq(),
        });
    }

    fn clear_last(&self) {
        *lock(&self.last) = (None, None);
    }

    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn enqueue(&self, job: Pending) {
        let mut q = lock(&self.pending);
        let same_kind = q
            .last()
            .is_some_and(|l| std::mem::discriminant(l) == std::mem::discriminant(&job));
        if same_kind {
            q.pop(); // keep only the latest of a kind
        }
        q.push(job);
        while q.len() > 16 {
            q.remove(0);
        }
        drop(q);
        self.cv.notify_one();
    }

    fn session_path(&self) -> Option<String> {
        lock(&self.session_path).clone()
    }

    fn resume_argv(&self) -> Vec<String> {
        lock(&self.resume).clone()
    }

    /// Assemble the full argv for one report (unit tests assert on this).
    fn argv(&self, job: &Pending) -> Vec<String> {
        let mut argv = vec!["pane".to_string()];
        match job {
            Pending::State {
                state,
                message,
                seq,
            } => {
                argv.push("report-agent".to_string());
                argv.push(self.pane.clone());
                argv.push("--source".to_string());
                argv.push(SOURCE.to_string());
                argv.push("--agent".to_string());
                argv.push(AGENT.to_string());
                argv.push("--state".to_string());
                argv.push(state.as_str().to_string());
                if let Some(m) = message {
                    argv.push("--message".to_string());
                    argv.push(m.clone());
                }
                argv.push("--seq".to_string());
                argv.push(seq.to_string());
                if let Some(p) = self.session_path() {
                    argv.push("--agent-session-path".to_string());
                    argv.push(p);
                }
            }
            Pending::Session { start_source, seq } => {
                argv.push("report-agent-session".to_string());
                argv.push(self.pane.clone());
                argv.push("--source".to_string());
                argv.push(SOURCE.to_string());
                argv.push("--agent".to_string());
                argv.push(AGENT.to_string());
                argv.push("--seq".to_string());
                argv.push(seq.to_string());
                if let Some(p) = self.session_path() {
                    argv.push("--agent-session-path".to_string());
                    argv.push(p);
                }
                argv.push("--session-start-source".to_string());
                argv.push(start_source.clone());
            }
            Pending::Release { seq } => {
                argv.push("release-agent".to_string());
                argv.push(self.pane.clone());
                argv.push("--source".to_string());
                argv.push(SOURCE.to_string());
                argv.push("--agent".to_string());
                argv.push(AGENT.to_string());
                argv.push("--seq".to_string());
                argv.push(seq.to_string());
            }
        }
        let resume = self.resume_argv();
        if !matches!(job, Pending::Release { .. }) && !resume.is_empty() {
            argv.push("--".to_string());
            argv.extend(resume);
        }
        argv
    }

    /// Run one report with a hard wall-clock budget; every failure is dropped.
    fn run(&self, job: &Pending) {
        let mut child = match Command::new(&self.bin)
            .args(self.argv(job))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => return,
        };
        let deadline = std::time::Instant::now() + REPORT_TIMEOUT;
        loop {
            match child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(_) => return,
            }
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Single delivery thread: drains the whole queue, then waits again. While a
/// delivery is in flight, `enqueue` coalesces same-kind reports, so at most a
/// few stale states wait — older duplicates are dropped per the docs.
fn worker(weak: std::sync::Weak<Herdr>) {
    loop {
        let jobs = {
            let this = match weak.upgrade() {
                Some(t) => t,
                None => return, // reporter dropped: nothing left to send
            };
            let mut q = lock(&this.pending);
            loop {
                if !q.is_empty() {
                    break std::mem::take(&mut *q);
                }
                let (guard, _) = this
                    .cv
                    .wait_timeout(q, Duration::from_millis(500))
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                q = guard;
            }
        };
        for job in jobs {
            match weak.upgrade() {
                Some(this) => this.run(&job),
                None => return,
            }
        }
    }
}

static GLOBAL: std::sync::LazyLock<Option<Arc<Herdr>>> = std::sync::LazyLock::new(Herdr::detect);

/// One bounded `pane process-info` probe: true when the pane's foreground
/// process is the TS `omp` agent (i.e. the pane is owned by the `herdr:omp`
/// integration). Any failure (spawn, timeout, parse) returns false so the
/// reporter keeps its previous behaviour outside that one known-conflict case.
fn pane_foreground_is_ts_omp(bin: &Path, pane: &str) -> bool {
    let mut child = match Command::new(bin)
        .args(["pane", "process-info", "--pane", pane])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let deadline = std::time::Instant::now() + Duration::from_millis(1500);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return false;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return false,
        }
    }
    let Ok(output) = child.wait_with_output() else {
        return false;
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&output.stdout) else {
        return false;
    };
    value["result"]["process_info"]["foreground_processes"]
        .as_array()
        .is_some_and(|processes| {
            processes.iter().any(|p| {
                p["name"]
                    .as_str()
                    .is_some_and(|name| name == "omp" || name.ends_with("/omp"))
            })
        })
}

/// Process-wide reporter (env-detected once); `None` outside Herdr.
pub fn global() -> Option<Arc<Herdr>> {
    GLOBAL.clone()
}

/// Stdout capture for tests: `argv()` is observable through the stub binary.
#[cfg(test)]
pub(crate) fn argv_for_test(h: &Herdr, job_state: State) -> Vec<String> {
    h.argv(&Pending::State {
        state: job_state,
        message: None,
        seq: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Serialize env-mutating tests (and anything touching the global stub dir).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn stub_bin(dir: &Path) -> PathBuf {
        let bin = dir.join("herdr-stub");
        let log = dir.join("herdr-stub.log");
        let script = format!("#!/bin/sh\nprintf '%s\\n' \"$@\" >> '{}'\n", log.display());
        std::fs::write(&bin, script).unwrap();
        std::fs::set_permissions(&bin, {
            use std::os::unix::fs::PermissionsExt;
            std::fs::Permissions::from_mode(0o755)
        })
        .unwrap();
        bin
    }

    fn read_log(dir: &Path) -> Vec<String> {
        std::fs::read_to_string(dir.join("herdr-stub.log"))
            .map(|s| s.lines().map(str::to_string).collect())
            .unwrap_or_default()
    }

    /// Poll the stub log until `pred` holds (or the deadline), then return
    /// the log lines — deliveries are async, so assertions must not race them.
    fn wait_until(dir: &Path, pred: impl Fn(&[String]) -> bool) -> Vec<String> {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let lines = read_log(dir);
            if pred(&lines) || std::time::Instant::now() >= deadline {
                return lines;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("omp-herdr-test-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Reports coalesce by design (only the latest queued state is sent), so
    /// tests settle briefly between reports to observe each delivery.
    fn settle() {
        std::thread::sleep(Duration::from_millis(200));
    }

    #[test]
    fn disabled_outside_herdr_env() {
        let _g = lock(&ENV_LOCK);
        let dir = tmp("disabled");
        for (k, v) in [
            ("HERDR_ENV", None),
            ("HERDR_ENV", Some("0")),
            ("HERDR_ENV", Some("1")), // missing bin/pane
        ] {
            std::env::remove_var("HERDR_BIN_PATH");
            std::env::remove_var("HERDR_PANE_ID");
            if let Some(v) = v {
                std::env::set_var("HERDR_ENV", v);
            } else {
                std::env::remove_var("HERDR_ENV");
            }
            // fresh detection per case (bypass the process-global cache)
            assert!(Herdr::detect().is_none(), "case {k:?}={v:?}");
        }
        std::env::remove_var("HERDR_ENV");
        let _ = dir;
    }

    #[test]
    fn enabled_inside_herdr_env() {
        let _g = lock(&ENV_LOCK);
        let dir = tmp("enabled");
        let bin = stub_bin(&dir);
        std::env::set_var("HERDR_ENV", "1");
        std::env::set_var("HERDR_BIN_PATH", &bin);
        std::env::set_var("HERDR_PANE_ID", "pane-42");
        let h = Herdr::detect().expect("detected inside herdr");
        assert_eq!(h.pane, "pane-42");
        std::env::remove_var("HERDR_ENV");
        std::env::remove_var("HERDR_BIN_PATH");
        std::env::remove_var("HERDR_PANE_ID");
    }

    #[test]
    fn foreground_ts_omp_probe_parses_stub_output() {
        let dir = std::env::temp_dir().join(format!("herdr-probe-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("herdr");
        std::fs::write(
            &bin,
            "#!/bin/sh\nprintf '%s' '{\"result\":{\"process_info\":{\"foreground_processes\":[{\"name\":\"omp\"}]}}}'\n",
        )
        .unwrap();
        #[cfg(unix)]
        std::fs::set_permissions(&bin, {
            use std::os::unix::fs::PermissionsExt;
            std::fs::Permissions::from_mode(0o755)
        })
        .unwrap();
        assert!(pane_foreground_is_ts_omp(&bin, "w1:p1"));
        std::fs::write(
            &bin,
            "#!/bin/sh\nprintf '%s' '{\"result\":{\"process_info\":{\"foreground_processes\":[{\"name\":\"omt\"}]}}}'\n",
        )
        .unwrap();
        assert!(!pane_foreground_is_ts_omp(&bin, "w1:p1"));
        std::fs::write(&bin, "#!/bin/sh\nexit 1\n").unwrap();
        assert!(!pane_foreground_is_ts_omp(&bin, "w1:p1"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reports_state_with_increasing_seq_and_resume_argv() {
        let dir = tmp("state");
        let bin = stub_bin(&dir);
        let h = Herdr::new(bin, "pane-7".into());
        h.set_session(
            Some(Path::new("/tmp/sess/abc.jsonl")),
            Some("anthropic/claude-sonnet-4-5"),
            "startup",
        );
        h.report(State::Idle, None);
        settle();
        h.report(State::Working, None);
        settle();
        h.report(State::Blocked, Some("need approval"));
        settle();
        h.report(State::Idle, None);

        let lines = wait_until(&dir, |l| {
            l.iter().any(|l| l == "blocked") && l.iter().any(|l| l == "need approval")
        });
        assert!(lines.iter().any(|l| l == "report-agent"));
        assert!(lines.iter().any(|l| l == "pane-7"));
        assert!(lines.iter().any(|l| l == "--source"));
        assert!(lines.iter().any(|l| l == "omt"));
        assert!(lines.iter().any(|l| l == "--state"));
        // resume argv attached after `--` in every state/session report
        assert!(lines.windows(6).any(|w| {
            w[0] == "--"
                && w[1] == "omt"
                && w[2] == "--resume"
                && w[3] == "/tmp/sess/abc.jsonl"
                && w[4] == "--model"
                && w[5] == "anthropic/claude-sonnet-4-5"
        }));
        // session identity rides along
        assert!(lines
            .windows(2)
            .any(|w| w == ["--agent-session-path", "/tmp/sess/abc.jsonl"]));
        // seq strictly increases across reports
        let seqs: Vec<u64> = lines
            .iter()
            .zip(lines.iter().skip(1))
            .filter(|(a, _)| *a == "--seq")
            .filter_map(|(_, b)| b.parse().ok())
            .collect();
        assert!(seqs.len() >= 3, "seqs: {seqs:?}");
        assert!(seqs.windows(2).all(|w| w[0] < w[1]), "seqs: {seqs:?}");
    }

    #[test]
    fn dedupes_identical_states_unless_forced() {
        let dir = tmp("dedupe");
        let bin = stub_bin(&dir);
        let h = Herdr::new(bin, "pane-8".into());
        h.report(State::Working, None);
        settle();
        h.report(State::Working, None);
        settle();
        h.report(State::Working, None);
        // no session: only state reports, each `--seq` once
        let lines = wait_until(&dir, |l| !l.is_empty());
        let seq_count = |l: &[String]| {
            l.iter()
                .zip(l.iter().skip(1))
                .filter(|(a, _)| *a == "--seq")
                .count()
        };
        assert_eq!(seq_count(&lines), 1, "duplicate states dropped: {lines:?}");
        // different message → new report
        h.report(State::Working, Some("still going"));
        let lines = wait_until(&dir, |l| seq_count(l) == 2);
        assert_eq!(seq_count(&lines), 2, "message change re-reports: {lines:?}");
    }

    #[test]
    fn session_switch_and_release_reports() {
        let dir = tmp("session");
        let bin = stub_bin(&dir);
        let h = Herdr::new(bin, "pane-9".into());
        h.set_session(
            Some(Path::new("/tmp/a.jsonl")),
            Some("openai/gpt-5"),
            "resume",
        );
        settle();
        h.set_session(
            Some(Path::new("/tmp/b.jsonl")),
            Some("openai/gpt-5"),
            "fork",
        );
        settle();
        h.release_blocking();

        let lines = wait_until(&dir, |l| {
            l.iter().any(|l| l == "release-agent") && l.iter().any(|l| l == "report-agent-session")
        });
        assert!(lines.iter().any(|l| l == "--session-start-source"));
        assert!(lines.iter().any(|l| l == "fork"));
        assert!(lines.iter().any(|l| l == "/tmp/b.jsonl"));
        // release must not carry a resume argv (no `--` in its arg block)
        let rel = lines.iter().position(|l| l == "release-agent").unwrap();
        assert!(!lines[rel..].iter().take(8).any(|l| l == "--"));
        // session reports carry the resume command
        assert!(lines.windows(6).any(|w| {
            w[0] == "--"
                && w[1] == "omt"
                && w[2] == "--resume"
                && w[3] == "/tmp/b.jsonl"
                && w[4] == "--model"
                && w[5] == "openai/gpt-5"
        }));
    }

    #[test]
    fn report_survives_a_hanging_stub() {
        let dir = tmp("hang");
        let bin = dir.join("herdr-hang-stub");
        std::fs::write(&bin, "#!/bin/sh\nsleep 60\n").unwrap();
        std::fs::set_permissions(&bin, {
            use std::os::unix::fs::PermissionsExt;
            std::fs::Permissions::from_mode(0o755)
        })
        .unwrap();
        let h = Herdr::new(bin, "pane-10".into());
        let t0 = std::time::Instant::now();
        h.report(State::Idle, None); // returns immediately
        assert!(t0.elapsed() < Duration::from_millis(200));
        // a later report to a working stub still gets through after the hung one is killed
        let bin2 = stub_bin(&dir);
        let h2 = Herdr::new(bin2, "pane-11".into());
        h2.report(State::Working, None);
        let lines = wait_until(&dir, |l| {
            l.iter().any(|l| l == "--seq") && l.iter().any(|l| l == "pane-11")
        });
        assert!(lines.iter().any(|l| l == "--seq"));
    }

    #[test]
    fn argv_for_state_is_stable() {
        let dir = tmp("argv");
        let bin = stub_bin(&dir);
        let h = Herdr::new(bin, "pane-12".into());
        let argv = argv_for_test(&h, State::Blocked);
        assert_eq!(argv[0..2], ["pane", "report-agent"]);
        // no resume/session configured: nothing after flags
        assert!(!argv.iter().any(|a| a == "--"));
        let mut f = std::fs::File::create(dir.join("marker")).unwrap();
        f.write_all(b"ok").unwrap();
    }
}
