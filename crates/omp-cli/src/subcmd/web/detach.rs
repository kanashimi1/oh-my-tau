//! Detached rpc agents: the registry and liveness/cleanup helpers behind
//! `omp-rs web`'s restart-surviving children.
//!
//! Every open session is an `omp-rs --mode rpc` child started with
//! `OMP_RPC_SOCKET` set. The child lives in its own process group, logs
//! stdout/stderr to `~/.omp-rs/agents/<id>/log`, and serves command frames on
//! a per-agent Unix socket `~/.omp-rs/agents/<id>.sock`. A `registry.json`
//! records each child (`{id,pid,pgid,session,socket,startedMs,hubPid}`) so a
//! restarting hub can adopt live children (and reclaim dead ones), and so two
//! hubs never write to the same agent at once.
//!
//! Orchestrator scenario (manual; not run by the build):
//!   1. `omp-rs web &`                     # hub A
//!   2. `POST /api/agents {cwd,...}`        # spawn a child
//!   3. `kill -TERM <hubA pid>`             # stop A; the child survives
//!   4. `ps -p <child pid>`                 # the child is still running
//!   5. `omp-rs web &`                      # hub B adopts the child
//!   6. `GET /api/agents`                   # adopted agent present (adopted:true)
//!   7. `DELETE /api/agents/<id>`           # SIGTERM the group, registry cleared

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One detached child, as persisted in `registry.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Entry {
    pub id: String,
    pub pid: u32,
    pub pgid: u32,
    /// Session file the child was asked to resume (best effort; the live state
    /// is re-probed after adoption).
    #[serde(default)]
    pub session: Option<String>,
    /// `--approval-mode` the child was started with, so an adopted agent keeps
    /// reporting it after a hub restart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_mode: Option<String>,
    pub socket: PathBuf,
    pub started_ms: i64,
    /// Hub that currently owns this child (`None` once released on shutdown).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hub_pid: Option<u32>,
}

/// `~/.omp-rs/agents`.
pub fn agents_dir() -> PathBuf {
    home_dir().join(".omp-rs/agents")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from)
}

/// Per-agent stdout/stderr log.
pub fn log_path(id: &str) -> PathBuf {
    agents_dir().join(id).join("log")
}

/// Per-agent control socket.
pub fn socket_path(id: &str) -> PathBuf {
    agents_dir().join(format!("{id}.sock"))
}

/// Per-agent bridge state (`state.json`), written by the web bridge and read by
/// adoption so a restarting hub restores dialogs / run markers without scanning
/// the (possibly multi-GB) log.
pub fn state_path(id: &str) -> PathBuf {
    agents_dir().join(id).join("state.json")
}

/// Load persisted bridge state; missing or corrupt files read as `None`.
pub fn load_state<T: serde::de::DeserializeOwned>(id: &str) -> Option<T> {
    let text = fs::read_to_string(state_path(id)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Persist bridge state atomically (the per-agent directory is created on
/// demand), so a hub crash cannot leave a half-written JSON document.
pub fn save_state<T: serde::Serialize>(id: &str, state: &T) {
    let dir = agents_dir().join(id);
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(text) = serde_json::to_string(state) {
        let path = state_path(id);
        let tmp = dir.join("state.json.tmp");
        if fs::write(&tmp, text).is_ok() {
            let _ = fs::rename(tmp, path);
        }
    }
}
fn registry_path() -> PathBuf {
    agents_dir().join("registry.json")
}

/// Load the registry; missing or corrupt files read as empty.
pub fn load_registry() -> Vec<Entry> {
    let Ok(text) = fs::read_to_string(registry_path()) else {
        return Vec::new();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

/// Persist the registry, creating the agents directory as needed.
pub fn save_registry(entries: &[Entry]) {
    let _ = fs::create_dir_all(agents_dir());
    if let Ok(text) = serde_json::to_string_pretty(entries) {
        let _ = fs::write(registry_path(), text);
    }
}

/// Is `pid` alive? On Linux, a `/proc/<pid>` entry exists. No libc/nix.
pub fn pid_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        pid > 0 && Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Portable fallback: no signal-free probe; assume alive.
        let _ = pid;
        true
    }
}

/// Working directory of a live child, read from `/proc/<pid>/cwd`.
pub fn pid_cwd(pid: u32) -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        fs::read_link(format!("/proc/{pid}/cwd")).ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        None
    }
}

/// Can a fresh client reach the child's control socket?
pub fn socket_connects(path: &Path) -> bool {
    #[cfg(unix)]
    {
        std::os::unix::net::UnixStream::connect(path).is_ok()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        false
    }
}

/// SIGTERM the child: its process group (descendants such as running tool
/// subprocesses) and the leader itself. Returns false when neither signal was
/// delivered so callers can fall back.
///
/// Signals go through `killpg(2)`/`kill(2)` directly, never through the
/// `kill` binary: procps-ng 4.0.4 parses `kill -TERM -<pgid>` (no `--`) as a
/// broadcast to every process of the user. On 2026-10-24 one
/// `DELETE /api/agents/<id>` took down the whole user session that way
/// (systemd --user, herdr, every agent).
#[allow(clippy::similar_names)] // pid and pgid are the two distinct ids a kill targets
pub fn kill_group(pid: u32, pgid: u32) -> bool {
    // pgid 0 = caller's own group, 1 = init: never valid targets here.
    if pgid <= 1 || pid <= 1 {
        return false;
    }
    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, killpg, Signal};
        use nix::unistd::Pid;
        let (Ok(pgid), Ok(pid)) = (i32::try_from(pgid), i32::try_from(pid)) else {
            return false;
        };
        let group = killpg(Pid::from_raw(pgid), Signal::SIGTERM).is_ok();
        let leader = kill(Pid::from_raw(pid), Signal::SIGTERM).is_ok();
        group || leader
    }
    #[cfg(not(unix))]
    {
        false
    }
}

/// Drop a dead agent: remove its registry record and its control socket. The
/// agent log is kept for reference.
pub fn remove_entry(entries: &mut Vec<Entry>, id: &str) {
    entries.retain(|e| e.id != id);
    let _ = fs::remove_file(socket_path(id));
}

/// Insert or replace one entry on disk.
pub fn upsert(entry: Entry) {
    let mut entries = load_registry();
    entries.retain(|e| e.id != entry.id);
    entries.push(entry);
    save_registry(&entries);
}

/// Point one entry's `hubPid` at `hub_pid` (`None` = unowned) and persist.
pub fn set_hub_pid(id: &str, hub_pid: Option<u32>) {
    let mut entries = load_registry();
    if let Some(entry) = entries.iter_mut().find(|e| e.id == id) {
        entry.hub_pid = hub_pid;
        save_registry(&entries);
    }
}

/// Forget one agent on disk: remove its record and control socket (the log is
/// kept for reference). No-op when the record is already gone.
pub fn unregister(id: &str) {
    let mut entries = load_registry();
    let had = entries.iter().any(|e| e.id == id);
    remove_entry(&mut entries, id);
    if had {
        save_registry(&entries);
    }
}

/// Number of rotated logs kept per agent. `log` is the live one; completed runs
/// are shifted to `log.1`, `log.2`, … as the child exits.
const LOG_ROTATION_KEEP: usize = 3;

/// Rotate a *completed* agent log (`log` → `log.1`, …, dropping anything past
/// `LOG_ROTATION_KEEP`). Safe only when the child is dead (no live writer or
/// tail); callers invoke it from the exit path, never for a running agent.
pub fn rotate_log(id: &str) {
    let dir = agents_dir().join(id);
    for i in (1..LOG_ROTATION_KEEP).rev() {
        let from = dir.join(format!("log.{i}"));
        let to = dir.join(format!("log.{}", i + 1));
        if from.exists() {
            let _ = fs::rename(&from, &to);
        }
    }
    let live = dir.join("log");
    if live.exists() {
        let _ = fs::rename(&live, dir.join("log.1"));
    }
    // Drop anything beyond the cap (e.g. from a larger rotation).
    for i in (LOG_ROTATION_KEEP + 1)..LOG_ROTATION_KEEP + 8 {
        let p = dir.join(format!("log.{i}"));
        if p.exists() {
            let _ = fs::remove_file(&p);
        } else {
            break;
        }
    }
}

/// Owning hub of this entry, when a different *live* hub still holds it. That
/// hub may be mid-write to the child; a second writer would corrupt the
/// session, so the entry must be left alone.
pub fn foreign_owner(entry: &Entry, my_pid: u32) -> Option<u32> {
    let owner = entry.hub_pid?;
    (owner != my_pid && pid_alive(owner)).then_some(owner)
}

#[allow(clippy::unwrap_used, clippy::expect_used)] // test helpers (cfg(all(test, …)) is not seen as test code)
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::process::CommandExt;
    use std::process::{Child, Command};
    use std::time::{Duration, Instant};

    fn sleeper() -> Child {
        Command::new("sleep")
            .arg("30")
            .process_group(0)
            .spawn()
            .expect("spawn sleep")
    }

    fn exited(child: &mut Child) -> bool {
        let until = Instant::now() + Duration::from_secs(5);
        while Instant::now() < until {
            if child.try_wait().expect("try_wait").is_some() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    /// Regression for the 2026-10-24 session wipe: signalling one agent's
    /// group must leave every other process group alone.
    #[test]
    fn kill_group_hits_only_the_target_group() {
        let mut target = sleeper();
        let mut bystander = sleeper();
        assert!(kill_group(target.id(), target.id()));
        assert!(exited(&mut target), "target group survived");
        assert!(
            bystander.try_wait().expect("try_wait").is_none(),
            "bystander was signalled"
        );
        let _ = bystander.kill();
        let _ = bystander.wait();
    }

    #[test]
    fn kill_group_refuses_own_group_and_init() {
        assert!(!kill_group(0, 0));
        assert!(!kill_group(1, 1));
        assert!(!kill_group(12345, 1));
    }

    /// Registry records written before `approvalMode` existed must still load,
    /// and the field must round-trip (as `approvalMode`) once set.
    #[test]
    fn entry_round_trips_with_and_without_approval_mode() {
        let old: Entry = serde_json::from_str(
            r#"{"id":"a","pid":1,"pgid":1,"socket":"/tmp/a.sock","startedMs":0}"#,
        )
        .expect("pre-approvalMode record loads");
        assert_eq!(old.approval_mode, None);

        let entry = Entry {
            id: "b".into(),
            pid: 2,
            pgid: 2,
            session: None,
            approval_mode: Some("always-ask".into()),
            socket: PathBuf::from("/tmp/b.sock"),
            started_ms: 1,
            hub_pid: None,
        };
        let text = serde_json::to_string(&entry).expect("serialize");
        assert!(text.contains(r#""approvalMode":"always-ask""#));
        let back: Entry = serde_json::from_str(&text).expect("round-trip");
        assert_eq!(back.approval_mode.as_deref(), Some("always-ask"));
    }

    /// Rotation only runs on a *completed* log (child dead): the live `log`
    /// shifts to `log.1`, older runs shift up, and everything past the cap is
    /// dropped — never touching a running writer/tail (those callers only
    /// rotate after exit).
    #[test]
    fn rotate_log_shifts_and_caps_completed_logs() {
        let home = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", home.path());
        let dir = agents_dir().join("a");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("log"), "live").unwrap();
        std::fs::write(dir.join("log.1"), "old1").unwrap();
        std::fs::write(dir.join("log.2"), "old2").unwrap();
        std::fs::write(dir.join("log.3"), "old3").unwrap();
        rotate_log("a");
        assert_eq!(std::fs::read_to_string(dir.join("log.1")).unwrap(), "live");
        assert_eq!(std::fs::read_to_string(dir.join("log.2")).unwrap(), "old1");
        assert_eq!(std::fs::read_to_string(dir.join("log.3")).unwrap(), "old2");
        assert!(!dir.join("log.4").exists(), "past-cap log kept");
    }
}
