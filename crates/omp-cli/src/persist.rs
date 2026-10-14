//! Session persistence (omp session JSONL v3) via `omp-session`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use omp_session::SessionWriter;
use pa_agent::agent::Agent;
use pa_agent::types::AgentEvent;

use crate::agent::civil;

/// ISO-8601 UTC with milliseconds, like omp ("2026-10-11T14:08:39.790Z").
#[must_use]
pub fn iso_from_ms(ms: i64) -> String {
    let (y, mo, d) = civil(ms.div_euclid(86_400_000));
    let rem = ms.rem_euclid(86_400_000);
    format!(
        "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3_600_000,
        rem / 60_000 % 60,
        rem / 1000 % 60,
        rem % 1000
    )
}

#[must_use]
pub fn now_iso() -> String {
    iso_from_ms(pa_agent::now_ms())
}

/// UUID v7 string (48-bit ms timestamp + random).
#[must_use]
pub fn uuid_v7() -> String {
    use rand::RngCore;
    let mut b = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut b);
    let ms = u64::try_from(pa_agent::now_ms()).unwrap_or(0);
    b[..6].copy_from_slice(&ms.to_be_bytes()[2..]);
    b[6] = 0x70 | (b[6] & 0x0f);
    b[8] = 0x80 | (b[8] & 0x3f);
    let mut h = String::with_capacity(b.len() * 2);
    for x in &b {
        use std::fmt::Write as _;
        write!(h, "{x:02x}").expect("writing to a String is infallible");
    }
    format!(
        "{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    )
}

/// The default omp-rs session dir for `cwd` (`~/.omp-rs/sessions/<slug>`).
pub(crate) fn sessions_dir(cwd: &Path) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let slug = cwd.display().to_string().replace('/', "-");
    Some(PathBuf::from(home).join(".omp-rs/sessions").join(slug))
}

/// TS omp's session dir for `cwd` (`<agent dir>/sessions/<name>`, TS
/// `sessionDirForCwd`): read-only lookup, nothing is created or migrated.
/// The `/resume` picker and `--resume <id>` list it next to the omp-rs store.
#[must_use]
pub fn ts_sessions_dir(cwd: &Path) -> Option<PathBuf> {
    let home = PathBuf::from(std::env::var_os("HOME")?);
    let env_str = |k: &str| std::env::var(k).ok();
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let tmp = ts_tmpdir(env_str);
    let name = ts_session_dir_name(&canon(cwd), &canon(&home), &canon(Path::new(&tmp)));
    let env = TsDirsEnv {
        home,
        cwd: std::env::current_dir().unwrap_or_default(),
        config_dir: env_str("PI_CONFIG_DIR"),
        agent_dir: env_str("PI_CODING_AGENT_DIR"),
        omp_profile: env_str("OMP_PROFILE"),
        pi_profile: env_str("PI_PROFILE"),
        xdg_data_home: env_str("XDG_DATA_HOME"),
    };
    Some(ts_sessions_root(&env, Path::exists).join(name))
}

/// Environment inputs of TS `getSessionsDir()` (pi-utils `dirs.ts`).
pub(crate) struct TsDirsEnv {
    pub home: PathBuf,
    pub cwd: PathBuf,
    pub config_dir: Option<String>,
    pub agent_dir: Option<String>,
    pub omp_profile: Option<String>,
    pub pi_profile: Option<String>,
    pub xdg_data_home: Option<String>,
}

/// TS `normalizeProfileName` (dirs.ts L63-80); invalid names are `None`
/// (the module-load path swallows the error, `readProfileFromEnvSafe`).
fn ts_normalize_profile(profile: Option<&str>) -> Option<String> {
    let p = profile?.trim();
    if p.is_empty() || p == "default" || p == "." || p == ".." || p.ends_with('.') {
        return None;
    }
    let mut chars = p.chars();
    let first_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit());
    let rest_ok = chars.all(|c| {
        c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-')
    });
    if !first_ok || !rest_ok || p.len() > 64 {
        return None;
    }
    let stem = p.split('.').next().unwrap_or(p).to_ascii_uppercase();
    let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || ((stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.len() == 4
            && stem.as_bytes()[3].is_ascii_digit());
    (!reserved).then(|| p.to_owned())
}

/// TS `getSessionsDir()` = `agentSubdir(undefined, "sessions", "data")`
/// (dirs.ts L956-958, `DirResolver` L342-413, L438-448), default mode:
/// 1. profile = `OMP_PROFILE` (when set, even empty) else `PI_PROFILE` (L90-96);
/// 2. config root = `$HOME/<PI_CONFIG_DIR|.omp>` (L115-122, L306-308), plus
///    `profiles/<profile>` for a named profile;
/// 3. agent dir = `<config root>/agent`, or `PI_CODING_AGENT_DIR` when no
///    profile is active and it is not a `PI_PROFILE`-derived agent dir
///    (L136-138, L461-482);
/// 4. only when the agent dir is the default one and `XDG_DATA_HOME` is set
///    and `$XDG_DATA_HOME/omp[/profiles/<p>]` exists (L363-385), the data
///    root flattens to that directory; else the agent dir.
///
/// `exists` is injected so the XDG probe stays testable.
pub(crate) fn ts_sessions_root(env: &TsDirsEnv, exists: impl Fn(&Path) -> bool) -> PathBuf {
    let profile = match env.omp_profile.as_deref() {
        Some(p) => ts_normalize_profile(Some(p)),
        None => ts_normalize_profile(env.pi_profile.as_deref()),
    };
    let base = env.home.join(env.config_dir.as_deref().filter(|s| !s.is_empty()).unwrap_or(".omp"));
    let config_root = match &profile {
        Some(p) => base.join("profiles").join(p),
        None => base.clone(),
    };
    let default_agent = config_root.join("agent");
    let agent_override = if profile.is_some() {
        None
    } else {
        let pi = ts_normalize_profile(env.pi_profile.as_deref());
        let derived = pi
            .map(|p| base.join("profiles").join(p).join("agent"))
            .is_some_and(|d| Some(d.as_path()) == env.agent_dir.as_deref().map(Path::new));
        env.agent_dir
            .as_deref()
            .filter(|s| !s.is_empty() && !derived)
            .map(|s| env.cwd.join(s))
    };
    let agent_dir = agent_override.unwrap_or_else(|| default_agent.clone());
    let xdg = (agent_dir == default_agent)
        .then(|| env.xdg_data_home.as_deref().filter(|s| !s.is_empty()))
        .flatten()
        .and_then(|v| {
            let app_root = Path::new(v).join("omp");
            let target = match &profile {
                Some(p) => app_root.join("profiles").join(p),
                None => app_root,
            };
            exists(&target).then_some(target)
        });
    xdg.unwrap_or(agent_dir).join("sessions")
}

/// Node `os.tmpdir()` on POSIX: first non-empty of `TMPDIR`, `TMP`, `TEMP`,
/// else `/tmp`; one trailing `/` stripped (unless the path is just `/`).
pub(crate) fn ts_tmpdir(get: impl Fn(&str) -> Option<String>) -> String {
    let mut dir = ["TMPDIR", "TMP", "TEMP"]
        .iter()
        .find_map(|k| get(k).filter(|v| !v.is_empty()))
        .unwrap_or_else(|| "/tmp".to_owned());
    if dir.len() > 1 && dir.ends_with('/') {
        dir.pop();
    }
    dir
}

/// TS `getDefaultSessionDirName` (session-paths.ts) over canonical paths:
/// temp root first (`-tmp[-rel]`), then home (`-[rel]`), else the legacy
/// absolute form `--<abs>--`; `/`, `\` and `:` become `-`.
fn ts_session_dir_name(cwd: &Path, home: &Path, tmp: &Path) -> String {
    let encode = |rel: &Path| {
        rel.to_string_lossy()
            .replace(['/', '\\', ':'], "-")
    };
    let relative = |prefix: &str, rel: &Path| {
        let encoded = encode(rel);
        if encoded.is_empty() {
            prefix.to_owned()
        } else if prefix.ends_with('-') {
            format!("{prefix}{encoded}")
        } else {
            format!("{prefix}-{encoded}")
        }
    };
    // TS isRelativeWithin: a remainder starting with `..` is outside.
    let within = |root: &Path| {
        cwd.strip_prefix(root)
            .ok()
            .filter(|rel| !rel.to_string_lossy().starts_with(".."))
    };
    if let Some(rel) = within(tmp) {
        return relative("-tmp", rel);
    }
    if let Some(rel) = within(home) {
        return relative("-", rel);
    }
    let abs = cwd.to_string_lossy();
    let abs = abs.strip_prefix(['/', '\\']).unwrap_or(&abs);
    format!("--{}--", abs.replace(['/', '\\', ':'], "-"))
}

/// Live session file: the agent's finished messages are appended to
/// `writer`, which `start_new` swaps for a fresh file (`new_session`).
///
/// An *ephemeral* session (`--no-session`) writes to a scratch directory that
/// is removed on drop and reports no `sessionFile`, like omp's in-memory
/// `SessionManager`.
pub struct Persistence {
    writer: Arc<Mutex<SessionWriter>>,
    session_id: Mutex<String>,
    ephemeral: bool,
    scratch: Option<PathBuf>,
    /// While set, an assistant error message is held back instead of written
    /// (omp persists a failed turn only once the retry saga gave up on it).
    defer_errors: Arc<AtomicBool>,
    deferred: Arc<Mutex<Option<(pa_types::session::AgentMessage, i64)>>>,
}

impl Drop for Persistence {
    fn drop(&mut self) {
        if let Some(dir) = &self.scratch {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

impl Persistence {
    #[must_use]
    pub fn session_id(&self) -> String {
        self.session_id.lock().unwrap().clone()
    }

    /// Path of the live session file (a scratch path when ephemeral).
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.writer.lock().unwrap().path().to_path_buf()
    }

    /// The session file omp would report (`sessionFile`); `None` for `--no-session`.
    #[must_use]
    pub fn file(&self) -> Option<PathBuf> {
        (!self.ephemeral).then(|| self.path())
    }

    /// Append a non-message entry (`custom`, ...); omp kinds have typed helpers below.
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    #[allow(dead_code)]
    pub fn append_entry(&self, kind: &str, payload: serde_json::Value) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .append_entry(kind, payload, &now_iso())
            .map(|_| ())
    }

    /// `model_change {model, role?, resolvedModelIsFallback}` (`SessionManager.appendModelChange`).
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    pub fn append_model_change(
        &self,
        model: &str,
        role: Option<&str>,
        resolved_model_is_fallback: bool,
    ) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .append_model_change(model, role, resolved_model_is_fallback, &now_iso())
            .map(|_| ())
    }

    /// `thinking_level_change {thinkingLevel, configured}` (`appendThinkingLevelChange`).
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    pub fn append_thinking_level_change(
        &self,
        level: Option<&str>,
        configured: Option<&str>,
    ) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .append_thinking_level_change(level, configured, &now_iso())
            .map(|_| ())
    }

    /// `title_change` audit entry.
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    pub fn append_title_change(
        &self,
        title: &str,
        source: &str,
        previous: Option<&str>,
    ) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .append_title_change(title, source, previous, None, &now_iso())
            .map(|_| ())
    }

    /// Append a finished conversation message.
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    pub fn append_message(
        &self,
        message: &pa_types::session::AgentMessage,
        timestamp_ms: i64,
    ) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .append_message(message, &iso_from_ms(timestamp_ms))
            .map(|_| ())
    }

    /// Run `f` on the live session writer (compaction entries need the writer's id chain).
    pub fn with_writer<R>(&self, f: impl FnOnce(&mut SessionWriter) -> R) -> R {
        f(&mut self.writer.lock().unwrap())
    }

    /// Hold back (`true`) or write straight through (`false`) assistant error turns.
    pub fn defer_errors(&self, on: bool) {
        self.defer_errors.store(on, Ordering::SeqCst);
    }

    /// The retry saga gave up: persist the held failed turn (`persistTerminalEmptyErrorTurn`),
    /// optionally with a rewritten `errorMessage`.
    pub fn flush_deferred_error(&self, error_message: Option<&str>) {
        let held = self.deferred.lock().unwrap().take();
        let Some((m, ts)) = held else { return };
        let m = match error_message {
            Some(e) => serde_json::to_value(&m)
                .ok()
                .map(|mut v| {
                    v["errorMessage"] = serde_json::json!(e);
                    v
                })
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or(m),
            None => m,
        };
        if let Err(e) = self.append_message(&m, ts) {
            eprintln!("omp-rs: session write failed: {e:#}");
        }
    }

    /// The failed turn was retried: it never reaches the session file.
    pub fn drop_deferred_error(&self) {
        self.deferred.lock().unwrap().take();
    }

    /// Switch to a brand-new session file in `dir` (default: the cwd slot).
    ///
    /// # Errors
    /// Fails when the file cannot be created.
    pub fn start_new(&self, cwd: &Path, dir: Option<&Path>) -> Result<()> {
        let dir = match (&self.scratch, dir) {
            (Some(scratch), _) => scratch.clone(),
            (None, Some(d)) => d.to_path_buf(),
            (None, None) => sessions_dir(cwd).ok_or_else(|| anyhow::anyhow!("HOME is not set"))?,
        };
        let id = uuid_v7();
        let fresh = SessionWriter::create(&dir, &cwd.display().to_string(), &id, &now_iso())?;
        *self.writer.lock().unwrap() = fresh;
        *self.session_id.lock().unwrap() = id;
        self.deferred.lock().unwrap().take();
        Ok(())
    }
}

/// Create a session file for this run and append every completed message.
///
/// # Errors
/// Fails when the session file cannot be created.
#[allow(dead_code)]
pub async fn attach(agent: &Agent, cwd: &Path) -> Result<Persistence> {
    attach_with(agent, cwd, false).await
}

/// [`attach`] with `ephemeral = true` for `--no-session`: nothing is kept after exit
/// and `sessionFile` is not reported.
///
/// # Errors
/// Fails when the (scratch) session file cannot be created.
pub async fn attach_with(agent: &Agent, cwd: &Path, ephemeral: bool) -> Result<Persistence> {
    let session_id = uuid_v7();
    let (dir, scratch) = if ephemeral {
        let d = std::env::temp_dir().join(format!(
            "omp-rs-nosession-{}-{}",
            std::process::id(),
            &session_id[..8]
        ));
        (d.clone(), Some(d))
    } else {
        (
            sessions_dir(cwd).ok_or_else(|| anyhow::anyhow!("HOME is not set"))?,
            None,
        )
    };
    let writer = SessionWriter::create(&dir, &cwd.display().to_string(), &session_id, &now_iso())?;
    let writer = Arc::new(Mutex::new(writer));
    let defer_errors = Arc::new(AtomicBool::new(false));
    let deferred = Arc::new(Mutex::new(None));
    let sub = agent
        .subscribe({
            let writer = writer.clone();
            let defer_errors = defer_errors.clone();
            let deferred = deferred.clone();
            move |event, _| {
                if let AgentEvent::MessageEnd { message } = &event {
                    let trace = std::env::var_os("OMP_RS_TURN_TRACE").is_some();
                    if trace {
                        crate::runtime::turn_trace!(
                            "[turn-trace] persist-begin role={} rss={}kB",
                            message.role(),
                            crate::runtime::rss_kb()
                        );
                    }
                    let converted = Ok::<_, anyhow::Error>(crate::events::message_value(message))
                        .and_then(|v| {
                            serde_json::from_value::<pa_types::session::AgentMessage>(v)
                                .map_err(anyhow::Error::from)
                        });
                    match converted {
                        Ok(m) => {
                            let held = defer_errors.load(Ordering::SeqCst)
                                && matches!(message, pa_agent::types::AgentMessage::Standard(pa_agent::types::Message::Assistant(a))
                                    if a.stop_reason == pa_agent::types::StopReason::Error);
                            if trace {
                                crate::runtime::turn_trace!(
                                    "[turn-trace] persist-converted rss={}kB",
                                    crate::runtime::rss_kb()
                                );
                            }
                            if held {
                                *deferred.lock().unwrap() = Some((m, message.timestamp()));
                            } else {
                                let ts = iso_from_ms(message.timestamp());
                                if let Err(e) = writer.lock().unwrap().append_message(&m, &ts) {
                                    eprintln!("omp-rs: session write failed: {e:#}");
                                }
                            }
                        }
                        Err(e) => eprintln!("omp-rs: session message conversion failed: {e:#}"),
                    }
                }
                Box::pin(async { Ok(()) })
            }
        })
        .await;
    // Dropping a Subscription does not unsubscribe; the listener lives with the agent.
    drop(sub);
    Ok(Persistence {
        writer,
        session_id: Mutex::new(session_id),
        ephemeral,
        scratch,
        defer_errors,
        deferred,
    })
}

#[cfg(test)]
mod tests {
    use super::{ts_session_dir_name, ts_sessions_root, ts_tmpdir, TsDirsEnv};
    use std::path::{Path, PathBuf};

    #[test]
    fn ts_session_dir_names_match_ts_rule() {
        let home = Path::new("/home/u");
        let tmp = Path::new("/tmp");
        let name = |cwd: &str| ts_session_dir_name(Path::new(cwd), home, tmp);
        assert_eq!(name("/home/u/fleet-split"), "-fleet-split");
        assert_eq!(name("/home/u/a/b"), "-a-b");
        assert_eq!(name("/home/u/.omp/agent"), "-.omp-agent");
        assert_eq!(name("/home/u"), "-");
        assert_eq!(name("/tmp"), "-tmp");
        assert_eq!(name("/tmp/x/y"), "-tmp-x-y");
        assert_eq!(name("/mnt/c/Temp"), "--mnt-c-Temp--");
        // A sibling sharing a string prefix is not inside home.
        assert_eq!(name("/home/user2"), "--home-user2--");
        // `..`-prefixed remainder (TS isRelativeWithin) is not within home.
        assert_eq!(name("/home/u/..x"), "--home-u-..x--");
    }

    fn env(home: &str) -> TsDirsEnv {
        TsDirsEnv {
            home: PathBuf::from(home),
            cwd: PathBuf::from("/work"),
            config_dir: None,
            agent_dir: None,
            omp_profile: None,
            pi_profile: None,
            xdg_data_home: None,
        }
    }

    #[test]
    fn sessions_root_default_and_overrides() {
        let none = |_: &Path| false;
        let mut e = env("/home/u");
        assert_eq!(ts_sessions_root(&e, none), Path::new("/home/u/.omp/agent/sessions"));
        e.config_dir = Some("cfg".into());
        assert_eq!(ts_sessions_root(&e, none), Path::new("/home/u/cfg/agent/sessions"));
        e.agent_dir = Some("/custom".into());
        assert_eq!(ts_sessions_root(&e, none), Path::new("/custom/sessions"));
        e.agent_dir = Some(String::new());
        assert_eq!(ts_sessions_root(&e, none), Path::new("/home/u/cfg/agent/sessions"));
        e.agent_dir = Some("rel".into());
        assert_eq!(ts_sessions_root(&e, none), Path::new("/work/rel/sessions"));
    }

    #[test]
    fn sessions_root_profiles() {
        let none = |_: &Path| false;
        let mut e = env("/home/u");
        e.pi_profile = Some("old".into());
        assert_eq!(
            ts_sessions_root(&e, none),
            Path::new("/home/u/.omp/profiles/old/agent/sessions")
        );
        // OMP_PROFILE wins, even when empty (selects the default profile).
        e.omp_profile = Some(String::new());
        assert_eq!(ts_sessions_root(&e, none), Path::new("/home/u/.omp/agent/sessions"));
        e.omp_profile = Some("work".into());
        e.agent_dir = Some("/custom".into());
        assert_eq!(
            ts_sessions_root(&e, none),
            Path::new("/home/u/.omp/profiles/work/agent/sessions")
        );
        // A profile-derived PI_CODING_AGENT_DIR is not a default-mode override.
        let mut d = env("/home/u");
        d.omp_profile = Some(String::new());
        d.pi_profile = Some("old".into());
        d.agent_dir = Some("/home/u/.omp/profiles/old/agent".into());
        assert_eq!(ts_sessions_root(&d, none), Path::new("/home/u/.omp/agent/sessions"));
        // Invalid names fall back to the default profile.
        let mut bad = env("/home/u");
        bad.omp_profile = Some("Bad Name".into());
        assert_eq!(ts_sessions_root(&bad, none), Path::new("/home/u/.omp/agent/sessions"));
    }

    #[test]
    fn sessions_root_xdg_only_when_app_root_exists_and_default_agent() {
        let mut e = env("/home/u");
        e.xdg_data_home = Some("/x".into());
        let present = |p: &Path| p == Path::new("/x/omp");
        assert_eq!(ts_sessions_root(&e, present), Path::new("/x/omp/sessions"));
        assert_eq!(
            ts_sessions_root(&e, |_: &Path| false),
            Path::new("/home/u/.omp/agent/sessions")
        );
        e.agent_dir = Some("/custom".into());
        assert_eq!(ts_sessions_root(&e, present), Path::new("/custom/sessions"));
        e.agent_dir = None;
        e.omp_profile = Some("work".into());
        let prof = |p: &Path| p == Path::new("/x/omp/profiles/work");
        assert_eq!(ts_sessions_root(&e, prof), Path::new("/x/omp/profiles/work/sessions"));
        assert_eq!(
            ts_sessions_root(&e, present),
            Path::new("/home/u/.omp/profiles/work/agent/sessions")
        );
    }

    #[test]
    fn tmpdir_follows_node_rules() {
        let with = |pairs: &'static [(&'static str, &'static str)]| {
            ts_tmpdir(move |k| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| (*v).to_owned()))
        };
        assert_eq!(with(&[]), "/tmp");
        assert_eq!(with(&[("TMPDIR", "/var/t/")]), "/var/t");
        assert_eq!(with(&[("TMPDIR", ""), ("TMP", "/m")]), "/m");
        assert_eq!(with(&[("TMPDIR", ""), ("TMP", ""), ("TEMP", "/e")]), "/e");
        assert_eq!(with(&[("TMPDIR", "")]), "/tmp");
        assert_eq!(with(&[("TMPDIR", "/")]), "/");
    }
}
