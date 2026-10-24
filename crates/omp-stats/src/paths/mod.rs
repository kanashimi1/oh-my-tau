//! Config/session/db locations (port of the pi-utils `dirs.ts` helpers the
//! stats package uses). `HOME` and `PI_CONFIG_DIR` are honoured like TS, so
//! tests point everything at a fixture home.

mod roots;
mod ts_dirs;

use std::path::PathBuf;

pub use roots::{
    omp_rs_sessions_dir, relative_to_session_root, session_project_dir, session_root_of,
    session_roots,
};
pub use ts_dirs::{ts_session_dir_name, ts_sessions_root, ts_tmpdir, TsDirsEnv};

/// `~/.omp` (or `~/$PI_CONFIG_DIR`).
#[must_use]
pub fn config_root_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from);
    let name = std::env::var("PI_CONFIG_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| ".omp".into());
    home.join(name)
}

/// `~/.omp/agent`.
#[must_use]
pub fn agent_dir() -> PathBuf {
    config_root_dir().join("agent")
}

/// TS `getSessionsDir()`: the sessions dir of the omp TS agent, honouring
/// `PI_CONFIG_DIR`, `PI_CODING_AGENT_DIR`, `OMP_PROFILE`/`PI_PROFILE` and
/// `XDG_DATA_HOME` (see [`ts_sessions_root`]). Default `~/.omp/agent/sessions`.
#[must_use]
pub fn sessions_dir() -> PathBuf {
    let env = |k: &str| std::env::var(k).ok();
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from);
    ts_sessions_root(
        &TsDirsEnv {
            home,
            cwd: std::env::current_dir().unwrap_or_default(),
            config_dir: env("PI_CONFIG_DIR"),
            agent_dir: env("PI_CODING_AGENT_DIR"),
            omp_profile: env("OMP_PROFILE"),
            pi_profile: env("PI_PROFILE"),
            xdg_data_home: env("XDG_DATA_HOME"),
        },
        std::path::Path::exists,
    )
}

/// `~/.omp/stats.db`.
#[must_use]
pub fn stats_db_path() -> PathBuf {
    config_root_dir().join("stats.db")
}

/// `~/.omp/agent/agent.db`.
#[must_use]
pub fn agent_db_path() -> PathBuf {
    agent_dir().join("agent.db")
}

/// Classify which agent produced a transcript from its path (TS
/// `classifyAgentType` in parser.ts), as the wire string. Single
/// implementation shared by `parser` (typed wrapper) and `db` (agent_type
/// backfill); lives here because both feature trees can always see paths.
///
/// Layout: `<sessionsDir>/<project>/<file>.jsonl` is the `main` agent;
/// subagent and advisor transcripts live nested one level deeper inside the
/// session's artifacts dir. Any advisor transcript (`__advisor.jsonl` or
/// `__advisor.<slug>.jsonl`) — at any depth — counts as `advisor`; every
/// other nested transcript is a task `subagent`.
#[must_use]
pub fn classify_agent_type_str(session_path: &str) -> &'static str {
    const ADVISOR_BASENAME: &str = "__advisor.jsonl";
    let base = std::path::Path::new(session_path)
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("");
    if base == ADVISOR_BASENAME || (base.starts_with("__advisor.") && base.ends_with(".jsonl")) {
        return "advisor";
    }
    let rel = relative_to_session_root(std::path::Path::new(session_path))
        .unwrap_or_else(|| path_relative(&sessions_dir(), std::path::Path::new(session_path)));
    if rel.components().count() <= 2 {
        "main"
    } else {
        "subagent"
    }
}

/// `path.relative(from, to)` — trailing components of `to` after stripping
/// the shared prefix with `from` (`..` prefixes when `to` is outside).
#[must_use]
pub fn path_relative(from: &std::path::Path, to: &std::path::Path) -> PathBuf {
    let mut fc = from.components().peekable();
    let mut tc = to.components().peekable();
    while let (Some(a), Some(b)) = (fc.peek(), tc.peek()) {
        if a == b {
            fc.next();
            tc.next();
        } else {
            break;
        }
    }
    let mut out = PathBuf::new();
    if fc.peek().is_some() {
        for _ in fc {
            out.push("..");
        }
    }
    for c in tc {
        out.push(c.as_os_str());
    }
    out
}
