//! Port of the pi-utils `dirs.ts` session-dir resolution (`getSessionsDir`)
//! and `session-paths.ts` (`getDefaultSessionDirName`). Single implementation:
//! omp-cli resolves its TS-compatible session dir through here too.

use std::path::{Path, PathBuf};

/// Environment inputs of TS `getSessionsDir()` (pi-utils `dirs.ts`).
pub struct TsDirsEnv {
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
    let rest_ok =
        chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'));
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
pub fn ts_sessions_root(env: &TsDirsEnv, exists: impl Fn(&Path) -> bool) -> PathBuf {
    let profile = match env.omp_profile.as_deref() {
        Some(p) => ts_normalize_profile(Some(p)),
        None => ts_normalize_profile(env.pi_profile.as_deref()),
    };
    let base = env.home.join(
        env.config_dir
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or(".omp"),
    );
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
pub fn ts_tmpdir(get: impl Fn(&str) -> Option<String>) -> String {
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
pub fn ts_session_dir_name(cwd: &Path, home: &Path, tmp: &Path) -> String {
    let encode = |rel: &Path| rel.to_string_lossy().replace(['/', '\\', ':'], "-");
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

#[cfg(test)]
mod tests {
    use super::super::{ts_session_dir_name, ts_sessions_root, ts_tmpdir, TsDirsEnv};
    use std::path::{Path, PathBuf};

    #[test]
    fn ts_session_dir_names_match_ts_rule() {
        let home = Path::new("/home/u");
        let tmp = Path::new("/tmp");
        let name = |cwd: &str| ts_session_dir_name(Path::new(cwd), home, tmp);
        assert_eq!(name("/home/u/omrs"), "-omrs");
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
        assert_eq!(
            ts_sessions_root(&e, none),
            Path::new("/home/u/.omp/agent/sessions")
        );
        e.config_dir = Some("cfg".into());
        assert_eq!(
            ts_sessions_root(&e, none),
            Path::new("/home/u/cfg/agent/sessions")
        );
        e.agent_dir = Some("/custom".into());
        assert_eq!(ts_sessions_root(&e, none), Path::new("/custom/sessions"));
        e.agent_dir = Some(String::new());
        assert_eq!(
            ts_sessions_root(&e, none),
            Path::new("/home/u/cfg/agent/sessions")
        );
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
        assert_eq!(
            ts_sessions_root(&e, none),
            Path::new("/home/u/.omp/agent/sessions")
        );
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
        assert_eq!(
            ts_sessions_root(&d, none),
            Path::new("/home/u/.omp/agent/sessions")
        );
        // Invalid names fall back to the default profile.
        let mut bad = env("/home/u");
        bad.omp_profile = Some("Bad Name".into());
        assert_eq!(
            ts_sessions_root(&bad, none),
            Path::new("/home/u/.omp/agent/sessions")
        );
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
        assert_eq!(
            ts_sessions_root(&e, prof),
            Path::new("/x/omp/profiles/work/sessions")
        );
        assert_eq!(
            ts_sessions_root(&e, present),
            Path::new("/home/u/.omp/profiles/work/agent/sessions")
        );
    }

    #[test]
    fn tmpdir_follows_node_rules() {
        let with = |pairs: &'static [(&'static str, &'static str)]| {
            ts_tmpdir(move |k| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| (*v).to_owned())
            })
        };
        assert_eq!(with(&[]), "/tmp");
        assert_eq!(with(&[("TMPDIR", "/var/t/")]), "/var/t");
        assert_eq!(with(&[("TMPDIR", ""), ("TMP", "/m")]), "/m");
        assert_eq!(with(&[("TMPDIR", ""), ("TMP", ""), ("TEMP", "/e")]), "/e");
        assert_eq!(with(&[("TMPDIR", "")]), "/tmp");
        assert_eq!(with(&[("TMPDIR", "/")]), "/");
    }
}
