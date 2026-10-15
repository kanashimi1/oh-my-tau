//! Session roots: every directory the dashboard ingests transcripts from.
//!
//! TS omp lists only its own [`super::sessions_dir`]. omp-rs persists its
//! sessions under `~/.omp-rs/sessions/<slug>/`, so the dashboard reads both.
//! TS never deletes rows of files outside its own listing, so sharing
//! `stats.db` with a TS omp is safe.

use std::path::{Component, Path, PathBuf};

use super::{sessions_dir, ts_tmpdir};

/// `~/.omp-rs/sessions` (where omp-rs writes its transcripts).
#[must_use]
pub fn omp_rs_sessions_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from);
    home.join(".omp-rs").join("sessions")
}

/// All session roots, the TS root first, deduplicated.
#[must_use]
pub fn session_roots() -> Vec<PathBuf> {
    let mut roots = vec![sessions_dir()];
    let extra = omp_rs_sessions_dir();
    if !roots.contains(&extra) {
        roots.push(extra);
    }
    roots
}

fn relative_inside(root: &Path, path: &Path) -> Option<PathBuf> {
    let rel = super::path_relative(root, path);
    let escapes = rel.components().next().is_none_or(|c| matches!(c, Component::ParentDir | Component::RootDir | Component::Prefix(_)));
    (!escapes).then_some(rel)
}

/// The session root that (lexically) contains `path`.
#[must_use]
pub fn session_root_of(path: &Path) -> Option<PathBuf> {
    session_roots().into_iter().find(|root| relative_inside(root, path).is_some())
}

/// `path` relative to the session root containing it (`None` when outside all roots).
#[must_use]
pub fn relative_to_session_root(path: &Path) -> Option<PathBuf> {
    session_roots().iter().find_map(|root| relative_inside(root, path))
}

/// `/`-separated slug omp-rs derives from a cwd (`/home/u/x` → `-home-u-x`).
fn slugify(path: &Path) -> String {
    path.to_string_lossy().replace('/', "-")
}

/// Map an omp-rs project slug (`cwd` with `/` → `-`) to the TS project dir
/// name of the same cwd (TS `getDefaultSessionDirName`), so one project merges
/// in the dashboard whichever agent wrote the session. The slug is lossy, so
/// the cwd is not decoded: the rule works on prefixes of the home and temp
/// slugs, in TS order — temp (`-tmp[-rel]`), home (`-[rel]`), else the legacy
/// absolute form `--<abs>--`.
fn omp_rs_slug_to_ts_name(slug: &str, home: &Path, tmp: &Path) -> String {
    let under = |prefix: &str| -> Option<String> {
        if prefix.is_empty() || prefix == "-" {
            return None;
        }
        if slug == prefix {
            return Some(String::new());
        }
        slug.strip_prefix(prefix).and_then(|rest| rest.strip_prefix('-')).map(str::to_string)
    };
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    for tmp in [tmp.to_path_buf(), canon(tmp)] {
        if let Some(rel) = under(&slugify(&tmp)) {
            return if rel.is_empty() { "-tmp".to_string() } else { format!("-tmp-{rel}") };
        }
    }
    for home in [home.to_path_buf(), canon(home)] {
        if let Some(rel) = under(&slugify(&home)) {
            return format!("-{rel}");
        }
    }
    format!("--{}--", slug.trim_start_matches('-'))
}

/// The project directory name a transcript belongs to, in TS naming: the first
/// component below its session root, with omp-rs slugs mapped to the TS name.
/// Outside every root this is the first component relative to the TS root
/// (`..`), as TS does.
#[must_use]
pub fn session_project_dir(session_path: &str) -> String {
    let path = Path::new(session_path);
    let first = |rel: &Path| {
        rel.components().next().map(|c| c.as_os_str().to_string_lossy().into_owned()).unwrap_or_default()
    };
    let Some(root) = session_root_of(path) else {
        return first(&super::path_relative(&sessions_dir(), path));
    };
    let project = first(&super::path_relative(&root, path));
    if root == sessions_dir() {
        return project;
    }
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from);
    let tmp = ts_tmpdir(|k| std::env::var(k).ok());
    omp_rs_slug_to_ts_name(&project, &home, Path::new(&tmp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omp_rs_slugs_map_to_ts_names() {
        let (home, tmp) = (Path::new("/home/u"), Path::new("/tmp"));
        let map = |slug: &str| omp_rs_slug_to_ts_name(slug, home, tmp);
        assert_eq!(map("-home-u-fleet-split"), "-fleet-split");
        assert_eq!(map("-home-u"), "-");
        assert_eq!(map("-tmp"), "-tmp");
        assert_eq!(map("-tmp-x-y"), "-tmp-x-y");
        assert_eq!(map("-mnt-c-Temp"), "--mnt-c-Temp--");
        // A sibling sharing a string prefix is not inside home.
        assert_eq!(map("-home-user2"), "--home-user2--");
    }
}
