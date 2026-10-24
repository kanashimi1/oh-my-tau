//! Session roots: every directory the dashboard ingests transcripts from.
//!
//! TS omp lists only its own [`super::sessions_dir`]. omp-rs persists its
//! sessions under `~/.omp-rs/sessions/<slug>/`, so the dashboard reads both.
//! TS never deletes rows of files outside its own listing, so sharing
//! `stats.db` with a TS omp is safe.

use std::path::{Component, Path, PathBuf};

use super::{sessions_dir, ts_session_dir_name, ts_tmpdir};

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
    // Compare resolved paths: a TS root configured onto (or symlinked to) the
    // omp-rs store must not be listed, and ingested, twice.
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    if canon(&roots[0]) != canon(&extra) {
        roots.push(extra);
    }
    roots
}

fn relative_inside(root: &Path, path: &Path) -> Option<PathBuf> {
    let rel = super::path_relative(root, path);
    let escapes = rel.components().next().is_none_or(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    });
    (!escapes).then_some(rel)
}

/// The session root that (lexically) contains `path`.
#[must_use]
pub fn session_root_of(path: &Path) -> Option<PathBuf> {
    session_roots()
        .into_iter()
        .find(|root| relative_inside(root, path).is_some())
}

/// `path` relative to the session root containing it (`None` when outside all roots).
#[must_use]
pub fn relative_to_session_root(path: &Path) -> Option<PathBuf> {
    session_roots()
        .iter()
        .find_map(|root| relative_inside(root, path))
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
        slug.strip_prefix(prefix)
            .and_then(|rest| rest.strip_prefix('-'))
            .map(str::to_string)
    };
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    for tmp in [tmp.to_path_buf(), canon(tmp)] {
        if let Some(rel) = under(&slugify(&tmp)) {
            return if rel.is_empty() {
                "-tmp".to_string()
            } else {
                format!("-tmp-{rel}")
            };
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
        rel.components()
            .next()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .unwrap_or_default()
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
    // Exact: the session header's cwd, when it is the one the slug encodes.
    if let Some(cwd) = header_cwd(path).filter(|cwd| slugify(cwd) == project) {
        let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        return ts_session_dir_name(&canon(&cwd), &canon(&home), &canon(Path::new(&tmp)));
    }
    omp_rs_slug_to_ts_name(&project, &home, Path::new(&tmp))
}

/// `cwd` of the `{"type":"session"}` header within the first 32 non-empty lines
/// of a transcript (a `title` line may precede it). Lines that are not valid
/// UTF-8 JSON are skipped, not fatal.
fn header_cwd(path: &Path) -> Option<PathBuf> {
    use std::io::{BufRead, BufReader, Read};
    const HEADER_SCAN_LINES: usize = 32;
    let mut reader = BufReader::new(std::fs::File::open(path).ok()?.take(256 * 1024));
    let mut line = Vec::new();
    let mut scanned = 0;
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line).ok()? == 0 {
            return None;
        }
        let text = line.trim_ascii();
        if text.is_empty() {
            continue;
        }
        scanned += 1;
        if let Ok(header) = serde_json::from_slice::<serde_json::Value>(text) {
            if header.get("type").and_then(serde_json::Value::as_str) == Some("session") {
                if let Some(cwd) = header.get("cwd").and_then(serde_json::Value::as_str) {
                    return Some(PathBuf::from(cwd));
                }
            }
        }
        if scanned == HEADER_SCAN_LINES {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omp_rs_slugs_map_to_ts_names() {
        let (home, tmp) = (Path::new("/home/u"), Path::new("/tmp"));
        let map = |slug: &str| omp_rs_slug_to_ts_name(slug, home, tmp);
        assert_eq!(map("-home-u-omrs"), "-omrs");
        assert_eq!(map("-home-u"), "-");
        assert_eq!(map("-tmp"), "-tmp");
        assert_eq!(map("-tmp-x-y"), "-tmp-x-y");
        assert_eq!(map("-mnt-c-Temp"), "--mnt-c-Temp--");
        // A sibling sharing a string prefix is not inside home.
        assert_eq!(map("-home-user2"), "--home-user2--");
    }

    #[test]
    fn home_symlink_maps_like_its_target() {
        let dir = std::env::temp_dir().join(format!("omp-roots-home-{}", std::process::id()));
        let real = dir.join("real-home");
        let link = dir.join("link-home");
        std::fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let real = std::fs::canonicalize(&real).unwrap();
        let slug = format!("{}-proj", slugify(&real));
        // A tmp dir outside the slug's prefix, so only the home mapping applies.
        let mapped = omp_rs_slug_to_ts_name(&slug, &link, Path::new("/var/omp-no-such-tmp"));
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(mapped, "-proj");
    }

    #[test]
    fn header_cwd_skips_invalid_utf8_and_scans_32_non_empty_lines() {
        let dir = std::env::temp_dir().join(format!("omp-roots-hdr2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut late = Vec::new();
        for _ in 0..31 {
            late.extend_from_slice(b"\n{\"type\":\"title\"}\n");
        }
        late.extend_from_slice(b"{\"type\":\"session\",\"cwd\":\"/w/late\"}\n");
        let far = dir.join("far.jsonl");
        std::fs::write(&far, &late).unwrap();
        let mut bad = b"\xff\xfe not utf8\n".to_vec();
        bad.extend_from_slice(b"{\"type\":\"session\",\"cwd\":\"/w/after-garbage\"}\n");
        let garbled = dir.join("garbled.jsonl");
        std::fs::write(&garbled, &bad).unwrap();
        let mut beyond = Vec::new();
        for _ in 0..32 {
            beyond.extend_from_slice(b"{\"type\":\"title\"}\n");
        }
        beyond.extend_from_slice(b"{\"type\":\"session\",\"cwd\":\"/w/too-late\"}\n");
        let too_far = dir.join("too-far.jsonl");
        std::fs::write(&too_far, &beyond).unwrap();
        let got = (header_cwd(&far), header_cwd(&garbled), header_cwd(&too_far));
        std::fs::remove_dir_all(&dir).unwrap();
        // The 32nd non-empty line is the last one scanned.
        assert_eq!(got.0, Some(PathBuf::from("/w/late")));
        assert_eq!(got.1, Some(PathBuf::from("/w/after-garbage")));
        assert_eq!(got.2, None);
    }

    #[test]
    fn header_cwd_is_read_only_from_a_session_header() {
        let dir = std::env::temp_dir().join(format!("omp-roots-hdr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ok = dir.join("a.jsonl");
        std::fs::write(
            &ok,
            "{\"type\":\"title\"}\n{\"type\":\"session\",\"cwd\":\"/w/x\"}\n",
        )
        .unwrap();
        let other = dir.join("b.jsonl");
        std::fs::write(&other, "{\"type\":\"message\",\"cwd\":\"/w/x\"}\n").unwrap();
        assert_eq!(header_cwd(&ok), Some(PathBuf::from("/w/x")));
        assert_eq!(header_cwd(&other), None);
        assert_eq!(header_cwd(&dir.join("missing.jsonl")), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
