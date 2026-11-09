//! `omp-rs gc` — session-file garbage collection, scoped to session JSONL
//! files under `<agentDir>/sessions` and their per-session artifact
//! directories. Mirrors the archive phase of TS `cli/gc-cli.ts`: keep the
//! newest sessions globally and per project, archive older cold ones as gzip
//! into `<agentDir>/archive/sessions` (artifacts move with their session).
//! **Dry-run by default**; `--apply` performs the moves. Nothing is ever
//! deleted outright — archiving only relocates (and compresses) files; the TS
//! blob, WAL and stale-marker phases are not ported (they do not touch session
//! files).

use std::collections::BTreeMap;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::{Context as _, Result};
use serde_json::json;

const DAY: Duration = Duration::from_hours(24);
/// TS `GC_WRITE_GRACE_MS`: a session written within the last 5 minutes is active.
const WRITE_GRACE: Duration = Duration::from_mins(5);

const USAGE: &str = r"usage: omt gc [--apply] [--json] [--agent-dir <dir>] [--cold-archive-after-days <n>]
                  [--retain-newest-global <n>] [--retain-newest-per-cwd <n>]

Archives cold session JSONL files, with their artifact directories, as gzip
under <agentDir>/archive/sessions. Dry-run unless --apply is given.

  --apply                         perform the archive (default: dry-run, touches nothing)
  --json                          print the result as JSON
  --agent-dir <dir>               agent directory (default: $OMP_AGENT_DIR or ~/.omp/agent)
  --cold-archive-after-days <n>   archive sessions idle for n days (default: 30)
  --retain-newest-global <n>      always keep the n newest sessions (default: 20)
  --retain-newest-per-cwd <n>     keep the n newest sessions per project (default: 10)
  -h, --help                      show this help and exit
";

struct Options {
    help: bool,
    apply: bool,
    json: bool,
    cold_after: Duration,
    retain_global: usize,
    retain_per_cwd: usize,
}

struct SessionFile {
    path: PathBuf,
    /// Seconds since epoch of the last modification.
    mtime: u64,
    size: u64,
    /// Project slug (parent directory name), TS's per-cwd grouping key.
    project: String,
}

/// Per-session artifact directory (`<ts>_<uuid>/` beside `<ts>_<uuid>.jsonl`,
/// the layout `omp_urls::artifacts_directory_for` resolves), flattened.
struct ArtifactTree {
    dir: PathBuf,
    /// Regular files with their sizes.
    files: Vec<(PathBuf, u64)>,
    /// Subdirectories, parents before children.
    dirs: Vec<PathBuf>,
    /// Total bytes of `files`.
    bytes: u64,
}

/// Parse the gc flags (TS `parseArchiveGcArgs`).
///
/// # Errors
/// Unknown flags, missing values, or non-integer numeric values.
fn parse_options(args: &[String]) -> Result<(Options, PathBuf)> {
    let mut opts = Options {
        apply: false,
        help: false,
        json: false,
        cold_after: 30 * DAY,
        retain_global: 20,
        retain_per_cwd: 10,
    };
    let mut agent_dir: Option<PathBuf> = None;
    let mut it = args.iter().peekable();
    while let Some(arg) = it.next() {
        let mut number = |name: &str| -> Result<usize> {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("{name} needs a value"))?
                .parse()
                .map_err(|_| anyhow::anyhow!("{name} needs an integer"))
        };
        match arg.as_str() {
            "--apply" => opts.apply = true,
            "--json" => opts.json = true,
            "--help" | "-h" => opts.help = true,
            "--agent-dir" => {
                agent_dir = Some(PathBuf::from(
                    it.next().context("--agent-dir needs a value")?,
                ));
            }
            "--cold-archive-after-days" => {
                opts.cold_after = Duration::from_secs(u64::try_from(number(arg)?)? * 86_400);
            }
            "--retain-newest-global" => opts.retain_global = number(arg)?,
            "--retain-newest-per-cwd" => opts.retain_per_cwd = number(arg)?,
            other if other.starts_with('-') => anyhow::bail!("unknown flag: {other}"),
            other => anyhow::bail!("unexpected argument: {other}"),
        }
    }

    let agent_dir = agent_dir.unwrap_or_else(|| omp_providers::AgentDir::default().0);
    Ok((opts, agent_dir))
}

/// Entry point: `omp-rs gc [--apply] [--json] [--agent-dir <dir>]
/// [--cold-archive-after-days n] [--retain-newest-global n]
/// [--retain-newest-per-cwd n]`; `--help` prints the flags and defaults.
///
/// # Errors
/// I/O failure scanning or (with `--apply`) archiving session files.
#[allow(clippy::too_many_lines)] // one archive pass: parse, scan, plan, apply, report; the stages share state
pub fn run(args: &[String]) -> Result<i32> {
    let (opts, agent_dir) = parse_options(args)?;
    if opts.help {
        print!("{USAGE}");
        return Ok(0);
    }
    let sessions_root = agent_dir.join("sessions");
    let archive_root = agent_dir.join("archive").join("sessions");

    let scanned = scan_sessions(&sessions_root);
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let now_secs = now.as_secs();

    let mut skipped_active = 0usize;
    let mut candidates: Vec<SessionFile> = Vec::new();
    for file in scanned {
        if Duration::from_secs(now_secs.saturating_sub(file.mtime)) <= WRITE_GRACE {
            skipped_active += 1;
        } else {
            candidates.push(file);
        }
    }

    // Newest-first across everything, then mark the retained ones.
    candidates.sort_by_key(|f| std::cmp::Reverse(f.mtime));
    let mut kept = vec![false; candidates.len()];
    let mut per_cwd: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, file) in candidates.iter().enumerate() {
        if i < opts.retain_global {
            kept[i] = true;
        }
        let count = per_cwd.entry(&file.project).or_insert(0);
        if *count < opts.retain_per_cwd {
            *count += 1;
            kept[i] = true;
        }
    }

    let mut would_archive = 0usize;
    let mut archived = 0usize;
    // Session JSONL plus its artifacts (`bytes` includes `artifact_bytes`).
    let mut bytes: u64 = 0;
    let mut artifact_bytes: u64 = 0;
    let mut errors: Vec<String> = Vec::new();
    for (file, kept) in candidates.iter().zip(kept.iter()) {
        if *kept {
            continue;
        }
        if Duration::from_secs(now_secs.saturating_sub(file.mtime)) < opts.cold_after {
            continue; // not cold yet
        }
        if held_open_elsewhere(&file.path) {
            skipped_active += 1; // a running process still owns the file
            continue;
        }
        let artifacts = match scan_artifacts(&file.path) {
            Ok(tree) => tree,
            Err(e) => {
                // Unreadable artifacts: keep the session rather than split it.
                errors.push(format!("{}: {e:#}", file.path.display()));
                continue;
            }
        };
        would_archive += 1;
        bytes += file.size + artifacts.bytes;
        artifact_bytes += artifacts.bytes;
        if opts.apply {
            match archive_session(&sessions_root, &archive_root, &file.path, &artifacts) {
                Ok(()) => archived += 1,
                Err(e) => errors.push(format!("{}: {e:#}", file.path.display())),
            }
        }
    }

    let result = json!({
        "agentDir": agent_dir,
        "apply": opts.apply,
        "archive": {
            "scanned": candidates.len() + skipped_active,
            "skippedActive": skipped_active,
            "keptNewestGlobal": opts.retain_global,
            "keptNewestPerCwd": opts.retain_per_cwd,
            "wouldArchive": would_archive,
            "archived": archived,
            "bytes": bytes,
            "artifactBytes": artifact_bytes,
            "errors": errors,
        },
    });

    if opts.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        let line = if opts.apply { "applied" } else { "dry-run" };
        println!("GC {line} ({})", agent_dir.display());
        println!(
            "sessions: {archived}/{would_archive} archived ({})",
            format_bytes(bytes)
        );
        println!("  artifacts: {}", format_bytes(artifact_bytes));
        if skipped_active > 0 {
            println!("sessions skipped active: {skipped_active}");
        }
        let n_errors = result["archive"]["errors"].as_array().map_or(0, Vec::len);
        if n_errors > 0 {
            println!("session errors: {n_errors}");
        }
        if !opts.apply && would_archive > 0 {
            println!(
                "\nrerun with --apply to gzip + move these into {}",
                archive_root.display()
            );
        }
    }
    Ok(0)
}

fn scan_sessions(root: &Path) -> Vec<SessionFile> {
    let mut out = Vec::new();
    if !root.is_dir() {
        return out;
    }
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.filter_map(std::result::Result::ok) {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "jsonl") {
                let Ok(meta) = entry.metadata() else { continue };
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs());
                out.push(SessionFile {
                    project: path
                        .parent()
                        .and_then(Path::file_name)
                        .map_or_else(|| "?".to_string(), |p| p.to_string_lossy().into_owned()),
                    path,
                    mtime,
                    size: meta.len(),
                });
            }
        }
    }
    out
}

/// TS `tryAcquireSessionFileLease`: a session file a running process still
/// holds open is live and must not be archived (the port has no lease
/// sidecars, so probe `/proc/<pid>/fd` on Linux). Conservative: an
/// unprobeable state is treated as live, mirroring "an unknown owner is
/// treated as live". Non-Linux falls back to the write grace only.
#[cfg(target_os = "linux")]
fn held_open_elsewhere(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(real) = fs::canonicalize(path) else {
        return true; // vanished or unreadable: do not touch
    };
    let target = real.as_os_str().as_bytes();
    let Ok(procs) = fs::read_dir("/proc") else {
        return true; // unprobeable → live
    };
    for proc in procs.filter_map(std::result::Result::ok) {
        if !proc.file_name().as_bytes().iter().all(u8::is_ascii_digit) {
            continue;
        }
        let Ok(fds) = fs::read_dir(proc.path().join("fd")) else {
            continue;
        };
        for fd in fds.filter_map(std::result::Result::ok) {
            if let Ok(link) = fs::read_link(fd.path()) {
                if link.as_os_str().as_bytes() == target {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(not(target_os = "linux"))]
fn held_open_elsewhere(_path: &Path) -> bool {
    false
}

/// Flatten a session's artifact directory. A missing directory is an empty
/// tree. An entry that is neither a regular file nor a directory (a symlink,
/// say) is an error: the caller keeps the session rather than archive half of it.
fn scan_artifacts(session: &Path) -> Result<ArtifactTree> {
    let dir = session.with_extension(""); // `<ts>_<uuid>.jsonl` → `<ts>_<uuid>`
    let mut tree = ArtifactTree {
        dir: dir.clone(),
        files: Vec::new(),
        dirs: Vec::new(),
        bytes: 0,
    };
    if !dir.is_dir() {
        return Ok(tree);
    }
    let mut stack = vec![dir];
    while let Some(current) = stack.pop() {
        for entry in
            fs::read_dir(&current).with_context(|| format!("read {}", current.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                tree.dirs.push(path.clone());
                stack.push(path);
            } else if file_type.is_file() {
                let len = entry.metadata()?.len();
                tree.bytes += len;
                tree.files.push((path, len));
            } else {
                anyhow::bail!("not a regular file or directory: {}", path.display());
            }
        }
    }
    Ok(tree)
}

/// TS `runArchiveGc` destination: `<archiveRoot>/<relative-path>.gz`; the
/// source file is removed only after the archive lands intact. The session's
/// artifact directory moves too (same relative layout, each file gzipped), so
/// an archived session leaves nothing behind in `sessions/`.
fn archive_session(
    sessions_root: &Path,
    archive_root: &Path,
    path: &Path,
    artifacts: &ArtifactTree,
) -> Result<()> {
    let relative = path.strip_prefix(sessions_root).unwrap_or(path);
    let dest = archive_root.join(relative).with_extension("jsonl.gz");
    let artifact_dest = archive_root.join(
        artifacts
            .dir
            .strip_prefix(sessions_root)
            .unwrap_or(artifacts.dir.as_path()),
    );
    let has_dir = artifacts.dir.is_dir();
    if dest.exists() {
        anyhow::bail!("archive destination exists: {}", dest.display());
    }
    // Artifacts first, the JSONL last: if an artifact fails, the session
    // file is still in `sessions/`, so the next run picks the session up
    // again and resumes with the artifacts that are left (each archived
    // file is already gone from the source; a clash on one that is not
    // still fails that file).
    for (file, _) in &artifacts.files {
        let rel = file.strip_prefix(&artifacts.dir).unwrap_or(file);
        let mut name = rel.as_os_str().to_owned();
        name.push(".gz");
        archive_file(file, &artifact_dest.join(name))?;
    }
    if has_dir {
        // `dirs` lists parents before children, so reversed removes children first.
        for dir in artifacts
            .dirs
            .iter()
            .rev()
            .chain(std::iter::once(&artifacts.dir))
        {
            fs::remove_dir(dir).with_context(|| format!("remove {}", dir.display()))?;
        }
    }
    archive_file(path, &dest)
}

/// Gzip `src` into `dest` (parents created) and remove `src` only once the
/// archive is fully written, flushed, and reads back to the source size.
fn archive_file(src: &Path, dest: &Path) -> Result<()> {
    if dest.exists() {
        anyhow::bail!("archive destination exists: {}", dest.display());
    }
    let result = write_archive(src, dest);
    if result.is_err() && src.exists() {
        // Drop a partial archive so a rerun is not blocked by it; the source
        // is still intact.
        let _ = fs::remove_file(dest);
    }
    result
}

fn write_archive(src: &Path, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut input = fs::File::open(src).with_context(|| format!("open {}", src.display()))?;
    let mut encoder = flate2::write::GzEncoder::new(
        fs::File::create(dest).with_context(|| format!("create {}", dest.display()))?,
        flate2::Compression::default(),
    );
    std::io::copy(&mut input, &mut encoder)?;
    // `finish` writes the gzip trailer (`Drop` would swallow that error), then
    // the file is synced before the source may be released.
    encoder.finish()?.sync_all()?;
    // Verify the archive is readable before releasing the source. Streamed
    // size count (no read_to_end: sessions can be large).
    let mut check = flate2::read::GzDecoder::new(fs::File::open(dest)?);
    let mut counted = 0u64;
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        let n = check.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        counted += n as u64;
    }
    if counted != input.metadata()?.len() {
        anyhow::bail!("archive size mismatch: {}", dest.display());
    }
    fs::remove_file(src).with_context(|| format!("remove {}", src.display()))?;
    Ok(())
}

/// TS `formatBytes`: `512B`, `1.5KB`, `2.3MB`, `1.2GB`.
#[allow(clippy::cast_precision_loss)] // byte counts << 2^52; the division result is formatted to 1 decimal
fn format_bytes(bytes: u64) -> String {
    let f = bytes as f64;
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", f / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}MB", f / (1024.0 * 1024.0))
    } else {
        format!("{:.1}GB", f / (1024.0 * 1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_matches_ts() {
        assert_eq!(format_bytes(512), "512B");
        assert_eq!(format_bytes(1536), "1.5KB");
        assert_eq!(format_bytes(2_412_000), "2.3MB");
    }

    /// Review #5: a session file another (or this) process holds open is
    /// live: `gc --apply` must not archive it, and a closed file passes.
    #[test]
    fn held_open_file_is_treated_as_live() {
        let file =
            std::env::temp_dir().join(format!("omp-rs-gc-live-{}-{}", std::process::id(), "x"));
        std::fs::write(&file, "x").unwrap();
        let open = fs::File::open(&file).unwrap();
        assert!(held_open_elsewhere(&file), "open fd must read as live");
        drop(open);
        let _ = fs::remove_file(&file);
    }

    #[test]
    fn archive_roundtrip_gzips_and_removes_source() {
        let root = std::env::temp_dir().join(format!("omp-rs-gc-test-{}", std::process::id()));
        let sessions = root.join("sessions").join("proj");
        let archive = root.join("archive").join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        let file = sessions.join("2026-01-01_test.jsonl");
        std::fs::write(&file, "{\"type\":\"session\"}\n").unwrap();
        archive_session(
            &root.join("sessions"),
            &archive,
            &file,
            &scan_artifacts(&file).unwrap(),
        )
        .unwrap();
        assert!(!file.exists());
        let archived = archive.join("proj").join("2026-01-01_test.jsonl.gz");
        let mut decoded = String::new();
        flate2::read::GzDecoder::new(fs::File::open(&archived).unwrap())
            .read_to_string(&mut decoded)
            .unwrap();
        assert_eq!(decoded, "{\"type\":\"session\"}\n");
        fs::remove_dir_all(&root).ok();
    }

    fn gunzip(path: &Path) -> String {
        let mut decoded = String::new();
        flate2::read::GzDecoder::new(fs::File::open(path).unwrap())
            .read_to_string(&mut decoded)
            .unwrap();
        decoded
    }

    /// Every entry under `root`, sorted, for before/after comparisons.
    fn tree_of(root: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in fs::read_dir(&dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path.clone());
                }
                out.push(path);
            }
        }
        out.sort();
        out
    }

    /// Age `path` past the cold threshold so `run` treats it as a candidate.
    fn make_cold(path: &Path) {
        let old = SystemTime::now() - DAY * 40;
        fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(old)
            .unwrap();
    }

    #[test]
    fn archive_moves_artifact_dir_gzipped_and_leaves_nothing_behind() {
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("sessions");
        let archive = root.path().join("archive").join("sessions");
        let proj = sessions.join("proj");
        let artifacts = proj.join("2026-01-01_s1");
        fs::create_dir_all(artifacts.join("nested")).unwrap();
        let file = proj.join("2026-01-01_s1.jsonl");
        fs::write(&file, "{\"type\":\"session\"}\n").unwrap();
        fs::write(artifacts.join("0.log"), "out\n").unwrap();
        fs::write(artifacts.join("nested").join("1.bash.log"), "bash\n").unwrap();

        let tree = scan_artifacts(&file).unwrap();
        assert_eq!(tree.bytes, 4 + 5);
        archive_session(&sessions, &archive, &file, &tree).unwrap();

        assert!(!file.exists() && !artifacts.exists(), "session left behind");
        assert_eq!(fs::read_dir(&proj).unwrap().count(), 0);
        let archived_dir = archive.join("proj").join("2026-01-01_s1");
        assert_eq!(gunzip(&archived_dir.join("0.log.gz")), "out\n");
        assert_eq!(
            gunzip(&archived_dir.join("nested").join("1.bash.log.gz")),
            "bash\n"
        );
        assert_eq!(
            gunzip(&archive.join("proj").join("2026-01-01_s1.jsonl.gz")),
            "{\"type\":\"session\"}\n"
        );
    }

    /// A failing artifact leaves the session JSONL in place (so the next run
    /// retries it) and the rerun resumes with the remaining artifacts.
    #[cfg(unix)]
    #[test]
    fn failed_artifact_keeps_session_for_a_resumable_rerun() {
        use std::os::unix::fs::PermissionsExt as _;
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("sessions");
        let archive = root.path().join("archive").join("sessions");
        let proj = sessions.join("proj");
        let artifacts = proj.join("2026-01-01_s1");
        fs::create_dir_all(&artifacts).unwrap();
        let file = proj.join("2026-01-01_s1.jsonl");
        fs::write(&file, "{\"type\":\"session\"}\n").unwrap();
        fs::write(artifacts.join("0.log"), "zero\n").unwrap();
        let locked = artifacts.join("1.log");
        fs::write(&locked, "one\n").unwrap();
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
        if fs::File::open(&locked).is_ok() {
            eprintln!("running as root: permissions not enforced; skipping");
            return;
        }

        let tree = scan_artifacts(&file).unwrap();
        archive_session(&sessions, &archive, &file, &tree).expect_err("unreadable artifact");
        assert!(file.exists(), "the session must stay for a retry");

        fs::set_permissions(&locked, fs::Permissions::from_mode(0o644)).unwrap();
        let tree = scan_artifacts(&file).unwrap();
        archive_session(&sessions, &archive, &file, &tree).unwrap();
        assert_eq!(
            fs::read_dir(&proj).unwrap().count(),
            0,
            "session left behind"
        );
        let archived_dir = archive.join("proj").join("2026-01-01_s1");
        assert_eq!(gunzip(&archived_dir.join("0.log.gz")), "zero\n");
        assert_eq!(gunzip(&archived_dir.join("1.log.gz")), "one\n");
        assert!(archive.join("proj").join("2026-01-01_s1.jsonl.gz").exists());
    }

    #[test]
    fn dry_run_touches_nothing() {
        let root = tempfile::tempdir().unwrap();
        let proj = root.path().join("sessions").join("proj");
        fs::create_dir_all(proj.join("s1")).unwrap();
        let file = proj.join("2026-01-01_s1.jsonl");
        fs::write(&file, "{\"type\":\"session\"}\n").unwrap();
        fs::write(proj.join("s1").join("0.log"), "out\n").unwrap();
        make_cold(&file);
        let before = tree_of(root.path());
        let args: Vec<String> = [
            "--agent-dir",
            root.path().to_str().unwrap(),
            "--retain-newest-global",
            "0",
            "--retain-newest-per-cwd",
            "0",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        assert_eq!(run(&args).unwrap(), 0);
        assert_eq!(tree_of(root.path()), before);
        assert!(!root.path().join("archive").exists());
    }

    #[test]
    fn help_flag_exits_zero() {
        for flag in ["--help", "-h"] {
            assert_eq!(run(&[flag.to_string()]).unwrap(), 0);
        }
    }
}
