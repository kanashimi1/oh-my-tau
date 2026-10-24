//! `omp-rs gc` — session-file garbage collection, scoped strictly to session
//! JSONL files under `<agentDir>/sessions`. Mirrors the archive phase of TS
//! `cli/gc-cli.ts`: keep the newest sessions globally and per project, archive
//! older cold ones as gzip into `<agentDir>/archive/sessions`. **Dry-run by
//! default**; `--apply` performs the moves. Nothing is ever deleted outright —
//! archiving only relocates (and compresses) session files; the TS blob, WAL
//! and stale-marker phases are not ported (they do not touch session files).

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::{Context as _, Result};
use serde_json::json;

const DAY: Duration = Duration::from_hours(24);
/// TS `GC_WRITE_GRACE_MS`: a session written within the last 5 minutes is active.
const WRITE_GRACE: Duration = Duration::from_mins(5);

struct Options {
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

/// Parse the gc flags (TS `parseArchiveGcArgs`).
///
/// # Errors
/// Unknown flags, missing values, or non-integer numeric values.
fn parse_options(args: &[String]) -> Result<(Options, PathBuf)> {
    let mut opts = Options {
        apply: false,
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
            "--agent-dir" => {
                agent_dir = Some(PathBuf::from(
                    it.next().context("--agent-dir needs a value")?,
                ))
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
/// [--retain-newest-per-cwd n]`.
///
/// # Errors
/// I/O failure scanning or (with `--apply`) archiving session files.
pub fn run(args: &[String]) -> Result<i32> {
    let (opts, agent_dir) = parse_options(args)?;
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
    let mut bytes: u64 = 0;
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
        would_archive += 1;
        bytes += file.size;
        if opts.apply {
            match archive_session(&sessions_root, &archive_root, &file.path) {
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

/// TS `runArchiveGc` destination: `<archiveRoot>/<relative-path>.gz`; the
/// source file is removed only after the archive lands intact.
fn archive_session(sessions_root: &Path, archive_root: &Path, path: &Path) -> Result<()> {
    let relative = path.strip_prefix(sessions_root).unwrap_or(path);
    let dest = archive_root.join(relative).with_extension("jsonl.gz");
    if dest.exists() {
        anyhow::bail!("archive destination exists: {}", dest.display());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut input = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut encoder = flate2::write::GzEncoder::new(
        fs::File::create(&dest).with_context(|| format!("create {}", dest.display()))?,
        flate2::Compression::default(),
    );
    std::io::copy(&mut input, &mut encoder)?;
    encoder.flush()?;
    drop(encoder);
    // Verify the archive is readable before releasing the source. Streamed
    // size count (no read_to_end: sessions can be large).
    let mut check = flate2::read::GzDecoder::new(fs::File::open(&dest)?);
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
    fs::remove_file(path).with_context(|| format!("remove {}", path.display()))?;
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
        archive_session(&root.join("sessions"), &archive, &file).unwrap();
        assert!(!file.exists());
        let archived = archive.join("proj").join("2026-01-01_test.jsonl.gz");
        let mut decoded = String::new();
        flate2::read::GzDecoder::new(fs::File::open(&archived).unwrap())
            .read_to_string(&mut decoded)
            .unwrap();
        assert_eq!(decoded, "{\"type\":\"session\"}\n");
        fs::remove_dir_all(&root).ok();
    }
}
