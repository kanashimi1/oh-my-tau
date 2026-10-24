//! Session-scoped scheme providers: `local`, `artifact`, `agent`, `history`.
//! Ports of the TS `internal-urls/{local,artifact-agent,history}-protocol.ts`
//! handlers plus the shared `registry-helpers.ts` scans.

pub use super::{agent, artifact, history, local};

use std::path::{Path, PathBuf};

use crate::Registry;

/// Register the four session-scoped providers (TS router constructor order:
/// agent, history, artifact, local).
///
/// # Errors
/// Registration inconsistency is a programming error.
pub fn register(reg: &mut Registry) -> anyhow::Result<()> {
    reg.register(agent::AgentProvider)?;
    reg.register(history::HistoryProvider)?;
    reg.register(artifact::ArtifactProvider)?;
    reg.register(local::LocalProvider)?;
    Ok(())
}

/// TS `formatByteSize`: `512 B`, `1.5 KiB`, `2.0 MiB`, `1.0 GiB`.
pub fn format_byte_size(bytes: u64) -> String {
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let kib = bytes as f64 / 1024.0;
    if kib < 1024.0 {
        return format!("{kib:.1} KiB");
    }
    let mib = kib / 1024.0;
    if mib < 1024.0 {
        return format!("{mib:.1} MiB");
    }
    format!("{:.1} GiB", mib / 1024.0)
}

/// TS `formatAgo`: `Ns/Nm/Nh/Nd ago`.
pub fn format_ago(timestamp_ms: u64, now_ms: u64) -> String {
    let secs = now_ms.saturating_sub(timestamp_ms) / 1000;
    if secs < 60 {
        return format!("{secs}s ago");
    }
    let mins = secs / 60;
    if mins < 60 {
        return format!("{mins}m ago");
    }
    let hours = mins / 60;
    if hours < 24 {
        return format!("{hours}h ago");
    }
    format!("{}d ago", hours / 24)
}

/// Every artifacts dir the process knows (TS `artifactsDirsFromRegistry`
/// reduced to omp-rs's single-session world): the caller's session artifacts
/// dir first, then any extras registered globally.
pub fn artifacts_dirs(ctx: &crate::UrlCtx) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut add = |dir: Option<PathBuf>| {
        if let Some(d) = dir {
            if !dirs.contains(&d) {
                dirs.push(d);
            }
        }
    };
    add(ctx.artifacts_dir());
    add(crate::global_session_file().and_then(|f| crate::artifacts_directory_for(&f)));
    add(EXTRA_ARTIFACTS_DIR.read().expect("lock").clone());
    dirs
}

static EXTRA_ARTIFACTS_DIR: std::sync::RwLock<Option<PathBuf>> = std::sync::RwLock::new(None);

/// Register an extra artifacts dir (TS `registerArtifactsDir`).
pub fn register_artifacts_dir(dir: Option<PathBuf>) {
    *EXTRA_ARTIFACTS_DIR.write().expect("lock") = dir;
}

/// Recursively scan artifacts dirs for `<id>.jsonl` transcripts (TS
/// `sessionFilesFromDisk`): recursive to depth 8, first hit wins,
/// `__advisor*.jsonl` excluded.
pub fn session_files_from_disk(preferred: Option<&Path>) -> Vec<(String, PathBuf)> {
    let mut found: Vec<(String, PathBuf)> = Vec::new();
    let mut seen_dirs: Vec<PathBuf> = Vec::new();
    let scan =
        |dir: &Path, depth: usize, found: &mut Vec<(String, PathBuf)>, seen: &mut Vec<PathBuf>| {
            fn scan_impl(
                dir: &Path,
                depth: usize,
                found: &mut Vec<(String, PathBuf)>,
                seen: &mut Vec<PathBuf>,
            ) {
                if depth > 8 || seen.iter().any(|d| d == dir) {
                    return;
                }
                seen.push(dir.to_path_buf());
                let Ok(entries) = std::fs::read_dir(dir) else {
                    return;
                };
                for entry in entries.flatten() {
                    let path = entry.path();
                    let Ok(file_type) = entry.file_type() else {
                        continue;
                    };
                    if file_type.is_dir() {
                        scan_impl(&path, depth + 1, found, seen);
                        continue;
                    }
                    if !file_type.is_file() {
                        continue;
                    }
                    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                        continue;
                    };
                    if !name.ends_with(".jsonl") || name.starts_with("__advisor") {
                        continue;
                    }
                    let id = name[..name.len() - ".jsonl".len()].to_string();
                    if !found.iter().any(|(existing, _)| *existing == id) {
                        found.push((id, path));
                    }
                }
            }
            scan_impl(dir, depth, found, seen);
        };
    if let Some(p) = preferred {
        scan(p, 0, &mut found, &mut seen_dirs);
    }
    let empty_ctx = crate::UrlCtx::default();
    for dir in artifacts_dirs(&empty_ctx) {
        scan(&dir, 0, &mut found, &mut seen_dirs);
    }
    found
}

/// Simple fuzzy score for `Did you mean` suggestions (subsequence match;
/// TS uses `fuzzyFilter` — same candidates, same cap).
pub fn fuzzy_suggest(candidates: &[String], query: &str, max: usize) -> Vec<String> {
    let lower = query.to_lowercase();
    let mut scored: Vec<(usize, &String)> = candidates
        .iter()
        .filter(|c| **c != query)
        .filter_map(|c| {
            let cl = c.to_lowercase();
            let mut score = 0usize;
            let mut idx = 0usize;
            for ch in lower.chars() {
                match cl[idx..].find(ch) {
                    Some(off) => {
                        idx += off + ch.len_utf8();
                        score += off;
                    }
                    None => return None,
                }
            }
            Some((score, c))
        })
        .collect();
    scored.sort_by_key(|(s, _)| *s);
    scored
        .into_iter()
        .take(max)
        .map(|(_, c)| c.clone())
        .collect()
}
