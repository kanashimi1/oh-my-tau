//! Offline scoring of historical session files with the same view builder and scorer as the
//! shadow hook.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::episode::parse_entries;
use crate::scorer::{ModelPaths, Scorer};
use crate::shadow::row;
use crate::view::episode_view;

/// Session files under `root` (a file itself, or every `*.jsonl` below a directory), sorted.
///
/// # Errors
/// Unreadable directory.
pub fn collect_files(root: &Path) -> Result<Vec<PathBuf>> {
    if root.is_file() {
        return Ok(vec![root.to_owned()]);
    }
    let mut out = Vec::new();
    let mut stack = vec![root.to_owned()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).with_context(|| format!("read dir {}", d.display()))? {
            let p = e?.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "jsonl") {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn sessions_root() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".omp/agent/sessions"))
}

/// Subagent sessions live at `sessions/<project>/<parent session>/<agent>.jsonl` (depth >= 3
/// below the sessions root); a path outside the root is never a subagent.
#[must_use]
pub fn is_subagent_path(p: &Path) -> bool {
    sessions_root()
        .and_then(|root| {
            p.strip_prefix(root)
                .ok()
                .map(|r| r.components().count() >= 3)
        })
        .unwrap_or(false)
}

/// Totals of an offline run.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Summary {
    pub files: usize,
    pub episodes: usize,
    pub failed: usize,
}

/// Score every episode of `files`, writing one JSON row per episode to `out`.
///
/// # Errors
/// Model load or output write failure (per-file read problems are skipped and counted).
pub fn judge_files(
    files: &[PathBuf],
    out: &mut dyn std::io::Write,
    paths: &ModelPaths,
    progress: &mut dyn FnMut(&Path, usize),
) -> Result<Summary> {
    let mut scorer = Scorer::load(paths)?;
    let ckpt = paths.ckpt.to_string_lossy().into_owned();
    let mut sum = Summary::default();
    for f in files {
        let Ok(entries) = omp_session::read_values(f) else {
            sum.failed += 1;
            continue;
        };
        sum.files += 1;
        let is_sub = is_subagent_path(f);
        let session_id = entries
            .iter()
            .find(|e| e.get("type").and_then(Value::as_str) == Some("session"))
            .and_then(|e| e.get("id").and_then(Value::as_str))
            .map_or_else(
                || {
                    f.file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default()
                },
                str::to_owned,
            );
        let eps = parse_entries(&entries, is_sub);
        for (i, ep) in eps.iter().enumerate() {
            let t0 = Instant::now();
            let view = episode_view(ep, is_sub);
            let s = match scorer.score(&view) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("{}: episode {i}: {e:#}", f.display());
                    sum.failed += 1;
                    continue;
                }
            };
            let mut r = row(
                &session_id,
                i as u64,
                ep.ts.as_deref().unwrap_or(""),
                &s,
                &ckpt,
                t0.elapsed().as_millis(),
            );
            r.insert("session_file".into(), json!(f.to_string_lossy()));
            r.insert("n_steps".into(), json!(ep.steps.len()));
            r.insert(
                "request_head".into(),
                json!(crate::pyfmt::take_chars(&ep.request, 80)),
            );
            writeln!(out, "{}", serde_json::to_string(&r)?)?;
            sum.episodes += 1;
        }
        progress(f, eps.len());
    }
    Ok(sum)
}
