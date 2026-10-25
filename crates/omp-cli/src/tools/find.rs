//! `find`: behavior search — describe what code does, get files and line
//! ranges. omp's tool spends a judge (LLM) cascade over a lexical prior; omp-rs
//! has no judge provider yet, so this is the lexical stage alone: files are
//! ranked by case-insensitive keyword occurrences (omp `jfind/lexical.ts`),
//! scored as a 0–1 probability, and rendered in omp's exact report format.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use pa_agent::types::AgentToolResult;
use pi_walker::{WalkDetail, WalkRequest};
use rayon::prelude::*;

use super::util::{
    cancelled, format_bytes, format_path_relative_to_cwd, resolve_to_cwd, str_arg, CancelFlag,
};
use super::Env;

/// Ranges shown per hit in the model-facing text (omp `RANGES_SHOWN`).
const RANGES_SHOWN: usize = 3;
/// Files listed in the report.
const FILES_SHOWN: usize = 10;
/// Below this score hits are weak evidence (omp's τ).
const THRESHOLD: f64 = 0.40;
const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;
const SNIPPET_CHARS: usize = 100;

struct Hit {
    rel: String,
    score: f64,
    lines_seen: usize,
    ranges: Vec<(u32, u32, String)>, // start, end (1-indexed), snippet
}

/// omp `jfind/text.ts` keyword extraction, applied to `grep_keywords` and the
/// plain-language query: lowercase, quoted phrases kept whole.
fn keywords(query: &str, extra: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |kw: String| {
        let kw = kw.trim().to_lowercase();
        if !kw.is_empty() && !out.contains(&kw) {
            out.push(kw);
        }
    };
    for kw in extra {
        push(kw.clone());
    }
    // Quoted phrases match whole.
    let mut rest = query;
    while let Some(start) = rest.find('"') {
        let after = &rest[start + 1..];
        if let Some(end) = after.find('"') {
            push(after[..end].to_string());
            rest = &after[end + 1..];
        } else {
            rest = after;
            break;
        }
    }
    for tok in rest.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.') {
        push(tok.to_string());
    }
    out
}

/// Score from occurrence count: 1 - 1/(1+n), a 0–1 probability-ish scale.
fn score(count: usize) -> f64 {
    1.0 - 1.0 / (1.0 + f64::from(u32::try_from(count).unwrap_or(u32::MAX)))
}

fn walk_files(root: &Path) -> Result<Vec<PathBuf>> {
    let request = WalkRequest::new(root.to_path_buf())
        .hidden(false)
        .gitignore(true)
        .detail(WalkDetail::Minimal)
        .emit_root(false);
    Ok(request
        .collect_files()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .iter()
        .map(|e| e.absolute_path(root))
        .collect())
}

/// Judge one file against the keywords: `(hit, files_read, file_bytes)` —
/// counters are zero when the file is skipped (too large, binary, unreadable)
/// or cancelled mid-scan.
fn judge_file(
    file: &Path,
    keywords: &[String],
    env: &Env,
    cancel: &CancelFlag,
) -> (Option<Hit>, usize, usize) {
    if cancelled(cancel) {
        return (None, 0, 0);
    }
    let Ok(m) = std::fs::metadata(file) else {
        return (None, 0, 0);
    };
    if m.len() > MAX_FILE_BYTES {
        return (None, 0, 0);
    }
    let Ok(bytes) = std::fs::read(file) else {
        return (None, 0, 0);
    };
    if bytes.contains(&0) {
        return (None, 0, 0);
    }
    let text = String::from_utf8_lossy(&bytes);
    let mut matched_lines: Vec<(u32, &str)> = Vec::new();
    let mut count = 0usize;
    for (i, line) in text.lines().enumerate() {
        let lower = line.to_lowercase();
        let line_hits = keywords
            .iter()
            .filter(|kw| lower.contains(kw.as_str()))
            .count();
        if line_hits > 0 {
            count += line_hits;
            matched_lines.push((u32::try_from(i + 1).unwrap_or(u32::MAX), line.trim()));
        }
    }
    let file_score = score(count);
    if file_score < THRESHOLD {
        return (None, 1, bytes.len());
    }
    // Contiguous matched lines become ranges, strongest (most hits) first.
    let mut ranges: Vec<(u32, u32, usize)> = Vec::new();
    for (ln, _) in &matched_lines {
        if let Some(last) = ranges.last_mut() {
            if *ln == last.1 + 1 {
                last.1 = *ln;
                last.2 += 1;
                continue;
            }
        }
        ranges.push((*ln, *ln, 1));
    }
    ranges.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    let mut hit_ranges = Vec::new();
    for (start, end, n) in ranges.into_iter().take(RANGES_SHOWN) {
        let snippet = matched_lines
            .iter()
            .find(|(ln, _)| *ln == start)
            .map_or("", |(_, l)| l)
            .chars()
            .take(SNIPPET_CHARS)
            .collect::<String>();
        hit_ranges.push((start, end, snippet));
        let _ = n;
    }
    (
        Some(Hit {
            rel: format_path_relative_to_cwd(file, &env.cwd, false),
            score: file_score,
            lines_seen: text.lines().count(),
            ranges: hit_ranges,
        }),
        1,
        bytes.len(),
    )
}

#[cfg(test)]
/// Run the `find` tool (lexical stage of omp's cascade).
///
/// # Errors
/// Empty query or an unresolvable scope path.
pub fn find(env: &Env, args: &serde_json::Value) -> Result<AgentToolResult> {
    find_with_cancel(env, args, &CancelFlag::default())
}

/// [`find`] with a cooperative cancel flag: the registry adapter flips it on
/// abort; the per-file scoring loop stops early instead of reading the rest
/// of the tree.
///
/// Per-file work (read + keyword scan) runs on the `rayon` pool inside the
/// registry's `spawn_blocking` section; hits are collected in input order and
/// sorted afterwards exactly as before, so output stays byte-identical.
///
/// # Errors
/// Empty query or an unresolvable scope path.
#[allow(clippy::too_many_lines)]
pub fn find_with_cancel(
    env: &Env,
    args: &serde_json::Value,
    cancel: &CancelFlag,
) -> Result<AgentToolResult> {
    let query = str_arg(args, "query")?.trim().to_string();
    if query.is_empty() {
        bail!("`query` must be a non-empty description");
    }
    let extra: Vec<String> = args
        .get("grep_keywords")
        .and_then(serde_json::Value::as_array)
        .map_or_else(Vec::new, |a| {
            a.iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_string)
                .collect()
        });
    let keywords = keywords(&query, &extra);
    if keywords.is_empty() {
        bail!("`grep_keywords` must contain at least one term");
    }

    let scope_input = args
        .get("path")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let root = if scope_input.is_empty() {
        env.cwd.clone()
    } else {
        resolve_to_cwd(scope_input, &env.cwd)
    };
    let meta =
        std::fs::metadata(&root).map_err(|_| anyhow::anyhow!("Path not found: {scope_input}"))?;
    let scope_path = if root == env.cwd {
        None
    } else {
        Some(format_path_relative_to_cwd(&root, &env.cwd, meta.is_dir()))
    };

    let started = std::time::Instant::now();
    let mut hits: Vec<Hit> = Vec::new();
    let mut files_read = 0usize;
    let mut file_bytes = 0usize;
    let files: Vec<PathBuf> = if meta.is_file() {
        vec![root]
    } else {
        walk_files(&root)?
    };
    let listed = files.len();

    // Rayon judges files in parallel (read + keyword scan is CPU-bound);
    // results are collected in input order, and the sort below (score desc,
    // path asc) plus the summed counters depend only on the multiset, so the
    // report matches a serial scan byte for byte.
    let judged: Vec<(Option<Hit>, usize, usize)> = files
        .par_iter()
        .map(|file| judge_file(file, &keywords, env, cancel))
        .collect();
    for (hit, read, bytes) in judged {
        files_read += read;
        file_bytes += bytes;
        if let Some(hit) = hit {
            hits.push(hit);
        }
    }

    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.rel.cmp(&b.rel)));
    hits.truncate(FILES_SHOWN);

    let where_ = scope_path
        .as_deref()
        .map_or_else(String::new, |p| format!(" in {p}"));
    let mut out = String::new();
    if hits.is_empty() {
        let _ = writeln!(out, "no hits for \"{query}\"{where_} (τ {THRESHOLD:.2})");
    } else {
        let _ = writeln!(
            out,
            "{} hit(s) for \"{query}\"{where_} (τ {THRESHOLD:.2}), strongest first",
            hits.len()
        );
        out.push('\n');
        for hit in &hits {
            let _ = writeln!(
                out,
                "{}  {:.2}  {} lines matched",
                hit.rel, hit.score, hit.lines_seen
            );
            for (start, end, snippet) in &hit.ranges {
                let span = if start == end {
                    format!("{start}")
                } else {
                    format!("{start}-{end}")
                };
                let _ = writeln!(out, "  {}:{span}  {:.2}  {snippet}", hit.rel, hit.score);
            }
        }
    }
    out.push('\n');
    let _ = writeln!(
        out,
        "listed {listed} · judged 0 · read {files_read} files ({}) · 0 requests · 0 tokens · $0.0000 · {} wall / 0:00 api",
        format_bytes(file_bytes),
        super::jobs::format_duration(u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)),
    );
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let hit_values: Vec<serde_json::Value> = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "rel": h.rel,
                "contentScore": h.score,
                "linesSeen": h.lines_seen,
                "ranges": h.ranges.iter().map(|(start, end, snippet)| serde_json::json!({
                    "start": start,
                    "end": end,
                    "p": h.score,
                    "snippet": snippet,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let mut details = serde_json::json!({
        "query": query,
        "keywords": keywords,
        "threshold": THRESHOLD,
        "cwd": env.cwd.display().to_string(),
        "elapsedMs": elapsed_ms,
        "hits": hit_values,
        "stats": {
            "filesRead": files_read,
            "inputTokens": 0,
            "cost": 0.0,
            "errors": 0,
            "failures": [],
        },
    });
    if let Some(p) = scope_path {
        details["scopePath"] = serde_json::json!(p);
    }
    let mut result = AgentToolResult::text(out.trim_end().to_string());
    result.details = details;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (tempfile::TempDir, Env) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src/alpha.rs"),
            "pub fn login_user() {\n    verify_password();\n    create_session();\n}\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("src/beta.rs"),
            "pub fn logout() { destroy_session(); }\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("README.md"), "# demo\nnothing here\n").unwrap();
        let env = Env::new(dir.path(), false);
        (dir, env)
    }

    #[test]
    fn ranks_files_by_keyword_hits_in_omp_format() {
        let (_dir, env) = fixture();
        let r = find(
            &env,
            &serde_json::json!({"query": "user login session", "grep_keywords": ["session"]}),
        )
        .unwrap();
        let text = match &r.content[0] {
            pa_agent::types::ToolResultContent::Text(t) => t.text.clone(),
            pa_agent::types::ToolResultContent::Image(image) => {
                panic!("unexpected content: {image:?}")
            }
        };
        assert!(
            text.starts_with("2 hit(s) for \"user login session\" (τ 0.40), strongest first"),
            "{text}"
        );
        let alpha = text
            .lines()
            .find(|l| l.starts_with("src/alpha.rs"))
            .unwrap();
        assert!(alpha.contains("  0."), "{alpha}");
        // omp hit line format: `path:start-end score snippet`; matched lines
        // 1 and 3 are not contiguous, so they list as separate ranges.
        assert!(text.contains("  src/alpha.rs:1  "), "{text}");
        assert!(text.contains("  src/alpha.rs:3  "), "{text}");
        assert!(text.contains("pub fn login_user()"), "{text}");
        assert!(
            text.contains("listed 3 · judged 0 · read 3 files"),
            "{text}"
        );
    }

    #[test]
    fn scoped_search_reports_no_hits() {
        let (_dir, env) = fixture();
        let r = find(
            &env,
            &serde_json::json!({"query": "zzz absent topic", "grep_keywords": ["zznotthere"], "path": "src"}),
        )
        .unwrap();
        let text = match &r.content[0] {
            pa_agent::types::ToolResultContent::Text(t) => t.text.clone(),
            pa_agent::types::ToolResultContent::Image(image) => {
                panic!("unexpected content: {image:?}")
            }
        };
        assert!(
            text.starts_with("no hits for \"zzz absent topic\" in src/ (τ 0.40)"),
            "{text}"
        );
    }

    #[test]
    fn rejects_empty_query_and_scope() {
        let (_dir, env) = fixture();
        let e = find(
            &env,
            &serde_json::json!({"query": "  ", "grep_keywords": ["x"]}),
        )
        .unwrap_err();
        assert_eq!(e.to_string(), "`query` must be a non-empty description");
        let e = find(
            &env,
            &serde_json::json!({"query": "x", "grep_keywords": ["x"], "path": "nope"}),
        )
        .unwrap_err();
        assert_eq!(e.to_string(), "Path not found: nope");
    }
}
