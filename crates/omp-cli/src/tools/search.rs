//! `grep` (ripgrep engine crates over the `pi-walker` file walk) and `glob`
//! (`pi-walker`, mtime-ranked), with omp's result formats and notices.

use std::collections::{BTreeSet, HashMap};
use std::convert::Infallible;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{bail, Result};
use grep_regex::{RegexMatcher, RegexMatcherBuilder};
use grep_searcher::{
    BinaryDetection, Searcher, SearcherBuilder, Sink, SinkContext, SinkContextKind, SinkMatch,
};
use pa_agent::types::AgentToolResult;
use pi_edit::path_policy::canonical_key;
use pi_edit::store::seen_lines_from_body;
use pi_walker::{
    CompiledWalkGlob, DirectoryErrorMode, EmptyRecheck, FollowLinks, WalkDetail, WalkFilter,
    WalkOrder, WalkRank, WalkRequest,
};

use super::read::{parse_sel, LineRange, Selector};
use super::tree::{events, format_grouped_paths, Kind};
use super::util::{
    cut_bytes, format_bytes, format_path_relative_to_cwd, path_list, resolve_to_cwd, truncate_head,
    DEFAULT_MAX_BYTES,
};
use super::Env;

const DEFAULT_FILE_LIMIT: usize = 20;
const MULTI_FILE_PER_FILE_MATCHES: usize = 20;
const SINGLE_FILE_MATCHES: usize = 200;
const INTERNAL_TOTAL_CAP: usize = 2000;
const MAX_COLUMN: usize = 512;
const CONTEXT_BEFORE: usize = 1;
const CONTEXT_AFTER: usize = 3;
const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;
/// Files searched per parallel batch: bounds wasted work past
/// `INTERNAL_TOTAL_CAP` while staying above the walker pool's parallel
/// threshold (256 files).
const SEARCH_CHUNK_FILES: usize = 512;
const GLOB_DEFAULT_LIMIT: usize = 200;
const GLOB_MAX_LIMIT: usize = 200;

// -------------------------------------------------------------- patterns

fn has_glob_chars(s: &str) -> bool {
    s.contains(['*', '?', '[', '{'])
}

/// omp `parseFindPattern`: base directory, glob and whether a glob is present.
pub struct FindPattern {
    pub base: String,
    pub glob: String,
    pub has_glob: bool,
}

#[must_use]
pub fn parse_find_pattern(pattern: &str) -> FindPattern {
    let normalized = pattern.replace('\\', "/");
    let segments: Vec<&str> = normalized.split('/').collect();
    let Some(first) = segments.iter().position(|s| has_glob_chars(s)) else {
        return FindPattern {
            base: normalized,
            glob: "**/*".into(),
            has_glob: false,
        };
    };
    if first == 0 {
        let glob = if normalized.starts_with("**/") {
            normalized
        } else {
            format!("**/{normalized}")
        };
        return FindPattern {
            base: ".".into(),
            glob,
            has_glob: true,
        };
    }
    FindPattern {
        base: segments[..first].join("/"),
        glob: segments[first..].join("/"),
        has_glob: true,
    }
}

fn fix_unclosed_braces(mut p: String) -> String {
    let opens = p.matches('{').count();
    let closes = p.matches('}').count();
    for _ in closes..opens {
        p.push('}');
    }
    p
}

fn walk_depth_bound(pattern: &str) -> usize {
    if pattern.contains("**") || pattern.contains('{') {
        return usize::MAX;
    }
    pattern.split('/').filter(|s| !s.is_empty()).count().max(1)
}

fn build_walk_request(
    root: &Path,
    glob: &str,
    hidden: bool,
    gitignore: bool,
    detail: WalkDetail,
) -> Result<WalkRequest> {
    let pattern = fix_unclosed_braces(glob.replace('\\', "/"));
    let depth = walk_depth_bound(&pattern);
    let compiled = CompiledWalkGlob::new([pattern.clone()])
        .map_err(|e| anyhow::anyhow!("Invalid glob pattern: {e}"))?;
    let mentions_node_modules = pattern.contains("node_modules");
    Ok(WalkRequest::new(root.to_path_buf())
        .hidden(hidden)
        .gitignore(gitignore)
        .skip_git(true)
        .skip_node_modules(!mentions_node_modules)
        .follow_links(FollowLinks::Never)
        .detail(detail)
        .order(WalkOrder::Path)
        .emit_root(false)
        .depth(1, depth)
        .directory_errors(DirectoryErrorMode::SkipSkippable)
        .cache(false)
        .empty_recheck(EmptyRecheck::Configured)
        .filter(
            WalkFilter::all()
                .glob(compiled)
                .node_modules_unless_mentioned(mentions_node_modules),
        ))
}

// ------------------------------------------------------------------ glob

fn bool_arg(args: &serde_json::Value, key: &str) -> Option<bool> {
    args.get(key).and_then(serde_json::Value::as_bool)
}

/// Run the `glob` tool.
///
/// # Errors
/// Missing paths, invalid glob or limit.
// Layout mirrors omp's `glob` flow top-to-bottom; splitting it would obscure that parity.
// The limit casts below are lossless: both limits are small fixed constants (200).
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn with_details(text: String, details: serde_json::Value) -> AgentToolResult {
    let mut result = AgentToolResult::text(text);
    result.details = details;
    result
}

pub fn glob(env: &Env, args: &serde_json::Value) -> Result<AgentToolResult> {
    let mut patterns = path_list(args.get("path").and_then(serde_json::Value::as_str));
    if patterns.is_empty() {
        patterns.push(".".into());
    }
    let hidden = bool_arg(args, "hidden").unwrap_or(true);
    let gitignore = bool_arg(args, "gitignore").unwrap_or(true);
    let requested = args
        .get("limit")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(GLOB_DEFAULT_LIMIT as f64);
    if !requested.is_finite() || requested <= 0.0 {
        bail!("Limit must be a positive number");
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let limit = (requested.floor() as usize).clamp(1, GLOB_MAX_LIMIT);
    let clamp_notice = (requested > GLOB_MAX_LIMIT as f64)
        .then(|| format!("Requested limit {requested} clamped to the max of {GLOB_MAX_LIMIT}"));

    let single = patterns.len() == 1;
    let mut missing: Vec<String> = Vec::new();
    let mut merged: Vec<(String, f64)> = Vec::new();
    let mut scope = String::new();
    for pat in &patterns {
        let parsed = parse_find_pattern(pat.trim().trim_matches('"'));
        let base = resolve_to_cwd(&parsed.base, &env.cwd);
        if base == Path::new("/") {
            bail!("Searching from root directory '/' is not allowed");
        }
        if scope.is_empty() {
            scope = format_path_relative_to_cwd(&base, &env.cwd, false);
        }
        let Ok(meta) = std::fs::metadata(&base) else {
            if single {
                bail!("Path not found: {scope}");
            }
            missing.push(pat.clone());
            continue;
        };
        if !parsed.has_glob && meta.is_file() {
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0.0, |d| d.as_secs_f64() * 1000.0);
            merged.push((format_path_relative_to_cwd(&base, &env.cwd, false), mtime));
            continue;
        }
        if !meta.is_dir() {
            if single {
                bail!("Path is not a directory: {}", base.display());
            }
            continue;
        }
        let request = build_walk_request(&base, &parsed.glob, hidden, gitignore, WalkDetail::Full)?;
        let outcome = request
            .collect_ranked(WalkRank::MtimeDescPathAsc, limit)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        for entry in outcome.entries {
            let abs = entry.absolute_path(&base);
            let shown = format_path_relative_to_cwd(&abs, &env.cwd, entry.is_dir());
            merged.push((shown, entry.mtime.unwrap_or(0.0)));
        }
    }
    if patterns.len() > 1 && merged.is_empty() && missing.len() == patterns.len() {
        bail!("Path not found: {}", missing.join(", "));
    }
    let mut seen = BTreeSet::new();
    merged.retain(|(p, _)| seen.insert(p.clone()));
    merged.sort_by(|a, b| b.1.total_cmp(&a.1));
    let missing_note =
        (!missing.is_empty()).then(|| format!("Skipped missing paths: {}", missing.join(", ")));
    if merged.is_empty() {
        let mut parts = vec!["No files found matching pattern".to_string()];
        parts.extend(missing_note);
        let mut details = serde_json::json!({
            "scopePath": scope,
            "fileCount": 0,
            "files": [],
            "truncated": false,
            "cwd": env.cwd.display().to_string(),
        });
        if !missing.is_empty() {
            details["missingPaths"] = serde_json::json!(missing);
        }
        return Ok(with_details(parts.join("\n"), details));
    }
    let files: Vec<String> = merged.into_iter().map(|(p, _)| p).collect();
    let reached = files.len() >= limit;
    let limited = &files[..files.len().min(limit)];
    let mut out = format_grouped_paths(limited);
    let mut trailing: Vec<String> = Vec::new();
    trailing.extend(clamp_notice);
    trailing.extend(missing_note);
    if !trailing.is_empty() {
        let _ = write!(out, "\n\n{}", trailing.join("\n"));
    }
    let t = truncate_head(
        &out,
        if crate::methodology::raw_tool_output_requested() {
            usize::MAX
        } else {
            DEFAULT_MAX_BYTES
        },
    );
    let mut text = t.content.clone();
    let mut notices: Vec<String> = Vec::new();
    if t.truncated {
        notices.push(format!(
            "Showing lines 1-{} of {} ({} limit)",
            t.output_lines,
            t.total_lines,
            format_bytes(t.max_bytes)
        ));
    }
    if reached {
        let suggestion = (limit * 2).min(GLOB_MAX_LIMIT);
        notices.push(if suggestion > limit {
            format!("{limit} results limit reached. Use limit={suggestion} for more")
        } else {
            format!("{limit} results limit reached")
        });
    }
    if !notices.is_empty() {
        let _ = write!(text, "\n\n[{}]", notices.join(". "));
    }
    let mut details = serde_json::json!({
        "scopePath": scope,
        "fileCount": limited.len(),
        "files": limited,
        "truncated": reached || t.truncated,
        "cwd": env.cwd.display().to_string(),
    });
    if reached {
        details["resultLimitReached"] = serde_json::json!(limit);
    }
    if !missing.is_empty() {
        details["missingPaths"] = serde_json::json!(missing);
    }
    Ok(with_details(text, details))
}

// ------------------------------------------------------------------ grep

#[derive(Debug, Clone)]
struct Match {
    line_number: u64,
    line: String,
    matched: bool,
}

struct Collector {
    rows: Vec<Match>,
    matches: usize,
    cap: usize,
    truncated_lines: bool,
}

impl Collector {
    fn clip(&mut self, bytes: &[u8]) -> String {
        let text = String::from_utf8_lossy(bytes);
        let text = text.trim_end_matches(['\n', '\r']);
        if text.len() > MAX_COLUMN {
            self.truncated_lines = true;
            return format!("{}...", cut_bytes(text, MAX_COLUMN - 3));
        }
        text.to_string()
    }
}

impl Sink for Collector {
    type Error = std::io::Error;

    fn matched(&mut self, _s: &Searcher, m: &SinkMatch<'_>) -> Result<bool, Self::Error> {
        let first = m.line_number().unwrap_or(0);
        // A multi-line match yields one row per line.
        for (i, raw) in m.lines().enumerate() {
            let line = self.clip(raw);
            self.rows.push(Match {
                line_number: first + i as u64,
                line,
                matched: true,
            });
        }
        self.matches += 1;
        Ok(self.matches <= self.cap)
    }

    fn context(&mut self, _s: &Searcher, c: &SinkContext<'_>) -> Result<bool, Self::Error> {
        if matches!(c.kind(), SinkContextKind::Other) && self.matches > self.cap {
            return Ok(false);
        }
        let line = self.clip(c.bytes());
        self.rows.push(Match {
            line_number: c.line_number().unwrap_or(0),
            line,
            matched: false,
        });
        Ok(true)
    }
}

fn escape_unescaped_parens(p: &str) -> String {
    let mut out = String::with_capacity(p.len() + 4);
    let mut chars = p.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                out.push(c);
                if let Some(n) = chars.next() {
                    out.push(n);
                }
            }
            '(' | ')' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

fn build_matcher(pattern: &str, ignore_case: bool, multiline: bool) -> Result<RegexMatcher> {
    let build = |p: &str| -> Result<RegexMatcher, grep_regex::Error> {
        let make = |line_terminated: bool| {
            let mut b = RegexMatcherBuilder::new();
            b.case_insensitive(ignore_case).multi_line(multiline);
            if line_terminated {
                b.line_terminator(Some(b'\n'));
            }
            b.build(p)
        };
        if !multiline {
            if let Ok(m) = make(true) {
                return Ok(m);
            }
        }
        make(false)
    };
    let err = match build(pattern) {
        Ok(m) => return Ok(m),
        Err(e) => e,
    };
    let message = err.to_string();
    if message.contains("unclosed group") || message.contains("unopened group") {
        let escaped = escape_unescaped_parens(pattern);
        if escaped != pattern {
            if let Ok(m) = build(&escaped) {
                return Ok(m);
            }
        }
    }
    build(&regex::escape(pattern)).map_err(|_| anyhow::anyhow!("Invalid regex: {message}"))
}

#[derive(Debug, Clone)]
struct Target {
    base: PathBuf,
    glob: Option<String>,
    ranges: Option<Vec<LineRange>>,
}

struct FileHits {
    shown: String,
    absolute: PathBuf,
    matches: Vec<Match>,
    match_count: usize,
}

fn in_ranges(line: u64, ranges: &[LineRange]) -> bool {
    let line = usize::try_from(line).unwrap_or(usize::MAX);
    ranges
        .iter()
        .any(|r| line >= r.start && r.end.is_none_or(|e| line <= e))
}

fn search_file(
    searcher: &mut Searcher,
    matcher: &RegexMatcher,
    path: &Path,
    per_file_cap: usize,
    ranges: Option<&[LineRange]>,
) -> Option<(Vec<Match>, usize, bool)> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() {
        return None;
    }
    let mut sink = Collector {
        rows: Vec::new(),
        matches: 0,
        cap: per_file_cap,
        truncated_lines: false,
    };
    let file = std::fs::File::open(path).ok()?;
    let reader = std::io::Read::take(file, MAX_FILE_BYTES);
    searcher.search_reader(matcher, reader, &mut sink).ok()?;
    let mut rows = sink.rows;
    let mut count = sink.matches;
    if let Some(r) = ranges {
        rows.retain(|m| in_ranges(m.line_number, r));
        count = rows.iter().filter(|m| m.matched).count();
    }
    if count == 0 {
        return None;
    }
    Some((rows, count.min(per_file_cap + 1), sink.truncated_lines))
}

fn split_targets(env: &Env, entries: &[String]) -> (Vec<Target>, Vec<String>) {
    let mut targets = Vec::new();
    let mut missing = Vec::new();
    for entry in entries {
        let entry = entry.trim().trim_matches('"').to_string();
        // line-range selector on an existing file: `src/foo.ts:50-100`
        let mut ranges = None;
        let mut path_part = entry.clone();
        if !resolve_to_cwd(&entry, &env.cwd).exists() {
            if let Some((p, sel)) = entry.rsplit_once(':') {
                if let Ok(Selector::Lines { ranges: r, .. }) = parse_sel(sel) {
                    if resolve_to_cwd(p, &env.cwd).is_file() {
                        ranges = Some(r);
                        path_part = p.to_string();
                    }
                }
            }
        }
        let literal = resolve_to_cwd(&path_part, &env.cwd);
        if literal.exists() {
            targets.push(Target {
                base: literal,
                glob: None,
                ranges,
            });
            continue;
        }
        let parsed = parse_find_pattern(&path_part);
        if parsed.has_glob {
            // A glob spelled without a directory prefix matches at any depth.
            let no_dir_prefix = !path_part.contains('/');
            let glob = if no_dir_prefix && !parsed.glob.starts_with("**") {
                format!("**/{}", parsed.glob)
            } else {
                parsed.glob
            };
            let dir = resolve_to_cwd(&parsed.base, &env.cwd);
            if dir.is_dir() {
                targets.push(Target {
                    base: dir,
                    glob: Some(glob),
                    ranges: None,
                });
                continue;
            }
        }
        missing.push(entry);
    }
    (targets, missing)
}

/// Run the `grep` tool.
///
/// # Errors
/// Empty pattern, bad `skip`, or no path resolving.
#[allow(clippy::too_many_lines)]
pub fn grep(env: &Env, args: &serde_json::Value) -> Result<AgentToolResult> {
    let pattern = super::util::str_arg(args, "pattern")?;
    if pattern.trim().is_empty() {
        bail!("Pattern must not be empty");
    }
    let skip = match args.get("skip") {
        None | Some(serde_json::Value::Null) => 0usize,
        Some(v) => match v.as_f64() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Some(n) if n.is_finite() && n >= 0.0 => n.floor() as usize,
            _ => bail!("Skip must be a non-negative number"),
        },
    };
    let mut entries = path_list(args.get("path").and_then(serde_json::Value::as_str));
    if entries.is_empty() {
        entries.push(".".into());
    }
    let ignore_case = !bool_arg(args, "case").unwrap_or(true);
    let gitignore = bool_arg(args, "gitignore").unwrap_or(true);
    let multiline = pattern.contains('\n') || pattern.contains("\\n");

    let (targets, missing) = split_targets(env, &entries);
    if targets.is_empty() {
        bail!(
            "Path not found: {}; list each target in the semicolon-delimited `path`",
            missing.join(", ")
        );
    }
    let scope_path = targets
        .iter()
        .map(|t| format_path_relative_to_cwd(&t.base, &env.cwd, false))
        .collect::<Vec<_>>()
        .join(", ");
    let search_path = targets[0].base.display().to_string();
    let matcher = build_matcher(pattern, ignore_case, multiline)?;
    // One `Searcher` per walker-pool worker (a `Searcher` is not `Sync`).
    let new_searcher = move || {
        SearcherBuilder::new()
            .binary_detection(BinaryDetection::quit(0))
            .line_number(true)
            .multi_line(multiline)
            .before_context(CONTEXT_BEFORE)
            .after_context(CONTEXT_AFTER)
            .build()
    };

    let single_file = targets.len() == 1 && targets[0].glob.is_none() && targets[0].base.is_file();
    let is_multi_scope = !single_file;
    let per_file_cap = if is_multi_scope {
        MULTI_FILE_PER_FILE_MATCHES
    } else {
        SINGLE_FILE_MATCHES
    };

    let mut hits: Vec<FileHits> = Vec::new();
    let mut seen_files: BTreeSet<PathBuf> = BTreeSet::new();
    let mut total_matches = 0usize;
    let mut limit_reached = false;
    let mut lines_truncated = false;
    let mut oversized: Vec<String> = Vec::new();
    for target in &targets {
        let files: Vec<PathBuf> = if target.base.is_file() {
            vec![target.base.clone()]
        } else {
            let glob = target.glob.clone().unwrap_or_else(|| "**/*".to_string());
            let request =
                build_walk_request(&target.base, &glob, true, gitignore, WalkDetail::Minimal)?;
            let mut files: Vec<PathBuf> = request
                .collect_files()
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .iter()
                .map(|e| e.absolute_path(&target.base))
                .collect();
            files.sort();
            files
        };
        if target.base.is_file() {
            if let Ok(m) = std::fs::metadata(&target.base) {
                if m.len() > MAX_FILE_BYTES {
                    oversized.push(format_path_relative_to_cwd(&target.base, &env.cwd, false));
                }
            }
        }
        let files: Vec<PathBuf> = files
            .into_iter()
            .filter(|f| seen_files.insert(f.clone()))
            .collect();
        // Search each chunk on the walker pool (cold page cache makes this
        // per-file I/O latency bound), then fold results in path order so
        // output and the total-match cap stay identical to a serial scan.
        for chunk in files.chunks(SEARCH_CHUNK_FILES) {
            let slots: Vec<OnceLock<Option<(Vec<Match>, usize, bool)>>> =
                std::iter::repeat_with(OnceLock::new)
                    .take(chunk.len())
                    .collect();
            let indices: Vec<usize> = (0..chunk.len()).collect();
            let searched = pi_walker::parallel_for_each_init(
                &indices,
                new_searcher,
                |searcher, &i| -> std::result::Result<(), Infallible> {
                    let found = search_file(
                        searcher,
                        &matcher,
                        &chunk[i],
                        per_file_cap,
                        target.ranges.as_deref(),
                    );
                    let _ = slots[i].set(found);
                    Ok(())
                },
            );
            let Ok(()) = searched;
            for (file, slot) in chunk.iter().zip(slots) {
                let Some((rows, count, truncated)) = slot.into_inner().flatten() else {
                    continue;
                };
                lines_truncated |= truncated;
                total_matches += count;
                hits.push(FileHits {
                    shown: format_path_relative_to_cwd(file, &env.cwd, false),
                    absolute: file.clone(),
                    matches: rows,
                    match_count: count,
                });
                if total_matches >= INTERNAL_TOTAL_CAP {
                    limit_reached = true;
                    break;
                }
            }
            if limit_reached {
                break;
            }
        }
        if limit_reached {
            break;
        }
    }

    // Per-file cap (keep whole context rows up to the cap-th match).
    for h in &mut hits {
        if h.match_count > per_file_cap {
            let mut seen = 0usize;
            let mut cut = h.matches.len();
            for (i, m) in h.matches.iter().enumerate() {
                if m.matched {
                    seen += 1;
                    if seen > per_file_cap {
                        cut = i;
                        break;
                    }
                }
            }
            h.matches.truncate(cut);
        }
    }

    let total_files = hits.len();
    let total_files_label = if limit_reached {
        format!("{total_files}+")
    } else {
        total_files.to_string()
    };
    let can_paginate = is_multi_scope;
    let skip_files = if can_paginate {
        skip.min(total_files)
    } else {
        0
    };
    let window: Vec<&FileHits> = if can_paginate {
        hits.iter()
            .skip(skip_files)
            .take(DEFAULT_FILE_LIMIT)
            .collect()
    } else {
        hits.iter().collect()
    };
    let file_limit_reached = can_paginate && total_files > skip_files + DEFAULT_FILE_LIMIT;
    let next_skip = skip_files + window.len();
    let limit_message = file_limit_reached.then(|| {
        format!(
            "Showing files {}-{next_skip} of {total_files_label}. Use skip={next_skip} for the next page, or narrow paths/pattern.",
            skip_files + 1
        )
    });

    let missing_note =
        (!missing.is_empty()).then(|| format!("Skipped missing paths: {}", missing.join(", ")));
    let oversized_note = (!oversized.is_empty()).then(|| {
        format!(
            "Searched only the first {}MB of large files (matches past the {}MB window are not shown; use `read` for the rest): {}",
            MAX_FILE_BYTES / (1024 * 1024),
            MAX_FILE_BYTES / (1024 * 1024),
            oversized.join(", ")
        )
    });
    let warning = [missing_note, oversized_note]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("\n");

    if window.is_empty() {
        let no_match = if can_paginate && skip > 0 && total_files > 0 && skip_files >= total_files {
            format!(
                "No more results ({total_files_label} files total; skip={skip} is past the end)"
            )
        } else {
            "No matches found".to_string()
        };
        let text = if warning.is_empty() {
            no_match
        } else {
            format!("{no_match}\n{warning}")
        };
        let mut details = serde_json::json!({
            "scopePath": scope_path,
            "searchPath": search_path,
            "cwd": env.cwd.display().to_string(),
            "matchCount": 0,
            "fileCount": 0,
            "files": [],
            "fileMatches": [],
            "truncated": false,
            "displayContent": "",
        });
        if !missing.is_empty() {
            details["missingPaths"] = serde_json::json!(missing);
        }
        return Ok(with_details(text, details));
    }

    // Render per file; hashline snapshot tags bind to the displayed rows.
    let render = |h: &FileHits| -> (Vec<String>, Vec<String>, Option<String>) {
        let key = canonical_key(&h.absolute);
        let tag = if env.hashline {
            env.store.record_file(&key, None)
        } else {
            None
        };
        let mut out: Vec<String> = Vec::new();
        let mut disp: Vec<String> = Vec::new();
        let mut last: Option<u64> = None;
        let gutter_w = h
            .matches
            .iter()
            .map(|m| m.line_number.to_string().len())
            .max()
            .unwrap_or(1)
            + 1;
        for m in &h.matches {
            if last.is_some_and(|l| m.line_number > l + 1) {
                out.push("...".into());
                disp.push(format!("{:gutter_w$}\u{2502}...", ""));
            }
            let marker = if m.matched { '*' } else { ' ' };
            // TS `formatCodeFrameLine`: `*2│beta` / ` 1│alpha`, width-padded.
            let gutter = if m.matched {
                format!("*{}", m.line_number)
            } else {
                m.line_number.to_string()
            };
            disp.push(format!("{gutter:>gutter_w$}\u{2502}{}", m.line));
            if tag.is_some() {
                out.push(format!("{marker}{}:{}", m.line_number, m.line));
            } else {
                out.push(format!("{marker}{}|{}", m.line_number, m.line));
            }
            last = Some(m.line_number);
        }
        if let Some(t) = &tag {
            env.store
                .record_seen_lines(&key, t, &seen_lines_from_body(&out.join("\n")));
        }
        (out, disp, tag)
    };
    let mut model: Vec<String> = Vec::new();
    let mut display: Vec<String> = Vec::new();
    if is_multi_scope {
        let entries: Vec<(String, bool)> =
            window.iter().map(|h| (h.shown.clone(), false)).collect();
        let rendered: HashMap<usize, (Vec<String>, Vec<String>, Option<String>)> = window
            .iter()
            .enumerate()
            .map(|(i, h)| (i, render(h)))
            .collect();
        let mut emitted = false;
        for e in events(&entries) {
            let hashes = "#".repeat(e.depth + 1);
            if emitted && (e.depth == 0 || e.kind == Kind::Dir) {
                model.push(String::new());
                display.push(String::new());
            }
            match e.kind {
                Kind::Dir => {
                    model.push(format!("{hashes} {}/", e.name));
                    display.push(format!("{hashes} {}/", e.name));
                    emitted = true;
                }
                Kind::File => {
                    let (lines, disp, tag) = &rendered[&e.key];
                    if lines.is_empty() {
                        // Skipped files leave no header; undo the separator.
                        if model.last().is_some_and(String::is_empty) {
                            model.pop();
                            display.pop();
                        }
                        continue;
                    }
                    emitted = true;
                    let suffix = tag.as_ref().map(|t| format!("#{t}")).unwrap_or_default();
                    model.push(format!("{hashes} {}{suffix}", e.name));
                    display.push(format!("{hashes} {}{suffix}", e.name));
                    model.extend(lines.iter().cloned());
                    display.extend(disp.iter().cloned());
                }
            }
        }
    } else {
        for h in &window {
            let (lines, disp, tag) = render(h);
            if lines.is_empty() {
                continue;
            }
            if !model.is_empty() {
                model.push(String::new());
                display.push(String::new());
            }
            if let Some(t) = tag {
                model.push(format!("[{}#{t}]", h.shown));
            }
            model.extend(lines);
            display.extend(disp);
        }
    }
    if let Some(m) = limit_message {
        model.push(String::new());
        model.push(m);
    }
    if !warning.is_empty() {
        model.push(String::new());
        model.push(warning);
    }
    let raw = model.join("\n");
    let t = truncate_head(
        &raw,
        if crate::methodology::raw_tool_output_requested() {
            usize::MAX
        } else {
            DEFAULT_MAX_BYTES
        },
    );
    let mut text = t.content.clone();
    let mut notices: Vec<String> = Vec::new();
    if t.truncated {
        notices.push(format!(
            "Showing lines 1-{} of {} ({} limit)",
            t.output_lines,
            t.total_lines,
            format_bytes(t.max_bytes)
        ));
    }
    if lines_truncated {
        notices.push(format!("Some lines truncated to {MAX_COLUMN} bytes"));
    }
    if !notices.is_empty() {
        let _ = write!(text, "\n\n[{}]", notices.join(". "));
    }
    let match_total: usize = window
        .iter()
        .map(|h| h.matches.iter().filter(|m| m.matched).count())
        .sum();
    let files: Vec<String> = window.iter().map(|h| h.shown.clone()).collect();
    let file_matches: Vec<serde_json::Value> = window
        .iter()
        .map(|h| {
            serde_json::json!({
                "path": h.shown,
                "count": h.matches.iter().filter(|m| m.matched).count(),
            })
        })
        .collect();
    let mut details = serde_json::json!({
        "scopePath": scope_path,
        "searchPath": search_path,
        "cwd": env.cwd.display().to_string(),
        "matchCount": match_total,
        "fileCount": files.len(),
        "files": files,
        "fileMatches": file_matches,
        "truncated": file_limit_reached || limit_reached || t.truncated || lines_truncated,
        "displayContent": display.join("\n"),
    });
    if file_limit_reached {
        details["fileLimitReached"] = serde_json::json!(DEFAULT_FILE_LIMIT);
    }
    if lines_truncated {
        details["linesTruncated"] = serde_json::json!(true);
    }
    if !missing.is_empty() {
        details["missingPaths"] = serde_json::json!(missing);
    }
    Ok(with_details(text, details))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(r: &AgentToolResult) -> String {
        r.content
            .iter()
            .filter_map(|c| match c {
                pa_agent::types::ToolResultContent::Text(t) => Some(t.text.clone()),
                pa_agent::types::ToolResultContent::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::create_dir_all(p.join("src/inner")).unwrap();
        std::fs::create_dir_all(p.join("ignored")).unwrap();
        std::fs::create_dir_all(p.join(".git")).unwrap();
        std::fs::write(p.join(".gitignore"), "ignored/\n").unwrap();
        std::fs::write(p.join("src/a.rs"), "fn alpha() {}\nfn beta() {}\n").unwrap();
        std::fs::write(p.join("src/inner/b.rs"), "// beta here\nlet x = 1;\n").unwrap();
        std::fs::write(p.join("ignored/c.rs"), "fn beta_ignored() {}\n").unwrap();
        std::fs::write(p.join("notes.md"), "beta in notes\n").unwrap();
        dir
    }

    #[test]
    fn find_pattern_parsing_matches_omp() {
        let p = parse_find_pattern("**/*.json");
        assert_eq!(
            (p.base.as_str(), p.glob.as_str(), p.has_glob),
            (".", "**/*.json", true)
        );
        let p = parse_find_pattern("*.ts");
        assert_eq!((p.base.as_str(), p.glob.as_str()), (".", "**/*.ts"));
        let p = parse_find_pattern("/abs/path/**/*.ts");
        assert_eq!((p.base.as_str(), p.glob.as_str()), ("/abs/path", "**/*.ts"));
        let p = parse_find_pattern("src/app");
        assert_eq!(
            (p.base.as_str(), p.glob.as_str(), p.has_glob),
            ("src/app", "**/*", false)
        );
    }

    #[test]
    fn glob_lists_grouped_respecting_gitignore() {
        let dir = fixture();
        let env = Env::new(dir.path(), true);
        let out = text_of(&glob(&env, &serde_json::json!({"path": "**/*.rs"})).unwrap());
        assert!(out.contains("a.rs") && out.contains("b.rs"), "{out}");
        assert!(!out.contains("c.rs"), "gitignored file leaked: {out}");
        assert!(out.contains("# src/"), "{out}");
        let none = text_of(&glob(&env, &serde_json::json!({"path": "*.zzz"})).unwrap());
        assert_eq!(none, "No files found matching pattern");
        let err = glob(&env, &serde_json::json!({"path": "missing/dir"}))
            .unwrap_err()
            .to_string();
        assert_eq!(err, "Path not found: missing/dir");
    }

    #[test]
    fn glob_and_grep_emit_card_details() {
        let dir = fixture();
        let env = Env::new(dir.path(), true);
        let g = glob(&env, &serde_json::json!({"path": "**/*.rs"})).unwrap();
        assert_eq!(g.details["fileCount"], 2, "{:?}", g.details);
        assert_eq!(g.details["scopePath"], ".");
        assert_eq!(g.details["files"].as_array().unwrap().len(), 2);
        let r = grep(&env, &serde_json::json!({"pattern": "beta", "path": "."})).unwrap();
        assert_eq!(r.details["matchCount"], 3, "{:?}", r.details);
        assert_eq!(r.details["fileCount"], 3);
        let display = r.details["displayContent"].as_str().unwrap();
        assert!(display.contains("*2\u{2502}fn beta() {}"), "{display}");
        assert!(display.starts_with("# "), "{display}");
        let none = grep(&env, &serde_json::json!({"pattern": "zzzz", "path": "."})).unwrap();
        assert_eq!(none.details["matchCount"], 0);
    }

    #[test]
    fn glob_limit_notice() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..5 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), "x").unwrap();
        }
        let env = Env::new(dir.path(), true);
        let out = text_of(&glob(&env, &serde_json::json!({"path": "*.txt", "limit": 3})).unwrap());
        assert!(
            out.ends_with("[3 results limit reached. Use limit=6 for more]"),
            "{out}"
        );
    }

    #[test]
    fn grep_directory_scope_groups_files_with_tags_and_context() {
        let dir = fixture();
        let env = Env::new(dir.path(), true);
        let out = text_of(&grep(&env, &serde_json::json!({"pattern": "beta"})).unwrap());
        assert!(
            out.contains("## a.rs#") || out.contains("# a.rs#") || out.contains("a.rs#"),
            "{out}"
        );
        assert!(out.contains("*2:fn beta() {}"), "{out}");
        assert!(
            out.contains(" 1:fn alpha() {}"),
            "context row missing: {out}"
        );
        assert!(out.contains("*1:// beta here"), "{out}");
        assert!(!out.contains("beta_ignored"), "{out}");
        assert!(out.contains("notes.md"), "{out}");
    }

    #[test]
    fn grep_single_file_scope_flat_with_header() {
        let dir = fixture();
        let env = Env::new(dir.path(), true);
        let out = text_of(
            &grep(
                &env,
                &serde_json::json!({"pattern": "alpha", "path": "src/a.rs"}),
            )
            .unwrap(),
        );
        assert!(out.starts_with("[src/a.rs#"), "{out}");
        assert!(out.contains("*1:fn alpha() {}"), "{out}");
    }

    #[test]
    fn grep_case_flag_no_match_and_invalid_input() {
        let dir = fixture();
        let env = Env::new(dir.path(), true);
        let none = text_of(&grep(&env, &serde_json::json!({"pattern": "ALPHA"})).unwrap());
        assert_eq!(none, "No matches found");
        let some = text_of(
            &grep(
                &env,
                &serde_json::json!({"pattern": "ALPHA", "case": false}),
            )
            .unwrap(),
        );
        assert!(some.contains("alpha"), "{some}");
        assert_eq!(
            grep(&env, &serde_json::json!({"pattern": "  "}))
                .unwrap_err()
                .to_string(),
            "Pattern must not be empty"
        );
        // unbalanced paren is escaped instead of failing
        let paren = text_of(&grep(&env, &serde_json::json!({"pattern": "alpha("})).unwrap());
        assert!(paren.contains("No matches found") || paren.contains("alpha"));
    }

    #[test]
    fn grep_line_range_selector_filters_matches() {
        let dir = fixture();
        let env = Env::new(dir.path(), true);
        let out = text_of(
            &grep(
                &env,
                &serde_json::json!({"pattern": "fn", "path": "src/a.rs:2-2"}),
            )
            .unwrap(),
        );
        assert!(out.contains("*2:fn beta() {}"), "{out}");
        assert!(!out.contains("*1:fn alpha"), "{out}");
    }

    #[test]
    fn grep_glob_scope_and_bare_glob_any_depth() {
        let dir = fixture();
        let env = Env::new(dir.path(), true);
        let out = text_of(
            &grep(
                &env,
                &serde_json::json!({"pattern": "beta", "path": "*.rs"}),
            )
            .unwrap(),
        );
        assert!(out.contains("a.rs") && out.contains("b.rs"), "{out}");
        assert!(!out.contains("notes.md"), "{out}");
        let direct = text_of(
            &grep(
                &env,
                &serde_json::json!({"pattern": "beta", "path": "src/*.rs"}),
            )
            .unwrap(),
        );
        assert!(
            direct.contains("a.rs") && !direct.contains("b.rs"),
            "{direct}"
        );
    }
}
