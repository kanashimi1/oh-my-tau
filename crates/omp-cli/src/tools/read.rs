//! `read`: one `path` parameter with inline selectors (`:50-200`, `:-60`,
//! `:raw`, `a-b,c-d`). Text files render as a `[path#TAG]` hashline snapshot
//! header followed by `N:line` rows (omp `read.ts`, hashline display mode);
//! directories render as omp's workspace tree.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::LazyLock;
use std::time::SystemTime;

use anyhow::{bail, Context, Result};
use pa_agent::types::AgentToolResult;
use pi_edit::path_policy::canonical_key;
use pi_edit::store::seen_lines_from_body;
use pi_edit::text::{normalize_to_lf, strip_bom};
use regex::Regex;

use super::util::{
    format_age, format_bytes, format_path_relative_to_cwd, resolve_to_cwd, shorten_path,
    truncate_head, DEFAULT_MAX_BYTES,
};
use super::Env;

const DEFAULT_LIMIT: usize = 300;
const MAX_LINES: usize = 3000;
const LEADING_CONTEXT: usize = 1;
const TRAILING_CONTEXT: usize = 3;
const MAX_COLUMNS: usize = 768;
const DIR_MAX_DEPTH: usize = 2;
const DIR_CHILD_LIMIT: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineRange {
    pub start: usize,
    pub end: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selector {
    None,
    Raw,
    Lines { ranges: Vec<LineRange>, raw: bool },
    Tail { count: usize, raw: bool },
}

static RANGE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^L?(\d+)(?:(\.\.|[-+])L?(\d+)?)?$").expect("valid line range regex")
});
static TAIL_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^-(\d+)$").expect("valid tail regex"));

fn parse_chunk(chunk: &str, pin_bare: bool) -> Result<Option<LineRange>> {
    let Some(c) = RANGE_RE.captures(chunk) else { return Ok(None) };
    let start: usize = c[1].parse().unwrap_or(usize::MAX);
    if start < 1 {
        bail!("Line selector 0 is invalid; lines are 1-indexed. Use :1.");
    }
    let sep = c.get(2).map(|m| if m.as_str() == ".." { "-" } else { m.as_str() });
    let rhs: Option<usize> = c.get(3).and_then(|m| m.as_str().parse().ok());
    let end = match sep {
        Some("+") => match rhs {
            Some(n) if n >= 1 => Some(start + n - 1),
            _ => bail!("Invalid range {start}+{}: count must be >= 1.", rhs.unwrap_or(0)),
        },
        Some(_) => match rhs {
            Some(e) if e < start => bail!("Invalid range {start}-{e}: end must be >= start."),
            other => other,
        },
        None => pin_bare.then_some(start),
    };
    Ok(Some(LineRange { start, end }))
}

/// omp `parseLineRanges`: comma list, sorted and merged.
fn parse_ranges(sel: &str) -> Result<Option<Vec<LineRange>>> {
    let chunks: Vec<&str> = sel.split(',').collect();
    let pin = chunks.len() > 1;
    let mut parsed = Vec::new();
    for ch in chunks {
        match parse_chunk(ch, pin)? {
            Some(r) => parsed.push(r),
            None => return Ok(None),
        }
    }
    parsed.sort_by_key(|r| r.start);
    let mut merged: Vec<LineRange> = Vec::new();
    for r in parsed {
        match merged.last_mut() {
            None => merged.push(r),
            Some(last) => {
                let Some(last_end) = last.end else { continue };
                if r.start <= last_end + 1 {
                    if r.end.is_none_or(|e| e > last_end) {
                        last.end = r.end;
                    }
                } else {
                    merged.push(r);
                }
            }
        }
    }
    Ok(Some(merged))
}

fn parse_range_or_tail(chunk: &str, raw: bool) -> Result<Option<Selector>> {
    if let Some(ranges) = parse_ranges(chunk)? {
        return Ok(Some(Selector::Lines { ranges, raw }));
    }
    if let Some(c) = TAIL_RE.captures(chunk) {
        let count: usize = c[1].parse().unwrap_or(0);
        if count < 1 {
            bail!("Tail selector -0 is invalid; use :-N with N >= 1 to read the last N lines.");
        }
        return Ok(Some(Selector::Tail { count, raw }));
    }
    Ok(None)
}

/// omp `parseSel` for the selector kinds this port serves.
///
/// # Errors
/// Invalid range bounds or a recognised-but-malformed compound selector.
pub fn parse_sel(sel: &str) -> Result<Selector> {
    if sel.is_empty() {
        return Ok(Selector::None);
    }
    let invalid = || {
        anyhow::anyhow!(
            "Invalid selector ':{sel}'. Use :N, :N-M, :N+K, :N- (open-ended), :-N (last N lines), a comma-separated list of ranges, :raw, or a range combined with raw (e.g. :raw:50-100)."
        )
    };
    let looks_read_like = |c: &str| {
        let l = c.to_lowercase();
        l == "raw"
            || l == "conflicts"
            || l == "img"
            || TAIL_RE.is_match(c)
            || matches!(parse_ranges(c), Ok(Some(_)))
    };
    if sel.contains(':') {
        let chunks: Vec<&str> = sel.split(':').collect();
        if let [a, b] = chunks[..] {
            let a_raw = a.eq_ignore_ascii_case("raw");
            let b_raw = b.eq_ignore_ascii_case("raw");
            let range_chunk = if a_raw {
                Some(b)
            } else if b_raw {
                Some(a)
            } else {
                None
            };
            if let Some(rc) = range_chunk {
                if let Some(parsed) = parse_range_or_tail(rc, true)? {
                    return Ok(parsed);
                }
            }
        }
        if chunks.iter().all(|c| looks_read_like(c)) {
            return Err(invalid());
        }
        return Ok(Selector::None);
    }
    if sel.eq_ignore_ascii_case("raw") {
        return Ok(Selector::Raw);
    }
    if sel.eq_ignore_ascii_case("conflicts") || sel.eq_ignore_ascii_case("img") {
        bail!("Selector ':{sel}' is not supported by this build; use :N-M, :-N or :raw.");
    }
    Ok(parse_range_or_tail(sel, false)?.unwrap_or(Selector::None))
}

fn is_raw(sel: &Selector) -> bool {
    matches!(sel, Selector::Raw | Selector::Lines { raw: true, .. } | Selector::Tail { raw: true, .. })
}

/// Split `raw` into a filesystem path (existing) plus selector. A literal path
/// that exists always wins (omp `splitPathAndSelPreferringLiteral`).
fn split_path_and_sel(raw: &str, cwd: &Path) -> Result<(String, Selector)> {
    if resolve_to_cwd(raw, cwd).exists() {
        return Ok((raw.to_string(), Selector::None));
    }
    let chunks: Vec<&str> = raw.split(':').collect();
    for take in 1..=2usize.min(chunks.len().saturating_sub(1)) {
        let path = chunks[..chunks.len() - take].join(":");
        let sel = chunks[chunks.len() - take..].join(":");
        if path.is_empty() {
            continue;
        }
        let parsed = parse_sel(&sel)?;
        if parsed != Selector::None && resolve_to_cwd(&path, cwd).exists() {
            return Ok((path, parsed));
        }
    }
    Ok((raw.to_string(), Selector::None))
}

/// Lines addressable by hashline anchors (omp `splitAddressableFileLines`).
fn addressable_lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<&str> = text.split('\n').collect();
    if text.ends_with('\n') {
        lines.pop();
    }
    lines
}

fn truncate_columns(line: &str) -> String {
    let mut chars = line.char_indices();
    match chars.nth(MAX_COLUMNS) {
        Some((idx, _)) => format!("{}…", &line[..idx]),
        None => line.to_string(),
    }
}

/// Run the `read` tool.
///
/// # Errors
/// Missing path, unreadable file, binary content or invalid selector.
// Layout mirrors omp's `read` flow top-to-bottom; splitting it would obscure that parity.
#[allow(clippy::too_many_lines)]
pub fn read(env: &Env, args: &serde_json::Value) -> Result<AgentToolResult> {
    let raw_path = super::util::str_arg(args, "path")?;
    if let Some(rest) = raw_path.strip_prefix("proc://") {
        return super::jobs::read_proc(env, rest);
    }
    let (path_part, sel) = split_path_and_sel(raw_path, &env.cwd)?;
    let absolute = resolve_to_cwd(&path_part, &env.cwd);
    let meta = std::fs::metadata(&absolute)
        .map_err(|_| anyhow::anyhow!("Path '{path_part}' not found"))?;
    if meta.is_dir() {
        return read_directory(&absolute, &sel);
    }
    if !meta.is_file() {
        bail!("Cannot read '{path_part}': it is a special file, not a regular file or directory.");
    }
    let bytes = std::fs::read(&absolute).with_context(|| format!("cannot read {path_part}"))?;
    if bytes.contains(&0) {
        bail!("Cannot read '{path_part}': binary file ({}).", format_bytes(bytes.len()));
    }
    let text = String::from_utf8_lossy(&bytes);
    let (_, without_bom) = strip_bom(&text);
    let normalized = normalize_to_lf(without_bom);
    let lines = addressable_lines(&normalized);
    let total = lines.len();
    let raw = is_raw(&sel);
    let hash_lines = !raw && env.hashline;

    // Resolve the selector against the line count.
    let sel = match sel {
        Selector::Tail { count, raw } => {
            let start = total.saturating_sub(count) + 1;
            Selector::Lines { ranges: vec![LineRange { start: start.max(1), end: Some(total.max(start)) }], raw }
        }
        other => other,
    };
    let ranges: Vec<LineRange> = match &sel {
        Selector::Lines { ranges, .. } => ranges.clone(),
        _ => Vec::new(),
    };
    let selector_base = path_part;
    let display = format_path_relative_to_cwd(&absolute, &env.cwd, false);
    let display = if Path::new(&display).is_absolute() { shorten_path(&display) } else { display };

    let mut body = String::new();
    let mut rows = 0usize;
    let mut last_shown = 0usize; // 1-indexed last displayed line
    let mut empty_window = false;
    let first_start = ranges.first().map_or(1, |r| r.start);
    let multi = ranges.len() > 1;
    let windows: Vec<(usize, usize)> = if ranges.is_empty() {
        vec![(0, DEFAULT_LIMIT.min(MAX_LINES))]
    } else {
        ranges
            .iter()
            .map(|r| {
                let requested_start = r.start - 1;
                let lead = if !raw && r.start > 1 { LEADING_CONTEXT.min(requested_start) } else { 0 };
                let trail = if !raw && r.end.is_some() { TRAILING_CONTEXT } else { 0 };
                let limit = r.end.map_or(DEFAULT_LIMIT, |e| e - r.start + 1);
                (requested_start - lead, (limit + lead + trail).min(MAX_LINES))
            })
            .collect()
    };
    let mut truncated_by_bytes = false;
    let max_bytes = DEFAULT_MAX_BYTES.max(windows.iter().map(|w| w.1).max().unwrap_or(0) * 512);
    for (wi, (start, count)) in windows.iter().enumerate() {
        if *start >= total {
            if wi == 0 {
                empty_window = true;
            }
            break;
        }
        let end = (start + count).min(total);
        for (i, line) in lines.iter().enumerate().take(end).skip(*start) {
            let shown = if raw { (*line).to_string() } else { truncate_columns(line) };
            let row = if hash_lines { format!("{}:{shown}", i + 1) } else { shown };
            if body.len() + row.len() + 1 > max_bytes && rows > 0 {
                truncated_by_bytes = true;
                break;
            }
            if rows > 0 {
                body.push('\n');
            }
            body.push_str(&row);
            rows += 1;
            last_shown = i + 1;
        }
        if truncated_by_bytes {
            break;
        }
    }

    if empty_window {
        let requested = first_start;
        let suggestion = if total == 0 {
            "The file is empty.".to_string()
        } else {
            format!("Use {selector_base}:1 to read from the start, or {selector_base}:{total} to read the last line.")
        };
        return Ok(AgentToolResult::text(format!(
            "Line {requested} is beyond end of file ({total} lines total). {suggestion}"
        )));
    }

    let mut out = String::new();
    if hash_lines && rows > 0 {
        let key = canonical_key(&absolute);
        let tag = env.store.record(&key, &normalized, None);
        let seen = seen_lines_from_body(&body);
        env.store.record_seen_lines(&key, &tag, &seen);
        let _ = writeln!(out, "[{display}#{tag}]");
    } else if raw && rows > 0 {
        // A raw read emits no header, but records the displayed range.
        let key = canonical_key(&absolute);
        let first = windows.first().map_or(0, |w| w.0) + 1;
        let seen: Vec<u32> = (first..=last_shown).filter_map(|n| u32::try_from(n).ok()).collect();
        env.store.record(&key, &normalized, Some(&seen));
    }
    out.push_str(&body);

    if truncated_by_bytes {
        let t = truncate_head(&body, max_bytes);
        let _ = write!(
            out,
            "\n\n[Showing lines {first_start}-{last_shown} of {total} ({} limit). Use {selector_base}:{} to continue]",
            format_bytes(t.max_bytes),
            last_shown + 1
        );
    } else if last_shown < total && !multi {
        let _ = write!(
            out,
            "\n\n[{} more lines in file. Use {selector_base}:{} to continue]",
            total - last_shown,
            last_shown + 1
        );
    }
    Ok(AgentToolResult::text(out))
}

// ------------------------------------------------------------- directories

struct Node {
    name: String,
    is_dir: bool,
    mtime: f64,
    size: u64,
    children: Vec<Node>,
    /// Entries cut from this directory's listing (`… N more`).
    dropped: usize,
}

fn mtime_ms(meta: &std::fs::Metadata) -> f64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map_or(0.0, |d| d.as_secs_f64() * 1000.0)
}

fn skip_dir(name: &str) -> bool {
    matches!(name, ".git" | "node_modules")
}

/// Children of `dir` (depth 1 = the root's entries), newest first. Below the
/// root a directory keeps `limit - 1` recent entries plus its oldest.
fn load_children(dir: &Path, depth: usize) -> (Vec<Node>, usize) {
    let Ok(rd) = std::fs::read_dir(dir) else { return (Vec::new(), 0) };
    let mut nodes: Vec<Node> = rd
        .filter_map(Result::ok)
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let meta = e.metadata().ok()?;
            let is_dir = meta.is_dir();
            if is_dir && skip_dir(&name) {
                return None;
            }
            Some(Node { name, is_dir, mtime: mtime_ms(&meta), size: meta.len(), children: Vec::new(), dropped: 0 })
        })
        .collect();
    nodes.sort_by(|a, b| b.mtime.total_cmp(&a.mtime).then_with(|| a.name.cmp(&b.name)));
    let mut dropped = 0;
    if depth > 1 && nodes.len() > DIR_CHILD_LIMIT {
        dropped = nodes.len() - DIR_CHILD_LIMIT;
        let oldest = nodes.pop();
        nodes.truncate(DIR_CHILD_LIMIT - 1);
        nodes.extend(oldest);
    }
    for n in &mut nodes {
        if n.is_dir && depth < DIR_MAX_DEPTH {
            let (children, child_dropped) = load_children(&dir.join(&n.name), depth + 1);
            n.children = children;
            n.dropped = child_dropped;
        }
    }
    (nodes, dropped)
}

struct Line {
    label: String,
    size: Option<String>,
    age: Option<String>,
}

fn render(nodes: &[Node], depth: usize, dropped: usize, now_ms: f64, out: &mut Vec<Line>) {
    let indent = "  ".repeat(depth);
    let one = |n: &Node, out: &mut Vec<Line>| {
        let suffix = if n.is_dir { "/" } else { "" };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let age = format_age(((now_ms - n.mtime).max(0.0) / 1000.0) as u64);
        out.push(Line {
            label: format!("{indent}- {}{suffix}", n.name),
            size: (!n.is_dir).then(|| format_bytes(usize::try_from(n.size).unwrap_or(usize::MAX))),
            age: Some(age),
        });
        render(&n.children, depth + 1, n.dropped, now_ms, out);
    };
    if dropped == 0 {
        for n in nodes {
            one(n, out);
        }
        return;
    }
    // Layout: recent entries, the "… N more" marker, then the oldest entry.
    let (oldest, recent) = nodes.split_last().map_or((None, nodes), |(l, r)| (Some(l), r));
    for n in recent {
        one(n, out);
    }
    out.push(Line { label: format!("{indent}- … {dropped} more"), size: None, age: None });
    if let Some(n) = oldest {
        one(n, out);
    }
}

fn read_directory(absolute: &Path, sel: &Selector) -> Result<AgentToolResult> {
    let (top, _) = load_children(absolute, 1);
    let now_ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64() * 1000.0);
    let mut lines = vec![Line { label: ".".to_string(), size: None, age: None }];
    render(&top, 1, 0, now_ms, &mut lines);
    let rendered = if lines.len() <= 1 {
        "(empty directory)".to_string()
    } else {
        let width = lines.iter().map(|l| l.label.chars().count()).max().unwrap_or(0);
        lines
            .iter()
            .map(|l| match &l.age {
                None => l.label.clone(),
                Some(age) => {
                    let pad = " ".repeat(width + 2 - l.label.chars().count());
                    format!("{}{pad}{:<8}  {:<4}", l.label, l.size.clone().unwrap_or_default(), age)
                        .trim_end()
                        .to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let sel = match sel {
        Selector::Tail { count, raw } => {
            let total = rendered.split('\n').count();
            let start = total.saturating_sub(*count) + 1;
            Selector::Lines { ranges: vec![LineRange { start, end: Some(total) }], raw: *raw }
        }
        other => other.clone(),
    };
    if let Selector::Lines { ranges, .. } = &sel {
        if ranges.len() > 1 {
            bail!("Multi-range line selectors are not supported for directory listings.");
        }
        let all: Vec<&str> = rendered.split('\n').collect();
        let r = ranges[0];
        let start = r.start - 1;
        if start >= all.len() {
            return Ok(AgentToolResult::text(format!(
                "Line {} is beyond end of listing ({} lines total). Use :1 to read from the start, or :{} to read the last line.",
                start + 1,
                all.len(),
                all.len()
            )));
        }
        let end = r.end.map_or(all.len(), |e| e.min(all.len()));
        let mut text = all[start..end].join("\n");
        if end < all.len() {
            let _ = write!(text, "\n\n[{} more lines in listing. Use :{} to continue]", all.len() - end, end + 1);
        }
        return Ok(AgentToolResult::text(text));
    }
    let t = truncate_head(&rendered, DEFAULT_MAX_BYTES);
    let mut text = t.content;
    if t.truncated {
        let _ = write!(text, "\n\n[Showing lines 1-{} of {} ({} limit)]", t.output_lines, t.total_lines, format_bytes(t.max_bytes));
    }
    Ok(AgentToolResult::text(text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Env;

    fn env(dir: &Path) -> Env {
        Env::new(dir, true)
    }

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

    #[test]
    fn selectors_parse_like_omp() {
        assert_eq!(parse_sel("").unwrap(), Selector::None);
        assert_eq!(parse_sel("raw").unwrap(), Selector::Raw);
        assert_eq!(
            parse_sel("5-16,960-973").unwrap(),
            Selector::Lines {
                ranges: vec![LineRange { start: 5, end: Some(16) }, LineRange { start: 960, end: Some(973) }],
                raw: false
            }
        );
        assert_eq!(
            parse_sel("50+10").unwrap(),
            Selector::Lines { ranges: vec![LineRange { start: 50, end: Some(59) }], raw: false }
        );
        assert_eq!(parse_sel("-60").unwrap(), Selector::Tail { count: 60, raw: false });
        assert_eq!(
            parse_sel("raw:2-4").unwrap(),
            Selector::Lines { ranges: vec![LineRange { start: 2, end: Some(4) }], raw: true }
        );
        assert!(parse_sel("0").is_err());
        assert!(parse_sel("9-3").is_err());
    }

    #[test]
    fn hashline_read_header_rows_and_footer() {
        let dir = tempfile::tempdir().unwrap();
        let mut body = String::new();
        for i in 1..=5 {
            use std::fmt::Write as _;
            let _ = writeln!(body, "line {i}");
        }
        std::fs::write(dir.path().join("a.txt"), &body).unwrap();
        let e = env(dir.path());
        let out = text_of(&read(&e, &serde_json::json!({"path": "a.txt"})).unwrap());
        let mut it = out.lines();
        let header = it.next().unwrap();
        assert!(header.starts_with("[a.txt#") && header.ends_with(']'), "{header}");
        assert_eq!(header.len(), "[a.txt#ABCD]".len());
        assert_eq!(it.next(), Some("1:line 1"));
        assert_eq!(out.lines().last(), Some("5:line 5"));
        // window with selector: one leading + three trailing context lines, footer
        let out = text_of(&read(&e, &serde_json::json!({"path": "a.txt:2-2"})).unwrap());
        assert!(out.contains("1:line 1\n2:line 2\n3:line 3"), "{out}");
        assert!(out.contains("5:line 5"), "{out}");
        let raw = text_of(&read(&e, &serde_json::json!({"path": "a.txt:raw:2-3"})).unwrap());
        assert_eq!(raw, "line 2\nline 3\n\n[2 more lines in file. Use a.txt:4 to continue]");
        let tail = text_of(&read(&e, &serde_json::json!({"path": "a.txt:-2"})).unwrap());
        assert!(tail.contains("4:line 4\n5:line 5"), "{tail}");
    }

    #[test]
    fn hashline_output_matches_ts_natives() {
        // Same fixture and TS formatting functions (bun cross-check, 2026-10-12):
        //   hashlineFileHash → "0793", hashlineFormatHeader → "[a.txt#0793]",
        //   rows via formatNumberedLine → "N:line".
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "line 1\nline 2\nline 3\nline 4\nline 5\n").unwrap();
        let e = env(dir.path());
        let out = text_of(&read(&e, &serde_json::json!({"path": "a.txt"})).unwrap());
        let expected = "[a.txt#0793]\n1:line 1\n2:line 2\n3:line 3\n4:line 4\n5:line 5";
        assert_eq!(out, expected, "omp-rs read output diverged from TS formatting");
    }

    #[test]
    fn read_reports_missing_and_beyond_eof() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "x\n").unwrap();
        let e = env(dir.path());
        let err = read(&e, &serde_json::json!({"path": "nope.txt"})).unwrap_err().to_string();
        assert_eq!(err, "Path 'nope.txt' not found");
        let out = text_of(&read(&e, &serde_json::json!({"path": "a.txt:9"})).unwrap());
        assert!(out.starts_with("Line 9 is beyond end of file (1 lines total)."), "{out}");
    }

    #[test]
    fn long_files_page_with_continue_footer() {
        let dir = tempfile::tempdir().unwrap();
        let mut body = String::new();
        for i in 1..=400 {
            use std::fmt::Write as _;
            let _ = writeln!(body, "l{i}");
        }
        std::fs::write(dir.path().join("big.txt"), body).unwrap();
        let out = text_of(&read(&env(dir.path()), &serde_json::json!({"path": "big.txt"})).unwrap());
        assert!(out.contains("300:l300"));
        assert!(!out.contains("301:l301"));
        assert!(out.ends_with("[100 more lines in file. Use big.txt:301 to continue]"), "{out}");
    }
}
