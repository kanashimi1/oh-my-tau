//! Shared helpers for the built-in tools: path normalisation (omp's
//! `expandPath` / `resolveToCwd` / `formatPathRelativeToCwd`), size and age
//! formatting, and head/tail truncation with omp's notice texts.

use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, Result};
use serde_json::Value;

/// `DEFAULT_MAX_BYTES` of omp's streaming output: inline output budget.
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

pub fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing string argument `{key}`"))
}

/// omp `formatBytes`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// omp `formatAge` (seconds → "5m ago").
#[must_use]
pub fn format_age(seconds: u64) -> String {
    if seconds == 0 {
        return String::new();
    }
    let mins = seconds / 60;
    let hours = mins / 60;
    let days = hours / 24;
    let weeks = days / 7;
    let months = days / 30;
    if months > 0 {
        format!("{months}mo ago")
    } else if weeks > 0 {
        format!("{weeks}w ago")
    } else if days > 0 {
        format!("{days}d ago")
    } else if hours > 0 {
        format!("{hours}h ago")
    } else if mins > 0 {
        format!("{mins}m ago")
    } else {
        "just now".to_string()
    }
}

/// omp `expandPath`: strips a stray leading `:` before a path shape, the
/// `@` mention prefix, `file://` and expands `~`.
#[must_use]
pub fn expand_path(raw: &str) -> String {
    let mut p = raw;
    let bytes = p.as_bytes();
    if bytes.first() == Some(&b':')
        && bytes
            .get(1)
            .is_some_and(|c| matches!(c, b'/' | b'\\' | b'~') || (*c == b'.'))
    {
        p = &p[1..];
    }
    p = p.strip_prefix('@').unwrap_or(p);
    p = p.strip_prefix("file://").unwrap_or(p);
    if p == "~" {
        return home().map_or_else(|| p.to_string(), |h| h.display().to_string());
    }
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(h) = home() {
            return h.join(rest).display().to_string();
        }
    }
    p.to_string()
}

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Lexical normalisation (`.`/`..` folded) without touching the filesystem.
#[must_use]
pub fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// omp `resolveToCwd`.
#[must_use]
pub fn resolve_to_cwd(raw: &str, cwd: &Path) -> PathBuf {
    let expanded = expand_path(raw.trim());
    if !expanded.is_empty() && expanded.bytes().all(|b| b == b'/') {
        return cwd.to_path_buf();
    }
    let p = PathBuf::from(&expanded);
    normalize(&if p.is_absolute() { p } else { cwd.join(p) })
}

/// omp `formatPathRelativeToCwd`: cwd-relative when inside, absolute otherwise.
#[must_use]
pub fn format_path_relative_to_cwd(path: &Path, cwd: &Path, trailing_slash: bool) -> String {
    let abs = normalize(&if path.is_absolute() { path.to_path_buf() } else { cwd.join(path) });
    let cwd = normalize(cwd);
    let mut display = match abs.strip_prefix(&cwd) {
        Ok(rel) if rel.as_os_str().is_empty() => ".".to_string(),
        Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
        Err(_) => abs.to_string_lossy().replace('\\', "/"),
    };
    if trailing_slash && display != "." && !display.ends_with('/') {
        display.push('/');
    }
    display
}

/// `[home-shortened]` display of an absolute path (omp `shortenPath`).
#[must_use]
pub fn shorten_path(path: &str) -> String {
    if let Some(h) = home() {
        let h = h.display().to_string();
        if path == h {
            return "~".to_string();
        }
        if let Some(rest) = path.strip_prefix(&format!("{h}/")) {
            return format!("~/{rest}");
        }
    }
    path.to_string()
}

fn floor_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_boundary(s: &str, mut i: usize) -> usize {
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Result of a head/tail truncation (omp `TruncationResult`, the used part).
#[derive(Debug, Clone)]
pub struct Truncation {
    pub content: String,
    pub truncated: bool,
    pub total_lines: usize,
    pub output_lines: usize,
    pub max_bytes: usize,
}

fn count_lines(s: &str) -> usize {
    s.bytes().filter(|b| *b == b'\n').count() + 1
}

/// omp `truncateHead`: keep whole leading lines within `max_bytes`.
#[must_use]
pub fn truncate_head(content: &str, max_bytes: usize) -> Truncation {
    let total_lines = count_lines(content);
    if content.len() <= max_bytes {
        return Truncation {
            content: content.to_string(),
            truncated: false,
            total_lines,
            output_lines: total_lines,
            max_bytes,
        };
    }
    let mut end = 0;
    let mut lines = 0;
    for line in content.split('\n') {
        let sep = usize::from(lines > 0);
        if end + sep + line.len() > max_bytes {
            break;
        }
        end += sep + line.len();
        lines += 1;
    }
    Truncation {
        content: content[..end].to_string(),
        truncated: true,
        total_lines,
        output_lines: lines,
        max_bytes,
    }
}

/// omp `truncateTail`: keep whole trailing lines within `max_bytes`; a single
/// over-long last line is cut to a UTF-8-safe byte window.
#[must_use]
pub fn truncate_tail(content: &str, max_bytes: usize) -> Truncation {
    let total_lines = count_lines(content);
    if content.len() <= max_bytes {
        return Truncation {
            content: content.to_string(),
            truncated: false,
            total_lines,
            output_lines: total_lines,
            max_bytes,
        };
    }
    let mut start = content.len();
    let mut lines = 0;
    for line_start in std::iter::once(0)
        .chain(content.match_indices('\n').map(|(i, _)| i + 1))
        .rev()
    {
        if content.len() - line_start > max_bytes {
            break;
        }
        start = line_start;
        lines += 1;
    }
    if lines == 0 {
        start = ceil_boundary(content, content.len() - max_bytes);
        lines = 1;
    }
    Truncation {
        content: content[start..].to_string(),
        truncated: true,
        total_lines,
        output_lines: lines,
        max_bytes,
    }
}

/// Cut `s` to at most `max` bytes at a char boundary (display-only column cap).
#[must_use]
pub fn cut_bytes(s: &str, max: usize) -> &str {
    &s[..floor_boundary(s, max.min(s.len()))]
}

/// Split `a;b;c` path lists like omp `toPathList` (semicolon-delimited).
#[must_use]
pub fn path_list(raw: Option<&str>) -> Vec<String> {
    raw.map(|r| {
        r.split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    })
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_truncation_keeps_whole_trailing_lines() {
        let text = (1..=100).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let t = truncate_tail(&text, 50);
        assert!(t.truncated);
        assert!(t.content.ends_with("line 100"));
        assert!(t.content.len() <= 50);
        assert!(!t.content.starts_with("ine"));
        assert_eq!(t.output_lines, t.content.lines().count());
    }

    #[test]
    fn relative_paths_inside_and_outside_cwd() {
        let cwd = Path::new("/w/proj");
        assert_eq!(format_path_relative_to_cwd(Path::new("/w/proj/a/b.rs"), cwd, false), "a/b.rs");
        assert_eq!(format_path_relative_to_cwd(Path::new("/w/proj/a"), cwd, true), "a/");
        assert_eq!(format_path_relative_to_cwd(Path::new("/etc/x"), cwd, false), "/etc/x");
        assert_eq!(resolve_to_cwd("../x/./y", cwd), PathBuf::from("/w/x/y"));
        assert_eq!(resolve_to_cwd("@src/a.rs", cwd), PathBuf::from("/w/proj/src/a.rs"));
    }
}
