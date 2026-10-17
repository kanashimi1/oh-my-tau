//! Shared helpers for the built-in tools: path normalisation (omp's
//! `expandPath` / `resolveToCwd` / `formatPathRelativeToCwd`), size and age
//! formatting, and head/tail truncation with omp's notice texts.

use std::fmt::Write as _;
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
    let abs = normalize(&if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    });
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
    truncate_head_lines(content, max_bytes, usize::MAX)
}

/// omp `truncateHead` with a line cap: whole leading lines within both
/// `max_bytes` and `max_lines`.
#[must_use]
pub fn truncate_head_lines(content: &str, max_bytes: usize, max_lines: usize) -> Truncation {
    let total_lines = count_lines(content);
    if content.len() <= max_bytes && total_lines <= max_lines {
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
        if lines >= max_lines || end + sep + line.len() > max_bytes {
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
    truncate_tail_lines(content, max_bytes, usize::MAX)
}

/// omp `truncateTail` with a line cap: whole trailing lines within both
/// `max_bytes` and `max_lines`; a single over-long last line is cut to a
/// UTF-8-safe byte window.
#[must_use]
pub fn truncate_tail_lines(content: &str, max_bytes: usize, max_lines: usize) -> Truncation {
    let total_lines = count_lines(content);
    if content.len() <= max_bytes && total_lines <= max_lines {
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
        if lines >= max_lines || content.len() - line_start > max_bytes {
            break;
        }
        start = line_start;
        lines += 1;
    }
    if lines == 0 {
        start = ceil_boundary(content, content.len().saturating_sub(max_bytes));
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

/// Slack over the spill threshold before bash's final-defense inline cap
/// fires (TS `INLINE_CAP_SLACK_BYTES`).
const INLINE_CAP_SLACK_BYTES: usize = 2 * 1024;

/// Output limits from the `tools.artifact*` settings (TS `getSpillConfig`;
/// settings are KB, fields bytes). [`Default`] is the TS schema default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillConfig {
    /// `tools.artifactSpillThreshold`: results above it spill to an artifact.
    pub threshold: usize,
    /// `tools.artifactHeadBytes`: head kept beside the tail; 0 = tail only.
    pub head_bytes: usize,
    /// `tools.artifactTailBytes`: tail kept inline after a spill.
    pub tail_bytes: usize,
    /// `tools.artifactTailLines`: line cap of the kept tail (and head).
    pub tail_lines: usize,
}

impl Default for SpillConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_MAX_BYTES,
            head_bytes: 20 * 1024,
            tail_bytes: 20 * 1024,
            tail_lines: 500,
        }
    }
}

impl SpillConfig {
    /// Effective config for `cwd` (global + project settings layers).
    #[must_use]
    pub fn load(cwd: &Path) -> Self {
        Self::from_settings(&omp_settings::Settings::load(cwd))
    }

    /// Read the four keys; a missing, negative or non-numeric value keeps
    /// the default.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // finite, non-negative sizes
    pub fn from_settings(settings: &omp_settings::Settings) -> Self {
        let d = Self::default();
        let num = |id: &str| {
            settings
                .get_number(id)
                .filter(|v| v.is_finite() && *v >= 0.0)
        };
        let kb = |id: &str, fallback: usize| num(id).map_or(fallback, |v| (v * 1024.0) as usize);
        Self {
            threshold: kb("tools.artifactSpillThreshold", d.threshold),
            head_bytes: kb("tools.artifactHeadBytes", d.head_bytes),
            tail_bytes: kb("tools.artifactTailBytes", d.tail_bytes),
            tail_lines: num("tools.artifactTailLines").map_or(d.tail_lines, |v| v as usize),
        }
    }

    /// TS `OutputSink` head window: `headBytes` clamped to half the stream
    /// budget ([`DEFAULT_MAX_BYTES`], which the sink uses regardless of
    /// settings).
    #[must_use]
    pub fn sink_head_bytes(&self) -> usize {
        self.head_bytes.min(DEFAULT_MAX_BYTES / 2)
    }

    /// TS `resolveInlineByteCapBudget`: spill threshold plus notice slack.
    #[must_use]
    pub fn inline_cap(&self) -> usize {
        self.threshold + INLINE_CAP_SLACK_BYTES
    }
}

/// An elided view of over-long output plus its TS truncation notice (without
/// the artifact reference; see [`notice_with_artifact`]).
#[derive(Debug, Clone)]
pub struct Elided {
    pub content: String,
    pub notice: String,
}

/// TS `formatMiddleElisionMarker`.
fn elision_marker(elided_lines: usize, elided_bytes: usize) -> String {
    if elided_lines <= 1 {
        format!("[…{elided_bytes}B elided…]")
    } else {
        format!("[…{elided_lines}ln elided…]")
    }
}

/// One-sided view (TS `truncateTail`/`truncateHead` + tail notice).
fn one_sided(content: &str, t: Truncation, from_head: bool, max_lines: usize) -> Elided {
    let total = t.total_lines;
    let start = if from_head {
        1
    } else {
        total + 1 - t.output_lines.min(total)
    };
    let end = if from_head { t.output_lines } else { total };
    let shown_bytes = t.content.len();
    let partial_tail = !from_head && {
        let cut = content.len() - shown_bytes;
        cut > 0 && content.as_bytes()[cut - 1] != b'\n'
    };
    let mut notice = if partial_tail {
        format!(
            "Showing line {start} (partial, {} of {}) of {total}",
            format_bytes(shown_bytes),
            format_bytes(content.len())
        )
    } else {
        format!("Showing lines {start}-{end} of {total}")
    };
    if !partial_tail && t.output_lines < max_lines {
        // Cut by bytes, not by the line cap.
        let _ = write!(notice, " ({} limit)", format_bytes(t.max_bytes));
    }
    Elided {
        content: t.content,
        notice,
    }
}

/// omp `truncateMiddle`: keep leading lines within `head_bytes`/`head_lines`
/// and trailing lines within the rest of `max_bytes`/`max_lines`, joined by
/// an elision marker. A zero head budget degrades to `truncateTail`, a zero
/// tail budget to `truncateHead`. `None` when `content` fits.
#[must_use]
pub fn truncate_middle(
    content: &str,
    max_bytes: usize,
    max_lines: usize,
    head_bytes: usize,
    head_lines: usize,
) -> Option<Elided> {
    let total = count_lines(content);
    if content.len() <= max_bytes && total <= max_lines {
        return None;
    }
    let tail_bytes = max_bytes.saturating_sub(head_bytes);
    let tail_lines = max_lines.saturating_sub(head_lines);
    if head_bytes == 0 || head_lines == 0 {
        let bytes = if tail_bytes > 0 {
            tail_bytes
        } else {
            max_bytes
        };
        let lines = if tail_lines > 0 {
            tail_lines
        } else {
            max_lines
        };
        return Some(one_sided(
            content,
            truncate_tail_lines(content, bytes, lines),
            false,
            lines,
        ));
    }
    if tail_bytes == 0 || tail_lines == 0 {
        let t = truncate_head_lines(content, head_bytes, head_lines);
        return Some(one_sided(content, t, true, head_lines));
    }
    let head = truncate_head_lines(content, head_bytes, head_lines);
    // A giant first line fits no whole-line head window: keep a byte window,
    // and give the tail only the bytes after it.
    let byte_head = head.output_lines == 0;
    let head_text = if byte_head {
        &content[..floor_boundary(content, head_bytes.min(content.len()))]
    } else {
        head.content.as_str()
    };
    let tail_budget = if byte_head {
        tail_bytes.min(content.len() - head_text.len())
    } else {
        tail_bytes
    };
    let tail = truncate_tail_lines(content, tail_budget, tail_lines);
    let head_kept = head.output_lines.max(1);
    let tail_start = content.len() - tail.content.len();
    let partial_tail = tail_start > 0 && content.as_bytes()[tail_start - 1] != b'\n';
    if !byte_head && !partial_tail && head_kept + tail.output_lines >= total {
        return None;
    }
    let elided_bytes = content
        .len()
        .saturating_sub(head_text.len() + tail.content.len());
    if elided_bytes == 0 {
        return None;
    }
    let elided_lines = total.saturating_sub(head_kept + tail.output_lines);
    let notice = if byte_head || partial_tail {
        format!(
            "Showing head and tail bytes of {total} line{}; {} elided",
            if total == 1 { "" } else { "s" },
            format_bytes(elided_bytes)
        )
    } else {
        format!(
            "Showing lines 1-{head_kept} and {}-{total} of {total}; {elided_lines} middle line{} ({}) elided",
            total + 1 - tail.output_lines,
            if elided_lines == 1 { "" } else { "s" },
            format_bytes(elided_bytes)
        )
    };
    Some(Elided {
        content: format!(
            "{head_text}\n{}\n{}",
            elision_marker(elided_lines, elided_bytes),
            tail.content
        ),
        notice,
    })
}

/// TS `OutputSink` stream window: past [`DEFAULT_MAX_BYTES`] keep
/// [`SpillConfig::sink_head_bytes`] of head and the rest of that budget as
/// tail (no line cap). `None` when the output fits.
#[must_use]
pub fn sink_window(text: &str, cfg: &SpillConfig) -> Option<Elided> {
    if text.len() <= DEFAULT_MAX_BYTES {
        return None;
    }
    let head = cfg.sink_head_bytes();
    let head_lines = if head == 0 { 0 } else { usize::MAX / 2 };
    truncate_middle(text, DEFAULT_MAX_BYTES, usize::MAX, head, head_lines)
}

/// TS `spillLargeResultToArtifact` window: above the spill threshold keep
/// `head_bytes` + `tail_bytes` within `2 × tail_lines` lines (middle
/// elision), or only the tail when `head_bytes` is 0. `None` when the text
/// is under the threshold or already fits the window.
#[must_use]
pub fn spill_window(text: &str, cfg: &SpillConfig) -> Option<Elided> {
    if text.len() <= cfg.threshold {
        return None;
    }
    if cfg.head_bytes > 0 {
        truncate_middle(
            text,
            cfg.head_bytes + cfg.tail_bytes,
            cfg.tail_lines.saturating_mul(2),
            cfg.head_bytes,
            cfg.tail_lines,
        )
    } else {
        truncate_middle(text, cfg.tail_bytes, cfg.tail_lines, 0, 0)
    }
}

/// TS `enforceInlineByteCap`: past `max_bytes` keep ~60% head and ~25% tail
/// on line boundaries with a byte-count marker between; `artifact` (called
/// only when cutting) supplies the `[raw output: artifact://N]` footer.
#[must_use]
pub fn enforce_inline_byte_cap(
    text: &str,
    max_bytes: usize,
    artifact: impl FnOnce() -> Option<String>,
) -> String {
    if max_bytes == 0 || text.len() <= max_bytes {
        return text.to_owned();
    }
    let head = &text[..floor_boundary(text, max_bytes * 3 / 5)];
    let head = match head.rfind('\n') {
        Some(i) if i > 0 => &head[..i],
        _ => head,
    };
    let tail = &text[ceil_boundary(text, text.len() - max_bytes / 4)..];
    let tail = match tail.find('\n') {
        Some(i) if i + 1 < tail.len() => &tail[i + 1..],
        _ => tail,
    };
    let elided = text.len().saturating_sub(head.len() + tail.len());
    let mut out = format!("{head}\n[…{elided}B elided…]\n{tail}");
    if let Some(url) = artifact() {
        if !out.ends_with('\n') {
            out.push('\n');
        }
        let _ = write!(out, "[raw output: {url}]");
    }
    out
}

/// TS `formatTruncationMetaNotice` artifact suffix.
#[must_use]
pub fn notice_with_artifact(notice: &str, artifact: Option<&str>) -> String {
    match artifact {
        Some(url) => format!("{notice}. Read {url} for full output"),
        None => notice.to_owned(),
    }
}

/// Save `text` to the session artifacts dir as the next free `<id>.log`
/// (omp `session.allocateOutputArtifact`); returns its `artifact://<id>`
/// URL. `None` without a session or when the write fails.
#[must_use]
pub fn save_artifact(text: &str) -> Option<String> {
    let dir = omp_urls::global_session_file()
        .as_ref()
        .and_then(|p| omp_urls::artifacts_directory_for(p))?;
    std::fs::create_dir_all(&dir).ok()?;
    let used: Vec<u64> = std::fs::read_dir(&dir)
        .ok()?
        .flatten()
        .filter_map(|e| {
            e.file_name()
                .to_str()
                .and_then(|n| n.split('.').next())
                .and_then(|p| p.parse().ok())
        })
        .collect();
    let id = (0u64..).find(|i| !used.contains(i))?;
    std::fs::write(dir.join(format!("{id}.log")), text).ok()?;
    Some(format!("artifact://{id}"))
}

/// Serializes tests that install the process-global session file.
#[cfg(test)]
pub(crate) static SESSION_FILE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    fn spill_config_reads_kb_settings_with_default_fallbacks() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.yml"),
            "tools:\n  artifactSpillThreshold: 8\n  artifactHeadBytes: 2.5\n  artifactTailLines: -3\n",
        )
        .unwrap();
        let cfg = SpillConfig::from_settings(&omp_settings::Settings::load_from(None, dir.path()));
        assert_eq!(cfg.threshold, 8 * 1024);
        assert_eq!(cfg.head_bytes, 2560);
        assert_eq!(cfg.tail_bytes, SpillConfig::default().tail_bytes);
        assert_eq!(cfg.tail_lines, SpillConfig::default().tail_lines);
        assert_eq!(cfg.inline_cap(), 10 * 1024);
        let big = SpillConfig {
            head_bytes: 40 * 1024,
            ..cfg
        };
        assert_eq!(big.sink_head_bytes(), DEFAULT_MAX_BYTES / 2);
    }

    #[test]
    fn spill_window_keeps_head_and_tail_lines() {
        let text = (1..=1000)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let cfg = SpillConfig {
            threshold: 1024,
            head_bytes: 200,
            tail_bytes: 300,
            tail_lines: 10,
        };
        let e = spill_window(&text, &cfg).unwrap();
        assert!(e.content.starts_with("line 1\n"));
        assert!(e.content.ends_with("line 1000"));
        assert!(e.content.lines().count() <= 21, "{}", e.content);
        assert!(
            e.notice
                .starts_with("Showing lines 1-10 and 991-1000 of 1000;"),
            "{}",
            e.notice
        );
        // Tail only when the head budget is 0.
        let tail_only = spill_window(
            &text,
            &SpillConfig {
                head_bytes: 0,
                ..cfg
            },
        )
        .unwrap();
        assert!(tail_only.content.ends_with("line 1000") && !tail_only.content.contains("elided"));
        assert_eq!(tail_only.notice, "Showing lines 991-1000 of 1000");
        // Under the threshold nothing is cut.
        assert!(spill_window("short", &cfg).is_none());
    }

    #[test]
    fn inline_byte_cap_cuts_on_lines_and_appends_artifact_footer() {
        let text = (1..=2000)
            .map(|i| format!("row {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let out = enforce_inline_byte_cap(&text, 1000, || Some("artifact://7".to_string()));
        assert!(out.len() < 1000, "{}", out.len());
        assert!(out.starts_with("row 1\n"));
        assert!(out.contains("B elided…]\n"));
        assert!(
            out.ends_with("row 2000\n[raw output: artifact://7]"),
            "{out}"
        );
        assert_eq!(
            enforce_inline_byte_cap("small", 1000, || unreachable!()),
            "small"
        );
    }

    #[test]
    fn tail_truncation_keeps_whole_trailing_lines() {
        let text = (1..=100)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
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
        assert_eq!(
            format_path_relative_to_cwd(Path::new("/w/proj/a/b.rs"), cwd, false),
            "a/b.rs"
        );
        assert_eq!(
            format_path_relative_to_cwd(Path::new("/w/proj/a"), cwd, true),
            "a/"
        );
        assert_eq!(
            format_path_relative_to_cwd(Path::new("/etc/x"), cwd, false),
            "/etc/x"
        );
        assert_eq!(resolve_to_cwd("../x/./y", cwd), PathBuf::from("/w/x/y"));
        assert_eq!(
            resolve_to_cwd("@src/a.rs", cwd),
            PathBuf::from("/w/proj/src/a.rs")
        );
    }
}
