//! Search-family renderers (`glob`/`grep`) — ports of `pi-tui/src/tools/glob.ts`
//! and `grep.ts`, plus the grouped-file-output helpers from
//! `grouped-file-output.ts` (also used by `find`).
//!
//! Results come from the settled tool-call payload: `result.content[].text`
//! (model-facing lines) and `result.details` (counts, file lists, truncation).
//! List bodies are capped: collapsed 6/8 rows, expanded 24, with a
//! `… N more …` summary line and a `truncated` meta when the tool cut results.

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use serde_json::Value;

use crate::theme::Theme;

use super::ToolRenderer;

/// Collapsed rows for the grep match body (`PREVIEW_LIMITS.COLLAPSED_LINES * 2`).
pub(crate) const COLLAPSED_TEXT_LIMIT: usize = 6;
/// Expanded rows for the grep match body.
/// Expanded rows for the grep match body. Reserved for the ctrl+o expand
/// layer (`RenderResultOptions` equivalent) — unused until then.
#[allow(dead_code)]
pub(crate) const EXPANDED_TEXT_LIMIT: usize = 24;
/// Collapsed rows for the glob file list (`PREVIEW_LIMITS.COLLAPSED_ITEMS`).
pub(crate) const COLLAPSED_ITEMS: usize = 8;

/// Borrow the arguments object of a tool-call payload.
pub(crate) fn args_of(call: &Value) -> Option<&Value> {
    call.get("arguments").or_else(|| call.get("input"))
}

/// Borrow the settled details object, if any.
pub(crate) fn details_of(call: &Value) -> Option<&Value> {
    call.get("result")
        .and_then(|r| r.get("details"))
        .or_else(|| call.get("details"))
}

/// Collect the model-facing text of a settled result.
pub(crate) fn result_text(call: &Value) -> Option<String> {
    let result = call.get("result").or(Some(call));
    if let Some(text) = result.and_then(|r| r.get("text")).and_then(Value::as_str) {
        return Some(text.to_owned());
    }
    if let Some(out) = result.and_then(|r| r.get("output")).and_then(Value::as_str) {
        return Some(out.to_owned());
    }
    let content = result?.get("content")?;
    let mut joined = String::new();
    for piece in content.as_array()? {
        if piece.get("type").and_then(Value::as_str) == Some("text") {
            if !joined.is_empty() {
                joined.push('\n');
            }
            joined.push_str(piece.get("text").and_then(Value::as_str).unwrap_or(""));
        }
    }
    Some(joined).filter(|s| !s.is_empty())
}

/// One styled, owned display line.
pub(crate) fn styled(theme: &Theme, token: &str, text: impl Into<String>) -> Span<'static> {
    Span::styled(text.into(), Style::default().fg(theme.color(token)))
}

/// Render accumulated lines into the frame, truncated by the area.
pub(crate) fn paint(frame: &mut ratatui::Frame<'_>, area: Rect, lines: Vec<Line<'static>>) {
    frame.render_widget(Paragraph::new(lines), area);
}

/// `renderStatusLine` equivalent: `icon title: description · meta…`.
pub(crate) fn status_line(
    theme: &Theme,
    icon: &str,
    title: &str,
    description: &str,
    meta: Vec<Span<'static>>,
) -> Line<'static> {
    let mut spans = vec![styled(theme, "accent", format!("{icon} {title}:"))];
    if !description.is_empty() {
        spans.push(Span::raw(" "));
        spans.push(styled(theme, "text", description));
    }
    for m in meta {
        spans.push(Span::raw(" · "));
        spans.push(m);
    }
    Line::from(spans)
}

/// Warning-colored meta span (used for `truncated`, `skipped missing: …`).
pub(crate) fn warn(theme: &Theme, text: impl Into<String>) -> Span<'static> {
    styled(theme, "warning", text)
}

/// `… N more <item>s` summary span, muted.
pub(crate) fn more_items(theme: &Theme, count: usize, item: &str) -> Span<'static> {
    styled(theme, "text_muted", more_label(count, item))
}

fn more_label(count: usize, item: &str) -> String {
    if count == 1 {
        format!("… 1 more {item}")
    } else if item == "match" {
        format!("… {count} more matches")
    } else {
        format!("… {count} more {item}s")
    }
}

/// Tree branch prefix (`├─` / `└─`) as a muted span.
pub(crate) fn branch(theme: &Theme, last: bool) -> Span<'static> {
    styled(theme, "text_muted", if last { "└─" } else { "├─" })
}

/// `⚠ message` line (empty-result notice).
fn empty_message(theme: &Theme, message: &str) -> Line<'static> {
    Line::from(vec![warn(theme, format!("⚠ {message}"))])
}

/// Count a details array, falling back to line-counting legacy text output.
#[allow(clippy::cast_possible_truncation)]
fn detail_count(details: Option<&Value>, key: &str) -> Option<usize> {
    details
        .and_then(|d| d.get(key))
        .and_then(Value::as_u64)
        .map(|n| usize::try_from(n).unwrap_or(usize::MAX))
}

fn scope_of(args: Option<&Value>) -> Option<String> {
    let a = args?;
    let path = a.get("path").or_else(|| a.get("paths"));
    match path {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(items)) => {
            let list: Vec<&str> = items.iter().filter_map(Value::as_str).collect();
            if list.is_empty() {
                None
            } else {
                Some(list.join(", "))
            }
        }
        _ => None,
    }
}

fn str_list(details: Option<&Value>, key: &str) -> Vec<String> {
    details
        .and_then(|d| d.get(key))
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn truncated_of(call: &Value, details: Option<&Value>) -> bool {
    for d in [details, Some(call)] {
        if d.and_then(|d| d.get("truncated")).and_then(Value::as_bool) == Some(true) {
            return true;
        }
    }
    details
        .and_then(|d| d.get("meta"))
        .and_then(|m| m.get("truncation"))
        .is_some()
}

/// `glob` — file list with counts (`glob.ts`).
pub(super) struct Glob;

impl ToolRenderer for Glob {
    fn summary(&self, call: &Value) -> String {
        let args = args_of(call);
        let pattern = args
            .and_then(|a| a.get("pattern"))
            .and_then(Value::as_str)
            .unwrap_or("*");
        match scope_of(args) {
            Some(p) => format!("{pattern} in {p}"),
            None => pattern.to_owned(),
        }
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
        let details = details_of(call);
        let args = args_of(call);
        let truncated = truncated_of(call, details);
        let mut files = str_list(details, "files");
        if files.is_empty() {
            if let Some(text) = result_text(call) {
                files = text
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty() && !l.starts_with("No "))
                    .map(ToOwned::to_owned)
                    .collect();
            }
        }
        let count = detail_count(details, "fileCount").unwrap_or(files.len());

        let mut lines = Vec::new();
        if files.is_empty() && count == 0 {
            let label = if truncated {
                "No matches before timeout (scan incomplete)"
            } else {
                "No files found"
            };
            lines.push(status_line(theme, "⚠", "Glob", &self.summary(call), vec![]));
            lines.push(empty_message(theme, label));
            paint(frame, area, lines);
            return;
        }

        let mut meta = vec![styled(theme, "text_muted", format!("{count} files"))];
        if let Some(scope) = scope_of(args) {
            meta.push(styled(theme, "text_muted", format!("in {scope}")));
        }
        if truncated {
            meta.push(warn(theme, "truncated"));
        }
        lines.push(status_line(theme, "🔍", "Glob", &self.summary(call), meta));

        let shown = files.len().min(COLLAPSED_ITEMS);
        for (i, file) in files.iter().enumerate().take(shown) {
            let last = i + 1 == shown && files.len() <= COLLAPSED_ITEMS;
            let mut spans = vec![branch(theme, last), Span::raw(" ")];
            if file.ends_with('/') {
                spans.push(styled(theme, "accent", file.clone()));
            } else {
                spans.push(styled(theme, "text", file.clone()));
            }
            lines.push(Line::from(spans));
        }
        if files.len() > COLLAPSED_ITEMS {
            lines.push(Line::from(vec![
                branch(theme, true),
                Span::raw(" "),
                more_items(theme, files.len() - COLLAPSED_ITEMS, "file"),
            ]));
        }
        paint(frame, area, lines);
    }
}

/// A display line classified for the grep body (`classifyGroupedLines`
/// light): directory header (`# src/`), file header (`## main.rs`), or body.
enum GrepLine {
    Dir,
    File(String),
    Body,
}

fn classify(line: &str) -> GrepLine {
    if !line.trim_start().starts_with('#') {
        return GrepLine::Body;
    }
    let hash = line.find('#').unwrap_or(0);
    let Some(space) = line.find(' ') else {
        return GrepLine::Body;
    };
    if space <= hash {
        return GrepLine::Body;
    }
    let rest = line[hash + 1..space].trim();
    if rest.is_empty() {
        return GrepLine::Body;
    }
    if rest.ends_with('/') {
        GrepLine::Dir
    } else {
        GrepLine::File(rest.to_owned())
    }
}

/// True for marked match rows (`*12│ …`) or model-facing rows (`12: …` / `12| …`).
pub(crate) fn is_match_line(line: &str) -> bool {
    let t = line.trim_start();
    let t = t.strip_prefix('*').unwrap_or(t);
    let Some(digits) = t.find(|c: char| !c.is_ascii_digit()) else {
        return false;
    };
    if digits == 0 {
        return false;
    }
    matches!(t[digits..].chars().next(), Some('\u{2502}' | ':' | '|'))
}

/// `groupLineIndicesByBlank`: split into blank-separated groups of indices.
pub(crate) fn group_line_indices_by_blank(lines: &[&str]) -> Vec<Vec<usize>> {
    let has_seps = lines.iter().any(|l| l.trim().is_empty());
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            if has_seps && !current.is_empty() {
                groups.push(std::mem::take(&mut current));
            }
            continue;
        }
        current.push(i);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// `grep` — grouped match rows with counts and a line budget (`grep.ts`).
pub(super) struct Grep;

const SEARCH_GLYPH: &str = "\u{1f50d}"; // magnifier (theme icon.search)

impl ToolRenderer for Grep {
    fn summary(&self, call: &Value) -> String {
        let args = args_of(call);
        let pattern = args
            .and_then(|a| a.get("pattern").or_else(|| a.get("regex")))
            .and_then(Value::as_str)
            .unwrap_or("?");
        match scope_of(args) {
            Some(p) => format!("{pattern} in {p}"),
            None => pattern.to_owned(),
        }
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
        let details = details_of(call);
        let args = args_of(call);
        let text = result_text(call).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        let match_count = detail_count(details, "matchCount")
            .unwrap_or_else(|| lines.iter().filter(|l| is_match_line(l)).count());
        let file_count = detail_count(details, "fileCount").unwrap_or_else(|| {
            lines
                .iter()
                .filter(|l| matches!(classify(l), GrepLine::File(_)))
                .count()
        });
        let truncated = truncated_of(call, details);
        let missing = str_list(details, "missingPaths");

        if text.is_empty() || match_count == 0 {
            let mut meta = vec![styled(theme, "text_muted", "0 matches")];
            if let Some(scope) = scope_of(args) {
                meta.push(styled(theme, "text_muted", format!("in {scope}")));
            }
            let mut out = vec![status_line(
                theme,
                "\u{26a0}",
                "Grep",
                &self.summary(call),
                meta,
            )];
            out.push(empty_message(theme, "No matches found"));
            if !missing.is_empty() {
                out.push(Line::from(vec![warn(
                    theme,
                    format!("skipped missing: {}", missing.join(", ")),
                )]));
            }
            paint(frame, area, out);
            return;
        }

        let mut meta = vec![
            styled(theme, "text_muted", format!("{match_count} matches")),
            styled(theme, "text_muted", format!("{file_count} files")),
        ];
        if let Some(scope) = scope_of(args) {
            meta.push(styled(theme, "text_muted", format!("in {scope}")));
        }
        if truncated {
            meta.push(warn(theme, "truncated"));
        }
        let mut out = vec![status_line(
            theme,
            SEARCH_GLYPH,
            "Grep",
            &self.summary(call),
            meta,
        )];

        // Collapsed body: header + match rows only, budget COLLAPSED_TEXT_LIMIT
        // (context rows and dir headers are dropped, per `compactSearchPreviewGroup`).
        let budget = COLLAPSED_TEXT_LIMIT;
        let groups = group_line_indices_by_blank(&lines);
        let mut used = 0usize;
        let mut shown_matches = 0usize;
        let mut rendered: Vec<Line<'static>> = Vec::new();
        'outer: for group in &groups {
            // Compact mode keeps only match rows — unless the group carries no
            // detectable marker (e.g. `path:12: text` model format), in which
            // case all of its lines are kept as matches.
            let marked = group.iter().any(|&i| is_match_line(lines[i]));
            for &i in group {
                if used >= budget {
                    break 'outer;
                }
                let line = lines[i];
                let kind = classify(line);
                let is_match = is_match_line(line);
                let is_body = matches!(kind, GrepLine::Body);
                let keep = match &kind {
                    GrepLine::Dir => false,
                    GrepLine::File(_) => true,
                    GrepLine::Body => !marked || is_match,
                };
                if !keep {
                    continue;
                }
                let row = match kind {
                    GrepLine::File(f) => {
                        Line::from(vec![styled(theme, "accent", format!("## {f}"))])
                    }
                    _ => Line::from(vec![styled(theme, "text", line.to_owned())]),
                };
                if is_body {
                    shown_matches += 1;
                }
                rendered.push(row);
                used += 1;
            }
        }
        let total = rendered.len();
        for (idx, row) in rendered.into_iter().enumerate() {
            let mut spans = vec![branch(theme, idx + 1 == total), Span::raw(" ")];
            spans.extend(row.spans);
            out.push(Line::from(spans));
        }
        if shown_matches < match_count {
            out.push(Line::from(vec![
                branch(theme, true),
                Span::raw(" "),
                more_items(theme, match_count - shown_matches, "match"),
            ]));
        }
        if !missing.is_empty() {
            out.push(Line::from(vec![warn(
                theme,
                format!("skipped missing: {}", missing.join(", ")),
            )]));
        }
        paint(frame, area, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn render_lines<R: ToolRenderer>(r: &R, call: &Value, w: u16, h: u16) -> Vec<String> {
        let backend = TestBackend::new(w, h);
        let mut term = Terminal::new(backend).unwrap();
        let theme = Theme::default();
        term.draw(|f| r.render(f, f.area(), call, &theme)).unwrap();
        let buffer = term.backend().buffer().clone();
        (0..buffer.area.height)
            .filter_map(|y| {
                let row: String = (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_owned())
                    .collect();
                let trimmed = row.trim_end().to_owned();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            })
            .collect()
    }

    #[test]
    fn is_match_line_formats() {
        assert!(is_match_line("*12│ fn main()"));
        assert!(is_match_line("12: let x = 1;"));
        assert!(is_match_line("3| return"));
        assert!(!is_match_line("  context line"));
        assert!(!is_match_line("## src/lib.rs"));
    }

    #[test]
    fn groups_split_on_blank_lines() {
        let lines = ["a", "", "b", "c"];
        let groups = group_line_indices_by_blank(&lines);
        assert_eq!(groups, vec![vec![0], vec![2, 3]]);
        let no_seps = ["a", "b"];
        assert_eq!(group_line_indices_by_blank(&no_seps), vec![vec![0, 1]]);
    }

    #[test]
    fn glob_summary_is_pattern_and_scope() {
        let call = serde_json::json!({"name":"glob","arguments":{"pattern":"*.rs","path":"src"}});
        assert_eq!(Glob.summary(&call), "*.rs in src");
    }

    #[test]
    fn glob_caps_list_and_counts() {
        let files: Vec<String> = (0..12).map(|i| format!("f{i}.rs")).collect();
        let call = serde_json::json!({
            "name": "glob",
            "arguments": {"pattern": "*"},
            "result": {"details": {"fileCount": 12, "files": files}}
        });
        let out = render_lines(&Glob, &call, 100, 20);
        assert!(out[0].contains("12 files"));
        assert_eq!(
            out.iter()
                .filter(|l| l.contains("\u{251c}\u{2500}"))
                .count(),
            8
        );
        assert!(out.last().unwrap().contains("\u{2026} 4 more files"));
    }

    #[test]
    fn glob_empty_shows_warning() {
        let call = serde_json::json!({"name":"glob","arguments":{"pattern":"*.zzz"},"result":{"text":"No files found"}});
        let out = render_lines(&Glob, &call, 80, 10);
        assert!(out.iter().any(|l| l.contains("\u{26a0} No files found")));
    }

    #[test]
    fn grep_truncated_lists_more_matches() {
        let body: String = (1..=10)
            .map(|i| format!("src/lib.rs:{i}: match line {i}\n\n"))
            .collect();
        let call = serde_json::json!({
            "name": "grep",
            "arguments": {"pattern": "match"},
            "result": {"text": body.trim_end(), "details": {"matchCount": 10, "fileCount": 1}}
        });
        let out = render_lines(&Grep, &call, 100, 30);
        assert!(out[0].contains("10 matches"));
        assert!(out[0].contains("1 files"));
        assert!(out.last().unwrap().contains("\u{2026} 4 more matches"));
    }

    #[test]
    fn grep_zero_matches_warns() {
        let call = serde_json::json!({
            "name": "grep",
            "arguments": {"pattern": "zz"},
            "result": {"text": "No matches found", "details": {"matchCount": 0, "fileCount": 0}}
        });
        let out = render_lines(&Grep, &call, 80, 10);
        assert!(out.iter().any(|l| l.contains("No matches found")));
        assert!(out[0].contains("0 matches"));
    }
}
