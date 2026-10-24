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

use super::bash::replace_tabs;
use super::frame::{lang_icon, status_header, token};
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

/// Warning-colored meta span (used for `truncated`, `skipped missing: …`).
pub(crate) fn warn(theme: &Theme, text: impl Into<String>) -> Span<'static> {
    styled(theme, "warning", text)
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

/// `pluralize` + `formatCount`: `1 match`, `2 matches`, `1 file`, `0 files`.
pub(crate) fn format_count(label: &str, n: usize) -> String {
    let plural = if n == 1 {
        label.to_owned()
    } else if let Some(stem) = label.strip_suffix('y') {
        format!("{stem}ies")
    } else if label.ends_with('s') || label.ends_with("ch") || label.ends_with("sh") {
        format!("{label}es")
    } else {
        format!("{label}s")
    };
    format!("{n} {plural}")
}

fn dim_span(theme: &Theme, text: impl Into<String>) -> Span<'static> {
    Span::styled(
        text.into(),
        Style::default().fg(token(theme, "dim", "text_muted")),
    )
}

fn muted_span(theme: &Theme, text: impl Into<String>) -> Span<'static> {
    Span::styled(
        text.into(),
        Style::default().fg(token(theme, "muted", "text_muted")),
    )
}

fn warning_span(theme: &Theme, text: impl Into<String>) -> Span<'static> {
    Span::styled(text.into(), Style::default().fg(theme.color("warning")))
}

/// `formatEmptyMessage`: warning glyph + muted message.
pub(crate) fn empty_row(theme: &Theme, message: &str) -> Line<'static> {
    Line::from(vec![
        status_icon(theme, "status.warning", "warning"),
        Span::raw(" "),
        muted_span(theme, message),
    ])
}

/// `formatErrorMessage`.
pub(crate) fn error_row(theme: &Theme, message: &str) -> Line<'static> {
    Line::from(vec![
        status_icon(theme, "status.error", "error"),
        Span::raw(" "),
        Span::styled(
            format!("Error: {}", message.trim()),
            Style::default().fg(theme.color("error")),
        ),
    ])
}

/// Status-header icon span of a `status.*`/`icon.*` symbol in a theme color.
pub(crate) fn status_icon(theme: &Theme, key: &str, color_token: &str) -> Span<'static> {
    Span::styled(
        theme.symbol(key).to_owned(),
        Style::default().fg(token(theme, color_token, "text_muted")),
    )
}

/// Search-family result icon (`icon.search`, tool-title colored).
pub(crate) fn search_icon(theme: &Theme) -> Span<'static> {
    status_icon(theme, "icon.search", "toolTitle")
}

/// TS `Text(…, paddingX 1)`: one leading space per row.
pub(crate) fn pad_rows(lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    lines
        .into_iter()
        .map(|mut l| {
            l.spans.insert(0, Span::raw(" "));
            l
        })
        .collect()
}

/// `renderTreeListRows` for single-line items: `├─ item` … `└─ item`, with
/// an optional muted trailing summary row that takes the last branch.
pub(crate) fn tree_rows(
    theme: &Theme,
    items: Vec<Line<'static>>,
    summary: Option<String>,
) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let n = items.len();
    for (i, mut line) in items.into_iter().enumerate() {
        let last = summary.is_none() && i + 1 == n;
        let branch = format!(
            "{} ",
            theme.symbol(if last { "tree.last" } else { "tree.branch" })
        );
        line.spans.insert(0, dim_span(theme, branch));
        out.push(line);
    }
    if let Some(text) = summary {
        out.push(Line::from(vec![
            dim_span(theme, format!("{} ", theme.symbol("tree.last"))),
            muted_span(theme, text),
        ]));
    }
    out
}

/// TS `formatMoreItems`: `… N more <item>(s)`.
fn more_text(n: usize, item: &str) -> String {
    let counted = format_count(item, n);
    let word = counted.split_once(' ').map_or(item, |(_, w)| w);
    format!("… {n} more {word}")
}

fn call_pending(call: &Value) -> bool {
    call.get("result").is_none() && call.get("isError").is_none()
}

/// Path list of a glob/grep call (`toPathList(args.path ?? args.paths)`).
fn path_list(args: Option<&Value>) -> Vec<String> {
    let Some(a) = args else {
        return Vec::new();
    };
    match a.get("path").or_else(|| a.get("paths")) {
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(ToOwned::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn is_error_result(call: &Value, details: Option<&Value>) -> Option<String> {
    let flagged = call
        .get("result")
        .and_then(|r| r.get("isError"))
        .or_else(|| call.get("isError"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let detail_err = details
        .and_then(|d| d.get("error"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    if flagged || detail_err.is_some() {
        Some(
            detail_err
                .or_else(|| result_text(call))
                .unwrap_or_else(|| "Unknown error".to_owned()),
        )
    } else {
        None
    }
}

fn has_value(details: Option<&Value>, key: &str) -> bool {
    details
        .and_then(|d| d.get(key))
        .is_some_and(|v| !v.is_null())
}

/// Glob transcript rows (TS `globToolRenderer.renderCall/renderResult`).
#[must_use]
pub fn glob_lines(call: &Value, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
    let args = args_of(call);
    let paths = path_list(args);
    let call_desc = if paths.is_empty() {
        "*".to_owned()
    } else {
        paths.join(", ")
    };
    let result_desc = paths.join(", ");
    let desc = |text: &str| {
        if text.is_empty() {
            Vec::new()
        } else {
            vec![muted_span(theme, text)]
        }
    };
    if call_pending(call) {
        let mut meta = Vec::new();
        if let Some(limit) = args.and_then(|a| a.get("limit")).and_then(Value::as_u64) {
            meta.push(dim_span(theme, format!("limit:{limit}")));
        }
        let header = status_header(
            theme,
            Some(status_icon(theme, "status.pending", "muted")),
            "Glob",
            "toolTitle",
            desc(&call_desc),
            meta,
        );
        return pad_rows(vec![header]);
    }
    let details = details_of(call);
    if let Some(err) = is_error_result(call, details) {
        return pad_rows(vec![error_row(theme, &err)]);
    }
    let max = if expanded {
        usize::MAX
    } else {
        COLLAPSED_ITEMS
    };
    if detail_count(details, "fileCount").is_none() {
        let text = result_text(call).unwrap_or_default();
        if text.trim().is_empty()
            || text.contains("No files matching")
            || text.contains("No files found")
        {
            return pad_rows(vec![empty_row(theme, "No files found")]);
        }
        let listed: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        let header = status_header(
            theme,
            Some(search_icon(theme)),
            "Glob",
            "toolTitle",
            desc(&result_desc),
            vec![dim_span(theme, format_count("file", listed.len()))],
        );
        let shown = listed.len().min(max);
        let items = listed[..shown]
            .iter()
            .map(|l| {
                Line::from(Span::styled(
                    (*l).to_owned(),
                    Style::default().fg(theme.color("accent")),
                ))
            })
            .collect();
        let summary = (listed.len() > shown).then(|| more_text(listed.len() - shown, "file"));
        let mut rows = vec![header];
        rows.extend(tree_rows(theme, items, summary));
        return pad_rows(rows);
    }
    let file_count = detail_count(details, "fileCount").unwrap_or(0);
    let files = str_list(details, "files");
    let limit_reached = detail_count(details, "resultLimitReached");
    let truncation = has_value(details, "truncation")
        || details
            .and_then(|d| d.get("meta"))
            .and_then(|m| m.get("truncation"))
            .is_some_and(|v| !v.is_null());
    let truncated = truncated_of(call, details) || truncation || limit_reached.is_some();
    let timed_out = details
        .and_then(|d| d.get("timedOut"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let missing = str_list(details, "missingPaths");
    let missing_note = (!missing.is_empty()).then(|| {
        Line::from(warning_span(
            theme,
            format!("skipped missing: {}", missing.join(", ")),
        ))
    });
    if file_count == 0 {
        let empty_timed_out = timed_out || truncated;
        let label = if empty_timed_out {
            "No matches before timeout (scan incomplete)"
        } else {
            "No files found"
        };
        let mut meta = vec![dim_span(theme, "0 files")];
        if empty_timed_out {
            meta.push(warning_span(theme, "timed out"));
        }
        let header = status_header(
            theme,
            Some(status_icon(theme, "status.warning", "warning")),
            "Glob",
            "toolTitle",
            desc(&result_desc),
            meta,
        );
        let mut rows = vec![header, empty_row(theme, label)];
        rows.extend(missing_note);
        return pad_rows(rows);
    }
    let mut meta = vec![dim_span(theme, format_count("file", file_count))];
    if let Some(scope) = details
        .and_then(|d| d.get("scopePath"))
        .and_then(Value::as_str)
    {
        meta.push(dim_span(theme, format!("in {scope}")));
    }
    if truncated {
        meta.push(warning_span(
            theme,
            if timed_out { "timed out" } else { "truncated" },
        ));
    }
    let header = status_header(
        theme,
        Some(if truncated {
            status_icon(theme, "status.warning", "warning")
        } else {
            search_icon(theme)
        }),
        "Glob",
        "toolTitle",
        desc(&result_desc),
        meta,
    );
    let shown = files.len().min(max);
    let items = files[..shown].iter().map(|f| file_row(theme, f)).collect();
    let summary = (files.len() > shown).then(|| more_text(files.len() - shown, "file"));
    let mut rows = vec![header];
    rows.extend(tree_rows(theme, items, summary));
    let mut reasons: Vec<String> = Vec::new();
    if let Some(n) = limit_reached {
        reasons.push(format!("limit {n} results"));
    }
    if truncation {
        reasons.push("size limit".to_owned());
    }
    if !reasons.is_empty() {
        rows.push(Line::from(warning_span(
            theme,
            format!("truncated: {}", reasons.join(", ")),
        )));
    }
    rows.extend(missing_note);
    pad_rows(rows)
}

/// `renderFileList` item: folder/language icon + path.
pub(crate) fn file_row(theme: &Theme, path: &str) -> Line<'static> {
    let is_dir = path.ends_with('/');
    let (icon, icon_token, label_token) = if is_dir {
        (theme.symbol("icon.folder"), "accent", "accent")
    } else {
        (lang_icon(theme, path), "muted", "toolOutput")
    };
    Line::from(vec![
        Span::styled(
            icon.to_owned(),
            Style::default().fg(token(theme, icon_token, "text_muted")),
        ),
        Span::raw(" "),
        Span::styled(
            path.to_owned(),
            Style::default().fg(token(theme, label_token, "text")),
        ),
    ])
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
        paint(frame, area, glob_lines(call, false, theme));
    }
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

fn is_header_line(line: &str) -> bool {
    line.starts_with('#') && line.trim_start_matches('#').starts_with(' ')
}

fn is_marked_match(line: &str) -> bool {
    let Some(t) = line.trim_start().strip_prefix('*') else {
        return false;
    };
    let digits = t.chars().take_while(char::is_ascii_digit).count();
    digits > 0 && matches!(t[digits..].chars().next(), Some('\u{2502}' | ':' | '|'))
}

fn style_search_line(theme: &Theme, line: &str) -> Span<'static> {
    let tok = if line.starts_with("# ") {
        "accent"
    } else if is_header_line(line) {
        "dim"
    } else {
        "toolOutput"
    };
    Span::styled(
        replace_tabs(line),
        Style::default().fg(token(theme, tok, "text")),
    )
}

/// TS `renderBudgetedSearchGroups`.
fn budgeted_groups(
    theme: &Theme,
    groups: &[Vec<usize>],
    lines: &[&str],
    max_lines: usize,
    match_count: usize,
    compact: bool,
) -> Vec<Line<'static>> {
    if max_lines == 0 {
        return Vec::new();
    }
    let rendered: Vec<Vec<usize>> = groups
        .iter()
        .map(|g| {
            if compact {
                let kept: Vec<usize> = g
                    .iter()
                    .copied()
                    .filter(|&i| is_header_line(lines[i]) || is_marked_match(lines[i]))
                    .collect();
                if kept.is_empty() {
                    g.clone()
                } else {
                    kept
                }
            } else {
                g.clone()
            }
        })
        .filter(|g| !g.is_empty())
        .collect();
    if rendered.is_empty() {
        return Vec::new();
    }
    let count = |g: &[usize], marked: bool| {
        g.iter()
            .filter(|&&i| {
                if marked {
                    is_marked_match(lines[i])
                } else {
                    !is_header_line(lines[i]) && !lines[i].is_empty()
                }
            })
            .count()
    };
    let total_lines: usize = rendered.iter().map(Vec::len).sum();
    let total_marked: usize = rendered.iter().map(|g| count(g, true)).sum();
    let total_fallback: usize = rendered.iter().map(|g| count(g, false)).sum();
    let has_marked = total_marked > 0;
    let needs_summary = total_lines > max_lines;
    let budget = if needs_summary {
        max_lines.saturating_sub(1)
    } else {
        max_lines
    };
    let mut visible: Vec<Vec<usize>> = Vec::new();
    let mut visible_lines = 0;
    let mut visible_matches = 0;
    for g in &rendered {
        if visible_lines >= budget {
            break;
        }
        let take = g.len().min(budget - visible_lines);
        if take == 0 {
            break;
        }
        let vg = g[..take].to_vec();
        visible_lines += vg.len();
        visible_matches += count(&vg, has_marked);
        visible.push(vg);
    }
    let total_matches = if has_marked {
        total_marked
    } else {
        match_count.max(total_fallback)
    };
    let hidden_matches = total_matches.saturating_sub(visible_matches);
    let hidden_lines = total_lines.saturating_sub(visible_lines);
    let has_summary = needs_summary && (hidden_matches > 0 || hidden_lines > 0);
    let mut out = Vec::new();
    for (gi, g) in visible.iter().enumerate() {
        let last = !has_summary && gi + 1 == visible.len();
        let prefix = format!(
            "{} ",
            theme.symbol(if last { "tree.last" } else { "tree.branch" })
        );
        let cont = if last {
            "   ".to_owned()
        } else {
            format!("{}  ", theme.symbol("tree.vertical"))
        };
        for (j, &i) in g.iter().enumerate() {
            out.push(Line::from(vec![
                dim_span(theme, if j == 0 { prefix.clone() } else { cont.clone() }),
                style_search_line(theme, lines[i]),
            ]));
        }
    }
    if has_summary {
        let label = if hidden_matches > 0 {
            more_text(hidden_matches, "match")
        } else {
            more_text(hidden_lines, "line")
        };
        out.push(Line::from(vec![
            dim_span(theme, format!("{} ", theme.symbol("tree.last"))),
            muted_span(theme, label),
        ]));
    }
    out
}

/// Grep transcript rows (TS `grepToolRenderer.renderCall/renderResult`).
#[must_use]
pub fn grep_lines(call: &Value, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
    let args = args_of(call);
    let pattern = args
        .and_then(|a| a.get("pattern").or_else(|| a.get("regex")))
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_owned();
    let description = || vec![muted_span(theme, pattern.clone())];
    if call_pending(call) {
        let paths = path_list(args);
        let mut meta = Vec::new();
        if !paths.is_empty() {
            meta.push(dim_span(theme, format!("in {}", paths.join(", "))));
        }
        if args.and_then(|a| a.get("case")).and_then(Value::as_bool) == Some(false) {
            meta.push(dim_span(theme, "case:insensitive"));
        }
        if args
            .and_then(|a| a.get("gitignore"))
            .and_then(Value::as_bool)
            == Some(false)
        {
            meta.push(dim_span(theme, "gitignore:false"));
        }
        if let Some(skip) = args
            .and_then(|a| a.get("skip"))
            .and_then(Value::as_u64)
            .filter(|s| *s > 0)
        {
            meta.push(dim_span(theme, format!("skip:{skip}")));
        }
        let header = status_header(
            theme,
            Some(status_icon(theme, "status.pending", "muted")),
            "Grep",
            "toolTitle",
            description(),
            meta,
        );
        return pad_rows(vec![header]);
    }
    let details = details_of(call);
    if let Some(err) = is_error_result(call, details) {
        return pad_rows(vec![error_row(theme, &err)]);
    }
    let text = details
        .and_then(|d| d.get("displayContent"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| result_text(call))
        .unwrap_or_default();
    let has_detail = detail_count(details, "matchCount").is_some()
        || detail_count(details, "fileCount").is_some();
    let scope_meta = details
        .and_then(|d| d.get("scopePath"))
        .and_then(Value::as_str)
        .map(|s| dim_span(theme, format!("in {s}")));
    if !has_detail {
        if text.is_empty() || text == "No matches found" {
            return pad_rows(vec![empty_row(theme, "No matches found")]);
        }
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        let header = status_header(
            theme,
            Some(search_icon(theme)),
            "Grep",
            "toolTitle",
            description(),
            vec![dim_span(theme, format_count("item", lines.len()))],
        );
        let max = if expanded {
            EXPANDED_TEXT_LIMIT
        } else {
            COLLAPSED_TEXT_LIMIT
        };
        let shown = lines.len().min(max);
        let items = lines[..shown]
            .iter()
            .map(|l| {
                Line::from(Span::styled(
                    (*l).to_owned(),
                    Style::default().fg(token(theme, "toolOutput", "text")),
                ))
            })
            .collect();
        let summary = (lines.len() > shown).then(|| more_text(lines.len() - shown, "item"));
        let mut rows = vec![header];
        rows.extend(tree_rows(theme, items, summary));
        return pad_rows(rows);
    }
    let match_count = detail_count(details, "matchCount").unwrap_or(0);
    let file_count = detail_count(details, "fileCount").unwrap_or(0);
    let missing = str_list(details, "missingPaths");
    let missing_note = (!missing.is_empty()).then(|| {
        Line::from(warning_span(
            theme,
            format!("skipped missing: {}", missing.join(", ")),
        ))
    });
    if match_count == 0 {
        let mut meta = vec![dim_span(theme, "0 matches")];
        meta.extend(scope_meta);
        let header = status_header(
            theme,
            Some(status_icon(theme, "status.warning", "warning")),
            "Grep",
            "toolTitle",
            description(),
            meta,
        );
        let mut rows = vec![header, empty_row(theme, "No matches found")];
        rows.extend(missing_note);
        return pad_rows(rows);
    }
    let truncated = truncated_of(call, details)
        || details
            .and_then(|d| d.get("meta"))
            .and_then(|m| m.get("limits"))
            .and_then(|l| l.get("columnTruncated"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let mut meta = vec![
        dim_span(theme, format_count("match", match_count)),
        dim_span(theme, format_count("file", file_count)),
    ];
    meta.extend(scope_meta);
    if truncated {
        meta.push(warning_span(theme, "truncated"));
    }
    let header = status_header(
        theme,
        Some(if truncated {
            status_icon(theme, "status.warning", "warning")
        } else {
            search_icon(theme)
        }),
        "Grep",
        "toolTitle",
        description(),
        meta,
    );
    let all: Vec<&str> = text.split('\n').collect();
    let groups = group_line_indices_by_blank(&all);
    let extra = usize::from(missing_note.is_some());
    let budget = (if expanded {
        EXPANDED_TEXT_LIMIT
    } else {
        COLLAPSED_TEXT_LIMIT
    })
    .saturating_sub(extra);
    let mut rows = vec![header];
    rows.extend(budgeted_groups(
        theme,
        &groups,
        &all,
        budget,
        match_count,
        !expanded,
    ));
    rows.extend(missing_note);
    pad_rows(rows)
}

/// `grep` — grouped match rows with counts and a line budget (`grep.ts`).
pub(super) struct Grep;

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
        paint(frame, area, grep_lines(call, false, theme));
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
    fn marked_match_line_formats() {
        assert!(is_marked_match("*12│ fn main()"));
        assert!(is_marked_match("*12: let x = 1;"));
        assert!(!is_marked_match(" 3| return"));
        assert!(!is_marked_match("  context line"));
        assert!(!is_marked_match("# src/lib.rs"));
    }

    fn plain(lines: Vec<Line<'static>>) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn grep_card_matches_ts_title_and_hashline_rows() {
        let theme = Theme::default();
        let call = serde_json::json!({
            "name": "grep",
            "arguments": {"pattern": "beta", "path": "."},
            "result": {
                "content": [{"type": "text", "text": "# e2e-new.txt#0063\n 1:alpha\n*2:beta"}],
                "details": {
                    "scopePath": ".", "matchCount": 1, "fileCount": 1,
                    "displayContent": "# e2e-new.txt#0063\n 1│alpha\n*2│beta"
                }
            }
        });
        let rows = plain(grep_lines(&call, false, &theme));
        assert_eq!(rows[0], " ⌕ Grep: beta 1 match · 1 file · in .");
        assert_eq!(rows[1], " └─ # e2e-new.txt#0063");
        assert_eq!(rows[2], "    *2│beta");
        assert_eq!(rows.len(), 3, "context row compacted away: {rows:?}");
    }

    #[test]
    fn glob_card_matches_ts_title_and_file_rows() {
        let theme = Theme::default();
        let call = serde_json::json!({
            "name": "glob",
            "arguments": {"path": "*.txt"},
            "result": {"text": "e2e-new.txt\ne2e-read-me.txt", "details": {
                "scopePath": ".", "fileCount": 2, "files": ["e2e-new.txt", "e2e-read-me.txt"]
            }}
        });
        let rows = plain(glob_lines(&call, false, &theme));
        assert_eq!(rows[0], " ⌕ Glob: *.txt 2 files · in .");
        assert_eq!(rows[1], " ├─ ◦ e2e-new.txt");
        assert_eq!(rows[2], " └─ ◦ e2e-read-me.txt");
    }

    #[test]
    fn pending_calls_show_hourglass_header() {
        let theme = Theme::default();
        let call = serde_json::json!({"name":"grep","arguments":{"pattern":"x","path":"src"}});
        assert_eq!(
            plain(grep_lines(&call, false, &theme))[0],
            " ◷ Grep: x in src"
        );
        let call = serde_json::json!({"name":"glob","arguments":{}});
        assert_eq!(plain(glob_lines(&call, false, &theme))[0], " ◷ Glob: *");
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
        assert!(out.iter().any(|l| l.contains("▲ No files found")));
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
        assert!(out[0].contains("1 file"));
        assert!(out.last().unwrap().contains("\u{2026} 5 more matches"));
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
