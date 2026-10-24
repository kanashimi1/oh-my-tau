//! `edit` tool renderer — port of `pi-tui/src/tools/edit.ts` (P0).
//!
//! Covers the unified-edit, hashline line-edit and delete/move shapes:
//! pending (streaming) cards, success/error result cards with a numbered
//! unified diff, per-hunk collapse budgets, and the multi-file fan-out.
//! The `write` tool (`tools/write.rs`) reuses the shared helpers here.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use serde_json::Value;

use crate::theme::Theme;

use super::{error_detail_lines, ToolRenderer};

/// Collapse budgets (TS `PREVIEW_LIMITS` edit-specific rows:
/// `DIFF_COLLAPSED_HUNKS` / `DIFF_COLLAPSED_LINES`).
pub const DIFF_COLLAPSED_HUNKS: usize = 8;
/// Collapse budgets: max rendered rows while collapsed.
pub const DIFF_COLLAPSED_LINES: usize = 40;
/// Rows kept in the collapsed streaming/pending preview.
pub const PREVIEW_TAIL_LINES: usize = 12;

/// Lifecycle phase of an edit card (TS `ToolCardPhase`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Call issued, no result yet.
    Pending,
    /// Result still streaming.
    Partial,
    /// Settled successfully.
    Success,
    /// Settled with an error.
    Error,
}

/// Renderer for the `edit` tool family (unified, hashline, patch, move).
pub struct Edit;

impl ToolRenderer for Edit {
    fn summary(&self, call: &Value) -> String {
        let path = edit_path(call).unwrap_or_else(|| "…".into());
        let op = edit_op(call);
        if op == "Move" {
            let target = move_target(call).unwrap_or_else(|| "…".into());
            format!("{path} → {target}")
        } else {
            format!("{op} {path}")
        }
    }

    fn render(
        &self,
        frame: &mut ratatui::Frame<'_>,
        area: ratatui::layout::Rect,
        call: &Value,
        theme: &Theme,
    ) {
        let width = area.width as usize;
        let lines = build_card_lines(call, None, false, width, theme);
        frame.render_widget(Paragraph::new(lines), area);
    }
}

/// The arguments object of a call, whichever wire key carries it.
#[must_use]
pub(crate) fn args_of(call: &Value) -> &Value {
    call.get("arguments")
        .or_else(|| call.get("input"))
        .unwrap_or(&Value::Null)
}

/// First non-empty string among `keys` on `v`.
pub(crate) fn arg_str<'v>(v: &'v Value, keys: &[&str]) -> Option<&'v str> {
    keys.iter().find_map(|k| v.get(k).and_then(Value::as_str))
}

/// The target path of an edit call (unified `path`/`file_path`, or the
/// first entry of a hashline `edits` array).
#[must_use]
pub fn edit_path(call: &Value) -> Option<String> {
    let args = args_of(call);
    arg_str(args, &["file_path", "filePath", "path"])
        .map(str::to_owned)
        .or_else(|| {
            first_edit(call)
                .and_then(|e| arg_str(e, &["file_path", "filePath", "path"]).map(str::to_owned))
        })
}

fn first_edit(call: &Value) -> Option<&Value> {
    args_of(call)
        .get("edits")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
}

/// Human verb for the header: `Create`/`Delete`/`Move`/`Edit`.
#[must_use]
pub fn edit_op(call: &Value) -> String {
    let args = args_of(call);
    let raw = arg_str(args, &["op", "operation"]).unwrap_or("");
    match raw {
        "" => {
            if move_target(call).is_some() {
                "Move".to_owned()
            } else {
                "Edit".to_owned()
            }
        }
        s => {
            let mut c = s.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => "Edit".to_owned(),
            }
        }
    }
}

fn move_target(call: &Value) -> Option<String> {
    let args = args_of(call);
    arg_str(args, &["to", "dst", "new_path", "newPath", "target"]).map(str::to_owned)
}

/// True for delete/move calls that carry no replacement payload — those
/// render as a bare inline row, not a framed card (TS `renderInlineEditRow`).
fn is_payload_free(call: &Value) -> bool {
    let args = args_of(call);
    let op = arg_str(args, &["op", "operation"])
        .unwrap_or("")
        .to_ascii_lowercase();
    let has_payload = arg_str(
        args,
        &["old_string", "new_string", "patch", "content", "edits"],
    )
    .is_some()
        || args
            .get("edits")
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty());
    (op == "delete" || op == "move" || move_target(call).is_some()) && !has_payload
}

/// Derive the card phase from a call and its optional tool result
/// (`pa-types` `toolResult` message shape: `isError`, `details.*`).
#[must_use]
pub fn phase_of(call: &Value, result: Option<&Value>) -> Phase {
    let _ = call;
    match result {
        None => Phase::Pending,
        Some(r) if r.get("isError").and_then(Value::as_bool).unwrap_or(false) => Phase::Error,
        Some(r) => {
            let streaming = r
                .get("details")
                .and_then(|d| d.get("isPartial").or_else(|| d.get("streaming")))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if streaming {
                Phase::Partial
            } else {
                Phase::Success
            }
        }
    }
}

/// Truncate `s` to `width` display cells with a Unicode ellipsis.
#[must_use]
pub(crate) fn truncate_width(s: &str, width: usize) -> String {
    let count = s.chars().count();
    if count <= width {
        return s.to_owned();
    }
    match width {
        0 => String::new(),
        1 => "…".to_owned(),
        _ => {
            let kept: String = s.chars().take(width - 1).collect();
            format!("{kept}…")
        }
    }
}

/// Frame a card: `body[0]` is the header row and rides in the top bar
/// (`╭─── ✎ Edit: 🗒 path ⟦+1/-1⟧ ──╮`); the rest are flush-left content rows
/// (TS edit frames use `contentPaddingLeft: 0`).
pub(crate) fn frame_card(
    mut body: Vec<Line<'static>>,
    width: usize,
    theme: &Theme,
    border_token: &str,
) -> Vec<Line<'static>> {
    let header = (!body.is_empty()).then(|| body.remove(0));
    let border = if border_token == "border" {
        super::frame::token(theme, "borderMuted", "text_muted")
    } else {
        theme.color(border_token)
    };
    let sections = if body.is_empty() {
        Vec::new()
    } else {
        vec![super::frame::CardSection::plain(body)]
    };
    super::frame::framed_card_padded(header, sections, width, border, 0)
}

/// Concatenated text blocks of a tool-result message.
#[must_use]
pub(crate) fn result_text(result: &Value) -> String {
    // The chat card hands the flattened text as `{"text": ...}` (a settled toolResult message
    // carries `content` blocks instead).
    if let Some(text) = result.get("text").and_then(Value::as_str) {
        return text.to_owned();
    }
    let Some(blocks) = result.get("content").and_then(Value::as_array) else {
        return String::new();
    };
    blocks
        .iter()
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Tail-window streaming preview: when capped, a `… (N earlier lines)` row
/// followed by the last `budget` rows with dim zero-padded gutter numbers,
/// then a dim `(streaming)` row.
pub(crate) fn streaming_preview_lines(
    content: &str,
    budget: usize,
    width: usize,
    theme: &Theme,
    streaming: bool,
) -> Vec<Line<'static>> {
    let inner = width.saturating_sub(4).max(1);
    let rows: Vec<&str> = content.lines().collect();
    let mut lines = Vec::new();
    let tail_start = rows.len().saturating_sub(budget);
    if tail_start > 0 {
        lines.push(Line::from(Span::styled(
            truncate_width(&format!("… ({tail_start} earlier lines)"), inner),
            Style::default().fg(theme.color("text_muted")),
        )));
    }
    let gutter = rows.len().to_string().len().max(2);
    for (i, row) in rows.iter().enumerate().skip(tail_start) {
        let gutter_text = format!("{:>gutter$} ", i + 1);
        lines.push(Line::from(vec![
            Span::styled(gutter_text, Style::default().fg(theme.color("text_muted"))),
            Span::raw(truncate_width(row, inner.saturating_sub(gutter + 1))),
        ]));
    }
    if streaming {
        lines.push(Line::from(Span::styled(
            "(streaming)",
            Style::default().fg(theme.color("text_muted")),
        )));
    }
    lines
}

/// `+added`/`-removed` counts from numbered diff rows (`+13|…`, `-4|…`;
/// `+++`/`---` file headers never counted).
#[must_use]
pub fn diff_stats(diff: &str) -> (usize, usize) {
    let mut added = 0;
    let mut removed = 0;
    for row in diff.lines() {
        if let Some(rest) = row.strip_prefix('+') {
            if !rest.starts_with('+') {
                added += 1;
            }
        } else if let Some(rest) = row.strip_prefix('-') {
            if !rest.starts_with('-') {
                removed += 1;
            }
        }
    }
    (added, removed)
}

/// Resolve the numbered diff for a call/result pair, most authoritative
/// first: `details.diff`, then `details.oldText/newText`, then a fallback
/// computed from the unified `old_string`/`new_string` args via the vendored
/// `pi-edit` engine.
#[must_use]
pub fn resolve_diff(call: &Value, result: Option<&Value>, streaming: bool) -> Option<String> {
    if let Some(details) = result.and_then(|r| r.get("details")) {
        if let Some(diff) = details.get("diff").and_then(Value::as_str) {
            if !diff.is_empty() {
                return Some(diff.to_owned());
            }
        }
        let old = arg_str(details, &["oldText", "old_text", "before"]);
        let new = arg_str(details, &["newText", "new_text", "after"]);
        if let (Some(old), Some(new)) = (old, new) {
            if old == new {
                return None;
            }
            let source = pi_edit::diff_string::BlockContextSource {
                path: None,
                lang: None,
                streaming,
            };
            let output = pi_edit::diff_string::LineDiff::new(old, new).numbered(None, &source);
            return Some(output.diff);
        }
    }
    let args = args_of(call);
    let old = arg_str(args, &["old_string", "oldString"])?;
    let new = arg_str(args, &["new_string", "newString"])?;
    if old == new {
        return None;
    }
    let source = pi_edit::diff_string::BlockContextSource {
        path: None,
        lang: None,
        streaming,
    };
    Some(
        pi_edit::diff_string::LineDiff::new(old, new)
            .numbered(None, &source)
            .diff,
    )
}

/// Convert hashline edit text (`*13:new text`, context ` 14|old`) into
/// numbered diff rows (`+13|new text`).
#[must_use]
pub fn hashline_rows(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let bytes = line.as_bytes();
            match bytes.first().copied() {
                Some(b'*') => Some(format!("+{}", &line[1..])),
                Some(b'+' | b'-') => Some(line.to_owned()),
                Some(b' ') if bytes.get(1).is_some_and(u8::is_ascii_digit) => Some(line.to_owned()),
                _ => None,
            }
        })
        .collect()
}

/// Collapse numbered diff rows by hunk: at most `max_hunks` hunks and
/// `max_lines` rows are kept; returns the kept rows plus the surplus counts
/// for the `… (N more hunks, M more lines)` footer.
#[must_use]
pub fn truncate_diff_rows(
    rows: &[String],
    max_hunks: usize,
    max_lines: usize,
) -> (Vec<String>, usize, usize) {
    // A new hunk starts at a change row preceded by a context gap of > 3
    // rows (mirrors the unified context the rows were built with).
    let is_change = |r: &str| {
        (r.starts_with('+') && !r.starts_with("++")) || (r.starts_with('-') && !r.starts_with("--"))
    };
    let mut hunk_of_row = Vec::with_capacity(rows.len());
    let mut current_hunk = 0_usize;
    let mut total_hunks = 0_usize;
    let mut ctx_run = 0_usize;
    for row in rows {
        if is_change(row) {
            if total_hunks == 0 || ctx_run > 3 {
                current_hunk += 1;
                total_hunks += 1;
            }
            ctx_run = 0;
        } else if !row.starts_with('\\') && !row.starts_with("@@") {
            ctx_run += 1;
        }
        hunk_of_row.push(current_hunk);
    }
    let mut kept = Vec::new();
    let mut kept_hunks = 0_usize;
    let mut last_hunk = 0_usize;
    let mut more_lines = 0_usize;
    for (i, row) in rows.iter().enumerate() {
        let hunk = hunk_of_row[i];
        let new_hunk = hunk != last_hunk;
        if (new_hunk && kept_hunks >= max_hunks) || kept.len() >= max_lines {
            more_lines += 1;
            continue;
        }
        if new_hunk {
            kept_hunks += 1;
            last_hunk = hunk;
        }
        kept.push(row.clone());
    }
    (kept, total_hunks.saturating_sub(kept_hunks), more_lines)
}

/// Style one numbered diff row: gutter (up to `|`) dim, body colored by
/// prefix (`+` success, `-` error, `@@` accent).
fn diff_row_line(row: &str, width: usize, theme: &Theme) -> Line<'static> {
    let inner = width.saturating_sub(4).max(1);
    let truncated = truncate_width(row, inner);
    let split = truncated.find('|');
    let (gutter, body) = match split {
        Some(i) => (&truncated[..i], &truncated[i..]),
        None => ("", truncated.as_str()),
    };
    let body_style = match truncated.as_bytes().first().copied() {
        Some(b'+') => Style::default().fg(theme.color("success")),
        Some(b'-') => Style::default().fg(theme.color("error")),
        Some(b'@') => Style::default().fg(theme.color("accent")),
        _ => Style::default(),
    };
    let mut spans = vec![Span::styled(
        gutter.to_owned(),
        Style::default().fg(theme.color("text_muted")),
    )];
    if body.starts_with('+') || body.starts_with('-') {
        spans.push(Span::styled(
            body[..1].to_owned(),
            body_style.add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(body[1..].to_owned(), body_style));
    } else {
        spans.push(Span::styled(body.to_owned(), body_style));
    }
    Line::from(spans)
}

/// Header row (TS `renderEditHeader`): `[icon ]Edit: 🗒 path ⟦+a/-r⟧`.
fn header_line(
    op: &str,
    path: &str,
    phase: Phase,
    stats: Option<(usize, usize)>,
    _width: usize,
    theme: &Theme,
) -> Line<'static> {
    use super::frame::{lang_icon, status_header, token};
    let icon = match phase {
        Phase::Pending => Some(("status.pending", "muted")),
        Phase::Partial => Some(("status.running", "accent")),
        Phase::Success => Some((
            if op == "Delete" {
                "tool.delete"
            } else {
                "tool.edit"
            },
            "accent",
        )),
        Phase::Error => Some(("status.error", "error")),
    }
    .map(|(key, tok)| {
        Span::styled(
            theme.symbol(key).to_owned(),
            Style::default().fg(token_of(theme, tok)),
        )
    });
    let dim = Style::default().fg(token(theme, "dim", "text_muted"));
    let mut description = vec![
        Span::styled(
            format!("{} ", lang_icon(theme, path)),
            Style::default().fg(token(theme, "muted", "text_muted")),
        ),
        Span::styled(path.to_owned(), Style::default().fg(theme.color("accent"))),
    ];
    if let Some((a, r)) = stats.filter(|(a, r)| a + r > 0) {
        description.push(Span::styled(
            format!(" {}", theme.symbol("format.bracketLeft")),
            dim,
        ));
        let mut parts: Vec<Span<'static>> = Vec::new();
        if a > 0 {
            parts.push(Span::styled(
                format!("+{a}"),
                Style::default().fg(token(theme, "toolDiffAdded", "success")),
            ));
        }
        if r > 0 {
            if !parts.is_empty() {
                parts.push(Span::styled("/".to_owned(), dim));
            }
            parts.push(Span::styled(
                format!("-{r}"),
                Style::default().fg(token(theme, "toolDiffRemoved", "error")),
            ));
        }
        description.extend(parts);
        description.push(Span::styled(
            theme.symbol("format.bracketRight").to_owned(),
            dim,
        ));
    }
    status_header(theme, icon, op, "accent", description, Vec::new())
}

fn token_of(theme: &Theme, name: &str) -> ratatui::style::Color {
    super::frame::token(theme, name, "text_muted")
}

/// Dim expand-hint footer (TS `formatExpandHint`).
fn expand_hint_line(
    more_hunks: usize,
    more_lines: usize,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Line<'static> {
    let inner = width.saturating_sub(4).max(1);
    let text = if expanded {
        format!("… ({more_hunks} more hunks, {more_lines} more lines)")
    } else {
        format!("… ({more_hunks} more hunks, {more_lines} more lines) [ctrl+o: Expand]")
    };
    Line::from(Span::styled(
        truncate_width(&text, inner),
        Style::default().fg(theme.color("text_muted")),
    ))
}

/// Dim diagnostics lines from `details.diagnostics` (TS `formatDiagnostics`).
fn diagnostics_lines(result: &Value, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let inner = width.saturating_sub(4).max(1);
    result
        .get("details")
        .and_then(|d| d.get("diagnostics"))
        .and_then(Value::as_array)
        .map(|diags| {
            diags
                .iter()
                .filter_map(Value::as_str)
                .map(|d| {
                    Line::from(Span::styled(
                        truncate_width(d, inner),
                        Style::default().fg(theme.color("text_muted")),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Multi-file result rows from `details.perFileResults`
/// (`[{path, diff?, isError?}]`), one framed block per finished file.
fn per_file_results(result: &Value) -> Option<Vec<(String, Option<String>, bool)>> {
    result
        .get("details")?
        .get("perFileResults")?
        .as_array()
        .map(|files| {
            files
                .iter()
                .map(|f| {
                    (
                        arg_str(f, &["path", "file_path"]).unwrap_or("…").to_owned(),
                        f.get("diff").and_then(Value::as_str).map(str::to_owned),
                        f.get("isError").and_then(Value::as_bool).unwrap_or(false),
                    )
                })
                .collect()
        })
}

/// Build the full card for one edit call. `result` is the settled
/// `toolResult` message when present; `expanded` selects collapsed (default)
/// vs expanded previews.
#[must_use]
pub fn build_card_lines(
    call: &Value,
    result: Option<&Value>,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let width = width.max(8);
    let phase = phase_of(call, result);
    let op = edit_op(call);
    let path = edit_path(call).unwrap_or_else(|| "…".into());

    // Delete / move with no payload: bare inline row, no frame.
    if is_payload_free(call) {
        let glyph = theme.glyph("collapsed");
        let text = if op == "Move" {
            format!(
                "{glyph} {path} → {}",
                move_target(call).unwrap_or_else(|| "…".into())
            )
        } else {
            format!("{glyph} {op} {path}")
        };
        let (icon, color) = match phase {
            Phase::Error => (theme.glyph("failure"), theme.color("error")),
            Phase::Pending => ("⧗", theme.color("warning")),
            _ => (theme.glyph("success"), theme.color("success")),
        };
        return vec![Line::from(Span::styled(
            truncate_width(&format!("{icon} {text}"), width),
            Style::default().fg(color),
        ))];
    }

    let mut body = Vec::new();
    let mut border_token = "border";
    let mut stats = None;
    if phase == Phase::Success || phase == Phase::Partial {
        if let Some(diff) = resolve_diff(call, result, phase == Phase::Partial) {
            stats = Some(diff_stats(&diff));
        }
    }
    body.push(header_line(&op, &path, phase, stats, width, theme));

    match phase {
        Phase::Pending => {
            let new_text = arg_str(args_of(call), &["new_string", "newString"]).unwrap_or("");
            if new_text.is_empty() {
                body.push(Line::from(Span::styled(
                    "…",
                    Style::default().fg(theme.color("text_muted")),
                )));
            } else {
                body.extend(streaming_preview_lines(
                    new_text,
                    PREVIEW_TAIL_LINES,
                    width,
                    theme,
                    true,
                ));
            }
        }
        Phase::Error => {
            border_token = "error";
            let text = result.map(result_text).unwrap_or_default();
            if text.is_empty() {
                body.push(Line::from(Span::styled(
                    "  Unknown error",
                    Style::default().fg(theme.color("error")),
                )));
            } else {
                body.extend(error_detail_lines(&text, width, theme));
            }
        }
        Phase::Partial | Phase::Success => {
            if let Some(files) = result.and_then(per_file_results) {
                if files.len() > 1 {
                    return multi_file_card(&files, phase, expanded, width, theme);
                }
            }
            match resolve_diff(call, result, phase == Phase::Partial) {
                Some(diff) => {
                    let rows: Vec<String> = diff.lines().map(str::to_owned).collect();
                    let (kept, more_hunks, more_lines) = if expanded {
                        (rows, 0, 0)
                    } else {
                        truncate_diff_rows(&rows, DIFF_COLLAPSED_HUNKS, DIFF_COLLAPSED_LINES)
                    };
                    for row in &kept {
                        body.push(diff_row_line(row, width, theme));
                    }
                    if more_hunks > 0 || more_lines > 0 {
                        body.push(expand_hint_line(
                            more_hunks, more_lines, expanded, width, theme,
                        ));
                    }
                }
                None => {
                    // create / no-op: nothing to diff against.
                    body.push(Line::from(Span::styled(
                        truncate_width(&format!("No changes were made to {path}."), width - 4),
                        Style::default().fg(theme.color("text_muted")),
                    )));
                }
            }
            if let Some(result) = result {
                body.extend(diagnostics_lines(result, width, theme));
            }
        }
    }
    frame_card(body, width, theme, border_token)
}

/// One framed block per finished file, blank line between, then a spinner
/// row for still-pending files (TS `renderMultiFileResult`).
fn multi_file_card(
    files: &[(String, Option<String>, bool)],
    phase: Phase,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut pending = 0_usize;
    for (path, diff, is_error) in files {
        let file_phase = if *is_error {
            Phase::Error
        } else {
            Phase::Success
        };
        let mut body = vec![header_line(
            "Edit",
            path,
            file_phase,
            diff.as_ref().map(|d| diff_stats(d)),
            width,
            theme,
        )];
        let border = if *is_error { "error" } else { "border" };
        match diff {
            Some(diff) => {
                let rows: Vec<String> = diff.lines().map(str::to_owned).collect();
                let (kept, more_hunks, more_lines) = if expanded {
                    (rows, 0, 0)
                } else {
                    truncate_diff_rows(&rows, DIFF_COLLAPSED_HUNKS, DIFF_COLLAPSED_LINES)
                };
                for row in &kept {
                    body.push(diff_row_line(row, width, theme));
                }
                if more_hunks > 0 || more_lines > 0 {
                    body.push(expand_hint_line(
                        more_hunks, more_lines, expanded, width, theme,
                    ));
                }
            }
            None => pending += 1,
        }
        if !out.is_empty() {
            out.push(Line::default());
        }
        out.extend(frame_card(body, width, theme, border));
    }
    if phase == Phase::Partial || pending > 0 {
        out.push(Line::from(Span::styled(
            truncate_width(&format!("Edit — {pending} more files pending…"), width),
            Style::default().fg(theme.color("accent")),
        )));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    fn render_to_string(lines: Vec<Line<'static>>, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                let area = ratatui::layout::Rect::new(0, 0, width, height);
                f.render_widget(Paragraph::new(lines), area);
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        let mut out = String::new();
        for y in 0..height {
            for x in 0..width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    fn ok_result(diff: &str) -> Value {
        json!({"role":"toolResult","toolCallId":"c1","toolName":"edit","isError":false,
               "details":{"diff": diff},"timestamp":1})
    }

    #[test]
    fn summary_unified_edit() {
        let call = json!({"name":"edit","arguments":{"path":"src/main.rs","old_string":"a","new_string":"b"}});
        assert_eq!(Edit.summary(&call), "Edit src/main.rs");
    }

    #[test]
    fn summary_move() {
        let call = json!({"name":"edit","arguments":{"path":"a.rs","op":"move","to":"b.rs"}});
        assert_eq!(Edit.summary(&call), "a.rs → b.rs");
    }

    #[test]
    fn phase_detection() {
        let call = json!({"name":"edit","arguments":{"path":"p"}});
        assert_eq!(phase_of(&call, None), Phase::Pending);
        assert_eq!(phase_of(&call, Some(&ok_result("+1|x"))), Phase::Success);
        assert_eq!(
            phase_of(&call, Some(&json!({"isError": true}))),
            Phase::Error
        );
        assert_eq!(
            phase_of(
                &call,
                Some(&json!({"isError": false, "details": {"isPartial": true}}))
            ),
            Phase::Partial
        );
    }

    #[test]
    fn diff_stats_counts_rows() {
        assert_eq!(diff_stats("+1|a\n-2|b\n 3|c\n+++ x\n--- y\n"), (1, 1));
    }

    #[test]
    fn unified_fallback_diff_via_pi_edit() {
        let call = json!({"name":"edit","arguments":{
            "path":"f.rs",
            "old_string":"let a = 1;\nlet b = 2;\n",
            "new_string":"let a = 1;\nlet b = 3;\n"}});
        let diff = resolve_diff(&call, None, false).expect("fallback diff");
        assert!(diff.contains("-2|let b = 2;"), "got: {diff}");
        assert!(diff.contains("+2|let b = 3;"), "got: {diff}");
    }

    #[test]
    fn hashline_rows_convert_star_prefix() {
        let rows = hashline_rows("*13:new line\n 12|ctx\nnot a row\n");
        assert_eq!(rows, vec!["+13:new line".to_owned(), " 12|ctx".to_owned()]);
    }

    #[test]
    fn collapse_budgets_hunks_and_lines() {
        // 60 change rows with no context between them form ONE hunk; the
        // line budget (40) caps the kept rows.
        let rows: Vec<String> = (1..=60).map(|i| format!("+{i}|line {i}")).collect();
        let (kept, more_hunks, more_lines) = truncate_diff_rows(&rows, 8, 40);
        assert_eq!(kept.len(), 40);
        assert_eq!(more_hunks, 0);
        assert_eq!(more_lines, 20);
    }

    #[test]
    fn collapse_ignores_context_gaps() {
        // Two hunks: change, 5 context rows, change. Cap to 1 hunk.
        let mut rows = vec!["+1|a".to_owned()];
        rows.extend((2..=6).map(|i| format!(" {i}|ctx")));
        rows.push("+7|b".to_owned());
        let (kept, more_hunks, _) = truncate_diff_rows(&rows, 1, 40);
        assert!(kept.iter().any(|r| r == "+1|a"));
        assert!(!kept.iter().any(|r| r == "+7|b"));
        assert_eq!(more_hunks, 1);
    }

    #[test]
    fn pending_card_renders_streaming_preview() {
        let call = json!({"name":"edit","arguments":{
            "path":"src/lib.rs",
            "old_string":"x",
            "new_string":"line1\nline2\nline3"}});
        let theme = Theme::default();
        let lines = build_card_lines(&call, None, false, 40, &theme);
        let text = render_to_string(lines, 40, 12);
        assert!(
            text.contains("Edit: ") && text.contains("src/lib.rs"),
            "got:\n{text}"
        );
        assert!(text.contains("line3"), "got:\n{text}");
        assert!(text.contains("(streaming)"), "got:\n{text}");
        assert!(text.contains('╭'), "framed card expected:\n{text}");
    }

    #[test]
    fn success_card_shows_diff_and_stats() {
        let call = json!({"name":"edit","arguments":{"path":"src/lib.rs"}});
        let result = ok_result(" 1|keep\n-2|old\n+2|new");
        let theme = Theme::default();
        let lines = build_card_lines(&call, Some(&result), false, 40, &theme);
        let text = render_to_string(lines, 40, 12);
        assert!(
            text.contains("Edit: ") && text.contains("src/lib.rs"),
            "got:\n{text}"
        );
        assert!(text.contains("⟦+1/-1⟧"), "stats missing:\n{text}");
        assert!(text.contains("-2|old"), "got:\n{text}");
        assert!(text.contains("+2|new"), "got:\n{text}");
    }

    #[test]
    fn error_card_red_with_stripped_prefix() {
        let call = json!({"name":"edit","arguments":{"path":"src/lib.rs","old_string":"a","new_string":"b"}});
        let result = json!({"isError": true,
            "content":[{"type":"text","text":"Error: string not found"}]});
        let theme = Theme::default();
        let lines = build_card_lines(&call, Some(&result), false, 40, &theme);
        let text = render_to_string(lines, 40, 8);
        assert!(text.contains("string not found"), "got:\n{text}");
        assert!(
            !text.contains("Error: string not found"),
            "prefix should be stripped:\n{text}"
        );
    }

    #[test]
    fn no_changes_message() {
        let call = json!({"name":"edit","arguments":{"path":"src/lib.rs","old_string":"a","new_string":"a"}});
        let result = ok_result("");
        let theme = Theme::default();
        let lines = build_card_lines(&call, Some(&result), false, 40, &theme);
        let text = render_to_string(lines, 40, 8);
        assert!(
            text.contains("No changes were made to src/lib.rs."),
            "got:\n{text}"
        );
    }

    #[test]
    fn delete_renders_inline_row_not_framed() {
        let call = json!({"name":"edit","arguments":{"path":"old.rs","op":"delete"}});
        let theme = Theme::default();
        let lines = build_card_lines(&call, None, false, 40, &theme);
        assert_eq!(lines.len(), 1);
        let text = render_to_string(lines, 40, 4);
        assert!(text.contains("Delete old.rs"), "got:\n{text}");
    }

    #[test]
    fn multi_file_result_renders_blocks_and_pending_row() {
        let call = json!({"name":"edit","arguments":{"path":"a.rs"}});
        let result = json!({"isError": false, "details": {"isPartial": true, "perFileResults": [
            {"path":"a.rs","diff":"-1|x\n+1|y"},
            {"path":"b.rs"},
        ]}});
        let theme = Theme::default();
        let lines = build_card_lines(&call, Some(&result), false, 40, &theme);
        let text = render_to_string(lines, 40, 16);
        assert!(
            text.contains("Edit: ") && text.contains("a.rs"),
            "got:\n{text}"
        );
        assert!(text.contains("b.rs"), "got:\n{text}");
        assert!(text.contains("1 more files pending"), "got:\n{text}");
    }
}
