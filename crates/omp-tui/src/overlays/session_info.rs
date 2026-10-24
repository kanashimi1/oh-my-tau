//! `/session` info panel — port of `pi-tui/src/overlays/session-info-overlay.ts`
//! drawn in the `chrome/overlay-box.ts` `OverlayPanel` chrome: a full-width,
//! bottom-anchored rounded box titled "Session Info", the report body in a
//! scroll view (scrollbar column only when it overflows), a divider and a
//! `↑/↓ scroll · ⎋ close` footer.
//!
//! The body comes from the runtime's `{"type":"session_info","lines":[…]}`
//! frame (omp-cli `tui.rs` `session_info_frame`, mirroring TS
//! `CommandController.handleSessionCommand`): one [`SessionInfoLine`] per
//! report line, word-wrapped to the panel's inner width like TS `Text`.

use std::cell::Cell;

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};
use ratatui::Frame;
use serde_json::Value;
use unicode_width::UnicodeWidthChar;

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::theme::Theme;

/// Rows of panel chrome around the body: top border, divider, footer, bottom border.
const PANEL_CHROME_ROWS: u16 = 4;
/// Rows moved by Shift+Up/Down (TS `ScrollView` `fastScrollLines`).
const FAST_SCROLL_LINES: usize = 5;

/// One styled run of a value: text plus an optional theme token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfoSpan {
    /// Text of the run.
    pub text: String,
    /// Theme color token (`success`, `dim`, …); `None` = default foreground.
    pub color: Option<String>,
}

/// One line of the `/session` report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionInfoLine {
    /// `{"k","v"|"vs"}`: dim `Key:`, a space, then the value runs.
    Kv {
        /// Key without the colon.
        key: String,
        /// Value runs.
        value: Vec<InfoSpan>,
    },
    /// `{"head"}`: a bold section heading.
    Head(String),
    /// `{"text","dim"?}`: a free line (`""` = blank).
    Text {
        /// Line text.
        text: String,
        /// Render dim.
        dim: bool,
    },
}

impl SessionInfoLine {
    /// Parse one frame entry; unknown shapes are skipped.
    #[must_use]
    pub fn from_value(v: &Value) -> Option<Self> {
        if let Some(key) = v.get("k").and_then(Value::as_str) {
            let value = if let Some(spans) = v.get("vs").and_then(Value::as_array) {
                spans
                    .iter()
                    .map(|s| InfoSpan {
                        text: s
                            .get("t")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        color: s.get("c").and_then(Value::as_str).map(str::to_owned),
                    })
                    .collect()
            } else {
                vec![InfoSpan {
                    text: v
                        .get("v")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    color: None,
                }]
            };
            return Some(Self::Kv {
                key: key.to_owned(),
                value,
            });
        }
        if let Some(head) = v.get("head").and_then(Value::as_str) {
            return Some(Self::Head(head.to_owned()));
        }
        let text = v.get("text").and_then(Value::as_str)?;
        Some(Self::Text {
            text: text.to_owned(),
            dim: v.get("dim").and_then(Value::as_bool).unwrap_or(false),
        })
    }

    /// Parse the `lines` array of a `session_info` frame.
    #[must_use]
    pub fn parse_frame(frame: &Value) -> Vec<Self> {
        frame
            .get("lines")
            .and_then(Value::as_array)
            .map(|lines| lines.iter().filter_map(Self::from_value).collect())
            .unwrap_or_default()
    }

    /// The line as styled characters (before wrapping).
    fn cells(&self, theme: &Theme) -> Vec<(char, Style)> {
        let dim = Style::default().fg(theme.color("dim"));
        let mut out = Vec::new();
        let mut push = |text: &str, style: Style| out.extend(text.chars().map(|c| (c, style)));
        match self {
            Self::Kv { key, value } => {
                push(&format!("{key}:"), dim);
                push(" ", Style::default());
                for span in value {
                    let style = span
                        .color
                        .as_deref()
                        .map_or_else(Style::default, |c| Style::default().fg(theme.color(c)));
                    push(&span.text, style);
                }
            }
            Self::Head(text) => push(text, Style::default().add_modifier(Modifier::BOLD)),
            Self::Text { text, dim: is_dim } => {
                push(text, if *is_dim { dim } else { Style::default() })
            }
        }
        out
    }
}

fn char_width(c: char) -> usize {
    c.width().unwrap_or(0)
}

/// Word-wrap styled characters to `width` columns (TS `wrapTextWithAnsi`):
/// words move whole to the next row; a word wider than a row hard-breaks.
/// Inter-word spaces at a break are dropped. An empty line stays one row.
fn wrap_cells(cells: &[(char, Style)], width: usize) -> Vec<Vec<(char, Style)>> {
    let width = width.max(1);
    let mut rows: Vec<Vec<(char, Style)>> = Vec::new();
    let mut line: Vec<(char, Style)> = Vec::new();
    let mut line_w = 0usize;
    let mut i = 0usize;
    while i < cells.len() {
        // Gather the run of spaces then the following word.
        let space_start = i;
        while i < cells.len() && cells[i].0 == ' ' {
            i += 1;
        }
        let spaces = &cells[space_start..i];
        let word_start = i;
        while i < cells.len() && cells[i].0 != ' ' {
            i += 1;
        }
        let word = &cells[word_start..i];
        let spaces_w = spaces.len();
        let word_w: usize = word.iter().map(|(c, _)| char_width(*c)).sum();
        if line_w + spaces_w + word_w <= width {
            line.extend_from_slice(spaces);
            line.extend_from_slice(word);
            line_w += spaces_w + word_w;
            continue;
        }
        if line_w > 0 {
            rows.push(std::mem::take(&mut line));
            line_w = 0;
        } else if word.is_empty() {
            // Leading spaces wider than the row: keep what fits.
            line.extend(spaces.iter().take(width).copied());
            line_w = line.len();
            continue;
        }
        for &(c, style) in word {
            let w = char_width(c);
            if line_w + w > width && line_w > 0 {
                rows.push(std::mem::take(&mut line));
                line_w = 0;
            }
            line.push((c, style));
            line_w += w;
        }
    }
    rows.push(line);
    rows
}

/// Merge styled characters into spans.
fn to_line(cells: &[(char, Style)]) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut text = String::new();
    let mut current: Option<Style> = None;
    for &(c, style) in cells {
        if current.is_some_and(|s| s != style) {
            spans.push(Span::styled(
                std::mem::take(&mut text),
                current.unwrap_or_default(),
            ));
        }
        current = Some(style);
        text.push(c);
    }
    if !text.is_empty() {
        spans.push(Span::styled(text, current.unwrap_or_default()));
    }
    Line::from(spans)
}

/// TS `scrollbarThumbRange`: `(start, end)` rows of the thumb.
fn thumb_range(height: usize, total: usize, offset: usize) -> (usize, usize) {
    if height == 0 {
        return (0, 0);
    }
    if total <= height {
        return (0, height);
    }
    let size = ((height * height) / total).clamp(1, height);
    let travel = height - size;
    let max_offset = total - height;
    let offset = offset.min(max_offset);
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let start = ((offset as f64 / max_offset as f64) * travel as f64).round() as usize;
    (start, start + size)
}

/// The focused, dismissible `/session` information panel.
#[derive(Debug, Clone, Default)]
pub struct SessionInfoOverlay {
    /// Panel title; empty = "Session Info" (the `/usage` report reuses the panel).
    title: String,
    lines: Vec<SessionInfoLine>,
    scroll: usize,
    /// `(viewport rows, total rows)` from the last render, for key clamping.
    viewport: Cell<(usize, usize)>,
}

impl SessionInfoOverlay {
    /// Replace the report and reset the scroll position.
    pub fn set_lines(&mut self, lines: Vec<SessionInfoLine>) {
        self.lines = lines;
        self.scroll = 0;
        self.viewport.set((0, 0));
    }

    /// Set the panel title (`None` = "Session Info").
    pub fn set_title(&mut self, title: Option<&str>) {
        self.title = title.unwrap_or_default().to_owned();
    }

    /// The report lines.
    #[must_use]
    pub fn lines(&self) -> &[SessionInfoLine] {
        &self.lines
    }

    /// Current scroll offset (first visible body row).
    #[must_use]
    pub fn scroll(&self) -> usize {
        self.scroll
    }

    fn body_rows(&self, width: usize, theme: &Theme) -> Vec<Vec<(char, Style)>> {
        self.lines
            .iter()
            .flat_map(|line| wrap_cells(&line.cells(theme), width))
            .collect()
    }

    fn scroll_by(&mut self, delta: isize) {
        let (height, total) = self.viewport.get();
        let max = total.saturating_sub(height);
        self.scroll = self.scroll.saturating_add_signed(delta).min(max);
    }
}

impl Component for SessionInfoOverlay {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        if area.width < 4 || area.height < PANEL_CHROME_ROWS + 1 {
            return;
        }
        let width = usize::from(area.width);
        let inner_width = width - 4;
        let max_body = usize::from(area.height - PANEL_CHROME_ROWS).max(1);
        let full = self.body_rows(inner_width, theme);
        let overflow = full.len() > max_body;
        let (rows, content_width) = if overflow {
            let w = inner_width.saturating_sub(1).max(1);
            (self.body_rows(w, theme), w)
        } else {
            (full, inner_width)
        };
        let height = rows.len().clamp(1, max_body);
        self.viewport.set((height, rows.len()));
        let offset = self.scroll.min(rows.len().saturating_sub(height));

        let border = Style::default().fg(theme.color("border"));
        let sym = |k: &str| theme.symbol(k);
        let horizontal = sym("boxRound.horizontal");
        let vertical = sym("boxRound.vertical");

        let mut out: Vec<Line<'static>> = Vec::with_capacity(height + 4);
        // Top border with the inset title (overlay-box `topBorder`).
        let title = format!(
            " {} ",
            if self.title.is_empty() {
                "Session Info"
            } else {
                self.title.as_str()
            }
        );
        let title_w: usize = title.chars().map(char_width).sum();
        let fill = (width - 2).saturating_sub(1 + title_w);
        out.push(Line::from(vec![
            Span::styled(format!("{}{horizontal}", sym("boxRound.topLeft")), border),
            Span::styled(
                title,
                Style::default()
                    .fg(theme.color("accent"))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{}{}", horizontal.repeat(fill), sym("boxRound.topRight")),
                border,
            ),
        ]));
        let boxed = |content: Line<'static>, content_w: usize| {
            let mut spans = vec![Span::styled(format!("{vertical} "), border)];
            spans.extend(content.spans);
            spans.push(Span::raw(" ".repeat(inner_width.saturating_sub(content_w))));
            spans.push(Span::styled(format!(" {vertical}"), border));
            Line::from(spans)
        };
        let (thumb_start, thumb_end) = thumb_range(height, rows.len(), offset);
        for (i, row) in rows.iter().skip(offset).take(height).enumerate() {
            let row_w: usize = row.iter().map(|(c, _)| char_width(*c)).sum();
            let mut line = to_line(row);
            let mut w = row_w;
            if overflow {
                line.spans
                    .push(Span::raw(" ".repeat(content_width.saturating_sub(row_w))));
                let (glyph, token) = if i >= thumb_start && i < thumb_end {
                    ("█", "accent")
                } else {
                    ("│", "dim")
                };
                line.spans
                    .push(Span::styled(glyph, Style::default().fg(theme.color(token))));
                w = content_width + 1;
            }
            out.push(boxed(line, w));
        }
        // Divider (overlay-box `divider`).
        out.push(Line::styled(
            format!(
                "{}{}{}",
                sym("boxSharp.teeRight"),
                horizontal.repeat(width - 2),
                sym("boxSharp.teeLeft")
            ),
            border,
        ));
        // Footer hint (TS `formatKeyHints(["up","down"]) scroll · editorKey(cancel) close`).
        let hint = format!(
            "{}/{} scroll · {} close",
            sym("key.up"),
            sym("key.down"),
            sym("key.esc")
        );
        let hint: String = {
            let mut w = 0;
            hint.chars()
                .take_while(|c| {
                    w += char_width(*c);
                    w <= inner_width
                })
                .collect()
        };
        let hint_w: usize = hint.chars().map(char_width).sum();
        out.push(boxed(
            Line::styled(hint, Style::default().fg(theme.color("dim"))),
            hint_w,
        ));
        out.push(Line::styled(
            format!(
                "{}{}{}",
                sym("boxRound.bottomLeft"),
                horizontal.repeat(width - 2),
                sym("boxRound.bottomRight")
            ),
            border,
        ));

        #[allow(clippy::cast_possible_truncation)]
        let panel_h = (out.len() as u16).min(area.height);
        let panel = Rect {
            x: area.x,
            y: area.y + area.height - panel_h,
            width: area.width,
            height: panel_h,
        };
        frame.render_widget(Clear, panel);
        frame.render_widget(Paragraph::new(out), panel);
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        let (height, total) = self.viewport.get();
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        #[allow(clippy::cast_possible_wrap)]
        let page = height.max(1) as isize;
        #[allow(clippy::cast_possible_wrap)]
        let fast = FAST_SCROLL_LINES as isize;
        match key.code {
            KeyCode::Up => self.scroll_by(if shift { -fast } else { -1 }),
            KeyCode::Down => self.scroll_by(if shift { fast } else { 1 }),
            KeyCode::PageUp => self.scroll_by(-page),
            KeyCode::PageDown => self.scroll_by(page),
            KeyCode::Home => self.scroll = 0,
            KeyCode::End => self.scroll = total.saturating_sub(height),
            // Esc bubbles: the app closes the top overlay on an ignored Esc.
            _ => return EventOutcome::Ignored,
        }
        EventOutcome::Consumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    /// The e2e `/session` report for a fresh mock session.
    fn e2e_frame() -> Value {
        json!({"type": "session_info", "lines": [
            {"k": "File", "v": "/tmp/e2e-home-yjtazx1t/.omp/agent/sessions/-tmp-e2e-home-yjtazx1t-work/2026-10-16T00-04-04-462Z_01a14205-e2ee-7000-be2b-883cca3faf0d.jsonl"},
            {"k": "ID", "v": "01a14205-e2ee-7000-be2b-883cca3faf0d"},
            {"text": ""},
            {"head": "Provider"},
            {"k": "Name", "v": "mock"},
            {"k": "Model", "v": "text"},
            {"k": "API", "v": "openai-completions"},
            {"k": "Auth", "v": "runtime/fallback"},
            {"k": "Endpoint", "v": "http://127.0.0.1:45133/v1"},
            {"k": "Source", "v": "config override (models.yml)"},
            {"text": ""},
            {"head": "Messages"},
            {"k": "User", "v": "0"},
            {"k": "Assistant", "v": "0"},
            {"k": "Tool Calls", "v": "0"},
            {"k": "Tool Results", "v": "0"},
            {"k": "Total", "v": "0"},
            {"text": ""},
            {"k": "Append-Only", "vs": [{"t": "active", "c": "success"}, {"t": " (setting: auto (mock))"}]},
            {"head": "Tokens"},
            {"k": "Input", "v": "0"},
            {"k": "Output", "v": "0"},
            {"k": "Total", "v": "0"},
            {"text": ""},
            {"head": "MCP Servers"},
            {"text": "None connected", "dim": true},
            {"text": ""},
        ]})
    }

    fn render(overlay: &SessionInfoOverlay, w: u16, h: u16) -> Vec<String> {
        let theme = Theme::default();
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        term.draw(|f| overlay.render(f, f.area(), &theme)).unwrap();
        let buf = term.backend().buffer().clone();
        (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| buf[(x, y)].symbol().to_owned())
                    .collect::<String>()
                    .trim_end()
                    .to_owned()
            })
            .collect()
    }

    fn overlay() -> SessionInfoOverlay {
        let mut o = SessionInfoOverlay::default();
        o.set_lines(SessionInfoLine::parse_frame(&e2e_frame()));
        o
    }

    #[test]
    fn session_info_render_matches_ts_screen() {
        let rows = render(&overlay(), 120, 40);
        // Optional TS ground-truth capture, supplied via the environment.
        let ts = std::env::var_os("OMT_TS_SLASH_SESSION_CAPTURE")
            .ok_or(())
            .and_then(|p| std::fs::read_to_string(p).map_err(|_| ()));
        // Panel: 29 body rows + 4 chrome rows, bottom-anchored.
        assert!(rows[..7].iter().all(String::is_empty), "{rows:#?}");
        let panel = &rows[7..];
        assert!(panel[0].starts_with("╭─ Session Info ─"), "{}", panel[0]);
        assert!(panel[0].ends_with('╮'));
        assert_eq!(panel[0].chars().count(), 120);
        assert_eq!(panel[1], format!("│ File:{} │", " ".repeat(111)));
        assert!(panel[2].starts_with("│ /tmp/e2e-home-yjtazx1t/"));
        assert!(panel[3].starts_with("│ e2b-883cca3faf0d.jsonl "));
        assert!(panel
            .iter()
            .any(|r| r.starts_with("│ Endpoint: http://127.0.0.1:45133/v1 ")));
        assert_eq!(panel[29].chars().count(), 120);
        assert!(panel[30].starts_with("├─") && panel[30].ends_with('┤'));
        assert!(
            panel[31].starts_with("│ ↑/↓ scroll · ⎋ close "),
            "{}",
            panel[31]
        );
        assert!(panel[32].starts_with("╰─") && panel[32].ends_with('╯'));
        // Text-exact against the TS ground truth when the capture is present.
        if let Ok(ts) = ts {
            let ts_panel: Vec<&str> = ts.lines().skip_while(|l| !l.starts_with('╭')).collect();
            let ours: Vec<&str> = panel.iter().map(String::as_str).collect();
            assert_eq!(
                ours,
                ts_panel.iter().map(|l| l.trim_end()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn session_info_scrolls_when_overflowing() {
        let mut o = overlay();
        // 12 rows: 8 body rows visible out of 30 (rewrapped at width-1).
        let rows = render(&o, 120, 12);
        assert!(rows[0].starts_with("╭─ Session Info"));
        assert!(rows[1].ends_with("█ │"), "thumb at top: {}", rows[1]);
        assert!(rows[1].starts_with("│ File:"));
        assert!(o.handle_key(KeyEvent::from(KeyCode::Down)).consumed());
        assert_eq!(o.scroll(), 1);
        o.handle_key(KeyEvent::from(KeyCode::End));
        let rows = render(&o, 120, 12);
        assert!(rows[7].starts_with("│ None connected"), "{rows:#?}");
        assert!(o.handle_key(KeyEvent::from(KeyCode::PageDown)).consumed());
        let max = o.scroll();
        o.handle_key(KeyEvent::from(KeyCode::Down));
        assert_eq!(o.scroll(), max, "clamped at the end");
        // Esc bubbles to the host (which pops the overlay).
        assert!(!o.handle_key(KeyEvent::from(KeyCode::Esc)).consumed());
    }

    #[test]
    fn session_info_wrap_hard_breaks_long_words() {
        let style = Style::default();
        let cells: Vec<(char, Style)> = "ab cdefghij k".chars().map(|c| (c, style)).collect();
        let rows: Vec<String> = wrap_cells(&cells, 4)
            .iter()
            .map(|r| r.iter().map(|(c, _)| *c).collect())
            .collect();
        assert_eq!(rows, ["ab", "cdef", "ghij", "k"]);
        assert_eq!(wrap_cells(&[], 4).len(), 1);
    }
}
