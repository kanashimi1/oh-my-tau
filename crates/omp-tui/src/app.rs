//! The root application state: owns every component and routes input.
//!
//! Layout (root vertical split, matching pi-tui's default chrome):
//!
//! ```text
//! ┌────────────────────────────────────────┐
//! │ chat: scrollback (messages, tools)     │  fill
//! ├────────────────────────────────────────┤
//! │ ❯ prompt editor                        │  3 rows
//! ├────────────────────────────────────────┤
//! │ model  cwd  state · hints              │  1 row
//! └────────────────────────────────────────┘
//! ```
//!
//! Input routing: an open overlay owns keys first (`?` toggles help); then
//! the editor; PgUp/PgDn scroll the chat; Ctrl+C aborts a streaming run or
//! quits when idle (TS: `ctrl+c` interrupt → "abort or exit" prompt).


use crossterm::event::Event;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::Frame;

use crate::action::SubmitBehavior;
use crate::chat::{ChatRole, Messages};
use crate::component::Component;
use crate::editor::{Editor, EditorOutcome};
use crate::events::AppEvent;
use crate::extension_ui::ExtensionUi;
use crate::keys::{KeyCode, KeyEvent};
use crate::overlays::{OpenOverlay, Overlays};
use crate::port::RuntimePort;
use crate::status_line::StatusLine;
use crate::theme::Theme;

/// Root state; single owner, lives on the controller loop.
#[derive(Debug)]
pub struct App {
    messages: Messages,
    editor: Editor,
    status: StatusLine,
    overlays: Overlays,
    extension_ui: ExtensionUi,
    theme: Theme,
    scroll: u16,
}

impl App {
    /// Build the initial app from the run configuration.
    #[must_use]
    pub fn new(model: String, cwd: &std::path::Path) -> Self {
        let dir = cwd
            .file_name()
            .map_or_else(|| cwd.display().to_string(), |n| n.to_string_lossy().into_owned());
        Self {
            messages: Messages::new(),
            editor: Editor::new(),
            status: StatusLine::new(model, dir),
            overlays: Overlays::default(),
            extension_ui: ExtensionUi::new(),
            theme: Theme::default(),
            scroll: 0,
        }
    }

    /// Apply one classified runtime event.
    pub fn apply_event(&mut self, event: &AppEvent) {
        match &event {
            AppEvent::AgentStart | AppEvent::TurnStart => self.status.streaming = true,
            AppEvent::AgentEnd(_) | AppEvent::TurnEnd(_) | AppEvent::SessionSettled => {
                self.status.streaming = false;
            }
            AppEvent::ModelChanged(frame) => {
                if let Some(m) = frame
                    .get("model")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| {
                        let p = frame.get("provider").and_then(serde_json::Value::as_str)?;
                        let id = frame.get("id").and_then(serde_json::Value::as_str)?;
                        Some(format!("{p}/{id}"))
                    })
                {
                    self.status.model = m;
                }
            }
            AppEvent::QueueUpdate {
                steering,
                follow_up,
            } => self.status.queued = steering.len() + follow_up.len(),
            AppEvent::ExtensionUiRequest(frame) => self.extension_ui.push(frame),
            _ => {}
        }
        self.messages.apply(event);
    }

    /// Handle one terminal event. Returns `false` when the app should quit.
    pub fn handle_terminal_event(&mut self, event: &Event, port: &impl RuntimePort) -> bool {
        let Event::Key(key) = event else {
            return true; // resize/paste/mouse: redraw-only for now
        };
        self.handle_key(*key, port).unwrap_or(false)
    }

    /// Handle a key event; `None` means quit.
    fn handle_key(&mut self, key: KeyEvent, port: &impl RuntimePort) -> Option<bool> {
        // Global keys.
        match key.code {
            KeyCode::Char('?') => {
                self.overlays.toggle(OpenOverlay::Help);
                return Some(true);
            }
            KeyCode::Esc if self.overlays.open() != OpenOverlay::None => {
                self.overlays.toggle(self.overlays.open());
                return Some(true);
            }
            KeyCode::PageUp => {
                self.scroll = self.scroll.saturating_add(1);
                return Some(true);
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_sub(1);
                return Some(true);
            }
            _ => {}
        }

        // Overlay focus (help is the only interactive one in the MVP).
        if self.overlays.open() != OpenOverlay::None {
            return Some(true);
        }

        match self.editor.handle_key(key) {
            EditorOutcome::Submit => {
                let text = self.editor.take();
                if !text.trim().is_empty() {
                    self.messages.push_user(&text);
                    let behavior = self
                        .messages
                        .is_streaming()
                        .then_some(SubmitBehavior::Steer);
                    port.submit(None, text, behavior);
                }
                Some(true)
            }
            EditorOutcome::Interrupt => {
                if self.messages.is_streaming() || self.status.streaming {
                    port.abort();
                    Some(true)
                } else {
                    // Idle: quit without aborting — aborting an idle runtime
                    // surfaces a spurious `prompt_result` failure on exit.
                    None
                }
            }
            EditorOutcome::Edited => Some(true),
        }
    }

    /// Render the full screen.
    pub fn render(&mut self, frame: &mut Frame<'_>) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(frame.area());
        self.render_chat(frame, chunks[0]);
        self.editor.render(frame, chunks[1], &self.theme);
        self.status.render(frame, chunks[2], &self.theme);
        self.overlays.render(frame, frame.area(), &self.theme);
    }

    /// The chat scrollback: one line per entry (wrap later with markdown).
    fn render_chat(&self, frame: &mut Frame<'_>, area: ratatui::layout::Rect) {
        use ratatui::style::Style;
        use ratatui::text::{Line, Span};
        use ratatui::widgets::Paragraph;
        let theme = &self.theme;
        let height = area.height as usize;
        let entries = self.messages.entries();
        let skip = entries.len().saturating_sub(height);
        let lines: Vec<Line<'_>> = entries
            .iter()
            .skip(skip)
            .map(|e| {
                let (glyph, color) = match e.role {
                    ChatRole::User => (theme.glyph("prompt"), theme.color("accent")),
                    ChatRole::Assistant => ("", theme.color("text")),
                    ChatRole::Tool => ("●", theme.color("warning")),
                    ChatRole::System => ("!", theme.color("text_muted")),
                };
                let text: String = e.text.lines().collect::<Vec<_>>().join(" ⏎ ");
                if glyph.is_empty() {
                    Line::from(Span::styled(text, Style::default().fg(color)))
                } else {
                    Line::from(vec![
                        Span::styled(format!("{glyph} "), Style::default().fg(color)),
                        Span::raw(text),
                    ])
                }
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port::FakePort;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn app() -> App {
        App::new("zai/glm-5.3-flash".into(), std::path::Path::new("/repo"))
    }

    fn type_key(app: &mut App, port: &FakePort, code: KeyCode, text: &str) {
        for c in text.chars() {
            let key = KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE);
            assert!(app.handle_terminal_event(&Event::Key(key), port));
        }
        let _ = code;
    }

    #[test]
    fn typing_then_enter_submits_and_echoes() {
        let mut app = app();
        let port = FakePort::default();
        type_key(&mut app, &port, KeyCode::Enter, "hi");
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.handle_terminal_event(&Event::Key(enter), &port));
        assert_eq!(app.messages.entries().len(), 1);
        assert_eq!(app.messages.entries()[0].role, ChatRole::User);
        let log = port.log.lock();
        assert_eq!(log.len(), 1);
        assert!(log[0].starts_with("submit("));
    }

    #[test]
    fn ctrl_c_quits_when_idle() {
        let mut app = app();
        let port = FakePort::default();
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(!app.handle_terminal_event(&Event::Key(ctrl_c), &port));
    }

    #[test]
    fn renders_chat_editor_status() {
        let mut app = app();
        let port = FakePort::default();
        type_key(&mut app, &port, KeyCode::Enter, "hello");
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        app.handle_terminal_event(&Event::Key(enter), &port);
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
        let buf = terminal.backend().buffer();
        let bottom: String = (0..80).map(|x| buf[(x, 23)].symbol()).collect();
        assert!(bottom.contains("zai/glm-5.3-flash"));
        let row: String = (0..80).map(|x| buf[(x, 20)].symbol()).collect();
        assert!(row.contains('❯'));
    }
}
