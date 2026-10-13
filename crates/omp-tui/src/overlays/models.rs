//! Model picker overlay — port of `pi-tui/src/overlays/model-picker.ts`
//! (plus the P1 slice of `model-hub.ts`: catalogue groups, search,
//! enter-select, esc) and `model-selector.ts` parsing helpers.
//!
//! Layout follows the TS picker: a floating, bottom-anchored panel with
//! `40%` of the terminal height (min list of `MIN_VISIBLE` rows), chrome =
//! top border, status hint, list window, footer hint. Models are grouped by
//! provider in catalogue order; the query filters case-insensitively over
//! `provider/id`; a leading `@` filters the quick-role rows instead (TS
//! `roleMode`). `tab` cycles the thinking level applied on Enter.
//!
//! Actions are queued and drained through the `Component::take_actions`
//! contract — `Action::SetModel` (and `Action::SetThinkingLevel` when a role
//! carries a level or a non-`off` level is armed).

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::action::Action;
use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::theme::Theme;

/// One selectable catalogue row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntry {
    /// Provider slug (`zai`, `openai`, ...).
    pub provider: String,
    /// Model id within the provider.
    pub id: String,
    /// Context window in tokens, when known (shown in the detail row).
    pub context_window: Option<u64>,
}

impl ModelEntry {
    /// Convenience constructor.
    #[must_use]
    pub fn new(provider: &str, id: &str, context_window: Option<u64>) -> Self {
        Self {
            provider: provider.to_owned(),
            id: id.to_owned(),
            context_window,
        }
    }

    /// `provider/id` selector.
    #[must_use]
    pub fn selector(&self) -> String {
        format!("{}/{}", self.provider, self.id)
    }
}

/// A quick-role row (TS `ResolvedRoleModel`): a named cycle entry that
/// resolves to a model + optional thinking level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleEntry {
    /// Role name (`main`, `smol`, `task`, ...).
    pub role: String,
    /// Resolved model.
    pub model: ModelEntry,
    /// Optional thinking level applied with the model.
    pub level: Option<String>,
}

/// Thinking levels offered by `tab`, in cycle order (TS
/// `ConfiguredThinkingLevel`).
pub const THINKING_LEVELS: [&str; 6] = ["off", "minimal", "low", "medium", "high", "xhigh"];

/// Minimum list-window rows on short terminals.
const MIN_VISIBLE: u16 = 4;
/// Fraction of the terminal height the panel occupies.
const HEIGHT_FRACTION: u16 = 40;
/// Width of the panel, percent of the terminal.
const WIDTH_PERCENT: u16 = 70;
/// Chrome rows around the list window (top border, status, footer, bottom).
const CHROME_ROWS: u16 = 4;

/// Status hint per mode (TS `STATUS_HINT` / `QUICK_ROLE_STATUS_HINT`).
#[must_use]
fn status_hint(role_mode: bool) -> &'static str {
    if role_mode {
        "Quick role switch — applies its model and thinking for this session"
    } else {
        "Session-only switch — role models stay unchanged"
    }
}

/// Case-insensitive substring filter (fuzzy scoring is a P2 upgrade shared
/// with the command palette).
#[must_use]
fn matches(query: &str, haystack: &str) -> bool {
    haystack.to_lowercase().contains(&query.to_lowercase())
}

/// Parse `provider/id[:level]` (TS `parseModelString`); `None` when either
/// part is missing.
#[must_use]
pub fn parse_model_string(s: &str) -> Option<(String, String, Option<String>)> {
    let (base, level) = match s.split_once(':') {
        Some((b, l)) if !b.is_empty() && !l.is_empty() => (b, Some(l.to_owned())),
        _ => (s, None),
    };
    let (provider, id) = base.split_once('/')?;
    if provider.is_empty() || id.is_empty() {
        return None;
    }
    Some((provider.to_owned(), id.to_owned(), level))
}

/// The `/model` (alt+p) picker overlay.
#[derive(Debug, Clone)]
pub struct ModelPicker {
    /// Full catalogue, grouped by provider in the given order.
    pub models: Vec<ModelEntry>,
    /// Quick-role rows (TS quick roles), selectable with a leading `@`.
    pub roles: Vec<RoleEntry>,
    /// `provider/id` of the session's active model (preselected on open).
    pub current: Option<String>,
    /// Active thinking level (`off` when unset).
    pub current_level: String,
    /// Query text; a leading `@` enters role mode.
    pub query: String,
    /// Selected row within the filtered list.
    pub selected: usize,
    /// Thinking level armed for the highlighted model (`tab` cycles).
    pub armed_level: String,
    /// Actions produced by Enter, drained by the app.
    pending: Vec<Action>,
}

impl Default for ModelPicker {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            roles: Vec::new(),
            current: None,
            current_level: "off".to_owned(),
            query: String::new(),
            selected: 0,
            armed_level: "off".to_owned(),
            pending: Vec::new(),
        }
    }
}

/// One filtered row: either a catalogue model or a quick role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Row {
    Model(ModelEntry),
    Role(RoleEntry),
}

impl Row {
    #[must_use]
    pub fn selector(&self) -> String {
        match self {
            Self::Model(m) => m.selector(),
            Self::Role(r) => format!("@{}", r.role),
        }
    }

    #[must_use]
    pub fn display(&self) -> String {
        match self {
            Self::Model(m) => m.selector(),
            Self::Role(r) => {
                let target = r.model.selector();
                match &r.level {
                    Some(level) => format!("@{} → {target}:{level}", r.role),
                    None => format!("@{} → {target}", r.role),
                }
            }
        }
    }
}

/// Next entry in `THINKING_LEVELS`, wrapping.
#[must_use]
fn next_level(current: &str) -> String {
    let pos = THINKING_LEVELS.iter().position(|l| *l == current);
    let next = pos.map_or(0, |p| (p + 1) % THINKING_LEVELS.len());
    THINKING_LEVELS[next].to_owned()
}

impl ModelPicker {
    /// Open-state reset: select the current model, clear the query.
    pub fn on_open(&mut self) {
        self.query.clear();
        self.selected = 0;
        self.armed_level = self.current_level.clone();
        if let Some(cur) = &self.current {
            let rows = self.rows();
            if let Some(pos) = rows.iter().position(|r| &r.selector() == cur) {
                self.selected = pos;
            }
        }
    }

    /// True when the query selects quick roles (leading `@`).
    #[must_use]
    pub fn role_mode(&self) -> bool {
        self.query.starts_with('@')
    }

    /// Filtered rows: role rows in role mode, otherwise catalogue models.
    #[must_use]
    pub fn rows(&self) -> Vec<Row> {
        let q = self.query.trim();
        if self.role_mode() {
            let needle = q.trim_start_matches('@');
            return self
                .roles
                .iter()
                .filter(|r| matches(needle, &r.role))
                .cloned()
                .map(Row::Role)
                .collect();
        }
        self.models
            .iter()
            .filter(|m| matches(q, &m.selector()))
            .cloned()
            .map(Row::Model)
            .collect()
    }

    /// Apply the highlighted row: queue `SetModel` (and `SetThinkingLevel`
    /// for roles / an armed non-`off` level).
    pub fn apply(&mut self, row: &Row) {
        match row {
            Row::Model(m) => {
                self.pending.push(Action::SetModel {
                    provider: m.provider.clone(),
                    id: m.id.clone(),
                });
                if self.armed_level != self.current_level && self.armed_level != "off" {
                    self.pending
                        .push(Action::SetThinkingLevel(self.armed_level.clone()));
                }
                self.current = Some(m.selector());
                self.current_level = self.armed_level.clone();
            }
            Row::Role(r) => {
                self.pending.push(Action::SetModel {
                    provider: r.model.provider.clone(),
                    id: r.model.id.clone(),
                });
                if let Some(level) = &r.level {
                    self.pending.push(Action::SetThinkingLevel(level.clone()));
                    self.current_level.clone_from(level);
                }
                self.current = Some(r.model.selector());
            }
        }
    }

    /// Handle one key event; queues `Action`s on Enter. The `Component`
    /// impl delegates here and reports consumption only.
    pub fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        let rows = self.rows();
        match (key.code, key.modifiers) {
            (KeyCode::Char(c), m) if m == KeyModifiers::NONE || m == KeyModifiers::SHIFT => {
                self.query.push(c);
                self.selected = 0;
            }
            (KeyCode::Backspace, _) => {
                self.query.pop();
                self.selected = 0;
            }
            (KeyCode::Up, _) => {
                self.selected = self.selected.saturating_sub(1);
            }
            (KeyCode::Down, _) if self.selected + 1 < rows.len() => self.selected += 1,
            (KeyCode::PageUp, _) => self.selected = self.selected.saturating_sub(5),
            (KeyCode::PageDown, _) => {
                self.selected = (self.selected + 5).min(rows.len().saturating_sub(1));
            }
            (KeyCode::Tab, _) => {
                self.armed_level = next_level(&self.armed_level);
            }
            (KeyCode::Enter, _) => {
                if let Some(row) = rows.get(self.selected) {
                    self.apply(row);
                }
            }
            _ => return EventOutcome::Ignored,
        }
        EventOutcome::Consumed
    }

    /// Bottom-anchored panel rect inside `area` (TS anchor bottom-center).
    fn panel(area: Rect) -> Rect {
        let height = (area.height * HEIGHT_FRACTION / 100).max(MIN_VISIBLE + CHROME_ROWS);
        let width = area.width * WIDTH_PERCENT / 100;
        Rect {
            x: area.x + (area.width.saturating_sub(width)) / 2,
            y: area.bottom().saturating_sub(height),
            width: width.min(area.width),
            height: height.min(area.height),
        }
    }
}

impl Component for ModelPicker {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let panel = Self::panel(area);
        frame.render_widget(Clear, panel);
        let block = Block::default()
            .title(" model ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.color("border")));
        let inner = block.inner(panel);
        frame.render_widget(block, panel);

        let rows = self.rows();
        let list_height = inner.height.saturating_sub(CHROME_ROWS - 1).max(1) as usize;
        // Scroll the window so the selection stays visible.
        let offset = self.selected.saturating_sub(list_height.saturating_sub(1));

        let mut lines = Vec::new();
        // Status hint row.
        lines.push(Line::from(Span::styled(
            status_hint(self.role_mode()),
            Style::default().fg(theme.color("dim")),
        )));
        // Search row (dim placeholder when empty).
        let query = if self.query.is_empty() {
            "type to search…".to_owned()
        } else {
            self.query.clone()
        };
        lines.push(Line::from(Span::styled(
            format!("search: {query}"),
            Style::default().fg(theme.color("accent")),
        )));

        if rows.is_empty() {
            lines.push(Line::from(Span::styled(
                "  no matching models",
                Style::default().fg(theme.color("muted")),
            )));
        } else {
            let mut last_provider: Option<&str> = None;
            for row in rows.iter().skip(offset).take(list_height) {
                let selected = rows.get(self.selected) == Some(row);
                let style = if selected {
                    Style::default().fg(theme.color("accent"))
                } else {
                    Style::default().fg(theme.color("text"))
                };
                // Provider group header (catalogue mode only, TS provider
                // groups).
                if let Row::Model(m) = row {
                    if last_provider != Some(m.provider.as_str()) {
                        last_provider = Some(m.provider.as_str());
                        lines.push(Line::from(Span::styled(
                            m.provider.clone(),
                            Style::default().fg(theme.color("muted")),
                        )));
                    }
                }
                let marker = if selected { "❯ " } else { "  " };
                let mut spans = vec![
                    Span::styled(marker, style),
                    Span::styled(row.display(), style),
                ];
                if selected && !self.role_mode() {
                    spans.push(Span::styled(
                        format!("  [{}]", self.armed_level),
                        Style::default().fg(theme.color("warning")),
                    ));
                }
                lines.push(Line::from(spans));
            }
        }

        // Detail row for the selected model.
        if let Some(Row::Model(m)) = rows.get(self.selected) {
            if let Some(ctx) = m.context_window {
                lines.push(Line::from(Span::styled(
                    format!("context {ctx} tok"),
                    Style::default().fg(theme.color("dim")),
                )));
            }
        }

        // Footer hint.
        lines.push(Line::from(Span::styled(
            "↑/↓ models · ⏎ use · tab thinking · type to search · esc close",
            Style::default().fg(theme.color("dim")),
        )));
        lines.truncate(inner.height as usize);
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        ModelPicker::handle_key(self, key)
    }

    fn take_actions(&mut self) -> Vec<Action> {
        std::mem::take(&mut self.pending)
    }

    fn wants_focus(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn catalog() -> Vec<ModelEntry> {
        vec![
            ModelEntry::new("zai", "glm-5.3-flash", Some(200_000)),
            ModelEntry::new("zai", "glm-5.3", Some(200_000)),
            ModelEntry::new("openai", "gpt-5.2", Some(400_000)),
        ]
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ch(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    #[test]
    fn parse_model_string_handles_level_suffix() {
        assert_eq!(
            parse_model_string("zai/glm-5.3:high"),
            Some(("zai".into(), "glm-5.3".into(), Some("high".into())))
        );
        assert_eq!(
            parse_model_string("zai/glm-5.3"),
            Some(("zai".into(), "glm-5.3".into(), None))
        );
        assert_eq!(parse_model_string("no-slash"), None);
        assert_eq!(parse_model_string("/id"), None);
    }

    #[test]
    fn filters_case_insensitive() {
        let mut p = ModelPicker::default();
        p.models = catalog();
        for c in "GLM-5.3-F".chars() {
            p.handle_key(ch(c));
        }
        let rows = p.rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].display(), "zai/glm-5.3-flash");
    }

    #[test]
    fn role_mode_filters_roles_and_applies_level() {
        let mut p = ModelPicker::default();
        p.roles = vec![RoleEntry {
            role: "smol".into(),
            model: ModelEntry::new("zai", "glm-5.3-flash", None),
            level: Some("low".into()),
        }];
        for c in "@sm".chars() {
            p.handle_key(ch(c));
        }
        assert_eq!(p.rows().len(), 1);
        p.handle_key(key(KeyCode::Enter));
        let actions = p.take_actions();
        assert_eq!(
            actions,
            vec![
                Action::SetModel {
                    provider: "zai".into(),
                    id: "glm-5.3-flash".into()
                },
                Action::SetThinkingLevel("low".into()),
            ]
        );
    }

    #[test]
    fn enter_queues_set_model_and_tabbed_level() {
        let mut p = ModelPicker::default();
        p.models = catalog();
        p.current_level = "off".into();
        p.handle_key(key(KeyCode::Tab)); // off -> minimal
        p.handle_key(key(KeyCode::Enter));
        let actions = p.take_actions();
        assert_eq!(actions.len(), 2);
        assert_eq!(
            actions[0],
            Action::SetModel {
                provider: "zai".into(),
                id: "glm-5.3-flash".into()
            }
        );
        assert_eq!(actions[1], Action::SetThinkingLevel("minimal".into()));
    }

    #[test]
    fn navigation_clamps() {
        let mut p = ModelPicker::default();
        p.models = catalog();
        assert!(p.handle_key(key(KeyCode::Up)).consumed());
        assert_eq!(p.selected, 0);
        p.handle_key(key(KeyCode::Down));
        p.handle_key(key(KeyCode::Down));
        assert_eq!(p.selected, 2);
        p.handle_key(key(KeyCode::Down)); // clamped
        assert_eq!(p.selected, 2);
    }

    #[test]
    fn on_open_preselects_current_model() {
        let mut p = ModelPicker::default();
        p.models = catalog();
        p.current = Some("openai/gpt-5.2".into());
        p.on_open();
        assert_eq!(p.selected, 2);
    }

    #[test]
    fn renders_grouped_panel() {
        let mut p = ModelPicker::default();
        p.models = catalog();
        p.current = Some("zai/glm-5.3-flash".into());
        p.on_open();
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let theme = Theme::default();
        terminal
            .draw(|f| Component::render(&p, f, f.area(), &theme))
            .unwrap();
        let buf = terminal.backend().buffer();
        // Panel is bottom-anchored: find its title row.
        let mut top = None;
        for y in 0..24u16 {
            let row: String = (0..80).map(|x| buf[(x, y)].symbol()).collect();
            if row.contains("┌ model") {
                top = Some(y);
                break;
            }
        }
        let top = top.expect("panel title row");
        let body: Vec<String> = ((top + 1)..24)
            .map(|y| (0..80).map(|x| buf[(x, y)].symbol()).collect())
            .collect();
        let joined = body.join("\n");
        assert!(joined.contains("zai"), "provider group header");
        assert!(joined.contains("zai/glm-5.3-flash"));
        assert!(joined.contains("Session-only switch"));
        assert!(joined.contains("❯"), "selection marker");
    }
}
