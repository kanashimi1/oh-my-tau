//! Settings overlay — port of `pi-tui/src/overlays/settings-selector.ts`
//! (the tabbed, sectioned `/settings` list) and `settings-defs.ts`
//! (`SETTING_TABS` / `TAB_GROUPS` / tab labels).
//!
//! Differences from TS, kept deliberate and documented:
//! - TS curates per-setting display metadata (`ui: {tab, group, label,
//!   description}`) registered across the agent; omp-settings' generated
//!   table carries only id/type/env/default, so this module ports the
//!   tab/group layout and classifies registry settings into it by
//!   id prefix (longest match). Registry entries no prefix explains are
//!   hidden — the same rule TS applies to settings without `ui` metadata.
//!   Labels are derived from the id's last segment (camelCase split).
//! - Secrets (`credential: true`) are omitted rather than mask-rendered.
//! - Row interaction matches the assignment spec: Enter/←→ cycle bool and
//!   enum values, Enter on text/number rows opens an inline editor.
//! - Values persist through `RuntimePort::settings_set` (global config.yml,
//!   TS `cfg://` set); rows whose values omp-rs cannot apply immediately are
//!   marked "no live effect" instead of pretending (TS applies everything
//!   live through the cfg:// host).

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;
use serde_json::Value;

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::theme::Theme;

/// One settings tab (TS `SettingTab`), in `SETTING_TABS` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Appearance,
    Model,
    Interaction,
    Context,
    Memory,
    Files,
    Shell,
    Tools,
    Tasks,
    Providers,
}

impl Tab {
    /// Every tab in display order (TS `SETTING_TABS`).
    pub const ALL: &[Tab] = &[
        Self::Appearance,
        Self::Model,
        Self::Interaction,
        Self::Context,
        Self::Memory,
        Self::Files,
        Self::Shell,
        Self::Tools,
        Self::Tasks,
        Self::Providers,
    ];

    /// Tab label (TS `TAB_METADATA`).
    fn label(self) -> &'static str {
        match self {
            Self::Appearance => "appearance",
            Self::Model => "model",
            Self::Interaction => "interaction",
            Self::Context => "context",
            Self::Memory => "memory",
            Self::Files => "files",
            Self::Shell => "shell",
            Self::Tools => "tools",
            Self::Tasks => "tasks",
            Self::Providers => "providers",
        }
    }

    /// Section order inside the tab (TS `TAB_GROUPS`).
    fn groups(self) -> &'static [&'static str] {
        match self {
            Self::Appearance => &["Theme", "Composer", "Status Line", "Display", "Images"],
            Self::Model => &["Thinking", "Sampling", "Prompt", "Retry & Fallback", "Advisor", "Prewalk", "Vision"],
            Self::Interaction => &[
                "Input",
                "Approvals",
                "Notifications",
                "Speech",
                "Collab",
                "Stream",
                "Magic Keywords",
                "Startup & Updates",
                "Power",
                "Agent",
                "Git",
                "Skills",
            ],
            Self::Context => &["General", "Compaction", "Rules (TTSR)", "Experimental"],
            Self::Memory => &["General", "Auto-Learn", "Mnemopi", "Hindsight", "Sharpshooter"],
            Self::Files => &["Editing", "Reading", "Read Summaries", "LSP"],
            Self::Shell => &["Bash", "Eval & Runtimes"],
            Self::Tools => &[
                "Available Tools",
                "Todos",
                "Grep & Browser",
                "Computer",
                "IDA Pro",
                "GitHub",
                "Output Limits",
                "Execution",
                "Discovery & MCP",
                "Extensions",
                "Developer",
            ],
            Self::Tasks => &["Modes", "Subagents", "Isolation", "Commands & Skills"],
            Self::Providers => &["Services", "Fireworks", "Tiny Model", "Protocol", "Timeouts", "Privacy"],
        }
    }
}

/// Registry-id prefix → (tab, section). Longest prefix wins at lookup; this
/// is the ported half of the TS `ui: {tab, group}` registrations.
const PREFIX_CLASS: &[(&str, Tab, &str)] = &[
    // appearance
    ("theme.", Tab::Appearance, "Theme"),
    ("composer.", Tab::Appearance, "Composer"),
    ("statusLine.", Tab::Appearance, "Status Line"),
    ("tui.", Tab::Appearance, "Display"),
    ("terminal.", Tab::Appearance, "Display"),
    ("images.", Tab::Appearance, "Images"),
    ("colorBlindMode", Tab::Appearance, "Display"),
    ("autocompleteMaxVisible", Tab::Appearance, "Composer"),
    // model
    ("defaultThinkingLevel", Tab::Model, "Thinking"),
    ("thinkingBudgets.", Tab::Model, "Thinking"),
    ("providers.autoThinkingMaxEffort", Tab::Model, "Thinking"),
    ("model.loopGuard", Tab::Model, "Retry & Fallback"),
    ("model.toolCallLoopGuard", Tab::Model, "Retry & Fallback"),
    ("maxTokens", Tab::Model, "Sampling"),
    ("minTokens", Tab::Model, "Sampling"),
    ("temperature", Tab::Model, "Sampling"),
    ("topP", Tab::Model, "Sampling"),
    ("minP", Tab::Model, "Sampling"),
    ("systemPrompt", Tab::Model, "Prompt"),
    ("modelProviderOrder", Tab::Model, "Prompt"),
    ("advisor.", Tab::Model, "Advisor"),
    // interaction
    ("verbose", Tab::Interaction, "Input"),
    ("steering.", Tab::Interaction, "Input"),
    ("tools.approval", Tab::Interaction, "Approvals"),
    ("ask.", Tab::Interaction, "Approvals"),
    ("notifications.", Tab::Interaction, "Notifications"),
    ("voice.", Tab::Interaction, "Speech"),
    ("tts.", Tab::Interaction, "Speech"),
    ("stt.", Tab::Interaction, "Speech"),
    ("collab.", Tab::Interaction, "Collab"),
    ("stream.", Tab::Interaction, "Stream"),
    ("magicKeywords.", Tab::Interaction, "Magic Keywords"),
    ("autoResume", Tab::Interaction, "Startup & Updates"),
    ("checkpoint.", Tab::Interaction, "Startup & Updates"),
    ("update.", Tab::Interaction, "Startup & Updates"),
    ("power.", Tab::Interaction, "Power"),
    ("git.", Tab::Interaction, "Git"),
    ("commit.", Tab::Interaction, "Git"),
    ("worktree.", Tab::Interaction, "Git"),
    // context
    ("compaction.", Tab::Context, "Compaction"),
    ("ttsr.", Tab::Context, "Rules (TTSR)"),
    ("contextWindow.", Tab::Context, "General"),
    ("rules.", Tab::Context, "General"),
    // memory
    ("memories.", Tab::Memory, "General"),
    ("autolearn.", Tab::Memory, "Auto-Learn"),
    ("mnemopi.", Tab::Memory, "Mnemopi"),
    ("hindsight.", Tab::Memory, "Hindsight"),
    ("sharpshooter.", Tab::Memory, "Sharpshooter"),
    // files
    ("edit.", Tab::Files, "Editing"),
    ("astEdit.", Tab::Files, "Editing"),
    ("read.", Tab::Files, "Reading"),
    ("lsp.", Tab::Files, "LSP"),
    // shell
    ("bash.", Tab::Shell, "Bash"),
    ("bashInterceptor.", Tab::Shell, "Bash"),
    ("eval.", Tab::Shell, "Eval & Runtimes"),
    ("exec.", Tab::Shell, "Eval & Runtimes"),
    // tools
    ("tools.", Tab::Tools, "Available Tools"),
    ("todo.", Tab::Tools, "Todos"),
    ("tasks.todo", Tab::Tools, "Todos"),
    ("browser.", Tab::Tools, "Grep & Browser"),
    ("grep.", Tab::Tools, "Grep & Browser"),
    ("webfetch.", Tab::Tools, "Grep & Browser"),
    ("computer.", Tab::Tools, "Computer"),
    ("ida.", Tab::Tools, "IDA Pro"),
    ("github.", Tab::Tools, "GitHub"),
    ("mcp.", Tab::Tools, "Discovery & MCP"),
    ("extensions.", Tab::Tools, "Extensions"),
    ("discover.", Tab::Tools, "Discovery & MCP"),
    // tasks
    ("subagents.", Tab::Tasks, "Subagents"),
    ("task.", Tab::Tasks, "Modes"),
    ("isolation.", Tab::Tasks, "Isolation"),
    ("commands.", Tab::Tasks, "Commands & Skills"),
    ("skills.", Tab::Tasks, "Commands & Skills"),
    // providers
    ("providers.", Tab::Providers, "Services"),
    ("tier.", Tab::Providers, "Services"),
    ("fireworks.", Tab::Providers, "Fireworks"),
    ("tinyModel.", Tab::Providers, "Tiny Model"),
    ("timeouts.", Tab::Providers, "Timeouts"),
    ("auth.", Tab::Providers, "Privacy"),
    ("privacy.", Tab::Providers, "Privacy"),
    ("telemetry.", Tab::Providers, "Privacy"),
];

/// Settings omp-rs applies immediately on edit; everything else is saved but
/// flagged "no live effect" in the row.
pub const LIVE_APPLIED: &[&str] = &["theme.dark", "theme.light", "defaultThinkingLevel"];

/// Which value editor a row uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Bool,
    Enum(&'static [&'static str]),
    Text,
    Number,
}

/// One editable settings row.
#[derive(Debug, Clone)]
pub struct SettingRow {
    /// Registered setting id.
    pub id: &'static str,
    /// Display label (derived from the id's last segment).
    pub label: String,
    tab: Tab,
    group: &'static str,
    kind: Kind,
    /// Current effective value (seeded from the port, updated on edit).
    pub value: Value,
}

impl SettingRow {
    /// Classify a registry setting; `None` when no prefix claims it (TS
    /// hides `ui`-less settings the same way).
    fn classify(def: &omp_settings::SettingDef) -> Option<(Tab, &'static str, Kind)> {
        if def.credential {
            return None;
        }
        let mut best: Option<(&str, Tab, &str)> = None;
        for (prefix, tab, group) in PREFIX_CLASS {
            if def.id.starts_with(prefix)
                && best.is_none_or(|(best_prefix, _, _)| prefix.len() > best_prefix.len())
            {
                best = Some((prefix, *tab, group));
            }
        }
        let (_, tab, group) = best?;
        let kind = match def.ty {
            omp_settings::SettingType::Boolean => Kind::Bool,
            omp_settings::SettingType::Enum(values) => Kind::Enum(values),
            omp_settings::SettingType::String => Kind::Text,
            omp_settings::SettingType::Number => Kind::Number,
            // Arrays and records need structured editors TS builds
            // per-shape; not ported.
            _ => return None,
        };
        Some((tab, group, kind))
    }

    /// Cycle the value one step (bool flips; enums advance through their
    /// value list, wrapping). Text/number rows do not cycle.
    fn cycled(&self) -> Option<Value> {
        match self.kind {
            Kind::Bool => Some(Value::Bool(!self.value.as_bool().unwrap_or(false))),
            Kind::Enum(values) => {
                let current = self.value.as_str().unwrap_or("");
                let next = values
                    .iter()
                    .position(|v| *v == current)
                    .map_or(0, |idx| (idx + 1) % values.len());
                Some(Value::String(values[next].to_string()))
            }
            Kind::Text | Kind::Number => None,
        }
    }

    /// Value rendered after the label.
    fn value_display(&self) -> String {
        match self.kind {
            Kind::Bool => self.value.as_bool().unwrap_or(false).to_string(),
            Kind::Number => match self.value.as_f64() {
                Some(n) => format!("{n}"),
                None => "unset".into(),
            },
            Kind::Enum(_) | Kind::Text => match self.value.as_str() {
                Some(s) if !s.is_empty() => format!("\"{s}\""),
                _ => "unset".into(),
            },
        }
    }
}

/// Derive a label from the last id segment: `defaultThinkingLevel` →
/// "default thinking level".
fn label_for(id: &str) -> String {
    let last = id.rsplit('.').next().unwrap_or(id);
    let mut out = String::with_capacity(last.len() + 8);
    for ch in last.chars() {
        if ch.is_uppercase() && !out.is_empty() {
            out.push(' ');
        }
        out.extend(ch.to_lowercase());
    }
    out
}

/// The `/settings` overlay: a tab strip over a sectioned, scrollable row
/// list, with an inline text editor for string/number values.
#[derive(Debug, Clone)]
pub struct SettingsOverlay {
    /// All rows, registry order (grouped per tab at render time).
    rows: Vec<SettingRow>,
    /// Active tab index into [`Tab::ALL`].
    pub tab: usize,
    /// Selected row index into `rows` (selection is global; the tab filter
    /// only affects what is rendered).
    pub selected: usize,
    /// First rendered row (scroll window).
    scroll: usize,
    /// Inline text editor state: row index being edited + buffer.
    edit: Option<(usize, String)>,
    /// Committed edits, drained by the host through [`Self::take_changes`].
    changes: Vec<(String, Value)>,
}

impl Default for SettingsOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl SettingsOverlay {
    /// Build the row list from the registry (values default to the TS
    /// default; the host seeds live values via [`Self::seed`]).
    #[must_use]
    pub fn new() -> Self {
        let rows = omp_settings::SETTINGS
            .iter()
            .filter_map(|def| {
                let (tab, group, kind) = SettingRow::classify(def)?;
                let value = def
                    .default_yaml
                    .and_then(|raw| serde_yaml_ng::from_str::<serde_yaml_ng::Value>(raw).ok())
                    .map(|v| match v {
                        serde_yaml_ng::Value::Bool(b) => Value::Bool(b),
                        serde_yaml_ng::Value::Number(n) => {
                            n.as_i64().map_or(Value::Null, Value::from)
                        }
                        serde_yaml_ng::Value::String(s) => Value::String(s),
                        _ => Value::Null,
                    })
                    .unwrap_or(Value::Null);
                Some(SettingRow {
                    id: def.id,
                    label: label_for(def.id),
                    tab,
                    group,
                    kind,
                    value,
                })
            })
            .collect();
        Self {
            rows,
            tab: 0,
            selected: 0,
            scroll: 0,
            edit: None,
            changes: Vec::new(),
        }
    }

    /// Rows of the active tab in render order: section headings (per
    /// `TAB_GROUPS` order, ungrouped first) then the group's rows.
    fn visible_rows(&self) -> Vec<VisibleRow<'_>> {
        let tab = Tab::ALL[self.tab];
        let mut out = Vec::new();
        let in_tab: Vec<(usize, &SettingRow)> = self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.tab == tab)
            .collect();
        let mut pushed: Vec<bool> = vec![false; in_tab.len()];
        let order = tab.groups();
        for group in order {
            let members: Vec<usize> = in_tab
                .iter()
                .enumerate()
                .filter(|(_, (_, row))| row.group == *group)
                .map(|(pos, _)| pos)
                .collect();
            if members.is_empty() {
                continue;
            }
            out.push(VisibleRow::Heading(group));
            for pos in members {
                pushed[pos] = true;
                out.push(VisibleRow::Row(in_tab[pos].0, in_tab[pos].1));
            }
        }
        for (pos, _) in in_tab.iter().enumerate() {
            if !pushed[pos] {
                out.insert(0, VisibleRow::Row(in_tab[pos].0, in_tab[pos].1));
            }
        }
        out
    }

    /// Replace row values with the port's effective values (`ids` and
    /// `values` must align; missing ids leave the default in place).
    pub fn seed(&mut self, values: &[(String, Value)]) {
        for (id, value) in values {
            if let Some(row) = self.rows.iter_mut().find(|row| row.id == id.as_str()) {
                if !value.is_null() {
                    row.value = value.clone();
                }
            }
        }
    }

    /// Move the selection to the row with `id` (also switching to its tab).
    /// Returns `false` when no such row is displayed.
    pub fn select(&mut self, id: &str) -> bool {
        let Some(idx) = self.rows.iter().position(|row| row.id == id) else {
            return false;
        };
        self.tab = Tab::ALL.iter().position(|tab| *tab == self.rows[idx].tab).unwrap_or(0);
        self.selected = idx;
        true
    }

    /// Registry ids of every row (for the port's batch get).
    #[must_use]
    pub fn ids(&self) -> Vec<String> {
        self.rows.iter().map(|row| row.id.to_string()).collect()
    }

    /// The row's current effective value (post-seed/post-edit).
    #[must_use]
    pub fn value_of(&self, id: &str) -> Option<Value> {
        self.rows
            .iter()
            .find(|row| row.id == id)
            .map(|row| row.value.clone())
    }

    /// Drain committed edits as (id, value) pairs for the host to persist.
    pub fn take_changes(&mut self) -> Vec<(String, Value)> {
        std::mem::take(&mut self.changes)
    }

    /// Whether the inline editor is active (the host must not close the
    /// overlay on Esc while it is).
    #[must_use]
    pub fn is_editing(&self) -> bool {
        self.edit.is_some()
    }

    /// The tab switch: `dir` +1/-1 wraps (TS tab bar).
    fn switch_tab(&mut self, dir: isize) {
        let len = Tab::ALL.len() as isize;
        self.tab = ((self.tab as isize + dir).rem_euclid(len)) as usize;
        self.scroll = 0;
        // Land the selection on the first row of the new tab.
        if let Some(pos) = self
            .rows
            .iter()
            .position(|row| row.tab == Tab::ALL[self.tab])
        {
            self.selected = pos;
        }
    }

    /// Move the selection by `delta` rows, skipping rows not on the active
    /// tab (headings are not selectable).
    fn move_selection(&mut self, delta: isize) {
        let tab = Tab::ALL[self.tab];
        let len = self.rows.len() as isize;
        let mut idx = self.selected as isize;
        for _ in 0..len {
            idx = (idx + delta).rem_euclid(len);
            if self.rows[idx as usize].tab == tab {
                self.selected = idx as usize;
                return;
            }
        }
    }

    /// Commit a new value for `idx`: queue the change and update the row.
    fn commit(&mut self, idx: usize, value: Value) {
        if let Some(row) = self.rows.get_mut(idx) {
            row.value = value.clone();
            self.changes.push((row.id.to_string(), value));
        }
    }

    fn panel(area: Rect) -> Rect {
        crate::overlays::centered(area, 80, 80)
    }
}

/// One rendered line of the tab's list: a section heading or a row.
enum VisibleRow<'a> {
    Heading(&'static str),
    Row(usize, &'a SettingRow),
}

impl Component for SettingsOverlay {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let panel = Self::panel(area);
        frame.render_widget(Clear, panel);
        let block = Block::default()
            .title(" settings ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.color("border")));
        let inner = block.inner(panel);
        frame.render_widget(block, panel);

        // Tab strip.
        let tabs: Vec<Span<'_>> = Tab::ALL
            .iter()
            .enumerate()
            .flat_map(|(idx, tab)| {
                let active = idx == self.tab;
                let style = if active {
                    Style::default()
                        .fg(theme.color("accent"))
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme.color("muted"))
                };
                let sep = if idx + 1 == Tab::ALL.len() { "" } else { " │ " };
                vec![
                    Span::styled(tab.label(), style),
                    Span::styled(sep, Style::default().fg(theme.color("dim"))),
                ]
            })
            .collect();
        frame.render_widget(Paragraph::new(Line::from(tabs)), inner);

        // Row window below the tab strip.
        let visible = self.visible_rows();
        let list_area = Rect {
            x: inner.x,
            y: inner.y + 1,
            width: inner.width,
            height: inner.height.saturating_sub(2),
        };
        let height = list_area.height as usize;
        // Scroll the window so the selection stays visible (stateless: the
        // window is derived from the selection every frame).
        let sel_pos = visible
            .iter()
            .position(|vr| matches!(vr, VisibleRow::Row(idx, _) if *idx == self.selected));
        let mut scroll = self.scroll.min(visible.len().saturating_sub(1));
        if let Some(pos) = sel_pos {
            if pos < scroll {
                scroll = pos;
            } else if pos >= scroll + height {
                scroll = pos + 1 - height;
            }
        }

        let mut lines = Vec::new();
        let mut armed_cycle_hint = false;
        for vr in visible.iter().skip(scroll).take(height) {
            match vr {
                VisibleRow::Heading(group) => {
                    lines.push(Line::from(Span::styled(
                        format!("─ {group} "),
                        Style::default()
                            .fg(theme.color("dim"))
                            .add_modifier(Modifier::BOLD),
                    )));
                }
                VisibleRow::Row(idx, row) => {
                    let selected = *idx == self.selected;
                    let editing = self
                        .edit
                        .as_ref()
                        .is_some_and(|(edit_idx, _)| *edit_idx == *idx);
                    let style = if selected {
                        Style::default().fg(theme.color("accent"))
                    } else {
                        Style::default().fg(theme.color("text"))
                    };
                    let marker = if selected { "❯ " } else { "  " };
                    let mut spans = vec![
                        Span::styled(marker, style),
                        Span::styled(format!("{:<34}", row.label), style),
                    ];
                    if editing {
                        let buf = &self.edit.as_ref().map_or("", |(_, buf)| buf.as_str());
                        spans.push(Span::styled(
                            format!("> {buf}_"),
                            Style::default().fg(theme.color("warning")),
                        ));
                    } else {
                        spans.push(Span::styled(
                            row.value_display(),
                            Style::default().fg(theme.color("accent")),
                        ));
                        if selected && matches!(row.kind, Kind::Bool | Kind::Enum(_)) {
                            armed_cycle_hint = true;
                        }
                        if selected && !LIVE_APPLIED.contains(&row.id) {
                            spans.push(Span::styled(
                                "  · no live effect in omp-rs",
                                Style::default().fg(theme.color("dim")),
                            ));
                        }
                    }
                    lines.push(Line::from(spans));
                }
            }
        }
        frame.render_widget(Paragraph::new(lines), list_area);

        // Footer hints.
        let edit_hint = if self.edit.is_some() {
            "⏎ save · esc cancel"
        } else if armed_cycle_hint {
            "⏎/←/→ change · tab/⇧tab tabs · ↑/↓ move · esc close"
        } else {
            "⏎ edit · tab/⇧tab tabs · ↑/↓ move · esc close"
        };
        let footer = Rect {
            x: inner.x,
            y: inner.y + inner.height.saturating_sub(1),
            width: inner.width,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                edit_hint,
                Style::default().fg(theme.color("dim")),
            ))),
            footer,
        );
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        // Inline text editor owns every key while active.
        if self.edit.is_some() {
            let (idx, mut buf) = self.edit.take().expect("checked is_some");
            match key.code {
                KeyCode::Esc => {
                    self.edit = None;
                }
                KeyCode::Enter => {
                    let row = &self.rows[idx];
                    match row.kind {
                        Kind::Number => match buf.trim().parse::<f64>() {
                            Ok(n) => self.commit(idx, Value::from(n)),
                            Err(_) => self.edit = Some((idx, buf)),
                        },
                        _ => self.commit(idx, Value::String(buf)),
                    }
                }
                KeyCode::Backspace => {
                    buf.pop();
                    self.edit = Some((idx, buf));
                }
                KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    buf.push(ch);
                    self.edit = Some((idx, buf));
                }
                _ => self.edit = Some((idx, buf)),
            }
            return EventOutcome::Consumed;
        }

        match key.code {
            // `?` bubbles to the host's global help toggle (help may stack
            // over any overlay, per the TS overlay manager).
            KeyCode::Char('?') => EventOutcome::Ignored,
            KeyCode::Tab => {
                self.switch_tab(1);
                EventOutcome::Consumed
            }
            KeyCode::BackTab => {
                self.switch_tab(-1);
                EventOutcome::Consumed
            }
            KeyCode::Up => {
                self.move_selection(-1);
                EventOutcome::Consumed
            }
            KeyCode::Down => {
                self.move_selection(1);
                EventOutcome::Consumed
            }
            KeyCode::Enter | KeyCode::Left | KeyCode::Right => {
                let Some(row) = self.rows.get(self.selected) else {
                    return EventOutcome::Consumed;
                };
                if let Some(next) = row.cycled() {
                    self.commit(self.selected, next);
                } else if key.code == KeyCode::Enter {
                    let current = match row.kind {
                        Kind::Number => row
                            .value
                            .as_f64()
                            .map_or(String::new(), |n| format!("{n}")),
                        _ => row.value.as_str().unwrap_or_default().to_string(),
                    };
                    self.edit = Some((self.selected, current));
                }
                EventOutcome::Consumed
            }
            _ => EventOutcome::Ignored,
        }
    }

    fn wants_focus(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_cover_every_tab_and_have_values() {
        let overlay = SettingsOverlay::new();
        assert!(!overlay.rows.is_empty());
        for tab in Tab::ALL {
            assert!(
                overlay.rows.iter().any(|row| row.tab == *tab),
                "tab {tab:?} has no settings"
            );
        }
        // Live-applied rows exist and are classified.
        for id in LIVE_APPLIED {
            assert!(
                overlay.rows.iter().any(|row| row.id == *id),
                "live row {id} missing"
            );
        }
    }

    #[test]
    fn bool_row_cycles() {
        let overlay = SettingsOverlay::new();
        let row = overlay
            .rows
            .iter()
            .find(|row| row.kind == Kind::Bool)
            .unwrap()
            .clone();
        let flipped = row.cycled().unwrap();
        assert_eq!(flipped, Value::Bool(!row.value.as_bool().unwrap()));
    }

    #[test]
    fn enum_row_cycles_through_values() {
        let overlay = SettingsOverlay::new();
        let row = overlay
            .rows
            .iter()
            .find(|row| matches!(row.kind, Kind::Enum(_)))
            .unwrap()
            .clone();
        let Kind::Enum(values) = row.kind else {
            panic!("expected enum");
        };
        let next = row.cycled().unwrap();
        let pos = values.iter().position(|v| *v == row.value.as_str().unwrap());
        assert_eq!(next.as_str().unwrap(), values[(pos.unwrap() + 1) % values.len()]);
    }

    #[test]
    fn labels_derive_from_last_segment() {
        assert_eq!(label_for("theme.dark"), "dark");
        assert_eq!(label_for("defaultThinkingLevel"), "default thinking level");
        assert_eq!(label_for("bash.autoBackground.enabled"), "enabled");
    }

    #[test]
    fn commit_queues_change_and_updates_row() {
        let mut overlay = SettingsOverlay::new();
        let idx = overlay
            .rows
            .iter()
            .position(|row| row.kind == Kind::Bool)
            .unwrap();
        let before = overlay.rows[idx].value.clone();
        overlay.commit(idx, Value::Bool(!before.as_bool().unwrap()));
        assert_eq!(overlay.rows[idx].value, Value::Bool(!before.as_bool().unwrap()));
        let changes = overlay.take_changes();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].1.as_bool(), Some(!before.as_bool().unwrap()));
        assert!(overlay.take_changes().is_empty());
    }
}
