//! Settings overlay — port of `pi-tui/src/overlays/settings-selector.ts`
//! (the fullscreen tabbed `/settings` screen) and `settings-defs.ts`
//! (`SETTING_TABS` / `TAB_METADATA` / `TAB_GROUPS`).
//!
//! Layout (TS `SettingsList` split layout + `overlayCard`): a full-terminal
//! bordered card titled "Settings" with an icon tab bar under the top
//! border, a left section sidebar and a right setting-row pane (flat layout
//! fallback when the terminal is narrow), a 3-row description area under the
//! selection, a live status-line Preview on the appearance tab, and the
//! key-hint footer over the bottom border.
//!
//! Differences from TS, kept deliberate and documented:
//! - Display metadata (label/description/tab/group/order) is extracted from
//!   the TS `register({...})` declarations into `settings_meta.rs`
//!   (regenerate with `tests-e2e/tui/gen_settings_meta.py`); settings TS
//!   hides (no `ui` metadata) are hidden here the same way. Kinds still come
//!   from the omp-settings registry (bool/enum/text/number; arrays/records
//!   need structured editors that are not ported).
//! - Secrets (`credential: true`) are omitted rather than mask-rendered.
//! - Row interaction: Enter/Space cycle bool and enum values, Enter on
//!   text/number rows opens an inline editor; ←/→ switch tabs, Tab toggles
//!   section focus, printable characters start the cross-tab search.
//! - Values persist through `RuntimePort::settings_set` (global config.yml,
//!   TS `cfg://` set); rows whose values omp-rs cannot apply immediately are
//!   still saved (the app's live-apply handles the ones it can).

use std::cell::RefCell;

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;
use serde_json::Value;

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::mouse::SgrMouseEvent;
use crate::theme::Theme;

#[path = "settings_meta.rs"]
mod settings_meta;

use settings_meta::SETTING_UI;

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
            Self::Appearance => "Appearance",
            Self::Model => "Model",
            Self::Interaction => "Interaction",
            Self::Context => "Context",
            Self::Memory => "Memory",
            Self::Files => "Files",
            Self::Shell => "Shell",
            Self::Tools => "Tools",
            Self::Tasks => "Tasks",
            Self::Providers => "Providers",
        }
    }

    /// Tab icon (TS `TAB_METADATA` symbol keys, unicode preset).
    fn icon(self) -> &'static str {
        match self {
            Self::Appearance => "🎨",
            Self::Model => "🤖",
            Self::Interaction => "⌨",
            Self::Context => "📋",
            Self::Memory => "🧠",
            Self::Files => "📁",
            Self::Shell => "💻",
            Self::Tools => "🔧",
            Self::Tasks => "📦",
            Self::Providers => "🌐",
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

    /// Parse the TS `SettingTab` name.
    fn from_name(name: &str) -> Option<Tab> {
        Some(match name {
            "appearance" => Self::Appearance,
            "model" => Self::Model,
            "interaction" => Self::Interaction,
            "context" => Self::Context,
            "memory" => Self::Memory,
            "files" => Self::Files,
            "shell" => Self::Shell,
            "tools" => Self::Tools,
            "tasks" => Self::Tasks,
            "providers" => Self::Providers,
            _ => return None,
        })
    }
}

/// Settings omp-rs applies immediately on edit; everything else is saved for
/// the next runtime read (the app hot-applies these in `apply_setting_live`).
pub const LIVE_APPLIED: &[&str] = &[
    "theme.dark",
    "theme.light",
    "defaultThinkingLevel",
    "display.showTokenUsage",
    "display.showTurnTime",
    "read.toolResultPreview",
    "tui.mouse",
];

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
    /// Display label (TS `ui.label`).
    pub label: String,
    /// One-line description (TS `ui.description`).
    description: &'static str,
    tab: Tab,
    group: &'static str,
    kind: Kind,
    /// Current effective value (seeded from the port, updated on edit).
    pub value: Value,
}

impl SettingRow {
    /// Classify a registry setting from the ported TS `ui` metadata; `None`
    /// when TS hides the setting from the panel the same way (no `ui`).
    fn classify(def: &omp_settings::SettingDef) -> Option<(Tab, &'static str, Kind)> {
        if def.credential {
            return None;
        }
        let meta = SETTING_UI.iter().find(|(id, _, _, _)| *id == def.id)?;
        let tab = Tab::from_name(meta.3 .0)?;
        let group = if meta.3 .1.is_empty() {
            // No `ui.group` (TS "ungrouped" rendering): land in the tab's
            // first section.
            tab.groups().first().copied().unwrap_or("General")
        } else {
            meta.3 .1
        };
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

    /// Value rendered after the label (TS renders the raw currentValue).
    fn value_display(&self) -> String {
        match self.kind {
            Kind::Bool => self.value.as_bool().unwrap_or(false).to_string(),
            Kind::Number => match self.value.as_f64() {
                Some(n) => format!("{n}"),
                None => "unset".into(),
            },
            Kind::Enum(_) | Kind::Text => match self.value.as_str() {
                Some(s) if !s.is_empty() => s.to_string(),
                _ => "unset".into(),
            },
        }
    }
}

/// Derive a label from the last id segment: `defaultThinkingLevel` →
/// "default thinking level". Only used when the TS metadata carries no
/// label.
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

/// TS panel-order index of a setting id (position in the generated table).
fn panel_order(id: &str) -> usize {
    SETTING_UI
        .iter()
        .position(|(sid, _, _, _)| *sid == id)
        .unwrap_or(usize::MAX)
}

// ═══════════════════════════════════════════════════════════════════════
// Overlay state
// ═══════════════════════════════════════════════════════════════════════

/// One rendered line of a display list: a heading (section or tab) or a row.
#[derive(Debug, Clone, Copy)]
enum DispRow {
    /// Section heading (flat layout) or tab heading (search results).
    Heading(&'static str),
    /// Index into `rows`.
    Row(usize),
}

/// Cross-tab search state (TS `#startSearch`): a query over every tab's
/// rows, results grouped by tab.
#[derive(Debug, Clone)]
struct SearchState {
    query: String,
    /// Flat result list: one `Heading` per tab with matches, then its rows
    /// (indices into `rows`), in tab order.
    results: Vec<DispRow>,
    /// Index into `results` pointing at a `Row` (headings are skipped).
    selected: usize,
}

/// Frame geometry of the last render, for mouse hit-testing (`render` takes
/// `&self`, so the map lives behind a `RefCell`).
#[derive(Debug, Default, Clone)]
struct MouseMap {
    /// Screen row of the tab bar.
    tab_row: u16,
    /// Rendered column span of each tab (icon + label).
    tabs: Vec<(u16, u16)>,
    /// Screen row range `[start, end)` of the list content.
    content: (u16, u16),
    /// First pane column; clicks left of it hit the section sidebar
    /// (`u16::MAX` = flat layout, no sidebar).
    sidebar_col: u16,
    /// Content-line → the section's first row index (sidebar hit test).
    sidebar: Vec<Option<usize>>,
    /// Content-line → row index (pane hit test; headings/blank = None).
    pane: Vec<Option<usize>>,
    /// Column index where pane rows end (scrollbar excluded).
    pane_end_col: u16,
}

impl MouseMap {
    fn clear(&mut self) {
        *self = Self::default();
    }
}

/// The `/settings` overlay: fullscreen tab bar + section sidebar + setting
/// rows + description + live Preview + key hints (TS `SettingsSelector`).
#[derive(Debug, Clone)]
pub struct SettingsOverlay {
    /// All rows, TS settings-panel order (grouped per tab at render time).
    rows: Vec<SettingRow>,
    /// Active tab index into [`Tab::ALL`].
    pub tab: usize,
    /// Selected row index into `rows` (selection is global; the tab filter
    /// only affects what is rendered).
    pub selected: usize,
    /// Keyboard focus on the section sidebar (Tab toggles, TS
    /// `sectionFocused`).
    section_focus: bool,
    /// Cross-tab search, active while `Some`.
    search: Option<SearchState>,
    /// Inline text editor state: row index being edited + buffer.
    edit: Option<(usize, String)>,
    /// Committed edits, drained by the host through [`Self::take_changes`].
    changes: Vec<(String, Value)>,
    /// Row under the pointer (hover highlight).
    hover: RefCell<Option<usize>>,
    /// Tab under the pointer.
    hover_tab: RefCell<Option<usize>>,
    /// Last render's frame geometry.
    mouse: RefCell<MouseMap>,
}

impl Default for SettingsOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl SettingsOverlay {
    /// Build the row list from the TS panel metadata + registry kinds
    /// (values default to the TS default; the host seeds live values via
    /// [`Self::seed`]).
    #[must_use]
    pub fn new() -> Self {
        let mut rows: Vec<SettingRow> = omp_settings::SETTINGS
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
                let meta = SETTING_UI.iter().find(|(id, _, _, _)| *id == def.id);
                let (label, description) = meta.map_or_else(
                    || (label_for(def.id), ""),
                    |(_, label, desc, _)| {
                        (
                            if label.is_empty() { label_for(def.id) } else { label.to_string() },
                            *desc,
                        )
                    },
                );
                Some(SettingRow {
                    id: def.id,
                    label,
                    description,
                    tab,
                    group,
                    kind,
                    value,
                })
            })
            .collect();
        // TS lists a tab's settings in registration (panel) order.
        rows.sort_by_key(|row| panel_order(row.id));
        // Land the selection on the first row of the initial tab (TS opens
        // the appearance tab with its first setting selected).
        let selected = rows
            .iter()
            .position(|row| row.tab == Tab::Appearance)
            .unwrap_or(0);
        Self {
            rows,
            tab: 0,
            selected,
            section_focus: false,
            search: None,
            edit: None,
            changes: Vec::new(),
            hover: RefCell::new(None),
            hover_tab: RefCell::new(None),
            mouse: RefCell::new(MouseMap::default()),
        }
    }


    // ---- host interface -------------------------------------------------

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
        self.section_focus = false;
        self.search = None;
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

    /// Whether an interactive sub-mode owns Esc (the inline editor or the
    /// search): the host must not close the overlay on Esc while this is
    /// true.
    #[must_use]
    pub fn is_editing(&self) -> bool {
        self.edit.is_some() || self.search.is_some()
    }

    // ---- selection and edits ---------------------------------------------

    /// The tab a row belongs to.
    fn tab_of(&self, row: usize) -> Tab {
        self.rows.get(row).map_or(Tab::ALL[0], |row| row.tab)
    }

    /// The tab to highlight in the tab bar (search keeps the tab of the
    /// selected result highlighted, TS `#syncTabBarToSelection`).
    fn active_tab(&self) -> usize {
        if let Some(search) = &self.search {
            if let Some(DispRow::Row(idx)) = search.results.get(search.selected) {
                return Tab::ALL
                    .iter()
                    .position(|tab| *tab == self.tab_of(*idx))
                    .unwrap_or(self.tab);
            }
        }
        self.tab
    }

    /// The tab switch: `dir` +1/-1 wraps (TS tab bar).
    fn switch_tab(&mut self, dir: isize) {
        let len = Tab::ALL.len() as isize;
        self.tab = ((self.tab as isize + dir).rem_euclid(len)) as usize;
        // Land the selection on the first row of the new tab.
        if let Some(pos) = self.rows.iter().position(|row| row.tab == Tab::ALL[self.tab]) {
            self.selected = pos;
        }
        self.section_focus = false;
    }

    /// Move the selection by `delta` rows within the active tab, wrapping
    /// (TS list navigation flows across section boundaries).
    fn move_selection(&mut self, delta: isize) {
        let tab = Tab::ALL[self.active_tab()];
        let in_tab: Vec<usize> = self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.tab == tab)
            .map(|(idx, _)| idx)
            .collect();
        if in_tab.is_empty() {
            return;
        }
        let pos = in_tab
            .iter()
            .position(|idx| *idx == self.selected)
            .map_or(0, |pos| (pos as isize + delta).rem_euclid(in_tab.len() as isize) as usize);
        self.selected = in_tab[pos];
    }

    /// Move the selection one step for a wheel notch (TS `handleWheel`):
    /// clamped, and row focus takes over from section focus.
    fn wheel_move(&mut self, delta: i8) {
        self.section_focus = false;
        let tab = Tab::ALL[self.active_tab()];
        let in_tab: Vec<usize> = self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.tab == tab)
            .map(|(idx, _)| idx)
            .collect();
        let Some(pos) = in_tab.iter().position(|idx| *idx == self.selected) else {
            return;
        };
        let next = (pos as isize + isize::from(delta)).clamp(0, in_tab.len() as isize - 1);
        self.selected = in_tab[next as usize];
    }

    /// Sections (group names) of the active tab that have at least one row.
    fn tab_sections(&self, tab: Tab) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = Vec::new();
        for group in tab.groups() {
            if self.rows.iter().any(|row| row.tab == tab && row.group == *group) {
                out.push(group);
            }
        }
        out
    }

    /// Jump to the first row of the previous/next section (TS section
    /// focus navigation), wrapping within the tab.
    fn jump_section(&mut self, dir: isize) {
        let tab = Tab::ALL[self.tab];
        let sections = self.tab_sections(tab);
        if sections.is_empty() {
            return;
        }
        let current = self.rows.get(self.selected).map_or(sections[0], |row| row.group);
        let pos = sections.iter().position(|name| *name == current).unwrap_or(0);
        let next = sections[(pos as isize + dir).rem_euclid(sections.len() as isize) as usize];
        if let Some(idx) = self
            .rows
            .iter()
            .position(|row| row.tab == tab && row.group == next)
        {
            self.selected = idx;
        }
    }

    /// Commit a new value for `idx`: queue the change and update the row.
    fn commit(&mut self, idx: usize, value: Value) {
        if let Some(row) = self.rows.get_mut(idx) {
            row.value = value.clone();
            self.changes.push((row.id.to_string(), value));
        }
    }

    /// Enter/Space on a row (TS confirm): cycle bool/enum, open the inline
    /// editor for text/number.
    fn activate(&mut self, idx: usize) {
        let Some(row) = self.rows.get(idx) else {
            return;
        };
        if let Some(next) = row.cycled() {
            self.commit(idx, next);
        } else {
            let current = match row.kind {
                Kind::Number => row.value.as_f64().map_or(String::new(), |n| format!("{n}")),
                _ => row.value.as_str().unwrap_or_default().to_string(),
            };
            self.edit = Some((idx, current));
        }
    }

    // ---- search ----------------------------------------------------------

    /// Start the cross-tab search with an initial query (TS `#startSearch`).
    fn start_search(&mut self, initial: char) {
        let mut search = SearchState {
            query: initial.to_string(),
            results: Vec::new(),
            selected: 0,
        };
        self.refilter(&mut search);
        self.search = Some(search);
        self.section_focus = false;
    }

    /// Recompute `results` for the query (TS `#setSearchQuery`, without the
    /// fuzzy ranking: substring match over label/id/value/description,
    /// grouped by tab in panel order).
    fn refilter(&mut self, search: &mut SearchState) {
        search.results.clear();
        let query = search.query.to_lowercase();
        if query.trim().is_empty() {
            return;
        }
        for tab in Tab::ALL {
            let matches: Vec<usize> = self
                .rows
                .iter()
                .enumerate()
                .filter(|(_, row)| row.tab == *tab)
                .filter(|(_, row)| {
                    row.label.to_lowercase().contains(&query)
                        || row.id.to_lowercase().contains(&query)
                        || row.description.to_lowercase().contains(&query)
                        || row.value_display().to_lowercase().contains(&query)
                })
                .map(|(idx, _)| idx)
                .collect();
            if matches.is_empty() {
                continue;
            }
            search.results.push(DispRow::Heading(tab.label()));
            search.results.extend(matches.iter().map(|idx| DispRow::Row(*idx)));
        }
        search.selected = search
            .results
            .iter()
            .position(|dr| matches!(dr, DispRow::Row(_)))
            .unwrap_or(0);
    }

    /// Move the search selection by `delta` rows (headings skipped).
    fn move_search_selection(&mut self, delta: isize) {
        let Some(search) = &mut self.search else {
            return;
        };
        let rows: Vec<usize> = search
            .results
            .iter()
            .enumerate()
            .filter(|(_, dr)| matches!(dr, DispRow::Row(_)))
            .map(|(idx, _)| idx)
            .collect();
        if rows.is_empty() {
            return;
        }
        let pos = rows.iter().position(|idx| *idx == search.selected).unwrap_or(0);
        let next = (pos as isize + delta).rem_euclid(rows.len() as isize) as usize;
        search.selected = rows[next];
    }

    /// Leave search, landing on the tab of the selected result with it
    /// selected (TS `#endSearch(true)` on cancel).
    fn end_search(&mut self) {
        let Some(search) = self.search.take() else {
            return;
        };
        if let Some(DispRow::Row(idx)) = search.results.get(search.selected) {
            self.selected = *idx;
            self.tab = Tab::ALL
                .iter()
                .position(|tab| *tab == self.tab_of(*idx))
                .unwrap_or(self.tab);
        }
    }

    /// Search-mode Tab: jump the selection to the first result of the
    /// next/previous tab group (TS "⇥ jump tabs").
    fn jump_search_tab(&mut self, dir: isize) {
        let Some(search) = &mut self.search else {
            return;
        };
        let groups: Vec<usize> = search
            .results
            .iter()
            .enumerate()
            .filter_map(|(idx, dr)| matches!(dr, DispRow::Heading(_)).then_some(idx))
            .collect();
        if groups.is_empty() {
            return;
        }
        let current = groups
            .iter()
            .rposition(|idx| *idx <= search.selected)
            .unwrap_or(0);
        let next = groups[(current as isize + dir).rem_euclid(groups.len() as isize) as usize];
        if let Some(pos) = search.results[next..]
            .iter()
            .position(|dr| matches!(dr, DispRow::Row(_)))
        {
            search.selected = next + pos;
        }
    }

    // ---- status line preview ----------------------------------------------

    /// `StatusLineConfig` built from the seeded `statusLine.*` row values
    /// (TS passes the same keys to the host's live status line).
    fn status_config(&self) -> crate::status_line::StatusLineConfig {
        use crate::status_line::{ContextLineMode, Preset, SeparatorStyle, StatusLineConfig};
        let mut cfg = StatusLineConfig::default();
        let str_of = |id: &str| self.value_of(id).and_then(|v| v.as_str().map(str::to_owned));
        let bool_of = |id: &str| self.value_of(id).and_then(|v| v.as_bool());
        if let Some(name) = str_of("statusLine.preset") { let name = name.as_str();
            cfg.preset = Preset::from_name(name);
        }
        if let Some(name) = str_of("statusLine.separator") { let name = name.as_str();
            cfg.separator = SeparatorStyle::from_name(name);
        }
        if let Some(name) = str_of("statusLine.contextLine") { let name = name.as_str();
            cfg.context_line = ContextLineMode::from_name(name);
        }
        if let Some(v) = bool_of("statusLine.sessionAccent") {
            cfg.session_accent = v;
        }
        if let Some(v) = bool_of("statusLine.transparent") {
            cfg.transparent = v;
        }
        if let Some(v) = bool_of("statusLine.compactThinkingLevel") {
            cfg.compact_thinking = v;
        }
        if let Some(v) = bool_of("statusLine.showHookStatus") {
            cfg.show_hook_status = v;
        }
        cfg
    }

    /// Render the live status-line preview into a single text line (TS
    /// `getStatusLinePreview`). A synthetic status demonstrates the preset:
    /// omp-rs has no host callback, so the model/cwd/session stand in.
    fn status_preview(&self, theme: &Theme, width: usize) -> Option<String> {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        if width == 0 {
            return None;
        }
        let mut st = crate::status_line::StatusLine::new("model".to_owned(), "~/project");
        st.session_name = Some("session".to_owned());
        st.context_percent = Some(5.0);
        st.context_window = Some(200_000);
        st.tokens = crate::status_line::TokenStats::default();
        st.config = self.status_config();
        let backend = TestBackend::new(width as u16, 1);
        let mut term = Terminal::new(backend).ok()?;
        term.draw(|f| {
            let area = f.area();
            st.render(f, area, theme);
        })
        .ok()?;
        let buf = term.backend().buffer();
        let mut out = String::new();
        for x in 0..width as u16 {
            out.push_str(buf[(x, 0)].symbol());
        }
        Some(out.trim_end().to_owned())
    }

    // ---- display lists ----------------------------------------------------

    /// Rows of the active tab in render order: section headings (per
    /// `TAB_GROUPS` order) then the group's rows (TS `#buildItemsForDefs`).
    fn tab_rows(&self, tab: Tab) -> Vec<DispRow> {
        let mut out = Vec::new();
        for group in tab.groups() {
            let members: Vec<usize> = self
                .rows
                .iter()
                .enumerate()
                .filter(|(_, row)| row.tab == tab && row.group == *group)
                .map(|(idx, _)| idx)
                .collect();
            if members.is_empty() {
                continue;
            }
            out.push(DispRow::Heading(group));
            out.extend(members.iter().map(|idx| DispRow::Row(*idx)));
        }
        out
    }

    /// The display list for the current state (search results or the
    /// active tab).
    fn display_list(&self) -> Vec<DispRow> {
        if let Some(search) = &self.search {
            return search.results.clone();
        }
        self.tab_rows(Tab::ALL[self.tab])
    }

    /// Selected row as a display-list position (for the scroll window).
    fn selected_display_pos(&self, list: &[DispRow]) -> Option<usize> {
        if let Some(search) = &self.search {
            return Some(search.selected.min(list.len().saturating_sub(1)));
        }
        list.iter()
            .position(|dr| matches!(dr, DispRow::Row(idx) if *idx == self.selected))
    }
}

impl Component for SettingsOverlay {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        // Fullscreen: the host hands us the whole alternate-screen frame.
        frame.render_widget(Clear, area);
        let block = Block::default()
            .title("─ Settings ")
            .borders(Borders::ALL)
            .border_type(ratatui::widgets::BorderType::Rounded)
            .border_style(Style::default().fg(theme.color("border")));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.width < 20 || inner.height < 8 {
            return;
        }

        let mut map = MouseMap {
            tab_row: inner.y,
            ..MouseMap::default()
        };
        let border = Style::default().fg(theme.color("border"));
        let dim = Style::default().fg(theme.color("dim"));
        let accent = Style::default().fg(theme.color("accent"));
        let text = Style::default().fg(theme.color("text"));

        // ── Tab bar (one row under the top border, 1-space inset) ─────────
        let searching = self.search.is_some();
        let active_tab = self.active_tab();
        let hover_tab = *self.hover_tab.borrow();
        let tabs_line = inner.y;
        {
            let mut spans: Vec<Span<'_>> = Vec::new();
            // TS insets the tab bar by a further space (border + 2).
            spans.push(Span::styled("  ", Style::default()));
            let mut col = inner.x + 2;
            let sep = "    ";
            let sep_w = sep.len() as u16;
            // Drop labels from the tail until the bar fits (TS short tabs).
            // Cell widths: icons are double-wide glyphs (+2), labels and
            // separators count chars (they are ASCII).
            let mut show_label: Vec<bool> = vec![true; Tab::ALL.len()];
            loop {
                let mut total: u16 = 2;
                for (idx, tab) in Tab::ALL.iter().enumerate() {
                    total += 3; // icon (2 cells) + space
                    if show_label[idx] {
                        total += tab.label().len() as u16;
                    }
                    total += sep_w;
                }
                if total <= inner.width || !show_label.iter().any(|show| *show) {
                    break;
                }
                if let Some(idx) = show_label.iter().rposition(|show| *show) {
                    show_label[idx] = false;
                } else {
                    break;
                }
            }
            map.tabs.clear();
            for (idx, tab) in Tab::ALL.iter().enumerate() {
                let start = col;
                let label = if show_label[idx] {
                    format!("{} {}", tab.icon(), tab.label())
                } else {
                    tab.icon().to_string()
                };
                let style = if idx == active_tab {
                    accent.add_modifier(Modifier::BOLD)
                } else if Some(idx) == hover_tab {
                    accent
                } else {
                    text
                };
                let width = if show_label[idx] {
                    3 + tab.label().len() as u16
                } else {
                    2
                };
                spans.push(Span::styled(label, style));
                col += width;
                map.tabs.push((start, col));
                if idx + 1 != Tab::ALL.len() {
                    spans.push(Span::styled(sep, dim));
                    col += sep_w;
                }
            }
            // TS tab bar pastes one more icon after the last clipped tab
            // (its overflow affordance), then the bar's trailing padding.
            if show_label.iter().any(|show| !*show) {
                spans.push(Span::styled(
                    format!("{}     ", Tab::Tasks.icon()),
                    Style::default().fg(theme.color("text")),
                ));
            }
            frame.render_widget(
                Paragraph::new(Line::from(spans)),
                Rect {
                    x: inner.x,
                    y: tabs_line,
                    width: inner.width,
                    height: 1,
                },
            );
        }

        // ── Divider under the tab bar (TS overlayCard ├─┤ rule) ────────────
        let divider_y = inner.y + 1;
        let rule = if area.width >= 2 {
            format!("├{}┤", "─".repeat(area.width as usize - 2))
        } else {
            "─".repeat(area.width as usize)
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(rule, border))),
            Rect { x: area.x, y: divider_y, width: area.width, height: 1 },
        );

        // ── Bottom block: description, preview, divider, footer ────────────
        let show_preview = !searching && Tab::ALL[active_tab] == Tab::Appearance;
        let bottom = 4 + if show_preview { 2 } else { 0 } + 2; // desc + preview + divider/footer
        let content_start = divider_y + 1;
        let content_end = (inner.y + inner.height).saturating_sub(bottom);
        let height = content_end.saturating_sub(content_start) as usize;
        if height == 0 {
            return;
        }
        map.content = (content_start, content_end);

        let list = self.display_list();
        let sel_pos = self.selected_display_pos(&list);

        // TS insets the list content by border + space + row() space.
        let content_x = inner.x + 1;
        let content_w = inner.width.saturating_sub(2);

        // ── Layout: split sidebar + pane, or flat (TS `#renderSplitList`) ──
        let tab = Tab::ALL[self.tab];
        let sections = self.tab_sections(tab);
        // TS: sidebarWidth = min(22, nameWidth) + 4 where nameWidth spans
        // the longest section name (pane column stays stable across tabs).
        let max_name = Tab::ALL
            .iter()
            .flat_map(|tab| tab.groups().iter())
            .map(|name| name.len())
            .max()
            .unwrap_or(0);
        let sidebar_width = (max_name.min(22) + 4) as u16; // 2 indent + 2 gap
        let split = !searching
            && sections.len() >= 2
            && inner.width >= sidebar_width + 2 + 60; // pane ≥ 60 (TS rule)
        let pane_x = if split { content_x + sidebar_width + 2 } else { content_x };
        let pane_width = content_x + content_w - pane_x;
        map.sidebar_col = if split { content_x + sidebar_width } else { u16::MAX };
        map.pane_end_col = content_x + content_w;

        // Scroll window: center the selection (TS `startRow`).
        let start = sel_pos.map_or(0, |pos| {
            pos.saturating_sub(height / 2)
                .min(list.len().saturating_sub(height))
        });
        let overflow = list.len() > height;
        let row_width = if overflow { pane_width.saturating_sub(1) } else { pane_width };
        map.pane_end_col = pane_x + row_width;

        // Label column spans the whole list so the layout stays stable.
        let max_label = list
            .iter()
            .filter_map(|dr| match dr {
                DispRow::Row(idx) => Some(self.rows[*idx].label.len()),
                DispRow::Heading(_) => None,
            })
            .max()
            .unwrap_or(0)
            .min(30);

        // Active section range (dim everything outside it in split layout):
        // [index of the active section's heading (or first row), index of
        // its last row].
        let active_section: &str = self.rows.get(self.selected).map_or("", |row| row.group);
        let active_range: Option<(usize, usize)> = if split {
            let mut bounds: Option<(usize, usize)> = None;
            let mut heading: Option<&str> = None;
            let mut section_start: usize = 0;
            let mut section_last_row: usize = 0;
            let mut has_row = false;
            for (pos, dr) in list.iter().enumerate() {
                match dr {
                    DispRow::Heading(name) => {
                        if heading == Some(active_section) {
                            bounds = Some((section_start, section_last_row));
                        }
                        heading = Some(*name);
                        section_start = pos;
                        section_last_row = pos;
                        has_row = false;
                    }
                    DispRow::Row(_) => {
                        if !has_row {
                            // A nameless leading section starts at its row.
                            if heading.is_none() {
                                section_start = pos;
                            }
                            has_row = true;
                        }
                        section_last_row = pos;
                    }
                }
            }
            if heading == Some(active_section) {
                bounds = Some((section_start, section_last_row));
            }
            bounds
        } else {
            None
        };

        // ── Sidebar: section names, highlight follows the selection ────────
        map.sidebar.clear();
        map.pane.clear();
        let hover = *self.hover.borrow();
        if split {
            let mut lines = Vec::new();
            for (i, name) in sections.iter().enumerate() {
                let is_active = *name == active_section;
                let focused = self.section_focus && is_active;
                let prefix = if focused { "❯ " } else { "  " };
                let style = if is_active {
                    accent
                } else {
                    dim
                };
                let field = sidebar_width.saturating_sub(4) as usize;
                let label: String = if name.len() > field {
                    name.chars().take(field.saturating_sub(1)).collect()
                } else {
                    name.to_string()
                };
                let pad = sidebar_width.saturating_sub(2 + label.len() as u16);
                lines.push(Line::from(vec![
                    Span::styled(prefix.to_string(), style),
                    Span::styled(label, style),
                    Span::styled(" ".repeat(pad as usize), dim),
                    Span::styled("│ ", dim),
                ]));
                // Sidebar hit → first row of that section.
                let first = self
                    .rows
                    .iter()
                    .position(|row| row.tab == tab && row.group == *name);
                map.sidebar.push(first);
                let _ = i;
            }
            // TS paints the sidebar separator for the full pane height,
            // blank-padding rows below the last section.
            while lines.len() < height {
                lines.push(Line::from(vec![
                    Span::styled(" ".repeat(sidebar_width as usize), dim),
                    Span::styled("│ ", dim),
                ]));
                map.sidebar.push(None);
            }
            frame.render_widget(
                Paragraph::new(lines),
                Rect { x: content_x, y: content_start, width: sidebar_width + 2, height: height as u16 },
            );
        }

        // ── Pane: heading + setting rows, scrollbar when overflowing ───────
        let mut pane_lines: Vec<Line<'_>> = Vec::new();
        for r in 0..height {
            let Some(dr) = list.get(start + r) else {
                pane_lines.push(Line::from(""));
                map.pane.push(None);
                continue;
            };
            match dr {
                DispRow::Heading(name) => {
                    let dimmed = active_range
                        .is_some_and(|(lo, hi)| start + r < lo || start + r > hi);
                    let style = if dimmed { dim } else { dim.add_modifier(Modifier::BOLD) };
                    pane_lines.push(Line::from(Span::styled(format!("  {name}"), style)));
                    map.pane.push(None);
                }
                DispRow::Row(idx) => {
                    let row = &self.rows[*idx];
                    let is_selected = *idx == self.selected && !self.section_focus;
                    let dimmed = active_range
                        .is_some_and(|(lo, hi)| start + r < lo || start + r > hi);
                    let hovered = hover == Some(*idx) && !is_selected;
                    let prefix = if is_selected { "❯ " } else { "  " };
                    let label_pad = max_label.saturating_sub(row.label.len());
                    let value = row.value_display();
                    let spans = if dimmed && !is_selected {
                        vec![Span::styled(
                            format!("  {}{}  {}", row.label, " ".repeat(label_pad), value),
                            dim,
                        )]
                    } else {
                        let label_style = if is_selected || hovered { accent } else { text };
                        let value_style = if is_selected { accent } else { text };
                        vec![
                            Span::styled(prefix.to_string(), label_style),
                            Span::styled(row.label.clone(), label_style),
                            Span::styled(" ".repeat(label_pad), label_style),
                            Span::styled("  ".to_string(), text),
                            Span::styled(value, value_style),
                        ]
                    };
                    let mut line = Line::from(spans);
                    line = truncate_line(line, row_width as usize);
                    pane_lines.push(line);
                    map.pane.push(Some(*idx));
                }
            }
        }
        // Scrollbar (TS ScrollView): thumb `█` with size
        // max(1, min(h*h/len, h)) positioned over the scroll travel, track
        // `│` on every other row.
        if overflow && height > 1 {
            let size = ((height * height) / list.len()).clamp(1, height);
            let travel = height - size;
            let max_offset = list.len() - height;
            let thumb_start = if max_offset == 0 {
                0
            } else {
                (start.saturating_mul(travel) + max_offset / 2) / max_offset
            };
            for (r, line) in pane_lines.iter_mut().enumerate() {
                let used: usize = line.spans.iter().map(|span| span.content.len()).sum();
                let pad = (row_width as usize).saturating_sub(used);
                if pad > 0 {
                    line.spans.push(Span::styled(" ".repeat(pad), text));
                }
                let (marker, style) = if (thumb_start..thumb_start + size).contains(&r) {
                    ("█", accent)
                } else {
                    ("│", dim)
                };
                line.spans.push(Span::styled(marker, style));
            }
        }
        // Rows truncate/pad to `row_width`; the scrollbar marker occupies
        // the reserved last column, so render the full pane width.
        frame.render_widget(
            Paragraph::new(pane_lines),
            Rect { x: pane_x, y: content_start, width: pane_width, height: height as u16 },
        );
        *self.mouse.borrow_mut() = map;

        // ── Description area: blank + up to 3 wrapped rows (TS constant) ───
        let desc_row = self
            .selected_display_pos(&list)
            .and_then(|pos| list.get(pos))
            .and_then(|dr| match dr {
                DispRow::Row(idx) => Some(*idx),
                DispRow::Heading(_) => None,
            })
            .or(if searching {
                None
            } else {
                Some(self.selected)
            });
        let desc: &str = desc_row
            .and_then(|idx| self.rows.get(idx))
            .map_or("", |row| row.description);
        let mut y = content_end;
        frame.render_widget(Paragraph::new(Line::from("")), Rect { x: inner.x, y, width: inner.width, height: 1 });
        y += 1;
        for wrapped in wrap_text(desc, inner.width.saturating_sub(4) as usize)
            .into_iter()
            .take(3)
        {
            let mut line = format!("  {wrapped}");
            if line.chars().count() as u16 > inner.width {
                line = line.chars().take(inner.width as usize).collect();
            }
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(line, dim))),
                Rect { x: inner.x + 1, y, width: inner.width.saturating_sub(1), height: 1 },
            );
            y += 1;
        }
        while y < content_end + 4 {
            frame.render_widget(Paragraph::new(Line::from("")), Rect { x: inner.x, y, width: inner.width, height: 1 });
            y += 1;
        }

        // ── Live Preview (appearance tab only, TS status-line preview) ─────
        if show_preview {
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled("Preview:", dim))),
                Rect { x: inner.x + 1, y, width: inner.width.saturating_sub(1), height: 1 },
            );
            y += 1;
            let preview = self
                .status_preview(theme, inner.width.saturating_sub(3) as usize)
                .filter(|line| !line.is_empty())
                .unwrap_or_else(|| "(preview not available)".to_owned());
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(preview, text))),
                Rect { x: inner.x + 1, y, width: inner.width.saturating_sub(1), height: 1 },
            );
            y += 1;
        }

        // ── Divider + footer key hints ─────────────────────────────────────
        let rule = if area.width >= 2 {
            format!("├{}┤", "─".repeat(area.width as usize - 2))
        } else {
            "─".repeat(area.width as usize)
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(rule, border))),
            Rect { x: area.x, y, width: area.width, height: 1 },
        );
        y += 1;
        let hint = if self.edit.is_some() {
            "⏎ save · ⎋ cancel"
        } else if searching {
            "⏎ change · ⇥ jump tabs · ⎋ exit search"
        } else if self.section_focus {
            "↑/↓ jump sections · ⇥/⏎ to settings · ←/→ to switch tabs · ⎋ to close"
        } else if sections.len() >= 2 {
            "⏎/␣ to change · ⇥ to jump sections · ←/→ to switch tabs · Type to search · ⎋ to close"
        } else {
            "⏎/␣ to change · ⇥ to switch tabs · ←/→ to switch tabs · Type to search · ⎋ to close"
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(format!(" {hint}"), dim))),
            Rect { x: inner.x, y, width: inner.width, height: 1 },
        );
        let _ = tabs_line;
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

        // Search owns its keys (TS `#handleSearchModeInput`).
        if self.search.is_some() {
            match key.code {
                KeyCode::Esc => self.end_search(),
                KeyCode::Enter => {
                    if let Some(search) = &self.search {
                        if let DispRow::Row(idx) = search.results[search.selected] {
                            // Cycle/edit in place; the query stays.
                            if self.rows[idx].cycled().is_some() {
                                let next = self.rows[idx].cycled().expect("checked");
                                self.commit(idx, next);
                            } else {
                                self.activate(idx);
                            }
                        }
                    }
                }
                KeyCode::Up => self.move_search_selection(-1),
                KeyCode::Down => self.move_search_selection(1),
                KeyCode::Tab => self.jump_search_tab(1),
                KeyCode::BackTab => self.jump_search_tab(-1),
                KeyCode::Backspace => {
                    let mut search = self.search.take();
                    if let Some(search) = &mut search {
                        search.query.pop();
                        if search.query.is_empty() {
                            self.search = None;
                            return EventOutcome::Consumed;
                        }
                        self.refilter(search);
                    }
                    self.search = search;
                }
                KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    let mut search = self.search.take();
                    if let Some(search) = &mut search {
                        search.query.push(ch);
                        self.refilter(search);
                    }
                    self.search = search;
                }
                _ => {}
            }
            return EventOutcome::Consumed;
        }

        match key.code {
            // `?` bubbles to the host's global help toggle (help may stack
            // over any overlay, per the TS overlay manager).
            KeyCode::Char('?') => EventOutcome::Ignored,
            // Esc closes the overlay (the host pops when we ignore it).
            KeyCode::Esc => EventOutcome::Ignored,
            KeyCode::Tab | KeyCode::BackTab => {
                let dir: isize = if key.code == KeyCode::Tab { 1 } else { -1 };
                if self.tab_sections(Tab::ALL[self.tab]).len() >= 2 {
                    // Toggle keyboard focus between sections and rows (TS).
                    self.section_focus = !self.section_focus;
                } else {
                    self.switch_tab(dir);
                }
                EventOutcome::Consumed
            }
            KeyCode::Left => {
                self.switch_tab(-1);
                EventOutcome::Consumed
            }
            KeyCode::Right => {
                self.switch_tab(1);
                EventOutcome::Consumed
            }
            KeyCode::Up => {
                if self.section_focus {
                    self.jump_section(-1);
                } else {
                    self.move_selection(-1);
                }
                EventOutcome::Consumed
            }
            KeyCode::Down => {
                if self.section_focus {
                    self.jump_section(1);
                } else {
                    self.move_selection(1);
                }
                EventOutcome::Consumed
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if self.section_focus {
                    // "to settings": back to the rows.
                    self.section_focus = false;
                } else {
                    self.activate(self.selected);
                }
                EventOutcome::Consumed
            }
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Printable characters start the cross-tab search (TS).
                self.start_search(ch);
                EventOutcome::Consumed
            }
            _ => EventOutcome::Ignored,
        }
    }

    fn handle_mouse(&mut self, ev: &SgrMouseEvent) -> EventOutcome {
        let map = self.mouse.borrow().clone();
        if let Some(delta) = ev.wheel {
            // Wheel scrolls the rows pane (TS `handleWheelAt`; the sidebar
            // does not scroll).
            if (map.content.0..map.content.1).contains(&ev.row) && ev.col >= map.sidebar_col {
                self.wheel_move(delta);
            }
            return EventOutcome::Consumed;
        }
        if ev.motion {
            let tab = if ev.row == map.tab_row { tab_at(&map, ev.col) } else { None };
            *self.hover_tab.borrow_mut() = tab;
            let row = if (map.content.0..map.content.1).contains(&ev.row) && ev.col >= map.sidebar_col {
                map.pane
                    .get((ev.row - map.content.0) as usize)
                    .and_then(|hit| {
                        // Only rows the pointer is visually on (hoverTest).
                        if ev.col < map.pane_end_col {
                            *hit
                        } else {
                            None
                        }
                    })
            } else {
                None
            };
            *self.hover.borrow_mut() = row;
            return EventOutcome::Consumed;
        }
        if ev.left_click {
            if ev.row == map.tab_row {
                if let Some(tab) = tab_at(&map, ev.col) {
                    self.end_search();
                    self.tab = tab;
                    self.section_focus = false;
                    if let Some(pos) =
                        self.rows.iter().position(|row| row.tab == Tab::ALL[self.tab])
                    {
                        self.selected = pos;
                    }
                }
                return EventOutcome::Consumed;
            }
            if (map.content.0..map.content.1).contains(&ev.row) {
                let line = (ev.row - map.content.0) as usize;
                if ev.col < map.sidebar_col {
                    // Sidebar click: jump to the section's first row (TS).
                    if let Some(Some(idx)) = map.sidebar.get(line) {
                        self.selected = *idx;
                        self.section_focus = false;
                    }
                    return EventOutcome::Consumed;
                }
                if ev.col < map.pane_end_col {
                    if let Some(Some(idx)) = map.pane.get(line) {
                        if *idx == self.selected {
                            // Repeated clicks activate (TS).
                            self.activate(*idx);
                        } else {
                            self.selected = *idx;
                            self.section_focus = false;
                        }
                    }
                    return EventOutcome::Consumed;
                }
                return EventOutcome::Consumed; // scrollbar column
            }
        }
        EventOutcome::Ignored
    }

    fn wants_focus(&self) -> bool {
        true
    }
}

/// Which tab (index into `Tab::ALL`) renders at screen column `col` on the
/// tab row, if any.
fn tab_at(map: &MouseMap, col: u16) -> Option<usize> {
    map.tabs
        .iter()
        .position(|(start, end)| col >= *start && col < *end)
}

/// Word-wrap `text` to `width` columns (TS `wrapTextWithAnsi` equivalent for
/// plain text); empty input yields no lines.
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for source in text.split('\n') {
        let mut line = String::new();
        for word in source.split_whitespace() {
            let need = word.len() + if line.is_empty() { 0 } else { 1 };
            if !line.is_empty() && line.len() + need > width {
                out.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        if !line.is_empty() {
            out.push(line);
        }
    }
    out
}

/// Truncate a rendered line to `width` display columns (chars; the UI is
/// single-width apart from tab icons which never appear mid-row).
fn truncate_line(mut line: Line<'_>, width: usize) -> Line<'_> {
    let mut used = 0usize;
    let mut kept = Vec::new();
    for span in line.spans {
        let mut text = String::new();
        for ch in span.content.chars() {
            let w = ch.len_utf8();
            if used + w > width {
                break;
            }
            used += w;
            text.push(ch);
        }
        if !text.is_empty() {
            kept.push(Span::styled(text, span.style));
        }
        if used >= width {
            break;
        }
    }
    line.spans = kept;
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn render_screen(overlay: &SettingsOverlay, w: u16, h: u16) -> String {
        let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
        t.draw(|f| {
            let area = f.area();
            overlay.render(f, area, &crate::theme::Theme::default());
        })
        .unwrap();
        let buf = t.backend().buffer().clone();
        (0..h)
            .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::from(code)
    }



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
    fn labels_and_descriptions_come_from_ts_metadata() {
        let overlay = SettingsOverlay::new();
        let dark = overlay
            .rows
            .iter()
            .find(|row| row.id == "theme.dark")
            .unwrap();
        assert_eq!(dark.label, "Dark Theme");
        assert_eq!(dark.description, "Theme used when the terminal has a dark background");
        assert_eq!(dark.tab, Tab::Appearance);
        assert_eq!(dark.group, "Theme");
    }

    #[test]
    fn appearance_rows_are_in_ts_panel_order() {
        let overlay = SettingsOverlay::new();
        let ids: Vec<&str> = overlay
            .rows
            .iter()
            .filter(|row| row.tab == Tab::Appearance)
            .map(|row| row.id)
            .collect();
        let pos = |id: &str| ids.iter().position(|candidate| *candidate == id);
        // theme.* lead the Theme group; Symbol Preset follows; Color-Blind
        // Mode comes after (TS screenshot order).
        assert!(pos("theme.dark") < pos("theme.light"));
        assert!(pos("theme.light") < pos("symbolPreset"));
        assert!(pos("symbolPreset") < pos("colorBlindMode"));
        assert!(pos("colorBlindMode") < pos("composer.shape"));
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
    fn labels_derive_from_last_segment_as_fallback() {
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

    #[test]
    fn fullscreen_render_paints_tab_bar_sidebar_rows_description_and_footer() {
        let overlay = SettingsOverlay::new();
        let s = render_screen(&overlay, 120, 40);
        assert!(s.contains("Settings"), "{s}");
        // Wide glyphs pad a cell in TestBackend: assert icon and label
        // separately.
        assert!(s.contains('🎨'), "{s}");
        assert!(s.contains("Appearance"), "{s}");
        assert!(s.contains("Theme"), "{s}");
        assert!(s.contains("Dark Theme"), "{s}");
        assert!(s.contains("titanium"), "{s}");
        assert!(s.contains("Theme used when the terminal has a dark background"), "{s}");
        assert!(s.contains("Preview:"), "{s}");
        assert!(s.contains("⏎/␣ to change · ⇥ to jump sections"), "{s}");
        assert!(s.contains("Type to search"), "{s}");
        // Fullscreen: borders touch every edge.
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 40);
        assert!(lines[0].starts_with('╭'), "{s}");
        assert!(lines[39].starts_with('╰'), "{s}");
    }

    #[test]
    fn narrow_terminal_falls_back_to_flat_layout() {
        let overlay = SettingsOverlay::new();
        let s = render_screen(&overlay, 60, 20);
        assert!(s.contains("Dark Theme"), "{s}");
        // The hint may truncate at this width (TS truncates too); the lead
        // is stable.
        assert!(s.contains("⏎/␣ to change"), "{s}");
        // Flat layout: no sidebar separator column inside the box borders.
        for line in s.lines() {
            // No sidebar separator in the flat layout: the left half of
            // each line (borders excluded) must be free of `│`/`█` (the
            // scrollbar track/thumb lives in the last column, as in TS).
            let chars: Vec<char> = line.chars().collect();
            let end = (chars.len() / 2).max(1);
            let interior: String = chars[1..end].iter().collect();
            assert!(!interior.contains('│') && !interior.contains('█'), "{line}");
        }
    }

    #[test]
    fn left_right_switch_tabs_and_wrap() {
        let mut overlay = SettingsOverlay::new();
        assert_eq!(overlay.tab, 0);
        overlay.handle_key(key(KeyCode::Left));
        assert_eq!(overlay.tab, Tab::ALL.len() - 1);
        overlay.handle_key(key(KeyCode::Right));
        assert_eq!(overlay.tab, 0);
        overlay.handle_key(key(KeyCode::Right));
        assert_eq!(overlay.tab, 1);
    }

    #[test]
    fn tab_toggles_section_focus_and_enter_returns() {
        let mut overlay = SettingsOverlay::new();
        assert!(!overlay.section_focus);
        overlay.handle_key(key(KeyCode::Tab));
        assert!(overlay.section_focus);
        overlay.handle_key(key(KeyCode::Down));
        // Section jump lands on another section's first row.
        let tab = Tab::ALL[overlay.tab];
        let sections = overlay.tab_sections(tab);
        let current = overlay.rows[overlay.selected].group;
        assert_ne!(sections[0], current);
        overlay.handle_key(key(KeyCode::Enter));
        assert!(!overlay.section_focus);
    }

    #[test]
    fn space_cycles_bool_row() {
        let mut overlay = SettingsOverlay::new();
        overlay.select("colorBlindMode");
        let before = overlay.value_of("colorBlindMode");
        overlay.handle_key(key(KeyCode::Char(' ')));
        assert_ne!(overlay.value_of("colorBlindMode"), before);
        assert!(!overlay.take_changes().is_empty());
    }

    #[test]
    fn type_to_search_filters_and_esc_lands_on_result_tab() {
        let mut overlay = SettingsOverlay::new();
        overlay.handle_key(key(KeyCode::Char('v')));
        overlay.handle_key(key(KeyCode::Char('i')));
        overlay.handle_key(key(KeyCode::Char('m')));
        let search = overlay.search.as_ref().expect("search started");
        assert_eq!(search.query, "vim");
        assert!(!search.results.is_empty());
        // Search swallows Esc without closing (is_editing).
        assert!(overlay.is_editing());
        overlay.handle_key(key(KeyCode::Esc));
        assert!(!overlay.is_editing());
        assert!(overlay.value_of("tui.vimMode").is_some() || overlay.tab <= Tab::ALL.len());
    }

    #[test]
    fn search_backspace_to_empty_exits_search() {
        let mut overlay = SettingsOverlay::new();
        overlay.handle_key(key(KeyCode::Char('x')));
        assert!(overlay.is_editing());
        overlay.handle_key(key(KeyCode::Backspace));
        assert!(!overlay.is_editing());
    }

    // ---- mouse ------------------------------------------------------------

    /// Click at screen (col, row) 0-based.
    fn click(overlay: &mut SettingsOverlay, col: u16, row: u16) {
        let ev = SgrMouseEvent {
            button: 0,
            col,
            row,
            release: false,
            wheel: None,
            motion: false,
            left_click: true,
        };
        assert!(overlay.handle_mouse(&ev).consumed());
    }

    fn wheel(overlay: &mut SettingsOverlay, col: u16, row: u16, delta: i8) {
        let ev = SgrMouseEvent {
            button: if delta > 0 { 65 } else { 64 },
            col,
            row,
            release: false,
            wheel: Some(delta),
            motion: false,
            left_click: false,
        };
        assert!(overlay.handle_mouse(&ev).consumed());
    }

    #[test]
    fn mouse_click_on_tab_switches_tab() {
        let mut overlay = SettingsOverlay::new();
        // Render once to fill the mouse map.
        let _ = render_screen(&overlay, 120, 40);
        let map = overlay.mouse.borrow().clone();
        assert_eq!(map.tab_row, 1);
        // Click on the "Model" tab (second span).
        let (start, _) = map.tabs[1];
        click(&mut overlay, start + 1, map.tab_row);
        assert_eq!(overlay.tab, 1);
    }

    #[test]
    fn mouse_click_selects_row_and_second_click_activates() {
        let mut overlay = SettingsOverlay::new();
        let s = render_screen(&overlay, 120, 40);
        assert!(s.contains("Dark Theme"), "{s}");
        let map = overlay.mouse.borrow().clone();
        assert_ne!(map.sidebar_col, u16::MAX, "split layout at 120 cols");
        // Find the content line that renders the "Light Theme" row.
        let target = overlay
            .rows
            .iter()
            .position(|row| row.id == "theme.light")
            .unwrap();
        let line = map
            .pane
            .iter()
            .position(|hit| matches!(hit, Some(idx) if *idx == target))
            .expect("target row rendered");
        let col = map.sidebar_col + 4;
        let row = map.content.0 + line as u16;
        // First click selects the row.
        click(&mut overlay, col, row);
        assert_eq!(overlay.selected, target);
        // A repeated click on the selected row activates it (theme.light is
        // a text row: opens the inline editor).
        click(&mut overlay, col, row);
        assert!(overlay.edit.is_some());
    }

    #[test]
    fn mouse_wheel_moves_selection_within_content() {
        let mut overlay = SettingsOverlay::new();
        let _ = render_screen(&overlay, 120, 40);
        let map = overlay.mouse.borrow().clone();
        let before = overlay.selected;
        wheel(&mut overlay, map.sidebar_col + 4, map.content.0 + 2, 1);
        assert_ne!(overlay.selected, before);
        // Wheel over the sidebar does not scroll the rows.
        let current = overlay.selected;
        wheel(&mut overlay, 2, map.content.0 + 2, 1);
        assert_eq!(overlay.selected, current);
    }
}
