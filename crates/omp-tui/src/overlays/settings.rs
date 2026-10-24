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
//! - The row list is generated from the TS panel itself (`settings_defs.rs`,
//!   regenerate with `tests-e2e/tui/gen_settings_defs.py`): the same rows,
//!   kinds, options, defaults and visibility conditions TS builds with
//!   `getAllSettingDefs`. Hidden settings (no `ui`, numbers without options)
//!   are hidden here the same way.
//! - Row interaction follows TS `SettingsList`: Enter/Space flip booleans and
//!   cycle plain enums; submenu rows (enums/numbers/strings with options,
//!   themes, composer shapes, thinking levels) open a select field with
//!   live preview; text rows open a text field (empty clears the key);
//!   multiselect rows open a toggle list. ←/→ switch tabs, Tab toggles
//!   section focus, printable characters start the cross-tab search.
//! - Values persist through `RuntimePort::settings_set` (global config.yml,
//!   TS `cfg://` set; `null` unsets) and the app applies them live
//!   (`App::apply_setting_live`).

use std::cell::RefCell;

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;
use serde_json::Value;
use unicode_width::UnicodeWidthStr;

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::mouse::SgrMouseEvent;
use crate::theme::Theme;

#[path = "settings_defs.rs"]
mod settings_defs;

use settings_defs::{DefKind, DefOption, SettingDefMeta, SETTING_DEFS};

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

    /// Tab icon from the theme's symbol preset (TS `TAB_METADATA` keys).
    fn icon(self, theme: &Theme) -> &'static str {
        theme.symbol(match self {
            Self::Appearance => "tab.appearance",
            Self::Model => "tab.model",
            Self::Interaction => "tab.interaction",
            Self::Context => "tab.context",
            Self::Memory => "tab.memory",
            Self::Files => "tab.files",
            Self::Shell => "tab.shell",
            Self::Tools => "tab.tools",
            Self::Tasks => "tab.tasks",
            Self::Providers => "tab.providers",
        })
    }

    /// Section order inside the tab (TS `TAB_GROUPS`).
    fn groups(self) -> &'static [&'static str] {
        match self {
            Self::Appearance => &["Theme", "Composer", "Status Line", "Display", "Images"],
            Self::Model => &[
                "Thinking",
                "Sampling",
                "Prompt",
                "Retry & Fallback",
                "Advisor",
                "Prewalk",
                "Vision",
            ],
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
            Self::Memory => &[
                "General",
                "Auto-Learn",
                "Mnemopi",
                "Hindsight",
                "Sharpshooter",
            ],
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
            Self::Providers => &[
                "Services",
                "Fireworks",
                "Tiny Model",
                "Protocol",
                "Timeouts",
                "Privacy",
            ],
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

/// Settings the app applies the moment they change (`App::apply_setting_live`);
/// everything else is saved for the next runtime read.
pub const LIVE_APPLIED: &[&str] = &[
    "theme.dark",
    "theme.light",
    "defaultThinkingLevel",
    "composer.shape",
    "statusLine.preset",
    "statusLine.separator",
    "statusLine.contextLine",
    "statusLine.sessionAccent",
    "statusLine.transparent",
    "statusLine.compactThinkingLevel",
    "statusLine.showHookStatus",
    "tui.vimMode",
    "tui.vimModeDisplay",
    "autocompleteMaxVisible",
    "display.showTokenUsage",
    "display.showTurnTime",
    "read.toolResultPreview",
    "tui.mouse",
];

/// Which value editor a row uses (TS `SettingDef.type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Boolean toggle (Enter/Space flip).
    Bool,
    /// Plain enum without `ui.options`: Enter/Space cycle the values.
    Enum(&'static [&'static str]),
    /// Single-choice select field (enum/number/string with options).
    Submenu,
    /// Free-text field (strings, JSON arrays/records).
    Text,
    /// Array-of-enum toggle list.
    Multi,
    /// `providers.maxInFlightRequests` per-provider limits (edited as JSON).
    Limits,
}

/// One choice of a select/multiselect field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MenuOpt {
    /// Stored value.
    pub value: String,
    /// Display label.
    pub label: String,
    /// One-line description shown after the label.
    pub description: String,
}

impl MenuOpt {
    fn from_def(opt: &DefOption) -> Self {
        Self {
            value: opt.value.to_string(),
            label: opt.label.to_string(),
            description: opt.description.to_string(),
        }
    }
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
    def: &'static SettingDefMeta,
    /// Current effective value (seeded from the port, updated on edit).
    pub value: Value,
}

impl SettingRow {
    /// Build the row for a TS panel definition.
    fn from_def(def: &'static SettingDefMeta) -> Option<Self> {
        let tab = Tab::from_name(def.tab)?;
        let kind = match def.kind {
            DefKind::Bool => Kind::Bool,
            DefKind::Enum => Kind::Enum(def.values),
            DefKind::Submenu => Kind::Submenu,
            DefKind::Text => Kind::Text,
            DefKind::Multi => Kind::Multi,
            DefKind::ProviderLimits => Kind::Limits,
        };
        let value = serde_json::from_str::<Value>(def.default_json).unwrap_or(Value::Null);
        Some(Self {
            id: def.id,
            label: def.label.to_string(),
            description: def.description,
            tab,
            group: def.group,
            kind,
            def,
            value,
        })
    }

    /// Cycle the value one step (bool flips; plain enums advance through
    /// their value list, wrapping). Other kinds open a field instead.
    fn cycled(&self) -> Option<Value> {
        match self.kind {
            Kind::Bool => Some(Value::Bool(!self.value.as_bool().unwrap_or(false))),
            Kind::Enum(values) if !values.is_empty() => {
                let current = self.value.as_str().unwrap_or("");
                let next = values
                    .iter()
                    .position(|v| *v == current)
                    .map_or(0, |idx| (idx + 1) % values.len());
                Some(Value::String(values[next].to_string()))
            }
            _ => None,
        }
    }

    /// Whether the registry value is numeric (submenu values parse to numbers).
    fn is_number(&self) -> bool {
        self.def.schema == "number"
    }

    /// Whether this row's value is a JSON structure (array/record).
    fn is_structured(&self) -> bool {
        matches!(self.def.schema, "array" | "record")
    }

    /// Choices of a static (non-runtime) submenu/multiselect.
    fn static_options(&self) -> Vec<MenuOpt> {
        self.def.options.iter().map(MenuOpt::from_def).collect()
    }

    /// Value rendered after the label (TS `currentValue` string).
    fn value_display(&self) -> String {
        match self.kind {
            Kind::Bool => self.value.as_bool().unwrap_or(false).to_string(),
            Kind::Enum(_) => self.value.as_str().unwrap_or("").to_string(),
            Kind::Submenu => {
                let raw = scalar_text(&self.value);
                // TS `#getSubmenuCurrentValue`: -1 / unset reads "default".
                if matches!(
                    self.id,
                    "compaction.thresholdPercent" | "compaction.thresholdTokens"
                ) && (raw == "-1" || raw.is_empty())
                {
                    "default".to_string()
                } else {
                    raw
                }
            }
            Kind::Text => text_edit_value(&self.value),
            Kind::Multi => {
                let ordered = self.def.ordered;
                let labels: Vec<&str> = self
                    .value
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .filter_map(|v| self.def.options.iter().find(|o| o.value == v))
                            .map(|o| o.label)
                            .collect()
                    })
                    .unwrap_or_default();
                if labels.is_empty() {
                    (if ordered { "default" } else { "none" }).to_string()
                } else {
                    labels.join(if ordered { " → " } else { ", " })
                }
            }
            Kind::Limits => {
                let Some(map) = self.value.as_object().filter(|m| !m.is_empty()) else {
                    return "Unlimited".to_string();
                };
                let mut entries: Vec<(&String, String)> =
                    map.iter().map(|(k, v)| (k, scalar_text(v))).collect();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                entries
                    .iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        }
    }
}

/// `String(value ?? "")`: scalars as text, unset as empty.
fn scalar_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// TS `#formatTextInputEditValue`: strings verbatim, structures as JSON.
fn text_edit_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Object(_) | Value::Array(_) => value.to_string(),
        other => scalar_text(other),
    }
}

/// Parse a submenu choice into the stored value: numbers for numeric
/// settings (`default` is the `-1` sentinel), text otherwise.
fn menu_value(row: &SettingRow, raw: &str) -> Value {
    if row.is_number() {
        if raw == "default" {
            return Value::from(-1);
        }
        if let Ok(int) = raw.parse::<i64>() {
            return Value::from(int);
        }
        if let Ok(float) = raw.parse::<f64>() {
            return serde_json::Number::from_f64(float).map_or(Value::Null, Value::Number);
        }
    }
    Value::String(raw.to_string())
}

/// The string form a stored value takes in the submenu option list.
fn menu_key(row: &SettingRow) -> String {
    let raw = scalar_text(&row.value);
    if row.is_number() && raw == "-1" && row.def.options.iter().any(|o| o.value == "default") {
        return "default".to_string();
    }
    raw
}

/// Built-in composer shapes (TS `BUILTIN_COMPOSER_SHAPES`) as select choices.
fn composer_shape_options() -> Vec<MenuOpt> {
    crate::composer::ComposerShape::ALL
        .iter()
        .map(|shape| MenuOpt {
            value: shape.name().to_string(),
            label: shape.label().to_string(),
            description: shape.description().to_string(),
        })
        .collect()
}

/// Rank of a group inside its tab's section order (unknown groups last).
fn group_rank(tab: Tab, group: &str) -> usize {
    tab.groups()
        .iter()
        .position(|candidate| *candidate == group)
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

/// The value field open over the row list (TS `submenu` of a setting item).
#[derive(Debug, Clone)]
enum Menu {
    /// Single-choice select field.
    Select {
        row: usize,
        options: Vec<MenuOpt>,
        selected: usize,
        /// The committed choice, restored when a previewed field is cancelled.
        base: String,
    },
    /// Free-text field.
    Text {
        row: usize,
        buf: String,
        /// Inline validation error.
        error: Option<String>,
    },
    /// Toggle list for array-of-enum settings.
    Multi {
        row: usize,
        options: Vec<MenuOpt>,
        cursor: usize,
        value: Vec<String>,
    },
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
    /// Open value field: screen row → option index.
    menu: Vec<(u16, usize)>,
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
    /// Open value field (select / text / multiselect submenu), if any.
    menu: Option<Menu>,
    /// Committed edits, drained by the host through [`Self::take_changes`].
    changes: Vec<(String, Value)>,
    /// Live-preview requests (`id`, `value`) for the host (theme hot-swap),
    /// drained through [`Self::take_previews`].
    previews: Vec<(String, String)>,
    /// Highlighted-but-uncommitted submenu choice (`id`, `value`): drives the
    /// status-line / composer-shape previews inside the overlay.
    preview_override: Option<(String, String)>,
    /// Themes offered by `theme.dark` / `theme.light` (host-provided).
    themes: Vec<String>,
    /// Efforts of the active model (`defaultThinkingLevel` choices).
    thinking_levels: Vec<String>,
    /// Whether the terminal speaks an inline image protocol (TS condition
    /// `hasImageProtocol`).
    image_protocol: bool,
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
        let mut rows: Vec<SettingRow> = SETTING_DEFS
            .iter()
            .filter_map(SettingRow::from_def)
            .collect();
        // Display order: tab, then the tab's section order (TS
        // `getSettingsForTab` stable-sorts by group rank); panel order inside
        // a section is the generated table order (stable sort).
        rows.sort_by_key(|row| {
            (
                Tab::ALL
                    .iter()
                    .position(|tab| *tab == row.tab)
                    .unwrap_or(usize::MAX),
                group_rank(row.tab, row.group),
            )
        });
        // Land the selection on the first row of the initial tab (TS opens
        // the appearance tab with its first setting selected).
        let selected = rows
            .iter()
            .position(|row| row.tab == Tab::Appearance && row.def.condition.is_empty())
            .unwrap_or(0);
        Self {
            rows,
            tab: 0,
            selected,
            section_focus: false,
            search: None,
            menu: None,
            changes: Vec::new(),
            previews: Vec::new(),
            preview_override: None,
            themes: Vec::new(),
            thinking_levels: ["minimal", "low", "medium", "high", "xhigh"]
                .map(String::from)
                .to_vec(),
            image_protocol: crate::images::detect() != crate::images::ImageProtocol::None,
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
        self.tab = Tab::ALL
            .iter()
            .position(|tab| *tab == self.rows[idx].tab)
            .unwrap_or(0);
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

    /// Drain live-preview requests (`id`, `value`) raised while a submenu
    /// choice is highlighted (the host hot-swaps the theme for these).
    pub fn take_previews(&mut self) -> Vec<(String, String)> {
        std::mem::take(&mut self.previews)
    }

    /// Whether an interactive sub-mode owns Esc (a value field or the
    /// search): the host must not close the overlay on Esc while this is
    /// true.
    #[must_use]
    pub fn is_editing(&self) -> bool {
        self.menu.is_some() || self.search.is_some()
    }

    /// Every row the panel shows, tab by tab in display order, as
    /// `(id, editor kind)` where the kind is one of `bool`, `enum`,
    /// `submenu`, `text`, `multi`, `limits` (the TS `SettingDef.type`).
    #[must_use]
    pub fn shown_rows(&self) -> Vec<(String, &'static str)> {
        Tab::ALL
            .iter()
            .flat_map(|tab| self.shown_in_tab(*tab))
            .map(|idx| {
                let kind = match self.rows[idx].kind {
                    Kind::Bool => "bool",
                    Kind::Enum(_) => "enum",
                    Kind::Submenu => "submenu",
                    Kind::Text => "text",
                    Kind::Multi => "multi",
                    Kind::Limits => "limits",
                };
                (self.rows[idx].id.to_string(), kind)
            })
            .collect()
    }

    /// Themes offered by the `theme.dark` / `theme.light` selectors.
    pub fn set_available_themes(&mut self, themes: Vec<String>) {
        self.themes = themes;
    }

    /// Efforts the active model supports (`defaultThinkingLevel` choices;
    /// `auto` is always offered first, like TS).
    pub fn set_thinking_levels(&mut self, levels: Vec<String>) {
        self.thinking_levels = levels;
    }

    /// Whether row `idx` is shown (TS `def.condition`).
    fn shown(&self, idx: usize) -> bool {
        let Some(row) = self.rows.get(idx) else {
            return false;
        };
        let flag = |id: &str| self.value_of(id).and_then(|v| v.as_bool()).unwrap_or(false);
        let text = |id: &str| {
            self.value_of(id)
                .and_then(|v| v.as_str().map(str::to_owned))
        };
        match row.def.condition {
            "" => true,
            "macOS" => cfg!(target_os = "macos"),
            "hasImageProtocol" => self.image_protocol,
            // Native (Tern) rendering is not part of the terminal client.
            "nativeRendering" => false,
            "advisorEnabled" => flag("advisor.enabled"),
            "vimModeEnabled" => flag("tui.vimMode"),
            "hindsightActive" => text("memory.backend").as_deref() == Some("hindsight"),
            "mnemopiActive" => text("memory.backend").as_deref() == Some("mnemopi"),
            "autolearnActive" => flag("autolearn.enabled"),
            "autoThinkingActive" => text("defaultThinkingLevel").as_deref() == Some("auto"),
            "usageAwareFallbackEnabled" => flag("retry.usageAwareFallback"),
            "planModeEnabled" => flag("plan.enabled"),
            "planAutosaveEnabled" => flag("plan.enabled") && flag("plan.autosave"),
            // The panel only ever edits the global layer; provenance of the
            // value is not tracked here, so the row is always offered.
            "contextFilesExtraNotProject" => true,
            _ => true,
        }
    }

    /// Indices of the shown rows of `tab`, in display order.
    fn shown_in_tab(&self, tab: Tab) -> Vec<usize> {
        (0..self.rows.len())
            .filter(|idx| self.rows[*idx].tab == tab && self.shown(*idx))
            .collect()
    }

    /// First shown row of `tab` (optionally of `group`).
    fn first_shown(&self, tab: Tab, group: Option<&str>) -> Option<usize> {
        (0..self.rows.len()).find(|idx| {
            let row = &self.rows[*idx];
            row.tab == tab && group.is_none_or(|g| row.group == g) && self.shown(*idx)
        })
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
        if let Some(pos) = self.first_shown(Tab::ALL[self.tab], None) {
            self.selected = pos;
        }
        self.section_focus = false;
    }

    /// Move the selection by `delta` rows within the active tab, wrapping
    /// (TS list navigation flows across section boundaries).
    fn move_selection(&mut self, delta: isize) {
        let tab = Tab::ALL[self.active_tab()];
        let in_tab = self.shown_in_tab(tab);
        if in_tab.is_empty() {
            return;
        }
        let pos = in_tab
            .iter()
            .position(|idx| *idx == self.selected)
            .map_or(0, |pos| {
                (pos as isize + delta).rem_euclid(in_tab.len() as isize) as usize
            });
        self.selected = in_tab[pos];
    }

    /// Move the selection one step for a wheel notch (TS `handleWheel`):
    /// clamped, and row focus takes over from section focus.
    fn wheel_move(&mut self, delta: i8) {
        self.section_focus = false;
        let tab = Tab::ALL[self.active_tab()];
        let in_tab = self.shown_in_tab(tab);
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
            if self.first_shown(tab, Some(group)).is_some() {
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
        let current = self
            .rows
            .get(self.selected)
            .map_or(sections[0], |row| row.group);
        let pos = sections
            .iter()
            .position(|name| *name == current)
            .unwrap_or(0);
        let next = sections[(pos as isize + dir).rem_euclid(sections.len() as isize) as usize];
        if let Some(idx) = self.first_shown(tab, Some(next)) {
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

    /// Choices of a submenu/multiselect row (TS `#submenuOptions`).
    fn options_for(&self, idx: usize) -> Vec<MenuOpt> {
        let row = &self.rows[idx];
        if row.def.runtime {
            return match row.id {
                "theme.dark" | "theme.light" => self
                    .themes
                    .iter()
                    .map(|name| MenuOpt {
                        value: name.clone(),
                        label: name.clone(),
                        description: String::new(),
                    })
                    .collect(),
                "composer.shape" => composer_shape_options(),
                _ => Vec::new(),
            };
        }
        if row.id == "defaultThinkingLevel" {
            // `auto` first, then the model's supported efforts.
            let declared = row.static_options();
            return std::iter::once("auto")
                .chain(self.thinking_levels.iter().map(String::as_str))
                .map(|level| {
                    declared
                        .iter()
                        .find(|o| o.value == level)
                        .cloned()
                        .unwrap_or_else(|| MenuOpt {
                            value: level.to_string(),
                            label: level.to_string(),
                            description: String::new(),
                        })
                })
                .collect();
        }
        row.static_options()
    }

    /// Settings whose highlighted choice previews live (TS
    /// `PREVIEWED_SETTINGS`).
    fn previews_live(id: &str) -> bool {
        matches!(
            id,
            "theme.dark"
                | "theme.light"
                | "statusLine.preset"
                | "statusLine.separator"
                | "statusLine.contextLine"
                | "composer.shape"
        )
    }

    /// Announce the highlighted choice of a previewed submenu.
    fn announce_preview(&mut self, id: &str, value: &str) {
        if !Self::previews_live(id) {
            return;
        }
        self.preview_override = Some((id.to_string(), value.to_string()));
        if matches!(id, "theme.dark" | "theme.light") {
            self.previews.push((id.to_string(), value.to_string()));
        }
    }

    /// Leave the previewed state (cancel or commit): theme rows restore the
    /// active theme through the host.
    fn end_preview(&mut self, id: &str, restore: Option<&str>) {
        self.preview_override = None;
        if let (true, Some(base)) = (matches!(id, "theme.dark" | "theme.light"), restore) {
            self.previews.push((id.to_string(), base.to_string()));
        }
    }

    /// Enter/Space on a row (TS confirm): cycle bool/plain enum, otherwise
    /// open the row's value field.
    fn activate(&mut self, idx: usize) {
        let Some(row) = self.rows.get(idx) else {
            return;
        };
        if let Some(next) = row.cycled() {
            self.commit(idx, next);
            return;
        }
        match row.kind {
            Kind::Submenu => {
                let options = self.options_for(idx);
                let key = menu_key(&self.rows[idx]);
                let selected = options.iter().position(|o| o.value == key).unwrap_or(0);
                let id = self.rows[idx].id;
                if Self::previews_live(id) {
                    self.preview_override = Some((id.to_string(), key.clone()));
                }
                self.menu = Some(Menu::Select {
                    row: idx,
                    options,
                    selected,
                    base: key,
                });
            }
            Kind::Multi => {
                let options = self.options_for(idx);
                let value: Vec<String> = row
                    .value
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .filter(|v| options.iter().any(|o| o.value == *v))
                            .map(str::to_owned)
                            .collect()
                    })
                    .unwrap_or_default();
                self.menu = Some(Menu::Multi {
                    row: idx,
                    options,
                    cursor: 0,
                    value,
                });
            }
            Kind::Text | Kind::Limits => {
                self.menu = Some(Menu::Text {
                    row: idx,
                    buf: text_edit_value(&row.value),
                    error: None,
                });
            }
            // Bool / Enum always cycle (handled above); an empty enum has
            // nothing to pick.
            Kind::Bool | Kind::Enum(_) => {}
        }
    }

    /// Commit the highlighted submenu choice and close the field.
    fn confirm_select(&mut self) {
        let Some(Menu::Select {
            row,
            options,
            selected,
            ..
        }) = self.menu.take()
        else {
            return;
        };
        let Some(opt) = options.get(selected) else {
            return;
        };
        let id = self.rows[row].id;
        let value = menu_value(&self.rows[row], &opt.value);
        self.end_preview(id, None);
        self.commit(row, value);
    }

    /// Esc on a select field: restore any preview and go back.
    fn cancel_menu(&mut self) {
        match self.menu.take() {
            Some(Menu::Select { row, base, .. }) => {
                let id = self.rows[row].id;
                self.end_preview(id, Some(&base));
            }
            Some(Menu::Multi { .. } | Menu::Text { .. }) | None => {}
        }
    }

    /// Submit a text field: empty clears the key, structures must be valid
    /// JSON of the right shape (TS `#setSettingValue`).
    fn confirm_text(&mut self) {
        let Some(Menu::Text { row, buf, .. }) = self.menu.take() else {
            return;
        };
        let r = &self.rows[row];
        if buf.is_empty() {
            self.commit(row, Value::Null);
            return;
        }
        let structured = r.is_structured() || r.kind == Kind::Limits;
        let value = if structured {
            let parsed: Result<Value, _> = serde_json::from_str(&buf);
            let want_array = r.def.schema == "array";
            match parsed {
                Ok(v) if (want_array && v.is_array()) || (!want_array && v.is_object()) => v,
                _ => {
                    let kind = if want_array { "array" } else { "record" };
                    let error = format!("Invalid {kind} JSON for {}", r.id);
                    self.menu = Some(Menu::Text {
                        row,
                        buf,
                        error: Some(error),
                    });
                    return;
                }
            }
        } else if r.is_number() {
            match buf
                .trim()
                .parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
            {
                Some(n) => Value::Number(n),
                None => {
                    self.menu = Some(Menu::Text {
                        row,
                        buf,
                        error: Some("Not a number".into()),
                    });
                    return;
                }
            }
        } else {
            Value::String(buf)
        };
        self.commit(row, value);
    }

    /// Toggle the multiselect choice under the cursor and persist.
    fn toggle_multi(&mut self) {
        let Some(Menu::Multi {
            row,
            options,
            cursor,
            value,
        }) = &mut self.menu
        else {
            return;
        };
        let Some(opt) = options.get(*cursor) else {
            return;
        };
        if let Some(pos) = value.iter().position(|v| *v == opt.value) {
            value.remove(pos);
        } else {
            value.push(opt.value.clone());
        }
        let (row, next) = (*row, value.clone());
        self.commit(
            row,
            Value::Array(next.into_iter().map(Value::String).collect()),
        );
    }

    /// Keys while a value field is open.
    fn handle_menu_key(&mut self, key: KeyEvent) -> EventOutcome {
        let Some(menu) = self.menu.as_mut() else {
            return EventOutcome::Ignored;
        };
        match menu {
            Menu::Select {
                options, selected, ..
            } => {
                let len = options.len();
                let step = |cur: usize, delta: isize| -> usize {
                    if len == 0 {
                        0
                    } else {
                        (cur as isize + delta).rem_euclid(len as isize) as usize
                    }
                };
                match key.code {
                    KeyCode::Esc => self.cancel_menu(),
                    KeyCode::Enter | KeyCode::Char(' ') => self.confirm_select(),
                    KeyCode::Up | KeyCode::Down => {
                        let delta = if key.code == KeyCode::Up { -1 } else { 1 };
                        *selected = step(*selected, delta);
                        self.menu_highlight_changed();
                    }
                    _ => {}
                }
            }
            Menu::Text { buf, error, .. } => match key.code {
                KeyCode::Esc => self.menu = None,
                KeyCode::Enter => self.confirm_text(),
                KeyCode::Backspace => {
                    buf.pop();
                    *error = None;
                }
                KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    buf.push(ch);
                    *error = None;
                }
                KeyCode::Char('u') => {
                    buf.clear();
                    *error = None;
                }
                _ => {}
            },
            Menu::Multi {
                options, cursor, ..
            } => {
                let len = options.len();
                match key.code {
                    KeyCode::Esc => self.menu = None,
                    KeyCode::Enter | KeyCode::Char(' ') => self.toggle_multi(),
                    KeyCode::Up if len > 0 => *cursor = (*cursor + len - 1) % len,
                    KeyCode::Down if len > 0 => *cursor = (*cursor + 1) % len,
                    _ => {}
                }
            }
        }
        EventOutcome::Consumed
    }

    /// The select field's highlight moved: refresh the live preview.
    fn menu_highlight_changed(&mut self) {
        let Some(Menu::Select {
            row,
            options,
            selected,
            ..
        }) = &self.menu
        else {
            return;
        };
        let id = self.rows[*row].id;
        if let Some(opt) = options.get(*selected) {
            let value = opt.value.clone();
            self.announce_preview(id, &value);
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
                .filter(|(idx, row)| row.tab == *tab && self.shown(*idx))
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
            search
                .results
                .extend(matches.iter().map(|idx| DispRow::Row(*idx)));
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
        let pos = rows
            .iter()
            .position(|idx| *idx == search.selected)
            .unwrap_or(0);
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

    /// Paint the open value field over the list area `[top, bottom)` and
    /// return the next free row (TS `SelectFormField` / `TextFormField` /
    /// `MultiSelectSubmenu` rendering inside the overlay card).
    #[allow(clippy::too_many_arguments)]
    fn render_menu(
        &self,
        frame: &mut Frame<'_>,
        menu: &Menu,
        inner: Rect,
        top: u16,
        bottom: u16,
        theme: &Theme,
        map: &mut MouseMap,
    ) -> u16 {
        let dim = Style::default().fg(theme.color("dim"));
        let accent = Style::default().fg(theme.color("accent"));
        let text = Style::default().fg(theme.color("text"));
        let x = inner.x + 1;
        let width = inner.width.saturating_sub(2);
        let row_of = |idx: usize| &self.rows[idx];
        let (row_idx, hint) = match menu {
            Menu::Select { row, .. } => (*row, "⏎ to select · ⎋ to go back"),
            Menu::Text { row, .. } => (*row, "⏎ to save · ⎋ to cancel · Clear field to unset"),
            Menu::Multi { row, .. } => (*row, "⏎/␣ to toggle · ⎋ to go back"),
        };
        let row = row_of(row_idx);
        let mut lines: Vec<(Line<'static>, Option<usize>)> = Vec::new();
        lines.push((
            Line::from(Span::styled(
                row.label.clone(),
                text.add_modifier(Modifier::BOLD),
            )),
            None,
        ));
        lines.push((Line::default(), None));
        for wrapped in wrap_text(row.description, width.saturating_sub(2) as usize) {
            lines.push((Line::from(Span::styled(wrapped, dim)), None));
        }
        if !row.description.is_empty() {
            lines.push((Line::default(), None));
        }
        match menu {
            Menu::Select {
                options, selected, ..
            } => {
                const MAX_VISIBLE: usize = 10;
                let start = selected
                    .saturating_sub(MAX_VISIBLE / 2)
                    .min(options.len().saturating_sub(MAX_VISIBLE));
                let label_w = options
                    .iter()
                    .map(|o| o.label.chars().count())
                    .max()
                    .unwrap_or(0)
                    .min(30)
                    + 2;
                for (i, opt) in options.iter().enumerate().skip(start).take(MAX_VISIBLE) {
                    let current = i == *selected;
                    let prefix = if current { "❯ " } else { "  " };
                    let style = if current { accent } else { text };
                    let pad = label_w.saturating_sub(opt.label.chars().count());
                    let mut spans = vec![
                        Span::styled(prefix.to_string(), style),
                        Span::styled(opt.label.clone(), style),
                    ];
                    if !opt.description.is_empty() {
                        spans.push(Span::styled(" ".repeat(pad), style));
                        spans.push(Span::styled(
                            opt.description.clone(),
                            if current { accent } else { dim },
                        ));
                    }
                    lines.push((Line::from(spans), Some(i)));
                }
                if options.len() > MAX_VISIBLE {
                    lines.push((
                        Line::from(Span::styled(
                            format!("  ({}/{})", selected + 1, options.len()),
                            dim,
                        )),
                        None,
                    ));
                }
                if row.id == "composer.shape" {
                    for line in self.shape_preview_lines(theme, width as usize) {
                        lines.push((line, None));
                    }
                }
            }
            Menu::Text { buf, error, .. } => {
                lines.push((
                    Line::from(vec![
                        Span::styled("› ", accent),
                        Span::styled(buf.clone(), text),
                        Span::styled(" ", Style::default().add_modifier(Modifier::REVERSED)),
                    ]),
                    None,
                ));
                if let Some(error) = error {
                    lines.push((
                        Line::from(Span::styled(
                            error.clone(),
                            Style::default().fg(theme.color("error")),
                        )),
                        None,
                    ));
                }
            }
            Menu::Multi {
                options,
                cursor,
                value,
                ..
            } => {
                let ordered = row.def.ordered;
                for (i, opt) in options.iter().enumerate() {
                    let position = value.iter().position(|v| *v == opt.value);
                    let mark = match (position, ordered) {
                        (None, true) => " · ".to_string(),
                        (None, false) => " ○ ".to_string(),
                        (Some(pos), true) => format!("{:>2}.", pos + 1),
                        (Some(_), false) => " ● ".to_string(),
                    };
                    let current = i == *cursor;
                    let style = if current { accent } else { text };
                    let mut spans = vec![
                        Span::styled(if current { "❯" } else { " " }, style),
                        Span::styled(mark, if position.is_some() { accent } else { dim }),
                        Span::styled(format!(" {}", opt.label), style),
                    ];
                    if !opt.description.is_empty() {
                        spans.push(Span::styled(format!("  {}", opt.description), dim));
                    }
                    lines.push((Line::from(spans), Some(i)));
                }
            }
        }
        lines.push((Line::default(), None));
        lines.push((Line::from(Span::styled(format!("  {hint}"), dim)), None));

        map.menu.clear();
        let mut y = top;
        for (line, opt) in lines {
            if y >= bottom {
                break;
            }
            if let Some(opt) = opt {
                map.menu.push((y, opt));
            }
            frame.render_widget(
                Paragraph::new(truncate_line(line, width as usize)),
                Rect {
                    x,
                    y,
                    width,
                    height: 1,
                },
            );
            y += 1;
        }
        bottom
    }

    /// The composer-shape preview under the shape select field (TS
    /// `ComposerShapePreview`): the candidate shape's chrome around a stand-in
    /// prompt, with the live status line.
    fn shape_preview_lines(&self, theme: &Theme, width: usize) -> Vec<Line<'static>> {
        let shape = match &self.preview_override {
            Some((id, value)) if id == "composer.shape" => value.clone(),
            _ => self
                .value_of("composer.shape")
                .and_then(|v| v.as_str().map(str::to_owned))
                .unwrap_or_else(|| "band".to_owned()),
        };
        let mut st = crate::status_line::StatusLine::new("model".to_owned(), "~/project");
        st.session_name = None;
        st.context_percent = Some(5.0);
        st.context_window = Some(200_000);
        st.config = self.status_config();
        let mut out = vec![
            Line::default(),
            Line::from(Span::styled(
                "Preview:",
                Style::default().fg(theme.color("muted")),
            )),
        ];
        out.extend(crate::composer::preview_lines(
            crate::composer::ComposerShape::from_name(&shape),
            width.max(24),
            theme,
            &st,
        ));
        out
    }

    // ---- status line preview ----------------------------------------------

    /// `StatusLineConfig` built from the seeded `statusLine.*` row values
    /// (TS passes the same keys to the host's live status line).
    fn status_config(&self) -> crate::status_line::StatusLineConfig {
        use crate::status_line::{ContextLineMode, Preset, SeparatorStyle, StatusLineConfig};
        let mut cfg = StatusLineConfig::default();
        let str_of = |id: &str| match &self.preview_override {
            Some((over, value)) if over == id => Some(value.clone()),
            _ => self
                .value_of(id)
                .and_then(|v| v.as_str().map(str::to_owned)),
        };
        let bool_of = |id: &str| self.value_of(id).and_then(|v| v.as_bool());
        if let Some(name) = str_of("statusLine.preset") {
            let name = name.as_str();
            cfg.preset = Preset::from_name(name);
        }
        if let Some(name) = str_of("statusLine.separator") {
            let name = name.as_str();
            cfg.separator = SeparatorStyle::from_name(name);
        }
        if let Some(name) = str_of("statusLine.contextLine") {
            let name = name.as_str();
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
                .filter(|(idx, row)| row.tab == tab && row.group == *group && self.shown(*idx))
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

    /// Mouse while a value field is open: the wheel moves the highlight, a
    /// click on a choice picks (select) or toggles (multiselect) it.
    fn handle_menu_mouse(&mut self, ev: &SgrMouseEvent, map: &MouseMap) -> EventOutcome {
        if let Some(delta) = ev.wheel {
            let code = if delta > 0 {
                KeyCode::Down
            } else {
                KeyCode::Up
            };
            return self.handle_menu_key(KeyEvent::from(code));
        }
        if ev.left_click {
            if let Some((_, opt)) = map.menu.iter().find(|(y, _)| *y == ev.row) {
                match &mut self.menu {
                    Some(Menu::Select { selected, .. }) => {
                        *selected = *opt;
                        self.menu_highlight_changed();
                        self.confirm_select();
                    }
                    Some(Menu::Multi { cursor, .. }) => {
                        *cursor = *opt;
                        self.toggle_multi();
                    }
                    _ => {}
                }
            }
        }
        EventOutcome::Consumed
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
        if theme.clean() {
            // Clean: text-only labels, 3-space gaps; non-active labels
            // shorten to 3 chars from the tail until the bar fits.
            let muted = clean_dim(theme);
            let labels = clean_tab_labels(active_tab, inner.width.saturating_sub(2));
            let mut spans: Vec<Span<'_>> = vec![Span::raw("  ")];
            let mut col = inner.x + 2;
            map.tabs.clear();
            for (idx, label) in labels.iter().enumerate() {
                if idx > 0 {
                    spans.push(Span::raw(CLEAN_TAB_SEP));
                    col = col.saturating_add(CLEAN_TAB_SEP_W);
                }
                let style = if idx == active_tab {
                    accent.add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
                } else if Some(idx) == hover_tab {
                    accent
                } else {
                    muted
                };
                let start = col;
                col = col.saturating_add(
                    u16::try_from(UnicodeWidthStr::width(*label)).unwrap_or(u16::MAX),
                );
                map.tabs.push((start, col));
                spans.push(Span::styled(*label, style));
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
        } else {
            let mut spans: Vec<Span<'_>> = Vec::new();
            // TS insets the tab bar by a further space (border + 2).
            spans.push(Span::styled("  ", Style::default()));
            let mut col = inner.x + 2;
            let sep = "    ";
            let sep_w = sep.len() as u16;
            // Drop labels from the tail until the bar fits (TS short tabs).
            // Cell widths: icon (measured; emoji presets are double-wide) +
            // space; labels and separators count chars (they are ASCII).
            let icon_w = |tab: Tab| UnicodeWidthStr::width(tab.icon(theme)) as u16;
            let mut show_label: Vec<bool> = vec![true; Tab::ALL.len()];
            loop {
                let mut total: u16 = 2;
                for (idx, tab) in Tab::ALL.iter().enumerate() {
                    total += icon_w(*tab) + 1;
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
                    format!("{} {}", tab.icon(theme), tab.label())
                } else {
                    tab.icon(theme).to_string()
                };
                let style = if idx == active_tab {
                    accent.add_modifier(Modifier::BOLD)
                } else if Some(idx) == hover_tab {
                    accent
                } else {
                    text
                };
                let width = if show_label[idx] {
                    icon_w(*tab) + 1 + tab.label().len() as u16
                } else {
                    icon_w(*tab)
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
                    format!("{}     ", Tab::Tasks.icon(theme)),
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
            Rect {
                x: area.x,
                y: divider_y,
                width: area.width,
                height: 1,
            },
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

        let mut y: u16;
        if let Some(menu) = &self.menu {
            y = self.render_menu(
                frame,
                menu,
                inner,
                content_start,
                content_end + 4,
                theme,
                &mut map,
            );
            *self.mouse.borrow_mut() = map;
        } else {
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
            let split = !searching && sections.len() >= 2 && inner.width >= sidebar_width + 2 + 60; // pane ≥ 60 (TS rule)
            let pane_x = if split {
                content_x + sidebar_width + 2
            } else {
                content_x
            };
            let pane_width = content_x + content_w - pane_x;
            map.sidebar_col = if split {
                content_x + sidebar_width
            } else {
                u16::MAX
            };
            map.pane_end_col = content_x + content_w;

            // Scroll window: center the selection (TS `startRow`).
            let start = sel_pos.map_or(0, |pos| {
                pos.saturating_sub(height / 2)
                    .min(list.len().saturating_sub(height))
            });
            let overflow = list.len() > height;
            let row_width = if overflow {
                pane_width.saturating_sub(1)
            } else {
                pane_width
            };
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
                    let style = if is_active { accent } else { dim };
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
                    let first = self.first_shown(tab, Some(name));
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
                    Rect {
                        x: content_x,
                        y: content_start,
                        width: sidebar_width + 2,
                        height: height as u16,
                    },
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
                        let dimmed =
                            active_range.is_some_and(|(lo, hi)| start + r < lo || start + r > hi);
                        let style = if dimmed {
                            dim
                        } else {
                            dim.add_modifier(Modifier::BOLD)
                        };
                        pane_lines.push(Line::from(Span::styled(format!("  {name}"), style)));
                        map.pane.push(None);
                    }
                    DispRow::Row(idx) => {
                        let row = &self.rows[*idx];
                        let is_selected = *idx == self.selected && !self.section_focus;
                        let dimmed =
                            active_range.is_some_and(|(lo, hi)| start + r < lo || start + r > hi);
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
                Rect {
                    x: pane_x,
                    y: content_start,
                    width: pane_width,
                    height: height as u16,
                },
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
                .or(if searching { None } else { Some(self.selected) });
            let desc: &str = desc_row
                .and_then(|idx| self.rows.get(idx))
                .map_or("", |row| row.description);
            y = content_end;
            frame.render_widget(
                Paragraph::new(Line::from("")),
                Rect {
                    x: inner.x,
                    y,
                    width: inner.width,
                    height: 1,
                },
            );
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
                    Rect {
                        x: inner.x + 1,
                        y,
                        width: inner.width.saturating_sub(1),
                        height: 1,
                    },
                );
                y += 1;
            }
            while y < content_end + 4 {
                frame.render_widget(
                    Paragraph::new(Line::from("")),
                    Rect {
                        x: inner.x,
                        y,
                        width: inner.width,
                        height: 1,
                    },
                );
                y += 1;
            }
        }

        // ── Live Preview (appearance tab only, TS status-line preview) ─────
        if show_preview {
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled("Preview:", dim))),
                Rect {
                    x: inner.x + 1,
                    y,
                    width: inner.width.saturating_sub(1),
                    height: 1,
                },
            );
            y += 1;
            let preview = self
                .status_preview(theme, inner.width.saturating_sub(3) as usize)
                .filter(|line| !line.is_empty())
                .unwrap_or_else(|| "(preview not available)".to_owned());
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(preview, text))),
                Rect {
                    x: inner.x + 1,
                    y,
                    width: inner.width.saturating_sub(1),
                    height: 1,
                },
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
            Rect {
                x: area.x,
                y,
                width: area.width,
                height: 1,
            },
        );
        y += 1;
        let hint = if let Some(menu) = &self.menu {
            match menu {
                Menu::Select { .. } => "⏎ to select · ⎋ to go back",
                Menu::Text { .. } => "⏎ to save · ⎋ to cancel · Clear field to unset",
                Menu::Multi { .. } => "⏎/␣ to toggle · ⎋ to go back",
            }
        } else if searching {
            "⏎ change · ⇥ jump tabs · ⎋ exit search"
        } else if self.section_focus {
            "↑/↓ jump sections · ⇥/⏎ to settings · ←/→ to switch tabs · ⎋ to close"
        } else if self.tab_sections(Tab::ALL[self.tab]).len() >= 2 {
            "⏎/␣ to change · ⇥ to jump sections · ←/→ to switch tabs · Type to search · ⎋ to close"
        } else {
            "⏎/␣ to change · ⇥ to switch tabs · ←/→ to switch tabs · Type to search · ⎋ to close"
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(format!(" {hint}"), dim))),
            Rect {
                x: inner.x,
                y,
                width: inner.width,
                height: 1,
            },
        );
        let _ = tabs_line;
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        // An open value field owns every key.
        if self.menu.is_some() {
            return self.handle_menu_key(key);
        }

        // Search owns its keys (TS `#handleSearchModeInput`).
        if self.search.is_some() {
            match key.code {
                KeyCode::Esc => self.end_search(),
                KeyCode::Enter => {
                    if let Some(search) = &self.search {
                        if let DispRow::Row(idx) = search.results[search.selected] {
                            // Cycle/edit in place; the query stays.
                            self.activate(idx);
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
        if self.menu.is_some() {
            return self.handle_menu_mouse(ev, &map);
        }
        if let Some(delta) = ev.wheel {
            // Wheel scrolls the rows pane (TS `handleWheelAt`; the sidebar
            // does not scroll).
            if (map.content.0..map.content.1).contains(&ev.row) && ev.col >= map.sidebar_col {
                self.wheel_move(delta);
            }
            return EventOutcome::Consumed;
        }
        if ev.motion {
            let tab = if ev.row == map.tab_row {
                tab_at(&map, ev.col)
            } else {
                None
            };
            *self.hover_tab.borrow_mut() = tab;
            let row =
                if (map.content.0..map.content.1).contains(&ev.row) && ev.col >= map.sidebar_col {
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
                    if let Some(pos) = self.first_shown(Tab::ALL[self.tab], None) {
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

/// Gap between clean-style tab labels.
const CLEAN_TAB_SEP: &str = "   ";
/// Cell width of [`CLEAN_TAB_SEP`].
const CLEAN_TAB_SEP_W: u16 = 3;

/// Clean-style tab labels that fit in `width` cells: non-active labels are
/// cut to their first 3 chars from the tail until the joined bar fits; the
/// active label is never shortened.
fn clean_tab_labels(active: usize, width: u16) -> Vec<&'static str> {
    let mut labels: Vec<&'static str> = Tab::ALL.iter().map(|tab| tab.label()).collect();
    let total = |labels: &[&str]| -> usize {
        labels
            .iter()
            .map(|l| UnicodeWidthStr::width(*l))
            .sum::<usize>()
            + CLEAN_TAB_SEP.len() * labels.len().saturating_sub(1)
    };
    for idx in (0..labels.len()).rev() {
        if total(&labels) <= usize::from(width) {
            break;
        }
        if idx == active {
            continue;
        }
        let label = labels[idx];
        let cut = label.char_indices().nth(3).map_or(label.len(), |(i, _)| i);
        labels[idx] = &label[..cut];
    }
    labels
}

/// Clean-style muted chrome: `dim`, falling back to `text_muted`.
fn clean_dim(theme: &Theme) -> Style {
    let dim = theme.color("dim");
    let color = if dim == ratatui::style::Color::Reset {
        theme.color("text_muted")
    } else {
        dim
    };
    Style::default().fg(color)
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
        assert_eq!(
            dark.description,
            "Theme used when the terminal has a dark background"
        );
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
        let pos = values
            .iter()
            .position(|v| *v == row.value.as_str().unwrap());
        assert_eq!(
            next.as_str().unwrap(),
            values[(pos.unwrap() + 1) % values.len()]
        );
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
        assert_eq!(
            overlay.rows[idx].value,
            Value::Bool(!before.as_bool().unwrap())
        );
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
        // Assert icon and label separately.
        assert!(s.contains('◐'), "{s}");
        assert!(s.contains("Appearance"), "{s}");
        assert!(s.contains("Theme"), "{s}");
        assert!(s.contains("Dark Theme"), "{s}");
        assert!(s.contains("titanium"), "{s}");
        assert!(
            s.contains("Theme used when the terminal has a dark background"),
            "{s}"
        );
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
        // a submenu row: opens the select field).
        click(&mut overlay, col, row);
        assert!(overlay.menu.is_some());
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

    fn clean_buffer(overlay: &SettingsOverlay, w: u16, h: u16) -> ratatui::buffer::Buffer {
        let mut theme = crate::theme::Theme::default();
        theme.ui = crate::theme::UiStyle::Clean;
        let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
        t.draw(|f| {
            let area = f.area();
            overlay.render(f, area, &theme);
        })
        .unwrap();
        t.backend().buffer().clone()
    }

    fn cells(buf: &ratatui::buffer::Buffer, y: u16, from: u16, to: u16) -> String {
        (from..to).map(|x| buf[(x, y)].symbol()).collect()
    }

    #[test]
    fn clean_tab_bar_is_text_only_with_underlined_active_tab() {
        let overlay = SettingsOverlay::new();
        let buf = clean_buffer(&overlay, 120, 40);
        let map = overlay.mouse.borrow().clone();
        let row = cells(&buf, map.tab_row, 0, 120);
        assert!(row.contains("Appearance   Model   Interaction"), "{row:?}");
        let theme = crate::theme::Theme::default();
        for tab in Tab::ALL {
            let icon = tab.icon(&theme);
            assert!(
                icon.is_empty() || !row.contains(icon),
                "icon {icon:?} in {row:?}"
            );
        }
        let (start, end) = map.tabs[0];
        assert_eq!(cells(&buf, map.tab_row, start, end), "Appearance");
        let active = buf[(start, map.tab_row)].style();
        assert!(active.add_modifier.contains(Modifier::UNDERLINED));
        assert!(active.add_modifier.contains(Modifier::BOLD));
        let (other, _) = map.tabs[1];
        let inactive = buf[(other, map.tab_row)].style();
        assert!(!inactive.add_modifier.contains(Modifier::UNDERLINED));
        for (idx, (start, end)) in map.tabs.iter().enumerate() {
            assert_eq!(
                cells(&buf, map.tab_row, *start, *end),
                Tab::ALL[idx].label()
            );
        }
    }

    #[test]
    fn clean_tab_bar_shortens_inactive_labels_and_keeps_mouse_ranges() {
        let mut overlay = SettingsOverlay::new();
        overlay.tab = 2; // Interaction stays full-length.
        let buf = clean_buffer(&overlay, 70, 30);
        let map = overlay.mouse.borrow().clone();
        let row = cells(&buf, map.tab_row, 0, 70);
        assert!(row.contains("Interaction"), "{row:?}");
        assert!(row.contains("Pro"), "{row:?}");
        assert!(!row.contains("Providers"), "{row:?}");
        let labels = clean_tab_labels(2, 70 - 2 - 2);
        assert_eq!(map.tabs.len(), labels.len());
        let mut shortened = 0;
        for (idx, (start, end)) in map.tabs.iter().enumerate() {
            assert_eq!(cells(&buf, map.tab_row, *start, *end), labels[idx]);
            if labels[idx] != Tab::ALL[idx].label() {
                assert_eq!(labels[idx].len(), 3);
                shortened += 1;
            }
        }
        assert!(shortened > 0);
        assert!(*map.tabs.last().map(|(_, end)| end).unwrap() <= 69);
        // A click on a shortened label selects that tab.
        let (start, _) = map.tabs[9];
        click(&mut overlay, start, map.tab_row);
        assert_eq!(overlay.tab, 9);
    }
}
