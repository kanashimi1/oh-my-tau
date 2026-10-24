//! Fullscreen model browser — port of TS `ModelHubComponent`
//! (`pi-tui/src/overlays/model-hub.ts`) + `ModelBrowser`
//! (`model-browser.ts`) + `HubFrame` (`hub-frame.ts`).
//!
//! Layout (TS `/model`): a rounded frame titled "Models" split into a
//! sidebar (Roles, All models, configured providers with counts, then every
//! catalogue provider that is not configured) and a body: status row, kind
//! tabs, search row, windowed model list (ctx / price columns, scrollbar),
//! two detail rows, and a footer hint row. Enter on a model opens the role
//! strip (default/smol/slow/…); assigning `default` switches the live model.
//!
//! Intents are queued and drained through `Component::take_actions`:
//! `Action::SetModel` / `Action::SetThinkingLevel` for the live session and
//! `Action::SettingChanged { id: "modelRoles", .. }` for role persistence.
//! Mouse (SGR, 0-based cells == frame coords): click sidebar rows, model
//! rows (click selects, click-again activates), footer chips; wheel scrolls
//! the pane under the pointer.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};
use ratatui::Frame;
use unicode_width::UnicodeWidthStr;

use crate::action::Action;
use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::mouse::SgrMouseEvent;
use crate::theme::Theme;

/// One selectable catalogue row.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelEntry {
    /// Provider slug (`zai`, `openai`, ...).
    pub provider: String,
    /// Model id within the provider.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Context window in tokens, when known.
    pub context_window: Option<u64>,
    /// Max output tokens, when known.
    pub max_tokens: Option<u64>,
    /// $/M input.
    pub cost_input: f64,
    /// $/M output.
    pub cost_output: f64,
    /// $/M cache read.
    pub cost_cache_read: f64,
    /// Model reasons (thinking levels apply).
    pub reasoning: bool,
    /// Model accepts image input.
    pub vision: bool,
    /// Provider is configured (declared in `models.yml` or has credentials).
    pub available: bool,
    /// Catalogue kind (`""` = chat; `tiny`, `image`, `tts`, `stt`, `search`,
    /// `judge`, `embedding`, `rerank`, `video`).
    pub kind: String,
    /// Catalogue `priority` (lower sorts first).
    pub priority: Option<i64>,
}

impl ModelEntry {
    /// Convenience constructor (free, available, text-only).
    #[must_use]
    pub fn new(provider: &str, id: &str, context_window: Option<u64>) -> Self {
        Self {
            provider: provider.to_owned(),
            id: id.to_owned(),
            name: id.to_owned(),
            context_window,
            max_tokens: None,
            cost_input: 0.0,
            cost_output: 0.0,
            cost_cache_read: 0.0,
            reasoning: false,
            vision: false,
            available: true,
            kind: String::new(),
            priority: None,
        }
    }

    /// `provider/id` selector.
    #[must_use]
    pub fn selector(&self) -> String {
        format!("{}/{}", self.provider, self.id)
    }

    /// Provider placeholder (no models yet, e.g. a keyless local endpoint):
    /// an entry with an empty `id` that only makes the provider appear in the
    /// sidebar.
    #[must_use]
    pub fn placeholder(provider: &str) -> Self {
        Self::new(provider, "", None)
    }
}

/// Thinking levels offered by the thinking strip, in cycle order.
pub const THINKING_LEVELS: [&str; 6] = ["off", "minimal", "low", "medium", "high", "xhigh"];

/// Built-in model roles in TS display order (`MODEL_ROLE_IDS`).
const ROLE_IDS: [&str; 15] = [
    "default",
    "smol",
    "slow",
    "vision",
    "plan",
    "commit",
    "tiny",
    "memory",
    "task",
    "advisor",
    "image",
    "web",
    "speech",
    "dictation",
    "judge",
];
/// Kind roles' auto-selection defaults (`rolePriorityDefaults` for the
/// non-chat roles): the model id taken when the role is unconfigured.
const AUTO_KIND_ROLES: [(&str, &str); 3] = [
    ("web", "parallel"),
    ("speech", "kokoro"),
    ("dictation", "parakeet-tdt-0.6b-v3"),
];
/// Kind tabs (TS `MODEL_KIND_TABS`); omp-rs only carries chat models.
const KIND_TABS: [&str; 11] = [
    "all",
    "chat",
    "tiny",
    "image",
    "tts",
    "stt",
    "search",
    "judge",
    "embedding",
    "rerank",
    "video",
];
const SIDEBAR_MIN: usize = 18;
const SIDEBAR_MAX: usize = 26;
const MIN_NAME_WIDTH: usize = 16;

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

/// One filtered row of the model list.
#[derive(Debug, Clone, PartialEq)]
pub enum Row {
    /// A catalogue model.
    Model(ModelEntry),
    /// Divider between recent/role models and the rest.
    Separator,
}

impl Row {
    /// `provider/id` text of a model row.
    #[must_use]
    pub fn display(&self) -> String {
        match self {
            Row::Model(m) => m.selector(),
            Row::Separator => String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryKind {
    Roles,
    All,
    Provider,
    Separator,
}

#[derive(Debug, Clone)]
struct Entry {
    id: String,
    kind: EntryKind,
    label: String,
    annotation: Option<String>,
    locked: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ChipAction {
    Assign(String),
    Unassign(String),
    Level(String),
}

#[derive(Debug, Clone)]
struct Chip {
    label: String,
    /// Role colour token / assigned flag for styling.
    assigned: bool,
    action: ChipAction,
}

#[derive(Debug, Clone)]
enum Strip {
    /// "model →" role chips.
    Role {
        model: ModelEntry,
        chips: Vec<Chip>,
        index: usize,
    },
    /// "role · model →" thinking-level chips.
    Thinking {
        model: ModelEntry,
        role: String,
        chips: Vec<Chip>,
        index: usize,
        initial: String,
    },
}

impl Strip {
    fn chips(&self) -> &[Chip] {
        match self {
            Strip::Role { chips, .. } | Strip::Thinking { chips, .. } => chips,
        }
    }
    fn index(&self) -> usize {
        match self {
            Strip::Role { index, .. } | Strip::Thinking { index, .. } => *index,
        }
    }
    fn index_mut(&mut self) -> &mut usize {
        match self {
            Strip::Role { index, .. } | Strip::Thinking { index, .. } => index,
        }
    }
}

/// Geometry of the last render, for mouse hit-testing.
#[derive(Debug, Clone, Default)]
struct Layout {
    sidebar_x: u16,
    sidebar_w: u16,
    body_x: u16,
    body_w: u16,
    content_y: u16,
    content_rows: u16,
    /// Per content row: sidebar entry index shown there.
    sidebar_rows: Vec<Option<usize>>,
    /// Absolute y of the first list row (browser) / first role row (roles view).
    list_y: u16,
    list_n: usize,
    roles_view: bool,
    footer_y: u16,
    /// `(x_start, x_end, chip index)` on the footer row.
    chips: Vec<(u16, u16, usize)>,
}

/// The `/model` (alt+p) browser overlay.
#[derive(Debug, Clone)]
pub struct ModelPicker {
    /// Full catalogue (configured and unconfigured providers).
    pub models: Vec<ModelEntry>,
    /// `provider/id` of the session's active model.
    pub current: Option<String>,
    /// Active thinking level (`off` when unset).
    pub current_level: String,
    /// Search query.
    pub query: String,
    /// Selected row within the visible (filtered) list.
    pub selected: usize,
    /// Persisted `modelRoles` (role → `provider/id[:level]`).
    pub role_map: BTreeMap<String, String>,
    /// Configured `defaultThinkingLevel` (shown next to the default role).
    pub default_level: String,
    active: String,
    focus_list: bool,
    kind_tab: usize,
    roles_index: usize,
    assigning: Option<String>,
    strip: Option<Strip>,
    pending: Vec<Action>,
    sidebar_scroll: Cell<usize>,
    sidebar_follow: Cell<bool>,
    list_start: Cell<usize>,
    list_follow: Cell<bool>,
    layout: RefCell<Layout>,
}

impl Default for ModelPicker {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            current: None,
            current_level: "off".to_owned(),
            query: String::new(),
            selected: 0,
            role_map: BTreeMap::new(),
            default_level: "high".to_owned(),
            active: "all".to_owned(),
            focus_list: false,
            kind_tab: 0,
            roles_index: 0,
            assigning: None,
            strip: None,
            pending: Vec::new(),
            sidebar_scroll: Cell::new(0),
            sidebar_follow: Cell::new(true),
            list_start: Cell::new(0),
            list_follow: Cell::new(true),
            layout: RefCell::new(Layout::default()),
        }
    }
}

// ---------------------------------------------------------------- helpers

fn width(s: &str) -> usize {
    UnicodeWidthStr::width(s)
}

fn spans_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(|s| width(&s.content)).sum()
}

/// Truncate to `max` columns, ending in `…` when cut (TS `truncateToWidth`).
fn truncate_spans(spans: Vec<Span<'static>>, max: usize) -> Vec<Span<'static>> {
    if spans_width(&spans) <= max {
        return spans;
    }
    let budget = max.saturating_sub(1);
    let mut out = Vec::new();
    let mut used = 0;
    for span in spans {
        let mut text = String::new();
        for ch in span.content.chars() {
            let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
            if used + cw > budget {
                if !text.is_empty() {
                    out.push(Span::styled(text, span.style));
                }
                out.push(Span::raw("…"));
                return out;
            }
            used += cw;
            text.push(ch);
        }
        out.push(Span::styled(text, span.style));
    }
    out
}

fn pad(mut spans: Vec<Span<'static>>, to: usize) -> Vec<Span<'static>> {
    let w = spans_width(&spans);
    if w < to {
        spans.push(Span::raw(" ".repeat(to - w)));
    }
    spans
}

/// TS `formatNumber`: `128k`, `4.1k`, `1M`.
fn format_number(n: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let f = n as f64;
    let trim = |v: f64, suffix: &str| {
        let s = format!("{v:.1}");
        format!("{}{suffix}", s.strip_suffix(".0").unwrap_or(&s))
    };
    if n >= 1_000_000 {
        trim(f / 1_000_000.0, "m")
    } else if n >= 1_000 {
        trim(f / 1_000.0, "k")
    } else {
        n.to_string()
    }
}

/// One per-million price leg (TS `formatCostLeg`).
fn cost_leg(n: f64) -> String {
    if !n.is_finite() || n < 0.0 {
        return "?".to_owned();
    }
    let s = if n >= 100.0 {
        format!("{}", n.round())
    } else if n >= 10.0 {
        format!("{n:.1}")
    } else {
        format!("{n:.2}")
    };
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        s
    }
}

fn is_free(m: &ModelEntry) -> bool {
    m.cost_input == 0.0 && m.cost_output == 0.0
}

fn cost_pair(m: &ModelEntry) -> String {
    if is_free(m) {
        "free".to_owned()
    } else {
        format!("${}/{}", cost_leg(m.cost_input), cost_leg(m.cost_output))
    }
}

fn matches_query(m: &ModelEntry, query: &str) -> bool {
    let hay = format!("{}/{} {}", m.provider, m.id, m.name).to_lowercase();
    query
        .split_whitespace()
        .all(|tok| hay.contains(&tok.to_lowercase()))
}

fn base_selector(value: &str) -> &str {
    match value.rsplit_once(':') {
        Some((b, l)) if THINKING_LEVELS.contains(&l) || l == "inherit" => b,
        _ => value,
    }
}

fn role_label(role: &str) -> &str {
    role
}

/// TS `extractVersionNumber`: first version number found in a model id.
fn version_number(id: &str) -> f64 {
    let b: Vec<char> = id.chars().collect();
    let digits = |from: usize| -> usize {
        let mut i = from;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        i
    };
    let sep_at = |i: usize| i == 0 || b[i - 1] == '-' || b[i - 1] == '_';
    // Dot-separated: "gemini-2.5-pro" → 2.5.
    for i in 0..b.len() {
        if !sep_at(i) || !b[i].is_ascii_digit() {
            continue;
        }
        let a = digits(i);
        if a < b.len() && b[a] == '.' {
            let c = digits(a + 1);
            if c > a + 1 {
                return b[i..c].iter().collect::<String>().parse().unwrap_or(0.0);
            }
        }
    }
    // Dash-separated short segments: "claude-sonnet-4-6" → 4.6.
    for i in 0..b.len() {
        if !sep_at(i) || !b[i].is_ascii_digit() {
            continue;
        }
        let a = digits(i);
        if a - i <= 2 && a < b.len() && b[a] == '-' {
            let c = digits(a + 1);
            if c > a + 1 && c - (a + 1) <= 2 && (c == b.len() || b[c] == '-') {
                let s: String = b[i..a]
                    .iter()
                    .chain(['.'].iter())
                    .chain(b[a + 1..c].iter())
                    .collect();
                return s.parse().unwrap_or(0.0);
            }
        }
    }
    // Single number after a separator: "gpt-4o" → 4.
    for i in 0..b.len() {
        if sep_at(i) && b[i].is_ascii_digit() {
            let a = digits(i);
            return b[i..a].iter().collect::<String>().parse().unwrap_or(0.0);
        }
    }
    0.0
}

fn locale_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.to_lowercase()
        .cmp(&b.to_lowercase())
        .then_with(|| b.cmp(a))
}

/// TS `modelItemComparator` below the role rank: provider, priority,
/// version (desc), recency (`-latest` / `-YYYYMMDD`), then id.
fn cmp_model(a: &ModelEntry, b: &ModelEntry) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let date = |id: &str| -> String {
        id.rsplit_once('-')
            .filter(|(_, d)| d.len() == 8 && d.chars().all(|c| c.is_ascii_digit()))
            .map(|(_, d)| d.to_owned())
            .unwrap_or_default()
    };
    locale_cmp(&a.provider, &b.provider)
        .then_with(|| {
            a.priority
                .unwrap_or(i64::MAX)
                .cmp(&b.priority.unwrap_or(i64::MAX))
        })
        .then_with(|| {
            version_number(&b.id)
                .partial_cmp(&version_number(&a.id))
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| {
            let (al, bl) = (a.id.ends_with("-latest"), b.id.ends_with("-latest"));
            let (ad, bd) = (date(&a.id), date(&b.id));
            let (ar, br) = (al || !ad.is_empty(), bl || !bd.is_empty());
            if ar != br {
                return if ar {
                    Ordering::Less
                } else {
                    Ordering::Greater
                };
            }
            if !ar {
                return locale_cmp(&a.id, &b.id);
            }
            if al != bl {
                return if al {
                    Ordering::Less
                } else {
                    Ordering::Greater
                };
            }
            if !ad.is_empty() && !bd.is_empty() {
                return bd.cmp(&ad);
            }
            locale_cmp(&a.id, &b.id)
        })
}

// ------------------------------------------------------------------ logic

impl ModelPicker {
    /// Open-state reset: back to "All models", clear the query, reload the
    /// persisted role assignments and select the current model.
    pub fn on_open(&mut self) {
        self.query.clear();
        self.active = "all".to_owned();
        self.focus_list = false;
        self.kind_tab = 0;
        self.assigning = None;
        self.strip = None;
        self.roles_index = 0;
        self.sidebar_scroll.set(0);
        self.sidebar_follow.set(true);
        self.list_start.set(0);
        self.list_follow.set(true);
        let cwd = std::env::current_dir().unwrap_or_default();
        let settings = omp_settings::Settings::load(&cwd);
        self.role_map = settings.model_roles();
        self.default_level = settings
            .get_string("defaultThinkingLevel")
            .unwrap_or_else(|| "high".to_owned());
        self.selected = 0;
        if let Some(cur) = &self.current {
            let rows = self.rows();
            if let Some(pos) = rows
                .iter()
                .position(|r| matches!(r, Row::Model(m) if &m.selector() == cur))
            {
                self.selected = pos;
            }
        }
    }

    /// Per provider: (available models, all models, keyless placeholder).
    fn provider_counts(&self) -> BTreeMap<&str, (usize, usize, bool)> {
        let mut counts: BTreeMap<&str, (usize, usize, bool)> = BTreeMap::new();
        for m in &self.models {
            let e = counts.entry(m.provider.as_str()).or_default();
            if m.id.is_empty() {
                e.2 |= m.available;
                continue;
            }
            e.1 += 1;
            if m.available {
                e.0 += 1;
            }
        }
        counts
    }

    fn assigned_roles(&self) -> usize {
        ROLE_IDS
            .iter()
            .filter(|r| self.role_selector(r, false).is_some())
            .count()
    }

    /// Effective selector of a role: explicit `modelRoles` entry, else the
    /// live model for `default`; with `auto`, the built-in default of the
    /// non-chat roles (TS auto-selection).
    fn role_selector(&self, role: &str, auto: bool) -> Option<String> {
        self.role_map
            .get(role)
            .map(|v| base_selector(v).to_owned())
            .or_else(|| (role == "default").then(|| self.current.clone()).flatten())
            .or_else(|| {
                if !auto {
                    return None;
                }
                let (_, id) = AUTO_KIND_ROLES.iter().find(|(r, _)| *r == role)?;
                self.models
                    .iter()
                    .find(|m| m.available && m.id == *id && !m.kind.is_empty())
                    .map(ModelEntry::selector)
            })
    }

    fn role_level(&self, role: &str) -> Option<String> {
        let value = self.role_map.get(role)?;
        let base = base_selector(value);
        value
            .strip_prefix(base)?
            .strip_prefix(':')
            .map(str::to_owned)
    }

    fn search_active(&self) -> bool {
        !self.query.trim().is_empty()
    }

    fn entries(&self) -> Vec<Entry> {
        let counts = self.provider_counts();
        let total: usize = counts.values().map(|c| c.0).sum();
        let searching = self.search_active();
        let match_count = |provider: Option<&str>| {
            self.models
                .iter()
                .filter(|m| {
                    m.available
                        && !m.id.is_empty()
                        && provider.is_none_or(|p| m.provider == p)
                        && matches_query(m, &self.query)
                })
                .count()
        };
        let mut out = vec![
            Entry {
                id: "roles".into(),
                kind: EntryKind::Roles,
                label: "Roles".into(),
                annotation: Some(format!("{}/{}", self.assigned_roles(), ROLE_IDS.len())),
                locked: false,
            },
            Entry {
                id: "all".into(),
                kind: EntryKind::All,
                label: "All models".into(),
                annotation: Some(if searching { match_count(None) } else { total }.to_string()),
                locked: false,
            },
        ];
        let mut unlocked: Vec<Entry> = Vec::new();
        let mut locked: Vec<Entry> = Vec::new();
        for (provider, (avail, _, keyless)) in &counts {
            let is_locked = *avail == 0 && !*keyless;
            let entry = Entry {
                id: format!("provider:{provider}"),
                kind: EntryKind::Provider,
                label: (*provider).to_owned(),
                annotation: (!is_locked).then(|| {
                    if searching {
                        match_count(Some(provider))
                    } else {
                        *avail
                    }
                    .to_string()
                }),
                locked: is_locked,
            };
            if is_locked {
                locked.push(entry);
            } else {
                unlocked.push(entry);
            }
        }
        unlocked.sort_by_key(|e| e.label.to_lowercase());
        locked.sort_by_key(|e| e.label.to_lowercase());
        if searching {
            // Providers with matches float to the top (each group alphabetical).
            unlocked.sort_by_key(|e| {
                let hit = e.annotation.as_deref().is_some_and(|a| a != "0");
                (!hit, e.label.to_lowercase())
            });
        }
        if !unlocked.is_empty() {
            out.push(Entry {
                id: "sep:providers".into(),
                kind: EntryKind::Separator,
                label: String::new(),
                annotation: None,
                locked: false,
            });
            out.extend(unlocked);
        }
        if !locked.is_empty() {
            out.push(Entry {
                id: "sep:locked".into(),
                kind: EntryKind::Separator,
                label: String::new(),
                annotation: None,
                locked: false,
            });
            out.extend(locked);
        }
        out
    }

    fn active_entry(&self) -> Entry {
        let entries = self.entries();
        entries
            .iter()
            .find(|e| e.id == self.active)
            .or_else(|| entries.iter().find(|e| e.id == "all"))
            .cloned()
            .unwrap_or(Entry {
                id: "all".into(),
                kind: EntryKind::All,
                label: "All models".into(),
                annotation: None,
                locked: false,
            })
    }

    fn roles_view(&self, e: &Entry) -> bool {
        e.kind == EntryKind::Roles && self.assigning.is_none()
    }

    fn locked_view(&self, e: &Entry) -> bool {
        e.kind == EntryKind::Provider && e.locked && self.assigning.is_none()
    }

    fn browser_view(&self, e: &Entry) -> bool {
        self.assigning.is_some()
            || e.kind == EntryKind::All
            || (e.kind == EntryKind::Provider && !e.locked)
    }

    /// Visible rows of the model list for the active scope, kind tab and query.
    #[must_use]
    pub fn rows(&self) -> Vec<Row> {
        let entry = self.active_entry();
        if !self.browser_view(&entry) {
            return Vec::new();
        }
        let provider = (entry.kind == EntryKind::Provider).then_some(entry.label.as_str());
        let kind = KIND_TABS[self.kind_tab];
        let assigned: Vec<(usize, String)> = ROLE_IDS
            .iter()
            .enumerate()
            .filter_map(|(i, r)| self.role_selector(r, true).map(|s| (i, s)))
            .collect();
        let rank = |m: &ModelEntry| {
            let sel = m.selector();
            assigned
                .iter()
                .find(|(_, s)| *s == sel)
                .map_or(ROLE_IDS.len(), |(i, _)| *i)
        };
        let mut pool: Vec<&ModelEntry> = self
            .models
            .iter()
            .filter(|m| m.available && !m.id.is_empty())
            .filter(|m| provider.is_none_or(|p| m.provider == p))
            .filter(|m| match kind {
                "all" => true,
                "chat" => m.kind.is_empty(),
                k => m.kind == k,
            })
            .filter(|m| matches_query(m, &self.query))
            .collect();
        if !self.search_active() {
            pool.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| cmp_model(a, b)));
        }
        let split = pool.iter().position(|m| rank(m) == ROLE_IDS.len());
        let mut out: Vec<Row> = Vec::new();
        for (i, m) in pool.iter().enumerate() {
            if !self.search_active() && split == Some(i) && i > 0 {
                out.push(Row::Separator);
            }
            out.push(Row::Model((*m).clone()));
        }
        out
    }

    fn set_active(&mut self, id: &str) {
        if self.active != id {
            self.active = id.to_owned();
            self.selected = 0;
            self.list_start.set(0);
            self.list_follow.set(true);
            self.sidebar_follow.set(true);
        }
    }

    fn move_sidebar(&mut self, delta: isize) {
        let entries = self.entries();
        let n = entries.len() as isize;
        let mut idx = entries
            .iter()
            .position(|e| e.id == self.active)
            .unwrap_or(0) as isize;
        for _ in 0..n {
            idx = (idx + delta).rem_euclid(n);
            let e = &entries[idx as usize];
            if e.kind != EntryKind::Separator {
                if e.kind == EntryKind::Roles {
                    self.assigning = None;
                }
                let id = e.id.clone();
                self.set_active(&id);
                return;
            }
        }
    }

    fn move_selection(&mut self, delta: isize, wrap: bool) {
        let rows = self.rows();
        let n = rows.len() as isize;
        if n == 0 {
            return;
        }
        let mut idx = self.selected.min(rows.len() - 1) as isize;
        let step = delta.signum();
        let mut remaining = delta.abs();
        while remaining > 0 {
            let mut next = idx + step;
            if wrap {
                next = next.rem_euclid(n);
            } else {
                next = next.clamp(0, n - 1);
            }
            if next == idx {
                break;
            }
            idx = next;
            if rows[idx as usize] != Row::Separator {
                remaining -= 1;
            }
        }
        // Never rest on a separator.
        if rows[idx as usize] == Row::Separator {
            idx = (idx + 1).min(n - 1);
        }
        self.selected = idx as usize;
        self.list_follow.set(true);
    }

    fn clamp_selected(&mut self) {
        let rows = self.rows();
        if rows.is_empty() {
            self.selected = 0;
        } else {
            self.selected = self.selected.min(rows.len() - 1);
            if rows[self.selected] == Row::Separator {
                self.selected = (self.selected + 1).min(rows.len() - 1);
            }
        }
    }

    fn selected_model(&self) -> Option<ModelEntry> {
        match self.rows().get(self.selected) {
            Some(Row::Model(m)) => Some(m.clone()),
            _ => None,
        }
    }

    fn emit_roles(&mut self) {
        let map: serde_json::Map<String, serde_json::Value> = self
            .role_map
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        self.pending.push(Action::SettingChanged {
            id: "modelRoles".to_owned(),
            value: serde_json::Value::Object(map),
        });
    }

    /// Roles that accept `m` (TS `getRoleInfo(role).accepts`), shown as chips.
    fn roles_for(m: &ModelEntry) -> Vec<&'static str> {
        match m.kind.as_str() {
            "" => {
                let mut roles = vec!["default", "smol", "slow"];
                if m.vision {
                    roles.push("vision");
                }
                roles.extend(["tiny", "memory"]);
                roles
            }
            "tiny" => vec!["tiny"],
            "image" => vec!["image"],
            "search" => vec!["web"],
            "tts" => vec!["speech"],
            "stt" => vec!["dictation"],
            "judge" => vec!["judge"],
            _ => Vec::new(),
        }
    }

    fn activate_model(&mut self, m: ModelEntry) {
        if let Some(role) = self.assigning.take() {
            self.assign_role(&m, &role, true);
            return;
        }
        let mut chips = Vec::new();
        for role in Self::roles_for(&m) {
            let explicit = self
                .role_map
                .get(role)
                .is_some_and(|v| base_selector(v) == m.selector());
            chips.push(Chip {
                label: role_label(role).to_owned(),
                assigned: explicit,
                action: if explicit {
                    ChipAction::Unassign(role.to_owned())
                } else {
                    ChipAction::Assign(role.to_owned())
                },
            });
        }
        self.strip = Some(Strip::Role {
            model: m,
            chips,
            index: 0,
        });
    }

    /// Persist `role → model` (keeping a still-valid thinking level), switch
    /// the live model for `default`, then offer the thinking strip.
    fn assign_role(&mut self, m: &ModelEntry, role: &str, return_to_roles: bool) {
        let level = self
            .role_level(role)
            .filter(|l| l != "inherit" && m.reasoning);
        let value = match &level {
            Some(l) => format!("{}:{l}", m.selector()),
            None => m.selector(),
        };
        self.role_map.insert(role.to_owned(), value);
        if role == "default" {
            self.pending.push(Action::SetModel {
                provider: m.provider.clone(),
                id: m.id.clone(),
            });
            if let Some(l) = &level {
                self.pending.push(Action::SetThinkingLevel(l.clone()));
                self.current_level.clone_from(l);
            }
            self.current = Some(m.selector());
        }
        self.emit_roles();
        if m.reasoning {
            let chips: Vec<Chip> = THINKING_LEVELS
                .iter()
                .map(|l| Chip {
                    label: (*l).to_owned(),
                    assigned: false,
                    action: ChipAction::Level((*l).to_owned()),
                })
                .collect();
            let initial = level.unwrap_or_else(|| "inherit".to_owned());
            let index = THINKING_LEVELS
                .iter()
                .position(|l| *l == initial)
                .unwrap_or(0);
            self.strip = Some(Strip::Thinking {
                model: m.clone(),
                role: role.to_owned(),
                chips,
                index,
                initial,
            });
        } else if return_to_roles {
            self.set_active("roles");
            self.focus_list = true;
        }
    }

    fn unassign_role(&mut self, role: &str) {
        if self.role_map.remove(role).is_some() {
            self.emit_roles();
        }
    }

    fn activate_strip_chip(&mut self) {
        let Some(strip) = self.strip.take() else {
            return;
        };
        let Some(chip) = strip.chips().get(strip.index()).cloned() else {
            return;
        };
        match (strip, chip.action) {
            (Strip::Role { model, .. }, ChipAction::Assign(role)) => {
                self.assign_role(&model, &role, false);
            }
            (Strip::Role { .. }, ChipAction::Unassign(role)) => self.unassign_role(&role),
            (
                Strip::Thinking {
                    model,
                    role,
                    initial,
                    ..
                },
                ChipAction::Level(level),
            ) => {
                if level != initial {
                    self.role_map
                        .insert(role.clone(), format!("{}:{level}", model.selector()));
                    if role == "default" {
                        self.pending.push(Action::SetThinkingLevel(level.clone()));
                        self.current_level = level;
                    }
                    self.emit_roles();
                }
            }
            _ => {}
        }
    }

    fn move_kind_tab(&mut self, delta: isize) {
        let n = KIND_TABS.len() as isize;
        self.kind_tab = (self.kind_tab as isize + delta).rem_euclid(n) as usize;
        self.selected = 0;
        self.list_start.set(0);
    }

    fn edit_query(&mut self, key: KeyEvent) -> bool {
        let before = self.query.clone();
        match key.code {
            KeyCode::Char(c)
                if key.modifiers == KeyModifiers::NONE || key.modifiers == KeyModifiers::SHIFT =>
            {
                self.query.push(c);
            }
            KeyCode::Backspace => {
                self.query.pop();
            }
            _ => return false,
        }
        if self.query != before {
            self.selected = 0;
            self.list_start.set(0);
            self.list_follow.set(true);
            self.clamp_selected();
        }
        true
    }

    /// Handle one key event; queues `Action`s on assignment.
    pub fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        let code = key.code;
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let enter = code == KeyCode::Enter;

        if let Some(strip) = &mut self.strip {
            let n = strip.chips().len().max(1);
            match code {
                KeyCode::Esc => self.strip = None,
                KeyCode::Left | KeyCode::Up | KeyCode::BackTab => {
                    let i = strip.index_mut();
                    *i = (*i + n - 1) % n;
                }
                KeyCode::Right | KeyCode::Down | KeyCode::Tab => {
                    let i = strip.index_mut();
                    *i = (*i + 1) % n;
                }
                KeyCode::Enter => self.activate_strip_chip(),
                _ => {}
            }
            return EventOutcome::Consumed;
        }

        let entry = self.active_entry();
        let roles_view = self.roles_view(&entry);
        let locked_view = self.locked_view(&entry);

        if code == KeyCode::Esc {
            if self.assigning.take().is_some() {
                return EventOutcome::Consumed;
            }
            if self.browser_view(&entry) && !self.query.is_empty() {
                self.query.clear();
                self.selected = 0;
                self.list_start.set(0);
                return EventOutcome::Consumed;
            }
            return EventOutcome::Ignored;
        }
        if matches!(code, KeyCode::Tab | KeyCode::BackTab) {
            self.focus_list = !self.focus_list;
            return EventOutcome::Consumed;
        }
        if alt && matches!(code, KeyCode::Left | KeyCode::Char('b')) {
            if !roles_view {
                self.move_kind_tab(-1);
            }
            return EventOutcome::Consumed;
        }
        if alt && matches!(code, KeyCode::Right | KeyCode::Char('f')) {
            if !roles_view {
                self.move_kind_tab(1);
            }
            return EventOutcome::Consumed;
        }
        if code == KeyCode::Left {
            self.focus_list = false;
            return EventOutcome::Consumed;
        }
        if code == KeyCode::Right {
            if roles_view || self.browser_view(&entry) {
                self.focus_list = true;
            }
            return EventOutcome::Consumed;
        }
        if !self.focus_list {
            match code {
                KeyCode::Up => {
                    self.move_sidebar(-1);
                    return EventOutcome::Consumed;
                }
                KeyCode::Down => {
                    self.move_sidebar(1);
                    return EventOutcome::Consumed;
                }
                _ => {}
            }
        }

        let typed = matches!(code, KeyCode::Char(c) if !c.is_whitespace())
            && !key.modifiers.contains(KeyModifiers::CONTROL)
            && !alt;
        if roles_view {
            let edits_query = !self.query.is_empty() && (typed || code == KeyCode::Backspace);
            if (!self.focus_list && typed) || edits_query {
                self.set_active("all");
                self.focus_list = true;
                self.edit_query(key);
                return EventOutcome::Consumed;
            }
            return self.roles_key(key);
        }
        if locked_view {
            if typed {
                self.set_active("all");
                self.focus_list = true;
                self.edit_query(key);
            }
            return EventOutcome::Consumed;
        }
        if !self.focus_list && enter {
            self.focus_list = true;
            return EventOutcome::Consumed;
        }

        let page = {
            let l = self.layout.borrow();
            l.list_n.max(1) as isize
        };
        match code {
            KeyCode::Up => self.move_selection(-1, true),
            KeyCode::Down => self.move_selection(1, true),
            KeyCode::PageUp => self.move_selection(-page, false),
            KeyCode::PageDown => self.move_selection(page, false),
            KeyCode::Home => self.move_selection(-(self.rows().len() as isize), false),
            KeyCode::End => self.move_selection(self.rows().len() as isize, false),
            KeyCode::Enter => {
                if let Some(m) = self.selected_model() {
                    self.activate_model(m);
                }
            }
            // F5: online catalog refresh (TS model hub refresh); the runtime
            // port updates `models.db`, and reopening the browser re-reads it.
            KeyCode::F(5) => self.pending.push(Action::RefreshModels),
            _ => {
                if self.edit_query(key) {
                    self.focus_list = true;
                } else {
                    return EventOutcome::Ignored;
                }
            }
        }
        EventOutcome::Consumed
    }

    fn roles_key(&mut self, key: KeyEvent) -> EventOutcome {
        let n = ROLE_IDS.len();
        match key.code {
            KeyCode::Up => self.roles_index = (self.roles_index + n - 1) % n,
            KeyCode::Down => self.roles_index = (self.roles_index + 1) % n,
            KeyCode::Enter => self.start_assign(ROLE_IDS[self.roles_index]),
            KeyCode::Char('x') => self.unassign_role(ROLE_IDS[self.roles_index]),
            _ => {}
        }
        EventOutcome::Consumed
    }

    fn start_assign(&mut self, role: &str) {
        self.assigning = Some(role.to_owned());
        self.query.clear();
        self.set_active("all");
        self.focus_list = true;
        self.selected = 0;
        if let Some(sel) = self.role_selector(role, true) {
            if let Some(pos) = self
                .rows()
                .iter()
                .position(|r| matches!(r, Row::Model(m) if m.selector() == sel))
            {
                self.selected = pos;
            }
        }
        self.list_follow.set(true);
    }

    /// Mouse routing (SGR cells == frame coords): click sidebar/model/role
    /// rows and footer chips, wheel scrolls the pane under the pointer.
    pub fn handle_mouse(&mut self, ev: &SgrMouseEvent) -> EventOutcome {
        let l = self.layout.borrow().clone();
        let (row, col) = (ev.row, ev.col);
        if let Some(strip) = &mut self.strip {
            if ev.left_click && row == l.footer_y {
                if let Some(&(_, _, idx)) = l.chips.iter().find(|(a, b, _)| col >= *a && col < *b) {
                    *strip.index_mut() = idx;
                    self.activate_strip_chip();
                }
            }
            return EventOutcome::Consumed;
        }
        let in_content = row >= l.content_y && row < l.content_y + l.content_rows;
        let over_sidebar = in_content && col >= l.sidebar_x && col < l.sidebar_x + l.sidebar_w;
        let over_body = in_content && col >= l.body_x && col < l.body_x + l.body_w;
        if let Some(dir) = ev.wheel {
            if over_sidebar {
                let max = self.entries().len().saturating_sub(l.content_rows as usize);
                let cur = self.sidebar_scroll.get() as isize + isize::from(dir);
                self.sidebar_scroll.set(cur.clamp(0, max as isize) as usize);
                self.sidebar_follow.set(false);
            } else if over_body && !l.roles_view {
                let total = self.rows().len();
                let max = total.saturating_sub(l.list_n.max(1));
                let cur = self.list_start.get() as isize + isize::from(dir);
                self.list_start.set(cur.clamp(0, max as isize) as usize);
                self.list_follow.set(false);
            }
            return EventOutcome::Consumed;
        }
        if !ev.left_click {
            return EventOutcome::Consumed;
        }
        if over_sidebar {
            let line = (row - l.content_y) as usize;
            if let Some(Some(idx)) = l.sidebar_rows.get(line) {
                let entries = self.entries();
                if let Some(e) = entries.get(*idx) {
                    if e.kind != EntryKind::Separator {
                        if e.kind == EntryKind::Roles {
                            self.assigning = None;
                        }
                        let id = e.id.clone();
                        self.set_active(&id);
                        self.focus_list = false;
                    }
                }
            }
        } else if over_body && row >= l.list_y && ((row - l.list_y) as usize) < l.list_n {
            let offset = (row - l.list_y) as usize;
            if l.roles_view {
                if offset < ROLE_IDS.len() {
                    let again = self.roles_index == offset && self.focus_list;
                    self.roles_index = offset;
                    self.focus_list = true;
                    if again {
                        self.start_assign(ROLE_IDS[offset]);
                    }
                }
            } else {
                let idx = self.list_start.get() + offset;
                if let Some(Row::Model(m)) = self.rows().get(idx).cloned() {
                    self.focus_list = true;
                    if idx == self.selected {
                        self.activate_model(m);
                    } else {
                        self.selected = idx;
                    }
                }
            }
        }
        EventOutcome::Consumed
    }
}

// ----------------------------------------------------------------- render

struct St {
    border: Style,
    dim: Style,
    muted: Style,
    accent: Style,
    success: Style,
    warning: Style,
    error: Style,
    text: Style,
    sel_bg: Option<Color>,
}

impl St {
    fn new(theme: &Theme) -> Self {
        let fg = |t: &str| Style::default().fg(theme.color(t));
        let bg = theme.color("selectedBg");
        Self {
            border: fg("border"),
            dim: fg("dim"),
            muted: fg("muted"),
            accent: fg("accent"),
            success: fg("success"),
            warning: fg("warning"),
            error: fg("error"),
            text: Style::default(),
            sel_bg: (bg != Color::Reset).then_some(bg),
        }
    }
}

fn sp(text: impl Into<String>, style: Style) -> Span<'static> {
    Span::styled(text.into(), style)
}

fn status_row_text(p: &ModelPicker, e: &Entry) -> (String, bool) {
    if let Some(role) = &p.assigning {
        return (format!(" Assigning {role} — ⏎ assigns, ⎋ cancels"), true);
    }
    let text = match e.kind {
        EntryKind::Roles => "Model roles — cleared roles fall back to auto-selection".to_owned(),
        EntryKind::Provider if e.locked => format!("{} · not configured", e.label),
        EntryKind::Provider => format!(
            "{} · {} models",
            e.label,
            e.annotation.as_deref().unwrap_or("0")
        ),
        _ => "All available models".to_owned(),
    };
    (format!(" {text}"), false)
}

impl ModelPicker {
    fn footer_hint(&self, e: &Entry) -> String {
        let cancel = "⎋";
        if self.assigning.is_some() {
            if !self.focus_list {
                return format!("⏎/→ models · ↑/↓ providers · type to search · Alt+←/Alt+→ kind · {cancel} cancel");
            }
            return format!("⏎ assign · ↑/↓ models · ← providers · type to search · Alt+←/Alt+→ kind · {cancel} cancel");
        }
        match e.kind {
            EntryKind::Roles => {
                if !self.focus_list {
                    return format!(
                        "↑/↓ providers · ⏎/→ roles · Alt+←/Alt+→ tabs · {cancel} close"
                    );
                }
                if !self.query.is_empty() {
                    return format!(
                        "↑/↓ rows · ⏎ pick · type to search · ← providers · {cancel} close"
                    );
                }
                "↑/↓ rows · ⏎ pick · x clear".to_owned()
            }
            EntryKind::Provider if e.locked => format!("↑/↓ providers · {cancel} close"),
            _ => {
                if self.focus_list {
                    format!("⏎ assign roles · ↑/↓ models · ← providers · type to search · Alt+←/Alt+→ kind · {cancel} close")
                } else {
                    format!("⏎/→ models · ↑/↓ providers · type to search · Alt+←/Alt+→ kind · {cancel} close")
                }
            }
        }
    }

    fn kind_tabs_line(&self, st: &St, bw: usize) -> Vec<Span<'static>> {
        let mut spans = vec![sp(" ", st.text), sp("Kind:", st.dim), sp(" ", st.text)];
        let active = self.kind_tab;
        for (i, label) in KIND_TABS.iter().enumerate() {
            if i > 0 {
                if i == active || i - 1 == active {
                    spans.push(sp("  ", st.text));
                } else {
                    spans.push(sp(" ┆ ", st.dim));
                }
            }
            if i == active {
                spans.push(sp(
                    format!("◀ {label} ▶"),
                    st.accent.add_modifier(Modifier::BOLD),
                ));
            } else {
                spans.push(sp(*label, st.dim));
            }
        }
        spans.push(sp("  Alt+←/Alt+→", st.dim));
        truncate_spans(spans, bw)
    }

    fn model_row(
        &self,
        st: &St,
        m: &ModelEntry,
        rw: usize,
        selected: bool,
        ctx_w: usize,
        cost_w: usize,
    ) -> Vec<Span<'static>> {
        let prefix = if selected && self.focus_list {
            vec![sp("❯", st.accent), sp(" ", st.text)]
        } else {
            vec![sp("  ", st.text)]
        };
        let mut left = prefix;
        left.push(sp(format!("{}/", m.provider), st.dim));
        left.push(sp(m.id.clone(), if selected { st.accent } else { st.text }));
        let ctx = m
            .context_window
            .filter(|c| *c > 0)
            .map(|c| format!("{} ◫", format_number(c)))
            .unwrap_or_default();
        let cost = cost_pair(m);
        let mut cols: Vec<(String, usize)> = Vec::new();
        if ctx_w > 0 {
            cols.push((ctx, ctx_w));
        }
        if cost_w > 0 {
            cols.push((cost, cost_w));
        }
        let meta: usize =
            cols.iter().map(|c| c.1).sum::<usize>() + 2 * cols.len().saturating_sub(1);
        let available = rw
            .saturating_sub(meta + usize::from(!cols.is_empty()))
            .max(1);
        let mut line = pad(truncate_spans(left, available), available);
        if !cols.is_empty() {
            line.push(sp(" ", st.text));
            let joined: Vec<String> = cols
                .iter()
                .map(|(t, w)| format!("{}{t}", " ".repeat(w.saturating_sub(width(t)))))
                .collect();
            line.push(sp(joined.join("  "), st.dim));
        }
        line
    }

    fn thinking_glyph(&self, level: &str, theme: &Theme) -> Option<String> {
        match level {
            "inherit" | "auto" => None,
            "off" => Some(theme.symbol("status.disabled").to_owned()),
            l => theme
                .symbol(&format!("thinking.{l}"))
                .split(' ')
                .next()
                .filter(|g| !g.is_empty())
                .map(str::to_owned),
        }
    }

    fn detail_lines(&self, st: &St, bw: usize, theme: &Theme) -> [Vec<Span<'static>>; 2] {
        let Some(m) = self.selected_model() else {
            return [Vec::new(), Vec::new()];
        };
        let mut facts = vec![m.name.clone()];
        if let Some(c) = m.context_window.filter(|c| *c > 0) {
            facts.push(format!("{} ctx", format_number(c)));
        }
        if let Some(t) = m.max_tokens.filter(|t| *t > 0) {
            facts.push(format!("{} out", format_number(t)));
        }
        facts.push(format!("{} per M", cost_pair(&m)));
        if m.reasoning {
            facts.push("reasoning".to_owned());
        }
        if m.vision {
            facts.push("vision".to_owned());
        }
        let line1 = truncate_spans(vec![sp(format!("  {}", facts.join(" · ")), st.muted)], bw);
        let mut chips: Vec<Span<'static>> = vec![sp("  ", st.text)];
        let mut first = true;
        for role in Self::roles_for(&m) {
            if !first {
                chips.push(sp(" · ", st.dim));
            }
            first = false;
            let held = self.role_selector(role, true).as_deref() == Some(m.selector().as_str());
            if held {
                chips.push(sp(format!("● {role}"), st.success));
                let level = self
                    .role_level(role)
                    .or_else(|| (role == "default").then(|| self.default_level.clone()));
                if let Some(glyph) = level.as_deref().and_then(|l| self.thinking_glyph(l, theme)) {
                    chips.push(sp(format!(" {glyph}"), st.dim));
                }
            } else {
                chips.push(sp(format!("○ {role}"), st.dim));
            }
        }
        [line1, truncate_spans(chips, bw)]
    }

    #[allow(clippy::too_many_lines)]
    fn browser_lines(
        &self,
        st: &St,
        bw: usize,
        rows: usize,
        entry: &Entry,
        layout: &mut Layout,
        list_abs_y: u16,
        theme: &Theme,
    ) -> Vec<Vec<Span<'static>>> {
        let mv = rows.saturating_sub(7).max(1);
        let mut lines: Vec<Vec<Span<'static>>> = Vec::new();
        let (status, accent) = status_row_text(self, entry);
        lines.push(truncate_spans(
            vec![sp(status, if accent { st.accent } else { st.muted })],
            bw,
        ));
        lines.push(self.kind_tabs_line(st, bw));
        // Search row.
        let icon = theme.symbol("icon.search");
        let mut search = vec![
            sp(" ", st.text),
            sp(icon, st.accent),
            sp(" ", st.text),
            sp("> ", st.accent),
            sp(self.query.clone(), st.text),
        ];
        if self.focus_list {
            search.push(sp(" ", Style::default().add_modifier(Modifier::REVERSED)));
        }
        lines.push(truncate_spans(search, bw));
        lines.push(Vec::new());

        let visible = self.rows();
        let total = visible.len();
        let mut start = self.list_start.get().min(total.saturating_sub(mv));
        if self.list_follow.get() && total > 0 {
            let sel = self.selected.min(total - 1);
            if sel < start {
                start = sel;
            } else if sel >= start + mv {
                start = sel + 1 - mv;
            }
            start = start.min(total.saturating_sub(mv));
            self.list_follow.set(false);
        }
        self.list_start.set(start);
        let end = (start + mv).min(total);
        layout.list_y = list_abs_y + 4;
        layout.list_n = end - start;
        let mut shown = 0;
        if total == 0 {
            let msg = if entry.kind == EntryKind::Provider && !self.query.trim().is_empty() {
                format!(
                    "  No matching models in {}. Switch to All models to search every provider.",
                    entry.label
                )
            } else if !self.query.trim().is_empty() {
                "  No matching models".to_owned()
            } else {
                "  No models available in this scope".to_owned()
            };
            lines.push(truncate_spans(vec![sp(msg, st.muted)], bw));
            shown = 1;
        } else {
            let window = &visible[start..end];
            let mut ctx_w = 0;
            let mut cost_w = 0;
            for r in window {
                if let Row::Model(m) = r {
                    let ctx = m
                        .context_window
                        .filter(|c| *c > 0)
                        .map(|c| format!("{} ◫", format_number(c)))
                        .unwrap_or_default();
                    ctx_w = ctx_w.max(width(&ctx));
                    cost_w = cost_w.max(width(&cost_pair(m)));
                }
            }
            let rw = bw.saturating_sub(1);
            let name_room = |c: usize, k: usize| {
                let n = usize::from(c > 0) + usize::from(k > 0);
                rw.saturating_sub(2 + c + k + 2 * n.saturating_sub(1))
            };
            if name_room(ctx_w, cost_w) < MIN_NAME_WIDTH {
                cost_w = 0;
            }
            if name_room(ctx_w, cost_w) < MIN_NAME_WIDTH {
                ctx_w = 0;
            }
            let bar = total > mv;
            let thumb = if bar {
                (mv * mv / total).clamp(1, mv)
            } else {
                0
            };
            let thumb_pos = if bar && total > mv {
                (start * (mv - thumb) + (total - mv) / 2) / (total - mv)
            } else {
                0
            };
            for (i, r) in window.iter().enumerate() {
                let mut line = match r {
                    Row::Separator => {
                        vec![
                            sp("  ", st.text),
                            sp("─".repeat(rw.saturating_sub(4)), st.muted),
                            sp("  ", st.text),
                        ]
                    }
                    Row::Model(m) => {
                        self.model_row(st, m, rw, start + i == self.selected, ctx_w, cost_w)
                    }
                };
                line = pad(truncate_spans(line, rw), rw);
                if bar {
                    if i >= thumb_pos && i < thumb_pos + thumb {
                        line.push(sp("█", st.accent));
                    } else {
                        line.push(sp("│", st.muted));
                    }
                }
                lines.push(line);
                shown += 1;
            }
        }
        while shown < mv {
            lines.push(Vec::new());
            shown += 1;
        }
        lines.push(Vec::new());
        let [d1, d2] = self.detail_lines(st, bw, theme);
        lines.push(d1);
        lines.push(d2);
        lines
    }

    fn roles_lines(
        &self,
        st: &St,
        bw: usize,
        entry: &Entry,
        layout: &mut Layout,
        list_abs_y: u16,
    ) -> Vec<Vec<Span<'static>>> {
        let (status, accent) = status_row_text(self, entry);
        let mut lines = vec![truncate_spans(
            vec![sp(status, if accent { st.accent } else { st.muted })],
            bw,
        )];
        lines.push(truncate_spans(
            vec![
                sp(" ", st.text),
                sp("Roles:", st.dim),
                sp(" ", st.text),
                sp("◀ all ▶", st.accent.add_modifier(Modifier::BOLD)),
                sp("  chat ┆ kinds", st.dim),
            ],
            bw,
        ));
        lines.push(Vec::new());
        layout.list_y = list_abs_y + 3;
        layout.list_n = ROLE_IDS.len();
        for (i, role) in ROLE_IDS.iter().enumerate() {
            let cursor = if self.focus_list && i == self.roles_index {
                sp("❯", st.accent)
            } else {
                sp(" ", st.text)
            };
            let value = match (self.role_selector(role, true), self.role_level(role)) {
                (Some(sel), Some(l)) => vec![sp(format!("{sel}:{l}"), st.text)],
                (Some(sel), None) => vec![sp(sel, st.text)],
                _ => vec![sp("auto", st.dim)],
            };
            let mut line = vec![
                cursor,
                sp(" ", st.text),
                sp(
                    format!("{:<10}", role_label(role)),
                    if i == self.roles_index {
                        st.accent
                    } else {
                        st.text
                    },
                ),
                sp("  ", st.text),
            ];
            line.extend(value);
            lines.push(truncate_spans(line, bw));
        }
        lines
    }

    fn locked_lines(&self, st: &St, bw: usize, entry: &Entry) -> Vec<Vec<Span<'static>>> {
        let (status, _) = status_row_text(self, entry);
        let count = self
            .models
            .iter()
            .filter(|m| m.provider == entry.label)
            .count();
        vec![
            truncate_spans(vec![sp(status, st.muted)], bw),
            Vec::new(),
            truncate_spans(
                vec![sp(
                    format!("  {} has no credentials configured.", entry.label),
                    st.muted,
                )],
                bw,
            ),
            truncate_spans(
                vec![sp(
                    format!("  Add an API key or login to use its {count} models."),
                    st.dim,
                )],
                bw,
            ),
        ]
    }

    fn strip_footer(
        &self,
        st: &St,
        strip: &Strip,
        fw: usize,
        layout: &mut Layout,
    ) -> Vec<Span<'static>> {
        let mut prefix: Vec<Span<'static>> = match strip {
            Strip::Role { model, .. } => vec![sp(model.id.clone(), st.accent), sp(" →", st.dim)],
            Strip::Thinking { model, role, .. } => vec![
                sp(role.to_lowercase(), st.muted),
                sp(format!(" · {} →", model.id), st.dim),
            ],
        };
        prefix.push(sp(" ", st.text));
        let chips = strip.chips();
        let index = strip.index();
        let body = |c: &Chip| -> (Vec<Span<'static>>, usize) {
            let mut v = vec![sp(" ", st.text)];
            if c.assigned {
                v.push(sp(format!("● {}", c.label), st.success));
                v.push(sp(" ✓", st.dim));
            } else {
                v.push(sp(c.label.clone(), st.muted));
            }
            v.push(sp(" ", st.text));
            let w = spans_width(&v);
            (v, w)
        };
        let avail = fw.saturating_sub(spans_width(&prefix)).max(1);
        let widths: Vec<usize> = chips
            .iter()
            .enumerate()
            .map(|(i, c)| body(c).1 + if i == index { 2 } else { 0 } + 1)
            .collect();
        let start_for = |target: usize| {
            let mut start = 0;
            while start < target {
                let sum: usize =
                    (if start > 0 { 2 } else { 0 }) + widths[start..=target].iter().sum::<usize>();
                if sum <= avail {
                    break;
                }
                start += 1;
            }
            start
        };
        let last = chips.len().saturating_sub(1);
        let mut start = start_for((index + 1).min(last));
        if start > index {
            start = start_for(index);
        }
        let mut line = prefix;
        let mut col = 2 + spans_width(&line);
        if start > 0 {
            line.push(sp("… ", st.dim));
            col += 2;
        }
        layout.chips.clear();
        for (i, c) in chips.iter().enumerate().skip(start) {
            let (mut spans, w) = body(c);
            let selected = i == index;
            let total_w = w + if selected { 2 } else { 0 };
            if selected {
                let mut s = vec![sp("[", st.accent)];
                s.append(&mut spans);
                s.push(sp("]", st.accent));
                if let Some(bg) = st.sel_bg {
                    s = s
                        .into_iter()
                        .map(|x| Span::styled(x.content, x.style.bg(bg)))
                        .collect();
                } else {
                    s = s
                        .into_iter()
                        .map(|x| Span::styled(x.content, x.style.add_modifier(Modifier::BOLD)))
                        .collect();
                }
                spans = s;
            }
            #[allow(clippy::cast_possible_truncation)]
            layout.chips.push((col as u16, (col + total_w) as u16, i));
            line.extend(spans);
            line.push(sp(" ", st.text));
            col += total_w + 1;
        }
        truncate_spans(line, fw)
    }
}

impl Component for ModelPicker {
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        frame.render_widget(Clear, area);
        let st = St::new(theme);
        let total_w = area.width as usize;
        let total_h = area.height as usize;
        if total_w < 12 || total_h < 5 {
            return;
        }
        let entries = self.entries();
        let longest = entries
            .iter()
            .map(|e| width(&e.label) + width(e.annotation.as_deref().unwrap_or("")) + 5)
            .max()
            .unwrap_or(0);
        let mut sw = longest.clamp(SIDEBAR_MIN, SIDEBAR_MAX);
        // Keep a usable body on narrow terminals.
        if total_w < sw + 7 + 28 {
            sw = total_w.saturating_sub(7 + 28).max(10);
        }
        let bw = total_w.saturating_sub(sw + 7).max(1);
        let content_rows = total_h.saturating_sub(4).max(1);
        let entry = self.active_entry();
        let roles_view = self.roles_view(&entry);
        let locked_view = self.locked_view(&entry);

        let mut layout = Layout {
            sidebar_x: area.x + 2,
            sidebar_w: sw as u16,
            body_x: area.x + (sw + 5) as u16,
            body_w: bw as u16,
            content_y: area.y + 1,
            content_rows: content_rows as u16,
            roles_view,
            footer_y: area.y + (total_h - 2) as u16,
            ..Layout::default()
        };

        // Sidebar viewport.
        let mut scroll = self.sidebar_scroll.get();
        if self.sidebar_follow.get() {
            let ai = entries
                .iter()
                .position(|e| e.id == self.active)
                .unwrap_or(0);
            if ai < scroll {
                scroll = ai;
            } else if ai >= scroll + content_rows {
                scroll = ai + 1 - content_rows;
            }
            self.sidebar_follow.set(false);
        }
        scroll = scroll.min(entries.len().saturating_sub(content_rows));
        self.sidebar_scroll.set(scroll);
        let searching = self.search_active();
        let mut sidebar: Vec<Vec<Span<'static>>> = Vec::new();
        for i in 0..content_rows {
            let idx = scroll + i;
            let Some(e) = entries.get(idx) else {
                layout.sidebar_rows.push(None);
                sidebar.push(Vec::new());
                continue;
            };
            layout.sidebar_rows.push(Some(idx));
            if e.kind == EntryKind::Separator {
                sidebar.push(vec![sp("─".repeat(sw), st.border)]);
                continue;
            }
            let active = e.id == self.active;
            let muted = e.locked
                || (searching
                    && (e.kind == EntryKind::Roles
                        || (e.kind == EntryKind::Provider
                            && e.annotation.as_deref() == Some("0"))));
            let (icon, icon_style) = match e.kind {
                EntryKind::Roles => ("✦", st.accent),
                EntryKind::All => ("⬢", st.accent),
                _ => {
                    if muted {
                        ("○", st.dim)
                    } else {
                        ("●", st.success)
                    }
                }
            };
            let label_style = if muted {
                st.dim
            } else if active {
                st.accent.add_modifier(Modifier::BOLD)
            } else {
                st.text
            };
            let cursor = if active && !self.focus_list {
                sp("❯", st.accent)
            } else {
                sp(" ", st.text)
            };
            let left = vec![
                cursor,
                sp(" ", st.text),
                sp(icon, if muted { st.dim } else { icon_style }),
                sp(" ", st.text),
                sp(e.label.clone(), label_style),
            ];
            let ann = e.annotation.clone().unwrap_or_default();
            let lw = spans_width(&left);
            let aw = width(&ann);
            let line = if lw + aw < sw {
                let mut l = left;
                l.push(sp(" ".repeat(sw - lw - aw), st.text));
                l.push(sp(ann, st.dim));
                l
            } else {
                pad(truncate_spans(left, sw), sw)
            };
            sidebar.push(line);
        }

        // Body.
        let list_abs_y = area.y + 1;
        let mut body = if roles_view {
            self.roles_lines(&st, bw, &entry, &mut layout, list_abs_y)
        } else if locked_view {
            self.locked_lines(&st, bw, &entry)
        } else {
            self.browser_lines(
                &st,
                bw,
                content_rows,
                &entry,
                &mut layout,
                list_abs_y,
                theme,
            )
        };
        body.resize(content_rows, Vec::new());

        // Footer.
        let footer = if let Some(strip) = &self.strip {
            self.strip_footer(&st, strip, total_w.saturating_sub(4), &mut layout)
        } else {
            layout.chips.clear();
            truncate_spans(
                vec![sp(self.footer_hint(&entry), st.dim)],
                total_w.saturating_sub(4),
            )
        };

        // Assemble screen lines.
        let mut screen: Vec<Vec<Span<'static>>> = Vec::with_capacity(total_h);
        let junction = 3 + sw;
        let fill = |left: char, mid: char, right: char, jc: char, title: &str| {
            let mut s = String::new();
            let mut col = 0;
            let mut chars: Vec<char> = Vec::new();
            for i in 0..total_w {
                chars.push(if i == 0 {
                    left
                } else if i == total_w - 1 {
                    right
                } else if i == junction {
                    jc
                } else {
                    mid
                });
            }
            if !title.is_empty() {
                let t = format!(" {title} ");
                for (k, ch) in t.chars().enumerate() {
                    if 2 + k < junction {
                        chars[2 + k] = ch;
                    }
                }
            }
            for ch in chars {
                s.push(ch);
                col += 1;
            }
            let _ = col;
            s
        };
        screen.push(vec![sp(fill('╭', '─', '╮', '┬', "Models"), st.border)]);
        for i in 0..content_rows {
            let mut line = vec![sp("│ ", st.border)];
            line.extend(pad(sidebar[i].clone(), sw));
            line.push(sp(" │ ", st.border));
            line.extend(pad(truncate_spans(body[i].clone(), bw), bw));
            line.push(sp(" │", st.border));
            screen.push(line);
        }
        screen.push(vec![sp(fill('├', '─', '┤', '┴', ""), st.border)]);
        let mut frow = vec![sp("│ ", st.border)];
        frow.extend(pad(footer, total_w.saturating_sub(4)));
        frow.push(sp(" │", st.border));
        screen.push(frow);
        screen.push(vec![sp(fill('╰', '─', '╯', '─', ""), st.border)]);

        for (i, spans) in screen.into_iter().enumerate().take(total_h) {
            let r = Rect {
                x: area.x,
                y: area.y + i as u16,
                width: area.width,
                height: 1,
            };
            frame.render_widget(Paragraph::new(Line::from(spans)), r);
        }
        *self.layout.borrow_mut() = layout;
        let _ = (&st.warning, &st.error);
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        ModelPicker::handle_key(self, key)
    }

    fn handle_mouse(&mut self, event: &SgrMouseEvent) -> EventOutcome {
        ModelPicker::handle_mouse(self, event)
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
        let mut locked = ModelEntry::new("anthropic", "claude-x", Some(200_000));
        locked.available = false;
        vec![
            ModelEntry::new("zai", "glm-5.3-flash", Some(200_000)),
            ModelEntry::new("zai", "glm-5.3", Some(200_000)),
            ModelEntry::new("openai", "gpt-5.2", Some(400_000)),
            locked,
        ]
    }

    fn picker() -> ModelPicker {
        let mut p = ModelPicker::default();
        p.models = catalog();
        p
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ch(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn click(col: u16, row: u16) -> SgrMouseEvent {
        SgrMouseEvent {
            button: 0,
            col,
            row,
            release: false,
            wheel: None,
            motion: false,
            left_click: true,
        }
    }

    fn draw(p: &ModelPicker, w: u16, h: u16) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        let theme = Theme::default();
        terminal
            .draw(|f| Component::render(p, f, f.area(), &theme))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..h)
            .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect())
            .collect()
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
    fn filters_case_insensitive_and_skips_unconfigured() {
        let mut p = picker();
        for c in "GLM-5.3-F".chars() {
            p.handle_key(ch(c));
        }
        let rows = p.rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].display(), "zai/glm-5.3-flash");
        p.query = "claude".into();
        assert!(
            p.rows().is_empty(),
            "unconfigured providers are not selectable"
        );
    }

    #[test]
    fn enter_then_default_chip_switches_model_and_persists_role() {
        let mut p = picker();
        p.on_open();
        p.role_map.clear();
        p.handle_key(key(KeyCode::Enter)); // sidebar -> list focus
        p.handle_key(key(KeyCode::Down));
        p.handle_key(key(KeyCode::Enter)); // open role strip
        assert!(p.strip.is_some());
        p.handle_key(key(KeyCode::Enter)); // default chip
        let actions = p.take_actions();
        assert!(actions.contains(&Action::SetModel {
            provider: p
                .current
                .as_deref()
                .unwrap()
                .split('/')
                .next()
                .unwrap()
                .into(),
            id: p
                .current
                .as_deref()
                .unwrap()
                .split_once('/')
                .unwrap()
                .1
                .into(),
        }));
        assert!(matches!(
            actions.last(),
            Some(Action::SettingChanged { id, .. }) if id == "modelRoles"
        ));
    }

    #[test]
    fn non_default_role_does_not_switch_live_model() {
        let mut p = picker();
        p.role_map.clear();
        p.focus_list = true;
        p.handle_key(key(KeyCode::Enter));
        p.handle_key(key(KeyCode::Right)); // smol
        p.handle_key(key(KeyCode::Enter));
        let actions = p.take_actions();
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], Action::SettingChanged { value, .. }
            if value["smol"].is_string()));
    }

    #[test]
    fn provider_scope_and_search_counts() {
        let mut p = picker();
        p.move_sidebar(1); // all -> first configured provider (separator skipped)
        let entries = p.entries();
        assert!(entries.iter().any(|e| e.label == "anthropic" && e.locked));
        assert_eq!(p.active_entry().label, "openai");
        assert_eq!(p.rows().len(), 1);
    }

    #[test]
    fn esc_ladder_clears_query_then_closes() {
        let mut p = picker();
        p.handle_key(ch('g'));
        assert!(p.handle_key(key(KeyCode::Esc)).consumed());
        assert!(p.query.is_empty());
        assert!(!p.handle_key(key(KeyCode::Esc)).consumed());
    }

    #[test]
    fn renders_fullscreen_two_pane_frame() {
        let mut p = picker();
        p.current = Some("zai/glm-5.3".into());
        p.on_open();
        let rows = draw(&p, 120, 40);
        assert!(rows[0].starts_with("╭─ Models "), "{}", rows[0]);
        assert!(rows[0].contains('┬'));
        assert!(rows[39].starts_with('╰'));
        let all = rows.join("\n");
        assert!(all.contains("Roles"));
        assert!(all.contains("All models"));
        assert!(all.contains("zai/glm-5.3"));
        assert!(all.contains("Kind:"));
        assert!(all.contains("⏎/→ models · ↑/↓ providers"));
        // Unconfigured provider listed (locked) in the sidebar.
        assert!(all.contains("anthropic"));
    }

    #[test]
    fn mouse_click_selects_then_activates_and_chip_click_assigns() {
        let mut p = picker();
        p.role_map.clear();
        p.current = Some("zai/glm-5.3-flash".into());
        p.on_open();
        let _ = draw(&p, 120, 40);
        let l = p.layout.borrow().clone();
        // Rows: current model, separator, then the rest — click the third.
        let row = l.list_y + 2;
        p.handle_mouse(&click(l.body_x + 4, row));
        assert_eq!(p.selected, 2);
        p.handle_mouse(&click(l.body_x + 4, row)); // click again = activate
        assert!(p.strip.is_some());
        let _ = draw(&p, 120, 40);
        let l = p.layout.borrow().clone();
        let (x0, _, _) = l.chips[0];
        p.handle_mouse(&click(x0 + 1, l.footer_y));
        let actions = p.take_actions();
        assert!(actions.iter().any(|a| matches!(a, Action::SetModel { .. })));
    }

    #[test]
    fn wheel_scrolls_list_window() {
        let mut p = ModelPicker::default();
        p.models = (0..100)
            .map(|i| ModelEntry::new("p", &format!("m{i:03}"), Some(1000)))
            .collect();
        p.on_open();
        let _ = draw(&p, 120, 40);
        let l = p.layout.borrow().clone();
        let ev = SgrMouseEvent {
            wheel: Some(1),
            left_click: false,
            ..click(l.body_x + 3, l.list_y)
        };
        p.handle_mouse(&ev);
        p.handle_mouse(&ev);
        assert_eq!(p.list_start.get(), 2);
        let rows = draw(&p, 120, 40);
        assert!(rows.iter().any(|r| r.contains("p/m002")));
        assert!(!rows.iter().any(|r| r.contains("p/m000")));
    }
}
