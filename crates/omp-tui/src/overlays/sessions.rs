//! Session picker — ports of `pi-tui/src/overlays/session-selector.ts`
//! (spec sec.21) and `pi-tui/src/apps/session-picker.ts` (sec.35).
//!
//! Lists sessions from the sessions dir (newest first, title + relative
//! time + cwd), fuzzy-filters as you type, and resolves to a picked session
//! file. Supports two scopes (current folder ↔ all projects, `tab`), a
//! two-step delete confirmation, and both embedding shapes: a centered
//! overlay ([`Component`]) and a standalone fullscreen picker
//! ([`select_session_standalone`]) for bare `omp --resume`.
//!
//! The list scan is lazy the way the TS picker is: per `*.jsonl` only the
//! first two lines are read — the fixed-width title slot (line 1) and the
//! `session` header (line 2, for id/cwd) — so a directory of large journals
//! costs O(files), not O(entries).

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph};
use ratatui::Frame;

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::theme::Theme;

/// One row of the session list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSummary {
    /// Session id (header id, else the file stem).
    pub id: String,
    /// Title from the fixed-width title slot, when the session has one.
    pub title: Option<String>,
    /// Working directory the session was started in.
    pub cwd: String,
    /// The journal file backing this session.
    pub path: PathBuf,
    /// Last modification (file mtime).
    pub modified: SystemTime,
    /// Pinned sessions sort first (reserved for the pinned-ids host hook).
    pub pinned: bool,
    /// Origin tag of the store the file came from (`None` = the native
    /// omp-rs store; e.g. `Some("ts")` for TS omp's `~/.omp/agent/sessions`).
    pub source: Option<String>,
}

/// One session store the picker lists: a per-project session dir plus an
/// optional origin tag shown on its rows. The all-projects scope scans every
/// sibling dir under the same root with the same tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSource {
    /// The current folder's session dir in this store.
    pub dir: PathBuf,
    /// Origin tag for rows from this store (`None` = native, untagged).
    pub label: Option<String>,
}

impl SessionSource {
    /// A native (untagged) store.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            label: None,
        }
    }

    /// A store whose rows carry `label`.
    #[must_use]
    pub fn labeled(dir: impl Into<PathBuf>, label: &str) -> Self {
        Self {
            dir: dir.into(),
            label: Some(label.to_owned()),
        }
    }
}

impl SessionSummary {
    /// The display title, falling back like TS (`(untitled)`).
    fn display_title(&self) -> &str {
        self.title.as_deref().unwrap_or("(untitled)")
    }
}

/// The picker's resolution, drained by the host via [`SessionPicker::take_pick`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionPick {
    /// Open this session file.
    Resume(PathBuf),
    /// The user cancelled (Esc, or Ctrl+C in the standalone picker).
    Cancelled,
}

/// Which sessions the picker lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SessionScope {
    /// Only the current folder's session dir.
    #[default]
    CurrentFolder,
    /// Every project dir under the sessions root.
    AllProjects,
}

/// Scan one session dir, returning summaries sorted newest-first.
///
/// Unreadable or header-less files are skipped silently (the TS picker
/// treats a broken file as "no session").
#[must_use]
pub fn scan_session_dir(dir: &Path) -> Vec<SessionSummary> {
    let mut out = Vec::new();
    let read_dir = match fs::read_dir(dir) {
        Ok(d) => d,
        Err(_) => return out,
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "jsonl") {
            continue;
        }
        if let Some(summary) = scan_session_file(&path) {
            out.push(summary);
        }
    }
    out.sort_by(|a, b| b.modified.cmp(&a.modified));
    out
}

/// Read just the title slot + header of one session file.
fn scan_session_file(path: &Path) -> Option<SessionSummary> {
    let file = fs::File::open(path).ok()?;
    let mut reader = BufReader::with_capacity(1024, file);
    let mut title = None;
    let mut id = None;
    let mut cwd = String::new();
    let mut line = String::new();
    for n in 0..2 {
        line.clear();
        if reader.read_line(&mut line).ok()? == 0 {
            break;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
            continue;
        };
        match value.get("type").and_then(serde_json::Value::as_str) {
            Some("title") if n == 0 => {
                title = value
                    .get("title")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
            }
            Some("session") => {
                id = value
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
                cwd = value
                    .get("cwd")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
            }
            _ => {}
        }
    }
    let id = id.or_else(|| {
        path.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
    })?;
    let modified = fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    Some(SessionSummary {
        id,
        title,
        cwd,
        path: path.to_path_buf(),
        modified,
        pinned: false,
        source: None,
    })
}

/// Scan one store's dir, tagging every row with the store's label.
fn scan_source(src: &SessionSource) -> Vec<SessionSummary> {
    let mut rows = scan_session_dir(&src.dir);
    if src.label.is_some() {
        for row in &mut rows {
            row.source.clone_from(&src.label);
        }
    }
    rows
}

/// Subsequence fuzzy score of `query` against `candidate` (both lowercased
/// by the caller), mirroring `rankSessionSearchMatches` loosely: matches
/// score higher when they start early and run consecutively.
fn fuzzy_score(query: &str, candidate: &str) -> Option<usize> {
    let q: Vec<char> = query.chars().collect();
    let c: Vec<char> = candidate.chars().collect();
    let mut score = 0usize;
    let mut qi = 0usize;
    let mut run = 0usize;
    let mut first = None;
    for (ci, &ch) in c.iter().enumerate() {
        if qi < q.len() && ch == q[qi] {
            if first.is_none() {
                first = Some(ci);
            }
            run += 1;
            score += 10 + run * 5 + (q.len().saturating_sub(qi));
            qi += 1;
        } else {
            run = 0;
        }
    }
    if qi < q.len() {
        return None;
    }
    // Prefix matches rank above mid-string scattered ones.
    if first == Some(0) {
        score += 50;
    }
    Some(score)
}

/// `format!("{} ago")`-style relative time, matching the TS picker strings.
fn relative_time(t: SystemTime) -> String {
    let Ok(age) = SystemTime::now().duration_since(t) else {
        return "just now".into();
    };
    if age < Duration::from_secs(45) {
        "just now".into()
    } else if age < Duration::from_secs(90) {
        "1m ago".into()
    } else if age < Duration::from_secs(3600) {
        format!("{}m ago", age.as_secs() / 60)
    } else if age < Duration::from_secs(5400) {
        "1h ago".into()
    } else if age < Duration::from_secs(86_400) {
        format!("{}h ago", age.as_secs() / 3600)
    } else if age < Duration::from_secs(1_209_600) {
        format!("{}d ago", age.as_secs() / 86_400)
    } else {
        format!("{}w ago", age.as_secs() / 1_209_600)
    }
}

/// Calendar date `YYYY-MM-DD` for a unix time (civil-from-days, no chrono).
#[cfg(test)]
fn civil_date(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let y = era * 400 + (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let yo = y - era * 400;
    let doy = doe - (365 * yo + yo / 4 - yo / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// The session picker overlay (`SessionSelectorComponent`).
#[derive(Clone)]
pub struct SessionPicker {
    folders: Vec<SessionSource>,
    all_dirs: Vec<SessionSource>,
    scope: SessionScope,
    sessions: Vec<SessionSummary>,
    /// Indices into `sessions`, filtered + ranked by the current query.
    filtered: Vec<usize>,
    query: String,
    selected: usize,
    offset: usize,
    list_height: usize,
    confirm_delete: bool,
    standalone: bool,
    standalone_title: String,
    pending: Option<SessionPick>,
    /// Allow deleting sessions (TS `allowDelete`, default on).
    pub allow_delete: bool,
    /// Allow the all-projects scope (TS `allowGlobalScope`, default on).
    pub allow_global_scope: bool,
}

impl std::fmt::Debug for SessionPicker {
    // Hand-written to skip noisy fields if needed later; keeps derive on
    // SessionSummary useful in tests.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionPicker")
            .field("scope", &self.scope)
            .field("sessions", &self.sessions.len())
            .field("query", &self.query)
            .field("selected", &self.selected)
            .finish()
    }
}

impl Default for SessionPicker {
    fn default() -> Self {
        Self {
            folders: Vec::new(),
            all_dirs: Vec::new(),
            scope: SessionScope::CurrentFolder,
            sessions: Vec::new(),
            filtered: Vec::new(),
            query: String::new(),
            selected: 0,
            offset: 0,
            list_height: 10,
            confirm_delete: false,
            standalone: false,
            standalone_title: String::new(),
            pending: None,
            allow_delete: true,
            allow_global_scope: true,
        }
    }
}

impl SessionPicker {
    /// Build a picker listing the sessions in `dir` (the current folder's
    /// session dir), newest first.
    #[must_use]
    pub fn load(dir: &Path) -> Self {
        Self::load_sources(&[SessionSource::new(dir)])
    }

    /// Build a picker over several stores (the native omp-rs dir plus TS
    /// omp's dir for the same folder), merged newest first.
    #[must_use]
    pub fn load_sources(sources: &[SessionSource]) -> Self {
        let mut picker = Self {
            folders: sources
                .iter()
                .filter(|s| !s.dir.as_os_str().is_empty())
                .cloned()
                .collect(),
            ..Default::default()
        };
        picker.rescan();
        picker
    }

    /// Render fullscreen (standalone `--resume` mode) instead of a centered
    /// popup (embedded `/resume` overlay).
    pub fn set_standalone(&mut self, standalone: bool) {
        self.standalone = standalone;
    }

    /// Window title for the standalone picker (TS `options.title`).
    pub fn set_title(&mut self, title: &str) {
        self.standalone_title = title.to_owned();
    }

    /// The picker resolution, if the user confirmed or cancelled.
    #[must_use]
    pub fn pick(&self) -> Option<&SessionPick> {
        self.pending.as_ref()
    }

    /// Drain the picker resolution (host consumes it once).
    pub fn take_pick(&mut self) -> Option<SessionPick> {
        self.pending.take()
    }

    /// Whether a modal question (delete confirmation) is showing.
    #[must_use]
    pub fn confirming(&self) -> bool {
        self.confirm_delete
    }

    /// The currently selected session, if any.
    #[must_use]
    pub fn selected_session(&self) -> Option<&SessionSummary> {
        self.filtered.get(self.selected).map(|&i| &self.sessions[i])
    }

    /// Rescan the session dir(s) for the current scope and re-apply the filter.
    pub fn rescan(&mut self) {
        let dirs: &[SessionSource] = match self.scope {
            SessionScope::CurrentFolder => &self.folders,
            SessionScope::AllProjects => &self.all_dirs,
        };
        self.sessions = dirs.iter().flat_map(scan_source).collect();
        self.sessions.sort_by(|a, b| b.modified.cmp(&a.modified));
        self.refilter();
    }

    /// Discover every project dir under each store's sessions root (lazy:
    /// only on the first `tab`). A folder dir's parent is its store's
    /// per-project root (`…/sessions/`); rows keep the store's tag.
    fn ensure_all_dirs(&mut self) {
        if !self.all_dirs.is_empty() || self.folders.is_empty() {
            return;
        }
        let mut dirs: Vec<SessionSource> = Vec::new();
        for folder in &self.folders {
            let mut found = Vec::new();
            if let Some(root) = folder.dir.parent() {
                if let Ok(read) = fs::read_dir(root) {
                    for entry in read.flatten() {
                        let p = entry.path();
                        if p.is_dir() {
                            found.push(p);
                        }
                    }
                }
            }
            if !found.contains(&folder.dir) {
                found.push(folder.dir.clone());
            }
            found.sort();
            for dir in found {
                if !dirs.iter().any(|d| d.dir == dir) {
                    dirs.push(SessionSource {
                        dir,
                        label: folder.label.clone(),
                    });
                }
            }
        }
        self.all_dirs = dirs;
    }

    /// Re-run the filter over `sessions`, preserving the query.
    fn refilter(&mut self) {
        self.filtered = if self.query.is_empty() {
            (0..self.sessions.len()).collect()
        } else {
            let query = self.query.to_lowercase();
            let mut scored: Vec<(usize, usize)> = self
                .sessions
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    let hay = format!("{} {} {}", s.display_title(), s.cwd, s.id).to_lowercase();
                    fuzzy_score(&query, &hay).map(|sc| (i, sc))
                })
                .collect();
            scored.sort_by(|a, b| {
                b.1.cmp(&a.1).then_with(|| {
                    self.sessions[b.0]
                        .modified
                        .cmp(&self.sessions[a.0].modified)
                })
            });
            scored.into_iter().map(|(i, _)| i).collect()
        };
        self.selected = 0;
        self.offset = 0;
    }

    /// Whether the selected row may be deleted: deletion enabled and the row
    /// belongs to the native store (foreign stores such as TS omp's are
    /// read-only from here; no notice mechanism exists, so it is silent).
    fn can_delete_selected(&self) -> bool {
        self.allow_delete
            && self
                .filtered
                .get(self.selected)
                .is_some_and(|&i| self.sessions[i].source.is_none())
    }

    /// Delete the selected session's file and drop it from the list.
    fn delete_selected(&mut self) {
        let Some(&i) = self.filtered.get(self.selected) else {
            return;
        };
        let path = self.sessions[i].path.clone();
        if fs::remove_file(&path).is_ok() {
            self.sessions.remove(i);
            self.refilter();
        }
        self.selected = self.selected.min(self.filtered.len().saturating_sub(1));
    }

    /// Toggle current-folder <-> all-projects scope (TS `tab`).
    fn toggle_scope(&mut self) {
        if !self.allow_global_scope {
            return;
        }
        self.scope = match self.scope {
            SessionScope::CurrentFolder => {
                self.ensure_all_dirs();
                SessionScope::AllProjects
            }
            SessionScope::AllProjects => SessionScope::CurrentFolder,
        };
        self.confirm_delete = false;
        self.rescan();
    }

    fn move_selection(&mut self, delta: isize) {
        if self.filtered.is_empty() {
            return;
        }
        let len = self.filtered.len() as isize;
        self.selected = (self.selected as isize + delta).clamp(0, len - 1) as usize;
    }

    /// The scope-toggle label in the footer hint (TS: `<all projects|current folder>`).
    fn scope_hint(&self) -> &'static str {
        match self.scope {
            SessionScope::CurrentFolder => "all projects",
            SessionScope::AllProjects => "current folder",
        }
    }
}

impl SessionPicker {
    /// One rendered row: `title  <relative time>  <cwd dim>` (TS row shape),
    /// plus a `[tag]` for sessions from a non-native store.
    fn row_spans<'a>(&self, s: &'a SessionSummary, theme: &Theme) -> Vec<Span<'a>> {
        let muted = Style::default().fg(theme.color("text_muted"));
        let mut spans = vec![
            Span::styled(s.display_title().to_owned(), Style::default()),
            Span::raw("  "),
            Span::styled(relative_time(s.modified), muted),
            Span::raw("  "),
            Span::styled(s.cwd.clone(), muted),
        ];
        if let Some(tag) = &s.source {
            spans.push(Span::raw("  "));
            spans.push(Span::styled(format!("[{tag}]"), muted));
        }
        spans
    }

    /// The footer hint (TS: `[delete/backspace delete · enter select · tab <scope> · esc cancel]`).
    fn hint_line(&self, theme: &Theme) -> Line<'static> {
        let dim = Style::default().fg(theme.color("text_muted"));
        let key = Style::default()
            .fg(theme.color("accent"))
            .add_modifier(Modifier::BOLD);
        let mut spans = vec![
            Span::styled("[", dim),
            Span::styled("delete/backspace", key),
            Span::styled(" delete · ", dim),
            Span::styled("enter", key),
            Span::styled(" select · ", dim),
        ];
        if self.allow_global_scope {
            spans.extend([
                Span::styled("tab", key),
                Span::styled(format!(" {} · ", self.scope_hint()), dim),
            ]);
        }
        spans.extend([Span::styled("esc", key), Span::styled(" cancel]", dim)]);
        Line::from(spans)
    }
}

impl Component for SessionPicker {
    fn wants_focus(&self) -> bool {
        self.pending.is_none()
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        if self.pending.is_some() {
            return EventOutcome::Consumed;
        }
        // Ctrl+C cancels, matching the standalone picker's process exit.
        if key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('c') {
            self.pending = Some(SessionPick::Cancelled);
            return EventOutcome::Consumed;
        }
        match key.code {
            KeyCode::Esc => {
                if self.confirm_delete {
                    self.confirm_delete = false;
                } else {
                    self.pending = Some(SessionPick::Cancelled);
                }
            }
            KeyCode::Enter => {
                if self.confirm_delete {
                    self.delete_selected();
                    self.confirm_delete = false;
                } else if let Some(s) = self.selected_session() {
                    self.pending = Some(SessionPick::Resume(s.path.clone()));
                }
            }
            KeyCode::Up => self.move_selection(-1),
            KeyCode::Down => self.move_selection(1),
            KeyCode::PageUp => {
                let h = self.list_height.max(1) as isize;
                self.move_selection(-h);
            }
            KeyCode::PageDown => {
                let h = self.list_height.max(1) as isize;
                self.move_selection(h);
            }
            KeyCode::Home => self.selected = 0,
            KeyCode::End => {
                self.selected = self.filtered.len().saturating_sub(1);
            }
            KeyCode::Tab | KeyCode::BackTab => self.toggle_scope(),
            KeyCode::Delete if self.can_delete_selected() => {
                self.confirm_delete = true;
            }
            KeyCode::Delete => {}
            KeyCode::Backspace => {
                if self.query.is_empty() {
                    if self.can_delete_selected() {
                        self.confirm_delete = true;
                    }
                } else {
                    self.query.pop();
                    self.refilter();
                }
            }
            KeyCode::Char(c)
                if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT =>
            {
                self.query.push(c);
                self.refilter();
            }
            _ => return EventOutcome::Ignored,
        }
        EventOutcome::Consumed
    }

    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let popup = if self.standalone {
            area
        } else {
            super::centered(area, 70, 70)
        };
        frame.render_widget(Clear, popup);
        let border = Style::default().fg(theme.color("border"));
        let title = if self.standalone && !self.standalone_title.is_empty() {
            self.standalone_title.clone()
        } else {
            match self.scope {
                SessionScope::CurrentFolder => " sessions ".to_owned(),
                SessionScope::AllProjects => " sessions (all projects) ".to_owned(),
            }
        };
        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(border);
        let inner = block.inner(popup);
        frame.render_widget(block, popup);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // filter query
                Constraint::Min(1),    // list / confirm
                Constraint::Length(1), // hint
            ])
            .split(inner);

        // Filter query line (`❯ query`).
        let prompt = theme.glyph("prompt");
        let query_line = if self.query.is_empty() {
            Line::from(Span::styled(
                format!("{prompt} "),
                Style::default().fg(theme.color("text_muted")),
            ))
        } else {
            Line::from(vec![
                Span::styled(
                    format!("{prompt} "),
                    Style::default().fg(theme.color("accent")),
                ),
                Span::styled(self.query.clone(), Style::default().fg(theme.color("text"))),
            ])
        };
        frame.render_widget(Paragraph::new(query_line), chunks[0]);

        let list_area = chunks[1];
        let height = list_area.height.max(1) as usize;

        if self.confirm_delete {
            // Two-step confirm replaces the list in the same slot (TS 1346).
            let label = self
                .selected_session()
                .map(|s| format!("Delete session \"{}\"?", s.display_title()))
                .unwrap_or_else(|| "Delete session?".into());
            let line = Line::from(vec![
                Span::styled(label, Style::default().fg(theme.color("warning"))),
                Span::raw("  "),
                Span::styled("[", Style::default().fg(theme.color("text_muted"))),
                Span::styled(
                    "enter confirm · esc cancel",
                    Style::default()
                        .fg(theme.color("accent"))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("]", Style::default().fg(theme.color("text_muted"))),
            ]);
            frame.render_widget(Paragraph::new(line), list_area);
        } else if self.filtered.is_empty() {
            let msg = if self.query.is_empty()
                && self.scope == SessionScope::CurrentFolder
                && self.allow_global_scope
            {
                "No sessions in current folder. Press tab to view all."
            } else {
                "No matching sessions."
            };
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    msg,
                    Style::default().fg(theme.color("text_muted")),
                ))),
                list_area,
            );
        } else {
            let selected_style = Style::default()
                .fg(theme.color("selected"))
                .add_modifier(Modifier::BOLD);
            // Scroll state is render-derived: compute the window without
            // mutating `self` (Component::render takes &self).
            let mut offset = self.offset.min(self.selected);
            if self.selected >= offset + height {
                offset = self.selected + 1 - height;
            }
            let items: Vec<ListItem<'_>> = self
                .filtered
                .iter()
                .enumerate()
                .skip(offset)
                .take(height)
                .map(|(row, &i)| {
                    let item = ListItem::new(Line::from(self.row_spans(&self.sessions[i], theme)));
                    if row == self.selected {
                        item.style(selected_style)
                    } else {
                        item
                    }
                })
                .collect();
            frame.render_widget(List::new(items), list_area);
        }

        frame.render_widget(Paragraph::new(self.hint_line(theme)), chunks[2]);
    }
}

/// Run the standalone fullscreen picker (bare `omp --resume`) and return the
/// chosen session file, or `None` if cancelled.
///
/// Owns the terminal lifecycle the way TS `runStandaloneTui` does: raw mode +
/// alternate screen for the picker's lifetime, restored on drop (including
/// panics); Ctrl+C resolves `None` like the TS `process.exit(0)` path.
pub fn select_session_standalone(sources: &[SessionSource], title: &str) -> Option<PathBuf> {
    use crossterm::event::{self, Event};
    use crossterm::terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    };
    use crossterm::ExecutableCommand;
    use ratatui::backend::CrosstermBackend;
    use std::io::stdout;

    struct TerminalGuard;
    impl Drop for TerminalGuard {
        fn drop(&mut self) {
            let _ = disable_raw_mode();
            let _ = std::io::stdout().execute(LeaveAlternateScreen);
        }
    }

    if enable_raw_mode().is_err() {
        return None;
    }
    if std::io::stdout().execute(EnterAlternateScreen).is_err() {
        let _ = disable_raw_mode();
        return None;
    }
    let _guard = TerminalGuard;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = ratatui::Terminal::new(backend).ok()?;

    let mut picker = SessionPicker::load_sources(sources);
    picker.set_standalone(true);
    picker.set_title(title);
    let theme = Theme::default();
    loop {
        if terminal
            .draw(|f| {
                let area = f.area();
                picker.render(f, area, &theme);
            })
            .is_err()
        {
            return None;
        }
        let Ok(Event::Key(key)) = event::read() else {
            continue; // resize/mouse: redraw-only
        };
        picker.handle_key(key);
        if let Some(pick) = picker.take_pick() {
            return match pick {
                SessionPick::Resume(path) => Some(path),
                SessionPick::Cancelled => None,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

    /// A fresh temp dir for fixtures (no tempfile dep).
    fn fixture_root() -> PathBuf {
        let n = FIXTURE_ID.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("omp-tui-sessions-test-{pid}-{n}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Write a minimal session file: title slot line + session header line.
    fn write_session(dir: &Path, id: &str, title: &str, cwd: &str) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let path = dir.join(format!("{id}.jsonl"));
        let mut f = fs::File::create(&path).unwrap();
        writeln!(f, "{{\"type\":\"title\",\"v\":1,\"title\":\"{title}\",\"updated_at\":\"x\",\"pad\":\"\"}}").unwrap();
        writeln!(f, "{{\"type\":\"session\",\"version\":3,\"id\":\"{id}\",\"timestamp\":\"2026-10-10T10:00:00Z\",\"cwd\":\"{cwd}\"}}").unwrap();
        path
    }

    /// Pushes mtime 60 s into the future so recency ordering is deterministic
    /// (appending alone can land in the same mtime tick under load).
    fn bump_mtime(path: &Path) {
        let f = fs::OpenOptions::new().append(true).open(path).unwrap();
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
        f.set_modified(later).unwrap();
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn char_key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn text_to_string(buffer: &ratatui::buffer::Buffer) -> String {
        let area = buffer.area;
        let mut out = String::new();
        for y in 0..area.height {
            for x in 0..area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn scan_reads_title_cwd_and_sorts_newest_first() {
        let root = fixture_root();
        let dir = root.join("proj");
        write_session(&dir, "aaa", "Alpha task", "/work/a");
        let bbb = write_session(&dir, "bbb", "Beta task", "/work/b");
        bump_mtime(&bbb);

        let sessions = scan_session_dir(&dir);
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].id, "bbb");
        assert_eq!(sessions[0].title.as_deref(), Some("Beta task"));
        assert_eq!(sessions[0].cwd, "/work/b");
        assert_eq!(sessions[1].id, "aaa");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn typing_filters_and_enter_resumes() {
        let root = fixture_root();
        let dir = root.join("proj");
        let alpha = write_session(&dir, "aaa", "Alpha task", "/work/a");
        write_session(&dir, "bbb", "Beta task", "/work/b");

        let mut picker = SessionPicker::load(&dir);
        assert_eq!(picker.filtered.len(), 2);

        for c in "alpha".chars() {
            assert!(picker.handle_key(char_key(c)).consumed());
        }
        assert_eq!(picker.filtered.len(), 1);
        assert_eq!(picker.selected_session().unwrap().id, "aaa");

        assert!(picker.handle_key(key(KeyCode::Enter)).consumed());
        assert_eq!(picker.take_pick(), Some(SessionPick::Resume(alpha)));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn esc_cancels_and_ctrl_c_cancels() {
        let root = fixture_root();
        let dir = root.join("proj");
        write_session(&dir, "aaa", "Alpha", "/w");
        let mut picker = SessionPicker::load(&dir);
        picker.handle_key(key(KeyCode::Esc));
        assert_eq!(picker.take_pick(), Some(SessionPick::Cancelled));

        let mut picker = SessionPicker::load(&dir);
        picker.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert_eq!(picker.take_pick(), Some(SessionPick::Cancelled));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_is_a_two_step_confirm() {
        let root = fixture_root();
        let dir = root.join("proj");
        let aaa = write_session(&dir, "aaa", "Alpha", "/w");
        let bbb = write_session(&dir, "bbb", "Beta", "/w");
        bump_mtime(&aaa); // aaa strictly newer -> row 0; Down lands on bbb.

        let mut picker = SessionPicker::load(&dir);
        picker.handle_key(key(KeyCode::Down));
        assert_eq!(picker.selected_session().unwrap().id, "bbb");

        picker.handle_key(key(KeyCode::Delete));
        assert!(picker.confirming());
        // Esc aborts the confirm, session survives.
        picker.handle_key(key(KeyCode::Esc));
        assert!(!picker.confirming());
        assert!(bbb.exists());

        // Backspace on an empty query also requests delete.
        picker.handle_key(key(KeyCode::Backspace));
        assert!(picker.confirming());
        picker.handle_key(key(KeyCode::Enter));
        assert!(!picker.confirming());
        assert!(!bbb.exists());
        assert_eq!(picker.sessions.len(), 1);
        assert_eq!(picker.selected, 0);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn foreign_store_rows_are_not_deletable() {
        let root = fixture_root();
        let dir = root.join("proj");
        let ts = write_session(&dir, "tsrow", "Foreign", "/w");

        let mut picker = SessionPicker::load(&dir);
        picker.sessions[0].source = Some("ts".into());
        picker.handle_key(key(KeyCode::Delete));
        assert!(!picker.confirming());
        picker.handle_key(key(KeyCode::Backspace));
        assert!(!picker.confirming());
        picker.handle_key(key(KeyCode::Enter));
        assert!(ts.exists());
        assert_eq!(picker.sessions.len(), 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn tab_toggles_current_folder_and_all_projects() {
        let root = fixture_root();
        let dir_a = root.join("proj-a");
        let dir_b = root.join("proj-b");
        write_session(&dir_a, "aaa", "Alpha", "/a");
        write_session(&dir_b, "bbb", "Beta", "/b");

        let mut picker = SessionPicker::load(&dir_a);
        assert_eq!(picker.filtered.len(), 1);

        picker.handle_key(key(KeyCode::Tab));
        assert_eq!(picker.scope, SessionScope::AllProjects);
        assert_eq!(picker.filtered.len(), 2);

        picker.handle_key(key(KeyCode::Tab));
        assert_eq!(picker.scope, SessionScope::CurrentFolder);
        assert_eq!(picker.filtered.len(), 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn multiple_sources_merge_newest_first_with_tags() {
        let root = fixture_root();
        let native = root.join("rs/sessions/-home-u-proj");
        let ts = root.join("ts/sessions/-proj");
        write_session(&native, "rs1", "Native", "/home/u/proj");
        let old_ts = write_session(&ts, "ts1", "FromTs", "/home/u/proj");
        let new_ts = write_session(&ts, "ts2", "NewestTs", "/home/u/proj");
        bump_mtime(&new_ts);
        // Another project in each store: only visible in the all-projects scope.
        write_session(&root.join("rs/sessions/-other"), "rs9", "RsOther", "/other");
        write_session(&root.join("ts/sessions/-other"), "ts9", "TsOther", "/other");

        let mut picker = SessionPicker::load_sources(&[
            SessionSource::new(&native),
            SessionSource::labeled(&ts, "ts"),
        ]);
        let ids: Vec<&str> = picker.sessions.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], "ts2", "newest across both stores first");
        let tag = |id: &str| {
            picker
                .sessions
                .iter()
                .find(|s| s.id == id)
                .and_then(|s| s.source.clone())
        };
        assert_eq!(tag("rs1"), None);
        assert_eq!(tag("ts1").as_deref(), Some("ts"));

        picker.handle_key(key(KeyCode::Tab));
        assert_eq!(picker.filtered.len(), 5);
        let ts_other = picker.sessions.iter().find(|s| s.id == "ts9").unwrap();
        assert_eq!(ts_other.source.as_deref(), Some("ts"));

        // Picking a TS session resolves to its file (resume reads it; the
        // runtime writes the continuation into the native store).
        picker.handle_key(key(KeyCode::Tab));
        for c in "fromts".chars() {
            picker.handle_key(char_key(c));
        }
        picker.handle_key(key(KeyCode::Enter));
        assert_eq!(picker.take_pick(), Some(SessionPick::Resume(old_ts)));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn tagged_rows_render_the_store_tag() {
        let root = fixture_root();
        let ts = root.join("ts/sessions/-proj");
        write_session(&ts, "ts1", "FromTs", "/home/u/proj");
        let picker = SessionPicker::load_sources(&[SessionSource::labeled(&ts, "ts")]);
        let theme = Theme::default();
        let backend = TestBackend::new(160, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| picker.render(f, f.area(), &theme))
            .unwrap();
        let text = text_to_string(terminal.backend().buffer());
        assert!(text.contains("FromTs"), "{text}");
        assert!(text.contains("[ts]"), "{text}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn navigation_home_end_paging() {
        let root = fixture_root();
        let dir = root.join("proj");
        for i in 0..20 {
            write_session(&dir, &format!("s{i:02}"), &format!("Session {i}"), "/w");
        }
        let mut picker = SessionPicker::load(&dir);
        picker.list_height = 5;

        picker.handle_key(key(KeyCode::End));
        assert_eq!(picker.selected, 19);
        picker.handle_key(key(KeyCode::PageUp));
        assert_eq!(picker.selected, 14);
        picker.handle_key(key(KeyCode::Home));
        assert_eq!(picker.selected, 0);
        picker.handle_key(key(KeyCode::PageDown));
        assert_eq!(picker.selected, 5);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn empty_folder_shows_scope_hint_message() {
        let root = fixture_root();
        let dir = root.join("empty");
        fs::create_dir_all(&dir).unwrap();
        let picker = SessionPicker::load(&dir);

        let backend = TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| picker.render(f, f.area(), &Theme::default()))
            .unwrap();
        let rendered = text_to_string(terminal.backend().buffer());
        assert!(rendered.contains("No sessions in current folder. Press tab to view all."));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn render_shows_rows_query_and_footer_hint() {
        let root = fixture_root();
        let dir = root.join("proj");
        write_session(&dir, "aaa", "Fix login bug", "/work/app");
        write_session(&dir, "bbb", "Refactor api", "/work/app");
        let mut picker = SessionPicker::load(&dir);
        for c in "fix".chars() {
            picker.handle_key(char_key(c));
        }

        let backend = TestBackend::new(120, 24);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| picker.render(f, f.area(), &Theme::default()))
            .unwrap();
        let rendered = text_to_string(terminal.backend().buffer());
        assert!(rendered.contains("sessions"));
        assert!(rendered.contains("Fix login bug"));
        assert!(rendered.contains("/work/app"));
        assert!(rendered.contains("❯ fix"));
        assert!(rendered.contains("enter"));
        assert!(rendered.contains("delete"));
        assert!(rendered.contains("all projects"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_confirm_replaces_list_in_render() {
        let root = fixture_root();
        let dir = root.join("proj");
        write_session(&dir, "aaa", "Doomed", "/w");
        let mut picker = SessionPicker::load(&dir);
        picker.handle_key(key(KeyCode::Delete));

        let backend = TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| picker.render(f, f.area(), &Theme::default()))
            .unwrap();
        let rendered = text_to_string(terminal.backend().buffer());
        assert!(rendered.contains("Delete session \"Doomed\"?"));
        assert!(rendered.contains("enter confirm · esc cancel"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn relative_time_buckets() {
        let now = SystemTime::now();
        assert_eq!(relative_time(now), "just now");
        assert_eq!(relative_time(now - Duration::from_secs(120)), "2m ago");
        assert_eq!(relative_time(now - Duration::from_secs(7200)), "2h ago");
        assert_eq!(relative_time(now - Duration::from_secs(172_800)), "2d ago");
    }

    #[test]
    fn civil_date_known_days() {
        assert_eq!(civil_date(0), "1970-01-01");
        assert_eq!(civil_date(86_400), "1970-01-02");
        // 2026-01-01 00:00:00 UTC = 1767225600.
        assert_eq!(civil_date(1_767_225_600), "2026-01-01");
    }
}
