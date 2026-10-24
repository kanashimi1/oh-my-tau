//! Session-tree navigator + user-message rewind selector — port of
//! `pi-tui/src/overlays/rewind-selector.ts` (MVP: single column of user
//! messages along the live branch) and the `/tree` view (branch layout).
//!
//! Data comes from the raw session journal entries (`omp-session::tree`):
//! entries form a tree via `parentId`; a rewind target is any *user message*
//! entry on the branch ending at the current leaf. Selecting one asks the
//! host to branch the session to that entry (`Runtime` `branch` rpc, omp
//! `AgentSession.branch`) — the selection is exposed via [`TreeNavigator::take_selection`]
//! because `Action` has no rewind variant yet (asked TuiArchitect for
//! `Action::Rewind { entry_id }`).

use omp_session::tree::{
    branch_path, build_tree, is_user_message_entry, leaf_id, user_message_text, TreeNode,
};
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent};
use crate::theme::Theme;

/// One selectable rewind target / tree leaf row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeRow {
    /// Entry id (branch target for rewind).
    pub id: String,
    /// One-line preview of the user prompt (or entry kind for `/tree`).
    pub label: String,
    /// Tree depth (0 = root level) — drives connector art.
    pub depth: usize,
    /// Number of forks (siblings) at this node — `⎇ i/n` caption.
    pub fork: Option<(usize, usize)>,
    /// True when this is the live branch's leaf.
    pub current: bool,
}

/// Flattened tree rows, depth-first (TS `SessionTreeNode` walk).
#[must_use]
pub fn flatten_tree(nodes: &[TreeNode], leaf: Option<&str>) -> Vec<TreeRow> {
    let mut out = Vec::new();
    walk(nodes, 0, &mut out, leaf);
    out
}

fn walk(nodes: &[TreeNode], depth: usize, out: &mut Vec<TreeRow>, leaf: Option<&str>) {
    let total = nodes.len();
    for (i, node) in nodes.iter().enumerate() {
        let label = node_label(node);
        out.push(TreeRow {
            id: node.id().unwrap_or_default().to_owned(),
            label,
            depth,
            fork: (total > 1).then(|| (i + 1, total)),
            current: leaf.is_some() && node.id() == leaf,
        });
        walk(&node.children, depth + 1, out, leaf);
    }
}

fn node_label(node: &TreeNode) -> String {
    let kind = node
        .entry
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("entry");
    if let Some(text) = user_message_text(&node.entry) {
        let one = text.lines().next().unwrap_or("").to_owned();
        if !one.is_empty() {
            return one;
        }
    }
    match node.label.as_deref() {
        Some(l) => format!("{kind}: {l}"),
        None => kind.to_owned(),
    }
}

/// User-message entries along the branch ending at `leaf`, oldest first —
/// the rewind selector's rows (TS `rewind-selector.ts` column).
#[must_use]
pub fn rewind_rows<'a>(
    entries: &'a [serde_json::Value],
    leaf: Option<&str>,
) -> Vec<&'a serde_json::Value> {
    let Some(leaf) = leaf else { return Vec::new() };
    branch_path(entries, leaf)
        .into_iter()
        .filter(|e| is_user_message_entry(e))
        .collect()
}

/// Filter rows by a case-insensitive substring query (rewind `f` filter).
#[must_use]
pub fn filter_rows(rows: &[TreeRow], query: &str) -> Vec<usize> {
    let q = query.to_lowercase();
    rows.iter()
        .enumerate()
        .filter(|(_, r)| q.is_empty() || r.label.to_lowercase().contains(&q))
        .map(|(i, _)| i)
        .collect()
}

/// Interactive selector: rewind column (double-Esc) or full tree (`/tree`).
#[derive(Debug, Clone, Default)]
pub struct TreeNavigator {
    /// Which view the host asked for.
    pub mode: TreeMode,
    /// Title for the panel border (`rewind` / `tree`).
    pub title: String,
    rows: Vec<TreeRow>,
    visible: Vec<usize>,
    cursor: usize,
    filter: Option<String>,
}

/// What [`TreeNavigator`] shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TreeMode {
    /// User-message rewind column (double-Esc default action).
    #[default]
    Rewind,
    /// Full entry tree.
    Tree,
}

/// A confirmed selection the host must act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// Entry id of the chosen user message.
    pub entry_id: String,
    /// Its prompt text (the host re-prompts after branching).
    pub text: String,
}

impl TreeNavigator {
    /// Rebuild from raw journal entries. `mode` picks rewind vs tree rows.
    pub fn set_entries(&mut self, entries: &[serde_json::Value], mode: TreeMode) {
        self.mode = mode;
        let leaf = leaf_id(entries).map(str::to_owned);
        self.rows = match mode {
            TreeMode::Rewind => rewind_rows(entries, leaf.as_deref())
                .iter()
                .map(|e| TreeRow {
                    id: e
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    label: user_message_text(e).unwrap_or_default(),
                    depth: 0,
                    fork: None,
                    current: false,
                })
                .collect(),
            TreeMode::Tree => flatten_tree(&build_tree(entries), leaf.as_deref()),
        };
        self.cursor = self.rows.len().saturating_sub(1); // "You are here" at the leaf
        self.refresh_visible();
    }

    fn refresh_visible(&mut self) {
        let query = self.filter.as_deref().unwrap_or_default();
        self.visible = filter_rows(&self.rows, query);
        if self.visible.is_empty() {
            self.cursor = 0;
        } else {
            self.cursor = self.cursor.min(self.visible.len() - 1);
        }
    }

    /// Start/clear the filter (`f`).
    pub fn toggle_filter(&mut self, on: bool) {
        self.filter = on.then(String::new);
        self.refresh_visible();
    }

    /// Selected row, if any.
    #[must_use]
    pub fn selected(&self) -> Option<&TreeRow> {
        self.visible
            .get(self.cursor)
            .and_then(|i| self.rows.get(*i))
    }

    /// Take the confirmed selection (`Enter`) for the host to branch on.
    pub fn take_selection(&mut self) -> Option<Selection> {
        let row = self.selected()?;
        Some(Selection {
            entry_id: row.id.clone(),
            text: row.label.clone(),
        })
    }
}

impl Component for TreeNavigator {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let popup = super::centered(area, 70, 70);
        if popup.width < 6 || popup.height < 4 {
            return;
        }
        frame.render_widget(Clear, popup);
        let title = match self.filter.as_deref() {
            Some(f) => format!(" {} — filter: {f}▏ ", self.title),
            None => format!(" {} ", self.title),
        };
        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.color("border")));
        let inner = block.inner(popup);
        frame.render_widget(block, popup);
        if self.visible.is_empty() {
            let msg = Line::from(Span::styled(
                if self.filter.is_some() {
                    "No turns match"
                } else {
                    "No turns yet"
                },
                Style::default().fg(theme.color("dim")),
            ));
            frame.render_widget(Paragraph::new(msg), inner);
            return;
        }
        let body = self.visible.len().min(usize::from(inner.height));
        let start = self.cursor.min(self.visible.len() - body);
        let lines: Vec<Line<'_>> = self.visible[start..start + body]
            .iter()
            .map(|&i| {
                let row = &self.rows[i];
                let sel = i == self.visible[self.cursor];
                let marker = if sel { "> " } else { "  " };
                let token = if sel { "accent" } else { "text" };
                let mut spans = Vec::new();
                if row.depth > 0 {
                    spans.push(Span::styled(
                        format!("{:width$}", "", width = row.depth * 2),
                        Style::default().fg(theme.color("dim")),
                    ));
                }
                if let Some((f, n)) = row.fork {
                    spans.push(Span::styled(
                        format!("⎇ {f}/{n} "),
                        Style::default().fg(theme.color("dim")),
                    ));
                }
                if row.current {
                    spans.push(Span::styled(
                        "● ",
                        Style::default().fg(theme.color("success")),
                    ));
                }
                spans.push(Span::styled(
                    format!("{marker}{}", truncate(&row.label, usize::from(inner.width))),
                    Style::default().fg(theme.color(token)),
                ));
                Line::from(spans)
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        if let Some(filter) = self.filter.as_mut() {
            match key.code {
                KeyCode::Char(c) => {
                    filter.push(c);
                    self.refresh_visible();
                    return EventOutcome::Consumed;
                }
                KeyCode::Backspace => {
                    if filter.pop().is_none() {
                        self.toggle_filter(false);
                    }
                    self.refresh_visible();
                    return EventOutcome::Consumed;
                }
                _ => {}
            }
        }
        let n = self.visible.len();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') if n > 0 => {
                self.cursor = self.cursor.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') if n > 0 => {
                self.cursor = (self.cursor + 1).min(n - 1);
            }
            KeyCode::PageUp if n > 0 => self.cursor = self.cursor.saturating_sub(10),
            KeyCode::PageDown if n > 0 => self.cursor = (self.cursor + 10).min(n - 1),
            KeyCode::Home if n > 0 => self.cursor = 0,
            KeyCode::End if n > 0 => self.cursor = n - 1,
            KeyCode::Char('f') => self.toggle_filter(self.filter.is_none()),
            KeyCode::Enter | KeyCode::Esc => return EventOutcome::Consumed,
            _ => return EventOutcome::Ignored,
        }
        EventOutcome::Consumed
    }

    fn wants_focus(&self) -> bool {
        true
    }
}

/// Truncate a label to `width` display columns with an ellipsis.
fn truncate(s: &str, width: usize) -> String {
    if s.chars().count() <= width.saturating_sub(1) {
        s.to_owned()
    } else {
        let keep = width.saturating_sub(2);
        format!("{}…", s.chars().take(keep).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::KeyModifiers;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn user_entries() -> Vec<serde_json::Value> {
        (0..3)
            .map(|i| {
                let id = format!("e{}", i + 1);
                let parent = (i > 0).then(|| format!("e{i}"));
                serde_json::json!({
                    "type": "message", "id": id,
                    "parentId": parent, "timestamp": "ts",
                    "message": {"role": "user", "content": format!("hello {id}")}
                })
            })
            .collect()
    }

    #[test]
    fn rewind_lists_user_messages_leaf_first_selection() {
        let mut nav = TreeNavigator::default();
        nav.set_entries(&user_entries(), TreeMode::Rewind);
        assert_eq!(nav.visible.len(), 3);
        let sel = nav.take_selection().expect("leaf selected");
        assert_eq!(sel.entry_id, "e3");
        nav.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(nav.take_selection().unwrap().entry_id, "e2");
    }

    #[test]
    fn tree_view_flattens_and_marks_leaf() {
        let entries = user_entries();
        let mut nav = TreeNavigator::default();
        nav.set_entries(&entries, TreeMode::Tree);
        assert_eq!(nav.visible.len(), 3);
        assert!(!nav.rows[0].current);
        assert!(nav.selected().unwrap().current);
        assert_eq!(nav.rows[2].depth, 2);
    }

    #[test]
    fn filter_narrows_and_clears() {
        let mut nav = TreeNavigator::default();
        nav.set_entries(&user_entries(), TreeMode::Rewind);
        nav.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE));
        nav.handle_key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::NONE));
        assert!(nav.visible.is_empty());
        nav.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(nav.visible.len(), 3);
    }

    #[test]
    fn render_snapshot() {
        let mut nav = TreeNavigator::default();
        nav.title = "rewind".into();
        nav.set_entries(&user_entries(), TreeMode::Rewind);
        let theme = Theme::default();
        let mut term = Terminal::new(TestBackend::new(40, 10)).unwrap();
        term.draw(|f| nav.render(f, f.area(), &theme)).unwrap();
        let out = term.backend().to_string();
        assert!(out.contains("rewind"), "{out}");
        assert!(out.contains("hello e1"), "{out}");
    }
}
