//! Modal overlays — ports of `pi-tui/src/overlays/` and the
//! interactive-mode controllers in `pi-coding-agent/src/modes/controllers/`.
//!
//! An overlay is a [`Component`](crate::component::Component) that, when
//! open, owns keyboard focus and paints over a centered area of the screen
//! (TS: bordered box at a percentage of the terminal, theme `"border"`
//! token). Each overlay lives in its own module so implementers never
//! collide.
//!
//! [`Overlays`] is the focus stack: at most one overlay owns focus at a
//! time, but overlays can stack (help over the model picker, per the TS
//! overlay-manager comment that most overlays are exclusive while help may
//! stack over others). Key events go to the top overlay first and bubble
//! downward while ignored; [`Overlays::open`] reports the top for callers
//! that predate the stack.
//!
//! Popup overlays render into the live region (centered, `Clear`ed). A
//! full-screen overlay ([`OpenOverlay::is_fullscreen`], e.g. the diff view)
//! instead borrows the alternate screen from the shell renderer for the
//! duration it is on top.

pub mod command_palette;
pub mod diff;
pub mod help;
pub mod large_paste;
pub mod models;
pub mod session_info;
pub mod sessions;
pub mod settings;
pub mod thinking;
pub mod tree;

use crate::component::{Component, EventOutcome};
use crate::keys::KeyEvent;
use crate::theme::Theme;
use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// A centered rectangle of `percent_x`/`percent_y` of `area`, shared by all
/// popups (TS overlays use the same percentage-of-terminal rule).
#[must_use]
pub fn centered(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let vert = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    let horiz = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vert[1]);
    horiz[1]
}

/// Which overlay (if any) is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenOverlay {
    /// Help/keyhints (`?`).
    Help,
    /// Model picker (`/model` or Tab).
    Models,
    /// Session picker (`/resume`).
    Sessions,
    /// Command palette (Cmd/Ctrl+K).
    CommandPalette,
    /// Full-screen diff view (from an edit tool entry).
    Diff,
    /// Session tree / rewind navigator (`/tree`, double-Esc rewind column).
    Tree,
    /// Settings editor (`/settings`).
    Settings,
    /// `/session` info panel (bottom-anchored, full width).
    SessionInfo,
    /// Thinking-level selector (bare `/effort`), anchored over the composer.
    Thinking,
    /// Large-paste chooser (`paste.largeMenuThreshold` gate).
    LargePaste,
}

impl OpenOverlay {
    /// Whether this overlay renders full-screen on the alternate screen
    /// rather than as a centered popup in the live region.
    ///
    /// TS mounts with `fullscreen: true`: the model hub (`/model`), the
    /// settings selector (`/settings`), the session selector (`/resume`),
    /// the esc-esc rewind selector and the annotation/diff overlay
    /// (`selector-controller.ts` `#showFullscreenMenu` + the explicit
    /// mounts). `/tree` (TS: editor-container replacement sized from the
    /// terminal rows) and `/session` (TS: bottom-anchored `maxHeight: 100%`
    /// overlay) are not alt-screen in TS but need the whole terminal height,
    /// which omp-rs's fixed inline viewport cannot provide, so they borrow
    /// the alternate screen too — without mouse tracking (see
    /// [`Overlays::wants_mouse`]).
    #[must_use]
    pub fn is_fullscreen(self) -> bool {
        matches!(
            self,
            Self::Models
                | Self::Settings
                | Self::Sessions
                | Self::Diff
                | Self::Tree
                | Self::SessionInfo
        )
    }
}

/// The focus stack: bottom-to-top order of open overlays. Empty means the
/// editor owns focus.
#[derive(Debug, Clone, Default)]
pub struct Overlays {
    help: help::Help,
    models: models::ModelPicker,
    sessions: sessions::SessionPicker,
    palette: command_palette::CommandPalette,
    diff: diff::DiffView,
    tree: tree::TreeNavigator,
    settings: settings::SettingsOverlay,
    session_info: session_info::SessionInfoOverlay,
    thinking: thinking::ThinkingSelector,
    large_paste: large_paste::LargePasteMenu,
    stack: Vec<OpenOverlay>,
}

impl Overlays {
    /// Toggle one overlay: if it is anywhere in the stack, remove it;
    /// otherwise push it on top. (TS: opening another overlay stacks it;
    /// re-toggling closes.)
    pub fn toggle(&mut self, which: OpenOverlay) {
        if let Some(pos) = self.stack.iter().position(|o| *o == which) {
            self.stack.remove(pos);
        } else {
            if which == OpenOverlay::Models {
                self.models.on_open();
            }
            self.stack.push(which);
        }
    }

    /// The overlay that currently owns focus (top of stack).
    #[must_use]
    pub fn open(&self) -> Option<OpenOverlay> {
        self.stack.last().copied()
    }

    /// The full stack, bottom-to-top.
    #[must_use]
    pub fn stack(&self) -> &[OpenOverlay] {
        &self.stack
    }

    /// Mutable access to the session picker (key routing + pick drain).
    pub fn sessions_mut(&mut self) -> &mut sessions::SessionPicker {
        &mut self.sessions
    }

    /// Mutable access to the model picker (catalogue seeding, `/model`
    /// query pre-fill).
    pub fn models_mut(&mut self) -> &mut models::ModelPicker {
        &mut self.models
    }

    /// Open the embedded session picker (`/resume`): rescan `sources`, reset
    /// the query/selection and push the overlay (replacing a stale picker).
    pub fn open_sessions(&mut self, sources: &[sessions::SessionSource]) {
        let mut picker = sessions::SessionPicker::load_sources(sources);
        picker.set_standalone(true); // TS mounts /resume fullscreen (selector-controller.ts)
        self.sessions = picker;
        if !self.stack.contains(&OpenOverlay::Sessions) {
            self.stack.push(OpenOverlay::Sessions);
        }
    }

    /// Open the tree navigator in `mode`, seeded with raw journal entries
    /// (`/tree` full view, double-Esc rewind column).
    pub fn open_tree(&mut self, entries: &[serde_json::Value], mode: tree::TreeMode) {
        self.tree.set_entries(entries, mode);
        self.tree.title = match mode {
            tree::TreeMode::Rewind => "rewind".to_owned(),
            tree::TreeMode::Tree => "tree".to_owned(),
        };
        if !self.stack.contains(&OpenOverlay::Tree) {
            self.stack.push(OpenOverlay::Tree);
        }
    }

    /// Open the `/settings` overlay (rows are seeded by the host through
    /// [`Self::settings_mut`] before or after opening).
    pub fn open_settings(&mut self) {
        if !self.stack.contains(&OpenOverlay::Settings) {
            self.stack.push(OpenOverlay::Settings);
        }
    }

    /// Mutable access to the settings overlay (seed values, drain edits).
    pub fn settings_mut(&mut self) -> &mut settings::SettingsOverlay {
        &mut self.settings
    }

    /// Open the `/session` info panel with a fresh report (replacing an
    /// open one, which moves back to the top of the stack). `title` names
    /// the panel (`None` = "Session Info"; `/usage` passes "Usage").
    pub fn open_session_info(
        &mut self,
        title: Option<&str>,
        lines: Vec<session_info::SessionInfoLine>,
    ) {
        self.session_info.set_title(title);
        self.session_info.set_lines(lines);
        self.close(OpenOverlay::SessionInfo);
        self.stack.push(OpenOverlay::SessionInfo);
    }

    /// Open the thinking-level selector over `levels` with `current`
    /// preselected (bare `/effort`).
    pub fn open_thinking(&mut self, levels: Vec<String>, current: &str) {
        self.thinking.open(levels, current);
        self.close(OpenOverlay::Thinking);
        self.stack.push(OpenOverlay::Thinking);
    }

    /// Open the large-paste chooser (TS `presentLargePasteMenu`).
    pub fn open_large_paste(&mut self, line_count: usize) {
        self.large_paste.open(line_count);
        self.close(OpenOverlay::LargePaste);
        self.stack.push(OpenOverlay::LargePaste);
    }

    /// Mutable access to the large-paste chooser (pick drain).
    pub fn large_paste_mut(&mut self) -> &mut large_paste::LargePasteMenu {
        &mut self.large_paste
    }

    /// Close the top overlay (esc routing).
    pub fn pop(&mut self) {
        self.stack.pop();
    }

    /// Whether any overlay is open (i.e. the editor does not have focus).
    #[must_use]
    pub fn wants_focus(&self) -> bool {
        !self.stack.is_empty()
    }

    /// Whether the topmost open overlay renders full-screen and thus needs
    /// the shell renderer's alternate screen (core.md §3). The controller
    /// borrows the alt screen per frame while this is true.
    #[must_use]
    pub fn is_fullscreen(&self) -> bool {
        self.open().is_some_and(OpenOverlay::is_fullscreen)
    }

    /// Back-compat alias for [`Self::is_fullscreen`].
    #[must_use]
    pub fn top_is_fullscreen(&self) -> bool {
        self.is_fullscreen()
    }

    /// Mutable access to the tree navigator (populate entries, drain picks).
    pub fn tree_mut(&mut self) -> &mut tree::TreeNavigator {
        &mut self.tree
    }

    fn component(&self, which: OpenOverlay) -> &dyn Component {
        match which {
            OpenOverlay::Help => &self.help,
            OpenOverlay::Models => &self.models,
            OpenOverlay::Sessions => &self.sessions,
            OpenOverlay::CommandPalette => &self.palette,
            OpenOverlay::Diff => &self.diff,
            OpenOverlay::Tree => &self.tree,
            OpenOverlay::Settings => &self.settings,
            OpenOverlay::SessionInfo => &self.session_info,
            OpenOverlay::Thinking => &self.thinking,
            OpenOverlay::LargePaste => &self.large_paste,
        }
    }

    fn component_mut(&mut self, which: OpenOverlay) -> &mut dyn Component {
        match which {
            OpenOverlay::Help => &mut self.help,
            OpenOverlay::Models => &mut self.models,
            OpenOverlay::Sessions => &mut self.sessions,
            OpenOverlay::CommandPalette => &mut self.palette,
            OpenOverlay::Diff => &mut self.diff,
            OpenOverlay::Tree => &mut self.tree,
            OpenOverlay::Settings => &mut self.settings,
            OpenOverlay::SessionInfo => &mut self.session_info,
            OpenOverlay::Thinking => &mut self.thinking,
            OpenOverlay::LargePaste => &mut self.large_paste,
        }
    }

    /// Route a key event to the open overlays, top first; bubbling stops at
    /// the first consumer. Returns `Ignored` when nothing is open or every
    /// overlay ignored the event.
    pub fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        for i in (0..self.stack.len()).rev() {
            let which = self.stack[i];
            if self.component_mut(which).handle_key(key).consumed() {
                return EventOutcome::Consumed;
            }
        }
        EventOutcome::Ignored
    }

    /// Drain every child's queued intents into `Action`s for the app to
    /// dispatch (TS: the interactive controller polls each overlay after
    /// key handling). Covers: model picker's SetModel/SetThinkingLevel,
    /// palette confirmations (`/cmd` → [`Action::Slash`]), session picker
    /// resolutions ([`Action::ResumeSession`]), tree selections
    /// ([`Action::Rewind`]). Resolved overlays (palette, sessions, tree) are
    /// popped automatically; Models stays open. The session picker is popped
    /// from the stack whenever its pick resolves (either variant).
    pub fn take_actions(&mut self) -> Vec<crate::action::Action> {
        use crate::action::Action;
        let mut out = self.models.take_actions();
        if let Some(label) = self.palette.take_picked() {
            let trimmed = label.trim_start_matches('/');
            let (name, args) = trimmed
                .split_once(char::is_whitespace)
                .map_or((trimmed, ""), |(n, a)| (n, a.trim_start()));
            if !name.is_empty() {
                out.push(Action::Slash {
                    name: name.to_string(),
                    args: args.to_string(),
                });
            }
            self.palette_close();
        }
        if let Some(sel) = self.tree.take_selection() {
            out.push(Action::Rewind {
                entry_id: sel.entry_id,
            });
            self.close(OpenOverlay::Tree);
        }
        if let Some(pick) = self.sessions.take_pick() {
            if let sessions::SessionPick::Resume(path) = pick {
                out.push(Action::ResumeSession { path, fork: false });
            }
            self.close(OpenOverlay::Sessions);
        }
        // Settings edits persist through the port and apply live where the
        // app can; the overlay stays open like the model picker.
        if let Some(level) = self.thinking.take_pick() {
            out.push(Action::SetThinkingLevel(level));
            self.close(OpenOverlay::Thinking);
        }
        for (id, value) in self.settings.take_previews() {
            out.push(Action::SettingPreview { id, value });
        }
        for (id, value) in self.settings.take_changes() {
            out.push(Action::SettingChanged { id, value });
        }
        out
    }

    fn close(&mut self, which: OpenOverlay) {
        if let Some(pos) = self.stack.iter().position(|o| *o == which) {
            self.stack.remove(pos);
        }
    }

    fn palette_close(&mut self) {
        self.close(OpenOverlay::CommandPalette);
    }

    /// Render every open overlay, bottom-to-top, each over the full area
    /// (popups clear and center themselves; the full-screen top overlay is
    /// expected to have been handed the alternate screen by the shell).
    pub fn render(
        &self,
        frame: &mut ratatui::Frame<'_>,
        area: ratatui::layout::Rect,
        theme: &Theme,
    ) {
        for &which in &self.stack {
            self.component(which).render(frame, area, theme);
        }
    }

    /// Index of the topmost full-screen overlay in the stack, if any.
    fn fullscreen_base(&self) -> Option<usize> {
        self.stack.iter().rposition(|o| o.is_fullscreen())
    }

    /// Whether any open overlay needs the alternate screen. A popup stacked
    /// over a full-screen overlay (help over `/model`) keeps the alt screen
    /// and paints over the full-screen frame.
    #[must_use]
    pub fn needs_alt_screen(&self) -> bool {
        self.fullscreen_base().is_some()
    }

    /// Whether terminal mouse reporting should be on for the overlay layer:
    /// the top overlay is a TS `fullscreen: true` mount (TS `tui.ts`
    /// `wantMouse = "full"`). Popups and the non-TS-fullscreen alt-screen
    /// borrowers (`/tree`, `/session`) leave tracking off so native text
    /// selection keeps working there.
    #[must_use]
    pub fn wants_mouse(&self) -> bool {
        match self.open() {
            Some(
                OpenOverlay::Models
                | OpenOverlay::Settings
                | OpenOverlay::Sessions
                | OpenOverlay::Diff,
            ) => true,
            Some(OpenOverlay::Tree) => self.tree.mode == tree::TreeMode::Rewind,
            _ => false,
        }
    }

    /// Route a mouse event to the top overlay only (TS: the focused
    /// fullscreen component receives SGR reports through its input
    /// handler). Mouse events never bubble to covered overlays.
    pub fn handle_mouse(&mut self, event: &crate::mouse::SgrMouseEvent) -> EventOutcome {
        match self.open() {
            Some(which) => self.component_mut(which).handle_mouse(event),
            None => EventOutcome::Ignored,
        }
    }

    /// Render the alternate-screen frame: the topmost full-screen overlay
    /// over the whole `area` (the full terminal, origin 0,0) plus any popups
    /// stacked above it. Overlays below the full-screen one are covered and
    /// not painted. No-op when no full-screen overlay is open.
    pub fn render_fullscreen(
        &self,
        frame: &mut ratatui::Frame<'_>,
        area: ratatui::layout::Rect,
        theme: &Theme,
    ) {
        let Some(base) = self.fullscreen_base() else {
            return;
        };
        for &which in &self.stack[base..] {
            self.component(which).render(frame, area, theme);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::KeyCode;

    #[test]
    fn centered_is_centered() {
        let area = Rect::new(0, 0, 100, 40);
        let rect = centered(area, 60, 50);
        assert_eq!(rect.width, 60);
        assert_eq!(rect.height, 20);
        assert!(rect.x > 0 && rect.y > 0);
        assert!(rect.x + rect.width <= 100);
    }

    #[test]
    fn toggle_pushes_and_removes() {
        let mut o = Overlays::default();
        assert_eq!(o.open(), None);
        o.toggle(OpenOverlay::Help);
        assert_eq!(o.open(), Some(OpenOverlay::Help));
        o.toggle(OpenOverlay::Models);
        assert_eq!(o.stack(), &[OpenOverlay::Help, OpenOverlay::Models]);
        // Re-toggling help removes it from the middle, models stays top.
        o.toggle(OpenOverlay::Help);
        assert_eq!(o.stack(), &[OpenOverlay::Models]);
        o.toggle(OpenOverlay::Models);
        assert_eq!(o.open(), None);
    }

    #[test]
    fn esc_close_routing() {
        let mut o = Overlays::default();
        o.toggle(OpenOverlay::Help);
        let outcome = o.handle_key(KeyEvent::from(KeyCode::Esc));
        // Help ignores esc (the host owns open/close state) so it bubbles.
        assert!(!outcome.consumed());
        o.pop();
        assert_eq!(o.open(), None);
    }

    #[test]
    fn fullscreen_marker() {
        assert!(OpenOverlay::Diff.is_fullscreen());
        assert!(!OpenOverlay::Help.is_fullscreen());
        let mut o = Overlays::default();
        assert!(!o.top_is_fullscreen());
        o.toggle(OpenOverlay::Diff);
        assert!(o.top_is_fullscreen());
    }

    #[test]
    fn ts_fullscreen_set_and_mouse_scope() {
        for o in [
            OpenOverlay::Models,
            OpenOverlay::Settings,
            OpenOverlay::Sessions,
        ] {
            assert!(o.is_fullscreen(), "{o:?} is a TS fullscreen:true mount");
        }
        for o in [OpenOverlay::Help, OpenOverlay::CommandPalette] {
            assert!(!o.is_fullscreen(), "{o:?} is a TS popup");
        }
        let mut o = Overlays::default();
        o.open_settings();
        assert!(o.needs_alt_screen() && o.wants_mouse());
        // Help popped over the fullscreen menu keeps the alt screen but, as
        // a non-fullscreen top overlay, turns reporting off (TS wantMouse).
        o.toggle(OpenOverlay::Help);
        assert!(o.needs_alt_screen());
        assert!(!o.wants_mouse());
        o.pop();
        // /session borrows the alt screen without TS fullscreen mouse.
        o.open_session_info(None, Vec::new());
        assert!(o.needs_alt_screen());
        assert!(!o.wants_mouse());
        o.pop();
        o.pop();
        assert!(!o.needs_alt_screen() && !o.wants_mouse());
    }
}
