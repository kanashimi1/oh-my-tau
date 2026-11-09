use super::turn_usage::format_tokens;
use super::{
    App, Component, Constraint, Direction, Frame, Instant, Layout, Line, Maintenance, Rect, Span,
    Style, SPINNER, SPINNER_ADVANCE_MS,
};

impl App {
    // ---- Live-region layout (ShellRender commit/live API) ----------------

    /// Height of the queued band in rows: a leading spacer, one heading +
    /// one row per message per non-empty group, and the trailing key-hint
    /// row (§5.2). Zero when nothing is queued.
    #[must_use]
    pub fn band_height(&self) -> u16 {
        let groups =
            usize::from(!self.steering.is_empty()) + usize::from(!self.follow_up.is_empty());
        if groups == 0 {
            0
        } else {
            // The queue holds a handful of entries, far below u16 range.
            #[allow(clippy::cast_possible_truncation)]
            {
                (1 + groups + self.steering.len() + self.follow_up.len() + 1) as u16
            }
        }
    }

    /// Whether the "Working…" loader row is visible.
    #[must_use]
    pub fn show_loader(&self) -> bool {
        self.status.streaming || self.maintenance.is_some()
    }

    /// Minimum height of the live region below the committed chat: band +
    /// loader + status line (1) + one editor row (the band composer has no
    /// frame). `ShellRender` sizes the inline viewport with this.
    #[must_use]
    pub fn live_height(&self) -> u16 {
        let width = crossterm::terminal::size().map_or(80, |(w, _)| w);
        let shape = self.editor.shape();
        // One content row inside the shape's chrome, plus the standalone bar.
        let editor = 1 + u16::from(shape.has_top()) + u16::from(shape.has_bottom());
        self.band_height()
            + self.error_banner_height(width)
            + u16::from(self.show_loader())
            + editor
            + self.bottom_bar_rows(width)
    }

    /// Rows of the standalone status bar under the editor (gap + bar) for
    /// the active composer shape (`ComposerStyle.bottomBar`).
    fn bottom_bar_rows(&self, width: u16) -> u16 {
        let shape = self.editor.shape();
        if crate::composer::bottom_bar(shape, &self.status, &self.theme, usize::from(width))
            .is_none()
        {
            return 0;
        }
        1 + u16::from(shape.bottom_bar_gap())
    }

    /// Push this frame's status content into the editor's top chrome (TS
    /// `attachToEditor` provider) for a composer `width` columns wide.
    fn sync_composer_chrome(&mut self, width: u16) {
        let shape = self.editor.shape();
        let top = crate::composer::top_border(shape, &self.status, &self.theme, usize::from(width));
        self.editor.set_top_border(top);
    }

    /// Paint the standalone bar (and its spacer row) under the editor.
    fn render_bottom_bar(&self, frame: &mut Frame<'_>, area: Rect) {
        let shape = self.editor.shape();
        let Some(bar) =
            crate::composer::bottom_bar(shape, &self.status, &self.theme, usize::from(area.width))
        else {
            return;
        };
        let y = area.y + u16::from(shape.bottom_bar_gap());
        if y < area.y + area.height {
            frame.render_widget(
                ratatui::widgets::Paragraph::new(bar),
                Rect {
                    y,
                    height: 1,
                    ..area
                },
            );
        }
    }

    /// Rows of the pinned provider-error banner (TS `ErrorBannerComponent`,
    /// between the queued-messages band and the loader); 0 when none.
    fn error_banner_height(&self, width: u16) -> u16 {
        self.messages.pinned_error().map_or(0, |message| {
            let lines =
                crate::chat::blocks::error_banner_lines(message, usize::from(width), &self.theme);
            u16::try_from(lines.len()).unwrap_or(u16::MAX)
        })
    }

    fn render_error_banner(&self, frame: &mut Frame<'_>, area: Rect) {
        if area.height == 0 {
            return;
        }
        if let Some(message) = self.messages.pinned_error() {
            let lines = crate::chat::blocks::error_banner_lines(
                message,
                usize::from(area.width),
                &self.theme,
            );
            frame.render_widget(ratatui::widgets::Paragraph::new(lines), area);
        }
    }

    /// Whether an open overlay (or a `fullscreen` extension dialog) is full-screen and the shell renderer must
    /// hold the alternate screen (`ShellRender` contract); while true the
    /// controller paints [`Self::render_fullscreen`] instead of the live
    /// region.
    #[must_use]
    pub fn has_fullscreen_overlay(&self) -> bool {
        self.overlays.needs_alt_screen() || self.extension_ui.wants_fullscreen()
    }

    /// Paint the alternate-screen frame: the topmost full-screen overlay
    /// (plus popups stacked over it) into the whole terminal area at origin
    /// (0,0), then any pending extension dialog on top.
    pub fn render_fullscreen(&mut self, frame: &mut Frame<'_>) {
        let area = frame.area();
        self.overlays.render_fullscreen(frame, area, &self.theme);
        self.extension_ui.render(frame, area, &self.theme);
    }

    /// Rows that settled since the last call and must enter terminal
    /// scrollback (`ShellRender::insert_before` source). Advances the chat
    /// watermark: committed entries are owned by the scrollback afterwards
    /// and are no longer drawn by [`Self::render_chat`]. With `rebuild =
    /// /// true` (width-change resize) returns the FULL transcript re-rendered
    /// at `width` and resets the watermark.
    /// Consume a pending scrollback erase (see [`AppEvent::SessionReset`]).
    pub fn take_clear_scrollback(&mut self) -> bool {
        std::mem::take(&mut self.clear_scrollback)
    }

    /// Drain rows for terminal scrollback (see the doc above
    /// [`Self::take_clear_scrollback`]: incremental, or the whole transcript
    /// when `rebuild`).
    ///
    /// The clean welcome is held back (drawn live by [`Self::render`]) while
    /// its startup shimmer runs; it commits with its static rows once the
    /// shimmer ends, or as soon as anything else retires or a rebuild runs.
    pub fn take_scrollback_lines(&mut self, width: u16, rebuild: bool) -> Vec<Line<'static>> {
        let transcript = if rebuild {
            self.messages.all_rows(width)
        } else {
            self.messages.take_committed_rows(width)
        };
        let hold =
            !rebuild && transcript.is_empty() && self.welcome_shimmer(Instant::now()).is_some();
        let mut rows = Vec::new();
        if !hold && self.welcome.as_ref().is_some_and(|w| rebuild || !w.emitted) {
            rows = self.welcome_rows(width, 0.0);
            if !rows.is_empty() {
                rows.push(Line::default());
            }
            if let Some(w) = self.welcome.as_mut() {
                w.emitted = true;
            }
        }
        rows.extend(transcript);
        rows
    }

    /// Render the live region (band + loader + editor + status) into
    /// `area`; `ShellRender` calls this each frame inside the inline
    /// viewport, below the committed scrollback rows.
    pub fn render_live(&mut self, frame: &mut Frame<'_>, area: Rect) {
        self.sync_frame_state();
        self.editor.set_hint(self.composer_hint());
        self.sync_vim_status();
        self.sync_composer_chrome(area.width);
        let bar_h = self.bottom_bar_rows(area.width);
        let editor_h = self
            .editor
            .desired_height(area.width)
            .min(area.height.saturating_sub(bar_h))
            .max(1);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(self.band_height()),
                Constraint::Length(self.error_banner_height(area.width)),
                Constraint::Length(u16::from(self.show_loader())),
                Constraint::Length(editor_h),
                Constraint::Length(bar_h),
                Constraint::Min(0),
            ])
            .split(area);
        self.render_band(frame, chunks[0]);
        self.render_error_banner(frame, chunks[1]);
        self.render_loader(frame, chunks[2]);
        self.editor.render(frame, chunks[3], &self.theme);
        self.render_bottom_bar(frame, chunks[4]);
        self.render_autocomplete(frame, area, chunks[3]);
        self.overlays.render(frame, area, &self.theme);
        self.extension_ui.render(frame, area, &self.theme);
    }

    /// Render the full screen (full-frame fallback renderer, used until the
    /// inline shell renderer replaces it; the fallback never commits, so
    /// the watermark stays 0 and the whole transcript renders).
    pub fn render(&mut self, frame: &mut Frame<'_>) {
        self.sync_frame_state();
        self.editor.set_hint(self.composer_hint());
        let area = frame.area();
        self.sync_composer_chrome(area.width);
        let bar_h = self.bottom_bar_rows(area.width);
        let editor_h = self
            .editor
            .desired_height(area.width)
            .min(area.height.saturating_sub(bar_h))
            .max(1);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),
                Constraint::Length(self.band_height()),
                Constraint::Length(self.error_banner_height(area.width)),
                Constraint::Length(u16::from(self.show_loader())),
                Constraint::Length(editor_h),
                Constraint::Length(bar_h),
            ])
            .split(area);
        self.render_chat(frame, chunks[0]);
        self.render_band(frame, chunks[1]);
        self.render_error_banner(frame, chunks[2]);
        self.render_loader(frame, chunks[3]);
        self.editor.render(frame, chunks[4], &self.theme);
        self.render_bottom_bar(frame, chunks[5]);
        self.render_autocomplete(frame, area, chunks[4]);
        self.overlays.render(frame, area, &self.theme);
        self.extension_ui.render(frame, area, &self.theme);
    }

    /// Draw the completion popup bottom-anchored just above the editor box.
    fn render_autocomplete(&self, frame: &mut Frame<'_>, area: Rect, editor: Rect) {
        let above = Rect {
            x: editor.x,
            y: area.y,
            width: editor.width,
            height: editor.y.saturating_sub(area.y),
        };
        self.autocomplete.render(frame, above, &self.theme);
    }

    /// The queued-messages band (§5.2 ANSI visuals): a spacer, per-group
    /// headings `Steering·n` / `After yield·n`, dim numbered rows, and a
    /// trailing `└ alt+up to edit` hint.
    fn render_band(&self, frame: &mut Frame<'_>, area: Rect) {
        use ratatui::widgets::Paragraph;
        if area.height == 0 {
            return;
        }
        let muted = Style::default().fg(self.theme.color("text_muted"));
        let mut lines: Vec<Line<'static>> = Vec::new();
        lines.push(Line::default());
        for (label, msgs) in [
            ("Steering", &self.steering),
            ("After yield", &self.follow_up),
        ] {
            if msgs.is_empty() {
                continue;
            }
            lines.push(Line::from(Span::styled(
                format!("{label}·{}", msgs.len()),
                muted,
            )));
            for (i, m) in msgs.iter().enumerate() {
                let text: String = m.replace('\t', " ").lines().collect::<Vec<_>>().join(" ↵ ");
                lines.push(Line::from(Span::styled(
                    format!("  {}. {text}", i + 1),
                    Style::default().fg(self.theme.color("text_muted")),
                )));
            }
        }
        lines.push(Line::from(Span::styled(
            format!("  └ {} to edit", self.dequeue_hint()),
            muted,
        )));
        frame.render_widget(Paragraph::new(lines), area);
    }

    /// The `app.message.dequeue` binding, formatted for the band hint
    /// (TS `formatKeyHint`): first resolved key, static fallback.
    fn dequeue_hint(&self) -> String {
        self.keys
            .first_key("app.message.dequeue")
            .unwrap_or_else(|| "alt+up".to_owned())
    }

    /// The "Working…" loader row (spinner + optional maintenance label).
    fn render_loader(&self, frame: &mut Frame<'_>, area: Rect) {
        use ratatui::widgets::Paragraph;
        if area.height == 0 {
            return;
        }
        let frame_idx = usize::try_from(
            self.spinner_start.elapsed().as_millis() / u128::from(SPINNER_ADVANCE_MS),
        )
        .unwrap_or(usize::MAX)
            % SPINNER.len();
        // Live token usage only while streaming (not during maintenance) and
        // only once there is something to show; dropped on narrow terminals.
        let usage = if area.width >= 60 {
            self.loader_usage_text()
        } else {
            None
        };
        if self.theme.clean() && self.maintenance.is_none() {
            let line = self.clean_loader_line(frame_idx, usage.as_deref());
            frame.render_widget(Paragraph::new(line), area);
            return;
        }
        let label = match self.maintenance {
            Some(Maintenance::Compaction) => " compacting… (esc to cancel)",
            Some(Maintenance::Retry) => " retrying… (esc to cancel)",
            None => " Working…",
        };
        let text = match &usage {
            Some(u) => format!("{} {label} · {u}", SPINNER[frame_idx]),
            None => format!("{} {label}", SPINNER[frame_idx]),
        };
        let line = Line::from(Span::styled(
            text,
            Style::default().fg(self.theme.color("accent")),
        ));
        frame.render_widget(Paragraph::new(line), area);
    }

    /// The live usage suffix for the loader row (`↑in ↓out (think t) · $c`),
    /// `None` until any token count is known this turn.
    fn loader_usage_text(&self) -> Option<String> {
        if !self.status.streaming {
            return None;
        }
        let input = self.turn_usage.input_tokens(self.status.context_tokens);
        let output = self.turn_usage.output_tokens();
        if input == 0 && output == 0 {
            return None;
        }
        let mut parts = vec![
            format!("↑{}", format_tokens(input)),
            format!("↓{}", format_tokens(output)),
        ];
        let think = self.turn_usage.thinking_tokens();
        if think > 0 {
            parts.push(format!("(think {})", format_tokens(think)));
        }
        if let Some((in_per_m, out_per_m)) = self.model_pricing {
            if let Some(cost) = self.turn_usage.cost(input, output, in_per_m, out_per_m) {
                parts.push(format!("${cost:.3}"));
            }
        }
        Some(parts.join(" "))
    }

    /// Re-resolve the active model's pricing (loader cost suffix) from the
    /// `/model` catalogue; `None` when the model is unknown or unpriced.
    pub(super) fn refresh_model_pricing(&mut self) {
        let model = self.status.model.clone();
        self.model_pricing = self
            .overlays
            .models_mut()
            .models
            .iter()
            .find(|m| format!("{}/{}", m.provider, m.id) == model)
            .map(|m| (m.cost_input, m.cost_output))
            .filter(|(i, o)| *i > 0.0 || *o > 0.0);
    }

    /// Clean loader: spinner, what is happening, Omi's face, elapsed time.
    fn clean_loader_line(&self, frame_idx: usize, usage: Option<&str>) -> Line<'static> {
        use crate::chrome::mascot::{face, Mood};
        let tool_running = self.messages.live_tail().iter().any(|e| match &e.kind {
            crate::chat::ChatKind::Tool(card) => !card.status.settled(),
            crate::chat::ChatKind::Bash(x) | crate::chat::ChatKind::Python(x) => {
                x.status == crate::chat::ExecStatus::Running
            }
            crate::chat::ChatKind::ReadGroup(g) => g.reads.iter().any(|r| !r.status.settled()),
            _ => false,
        });
        let (mood, label) = if tool_running {
            (Mood::Working, "Working…")
        } else {
            (Mood::Thinking, "Thinking…")
        };
        let elapsed = self.spinner_start.elapsed();
        let tick = u64::try_from(elapsed.as_millis() / 700).unwrap_or(u64::MAX);
        let accent = Style::default().fg(self.theme.color("accent"));
        let dim = Style::default().fg(self.theme.color("dim"));
        let pink = Style::default().fg(ratatui::style::Color::Rgb(236, 112, 214));
        let tail = match usage {
            Some(u) => format!("  {}s · {u} · esc to interrupt", elapsed.as_secs()),
            None => format!("  {}s · esc to interrupt", elapsed.as_secs()),
        };
        Line::from(vec![
            Span::styled(format!("{} ", SPINNER[frame_idx]), accent),
            Span::styled(
                label.to_owned(),
                accent.add_modifier(ratatui::style::Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(face(mood, tick).to_owned(), pink),
            Span::styled(tail, dim),
        ])
    }

    /// The uncommitted chat tail: entries at/after the commit watermark
    /// only — committed rows live in terminal scrollback and must never be
    /// re-drawn here (`ShellRender` invariant). Rows come from the chat block
    /// renderer (`chat::blocks`); the streaming reveal controller is ticked
    /// once per frame here in the draw loop (chat.md §8).
    fn render_chat(&mut self, frame: &mut Frame<'_>, area: Rect) {
        use ratatui::widgets::Paragraph;
        // The transcript renderer's theme is synced by `set_theme` only
        // (startup load + `/theme` hot-swap), not per frame.
        self.messages.tick_reveal();
        let mut area = area;
        let now = Instant::now();
        if let Some(w) = self
            .welcome
            .as_mut()
            .filter(|w| !w.emitted && w.started.is_none())
        {
            w.started = Some(now);
        }
        if let Some(phase) = self.welcome_shimmer(now) {
            // Startup shimmer: the welcome sits at the top of the live
            // region — exactly where its rows land when they commit.
            let rows = self.welcome_rows(area.width, phase);
            let h = u16::try_from(rows.len())
                .unwrap_or(u16::MAX)
                .min(area.height);
            frame.render_widget(Paragraph::new(rows), Rect { height: h, ..area });
            area.y += h;
            area.height -= h;
        }
        let height = area.height as usize;
        let width = usize::from(area.width).max(8);
        // Bottom-anchored: the last `height + scroll` rows of the tail.
        let lines = self
            .messages
            .tail_rows(width, height + self.scroll as usize);
        frame.render_widget(Paragraph::new(lines), area);
    }
}
