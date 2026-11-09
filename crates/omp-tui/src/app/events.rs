use super::{
    journal_entry, now_millis, turn_notification, welcome, Action, App, AppEvent, Duration,
    Instant, Line, Maintenance, OpenOverlay, RuntimePort, TerminalNotification, TitleState, Value,
    LOCAL_COMMANDS, SPINNER_ADVANCE_MS,
};

impl App {
    /// Mirror the runtime's thinking state (`thinkingLevel`, supported
    /// `thinkingLevels`, `autoThinking`) into the status line; a model
    /// without thinking shows no level (TS `thinkingLevelWord`).
    fn apply_thinking_state(&mut self, frame: &Value) {
        if let Some(levels) = frame.get("thinkingLevels").and_then(Value::as_array) {
            self.thinking_levels = levels
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
        }
        if let Some(auto) = frame.get("autoThinking").and_then(Value::as_bool) {
            self.status.auto_thinking = auto;
        }
        if let Some(level) = frame.get("thinkingLevel").and_then(Value::as_str) {
            self.set_thinking_display(level);
        }
        self.apply_ultra(frame);
        if self.thinking_levels.is_empty() {
            self.status.thinking = None;
            self.status.auto_thinking = false;
        }
    }

    /// Mirror the runtime's ultra mode (`"ultra": "<name>" | null`, wire
    /// contract 1); frames without the key leave it unchanged.
    fn apply_ultra(&mut self, frame: &Value) {
        if let Some(ultra) = frame.get("ultra") {
            self.status.ultra = ultra.as_str().and_then(crate::magic::Ultra::parse);
        }
    }

    fn set_thinking_display(&mut self, level: &str) {
        self.status.thinking = (!self.thinking_levels.is_empty()).then(|| level.to_owned());
    }

    /// Selector values for the active model in cycle order
    /// (`getAvailableEffortSelectors`): `off`, `auto`, then the efforts, then
    /// the omp-rs `ultracode` mode.
    pub(super) fn effort_selectors(&self) -> Vec<String> {
        if self.thinking_levels.is_empty() {
            return Vec::new();
        }
        ["off", "auto"]
            .iter()
            .map(|s| (*s).to_owned())
            .chain(self.thinking_levels.iter().cloned())
            .chain(std::iter::once(crate::magic::Ultra::Code.name().to_owned()))
            .collect()
    }

    /// The configured selector (`configuredThinkingLevel`): the ultra mode
    /// while one is active, `auto` while auto thinking is on, else the
    /// effective level. Any ultra mode (incl. the hybrids, which are not in
    /// the cycle) is followed by `off`.
    pub(super) fn current_effort_selector(&self) -> String {
        if let Some(ultra) = self.status.ultra {
            ultra.name().to_owned()
        } else if self.status.auto_thinking {
            "auto".to_owned()
        } else {
            self.status
                .thinking
                .clone()
                .unwrap_or_else(|| "off".to_owned())
        }
    }

    /// Shift+Tab (`app.thinking.cycle`): off → auto → minimal..max →
    /// ultracode → off (TS `cycleThinkingLevel` + omp-rs ultra).
    pub(super) fn cycle_thinking_level(&mut self, port: &impl RuntimePort) {
        let levels = self.effort_selectors();
        if levels.is_empty() {
            self.push_system("Current model does not support thinking");
            return;
        }
        let current = self.current_effort_selector();
        let next = levels
            .iter()
            .position(|l| *l == current)
            .map_or(0, |idx| (idx + 1) % levels.len());
        self.dispatch(Action::SetThinkingLevel(levels[next].clone()), port);
    }

    /// Bare `/effort`: open the thinking-level selector (TS
    /// `showThinkingSelector`).
    pub(super) fn open_thinking_selector(&mut self) {
        let mut levels = self.effort_selectors();
        if levels.is_empty() {
            self.push_system("Current model does not support thinking");
            return;
        }
        // TS lists `off, auto, …`; the selector shows the same order.
        let current = self.current_effort_selector();
        self.overlays
            .open_thinking(std::mem::take(&mut levels), &current);
    }

    /// Apply a `context_usage` frame: model identity plus the context gauge
    /// inputs (percent, window, compaction boundaries).
    fn apply_context_usage(&mut self, frame: &Value) {
        if let Some(m) = frame.get("model").and_then(Value::as_str) {
            if self.status.model != m {
                m.clone_into(&mut self.status.model);
                self.overlays.models_mut().current = Some(m.to_owned());
            }
        }
        if let Some(name) = frame.get("modelName") {
            self.status.model_name = name.as_str().filter(|n| !n.is_empty()).map(str::to_owned);
        }
        self.apply_thinking_state(frame);
        self.status.context_percent = frame.get("percent").and_then(Value::as_f64);
        self.status.context_tokens = frame.get("tokens").and_then(Value::as_u64);
        self.status.context_window = frame
            .get("contextWindow")
            .and_then(Value::as_u64)
            .filter(|w| *w > 0);
        self.status.compaction_pct = frame.get("thresholdPercent").and_then(Value::as_f64);
        self.status.speculation_pct = frame.get("speculationPercent").and_then(Value::as_f64);
        self.status.boundaries_known = self.status.compaction_pct.is_some();
    }

    /// The right-aligned composer hint (`resolveComposerHint`): the
    /// thinking-effort gesture until the conversation starts.
    pub(super) fn composer_hint(&self) -> Option<(String, String)> {
        if !self.messages.entries().is_empty() {
            return None;
        }
        let key = self.keys.first_key("app.thinking.cycle")?;
        Some((
            welcome::format_key_hint(&key),
            "to change thinking effort".to_owned(),
        ))
    }

    /// Enqueue one raw runtime frame for serialized, coalesced dispatch.
    pub fn push_frame(&mut self, frame: &Value, now: Instant) {
        self.pump.push_frame(frame, now);
    }

    /// Owned-frame variant of [`Self::push_frame`]: a `session_loaded`
    /// history is moved into its event instead of cloned.
    pub fn push_frame_owned(&mut self, frame: Value, now: Instant) {
        self.pump.push_frame_owned(frame, now);
    }

    /// Apply all due events. Returns `true` when anything was applied
    /// (the controller redraws after a pump that returns `true`).
    pub fn pump_events(&mut self, now: Instant) -> bool {
        // The pump lends itself out while draining to satisfy the borrow
        // checker (it calls back into `apply_event`).
        let mut pump = std::mem::take(&mut self.pump);
        let applied = pump.drain(self, now);
        self.pump = pump;
        applied > 0
    }

    /// Extension dialogs can time out; the controller calls this every
    /// tick. Expired dialogs are answered here (`cancelled` + `timedOut`).
    pub fn poll_deadlines(&mut self, now: Instant, port: &impl RuntimePort) {
        for (id, value) in self.extension_ui.poll_deadlines(now) {
            port.extension_ui_response(id, value);
        }
        self.poll_paste_results();
    }

    /// Apply finished paste results: a clipboard read (paste chord) and a
    /// resolved large-paste menu pick. The controller wakes on
    /// [`App::has_pending_paste_results`] so this runs — and the following
    /// paint lands — even when the loop would otherwise block with no
    /// terminal events.
    pub fn poll_paste_results(&mut self) {
        if let Some(rx) = &self.clipboard_rx {
            if let Ok(outcome) = rx.try_recv() {
                self.clipboard_rx = None;
                self.apply_clipboard_outcome(outcome);
            }
        }
        if let Some(pick) = self.overlays.large_paste_mut().take_pick() {
            self.apply_large_paste_pick(pick);
        }
    }

    /// Whether paste results are pending: a clipboard read in flight or an
    /// open large-paste menu whose pick still needs draining.
    #[must_use]
    pub fn has_pending_paste_results(&self) -> bool {
        self.clipboard_rx.is_some() || self.overlays.stack().contains(&OpenOverlay::LargePaste)
    }

    /// Whether a `message_update` is still inside its coalesce window; the
    /// controller uses this to schedule the next pump tick.
    #[must_use]
    pub fn has_pending_update(&self) -> bool {
        self.pump.has_pending_update()
    }

    /// The frame-timer period the controller loop must arm right now, or
    /// `None` when nothing animates (idle loops must not wake): 30 fps
    /// while the streaming reveal has a backlog (TS
    /// `STREAMING_REVEAL_FRAME_MS`), the 80 ms spinner step while the
    /// loader row is visible (TS `SPINNER_ADVANCE_MS`).
    #[must_use]
    pub fn animation_interval(&self) -> Option<Duration> {
        if self.messages.reveal_animating() || self.welcome_shimmer(Instant::now()).is_some() {
            Some(Duration::from_millis(crate::chat::reveal::FRAME_MS))
        } else if self.editor_shimmers() {
            Some(Duration::from_millis(crate::magic_paint::SHIMMER_FRAME_MS))
        } else if self.show_loader() {
            Some(Duration::from_millis(SPINNER_ADVANCE_MS))
        } else {
            None
        }
    }

    /// Startup shimmer phase of the clean welcome while it is still held in
    /// the live region (not yet committed), `None` once it froze.
    pub(super) fn welcome_shimmer(&self, now: Instant) -> Option<f64> {
        let w = self
            .welcome
            .as_ref()
            .filter(|w| !w.emitted && self.theme.clean())?;
        welcome::shimmer_phase(
            w.started
                .map_or(Duration::ZERO, |s| now.saturating_duration_since(s)),
        )
    }

    /// Welcome block rows at `width` (clean: OMT logo at `phase`).
    pub(super) fn welcome_rows(&self, width: u16, phase: f64) -> Vec<Line<'static>> {
        let Some(w) = &self.welcome else {
            return Vec::new();
        };
        if self.theme.clean() {
            welcome::clean_welcome_lines(&w.version, &w.tip, usize::from(width), &self.theme, phase)
        } else {
            welcome::welcome_lines(&w.version, &w.tip, usize::from(width), &self.theme)
        }
    }

    /// Whether the editor's magic keywords shimmer (TS: focused editor with
    /// a keyword on screen); no keyword → no animation timer.
    fn editor_shimmers(&self) -> bool {
        self.overlays.open().is_none()
            && !self.extension_ui.is_active()
            && crate::magic_paint::any_keyword(self.editor.buffer.lines(), self.theme.magic_off)
    }

    /// The clean status line's τ state: blocked on the user (dialog / ask)
    /// beats running, which beats a pinned turn error, else idle.
    pub(super) fn brand_state(&self) -> crate::status_line::Brand {
        use crate::status_line::Brand;
        let asking = self.messages.live_tail().iter().any(|e| {
            matches!(&e.kind, crate::chat::ChatKind::Tool(card) if card.name == "ask" && !card.status.settled())
        });
        if self.extension_ui.pending_len() > 0 || asking {
            Brand::Blocked
        } else if self.status.streaming || self.pending_submission {
            Brand::Running
        } else if self.messages.pinned_error().is_some() {
            Brand::Error
        } else {
            Brand::Idle
        }
    }

    /// Per-frame live state the status line and editor render from.
    pub(super) fn sync_frame_state(&mut self) {
        self.status.tick(now_millis());
        self.status.brand = self.brand_state();
        let phase = if self.editor_shimmers() {
            crate::magic_paint::shimmer_phase(now_millis())
        } else {
            0.0
        };
        self.editor.set_magic_phase(phase);
    }

    /// Apply one classified runtime event directly (tests + compat path).
    // One arm per runtime event keeps the event taxonomy in one place.
    #[allow(clippy::too_many_lines)]
    pub fn apply_event(&mut self, event: &AppEvent) {
        match &event {
            AppEvent::AgentStart => {
                self.last_stop_reason = None;
                self.spinner_start = Instant::now();
                self.turn_usage.reset();
                self.status.streaming = true;
                self.pending_submission = false;
                self.title
                    .set_state(TitleState::Working, Some(&self.title_label));
            }
            AppEvent::TurnStart => {
                self.turn_usage.reset();
                self.status.streaming = true;
                self.pending_submission = false;
            }
            AppEvent::AgentEnd(_) | AppEvent::TurnEnd(_) => {
                self.status.streaming = false;
                self.title
                    .set_state(TitleState::Idle, Some(&self.title_label));
            }
            AppEvent::SessionSettled => {
                self.status.streaming = false;
                self.title
                    .set_state(TitleState::Idle, Some(&self.title_label));
                self.notify_turn_settled(false);
            }
            AppEvent::PromptResult { success, .. } => {
                if !success {
                    // The transcript row is drawn by `messages.apply` below;
                    // here only the title and the settle notification.
                    self.title
                        .set_state(TitleState::Idle, Some(&self.title_label));
                    self.notify_turn_settled(true);
                }
            }
            AppEvent::MessageEnd(frame) => {
                if let Some(message) = frame.get("message") {
                    self.turn_usage.observe_message(message);
                }
                let message = frame.get("message");
                if message.and_then(|m| m.get("role")).and_then(Value::as_str) == Some("assistant")
                {
                    self.last_stop_reason = message
                        .and_then(|m| m.get("stopReason"))
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                }
                if let Some(entry) = journal_entry(frame) {
                    self.journal.push(entry);
                }
            }
            AppEvent::SessionNameChanged(name) => {
                self.title_label.clone_from(name);
                self.session_name = Some(name.clone()).filter(|n| !n.is_empty());
                // Re-emit the current state with the new label.
                let state = if self.status.streaming {
                    TitleState::Working
                } else {
                    TitleState::Idle
                };
                self.title.set_state(state, Some(name));
            }
            AppEvent::MessageStart(_) => self.pending_submission = false,
            AppEvent::MessageUpdate(frame) => {
                if let Some(message) = frame.get("message") {
                    self.turn_usage.observe_message(message);
                }
                // `delta_updates` mode carries only `assistantMessageEvent.delta`
                // (no cumulative `message` snapshot); fold it here.
                self.turn_usage.observe_delta(frame);
            }
            AppEvent::ModelChanged(frame) => {
                if let Some(m) = frame
                    .get("model")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| {
                        let p = frame.get("provider").and_then(Value::as_str)?;
                        let id = frame.get("id").and_then(Value::as_str)?;
                        Some(format!("{p}/{id}"))
                    })
                {
                    self.status.model.clone_from(&m);
                    self.overlays.models_mut().current = Some(m);
                }
                self.refresh_model_pricing();
            }
            AppEvent::ContextUsage(frame) => {
                self.apply_context_usage(frame);
                self.refresh_model_pricing();
            }
            AppEvent::ModelsChanged { models } => {
                let picker = self.overlays.models_mut();
                picker.models.clone_from(models);
                self.refresh_model_pricing();
            }
            AppEvent::ThinkingLevelChanged(frame) => {
                let configured_auto =
                    frame.get("configured").and_then(Value::as_str) == Some("auto");
                self.status.auto_thinking = configured_auto;
                if let Some(level) = frame.get("thinkingLevel").and_then(Value::as_str) {
                    self.set_thinking_display(level);
                }
                self.apply_ultra(frame);
            }
            AppEvent::QueueUpdate {
                steering,
                follow_up,
            } => {
                self.steering.clone_from(steering);
                self.follow_up.clone_from(follow_up);
                self.status.queued = steering.len() + follow_up.len();
            }
            AppEvent::AutoCompaction(frame) => {
                let start =
                    frame.get("type").and_then(Value::as_str) == Some("auto_compaction_start");
                self.maintenance = if start {
                    Some(Maintenance::Compaction)
                } else {
                    None
                };
                self.messages.apply(event);
            }
            AppEvent::Retry(frame) => {
                if frame.get("type").and_then(Value::as_str) == Some("auto_retry_start") {
                    self.maintenance = Some(Maintenance::Retry);
                } else if self.maintenance == Some(Maintenance::Retry) {
                    self.maintenance = None;
                }
                self.messages.apply(event);
            }
            AppEvent::ExtensionUiRequest(frame) => {
                // TS rpc `set_editor_text` (fire-and-forget): replace the prompt draft.
                if frame.get("method").and_then(Value::as_str) == Some("set_editor_text") {
                    let text = frame
                        .get("text")
                        .or_else(|| frame.pointer("/params/text"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    self.set_editor_text(&text);
                } else {
                    self.extension_ui.push(frame);
                }
            }
            AppEvent::ClipboardCopy(text) => {
                // TS `copyToClipboard`: OSC 52 + platform tools, best effort, off the
                // render thread.
                {
                    use std::io::{IsTerminal as _, Write as _};
                    let mut out = std::io::stdout();
                    if out.is_terminal() {
                        let _ = out.write_all(crate::clipboard::osc52(text).as_bytes());
                        let _ = out.flush();
                    }
                }
                let text = text.clone();
                std::thread::spawn(move || crate::clipboard::write_text(&text));
            }
            AppEvent::AvailableCommands(frame) => {
                self.autocomplete.update_commands(frame);
                // Merge the TUI-local entries (`/settings`, `/hotkeys`) into
                // the announced catalog so the `/` popup offers them even
                // though the runtime does not register them (TS registers
                // both host-side).
                let mut commands = self.autocomplete.commands().to_vec();
                for (name, description) in LOCAL_COMMANDS {
                    if !commands.iter().any(|cmd| cmd.name == *name) {
                        commands.push(crate::autocomplete::SlashCommand {
                            name: (*name).to_string(),
                            description: Some((*description).to_string()),
                            source: Some(crate::autocomplete::CommandSource::Builtin),
                            ..crate::autocomplete::SlashCommand::default()
                        });
                    }
                }
                self.autocomplete.set_commands(commands);
            }
            AppEvent::CommandOutput(text) => self.push_system(text),
            AppEvent::Notice { level, message } => {
                let glyph = match level.as_str() {
                    "warning" => self.theme.symbol("status.warning"),
                    "error" => self.theme.symbol("status.error"),
                    _ => self.theme.symbol("status.info"),
                };
                self.push_system(&format!("{glyph} {message}"));
            }
            AppEvent::SessionLoaded { messages, name } => {
                // The resumed history replaces the transcript; rows already
                // in terminal scrollback stay, the history is appended below.
                self.journal.clear();
                self.status.streaming = false;
                self.messages.rebuild(messages);
                if let Some(name) = name {
                    self.title_label.clone_from(name);
                    self.session_name = Some(name.clone()).filter(|n| !n.is_empty());
                }
                self.title
                    .set_state(TitleState::Idle, Some(&self.title_label));
            }
            AppEvent::SessionReset => {
                self.journal.clear();
                self.status.streaming = false;
                self.messages.rebuild(&[]);
                self.clear_scrollback = true;
            }
            AppEvent::SessionInfo { title, lines } => {
                self.overlays
                    .open_session_info(title.as_deref(), lines.clone());
            }
            _ => {}
        }
        self.messages.apply(event);
    }

    /// Turn-settle notifications (TS `sendErrorNotification` / `sendCompletionNotification`).
    /// A failed turn (an errored assistant message or a failed prompt) rings `request` with
    /// "Stopped with error" when `error.notify` is on (default off); any other non-aborted
    /// settle rings `done` with "Complete" unless `completion.notify` is off. The title is the
    /// session name, else `omp`.
    fn notify_turn_settled(&mut self, failed: bool) {
        let settings = self.load_settings();
        let on = |id: &str| settings.get_string(id).as_deref() != Some("off");
        let Some((body, kind)) = turn_notification(
            failed,
            self.last_stop_reason.as_deref(),
            on("error.notify"),
            on("completion.notify"),
        ) else {
            return;
        };
        let mut n = TerminalNotification::body(body);
        n.title = Some(
            self.session_name
                .clone()
                .unwrap_or_else(|| "omp".to_owned()),
        );
        n.kind = kind;
        self.notify.fire(&n);
    }
}
