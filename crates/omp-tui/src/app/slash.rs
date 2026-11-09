use super::{
    parse_model_string, Action, App, OpenOverlay, RuntimePort, TreeMode, Value, CONFIRM_BARE,
    P0_BUILTINS,
};

impl App {
    /// Steps 7-8: bare-slash confirmation arming and builtin dispatch.
    pub(super) fn slash_pipeline(&mut self, body: &str, port: &impl RuntimePort) -> bool {
        let mut parts = body.splitn(2, ' ');
        let name = parts.next().unwrap_or("");
        let args = parts.next().unwrap_or("").trim();

        // Local P0 builtins plus anything the runtime announced in its
        // catalog (builtins, extension and file commands, incl. aliases) go
        // through dispatch; the runtime reports commands it cannot run.
        let in_catalog = self
            .autocomplete
            .commands()
            .iter()
            .any(|c| c.name == name || c.aliases.iter().any(|a| a == name));
        if !P0_BUILTINS.contains(&name) && !in_catalog {
            self.push_system(&format!("unknown command: /{name}"));
            return true;
        }

        // Step 7: bare confirm-on-first-enter commands.
        if args.is_empty() && CONFIRM_BARE.contains(&name) {
            if self.armed_bare_command.as_deref() == Some(name) {
                self.armed_bare_command = None;
            } else {
                self.armed_bare_command = Some(name.to_owned());
                self.push_system(&format!(
                    "Press Enter again to run /{name}; add a leading space to send \"/{name}\" as a message"
                ));
                return true;
            }
        }
        self.execute_slash(name, args, port)
    }

    /// Step 8: run a P0 builtin slash command (`exit` locally quits, `help`
    /// opens the keyhints overlay, `model`/`resume`/`tree` open their
    /// overlays, `theme` hot-swaps locally, the rest dispatch to the
    /// runtime's slash module via [`Action::Slash`]).
    pub(super) fn execute_slash(
        &mut self,
        name: &str,
        args: &str,
        port: &impl RuntimePort,
    ) -> bool {
        self.armed_bare_command = None;
        match name {
            "exit" => self.quit_requested = true,
            "help" | "hotkeys" => self.overlays.toggle(OpenOverlay::Help),
            "settings" => self.slash_settings(port),
            // TS `/switch` opens the same model selector (session-only pick).
            "model" | "switch" => self.slash_model(name, args, port),
            "effort" if args.is_empty() => self.open_thinking_selector(),
            "resume" => self.slash_resume(),
            "tree" => {
                if self.journal.is_empty() {
                    self.push_system("no session tree yet");
                } else {
                    self.overlays
                        .open_tree(&self.journal.clone(), TreeMode::Tree);
                }
            }
            "theme" => self.slash_theme(args),
            // TS `/session` (bare or `info`) opens the session info panel;
            // other verbs (delete, pin, …) stay with the runtime.
            "session" if matches!(args.trim(), "" | "info") => port.request_session_info(),
            // TS `/usage` (bare or `show`) opens the usage panel; `reset`
            // stays with the runtime.
            "usage" if matches!(args.trim(), "" | "show") => port.request_usage(),
            _ => self.dispatch(
                Action::Slash {
                    name: name.to_owned(),
                    args: args.to_owned(),
                },
                port,
            ),
        }
        true
    }

    /// `/model`: bare or query form opens the picker (overlays-b P0 #1);
    /// an exact `provider/id` that exists in the catalogue still goes
    /// through the runtime slash dispatch (TS `setModel` path). Picking a
    /// row drains `Action::SetModel` → `port.set_model`.
    fn slash_model(&mut self, name: &str, args: &str, port: &impl RuntimePort) {
        let exact = parse_model_string(args).is_some_and(|(provider, id, _)| {
            let selector = format!("{provider}/{id}").to_lowercase();
            self.overlays
                .models_mut()
                .models
                .iter()
                .any(|m| m.selector().to_lowercase() == selector)
        });
        if exact {
            self.dispatch(
                Action::Slash {
                    name: name.to_owned(),
                    args: args.to_owned(),
                },
                port,
            );
            return;
        }
        // Picker path: open (or re-open) and pre-fill the search query.
        if self.overlays.open() == Some(OpenOverlay::Models) {
            self.overlays.pop();
        }
        self.overlays.toggle(OpenOverlay::Models);
        if !args.is_empty() {
            let picker = self.overlays.models_mut();
            args.clone_into(&mut picker.query);
            picker.selected = 0;
        }
    }

    /// `/settings`: open the tabbed settings overlay, seeded with the
    /// effective values from the port (TS opens the settings selector with a
    /// `cfg://` snapshot). The overlay stays open; edits drain as
    /// `Action::SettingChanged`.
    fn slash_settings(&mut self, port: &impl RuntimePort) {
        self.overlays.open_settings();
        let overlay = self.overlays.settings_mut();
        let ids = overlay.ids();
        let values = port.settings_get(&ids);
        overlay.seed(&values);
        overlay.set_available_themes(
            crate::theme::available_themes()
                .into_iter()
                .map(|t| t.name)
                .collect(),
        );
        overlay.set_thinking_levels(port.available_thinking_levels());
    }

    /// `/resume`: open the embedded session picker (overlays-b P0 #10); the
    /// pick resolves to `Action::ResumeSession` via the overlay drain.
    fn slash_resume(&mut self) {
        if self.session_sources.is_empty() {
            self.push_system("session storage is not available in this session");
            return;
        }
        self.overlays.open_sessions(&self.session_sources);
    }

    /// Live half of a persisted settings edit: apply what omp-rs can right
    /// now (TS applies everything through the cfg:// host; here only theme
    /// and thinking level have live consumers, everything else is saved for
    /// the next runtime read and flagged "no live effect" in the overlay).
    pub(super) fn apply_setting_live(&mut self, id: &str, value: &Value, port: &impl RuntimePort) {
        match id {
            "theme.dark" | "theme.light" | "symbolPreset" | "tui.style" => {
                // Re-read the layered settings and hot-swap the theme for
                // the terminal's current appearance.
                match self.load_settings_theme() {
                    Ok(()) => self.push_system(&format!("{id} applied")),
                    Err(e) => {
                        self.push_system(&format!("{id} saved, but theme reload failed: {e}"));
                    }
                }
            }
            "defaultThinkingLevel" => {
                if let Some(level) = value.as_str().filter(|level| *level != "auto") {
                    port.set_thinking_level(level.to_string());
                }
            }
            "display.showTokenUsage"
            | "display.showTurnTime"
            | "read.toolResultPreview"
            | "tui.mouse" => {
                self.reload_display_prefs();
            }
            id if id.starts_with("magicKeywords.") => {
                self.magic_off = crate::magic::disabled_mask(&self.load_settings());
                let theme = self.theme.clone();
                self.set_theme(theme);
            }
            "composer.shape"
            | "statusLine.preset"
            | "statusLine.separator"
            | "statusLine.contextLine"
            | "statusLine.sessionAccent"
            | "statusLine.transparent"
            | "statusLine.compactThinkingLevel"
            | "statusLine.showHookStatus"
            | "statusLine.leftSegments"
            | "statusLine.rightSegments"
            | "tui.vimMode"
            | "tui.vimModeDisplay"
            | "spelling.autocomplete"
            | "autocompleteMaxVisible" => {
                let settings = self.load_settings();
                self.reload_composer(&settings);
            }
            _ => {}
        }
    }

    /// `/theme`: bare lists the loadable themes; with a name it hot-swaps
    /// the app theme immediately (chrome.md §7, settings hot-swap half).
    fn slash_theme(&mut self, args: &str) {
        if args.is_empty() {
            let names: Vec<String> = crate::theme::available_themes()
                .into_iter()
                .map(|t| t.name)
                .collect();
            self.push_system(&format!("available themes: {}", names.join(", ")));
            return;
        }
        match crate::theme::load(args) {
            Ok(theme) => {
                self.set_theme(theme);
                self.push_system(&format!("theme set to {args}"));
            }
            Err(e) => self.push_system(&format!("theme error: {e}")),
        }
    }
}
