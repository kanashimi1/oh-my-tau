use super::turn_usage::{format_tokens, TurnUsage};
use super::*;
use crate::chat::ChatRole;
use crate::port::FakePort;
use crossterm::event::{KeyCode as CtKeyCode, KeyEvent as CtKeyEvent, KeyModifiers as CtMods};
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use serde_json::json;

/// An app pinned to built-in default settings: independent of the
/// process-global `HOME`, which the temp-HOME tests mutate.
fn app() -> App {
    let mut app = App::new("zai/glm-5.3-flash".into(), Path::new("/repo"));
    app.set_settings_global(None);
    // Capture deliveries instead of ringing the real terminal / desktop.
    app.set_notify_hook(Arc::new(|_| {}));
    app
}

fn port() -> FakePort {
    FakePort::default()
}

fn key(code: CtKeyCode) -> KeyEvent {
    CtKeyEvent::new(code, CtMods::NONE)
}

fn ctrl(code: char) -> KeyEvent {
    CtKeyEvent::new(CtKeyCode::Char(code), CtMods::CONTROL)
}

fn type_text(app: &mut App, port: &FakePort, text: &str) {
    for c in text.chars() {
        assert!(app
            .handle_key(
                KeyEvent::new(CtKeyCode::Char(c), CtMods::NONE),
                port,
                Instant::now()
            )
            .is_some());
    }
}

fn enter(app: &mut App, port: &FakePort) -> bool {
    app.handle_terminal_event(&Event::Key(key(CtKeyCode::Enter)), port)
}

fn press(app: &mut App, port: &FakePort, k: KeyEvent) -> bool {
    app.handle_terminal_event(&Event::Key(k), port)
}

fn log(port: &FakePort) -> Vec<String> {
    port.log.lock().clone()
}

fn start_streaming(app: &mut App) {
    app.apply_event(&AppEvent::AgentStart);
}

// ---- submit pipeline -------------------------------------------------

#[test]
fn typing_then_enter_submits_and_echoes() {
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "hi");
    assert!(enter(&mut app, &port));
    assert_eq!(app.messages.entries().len(), 1);
    assert_eq!(app.messages.entries()[0].role, ChatRole::User);
    assert_eq!(log(&port), vec!["submit(None, \"hi\", None)".to_string()]);
}

#[test]
fn empty_enter_while_streaming_aborts_for_drain() {
    let mut app = app();
    let port = port();
    start_streaming(&mut app);
    assert!(enter(&mut app, &port));
    assert_eq!(log(&port), vec!["abort()".to_string()]);
}

#[test]
fn dot_and_c_send_continue_directive() {
    for shortcut in [".", "c"] {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, shortcut);
        assert!(enter(&mut app, &port));
        assert_eq!(log(&port), vec!["continue_run()".to_string()]);
        assert!(app.messages.entries().is_empty()); // hidden, no user row
    }
}

#[test]
fn queue_shorthand_follows_up() {
    let mut app = app();
    let port = port();
    start_streaming(&mut app);
    type_text(&mut app, &port, ">> later");
    assert!(enter(&mut app, &port));
    assert_eq!(
        log(&port),
        vec!["submit(None, \"later\", Some(FollowUp))".to_string()]
    );
    assert_eq!(app.follow_up, vec!["later".to_string()]);
    assert_eq!(app.band_height(), 1 + 1 + 1 + 1); // spacer+heading+msg+hint
}

#[test]
fn steering_while_streaming_updates_band_optimistically() {
    let mut app = app();
    let port = port();
    start_streaming(&mut app);
    type_text(&mut app, &port, "faster");
    assert!(enter(&mut app, &port));
    assert_eq!(
        log(&port),
        vec!["submit(None, \"faster\", Some(Steer))".to_string()]
    );
    assert_eq!(app.steering, vec!["faster".to_string()]);
}

#[test]
fn free_text_while_streaming_steers() {
    let mut app = app();
    let port = port();
    start_streaming(&mut app);
    type_text(&mut app, &port, "side note");
    assert!(enter(&mut app, &port));
    assert_eq!(
        log(&port),
        vec!["submit(None, \"side note\", Some(Steer))".to_string()]
    );
}

#[test]
fn slash_model_exact_id_dispatches_to_runtime() {
    let mut app = app();
    let port = port();
    app.configure(
        vec![ModelEntry::new("zai", "glm", Some(128_000))],
        Vec::new(),
    );
    type_text(&mut app, &port, "/model zai/glm");
    assert!(enter(&mut app, &port));
    // An exact catalogue id goes through the runtime slash dispatch.
    assert_eq!(
        log(&port),
        vec!["run_slash(model, \"zai/glm\")".to_string()]
    );
    assert_eq!(app.overlays.open(), None);
}

#[test]
fn slash_model_bare_or_query_opens_picker() {
    for args in ["", "glm"] {
        let mut app = app();
        let port = port();
        app.configure(
            vec![ModelEntry::new("zai", "glm", Some(128_000))],
            Vec::new(),
        );
        type_text(&mut app, &port, &format!("/model {args}"));
        assert!(enter(&mut app, &port), "args: {args:?}");
        assert_eq!(
            app.overlays.open(),
            Some(OpenOverlay::Models),
            "args: {args:?}"
        );
        assert!(log(&port).is_empty(), "no port calls for picker");
    }
}

#[test]
fn model_picker_default_chip_dispatches_set_model_and_stays_open() {
    let _guard = HOME_LOCK.lock();
    let old_home = std::env::var_os("HOME");
    let home = with_temp_home("model-pick");
    {
        let mut app = home_app();
        let port = port();
        app.configure(
            vec![ModelEntry::new("zai", "glm", Some(128_000))],
            Vec::new(),
        );
        type_text(&mut app, &port, "/model");
        assert!(enter(&mut app, &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Models));
        // Enter focuses the model rows, Enter opens the role strip,
        // Enter on the first chip (default) switches the live model.
        assert!(enter(&mut app, &port));
        assert!(enter(&mut app, &port));
        assert!(enter(&mut app, &port));
        let calls = log(&port);
        assert_eq!(calls[0], "set_model(zai, glm)", "{calls:?}");
        assert!(
            calls
                .iter()
                .any(|c| c.starts_with("settings_set(modelRoles")),
            "{calls:?}"
        );
        // The browser stays open (TS model hub); Esc closes it.
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Models));
        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
        assert_eq!(app.overlays.open(), None);
    }
    restore_home(old_home);
    let _ = std::fs::remove_dir_all(home);
}

#[test]
fn slash_exit_quits_after_bare_confirmation() {
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "/exit");
    // Bare /exit must be confirmed with a second Enter (§4.1 step 7).
    // Bare /exit must be confirmed: first Enter arms, second quits.
    assert!(enter(&mut app, &port));
    assert!(!enter(&mut app, &port));
}

#[test]
fn slash_help_opens_help_overlay() {
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "/help");
    assert!(enter(&mut app, &port));
    assert_eq!(app.overlays.open(), Some(OpenOverlay::Help));
}

#[test]
fn slash_hotkeys_opens_help_overlay() {
    // TS alias: `/hotkeys` is `/help`.
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "/hotkeys");
    assert!(enter(&mut app, &port));
    assert_eq!(app.overlays.open(), Some(OpenOverlay::Help));
}

#[test]
fn slash_session_info_requests_report_and_opens_overlay() {
    let mut app = app();
    let port = port();
    app.apply_event(&AppEvent::AvailableCommands(json!({
        "commands": [{"name": "session", "source": "builtin"}]
    })));
    type_text(&mut app, &port, "/session");
    assert!(enter(&mut app, &port));
    assert!(
        log(&port).contains(&"request_session_info()".to_owned()),
        "{:?}",
        log(&port)
    );
    assert!(
        !log(&port).iter().any(|c| c.starts_with("run_slash")),
        "{:?}",
        log(&port)
    );
    // Other verbs still reach the runtime.
    type_text(&mut app, &port, "/session delete");
    assert!(enter(&mut app, &port));
    assert!(
        log(&port)
            .iter()
            .any(|c| c.starts_with("run_slash(session")),
        "{:?}",
        log(&port)
    );
    // The runtime's report frame opens the full-screen panel; Esc closes it.
    app.apply_event(&crate::events::parse_frame(&json!({
        "type": "session_info",
        "lines": [{"k": "ID", "v": "abc"}, {"head": "Provider"}, {"text": ""}]
    })));
    assert_eq!(app.overlays.open(), Some(OpenOverlay::SessionInfo));
    assert!(app.has_fullscreen_overlay());
    let s = screen(&mut app, 120, 40);
    assert!(s.contains("Session Info") && s.contains("ID: abc"), "{s}");
    assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
    assert_eq!(app.overlays.open(), None);
}

#[test]
fn slash_usage_requests_report_and_opens_panel() {
    let mut app = app();
    let port = port();
    app.apply_event(&AppEvent::AvailableCommands(json!({
        "commands": [{"name": "usage", "source": "builtin"}]
    })));
    type_text(&mut app, &port, "/usage");
    assert!(enter(&mut app, &port));
    assert!(
        log(&port).contains(&"request_usage()".to_owned()),
        "{:?}",
        log(&port)
    );
    assert!(
        !log(&port).iter().any(|c| c.starts_with("run_slash")),
        "{:?}",
        log(&port)
    );
    type_text(&mut app, &port, "/usage reset");
    assert!(enter(&mut app, &port));
    assert!(
        log(&port).iter().any(|c| c.starts_with("run_slash(usage")),
        "{:?}",
        log(&port)
    );
    app.apply_event(&crate::events::parse_frame(&json!({
        "type": "session_info", "title": "Usage",
        "lines": [{"head": "Provider limits (anthropic)"}, {"k": "5h", "v": "6%"}]
    })));
    assert_eq!(app.overlays.open(), Some(OpenOverlay::SessionInfo));
    let s = screen(&mut app, 120, 40);
    assert!(
        s.contains("Usage") && !s.contains("Session Info") && s.contains("5h: 6%"),
        "{s}"
    );
    assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
    assert_eq!(app.overlays.open(), None);
}

#[test]
fn slash_settings_opens_overlay_listing_sections() {
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "/settings");
    assert!(enter(&mut app, &port));
    assert_eq!(app.overlays.open(), Some(OpenOverlay::Settings));
    let s = screen(&mut app, 150, 30);
    // Fullscreen tab strip (TS labels) and the first tab's section
    // headings (sidebar + pane).
    assert!(s.contains("Appearance"), "{s}");
    assert!(s.contains("Model") && s.contains("Providers"), "{s}");
    assert!(s.contains("Theme"), "{s}");
    // Row labels render with their values (values are seeded from the
    // layered config, so assert the stable label half).
    assert!(s.contains("Dark Theme") && s.contains("Light Theme"), "{s}");
}

#[test]
fn settings_popup_lists_local_commands() {
    let mut app = app();
    let port = port();
    with_catalog(&mut app);
    type_text(&mut app, &port, "/se");
    let s = screen(&mut app, 60, 14);
    assert!(s.contains("settings"), "{s}");
    // New query: drop the old draft and type the other local command.
    for _ in 0..3 {
        assert!(press(&mut app, &port, key(CtKeyCode::Backspace)));
    }
    type_text(&mut app, &port, "/hot");
    let s = screen(&mut app, 60, 14);
    assert!(s.contains("hotkeys"), "{s}");
}

use crate::test_support::HOME_LOCK;

/// An app reading the user's config through `HOME`, for the temp-HOME
/// tests (call only while holding [`HOME_LOCK`]).
fn home_app() -> App {
    App::new("zai/glm-5.3-flash".into(), Path::new("/repo"))
}

fn with_temp_home(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("omp-tui-settings-{test}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::env::set_var("HOME", &dir);
    std::env::remove_var("OMP_PROFILE");
    dir
}

fn restore_home(old: Option<std::ffi::OsString>) {
    match old {
        Some(h) => std::env::set_var("HOME", h),
        None => std::env::remove_var("HOME"),
    }
}

#[test]
fn settings_bool_toggle_writes_config_and_rereads() {
    let _guard = HOME_LOCK.lock();
    let old_home = std::env::var_os("HOME");
    let home = with_temp_home("bool");
    {
        let mut app = home_app();
        let port = port();
        type_text(&mut app, &port, "/settings");
        assert!(enter(&mut app, &port));
        // Defaults before the edit.
        assert_eq!(
            app.overlays.settings_mut().value_of("colorBlindMode"),
            Some(json!(false))
        );
        // Cycle the bool row with ⏎ (TS confirm; → switches tabs).
        assert!(app.overlays.settings_mut().select("colorBlindMode"));
        assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
        // The overlay shows the new value and stays open.
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Settings));
        assert_eq!(
            app.overlays.settings_mut().value_of("colorBlindMode"),
            Some(json!(true))
        );
        // The write landed in ~/.omp/agent/config.yml…
        let config = home.join(".omp/agent/config.yml");
        let text = std::fs::read_to_string(&config).unwrap();
        assert!(text.contains("colorBlindMode: true"), "{text}");
        // …and a fresh layered read sees it (port round trip).
        assert_eq!(
            port.settings_get(&["colorBlindMode".to_string()])[0].1,
            json!(true)
        );
        let logged = log(&port);
        assert!(
            logged
                .iter()
                .any(|entry| entry.starts_with("settings_set(colorBlindMode")),
            "{logged:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&home);
    restore_home(old_home);
}

#[test]
fn settings_theme_change_applies_live() {
    let _guard = HOME_LOCK.lock();
    let old_home = std::env::var_os("HOME");
    let home = with_temp_home("theme");
    {
        let mut app = home_app();
        let port = port();
        app.set_appearance(crate::theme::Appearance::Dark);
        type_text(&mut app, &port, "/settings");
        assert!(enter(&mut app, &port));
        assert!(app.overlays.settings_mut().select("theme.dark"));
        // Submenu row: Enter opens the theme select field with the
        // current theme highlighted; ↓ to "light", Enter picks.
        assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
        let names: Vec<String> = crate::theme::available_themes()
            .into_iter()
            .map(|t| t.name)
            .collect();
        let from = names.iter().position(|n| n == "titanium").unwrap();
        let to = names.iter().position(|n| n == "light").unwrap();
        for _ in 0..(to + names.len() - from) % names.len() {
            assert!(press(&mut app, &port, key(CtKeyCode::Down)));
        }
        assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
        // Saved…
        assert_eq!(
            app.overlays.settings_mut().value_of("theme.dark"),
            Some(json!("light"))
        );
        let config = home.join(".omp/agent/config.yml");
        assert!(std::fs::read_to_string(&config).unwrap().contains("light"));
        // …and applied: the app theme is now the light theme.
        let expected = crate::theme::load("light").unwrap();
        assert_eq!(app.theme.color("text"), expected.color("text"));
        assert_eq!(app.theme.color("accent"), expected.color("accent"));
        // A notice confirms the live apply; it renders in the chat once
        // the fullscreen overlay closes.
        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
        assert_eq!(app.overlays.open(), None);
        let s = screen(&mut app, 150, 30);
        assert!(s.contains("theme.dark applied"), "{s}");
    }
    let _ = std::fs::remove_dir_all(&home);
    restore_home(old_home);
}

fn with_catalog(app: &mut App) {
    app.apply_event(&AppEvent::AvailableCommands(json!({
        "type": "available_commands_update",
        "commands": [
            {"name": "help", "description": "Show help", "source": "builtin"},
            {"name": "model", "description": "Pick model", "source": "builtin"},
            {"name": "compact", "description": "Compact context", "source": "builtin"}
        ]
    })));
}

/// Every row of every tab is editable like TS and the edit persists to
/// config.yml (temp HOME): bool toggle, enum cycle, submenu pick, text /
/// JSON input, multiselect toggle.
#[test]
fn every_settings_row_is_editable_and_persists() {
    let _guard = HOME_LOCK.lock();
    let old_home = std::env::var_os("HOME");
    let home = with_temp_home("all-rows");
    {
        let mut app = home_app();
        let port = port();
        type_text(&mut app, &port, "/settings");
        assert!(enter(&mut app, &port));
        let rows = app.overlays.settings_mut().shown_rows();
        assert!(
            rows.len() > 300,
            "TS panel lists ~380 rows, got {}",
            rows.len()
        );
        let kinds: std::collections::BTreeSet<&str> = rows.iter().map(|(_, k)| *k).collect();
        for kind in ["bool", "enum", "submenu", "text", "multi", "limits"] {
            assert!(kinds.contains(kind), "no {kind} row in the panel");
        }
        let mut unchanged: Vec<String> = Vec::new();
        for (id, kind) in &rows {
            let overlay = app.overlays.settings_mut();
            assert!(overlay.select(id), "{id}: not selectable");
            let before = overlay.value_of(id);
            assert!(press(&mut app, &port, key(CtKeyCode::Enter)), "{id}: enter");
            match *kind {
                // Enter/Space flip the bool / cycle the enum in place.
                "bool" | "enum" => {}
                // Submenu: ↓ moves to another choice, Enter picks it.
                "submenu" => {
                    assert!(
                        app.overlays.settings_mut().is_editing(),
                        "{id}: submenu did not open"
                    );
                    assert!(press(&mut app, &port, key(CtKeyCode::Down)));
                    assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
                }
                // Text field: type a value of the right shape, Enter saves.
                "text" | "limits" => {
                    assert!(
                        app.overlays.settings_mut().is_editing(),
                        "{id}: text field did not open"
                    );
                    let schema = if *kind == "limits" {
                        "record"
                    } else {
                        text_schema(id)
                    };
                    // The field opens pre-filled: clear it first.
                    for _ in 0..400 {
                        press(&mut app, &port, key(CtKeyCode::Backspace));
                    }
                    let input = match schema {
                        "array" => "[\"e2e\"]",
                        "record" if *kind == "limits" => "{\"openai\": 3}",
                        "record" => "{\"e2e\": \"1\"}",
                        _ => "e2e-value",
                    };
                    type_text(&mut app, &port, input);
                    assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
                }
                // Multiselect: Enter toggles the first choice, Esc goes back.
                "multi" => {
                    assert!(
                        app.overlays.settings_mut().is_editing(),
                        "{id}: multiselect did not open"
                    );
                    assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
                    assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
                }
                other => panic!("{id}: unknown kind {other}"),
            }
            let overlay = app.overlays.settings_mut();
            assert!(!overlay.is_editing(), "{id} ({kind}): field still open");
            let after = overlay.value_of(id);
            if after == before {
                unchanged.push(format!("{id} ({kind})"));
                continue;
            }
            // Persisted: a fresh layered read (config.yml + env) agrees.
            let stored = port.settings_get(std::slice::from_ref(id))[0].1.clone();
            assert_eq!(Some(stored), after, "{id} ({kind}): not persisted");
        }
        assert!(
            unchanged.is_empty(),
            "rows whose edit did not change the value: {unchanged:?}"
        );
        let config = std::fs::read_to_string(home.join(".omp/agent/config.yml")).unwrap();
        assert!(config.contains("composer:"), "{config}");
    }
    let _ = std::fs::remove_dir_all(&home);
    restore_home(old_home);
}

/// Registry value type of a text row (`array` JSON lists vs plain strings).
fn text_schema(id: &str) -> &'static str {
    match omp_settings::lookup(id).map(|d| d.ty) {
        Some(omp_settings::SettingType::Array) => "array",
        Some(omp_settings::SettingType::Record) => "record",
        _ => "string",
    }
}

/// Composer Shape edit changes the live editor look at once and persists
/// (TS `syncComposerShape`).
#[test]
fn composer_shape_edit_changes_editor_live_and_persists() {
    let _guard = HOME_LOCK.lock();
    let old_home = std::env::var_os("HOME");
    let home = with_temp_home("shape");
    {
        let mut app = home_app();
        let port = port();
        assert_eq!(app.editor.shape(), crate::composer::ComposerShape::Band);
        type_text(&mut app, &port, "/settings");
        assert!(enter(&mut app, &port));
        assert!(app.overlays.settings_mut().select("composer.shape"));
        assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
        // band → box (second choice).
        assert!(press(&mut app, &port, key(CtKeyCode::Down)));
        let s = screen(&mut app, 120, 40);
        assert!(s.contains("Rounded Box"), "{s}");
        assert!(s.contains("Preview:"), "{s}");
        assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
        assert_eq!(app.editor.shape(), crate::composer::ComposerShape::Box);
        let config = std::fs::read_to_string(home.join(".omp/agent/config.yml")).unwrap();
        assert!(config.contains("shape: box"), "{config}");
        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
        // The live region now draws the box frame around the prompt.
        let s = screen(&mut app, 80, 12);
        assert!(s.contains('╭') && s.contains('╯'), "{s}");
        // Re-seeding from disk (a fresh start) restores the same shape.
        let mut fresh = App::new("p/m".into(), Path::new("/nonexistent"));
        fresh.configure(Vec::new(), Vec::new());
        assert_eq!(fresh.editor.shape(), crate::composer::ComposerShape::Box);
    }
    let _ = std::fs::remove_dir_all(&home);
    restore_home(old_home);
}

/// Status-line settings apply live to the real status line.
#[test]
fn status_line_settings_apply_live() {
    let _guard = HOME_LOCK.lock();
    let old_home = std::env::var_os("HOME");
    let home = with_temp_home("statusline");
    {
        let mut app = home_app();
        let port = port();
        type_text(&mut app, &port, "/settings");
        assert!(enter(&mut app, &port));
        assert!(app
            .overlays
            .settings_mut()
            .select("statusLine.compactThinkingLevel"));
        assert!(app.status.config.compact_thinking);
        assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
        assert!(!app.status.config.compact_thinking);
        assert!(app.overlays.settings_mut().select("statusLine.separator"));
        assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
        assert!(press(&mut app, &port, key(CtKeyCode::Down)));
        assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
        assert_ne!(
            app.status.config.separator,
            crate::status_line::SeparatorStyle::PowerlineThin
        );
    }
    let _ = std::fs::remove_dir_all(&home);
    restore_home(old_home);
}

/// Shift+Tab cycles off → auto → efforts → ultracode → off and updates
/// the status line.
#[test]
fn shift_tab_cycles_thinking_effort() {
    let mut app = app();
    let port = port();
    // A model without thinking: nothing to cycle.
    let shift_tab = KeyEvent::new(CtKeyCode::BackTab, CtMods::SHIFT);
    assert!(press(&mut app, &port, shift_tab));
    assert!(!log(&port)
        .iter()
        .any(|c| c.starts_with("set_thinking_level")));
    // Reasoning model: context_usage announces the supported efforts.
    app.apply_event(&AppEvent::ContextUsage(json!({
        "type": "context_usage", "model": "p/m", "thinkingLevel": "high",
        "thinkingLevels": ["minimal", "low", "medium", "high"], "autoThinking": false,
        "contextWindow": 1000,
    })));
    assert_eq!(app.status.thinking.as_deref(), Some("high"));
    let mut seen = Vec::new();
    for _ in 0..7 {
        assert!(press(&mut app, &port, shift_tab));
        let last = log(&port).last().cloned().unwrap();
        let level = last
            .trim_start_matches("set_thinking_level(")
            .trim_end_matches(')')
            .to_owned();
        // The runtime echoes the change (ultra keeps the effort level).
        let configured = if level == "auto" {
            json!("auto")
        } else {
            json!(level)
        };
        let ultra = (level == "ultracode").then_some("ultracode");
        app.apply_event(&AppEvent::ThinkingLevelChanged(json!({
                "type": "thinking_level_changed",
                "thinkingLevel": match level.as_str() { "auto" => "medium", "ultracode" => "high", l => l },
                "configured": configured,
                "ultra": ultra,
            })));
        seen.push(level);
    }
    assert_eq!(
        seen,
        [
            "ultracode",
            "off",
            "auto",
            "minimal",
            "low",
            "medium",
            "high"
        ]
    );
    assert_eq!(app.status.thinking.as_deref(), Some("high"));
}

/// Bare `/effort` opens the thinking selector; Enter applies the pick.
#[test]
fn effort_selector_opens_and_applies_pick() {
    let mut app = app();
    let port = port();
    app.apply_event(&AppEvent::ContextUsage(json!({
        "type": "context_usage", "model": "p/m", "thinkingLevel": "low",
        "thinkingLevels": ["minimal", "low", "medium"], "contextWindow": 1000,
    })));
    type_text(&mut app, &port, "/effort");
    assert!(enter(&mut app, &port));
    assert_eq!(app.overlays.open(), Some(OpenOverlay::Thinking));
    assert!(press(&mut app, &port, key(CtKeyCode::Down)));
    assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
    assert!(
        log(&port).iter().any(|c| c == "set_thinking_level(medium)"),
        "{:?}",
        log(&port)
    );
    assert_eq!(app.overlays.open(), None);
}

fn screen(app: &mut App, w: u16, h: u16) -> String {
    let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
    t.draw(|f| app.render(f)).unwrap();
    let buf = t.backend().buffer().clone();
    (0..h)
        .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn fullscreen_overlays_want_alt_screen_and_mouse_then_restore_on_esc() {
    use crossterm::event::{MouseEvent, MouseEventKind};
    let mut app = app();
    let port = port();
    assert!(!app.has_fullscreen_overlay());
    assert!(!app.wants_mouse(), "tui.mouse defaults off");
    type_text(&mut app, &port, "/settings");
    assert!(enter(&mut app, &port));
    assert_eq!(app.overlays.open(), Some(OpenOverlay::Settings));
    assert!(app.has_fullscreen_overlay());
    assert!(
        app.wants_mouse(),
        "TS fullscreen overlays enable 1000/1003/1006"
    );
    // The alt-screen frame paints the overlay over the whole terminal.
    let mut t = Terminal::new(TestBackend::new(60, 20)).unwrap();
    t.draw(|f| app.render_fullscreen(f)).unwrap();
    let buf = t.backend().buffer().clone();
    let used = (0..20u16)
        .filter(|&y| (0..60u16).any(|x| buf[(x, y)].symbol() != " "))
        .count();
    assert!(
        used >= 18,
        "fullscreen overlay should fill the screen, used {used}/20"
    );
    // Mouse reports go to the overlay, never to the editor as input.
    let wheel = Event::Mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: 3,
        row: 3,
        modifiers: crossterm::event::KeyModifiers::NONE,
    });
    assert!(app.handle_terminal_event(&wheel, &port));
    assert_eq!(app.editor.text(), "");
    assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
    assert_eq!(app.overlays.open(), None);
    assert!(!app.has_fullscreen_overlay());
    assert!(!app.wants_mouse());
    // Popups (help) stay inline without mouse tracking.
    app.overlays.toggle(OpenOverlay::Help);
    assert!(!app.has_fullscreen_overlay());
    assert!(!app.wants_mouse());
}

#[test]
fn typing_slash_shows_command_popup_above_editor() {
    // Regression: `/` showed nothing — the popup was never wired.
    let mut app = app();
    let port = port();
    with_catalog(&mut app);
    type_text(&mut app, &port, "/");
    let s = screen(&mut app, 60, 14);
    assert!(
        s.contains("help") && s.contains("model") && s.contains("compact"),
        "{s}"
    );
    type_text(&mut app, &port, "mo");
    let s = screen(&mut app, 60, 14);
    // Best match first and selected (lower-ranked fuzzy hits may follow).
    assert!(s.contains("❯ model"), "filtered: {s}");
    assert!(!s.contains("help"), "non-matching rows dropped: {s}");
}

#[test]
fn enter_in_slash_popup_completes_and_runs_command() {
    let mut app = app();
    let port = port();
    with_catalog(&mut app);
    type_text(&mut app, &port, "/he");
    assert!(app.autocomplete.open);
    assert!(enter(&mut app, &port));
    assert_eq!(app.overlays.open(), Some(OpenOverlay::Help));
    assert!(app.editor.is_empty());
}

#[test]
fn esc_closes_popup_and_keeps_draft() {
    let mut app = app();
    let port = port();
    with_catalog(&mut app);
    type_text(&mut app, &port, "/co");
    assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
    assert!(!app.autocomplete.open);
    assert_eq!(app.editor.text(), "/co");
}

#[test]
fn catalog_command_dispatches_to_runtime() {
    // Regression: every command outside the 11 local builtins was
    // rejected as "unknown command" although the popup offered it.
    let mut app = app();
    let port = port();
    app.apply_event(&AppEvent::AvailableCommands(json!({
        "commands": [{"name": "session", "aliases": ["info"], "source": "builtin"}]
    })));
    type_text(&mut app, &port, "/info");
    app.autocomplete.close();
    assert!(enter(&mut app, &port));
    assert!(
        log(&port).iter().any(|l| l.contains("info")),
        "{:?}",
        log(&port)
    );
}

#[test]
fn command_output_and_prompt_errors_reach_the_transcript() {
    let mut app = app();
    app.apply_event(&parse_frame(
        &json!({"type": "command_output", "text": "Session: abc\nTitle: t\nCWD: /tmp"}),
    ));
    app.apply_event(&parse_frame(
            &json!({"type": "prompt_result", "status": "error", "error": {"message": "Unknown command: /zz"}}),
        ));
    let s = screen(&mut app, 60, 14);
    assert!(s.contains("Session: abc") && !s.contains("abcTitle"), "{s}");
    assert!(
        s.lines().any(|l| l.trim_start().starts_with("CWD: /tmp")),
        "line breaks kept: {s}"
    );
    assert!(s.contains("Unknown command: /zz"), "{s}");
}

#[test]
fn question_mark_is_text_inside_a_draft() {
    // Regression: `?` toggled help even mid-sentence.
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "why?");
    assert_eq!(app.editor.text(), "why?");
    assert_eq!(app.overlays.open(), None);
}

#[test]
fn unknown_slash_shows_notice_not_prompt() {
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "/frobnicate");
    assert!(enter(&mut app, &port));
    assert!(log(&port).is_empty());
    assert!(app
        .messages
        .entries()
        .iter()
        .any(|e| e.text.contains("unknown command")));
}

#[test]
fn local_command_shortcut_stays_hidden() {
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "!ls -la");
    assert!(enter(&mut app, &port));
    assert!(log(&port).is_empty());
    assert!(app
        .messages
        .entries()
        .iter()
        .any(|e| e.text.contains("not available yet")));
}

// ---- Esc ladder ------------------------------------------------------

#[test]
fn esc_aborts_streaming_turn() {
    let mut app = app();
    let port = port();
    start_streaming(&mut app);
    assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
    assert_eq!(log(&port), vec!["abort()".to_string()]);
}

#[test]
fn esc_never_destroys_a_draft() {
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "draft");
    assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
    assert_eq!(app.editor.text(), "draft");
    assert!(log(&port).is_empty());
}

/// Regression: a prompt submitted while streaming shows only in the
/// steering band, not also as a chat message (it entered both, so the user
/// saw the prompt "sent and queued" at once).
#[test]
fn steer_while_streaming_is_not_echoed_into_chat() {
    let mut app = app();
    let port = port();
    start_streaming(&mut app);
    let users_before = app
        .messages
        .entries()
        .iter()
        .filter(|e| e.role == ChatRole::User)
        .count();
    type_text(&mut app, &port, "steer me");
    assert!(enter(&mut app, &port));
    assert_eq!(app.steering, vec!["steer me".to_string()]);
    let users_after = app
        .messages
        .entries()
        .iter()
        .filter(|e| e.role == ChatRole::User)
        .count();
    assert_eq!(
        users_after, users_before,
        "queued prompt must not be echoed into the chat"
    );
    assert!(
        log(&port).iter().any(|c| c.contains("steer me")),
        "{:?}",
        log(&port)
    );
}

#[test]
fn esc_aborts_pending_submission_and_restores_queue() {
    let mut app = app();
    let port = port();
    start_streaming(&mut app);
    type_text(&mut app, &port, ">> queued one");
    assert!(enter(&mut app, &port));
    assert!(app.pending_submission);
    assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
    assert!(!app.pending_submission);
    assert_eq!(app.editor.text(), "queued one");
    assert!(app.follow_up.is_empty());
}

/// Regression: Esc during a run with an already-dispatched queued steer pulls it
/// out of the runtime queue into the editor before aborting (TS
/// `restoreQueuedMessagesToEditor({abort: true})`). Left queued, the post-abort
/// drain ran it under the interrupt and it ended as "Request was aborted".
#[test]
fn esc_while_streaming_restores_dispatched_queue_before_abort() {
    let mut app = app();
    let port = port();
    start_streaming(&mut app);
    app.steering.push("как успехи".into());
    type_text(&mut app, &port, "draft");
    assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
    let calls = log(&port);
    assert_eq!(
        calls
            .iter()
            .filter(|c| c.as_str() != "abort()")
            .cloned()
            .collect::<Vec<_>>(),
        vec![format!("remove_queued({:?}, Steering)", "как успехи")]
    );
    assert_eq!(calls.last().map(String::as_str), Some("abort()"));
    assert_eq!(app.editor.text(), "как успехи\ndraft");
    assert!(app.steering.is_empty());
}

#[test]
fn esc_cancels_maintenance_retry() {
    let mut app = app();
    let port = port();
    app.apply_event(&AppEvent::Retry(json!({"type":"auto_retry_start"})));
    assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
    assert_eq!(log(&port), vec!["abort_retry()".to_string()]);
    assert!(app
        .messages
        .entries()
        .iter()
        .any(|e| e.text.contains("retry aborted")));
}

// ---- Ctrl+C / Ctrl+D -------------------------------------------------

#[test]
fn ctrl_c_double_tap_quits_single_tap_clears_editor() {
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "draft");
    let t0 = Instant::now();
    assert_eq!(app.handle_key(ctrl('c'), &port, t0), Some(true));
    assert!(app.editor.is_empty());
    assert_eq!(
        app.handle_key(ctrl('c'), &port, t0 + Duration::from_millis(100)),
        None
    );
}

#[test]
fn ctrl_c_taps_beyond_window_do_not_quit() {
    let mut app = app();
    let port = port();
    let t0 = Instant::now();
    assert!(app.handle_key(ctrl('c'), &port, t0).is_some());
    assert!(app
        .handle_key(ctrl('c'), &port, t0 + Duration::from_millis(600))
        .is_some());
}

#[test]
fn ctrl_d_quits_immediately() {
    let mut app = app();
    let port = port();
    assert!(!press(&mut app, &port, ctrl('d')));
}

// ---- dequeue ----------------------------------------------------------

#[test]
fn alt_up_dequeues_newest_message_to_editor() {
    let mut app = app();
    let port = port();
    app.apply_event(&AppEvent::QueueUpdate {
        steering: vec!["s1".into()],
        follow_up: vec!["f1".into(), "f2".into()],
    });
    let alt_up = CtKeyEvent::new(CtKeyCode::Up, CtMods::ALT);
    // Steering pops before follow-up (§5.3): newest steering first.
    assert!(press(&mut app, &port, alt_up));
    assert_eq!(app.editor.text(), "s1");
    assert!(app.steering.is_empty());
    assert_eq!(
        log(&port),
        vec!["remove_queued(\"s1\", Steering)".to_string()]
    );
    assert!(press(&mut app, &port, alt_up));
    assert_eq!(app.editor.text(), "f2"); // then the newest follow-up
    assert_eq!(app.follow_up, vec!["f1".to_string()]);
}

// ---- event pump -------------------------------------------------------

#[test]
fn pump_coalesces_message_updates_within_33ms() {
    let mut app = app();
    let t0 = Instant::now();
    let upd = |t: &str| json!({"type":"message_update","message":{"role":"assistant","content":[{"type":"text","text":t}]}});
    app.push_frame(&upd("hel"), t0);
    app.push_frame(&upd("hello"), t0 + Duration::from_millis(5));
    assert!(!app.pump_events(t0 + Duration::from_millis(10)));
    assert!(app.messages.entries().is_empty());
    assert!(app.pump_events(t0 + Duration::from_millis(40)));
    assert_eq!(app.messages.entries().len(), 1);
    assert_eq!(app.messages.entries()[0].text, "hello");
}

#[test]
fn pump_holds_updates_until_window_elapses() {
    let mut app = app();
    let t0 = Instant::now();
    let upd = |t: &str| json!({"type":"message_update","message":{"role":"assistant","content":[{"type":"text","text":t}]}});
    app.push_frame(&upd("one"), t0);
    assert!(!app.pump_events(t0 + Duration::from_millis(10)));
    assert!(app.messages.entries().is_empty());
    assert!(app.pump_events(t0 + Duration::from_millis(40)));
    assert_eq!(app.messages.entries()[0].text, "one");
}

#[test]
fn pump_shows_a_continuous_stream_every_window() {
    // Updates every 15 ms never leave a 33 ms gap; the stream must still
    // reach the transcript once per window, not only when it pauses.
    let mut app = app();
    let t0 = Instant::now();
    let upd = |t: &str| json!({"type":"message_update","message":{"role":"assistant","content":[{"type":"text","text":t}]}});
    let mut shown = Vec::new();
    for i in 0..10u64 {
        let at = t0 + Duration::from_millis(15 * i);
        app.push_frame(&upd(&format!("t{i}")), at);
        app.pump_events(at + Duration::from_millis(1));
        if let Some(e) = app.messages.entries().first() {
            shown.push(e.text.clone());
        }
    }
    assert_eq!(
        shown.first().map(String::as_str),
        Some("t3"),
        "first snapshot after one window"
    );
    assert!(
        shown.contains(&"t7".to_string()),
        "a second snapshot mid-stream: {shown:?}"
    );
}

#[test]
fn pump_bumps_epoch_at_agent_start() {
    let mut app = app();
    let t0 = Instant::now();
    app.push_frame(&json!({"type":"agent_start"}), t0);
    app.push_frame(&json!({"type":"agent_end"}), t0);
    app.pump_events(t0);
    assert_eq!(app.run_epoch, 1);
}

#[test]
fn non_update_frame_flushes_pending_update_first() {
    let mut app = app();
    let t0 = Instant::now();
    let upd = |t: &str| json!({"type":"message_update","message":{"role":"assistant","content":[{"type":"text","text":t}]}});
    app.push_frame(&upd("hi"), t0);
    app.push_frame(&json!({"type":"agent_end"}), t0);
    app.pump_events(t0);
    assert_eq!(app.messages.entries()[0].text, "hi");
    assert!(!app.status.streaming);
}

// ---- rendering --------------------------------------------------------

#[test]
fn renders_chat_editor_status() {
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "hello");
    assert!(enter(&mut app, &port));
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let buf = terminal.backend().buffer();
    // Band composer: status line on top, then the `╰─ ` prompt row.
    let status: String = (0..80).map(|x| buf[(x, 22)].symbol()).collect();
    assert!(status.contains("zai/glm-5.3-flash"), "status: {status:?}");
    let editor: String = (0..80).map(|x| buf[(x, 23)].symbol()).collect();
    assert!(editor.starts_with("╰─"), "editor: {editor:?}");
}

#[test]
fn working_loader_shows_while_streaming() {
    let mut app = app();
    let _port = port();
    start_streaming(&mut app);
    let backend = TestBackend::new(80, 10);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let buf = terminal.backend().buffer();
    // rows: 10 - 1 editor - 1 status - 1 loader = 7 chat
    let row: String = (0..80).map(|x| buf[(x, 7)].symbol()).collect();
    assert!(row.contains("Working…"), "row: {row:?}");
}

#[test]
fn queued_band_renders_groups_and_hint() {
    let mut app = app();
    let _port = port();
    app.apply_event(&AppEvent::QueueUpdate {
        steering: vec!["left".into()],
        follow_up: vec!["after".into()],
    });
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let buf = terminal.backend().buffer();
    let band: Vec<String> = (15..22)
        .map(|y| (0..80).map(|x| buf[(x, y)].symbol()).collect())
        .collect();
    let text = band.join("\n");
    assert!(text.contains("Steering·1"), "{text}");
    assert!(text.contains("After yield·1"), "{text}");
    assert!(text.contains("alt+up to edit"), "{text}");
}

#[test]
fn maintenance_loader_advertises_esc_to_cancel() {
    let mut app = app();
    let _port = port();
    app.apply_event(&AppEvent::AutoCompaction(
        json!({"type":"auto_compaction_start","reason":"threshold"}),
    ));
    let backend = TestBackend::new(80, 6);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let buf = terminal.backend().buffer();
    // 6 - 1 editor - 1 status - 1 loader = 3 chat rows; loader at row 3
    let row: String = (0..80).map(|x| buf[(x, 3)].symbol()).collect();
    assert!(row.contains("esc to cancel"), "row: {row:?}");
}

#[test]
fn take_scrollback_lines_advances_watermark() {
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "done");
    assert!(enter(&mut app, &port));
    let committed = app.messages.committed_len();
    let lines = app.take_scrollback_lines(80, false);
    // Rich rendering emits several rows per entry; the watermark must
    // advance over every finished entry.
    assert!(!lines.is_empty());
    assert!(app.messages.committed_len() > committed);
    // The live tail must exclude everything committed so far.
    assert_eq!(
        app.messages.live_tail().len() + app.messages.committed_len(),
        app.messages.entries().len()
    );
    // Rebuild returns the full transcript and re-bases the watermark.
    let full = app.take_scrollback_lines(80, true);
    assert!(!full.is_empty());
    assert_eq!(app.messages.committed_len(), app.messages.finished_len());
}

#[test]
fn render_live_draws_editor_and_status() {
    let mut app = app();
    let _port = port();
    let backend = TestBackend::new(80, 5);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| app.render_live(f, Rect::new(0, 0, 80, 5)))
        .unwrap();
    let buf = terminal.backend().buffer();
    // Band composer: the status line row, then the `╰─ ` prompt row.
    let editor_row: String = (0..80).map(|x| buf[(x, 1)].symbol()).collect();
    assert!(editor_row.starts_with("╰─"), "row: {editor_row:?}");
    let status_row: String = (0..80).map(|x| buf[(x, 0)].symbol()).collect();
    assert!(
        status_row.contains("zai/glm-5.3-flash"),
        "row: {status_row:?}"
    );
}

// ---- /resume, /tree, /theme overlay wiring ----------------------------

#[test]
fn slash_resume_opens_embedded_session_picker() {
    let mut app = app();
    let port = port();
    let dir = std::env::temp_dir().join(format!("omp-resume-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // A minimal journal: title line + session header.
    let file = dir.join("sess.jsonl");
    std::fs::write(
        &file,
        concat!(
            "{\"type\":\"title\",\"title\":\"demo session\"}\n",
            "{\"type\":\"session\",\"id\":\"s1\",\"cwd\":\"/repo\"}\n",
        ),
    )
    .unwrap();
    app.configure(Vec::new(), vec![SessionSource::new(&dir)]);
    type_text(&mut app, &port, "/resume");
    assert!(enter(&mut app, &port));
    assert_eq!(app.overlays.open(), Some(OpenOverlay::Sessions));
    // Enter resumes the highlighted session through the port.
    assert!(enter(&mut app, &port));
    let calls = log(&port);
    assert!(
        calls
            .iter()
            .any(|c| c.starts_with("resume_session(") && c.contains("sess.jsonl")),
        "calls: {calls:?}"
    );
    assert_eq!(app.overlays.open(), None);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn slash_resume_beats_description_matches_in_the_popup() {
    // Regression: `/resume` was not in the popup catalog, so Enter
    // accepted `/pin` (its description mentions the resume list).
    let mut app = app();
    let port = port();
    let dir = std::env::temp_dir().join(format!("omp-resume-popup-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    app.configure(Vec::new(), vec![SessionSource::new(&dir)]);
    app.apply_event(&parse_frame(&json!({
            "type": "available_commands_update",
            "commands": [{"name": "pin", "description": "Pin this session to the top of the resume list",
                          "source": "builtin"}],
        })));
    type_text(&mut app, &port, "/resume");
    assert!(enter(&mut app, &port));
    assert_eq!(app.overlays.open(), Some(OpenOverlay::Sessions));
    assert!(
        !log(&port).iter().any(|c| c.contains("pin")),
        "calls: {:?}",
        log(&port)
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn session_loaded_rebuilds_transcript_from_history() {
    let mut app = app();
    app.push_system("stale notice");
    app.apply_event(&parse_frame(&json!({
        "type": "session_loaded",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "old question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "old answer"}],
             "stopReason": "stop"},
        ],
    })));
    let texts: Vec<String> = app
        .messages
        .entries()
        .iter()
        .map(|e| e.text.clone())
        .collect();
    assert_eq!(texts.len(), 2, "history replaces the transcript: {texts:?}");
    assert!(texts[0].contains("old question"), "{texts:?}");
    assert!(texts[1].contains("old answer"), "{texts:?}");
    // The loaded history is settled and flows into scrollback.
    let rows = app.take_scrollback_lines(80, false);
    let joined: String = rows
        .iter()
        .flat_map(|l| l.spans.iter().map(|s| s.content.to_string()))
        .collect();
    assert!(joined.contains("old answer"), "{joined}");
}

/// Regression: `/new` left the old conversation on screen. `session_reset`
/// empties the transcript and asks the controller to erase scrollback once.
#[test]
fn session_reset_clears_transcript_and_requests_scrollback_erase() {
    let mut app = app();
    app.apply_event(&parse_frame(&json!({
        "type": "session_loaded",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "old question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "old answer"}],
             "stopReason": "stop"},
        ],
    })));
    let _ = app.take_scrollback_lines(80, false);
    assert!(!app.take_clear_scrollback());
    app.apply_event(&parse_frame(&json!({"type": "session_reset"})));
    assert!(
        app.messages.entries().is_empty(),
        "transcript must be empty after /new"
    );
    assert!(app.take_clear_scrollback(), "scrollback erase requested");
    assert!(!app.take_clear_scrollback(), "erase is one-shot");
    let joined: String = app
        .take_scrollback_lines(80, true)
        .iter()
        .flat_map(|l| l.spans.iter().map(|s| s.content.to_string()))
        .collect();
    assert!(!joined.contains("old answer"), "{joined}");
}

#[test]
fn session_loaded_name_sets_title_label_and_owned_path_matches() {
    let mut app = app();
    let before = app.title_label.clone();
    // No name: label unchanged.
    app.push_frame_owned(
        json!({"type": "session_loaded", "messages": [], "name": null}),
        Instant::now(),
    );
    app.pump_events(Instant::now());
    assert_eq!(app.title_label, before);
    // Named: label follows the restored name; history moved, not lost.
    app.push_frame_owned(
        json!({"type": "session_loaded", "name": "Old title", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
        ]}),
        Instant::now(),
    );
    app.pump_events(Instant::now());
    assert_eq!(app.title_label, "Old title");
    assert_eq!(app.messages.entries().len(), 1);
    // Borrowed parse agrees with the owned parse.
    let frame = json!({"type": "session_loaded", "name": "n", "messages": [{"role": "user"}]});
    assert_eq!(parse_frame(&frame), parse_frame_owned(frame.clone()));
}

#[test]
fn slash_tree_opens_tree_overlay_from_journal() {
    let mut app = app();
    let port = port();
    push_user_turn(&mut app, "u1", None, "first question");
    push_user_turn(&mut app, "u2", Some("u1"), "follow up");
    type_text(&mut app, &port, "/tree");
    assert!(enter(&mut app, &port));
    assert_eq!(app.overlays.open(), Some(OpenOverlay::Tree));
    assert!(app.has_fullscreen_overlay());
}

#[test]
fn slash_theme_lists_and_hot_swaps() {
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "/theme");
    assert!(enter(&mut app, &port));
    assert!(app
        .messages
        .entries()
        .iter()
        .any(|e| e.text.contains("available themes:")));
    type_text(&mut app, &port, "/theme light");
    assert!(enter(&mut app, &port));
    // The transcript renderer follows the app theme (set_theme is the
    // single sync point, not a per-frame copy).
    assert_eq!(
        app.theme.color("accent"),
        app.messages.theme().color("accent")
    );
    assert_ne!(app.theme.color("accent"), Theme::default().color("accent"));
}

#[test]
fn theme_setter_syncs_transcript_renderer_once() {
    let mut app = app();
    let before = app.messages.theme().color("accent");
    let t = crate::theme::load("dark").expect("builtin dark theme");
    app.set_theme(t.clone());
    assert_eq!(app.messages.theme().color("accent"), t.color("accent"));
    assert_ne!(before, t.color("accent"));
}

// ---- double-Esc rewind --------------------------------------------------

#[test]
fn double_esc_opens_rewind_tree_and_confirms() {
    let mut app = app();
    let port = port();
    push_user_turn(&mut app, "u1", None, "first question");
    push_user_turn(&mut app, "u2", Some("u1"), "follow up");
    let t0 = Instant::now();
    let esc = || key(CtKeyCode::Esc);
    assert!(app.handle_key(esc(), &port, t0).is_some());
    assert_eq!(app.overlays.open(), None); // first Esc is timing state
    assert!(app
        .handle_key(esc(), &port, t0 + Duration::from_millis(100))
        .is_some());
    assert_eq!(app.overlays.open(), Some(OpenOverlay::Tree));
    // Enter confirms the rewind target → port.rewind_entry.
    assert!(enter(&mut app, &port));
    let calls = log(&port);
    assert!(
        calls.iter().any(|c| c.starts_with("rewind_entry(")),
        "calls: {calls:?}"
    );
    assert_eq!(app.overlays.open(), None);
}

#[test]
fn double_esc_without_history_shows_notice() {
    let mut app = app();
    let port = port();
    let t0 = Instant::now();
    assert!(app.handle_key(key(CtKeyCode::Esc), &port, t0).is_some());
    assert!(app
        .handle_key(key(CtKeyCode::Esc), &port, t0 + Duration::from_millis(50))
        .is_some());
    assert_eq!(app.overlays.open(), None);
    assert!(app
        .messages
        .entries()
        .iter()
        .any(|e| e.text.contains("nothing to rewind")));
}

// ---- terminal title -----------------------------------------------------

#[test]
fn title_follows_agent_start_and_end() {
    let mut app = app();
    let title = app.title().current_title().unwrap_or_default();
    assert!(title.contains("repo"), "initial title: {title:?}");
    app.apply_event(&AppEvent::AgentStart);
    let working = app.title().current_title().unwrap_or_default();
    assert!(working.starts_with('π'), "working title: {working:?}");
    assert!(working.contains("repo"), "working title: {working:?}");
    app.apply_event(&AppEvent::AgentEnd(json!({})));
    let idle = app.title().current_title().unwrap_or_default();
    assert!(idle.contains('>'), "idle title: {idle:?}");
}

// ---- notifications ------------------------------------------------------

#[test]
fn session_settled_fires_turn_notification() {
    let mut app = app();
    app.apply_event(&AppEvent::SessionSettled);
    assert_eq!(
        app.notify_sink().record(),
        vec!["omp: Complete".to_string()]
    );
}

#[test]
fn turn_notification_follows_ts_gates() {
    use crate::chrome::notify::NotifyKind;
    assert_eq!(
        turn_notification(false, Some("stop"), false, true),
        Some(("Complete", NotifyKind::Completion))
    );
    assert_eq!(
        turn_notification(false, None, false, true),
        Some(("Complete", NotifyKind::Completion))
    );
    assert_eq!(turn_notification(false, Some("aborted"), true, true), None);
    assert_eq!(turn_notification(false, Some("stop"), true, false), None);
    assert_eq!(
        turn_notification(false, Some("error"), true, true),
        Some(("Stopped with error", NotifyKind::Error))
    );
    assert_eq!(
        turn_notification(true, None, true, true),
        Some(("Stopped with error", NotifyKind::Error))
    );
    // error.notify defaults to off: an errored turn is silent, and never reads as "Complete".
    assert_eq!(turn_notification(true, Some("error"), false, true), None);
}

#[test]
fn failed_prompt_result_fires_error_notification() {
    let mut app = app();
    app.apply_event(&AppEvent::PromptResult {
        id: None,
        success: false,
        error: Some("kaboom".into()),
        frame: json!({"type":"prompt_result","status":"error","error":{"message":"kaboom"}}),
    });
    // `error.notify` defaults to off (TS): silent, and never a "Complete" toast.
    assert!(app
        .notify_sink()
        .record()
        .iter()
        .all(|l| l != "omp: Complete"));
}

#[test]
fn notify_hook_intercepts_delivery() {
    let mut app = app();
    let got = std::sync::Arc::new(parking_lot::Mutex::new(Vec::new()));
    let sink_got = got.clone();
    app.set_notify_hook(std::sync::Arc::new(move |n| {
        sink_got.lock().push(n.body.clone());
    }));
    app.apply_event(&AppEvent::SessionSettled);
    assert_eq!(got.lock().clone(), vec!["Complete".to_string()]);
}

// ---- keymap engine on the input path ---------------------------------

#[test]
fn input_path_uses_keymap_overrides() {
    let mut app = app();
    let port = port();
    // Rebind app.exit to ctrl+q; ctrl+d must stop quitting.
    let mut user = crate::keybindings::UserBindings::new();
    user.insert("app.exit".into(), vec!["ctrl+q".into()]);
    let mut km = crate::keybindings::KeybindingsManager::new(user);
    std::mem::swap(&mut app.keys, &mut km);
    assert!(app.handle_key(ctrl('d'), &port, Instant::now()).is_some());
    assert_eq!(app.handle_key(ctrl('q'), &port, Instant::now()), None);
}

#[test]
fn input_path_rebound_dequeue_chord() {
    let mut app = app();
    let port = port();
    app.apply_event(&AppEvent::QueueUpdate {
        steering: vec!["one".into()],
        follow_up: Vec::new(),
    });
    // Rebind dequeue to ctrl+n.
    let mut user = crate::keybindings::UserBindings::new();
    user.insert("app.message.dequeue".into(), vec!["ctrl+n".into()]);
    let mut km = crate::keybindings::KeybindingsManager::new(user);
    std::mem::swap(&mut app.keys, &mut km);
    assert!(press(&mut app, &port, ctrl('n')));
    assert_eq!(app.editor.text(), "one");
    assert_eq!(
        log(&port),
        vec!["remove_queued(\"one\", Steering)".to_string()]
    );
}

#[test]
fn band_hint_reflects_dequeue_binding() {
    let mut app = app();
    let _port = port();
    app.apply_event(&AppEvent::QueueUpdate {
        steering: vec!["x".into()],
        follow_up: Vec::new(),
    });
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let buf = terminal.backend().buffer();
    let band: Vec<String> = (15..22)
        .map(|y| (0..80).map(|x| buf[(x, y)].symbol()).collect())
        .collect();
    let text = band.join("\n");
    assert!(text.contains("alt+up to edit"), "{text}");
}

/// Push one user turn through `message_end` so the rewind journal has a
/// realistic entry (`{id, parentId?, role, content}` inside `message`).
fn push_user_turn(app: &mut App, id: &str, parent: Option<&str>, text: &str) {
    let mut message = serde_json::json!({
        "id": id,
        "role": "user",
        "content": [{"type": "text", "text": text}],
    });
    if let Some(p) = parent {
        message["parentId"] = Value::from(p);
    }
    app.apply_event(&AppEvent::MessageEnd(json!({
        "type": "message_end",
        "message": message,
    })));
}

// ---- omp-rs: OMT welcome, τ, ultra, magic keywords -------------------

fn clean_app() -> App {
    let mut app = app();
    let mut theme = Theme::default();
    theme.ui = crate::theme::UiStyle::Clean;
    app.set_theme(theme);
    app
}

fn reasoning(app: &mut App, ultra: &Value) {
    app.apply_event(&AppEvent::ContextUsage(json!({
        "type": "context_usage", "model": "p/m", "thinkingLevel": "high",
        "thinkingLevels": ["minimal", "low", "medium", "high"], "autoThinking": false,
        "contextWindow": 1000, "ultra": ultra,
    })));
}

/// Any ultra mode (hybrids too) is followed by `off`; snapshots carry
/// `ultra` and `null` clears it.
#[test]
fn ultra_state_mirrors_frames_and_cycles_to_off() {
    let mut app = app();
    let port = port();
    reasoning(&mut app, &json!("ultradeep"));
    assert_eq!(app.status.ultra, Some(crate::magic::Ultra::Deep));
    assert_eq!(app.current_effort_selector(), "ultradeep");
    assert!(press(
        &mut app,
        &port,
        KeyEvent::new(CtKeyCode::BackTab, CtMods::SHIFT)
    ));
    assert_eq!(
        log(&port).last().map(String::as_str),
        Some("set_thinking_level(off)")
    );
    // Frames without the key leave it; `null` clears it.
    app.apply_event(&AppEvent::ThinkingLevelChanged(json!({
        "type": "thinking_level_changed", "thinkingLevel": "high"
    })));
    assert_eq!(app.status.ultra, Some(crate::magic::Ultra::Deep));
    app.apply_event(&AppEvent::ThinkingLevelChanged(json!({
        "type": "thinking_level_changed", "thinkingLevel": "off", "ultra": null
    })));
    assert_eq!(app.status.ultra, None);
    // A model that cannot think offers no ultracode either.
    let mut plain = self::app();
    plain.apply_event(&AppEvent::ContextUsage(json!({
        "type": "context_usage", "model": "p/m", "thinkingLevels": []
    })));
    assert!(plain.effort_selectors().is_empty());
    assert_eq!(
        app.effort_selectors().last().map(String::as_str),
        Some("ultracode")
    );
}

/// `/ultra …` is a runtime slash command once the catalog lists it.
#[test]
fn slash_ultra_reaches_the_runtime() {
    let mut app = app();
    let port = port();
    app.apply_event(&AppEvent::AvailableCommands(json!({
        "commands": [{"name": "ultra", "description": "Ultra mode", "source": "builtin"}]
    })));
    assert!(app
        .autocomplete
        .commands()
        .iter()
        .any(|c| c.name == "ultra"));
    type_text(&mut app, &port, "/ultra deep");
    app.autocomplete.close();
    assert!(enter(&mut app, &port));
    assert!(
        log(&port).iter().any(|l| l == "run_slash(ultra, \"deep\")"),
        "{:?}",
        log(&port)
    );
}

/// Clean style renders the ultra mode in the status line.
#[test]
fn clean_status_shows_ultra_and_tau() {
    let mut app = clean_app();
    reasoning(&mut app, &json!("ultracode"));
    let s = screen(&mut app, 120, 12);
    assert!(s.contains("◉ ultracode"), "{s}");
    assert!(s.contains("τ∿"), "{s}");
    assert!(s.lines().any(|l| l.trim_start().starts_with("τ ·")), "{s}");
}

/// τ colour follows the run state: idle, running, blocked, error.
#[test]
fn brand_state_tracks_run_dialogs_and_errors() {
    use crate::status_line::Brand;
    let mut app = clean_app();
    assert_eq!(app.brand_state(), Brand::Idle);
    app.apply_event(&AppEvent::AgentStart);
    assert_eq!(app.brand_state(), Brand::Running);
    app.apply_event(&AppEvent::ExtensionUiRequest(json!({
        "type": "extension_ui_request", "id": "q1", "method": "confirm",
        "title": "Allow?", "message": "run it?"
    })));
    assert_eq!(app.brand_state(), Brand::Blocked);
    app.extension_ui.complete("q1");
    app.apply_event(&AppEvent::MessageEnd(json!({
            "type": "message_end",
            "message": {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": "boom"}
        })));
    app.apply_event(&AppEvent::AgentEnd(json!({"type": "agent_end"})));
    assert_eq!(app.brand_state(), Brand::Error);
    app.apply_event(&AppEvent::AgentStart);
    assert_eq!(app.brand_state(), Brand::Running);
    app.apply_event(&AppEvent::AgentEnd(json!({"type": "agent_end"})));
    assert_eq!(app.brand_state(), Brand::Idle);
    // Painted into the status line on the next frame.
    let _ = screen(&mut app, 100, 10);
    assert_eq!(app.status.brand, Brand::Idle);
}

/// The clean welcome stays live while it shimmers, then commits its
/// static rows; anything retiring earlier commits it immediately.
#[test]
fn clean_welcome_shimmers_live_then_commits_static_rows() {
    let mut app = clean_app();
    app.set_welcome("9.9.9");
    assert!(
        app.take_scrollback_lines(100, false).is_empty(),
        "held while shimmering"
    );
    assert_eq!(
        app.animation_interval(),
        Some(Duration::from_millis(crate::chat::reveal::FRAME_MS))
    );
    let s = screen(&mut app, 100, 14);
    assert!(
        s.lines().nth(2).is_some_and(|l| l.contains("OMT")),
        "drawn at the top of the live region:\n{s}"
    );
    // Time is up: the static rows commit, the live region drops them.
    let w = app.welcome.as_mut().unwrap();
    w.started = w.started.map(|s| {
        s.checked_sub(Duration::from_millis(welcome::SHIMMER_MS))
            .unwrap()
    });
    assert!(w.started.is_some(), "the clock starts at the first paint");
    assert_eq!(app.animation_interval(), None);
    let rows = app.take_scrollback_lines(100, false);
    let tip = app.welcome.as_ref().unwrap().tip.clone();
    let mut expected = welcome::clean_welcome_lines("9.9.9", &tip, 100, &app.theme, 0.0);
    expected.push(Line::default());
    assert_eq!(rows, expected);
    assert!(!screen(&mut app, 100, 14).contains("OMT"));
    assert!(app.take_scrollback_lines(100, false).is_empty());

    // A prompt retiring mid-shimmer commits the welcome first.
    let mut app = clean_app();
    let port = port();
    app.set_welcome("9.9.9");
    type_text(&mut app, &port, "hello");
    assert!(enter(&mut app, &port));
    let text: Vec<String> = app
        .take_scrollback_lines(100, false)
        .iter()
        .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
        .collect();
    assert!(text.iter().any(|r| r.contains("OMT")), "{text:#?}");
    let omt = text.iter().position(|r| r.contains("OMT")).unwrap();
    let hello = text.iter().position(|r| r.contains("hello")).unwrap();
    assert!(omt < hello);

    // `/new` (session_reset) rebuild re-commits the static welcome.
    let mut app = clean_app();
    app.set_welcome("9.9.9");
    app.apply_event(&AppEvent::SessionReset);
    assert!(app.take_clear_scrollback());
    let rows = app.take_scrollback_lines(100, true);
    assert!(rows
        .iter()
        .any(|l| l.spans.iter().any(|s| s.content == "O")));
}

/// Classic keeps committing the TS lockup at once (no shimmer).
#[test]
fn classic_welcome_commits_immediately() {
    let mut app = app();
    app.set_welcome("1.0.0");
    assert!(!app.take_scrollback_lines(120, false).is_empty());
    assert_eq!(app.animation_interval(), None);
}

/// Clean style brands the terminal title with τ, classic with π.
#[test]
fn title_brand_follows_the_style() {
    let app = clean_app();
    assert!(
        app.title()
            .current_title()
            .is_some_and(|t| t.starts_with("τ > ")),
        "{:?}",
        app.title().current_title()
    );
    let classic = self::app();
    assert!(classic
        .title()
        .current_title()
        .is_some_and(|t| t.starts_with("π > ")));
}

/// The editor paints magic keywords and shimmers only while one is typed.
#[test]
fn editor_paints_magic_keywords_and_shimmers_only_with_one() {
    let mut app = app();
    let port = port();
    type_text(&mut app, &port, "plain text");
    assert_eq!(app.animation_interval(), None, "no keyword, no timer");
    app.set_editor_text("please ultrathink now");
    assert_eq!(
        app.animation_interval(),
        Some(Duration::from_millis(crate::magic_paint::SHIMMER_FRAME_MS))
    );
    let mut t = Terminal::new(TestBackend::new(60, 10)).unwrap();
    t.draw(|f| app.render(f)).unwrap();
    let buf = t.backend().buffer().clone();
    let (x, y) = (0..10u16)
        .find_map(|y| {
            let row: String = (0..60u16).map(|x| buf[(x, y)].symbol()).collect();
            row.find("ultrathink")
                .map(|b| (u16::try_from(row[..b].chars().count()).unwrap(), y))
        })
        .expect("keyword on screen");
    assert!(matches!(buf[(x, y)].fg, ratatui::style::Color::Rgb(..)));
    assert!(matches!(buf[(x + 9, y)].fg, ratatui::style::Color::Rgb(..)));
    assert!(!matches!(
        buf[(x - 2, y)].fg,
        ratatui::style::Color::Rgb(..)
    ));
    // Disabled by settings: no paint, no timer.
    app.magic_off = 1;
    let theme = app.theme.clone();
    app.set_theme(theme);
    assert_eq!(app.animation_interval(), None);
}

/// Hidden notices (`display: false`) never render, live or rebuilt.
#[test]
fn hidden_custom_notices_never_render() {
    let notice = json!({
        "role": "custom", "customType": "ultracode-notice",
        "content": [{"type": "text", "text": "SECRET-NOTICE"}],
        "display": false, "attribution": "user", "timestamp": 1
    });
    let mut app = app();
    app.apply_event(&AppEvent::MessageStart(
        json!({"type": "message_start", "message": notice}),
    ));
    app.apply_event(&AppEvent::MessageEnd(
        json!({"type": "message_end", "message": notice}),
    ));
    let live: Vec<String> = app
        .take_scrollback_lines(80, true)
        .iter()
        .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
        .collect();
    assert!(
        live.iter()
            .all(|r| !r.contains("SECRET") && !r.contains("ultracode-notice")),
        "{live:#?}"
    );
    assert!(!screen(&mut app, 80, 12).contains("SECRET"));
    app.apply_event(&AppEvent::SessionLoaded {
        messages: vec![
            json!({"role": "user", "content": [{"type": "text", "text": "hi"}]}),
            notice,
        ],
        name: None,
    });
    let rebuilt: Vec<String> = app
        .take_scrollback_lines(80, true)
        .iter()
        .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
        .collect();
    assert!(rebuilt.iter().any(|r| r.contains("hi")));
    assert!(
        rebuilt
            .iter()
            .all(|r| !r.contains("SECRET") && !r.contains("ultracode-notice")),
        "{rebuilt:#?}"
    );
}

#[test]
fn format_tokens_compacts() {
    assert_eq!(format_tokens(0), "0");
    assert_eq!(format_tokens(999), "999");
    assert_eq!(format_tokens(1234), "1.2k");
    assert_eq!(format_tokens(18_200), "18.2k");
    assert_eq!(format_tokens(1_400), "1.4k");
    assert_eq!(format_tokens(1_500_000), "1.5M");
}

#[test]
fn turn_usage_estimates_from_deltas_then_exact() {
    let mut usage = TurnUsage::default();
    // 1200 text chars + 3600 thinking chars → (1200+3600)/4 = 1200 out.
    let text = "a".repeat(1200);
    let think = "t".repeat(3600);
    usage.observe_message(&json!({
        "id": "m1", "role": "assistant",
        "content": [{"type": "text", "text": text}, {"type": "thinking", "thinking": think}]
    }));
    assert_eq!(usage.output_tokens(), 1200);
    assert_eq!(usage.thinking_tokens(), 900);
    // No provider usage yet: input falls back to the context estimate.
    assert_eq!(usage.input_tokens(Some(18_200)), 18_200);
    // Second message accumulates per turn.
    usage.observe_message(&json!({
        "id": "m2", "role": "assistant",
        "content": [{"type": "text", "text": "b".repeat(400)}]
    }));
    assert_eq!(usage.estimated_output(), 1300);
    // Provider usage wins over the estimate once it arrives.
    usage.observe_message(&json!({
        "id": "m1", "role": "assistant", "usage": {"input": 100, "output": 500, "cacheRead": 50}
    }));
    assert_eq!(usage.output_tokens(), 500);
    assert_eq!(usage.input_tokens(None), 150);
    // Pricing math: 150 in @ $2/M + 500 out @ $8/M = $0.0043.
    let cost = usage.cost(150, 500, 2.0, 8.0).unwrap();
    assert!((cost - 0.0043).abs() < 1e-9, "cost: {cost}");
}
#[test]
fn turn_usage_deltas_add_thinking_and_text() {
    let mut usage = TurnUsage::default();
    // Thinking deltas (`delta_updates` mode: no cumulative `message`).
    usage.observe_delta(&json!({
        "type": "message_update", "messageId": "m1",
        "assistantMessageEvent": {"type": "thinking_delta", "delta": "t".repeat(3600)}
    }));
    assert_eq!(usage.output_tokens(), 900);
    // Text deltas add into the same message.
    usage.observe_delta(&json!({
        "type": "message_update", "messageId": "m1",
        "assistantMessageEvent": {"type": "text_delta", "delta": "a".repeat(1200)}
    }));
    assert_eq!(usage.output_tokens(), 1200);
    assert_eq!(usage.thinking_tokens(), 900);
    // A settled cumulative snapshot resets the deltas (no double count).
    usage.observe_message(&json!({
        "id": "m1", "role": "assistant",
        "content": [{"type": "thinking", "thinking": "t".repeat(3600)},
                    {"type": "text", "text": "a".repeat(1200)}]
    }));
    assert_eq!(usage.output_tokens(), 1200);
    assert_eq!(usage.thinking_tokens(), 900);
}

#[test]
fn loader_row_counts_thinking_deltas_live() {
    let mut app = app();
    let _port = port();
    app.apply_event(&AppEvent::AgentStart);
    app.apply_event(&AppEvent::ContextUsage(json!({
        "type": "context_usage", "model": "zai/glm-5.3-flash", "tokens": 1234
    })));
    let render_row = |app: &mut App| -> String {
        let backend = TestBackend::new(80, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
        (0..80)
            .map(|x| terminal.backend().buffer()[(x, 7)].symbol())
            .collect()
    };
    // Thinking-only deltas: `↓` grows with each delta.
    app.apply_event(&AppEvent::MessageUpdate(json!({
        "type": "message_update", "messageId": "m1",
        "assistantMessageEvent": {"type": "thinking_delta", "delta": "t".repeat(1800)}
    })));
    let row1 = render_row(&mut app);
    assert!(row1.contains("↓450"), "first row: {row1:?}");
    assert!(row1.contains("(think 450)"), "first row: {row1:?}");
    app.apply_event(&AppEvent::MessageUpdate(json!({
        "type": "message_update", "messageId": "m1",
        "assistantMessageEvent": {"type": "thinking_delta", "delta": "t".repeat(1800)}
    })));
    let row2 = render_row(&mut app);
    assert!(row2.contains("↓900"), "second row: {row2:?}");
    assert!(row2.contains("(think 900)"), "second row: {row2:?}");
}

#[test]
fn loader_row_shows_live_token_usage() {
    let mut app = app();
    let _port = port();
    app.apply_event(&AppEvent::AgentStart);
    app.apply_event(&AppEvent::ContextUsage(json!({
        "type": "context_usage", "model": "zai/glm-5.3-flash", "tokens": 1234
    })));
    // Stream ~1200 output tokens (~300 text + ~900 thinking).
    app.apply_event(&AppEvent::MessageUpdate(json!({
        "message": {"id": "m1", "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "t".repeat(3600)},
                                {"type": "text", "text": "a".repeat(1200)}]}
    })));
    let backend = TestBackend::new(80, 10);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let buf = terminal.backend().buffer();
    let row: String = (0..80).map(|x| buf[(x, 7)].symbol()).collect();
    assert!(row.contains("↑1.2k"), "row: {row:?}");
    assert!(row.contains("↓1.2k"), "row: {row:?}");
    assert!(row.contains("(think 900)"), "row: {row:?}");
}

#[test]
fn loader_usage_drops_on_narrow_terminals() {
    let mut app = app();
    let _port = port();
    app.apply_event(&AppEvent::AgentStart);
    app.apply_event(&AppEvent::ContextUsage(json!({
        "type": "context_usage", "model": "zai/glm-5.3-flash", "tokens": 1234
    })));
    app.apply_event(&AppEvent::MessageUpdate(json!({
        "message": {"id": "m1", "role": "assistant",
                    "content": [{"type": "text", "text": "a".repeat(1200)}]}
    })));
    let backend = TestBackend::new(40, 10);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let buf = terminal.backend().buffer();
    let row: String = (0..40).map(|x| buf[(x, 7)].symbol()).collect();
    assert!(!row.contains("↑1.2k"), "row: {row:?}");
    assert!(row.contains("Working…"), "row: {row:?}");
}

#[test]
fn turn_usage_resets_each_turn() {
    let mut app = app();
    let _port = port();
    app.apply_event(&AppEvent::AgentStart);
    app.apply_event(&AppEvent::MessageUpdate(json!({
        "message": {"id": "m1", "role": "assistant",
                    "content": [{"type": "text", "text": "a".repeat(1200)}]}
    })));
    assert_eq!(app.turn_usage.output_tokens(), 300);
    app.apply_event(&AppEvent::TurnStart);
    assert_eq!(app.turn_usage.output_tokens(), 0);
}
