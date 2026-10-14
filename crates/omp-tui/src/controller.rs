//! The interactive controller: terminal setup, the event pump and the draw
//! loop — the port of the wiring half of
//! `pi-coding-agent/src/modes/interactive-mode.ts` (its `run` loop) plus the
//! terminal plumbing of `pi-tui/src/tui.ts` / `terminal.ts`.
//!
//! ## Threading model
//!
//! One OS task owns the terminal: the async `run` loop below (living on
//! the tokio runtime omp-cli already started) does three things in a
//! `tokio::select!` per iteration:
//!
//! 1. **Input**: crossterm [`EventStream`] key/mouse/resize events.
//! 2. **Agent frames**: the `Runtime` frame channel from
//!    [`crate::RunConfig::frames`], queued via `App::push_frame` and applied
//!    by `App::pump_events` (the app coalesces `message_update` bursts; the
//!    controller re-arms a short pump tick while one is pending).
//! 3. **Draw**: after any state change, [`crate::App`] renders into the
//!    inline [`render::Shell`]: finished transcript rows are *committed* to
//!    the terminal scrollback; only the live region (streaming message,
//!    editor, status line) is repainted.
//!
//! [`crate::Action`]s produced by the app are pushed synchronously into the
//! [`crate::RuntimePort`] (the omp-cli bridge spawns the async `Runtime`
//! calls). There is exactly one mutable `App` and it lives on this loop —
//! components never share state; they receive `&mut` from the app.

use std::io;
use std::pin::Pin;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::event::{Event, EventStream};
use futures::StreamExt;
use ratatui::text::Line;
use tokio::time::{sleep, Sleep};

use crate::app::App;
use crate::port::RuntimePort;
use crate::render::{ResizeAction, Shell};
use crate::RunConfig;

/// Height of the live region (chat viewport + editor + status line) in rows.
///
/// The chat area fills whatever remains after the editor and status line, so
/// this bounds the inline viewport; committed history grows above it.
const LIVE_HEIGHT: u16 = 12;

/// Re-pump interval while a `message_update` coalesce window is open
/// (TS `EventController` `#MESSAGE_UPDATE_COALESCE_MS` is 33 ms; app.rs owns
/// the exact window, this only needs to be in the same ballpark).
const PUMP_TICK: Duration = Duration::from_millis(33);

type StdShell = Shell<crate::render::ShellBackend, io::Stdout>;

/// Entry point invoked by [`crate::run`]. Owns terminal lifecycle.
///
/// # Errors
///
/// Fails if raw mode / inline viewport setup fails or a draw errors.
pub async fn run(cfg: RunConfig, port: impl RuntimePort) -> Result<()> {
    install_panic_hook();
    let mut shell = Shell::enter(LIVE_HEIGHT).context("initialize terminal")?;
    let result = drive(cfg, port, &mut shell).await;
    let restore = shell.leave().context("restore terminal");
    result.and(restore)
}

/// Run the loop; terminal cleanup happens in the caller.
async fn drive(cfg: RunConfig, port: impl RuntimePort, shell: &mut StdShell) -> Result<()> {
    let mut keys = EventStream::new();
    let mut frames = cfg.frames;
    let mut app = App::new(cfg.model, &cfg.cwd);

    // Chrome/theme startup (chrome.md §13 P0):
    // - theme: OSC 11 / COLORFGBG appearance → `theme.dark`/`theme.light`
    //   (review2 §11 #6) instead of the 8-token hard-coded default.
    // - keybindings: user overrides from ~/.omp/agent/keybindings.yml
    //   (review2 §11 #15).
    // - model catalogue + session dir for the `/model` and `/resume`
    //   overlays.
    let appearance = crate::theme::detect_appearance(cfg.osc11_bg);
    app.set_appearance(appearance);
    match crate::theme::load_configured(None, None, appearance) {
        Ok(theme) => app.set_theme(theme),
        Err(e) => tracing::warn!("theme load failed, keeping default: {e}"),
    }
    if let Some(warning) = app.load_user_keybindings() {
        tracing::warn!("{warning}");
    }
    app.configure(cfg.models, cfg.session_sources);

    shell.draw_live(|f| app.render(f)).context("initial draw")?;

    loop {
        let now = Instant::now();
        app.poll_deadlines(now, &port);
        let mut pump_tick: Option<Pin<Box<Sleep>>> =
            app.has_pending_update().then(|| Box::pin(sleep(PUMP_TICK)));

        tokio::select! {
            biased;
            maybe_key = keys.next() => {
                let Some(Ok(event)) = maybe_key else { continue };
                match event {
                    Event::Resize(_, _) => {
                        if shell.on_resize().context("resize")? == ResizeAction::RebuildScrollback {
                            // "Rebuild" scrollback mode (mux owns the grid):
                            // re-commit the transcript at the new width.
                            let lines = take_scrollback(&mut app, shell.width(), Rebuild::Yes);
                            shell.commit(&lines).context("rebuild scrollback")?;
                        }
                        shell.draw_live(|f| app.render(f)).context("draw")?;
                    }
                    _ => {
                        if !app.handle_terminal_event(&event, &port) {
                            break; // Action::Quit
                        }
                    }
                }
            }
            maybe_frame = frames.recv() => {
                let Some(frame) = maybe_frame else { continue };
                app.push_frame(&frame, Instant::now());
                if app.pump_events(Instant::now()) {
                    commit_retired(&mut app, shell).context("commit scrollback")?;
                }
            }
            () = async { match pump_tick.as_mut() { Some(t) => t.as_mut().await, None => std::future::pending().await } } => {
                if app.pump_events(Instant::now()) {
                    commit_retired(&mut app, shell).context("commit scrollback")?;
                }
            }
        }

        // Key/resize paths also retire rows (e.g. the echoed user prompt on
        // submit is a settle point) — the drain is watermark-based and cheap.
        commit_retired(&mut app, shell).context("commit scrollback")?;

        if app.has_fullscreen_overlay() {
            // Full-screen overlays borrow the alternate screen for the
            // duration of the draw (kitty flags re-pushed there by Shell).
            shell
                .with_fullscreen(|f| app.render(f))
                .context("fullscreen paint")?;
        } else {
            shell.draw_live(|f| app.render(f)).context("draw")?;
        }
    }
    // Teardown the terminal title (stops the spinner thread) before the
    // shell restores the screen; the app's `Drop` is the panic-path backup.
    app.dispose_title();
    Ok(())
}

/// Rebuild vs incremental scrollback commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rebuild {
    No,
    Yes,
}

/// Take transcript rows that finished and are safe to commit to scrollback
/// (`rebuild` re-commits the full transcript after a width change inside a
/// mux; otherwise only rows retired since the last call — the watermark only
/// advances at settle events (`message_end` / tool end / turn end), so draining
/// once per tick is exactly pi-tui's history-batch ack points).
fn take_scrollback(app: &mut App, width: u16, rebuild: Rebuild) -> Vec<Line<'static>> {
    app.take_scrollback_lines(width, rebuild == Rebuild::Yes)
}

/// Drain retired rows and commit them above the live region. No-op when
/// nothing retired since the last call.
fn commit_retired(app: &mut App, shell: &mut StdShell) -> Result<()> {
    let lines = take_scrollback(app, shell.width(), Rebuild::No);
    shell.commit(&lines)?;
    Ok(())
}

/// Restore the terminal on panic so the shell is usable again.
fn install_panic_hook() {
    use std::io::Write as _;

    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = crossterm::terminal::disable_raw_mode();
        let mut out = io::stdout();
        // Best-effort unwind of everything `Shell::enter` enabled:
        // kitty pop, bracketed paste off, DECCKM normal, cursor visible.
        let _ = out.write_all(b"\x1b[<u\x1b[?2004l\x1b[?1l\x1b>\x1b[?25h");
        let _ = out.flush();
        default(info);
    }));
}

#[cfg(test)]
mod tests {
    //! FakePort-driven smoke tests for the controller's frame→app pipeline
    //! (review2 §8: controller.rs had zero tests). The terminal `Shell`
    //! itself needs a TTY, so these drive the same per-frame entry point the
    //! select loop uses (`App::push_frame` + `pump_events` + `handle_key`)
    //! and render through `TestBackend`.

    use super::*;
    use crate::app::App;
    use crate::port::FakePort;
    use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;
    use std::path::Path;

    /// The frame path of the `drive` loop: enqueue + apply.
    fn apply_frame(app: &mut App, frame: &serde_json::Value, now: Instant) -> bool {
        app.push_frame(frame, now);
        app.pump_events(now)
    }

    fn test_app() -> App {
        App::new("p/m".into(), Path::new("/repo"))
    }

    #[test]
    fn agent_frames_drive_title_and_notify() {
        let mut app = test_app();
        let t0 = Instant::now();

        apply_frame(&mut app, &json!({"type":"agent_start"}), t0);
        let title = app.title().current_title().unwrap_or_default();
        assert!(title.starts_with('π'), "title: {title:?}");
        assert!(title.contains("repo"), "title: {title:?}");

        app.push_frame(
            &json!({"type":"message_update","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}),
            t0,
        );
        assert!(app.pump_events(t0 + Duration::from_millis(40)));
        apply_frame(&mut app, &json!({"type":"agent_end"}), t0);
        let title = app.title().current_title().unwrap_or_default();
        assert!(title.contains('>'), "idle title: {title:?}");

        // Turn settle fires the notification (recorded by the sink).
        assert!(apply_frame(&mut app, &json!({"type":"session_settled"}), t0));
        assert_eq!(app.notify_sink().record(), vec!["omp: Turn complete".to_string()]);
    }

    #[test]
    fn failed_prompt_result_notifies_and_idles_title() {
        let mut app = test_app();
        let t0 = Instant::now();
        app.push_frame(&json!({"type":"agent_start"}), t0);
        app.pump_events(t0);
        app.push_frame(
            &json!({"type":"prompt_result","success":false,"error":"boom"}),
            t0,
        );
        app.pump_events(t0);
        assert_eq!(
            app.notify_sink().record(),
            vec!["omp: Turn failed: boom".to_string()]
        );
        let _port = FakePort::default();
    }

    #[test]
    fn frame_pipeline_renders_and_dispatches_keys() {
        let mut app = test_app();
        let port = FakePort::default();
        let t0 = Instant::now();

        apply_frame(
            &mut app,
            &json!({"type":"message_update","message":{"role":"assistant","content":[{"type":"text","text":"answer"}]}}),
            t0,
        );
        apply_frame(&mut app, &json!({"type":"agent_end"}), t0);

        // A keystroke through the controller entry point submits a prompt.
        assert!(app.handle_terminal_event(
            &Event::Key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE)),
            &port
        ));
        assert!(app.handle_terminal_event(
            &Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            &port
        ));
        assert_eq!(
            port.log.lock().clone(),
            vec!["submit(None, \"q\", None)".to_string()]
        );

        // The app still paints onto a TestBackend after the frame burst.
        let backend = TestBackend::new(80, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
        let buf = terminal.backend().buffer();
        let row: String = (0..80).map(|x| buf[(x, 7)].symbol()).collect();
        assert!(row.contains('❯'), "editor row: {row:?}");
    }

    #[test]
    fn fullscreen_overlay_flag_follows_tree_overlay() {
        let mut app = test_app();
        let port = FakePort::default();
        let t0 = Instant::now();
        // user message → assistant reply, with ids/parentIds
        apply_frame(
            &mut app,
            &json!({"type":"message_end","message":{"id":"u1","role":"user","content":[{"type":"text","text":"first"}]}}),
            t0,
        );
        apply_frame(
            &mut app,
            &json!({"type":"message_end","message":{"id":"a1","parentId":"u1","role":"assistant","content":[{"type":"text","text":"ok"}]}}),
            t0,
        );
        // double-Esc on an idle, empty editor opens the rewind tree
        let esc = Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(app.handle_terminal_event(&esc, &port));
        assert!(!app.has_fullscreen_overlay());
        assert!(app.handle_terminal_event(&esc, &port));
        assert!(app.has_fullscreen_overlay(), "rewind tree is fullscreen");

        // Enter confirms the rewind selection → port.rewind_entry.
        let enter = Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.handle_terminal_event(&enter, &port));
        assert!(port
            .log
            .lock()
            .iter()
            .any(|c| c.starts_with("rewind_entry(")));
        assert!(!app.has_fullscreen_overlay(), "tree closed after rewind");
    }
}
