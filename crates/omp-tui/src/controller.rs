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

/// Arm the animation frame timer: keep a running deadline, start one a full
/// `interval` out when animation begins, and drop it when nothing animates
/// (`interval == None`) so an idle loop never wakes.
fn arm_frame_timer(
    deadline: Option<Instant>,
    now: Instant,
    interval: Option<Duration>,
) -> Option<Instant> {
    let interval = interval?;
    Some(deadline.unwrap_or(now + interval))
}

/// After a frame deadline fired: schedule the next on the fixed cadence
/// (`fired + interval`, so loop latency does not stretch the period); when
/// the loop fell behind, resume from `now` instead of bursting catch-up
/// frames.
fn advance_frame_timer(
    deadline: Option<Instant>,
    now: Instant,
    interval: Option<Duration>,
) -> Option<Instant> {
    let interval = interval?;
    let next = deadline.map_or(now + interval, |d| d + interval);
    Some(if next > now { next } else { now + interval })
}

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
    if let Err(e) = app.load_settings_theme() {
        tracing::warn!("theme load failed, keeping default: {e}");
    }
    if let Some(warning) = app.load_user_keybindings() {
        tracing::warn!("{warning}");
    }
    app.configure(cfg.models, cfg.session_sources);
    app.set_model_name(cfg.model_name);
    app.set_welcome(&cfg.version);
    app.start_word_completion();
    commit_retired(&mut app, shell).context("commit welcome")?;

    shell.draw_live(|f| app.render(f)).context("initial draw")?;

    let mut signals = TermSignals::new();
    let mut frame_deadline: Option<Instant> = None;
    // Raw-mode watchdog: anything sharing our controlling terminal (a tool's
    // child that opens /dev/tty, `stty`, an interop .exe) can flip the tty
    // back to cooked mode; keys then sit in the line buffer and the TUI looks
    // frozen. Check once a second and re-assert raw mode when it was lost.
    let mut raw_guard = tokio::time::interval(RAW_GUARD_TICK);
    raw_guard.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        let now = Instant::now();
        // Keep the animation clock running only while something animates;
        // an idle app arms no timer and the loop blocks on input.
        frame_deadline = arm_frame_timer(frame_deadline, now, app.animation_interval());
        app.poll_deadlines(now, &port);
        let mut pump_tick: Option<Pin<Box<Sleep>>> =
            app.has_pending_update().then(|| Box::pin(sleep(PUMP_TICK)));
        // Pending paste results (in-flight clipboard read, open large-paste
        // menu pick): wake the select so `poll_paste_results` runs — and the
        // following paint lands — even with no further terminal events (an
        // idle loop otherwise never iterates).
        let mut paste_tick: Option<Pin<Box<Sleep>>> = app
            .has_pending_paste_results()
            .then(|| Box::pin(sleep(PUMP_TICK)));

        tokio::select! {
            biased;
            () = signals.recv() => {
                // SIGTERM/SIGHUP/SIGINT: leave through the normal teardown so
                // the alt screen, mouse tracking and keyboard modes are
                // restored (TS postmortem `terminal-restore`).
                break;
            }
            maybe_key = keys.next() => {
                let Some(Ok(event)) = maybe_key else { continue };
                match event {
                    Event::Resize(_, _) => {
                        // While a full-screen overlay covers the normal buffer
                        // the shell defers this to the alt exit.
                        if shell.on_resize().context("resize")? == ResizeAction::RebuildScrollback {
                            // "Rebuild" scrollback mode (mux owns the grid):
                            // re-commit the transcript at the new width.
                            let lines = take_scrollback(&mut app, shell.width(), Rebuild::Yes);
                            shell.commit(&lines).context("rebuild scrollback")?;
                        }
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
                app.push_frame_owned(frame, Instant::now());
                if app.pump_events(Instant::now()) {
                    commit_retired(&mut app, shell).context("commit scrollback")?;
                }
            }
            () = async { match pump_tick.as_mut() { Some(t) => t.as_mut().await, None => std::future::pending().await } } => {
                if app.pump_events(Instant::now()) {
                    commit_retired(&mut app, shell).context("commit scrollback")?;
                }
            }
            _ = raw_guard.tick() => {
                if restore_raw_mode_if_lost() {
                    tracing::warn!("terminal left raw mode (cooked tty); raw mode re-applied");
                }
            }
            () = async { match paste_tick.as_mut() { Some(t) => t.as_mut().await, None => std::future::pending().await } } => {
                app.poll_paste_results();
            }
            () = async { match frame_deadline { Some(at) => tokio::time::sleep_until(at.into()).await, None => std::future::pending().await } } => {
                // Animation frame: nothing to apply — the paint below ticks
                // the reveal and advances the spinner.
                frame_deadline = advance_frame_timer(frame_deadline, Instant::now(), app.animation_interval());
            }
        }

        paint(&mut app, shell)?;
    }
    // Teardown the terminal title (stops the spinner thread) before the
    // shell restores the screen; the app's `Drop` is the panic-path backup.
    app.persist_word_completion();
    app.dispose_title();
    Ok(())
}

const RAW_GUARD_TICK: Duration = Duration::from_secs(1);

/// `true` when stdin is a tty in canonical (cooked) mode — raw mode the TUI
/// and raw mode was re-applied. Windows consoles have no termios cooked/raw
/// duality crossterm cannot already see, so the guard is a no-op there.
#[cfg(unix)]
fn restore_raw_mode_if_lost() -> bool {
    let stdin = std::io::stdin();
    let Ok(t) = rustix::termios::tcgetattr(&stdin) else {
        return false;
    };
    if !t.local_modes.contains(rustix::termios::LocalModes::ICANON) {
        return false;
    }
    // crossterm no-ops `enable_raw_mode` while it believes raw mode is on:
    // drop its saved state first, then set raw mode again.
    let _ = crossterm::terminal::disable_raw_mode();
    crossterm::terminal::enable_raw_mode().is_ok()
}

/// Windows: the concept does not apply (no termios; crossterm owns the
/// console mode), so nothing to detect or re-apply.
#[cfg(not(unix))]
fn restore_raw_mode_if_lost() -> bool {
    false
}

/// One paint after a state change — the TS `tui.ts` render-pass
/// short-circuit: while an open overlay is full-screen, hold the alternate
/// screen and paint only the modal there (normal screen and scrollback
/// accounting untouched); when the last one closes, leave the alt screen
/// (the terminal restores the inline transcript + live region), apply any
/// deferred resize, commit rows that retired meanwhile and repaint the live
/// region. Mouse reporting follows [`App::wants_mouse`] either way.
fn paint(app: &mut App, shell: &mut StdShell) -> Result<()> {
    if app.has_fullscreen_overlay() {
        let entering = !shell.is_alt();
        shell
            .draw_fullscreen(|f| app.render_fullscreen(f))
            .context("fullscreen paint")?;
        if entering {
            tracing::debug!("fullscreen overlay: entered alternate screen");
        }
        shell
            .set_mouse_tracking(app.wants_mouse())
            .context("mouse tracking")?;
        return Ok(());
    }
    // Mouse off (or inline capture) first, then leave alt: TS fused exit
    // order `mouseExit + enhancementExit + ?1049l`.
    shell
        .set_mouse_tracking(app.wants_mouse())
        .context("mouse tracking")?;
    if shell.is_alt()
        && shell.leave_alt().context("leave alternate screen")? == ResizeAction::RebuildScrollback
    {
        let lines = take_scrollback(app, shell.width(), Rebuild::Yes);
        shell.commit(&lines).context("rebuild scrollback")?;
    }
    if app.take_clear_scrollback() {
        // `/new`: erase what the old session pushed into the terminal's
        // scrollback, then re-commit the header for the fresh transcript.
        shell.clear_history().context("clear scrollback")?;
        let lines = take_scrollback(app, shell.width(), Rebuild::Yes);
        shell.commit(&lines).context("rebuild scrollback")?;
    }
    // Key/resize paths also retire rows (e.g. the echoed user prompt on
    // submit is a settle point) — the drain is watermark-based and cheap.
    commit_retired(app, shell).context("commit scrollback")?;
    shell.draw_live(|f| app.render(f)).context("draw")?;
    Ok(())
}

/// Termination signals that must unwind through the terminal restore
/// (raw mode turns Ctrl+C into a key, so SIGINT only arrives via `kill`).
struct TermSignals {
    #[cfg(unix)]
    streams: Vec<tokio::signal::unix::Signal>,
}

impl TermSignals {
    fn new() -> Self {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let streams = [
                SignalKind::terminate(),
                SignalKind::hangup(),
                SignalKind::interrupt(),
            ]
            .into_iter()
            .filter_map(|kind| signal(kind).ok())
            .collect();
            Self { streams }
        }
        #[cfg(not(unix))]
        {
            Self {}
        }
    }

    /// Resolves when any watched signal arrives; pending forever otherwise.
    async fn recv(&mut self) {
        #[cfg(unix)]
        {
            if self.streams.is_empty() {
                return std::future::pending().await;
            }
            let futs = self.streams.iter_mut().map(|s| Box::pin(s.recv()));
            let _ = futures::future::select_all(futs).await;
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
    }
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
/// nothing retired since the last call, and deferred while a full-screen
/// overlay holds the alternate screen (inserting there would scroll the
/// rows into the alt buffer and lose them; the watermark keeps them until
/// the overlay closes).
fn commit_retired(app: &mut App, shell: &mut StdShell) -> Result<()> {
    if shell.is_alt() {
        return Ok(());
    }
    let lines = take_scrollback(app, shell.width(), Rebuild::No);
    shell.commit(&lines)?;
    Ok(())
}

/// Restore the terminal on panic so the shell is usable again (TS
/// `emergencyTerminalRestore`: mouse tracking off, leave the alt screen only
/// when a full-screen overlay holds it, keyboard modes popped).
fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        crate::render::emergency_restore();
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

    /// Pinned to built-in default settings so the notify gates never read
    /// the process-global `HOME` (mutated by the temp-HOME app tests).
    fn test_app() -> App {
        let mut app = App::new("p/m".into(), Path::new("/repo"));
        app.set_settings_global(None);
        // Capture deliveries instead of ringing the real terminal / desktop.
        app.set_notify_hook(std::sync::Arc::new(|_| {}));
        app
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
        assert!(apply_frame(
            &mut app,
            &json!({"type":"session_settled"}),
            t0
        ));
        assert_eq!(
            app.notify_sink().record(),
            vec!["omp: Complete".to_string()]
        );
    }

    #[test]
    fn failed_prompt_result_notifies_and_idles_title() {
        let mut app = test_app();
        let t0 = Instant::now();
        app.push_frame(&json!({"type":"agent_start"}), t0);
        app.pump_events(t0);
        app.push_frame(
            &json!({"type":"prompt_result","status":"error","error":{"message":"boom"}}),
            t0,
        );
        app.pump_events(t0);
        // `error.notify` defaults to off (TS): silent, and never a "Complete" toast.
        assert!(app
            .notify_sink()
            .record()
            .iter()
            .all(|l| l != "omp: Complete"));
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
        let row: String = (0..80).map(|x| buf[(x, 9)].symbol()).collect();
        assert!(row.starts_with("╰─"), "editor row: {row:?}");
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

#[cfg(test)]
mod frame_timer_tests {
    use super::*;

    const IV: Duration = Duration::from_millis(33);

    #[test]
    fn idle_arms_nothing_and_clears_a_running_timer() {
        let now = Instant::now();
        assert_eq!(arm_frame_timer(None, now, None), None);
        assert_eq!(arm_frame_timer(Some(now + IV), now, None), None);
    }

    #[test]
    fn animation_start_arms_one_interval_out_and_keeps_running_deadline() {
        let now = Instant::now();
        assert_eq!(arm_frame_timer(None, now, Some(IV)), Some(now + IV));
        let running = now + Duration::from_millis(10);
        // Unrelated loop wakeups (keys, frames) must not push the frame back.
        assert_eq!(arm_frame_timer(Some(running), now, Some(IV)), Some(running));
    }

    #[test]
    fn advance_keeps_fixed_cadence_and_never_bursts() {
        let t0 = Instant::now();
        let fired = t0 + IV;
        // On-time wake: next is fired + interval despite small latency.
        let now = fired + Duration::from_millis(2);
        assert_eq!(
            advance_frame_timer(Some(fired), now, Some(IV)),
            Some(fired + IV)
        );
        // Fell far behind: resume one interval from now, no catch-up burst.
        let late = fired + Duration::from_millis(500);
        assert_eq!(
            advance_frame_timer(Some(fired), late, Some(IV)),
            Some(late + IV)
        );
        // Animation ended meanwhile: timer disarms.
        assert_eq!(advance_frame_timer(Some(fired), now, None), None);
    }

    #[test]
    fn app_animates_only_while_streaming_or_revealing() {
        use serde_json::json;
        let mut app = App::new("p/m".into(), std::path::Path::new("/repo"));
        let t0 = Instant::now();
        assert_eq!(
            app.animation_interval(),
            None,
            "idle app must not arm a timer"
        );
        app.push_frame(&json!({"type":"agent_start"}), t0);
        app.pump_events(t0);
        // Loader visible: spinner cadence.
        assert_eq!(app.animation_interval(), Some(Duration::from_millis(80)));
        app.push_frame(
            &json!({"type":"message_start","message":{"role":"assistant","content":[]}}),
            t0,
        );
        let long = "word ".repeat(200);
        app.push_frame(
            &json!({"type":"message_update","message":{"role":"assistant","content":[{"type":"text","text":long}]}}),
            t0,
        );
        app.pump_events(t0 + Duration::from_millis(40));
        // Reveal backlog: 30 fps cadence until caught up.
        assert_eq!(app.animation_interval(), Some(Duration::from_millis(33)));
        app.push_frame(&json!({"type":"agent_end"}), t0);
        app.pump_events(t0 + Duration::from_millis(80));
    }
}
