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
