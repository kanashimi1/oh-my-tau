//! The interactive controller: terminal setup, the event pump and the draw
//! loop — the port of the wiring half of
//! `pi-coding-agent/src/modes/interactive-mode.ts` (its `run` loop) plus the
//! terminal plumbing of `pi-tui/src/tui.ts` / `terminal.ts`.
//!
//! ## Threading model
//!
//! One OS thread owns the terminal: the async `run` loop below (living on
//! the tokio runtime omp-cli already started) does three things in a
//! `tokio::select!` per iteration:
//!
//! 1. **Input**: crossterm [`EventStream`] key/mouse/resize events.
//! 2. **Agent frames**: the `Runtime` frame channel from
//!    [`crate::RunConfig::frames`], classified by [`crate::events::parse_frame`].
//! 3. **Draw**: after any state change, [`crate::App`] renders into a
//!    ratatui [`Terminal`] (crossterm backend, alternate screen, raw mode).
//!
//! [`crate::Action`]s produced by the app are pushed synchronously into the
//! [`crate::RuntimePort`] (the omp-cli bridge spawns the async `Runtime`
//! calls). There is exactly one mutable `App` and it lives on this loop —
//! components never share state; they receive `&mut` from the app.

use std::io::{self, Stdout};

use anyhow::{Context, Result};
use crossterm::event::EventStream;
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crate::events::parse_frame;
use crate::port::RuntimePort;
use crate::{App, RunConfig};

/// Entry point invoked by [`crate::run`]. Owns terminal lifecycle.
///
/// # Errors
///
/// Fails if raw mode / alternate screen setup fails or the terminal reports
/// a draw error.
pub async fn run(cfg: RunConfig, port: impl RuntimePort) -> Result<()> {
    install_panic_hook();
    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("enter alternate screen")?;
    let result = drive(cfg, port, &mut stdout).await;
    let restore = restore();
    result.and(restore)
}

/// Run the loop; terminal cleanup happens in the caller.
async fn drive(
    cfg: RunConfig,
    port: impl RuntimePort,
    stdout: &mut Stdout,
) -> Result<()> {
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("create terminal")?;
    let mut keys = EventStream::new();
    let mut frames = cfg.frames;
    let mut app = App::new(cfg.model, &cfg.cwd);

    terminal.draw(|f| app.render(f)).context("initial draw")?;

    loop {
        tokio::select! {
            biased;
            maybe_key = keys.next() => {
                let Some(Ok(event)) = maybe_key else { continue };
                if !app.handle_terminal_event(&event, &port) {
                    break; // Action::Quit
                }
            }
            maybe_frame = frames.recv() => {
                let Some(frame) = maybe_frame else { continue };
                app.apply_event(&parse_frame(&frame));
            }
        }
        terminal.draw(|f| app.render(f)).context("draw")?;
    }
    Ok(())
}

/// Restore terminal state (raw mode off, leave alternate screen).
fn restore() -> Result<()> {
    disable_raw_mode().context("disable raw mode")?;
    execute!(io::stdout(), LeaveAlternateScreen).context("leave alternate screen")?;
    Ok(())
}

/// Restore the terminal on panic so the shell is usable again.
fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        default(info);
    }));
}
