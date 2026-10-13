//! OSC 11 background-color probe — the startup half of TS
//! `terminal.ts#requestBackgroundColor` (chrome.md §13 "Dark/light:
//! COLORFGBG + OSC 11 luminance").
//!
//! Runs once, before the shell renderer starts (the response bytes would
//! otherwise sit in stdin and confuse the crossterm event parser). The
//! embedding binary (omp-cli) calls [`probe_osc11_background`] and hands the
//! result to the app through `RunConfig::osc11_bg`; the theme loader then
//! picks `theme.dark`/`theme.light` via [`crate::theme::detect_appearance`].
//!
//! Failure modes all return `None` (probe is best-effort; COLORFGBG and the
//! dark default remain as fallbacks): not a TTY, no response within the
//! deadline, or an unparseable response.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::time::{Duration, Instant};

use crate::theme::Rgb;

/// OSC 11 query: `OSC 11 ; ? ST`.
const QUERY: &[u8] = b"\x1b]11;?\x1b\\";

/// Parse an OSC 11 response (`...11;rgb:RRRR/GGGG/BBBB...` or `rgba:` with
/// trailing alpha) into an 8-bit [`Rgb`] (TS `colorLuma` input). Returns
/// `None` for anything else.
#[must_use]
pub fn parse_osc11_response(bytes: &[u8]) -> Option<Rgb> {
    let text = String::from_utf8_lossy(bytes);
    let marker = text.find("11;")?;
    let rest = &text[marker + 3..];
    let values = rest.strip_prefix("rgb:").or_else(|| rest.strip_prefix("rgba:"))?;
    // Channels are `/`-separated; the response may end with BEL or ST.
    let mut parts = values.split(|c: char| matches!(c, '/' | ';' | '\u{7}' | '\u{1b}'));
    let mut next = || {
        let hex = parts.next()?.trim_start_matches("0x");
        // 16-bit channel values (usually `0000`-`ffff`): take the high byte,
        // matching TS which scales to 0-255 the same way.
        let v = u32::from_str_radix(hex, 16).ok()?;
        Some((v >> 8) as u8)
    };
    let r = next()?;
    let g = next()?;
    let b = next()?;
    Some(Rgb { r, g, b })
}

/// Ask the terminal for its background color; `None` on any failure.
///
/// Enables raw mode so the response arrives byte-wise, writes the query to
/// `/dev/tty`, then polls (via `crossterm::event::poll`, which only checks
/// readiness — nothing is consumed from the buffer) and reads the response
/// from a dedicated `/dev/tty` handle. Restores the previous mode on every
/// path.
#[must_use]
pub fn probe_osc11_background(timeout: Duration) -> Option<Rgb> {
    use std::io::IsTerminal;
    if !std::io::stdout().is_terminal() {
        return None;
    }
    let mut tty = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")
        .ok()?;
    crossterm::terminal::enable_raw_mode().ok()?;
    let result = (|| {
        tty.write_all(QUERY).ok()?;
        tty.flush().ok()?;
        let deadline = Instant::now() + timeout;
        let mut buf: Vec<u8> = Vec::new();
        let mut byte = [0u8; 1];
        while Instant::now() < deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if !crossterm::event::poll(remaining).unwrap_or(false) {
                break;
            }
            if tty.read(&mut byte).ok()? == 0 {
                break;
            }
            buf.push(byte[0]);
            // ST (BEL or ESC \) terminates the response.
            if byte[0] == 0x07 || buf.ends_with(b"\x1b\\") {
                break;
            }
        }
        parse_osc11_response(&buf)
    })();
    let _ = crossterm::terminal::disable_raw_mode();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_standard_rgb_response() {
        let resp = b"\x1b]11;rgb:1c1c/1c1c/1c1c\x1b\\";
        assert_eq!(
            parse_osc11_response(resp),
            Some(Rgb { r: 0x1c, g: 0x1c, b: 0x1c })
        );
    }

    #[test]
    fn parses_bel_terminated_response() {
        let resp = b"\x1b]11;rgb:ffff/ffff/ffff\x07";
        assert_eq!(
            parse_osc11_response(resp),
            Some(Rgb { r: 0xff, g: 0xff, b: 0xff })
        );
    }

    #[test]
    fn ignores_garbage_and_missing_response() {
        assert_eq!(parse_osc11_response(b""), None);
        assert_eq!(parse_osc11_response(b"\x1b]10;rgb:0000/0000/0000\x1b\\"), None);
        assert_eq!(parse_osc11_response(b"hello"), None);
    }
}
