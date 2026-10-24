//! Bracketed paste handling — port of `pi-tui/src/bracketed-paste.ts` plus the
//! paste-relevant bits of `stdin-buffer.ts` (§6.2–6.3 of `tui-spec/core.md`).
//!
//! The embedding terminal loop (crossterm) already unwraps the
//! `\x1b[200~ … \x1b[201~` markers into [`crossterm::event::Event::Paste`];
//! this module owns the *semantics* on top of that payload:
//!
//! - tmux `extended-keys-format` re-encodes Ctrl+letters inside a paste as
//!   key-event escapes; [`decode_reencoded_paste_controls`] must run before any
//!   control-character stripping, otherwise the printable escape tail
//!   (`[106;5u`) leaks into the editor.
//! - A pasted trailing newline sharing the submit keypress is split off by
//!   [`split_trailing_enter`] so the editor receives the text and the app the
//!   submit gesture (TS: the `paste` event's `enter` parameter).
//!
//! [`BracketedPasteHandler`] itself is kept for callers that see raw bytes
//! (tests, future raw-stdin paths): it buffers chunks between the markers and
//! recovers from a lost end marker via a byte cap.

use crossterm::event::Event;

/// Byte cap for buffered paste content (TS `DEFAULT_BYTE_LIMIT`): 64 MiB.
pub const DEFAULT_BYTE_LIMIT: usize = 64 * 1024 * 1024;

const PASTE_START: &str = "\x1b[200~";
const PASTE_END: &str = "\x1b[201~";

/// Decode tmux's re-encoded control bytes (both `extended-keys-format`
/// variants) inside a bracketed-paste payload back to their literal byte, e.g.
/// `ESC [ 106 ; 5 u` (Ctrl+J) → `"\n"`. Only Ctrl+<letter> is decoded
/// (codepoints a-z/A-Z → `0x01..=0x1A`) — that is the set tmux actually
/// re-encodes from paste content; NUL/ESC/DEL are left untouched and stripped
/// later by the editor. Call **before** control stripping.
#[must_use]
pub fn decode_reencoded_paste_controls(text: &str) -> String {
    decode_reencoded(text)
}

fn decode_reencoded(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut pos = 0;
    while pos < bytes.len() {
        // csi-u form: ESC [ <codepoint> ; 5 u
        if let Some(rest) = strip_prefix_ascii(bytes, pos, b"\x1b[") {
            if let Some((cp, next)) = parse_decimal(bytes, rest) {
                if let Some(after) = strip_prefix_ascii(bytes, next, b";5u") {
                    if let Some(b) = ctrl_letter_byte(cp) {
                        out.push(b);
                        pos = after;
                        continue;
                    }
                }
            }
            // xterm form: ESC [ 27 ; 5 ; <codepoint> ~
            if let Some(after27) = strip_prefix_ascii(bytes, rest, b"27;5;") {
                if let Some((cp, next)) = parse_decimal(bytes, after27) {
                    if let Some(after) = strip_prefix_ascii(bytes, next, b"~") {
                        if let Some(b) = ctrl_letter_byte(cp) {
                            out.push(b);
                            pos = after;
                            continue;
                        }
                    }
                }
            }
        }
        out.push(bytes[pos]);
        pos += 1;
    }
    // Input came from a &str; decoding only replaces multi-byte sequences with
    // single ASCII bytes, so the result is still valid UTF-8.
    String::from_utf8(out).unwrap_or_default()
}

fn ctrl_letter_byte(cp: u64) -> Option<u8> {
    match cp {
        97..=122 => u8::try_from(cp - 96).ok(), // a-z → Ctrl+A..Ctrl+Z
        65..=90 => u8::try_from(cp - 64).ok(),  // A-Z → Ctrl+A..Ctrl+Z
        _ => None,
    }
}

fn strip_prefix_ascii(bytes: &[u8], i: usize, prefix: &[u8]) -> Option<usize> {
    if bytes.len() >= i + prefix.len() && &bytes[i..i + prefix.len()] == prefix {
        Some(i + prefix.len())
    } else {
        None
    }
}

fn parse_decimal(bytes: &[u8], mut i: usize) -> Option<(u64, usize)> {
    let start = i;
    let mut v: u64 = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        v = v
            .saturating_mul(10)
            .saturating_add(u64::from(bytes[i] - b'0'));
        i += 1;
    }
    if i == start {
        None
    } else {
        Some((v, i))
    }
}

/// Split a paste payload's trailing newline off as the submit gesture
/// (TS `StdinBuffer`: an Enter sharing the paste's final read is delivered as
/// the paste event's `enter` flag). CRLF counts once. Returns
/// `(text_without_enter, enter_seen)`.
#[must_use]
pub fn split_trailing_enter(content: &str) -> (String, bool) {
    if let Some(stripped) = content.strip_suffix("\r\n") {
        return (stripped.to_owned(), true);
    }
    if let Some(stripped) = content
        .strip_suffix('\n')
        .or_else(|| content.strip_suffix('\r'))
    {
        return (stripped.to_owned(), true);
    }
    (content.to_owned(), false)
}

/// Whether a crossterm event is a bracketed paste, and its payload.
#[must_use]
pub fn paste_of(event: &Event) -> Option<&str> {
    match event {
        Event::Paste(content) => Some(content.as_str()),
        _ => None,
    }
}

/// Outcome of feeding [`BracketedPasteHandler`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PasteResult {
    /// The data contained no paste sequence; process it normally.
    NotHandled,
    /// Consumed by paste buffering; `Some` when a complete paste (or an
    /// aborted runaway buffer) was assembled, `None` while still buffering.
    Handled(Option<String>),
}

/// State machine over the byte stream: outside → collecting → done (or
/// aborted by the byte limit / a missing end marker). Mirrors TS
/// `BracketedPasteHandler`, including nested-start stripping before the
/// current paste's first end marker.
#[derive(Debug)]
pub struct BracketedPasteHandler {
    buffer: String,
    active: bool,
    byte_limit: usize,
}

impl Default for BracketedPasteHandler {
    fn default() -> Self {
        Self::new(DEFAULT_BYTE_LIMIT)
    }
}

impl BracketedPasteHandler {
    /// New handler with the given buffered-paste byte cap.
    #[must_use]
    pub fn new(byte_limit: usize) -> Self {
        Self {
            buffer: String::new(),
            active: false,
            byte_limit,
        }
    }

    /// Feed one chunk of raw terminal data. See [`PasteResult`].
    ///
    /// Unlike the TS original (which returns a `remaining` tail for the
    /// caller to re-feed), this loops over the whole chunk: every *complete*
    /// paste found in the chunk is delivered — the first one as the returned
    /// content, any later ones appended to it — and a trailing incomplete
    /// paste stays buffered for the next chunk.
    pub fn process(&mut self, data: &str) -> PasteResult {
        let mut delivered = String::new();
        let mut any_delivered = false;
        let mut rest = data;
        loop {
            if !self.active {
                let Some(pos) = rest.find(PASTE_START) else {
                    break;
                };
                self.active = true;
                self.buffer.clear();
                rest = &rest[pos + PASTE_START.len()..];
            }
            let Some(end) = rest.find(PASTE_END) else {
                self.buffer.push_str(&rest.replace(PASTE_START, ""));
                break;
            };
            {
                self.buffer.push_str(&rest[..end].replace(PASTE_START, ""));
                if any_delivered {
                    delivered.push_str(&self.buffer);
                } else {
                    delivered = std::mem::take(&mut self.buffer);
                    any_delivered = true;
                }
                self.active = false;
                rest = &rest[end + PASTE_END.len()..];
                if rest.is_empty() {
                    break;
                }
            }
        }
        if self.active && self.buffer.len() > self.byte_limit {
            // Byte cap: a lost/corrupted end marker (ssh/tmux truncation)
            // must not consume unbounded memory. Deliver the accumulated
            // bytes so they are neither lost nor held forever.
            let content = std::mem::take(&mut self.buffer);
            self.active = false;
            return PasteResult::Handled(Some(content));
        }
        match (any_delivered, self.active) {
            (true, _) => PasteResult::Handled(Some(delivered)),
            (false, true) => PasteResult::Handled(None),
            (false, false) => PasteResult::NotHandled,
        }
    }

    /// Whether a paste is currently being buffered.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_tmux_csi_u_ctrl_letters() {
        assert_eq!(decode_reencoded_paste_controls("a\x1b[106;5ub"), "a\nb");
        assert_eq!(decode_reencoded_paste_controls("\x1b[27;5;105~"), "\t");
        assert_eq!(decode_reencoded_paste_controls("\x1b[65;5u"), "\x01");
        // Non-letter ctrl combos are left untouched.
        assert_eq!(decode_reencoded_paste_controls("\x1b[0;5u"), "\x1b[0;5u");
        assert_eq!(decode_reencoded_paste_controls("plain"), "plain");
    }

    #[test]
    fn splits_trailing_enter() {
        assert_eq!(split_trailing_enter("hi\n"), ("hi".into(), true));
        assert_eq!(split_trailing_enter("hi\r\n"), ("hi".into(), true));
        assert_eq!(split_trailing_enter("hi\r"), ("hi".into(), true));
        assert_eq!(split_trailing_enter("hi"), ("hi".into(), false));
        assert_eq!(split_trailing_enter("a\nb"), ("a\nb".into(), false));
    }

    #[test]
    fn assembles_paste_across_chunks() {
        let mut h = BracketedPasteHandler::default();
        assert_eq!(h.process("\x1b[200~hel"), PasteResult::Handled(None));
        assert_eq!(
            h.process("lo \x1b[201~next"),
            PasteResult::Handled(Some("hello ".into()))
        );
        // After completion the handler is back outside paste mode.
        assert_eq!(h.process("x"), PasteResult::NotHandled);
    }

    #[test]
    fn strips_nested_starts_and_handles_start_in_buffer() {
        let mut h = BracketedPasteHandler::default();
        assert_eq!(
            h.process("\x1b[200~\x1b[200~a\x1b[201~\x1b[200~b\x1b[201~"),
            PasteResult::Handled(Some("ab".into()))
        );
        assert!(!h.is_active());
    }

    #[test]
    fn byte_cap_aborts_runaway_paste() {
        let mut h = BracketedPasteHandler::new(4);
        assert_eq!(
            h.process("\x1b[200~abcdef"),
            PasteResult::Handled(Some("abcdef".into()))
        );
        assert!(!h.is_active());
    }
}
