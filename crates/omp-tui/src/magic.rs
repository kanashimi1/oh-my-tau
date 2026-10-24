//! Magic keywords and ultra modes — shared by the runtime (notices) and the
//! TUI (editor gradients, status line).
//!
//! Magic keywords port `pi-tui/src/prompt/magic-keywords.ts` +
//! `markdown-prose.ts`: a keyword fires only as a standalone, case-sensitive
//! prose word — never inside a fenced code block, an inline code span or an
//! HTML/XML section, and never glued into an identifier, path, file name,
//! `a::word` reference or `word()` call. The runtime turns each hit into a
//! hidden notice for that one message; the editor paints it with a gradient.
//!
//! Ultra modes are the omp-rs effort level after `max`: `ultracode` (workflow
//! by default on every substantial task) and its hybrids with RLM / CLM.

use std::ops::Range;

/// One magic keyword: trigger word, editor gradient, settings gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keyword {
    /// Settings key suffix (`magicKeywords.<id>`) and notice type prefix
    /// (`<id>-notice`).
    pub id: &'static str,
    /// Exact lowercase trigger.
    pub word: &'static str,
    /// HSL hue sweep `(from, to)` in degrees across the word; values outside
    /// `0..360` wrap.
    pub hue: (f64, f64),
    /// Tools the notice's contract needs; the notice is skipped when any is
    /// not active in the session.
    pub requires: &'static [&'static str],
}

/// The registered keywords (TS `MAGIC_KEYWORDS` + omp-rs `ultracode`).
pub const KEYWORDS: &[Keyword] = &[
    Keyword {
        id: "ultrathink",
        word: "ultrathink",
        hue: (0.0, 330.0),
        requires: &[],
    },
    Keyword {
        id: "orchestrate",
        word: "orchestrate",
        hue: (150.0, 280.0),
        requires: &["task"],
    },
    Keyword {
        id: "workflow",
        word: "workflowz",
        hue: (30.0, 150.0),
        requires: &["task", "eval"],
    },
    Keyword {
        id: "jevify",
        word: "jevify",
        hue: (300.0, 420.0),
        requires: &["eval"],
    },
    // Sunset: yellow (50) through red/pink to violet (270).
    Keyword {
        id: "ultracode",
        word: "ultracode",
        hue: (410.0, 270.0),
        requires: &["task", "eval"],
    },
];

/// Whether keyword `id` is enabled (`magicKeywords.enabled` and
/// `magicKeywords.<id>`, both default on).
#[must_use]
pub fn enabled(settings: &omp_settings::Settings, id: &str) -> bool {
    settings.get_bool("magicKeywords.enabled").unwrap_or(true)
        && settings
            .get_bool(&format!("magicKeywords.{id}"))
            .unwrap_or(true)
}

/// Byte ranges of every registered keyword standing as prose in `text`,
/// sorted by start.
#[must_use]
pub fn ranges(text: &str) -> Vec<(Range<usize>, &'static Keyword)> {
    if !KEYWORDS.iter().any(|k| text.contains(k.word)) {
        return Vec::new();
    }
    let masked = mask_non_prose(text);
    let mut out = Vec::new();
    for k in KEYWORDS {
        out.extend(word_hits(&masked, k.word).into_iter().map(|r| (r, k)));
    }
    out.sort_by_key(|(r, _)| r.start);
    out
}

/// Whether `word` stands as prose in `text` (TS `containsMagicKeyword`).
#[must_use]
pub fn contains(text: &str, word: &str) -> bool {
    text.contains(word) && !word_hits(&mask_non_prose(text), word).is_empty()
}

/// Keywords present as prose in `text`, in registry order.
#[must_use]
pub fn present(text: &str) -> Vec<&'static Keyword> {
    KEYWORDS.iter().filter(|k| contains(text, k.word)).collect()
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Standalone occurrences of `word` (TS `LEFT_BOUNDARY` / `RIGHT_BOUNDARY`).
fn word_hits(text: &str, word: &str) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(pos) = text[from..].find(word) {
        let start = from + pos;
        let end = start + word.len();
        from = end;
        let before = &text[..start];
        let prev = before.chars().next_back();
        if prev.is_some_and(|c| is_word_char(c) || matches!(c, '.' | '/' | '\\' | '-'))
            || before.ends_with("::")
        {
            continue;
        }
        let after = &text[end..];
        let mut it = after.chars();
        let next = it.next();
        if next.is_some_and(|c| is_word_char(c) || matches!(c, '/' | '\\' | '-' | '(')) {
            continue;
        }
        if next == Some('.') && it.next().is_some_and(|c| is_word_char(c) || c == '-') {
            continue;
        }
        out.push(start..end);
    }
    out
}

/// Length-preserving copy of `text` with fenced code blocks, inline code
/// spans and HTML/XML sections blanked to spaces (newlines kept), so byte
/// offsets of a match address the original text (TS `maskNonProse`).
#[must_use]
pub fn mask_non_prose(text: &str) -> String {
    if !text.contains('`') && !text.contains('<') && !text.contains("~~~") {
        return text.to_owned();
    }
    let b = text.as_bytes();
    let n = b.len();
    let mut masked = vec![false; n];

    // Phase 1: fenced code blocks, line by line.
    let mut fence: Option<(u8, usize)> = None;
    let mut line_start = 0;
    while line_start <= n {
        let nl = text[line_start..].find('\n').map_or(n, |p| line_start + p);
        let line = &text[line_start..nl];
        let open = fence_open(line);
        if let Some((ch, len)) = fence {
            masked[line_start..nl].iter_mut().for_each(|m| *m = true);
            if let Some((indent, marker)) = open {
                let rest = &line[indent + marker.len()..];
                if marker.as_bytes()[0] == ch && marker.len() >= len && rest.trim().is_empty() {
                    fence = None;
                }
            }
        } else if let Some((indent, marker)) = open {
            let ch = marker.as_bytes()[0];
            if !(ch == b'`' && line[indent + marker.len()..].contains('`')) {
                fence = Some((ch, marker.len()));
                masked[line_start..nl].iter_mut().for_each(|m| *m = true);
            }
        }
        if nl == n {
            break;
        }
        line_start = nl + 1;
    }

    // Phase 2: inline code spans and HTML/XML over not-yet-masked bytes.
    let mut i = 0;
    while i < n {
        if masked[i] {
            i += 1;
            continue;
        }
        match b[i] {
            b'`' => {
                let run_end = run_end(b, i);
                match backtick_close(b, run_end, run_end - i, &masked) {
                    Some(close) => {
                        masked[i..close].iter_mut().for_each(|m| *m = true);
                        i = close;
                    }
                    None => i = run_end,
                }
            }
            b'<' => {
                let end = mask_tag_at(text, i, &mut masked);
                i = if end > i { end } else { i + 1 };
            }
            _ => i += 1,
        }
    }

    let mut out = String::with_capacity(n);
    for (idx, ch) in text.char_indices() {
        if masked[idx] && ch != '\n' {
            // Keep byte length: one space per byte of the masked char.
            out.extend(std::iter::repeat_n(' ', ch.len_utf8()));
        } else {
            out.push(ch);
        }
    }
    out
}

/// `^( {0,3})([`~]{3,})` → (indent, marker).
fn fence_open(line: &str) -> Option<(usize, &str)> {
    let indent = line.bytes().take_while(|c| *c == b' ').count();
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let ch = *rest.as_bytes().first()?;
    if ch != b'`' && ch != b'~' {
        return None;
    }
    let len = rest.bytes().take_while(|c| *c == ch).count();
    (len >= 3).then(|| (indent, &rest[..len]))
}

fn run_end(b: &[u8], i: usize) -> usize {
    let mut j = i;
    while j < b.len() && b[j] == b'`' {
        j += 1;
    }
    j
}

fn backtick_close(b: &[u8], from: usize, run_len: usize, masked: &[bool]) -> Option<usize> {
    let mut k = from;
    while k < b.len() {
        if masked[k] {
            k += 1;
            continue;
        }
        if b[k] == b'`' {
            let e = run_end(b, k);
            if e - k == run_len {
                return Some(e);
            }
            k = e;
            continue;
        }
        k += 1;
    }
    None
}

/// Tag name at `j` (`[A-Za-z][A-Za-z0-9-]*`) → end index.
fn tag_name_end(b: &[u8], j: usize) -> Option<usize> {
    if !b.get(j)?.is_ascii_alphabetic() {
        return None;
    }
    let mut k = j + 1;
    while k < b.len() && (b[k].is_ascii_alphanumeric() || b[k] == b'-') {
        k += 1;
    }
    Some(k)
}

/// Index of the `>` closing a tag whose attributes start at `j`.
fn tag_end(b: &[u8], j: usize) -> Option<usize> {
    let mut quote = 0u8;
    for (k, &ch) in b.iter().enumerate().skip(j) {
        if quote != 0 {
            if ch == quote {
                quote = 0;
            }
            continue;
        }
        match ch {
            b'"' | b'\'' => quote = ch,
            b'>' => return Some(k),
            b'<' => return None,
            _ => {}
        }
    }
    None
}

fn matching_close(b: &[u8], start: usize, name: &[u8], masked: &[bool]) -> Option<usize> {
    let mut depth = 1;
    let mut k = start;
    while k < b.len() {
        if masked[k] || b[k] != b'<' {
            k += 1;
            continue;
        }
        let mut m = k + 1;
        let closing = b.get(m) == Some(&b'/');
        if closing {
            m += 1;
        }
        let Some(name_end) = tag_name_end(b, m) else {
            k += 1;
            continue;
        };
        let Some(gt) = tag_end(b, name_end) else {
            k += 1;
            continue;
        };
        if b[m..name_end].eq_ignore_ascii_case(name) {
            if closing {
                depth -= 1;
                if depth == 0 {
                    return Some(gt + 1);
                }
            } else if b[gt - 1] != b'/' {
                depth += 1;
            }
        }
        k = gt + 1;
    }
    None
}

fn mask_tag_at(text: &str, i: usize, masked: &mut [bool]) -> usize {
    let b = text.as_bytes();
    let n = b.len();
    if text[i..].starts_with("<!--") {
        let stop = text[i + 4..].find("-->").map_or(n, |p| i + 4 + p + 3);
        masked[i..stop].iter_mut().for_each(|m| *m = true);
        return stop;
    }
    let mut j = i + 1;
    let closing = b.get(j) == Some(&b'/');
    if closing {
        j += 1;
    }
    let Some(name_end) = tag_name_end(b, j) else {
        return i;
    };
    let Some(gt) = tag_end(b, name_end) else {
        return i;
    };
    let after = gt + 1;
    let self_closing = b[gt - 1] == b'/';
    masked[i..after].iter_mut().for_each(|m| *m = true);
    if closing || self_closing {
        return after;
    }
    match matching_close(b, after, &b[j..name_end], masked) {
        Some(close) => {
            masked[after..close].iter_mut().for_each(|m| *m = true);
            close
        }
        None => after,
    }
}

/// The omp-rs effort level after `max` and its RLM / CLM hybrids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ultra {
    /// Workflow by default on every substantial task.
    Code,
    /// `ultracode` + RLM (`llm_query`, λ-RLM, SERA rubrics) for huge inputs.
    Deep,
    /// `ultracode` + CLM (`clm-host`, `rule` policy) for long sessions.
    Long,
    /// `ultracode` + RLM + CLM.
    Hyper,
}

impl Ultra {
    /// Every mode, `ultracode` first.
    pub const ALL: [Self; 4] = [Self::Code, Self::Deep, Self::Long, Self::Hyper];

    /// Display / wire name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Code => "ultracode",
            Self::Deep => "ultradeep",
            Self::Long => "ultralong",
            Self::Hyper => "hypercode",
        }
    }

    /// Parse a wire name (`ultracode`, `ultradeep`, `ultralong`, `hypercode`)
    /// or a `/ultra` argument (`deep`, `long`, `hyper`).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "ultracode" | "code" => Some(Self::Code),
            "ultradeep" | "deep" => Some(Self::Deep),
            "ultralong" | "long" => Some(Self::Long),
            "hypercode" | "hyper" => Some(Self::Hyper),
            _ => None,
        }
    }

    /// Adds the RLM toolkit to the workflow contract.
    #[must_use]
    pub const fn rlm(self) -> bool {
        matches!(self, Self::Deep | Self::Hyper)
    }

    /// Runs the `clm-host` context strategy.
    #[must_use]
    pub const fn clm(self) -> bool {
        matches!(self, Self::Long | Self::Hyper)
    }
}

/// The omp-rs "sunset" palette, top → bottom: the OMT τ logo stripes and the
/// `hypercode` gradient.
pub const SUNSET: [(u8, u8, u8); 8] = [
    (255, 224, 102),
    (255, 176, 92),
    (255, 128, 128),
    (255, 92, 168),
    (236, 72, 210),
    (190, 78, 236),
    (140, 96, 246),
    (96, 120, 250),
];

impl Ultra {
    /// Gradient stops painting the mode's name in the status line.
    #[must_use]
    pub const fn gradient(self) -> &'static [(u8, u8, u8)] {
        match self {
            Self::Code => &[(255, 224, 102), (255, 92, 168), (170, 90, 240)],
            Self::Deep => &[(170, 90, 240), (80, 120, 250)],
            Self::Long => &[(255, 92, 168), (255, 224, 102)],
            Self::Hyper => &SUNSET,
        }
    }
}

/// Linear RGB colour at `t` (clamped to `0..=1`) along evenly spaced `stops`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]
pub fn gradient_at(stops: &[(u8, u8, u8)], t: f64) -> (u8, u8, u8) {
    match stops {
        [] => (255, 255, 255),
        [only] => *only,
        _ => {
            let seg = t.clamp(0.0, 1.0) * (stops.len() - 1) as f64;
            let i = (seg.floor() as usize).min(stops.len() - 2);
            let f = seg - i as f64;
            let (a, b) = (stops[i], stops[i + 1]);
            let ch =
                |x: u8, y: u8| (f64::from(x) + (f64::from(y) - f64::from(x)) * f).round() as u8;
            (ch(a.0, b.0), ch(a.1, b.1), ch(a.2, b.2))
        }
    }
}

/// HSL → RGB; `h` in degrees (wrapped), `s`/`l` in `0..=1`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]
pub fn hsl(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    let h = h.rem_euclid(360.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match h {
        h if h < 60.0 => (c, x, 0.0),
        h if h < 120.0 => (x, c, 0.0),
        h if h < 180.0 => (0.0, c, x),
        h if h < 240.0 => (0.0, x, c),
        h if h < 300.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let ch = |v: f64| ((v + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    (ch(r), ch(g), ch(b))
}

/// Palette size of a keyword gradient (TS `stops: 14`).
pub const KEYWORD_STOPS: usize = 14;
/// Keyword gradient HSL saturation (TS default 90%).
pub const KEYWORD_SATURATION: f64 = 0.90;
/// Keyword gradient HSL lightness (TS default 62%).
pub const KEYWORD_LIGHTNESS: f64 = 0.62;

/// Colour of character `i` of an `n`-character hit of `k`
/// (TS `createGradientHighlighter` `paint`): the palette index is
/// `floor(((i / n + phase) mod 1) · 14)`, stop `j` has hue
/// `round((from + j/14 · (to − from)) mod 360)`. `phase` 0 is the static
/// (sent-bubble) palette; increasing phase shimmers.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn keyword_rgb(k: &Keyword, i: usize, n: usize, phase: f64) -> (u8, u8, u8) {
    let n = n.max(1) as f64;
    let t = (i as f64 / n + phase.rem_euclid(1.0)).rem_euclid(1.0);
    let stop = ((t * KEYWORD_STOPS as f64).floor() as usize) % KEYWORD_STOPS;
    let (from, to) = k.hue;
    let hue = (from + stop as f64 / KEYWORD_STOPS as f64 * (to - from))
        .rem_euclid(360.0)
        .round();
    hsl(hue, KEYWORD_SATURATION, KEYWORD_LIGHTNESS)
}

/// Bitmask of keywords disabled by settings (bit `i` = [`KEYWORDS`]`[i]`).
#[must_use]
pub fn disabled_mask(settings: &omp_settings::Settings) -> u32 {
    KEYWORDS
        .iter()
        .enumerate()
        .filter(|(_, k)| !enabled(settings, k.id))
        .fold(0, |mask, (i, _)| mask | (1 << i))
}

/// [`ranges`] without the keywords whose bit is set in `disabled`
/// (see [`disabled_mask`]).
#[must_use]
pub fn enabled_ranges(text: &str, disabled: u32) -> Vec<(Range<usize>, &'static Keyword)> {
    let mut out = ranges(text);
    if disabled != 0 {
        out.retain(|(_, k)| {
            KEYWORDS
                .iter()
                .position(|x| x.id == k.id)
                .is_none_or(|i| disabled & (1 << i) == 0)
        });
    }
    out
}

#[cfg(test)]
mod paint_tests {
    use super::*;

    #[test]
    fn hsl_matches_css_reference_points() {
        assert_eq!(hsl(0.0, 1.0, 0.5), (255, 0, 0));
        assert_eq!(hsl(120.0, 1.0, 0.5), (0, 255, 0));
        assert_eq!(hsl(600.0, 1.0, 0.5), (0, 0, 255));
        // hsl(0, 90%, 62%) — the first ultrathink stop.
        assert_eq!(hsl(0.0, 0.9, 0.62), (245, 71, 71));
    }

    #[test]
    fn keyword_palette_sweeps_and_phase_rotates() {
        let k = &KEYWORDS[0]; // ultrathink 0 → 330
        assert_eq!(keyword_rgb(k, 0, 10, 0.0), hsl(0.0, 0.9, 0.62));
        // i/n = 0.5 → stop 7 → hue 165.
        assert_eq!(keyword_rgb(k, 5, 10, 0.0), hsl(165.0, 0.9, 0.62));
        // Phase 0.5 moves char 0 onto the mid stop; phase 1 ≡ 0.
        assert_eq!(keyword_rgb(k, 0, 10, 0.5), keyword_rgb(k, 5, 10, 0.0));
        assert_eq!(keyword_rgb(k, 3, 10, 1.0), keyword_rgb(k, 3, 10, 0.0));
        // ultracode 410 → 270 wraps through yellow.
        let u = KEYWORDS.iter().find(|k| k.id == "ultracode").unwrap();
        assert_eq!(keyword_rgb(u, 0, 9, 0.0), hsl(50.0, 0.9, 0.62));
    }

    #[test]
    fn gradient_at_hits_stops_and_masks_filter() {
        assert_eq!(gradient_at(&SUNSET, 0.0), SUNSET[0]);
        assert_eq!(gradient_at(&SUNSET, 1.0), SUNSET[7]);
        assert_eq!(gradient_at(&SUNSET, 3.0 / 7.0), SUNSET[3]);
        assert_eq!(gradient_at(Ultra::Deep.gradient(), 0.5), (125, 105, 245));
        let text = "ultrathink and jevify";
        assert_eq!(enabled_ranges(text, 0).len(), 2);
        let only = enabled_ranges(text, 1); // bit 0 = ultrathink off
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].1.id, "jevify");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standalone_prose_only() {
        assert!(contains("please ultracode this", "ultracode"));
        assert!(contains("ultracode.", "ultracode"));
        assert!(contains("(ultracode)", "ultracode"));
        assert!(!contains("my_ultracode", "ultracode"));
        assert!(!contains("ultracode()", "ultracode"));
        assert!(!contains("ultracode.rs", "ultracode"));
        assert!(!contains("src/ultracode", "ultracode"));
        assert!(!contains("a::ultracode", "ultracode"));
        assert!(!contains("ultracode-x", "ultracode"));
        assert!(!contains("Ultracode", "ultracode"));
        assert!(!contains("run `ultracode` now", "ultracode"));
        assert!(!contains("```\nultracode\n```", "ultracode"));
        assert!(!contains("<note>ultracode</note>", "ultracode"));
        assert!(contains("```\nx\n```\nultracode", "ultracode"));
        assert!(contains("a < b ultracode", "ultracode"));
    }

    #[test]
    fn ranges_address_original_bytes() {
        let text = "ж `jevify` ultrathink и jevify";
        let r = ranges(text);
        let words: Vec<&str> = r.iter().map(|(r, _)| &text[r.clone()]).collect();
        assert_eq!(words, ["ultrathink", "jevify"]);
        assert_eq!(mask_non_prose(text).len(), text.len());
    }

    #[test]
    fn ultra_names_round_trip() {
        for u in Ultra::ALL {
            assert_eq!(Ultra::parse(u.name()), Some(u));
        }
        assert_eq!(Ultra::parse("hyper"), Some(Ultra::Hyper));
        assert!(Ultra::Hyper.rlm() && Ultra::Hyper.clm());
        assert!(!Ultra::Code.rlm() && !Ultra::Code.clm());
        assert_eq!(Ultra::parse("max"), None);
    }
}
