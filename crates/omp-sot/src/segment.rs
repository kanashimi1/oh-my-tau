//! Sentence-level segmentation of generated text into reasoning units `y_t`.
//!
//! `SoT` executes reasoning at the sentence level (App. A): the sentence is the
//! smallest unit that still carries a locally coherent reasoning intention.
//! The splitter is deliberately conservative — deterministic, Unicode-aware,
//! and biased toward *not* splitting (a merged unit is recoverable, a wrongly
//! split reasoning step is not): it breaks only after sentence-final
//! punctuation (`.`, `!`, `?`, `…`) followed by whitespace or end of input,
//! and refuses to break after abbreviation periods (`Dr.`, `e.g.`, decimal
//! numbers) and inside ellipsis runs.

/// Sentence-final punctuation that can terminate a reasoning unit.
fn is_terminal(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '\u{2026}')
}

/// Whether the text immediately before a period marks an abbreviation (so the
/// period must not terminate the unit).
fn is_abbreviation_boundary(prefix: &str) -> bool {
    let mut rev = prefix.chars().rev();
    match rev.next() {
        // Decimal number ("3.14"): a digit directly before the period.
        Some(c) if c.is_ascii_digit() => return true,
        // Lowercase initial directly after a dot ("e.g", "i.e", "p.m").
        Some(c) if c.is_lowercase() && rev.next() == Some('.') => return true,
        _ => {}
    }
    // Whole-word abbreviations: "Dr", "Mr", "vs", "etc", ...
    let word: String = prefix
        .chars()
        .rev()
        .take_while(|c| c.is_alphabetic())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    matches!(
        word.as_str(),
        "Dr" | "Mr" | "Mrs" | "Ms" | "St" | "vs" | "etc" | "approx" | "Prof" | "No"
    )
}

/// Split `text` into sentence-level reasoning units (trimmed, non-empty).
/// Empty / whitespace-only input yields no units.
#[must_use]
pub fn sentences(text: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let mut units: Vec<String> = Vec::new();
    let mut current = String::new();
    // Held sentence-final char: the split decision needs the *next* char, so
    // a terminal char waits here until the following char (or end of input)
    // resolves it.
    let mut pending: Option<(usize, char)> = None;
    let mut chars = trimmed.char_indices().peekable();

    while let Some((i, c)) = chars.next() {
        if let Some((ti, tc)) = pending.take() {
            let boundary_ok = c.is_whitespace() && chars.peek().is_none_or(|&(_, n)| n != tc);
            // A run of identical terminals ("...", "!!!") is an ellipsis /
            // emphasis run, not a sentence boundary — never split inside it.
            let mid_run = current.ends_with(tc);
            if boundary_ok && !mid_run && !is_abbreviation_boundary(&trimmed[..ti]) {
                // The terminal char belongs to the finished unit.
                current.push(tc);
                units.push(std::mem::take(&mut current));
            } else {
                current.push(tc);
                current.push(c);
            }
            continue;
        }
        if is_terminal(c) {
            if chars.peek().is_none() {
                current.push(c);
                units.push(std::mem::take(&mut current));
            } else {
                pending = Some((i, c));
            }
        } else {
            current.push(c);
        }
    }
    // Flush a trailing unit that never met a resolving boundary (no terminal
    // punctuation, or a refused split); the final trim/filter below cleans up.
    if !current.is_empty() {
        units.push(current);
    }
    units
        .into_iter()
        .map(|u| u.trim().to_owned())
        .filter(|u| !u.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::sentences;

    #[test]
    fn basic_split() {
        let units = sentences("First step. Second step! Third step?");
        assert_eq!(units, ["First step.", "Second step!", "Third step?"]);
    }

    #[test]
    fn abbreviations_do_not_split() {
        let units = sentences("Dr. Smith arrived at 3.14 meters. He left.");
        assert_eq!(units, ["Dr. Smith arrived at 3.14 meters.", "He left."]);
    }

    #[test]
    fn lowercase_initials_do_not_split() {
        let units = sentences("Use e.g. apples or pears. Then stop.");
        assert_eq!(units, ["Use e.g. apples or pears.", "Then stop."]);
    }

    #[test]
    fn ellipsis_does_not_split_mid_run() {
        let units = sentences("Wait... think again. Done.");
        assert_eq!(units, ["Wait... think again.", "Done."]);
    }

    #[test]
    fn utf8_text() {
        let units = sentences("Первый шаг. Второй шаг! Третий?");
        assert_eq!(units, ["Первый шаг.", "Второй шаг!", "Третий?"]);
    }

    #[test]
    fn no_terminal_punctuation_is_one_unit() {
        let units = sentences("just one long run with no terminator");
        assert_eq!(units, ["just one long run with no terminator"]);
    }

    #[test]
    fn empty_and_whitespace() {
        assert!(sentences("").is_empty());
        assert!(sentences("   \n\t ").is_empty());
    }

    #[test]
    fn trailing_unit_without_space() {
        let units = sentences("Step one. Step two");
        assert_eq!(units, ["Step one.", "Step two"]);
    }
}
