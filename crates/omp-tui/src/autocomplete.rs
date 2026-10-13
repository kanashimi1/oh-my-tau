//! Slash-command and @-mention autocomplete — the port of
//! `pi-tui/src/autocomplete.ts` (+ `fuzzy.ts` for ranking).
//!
//! Stub. Planned surface: an [`Autocomplete`] widget owning a candidate
//! list (slash commands from `available_commands_update` frames, file paths
//! for `@`, model ids for `/model`), fuzzy-ranked with the same scoring as
//! TS `fuzzy.ts`, rendered as a popup above the editor, feeding accepted
//! text back into the editor buffer.

use crate::theme::Theme;

/// Ranking score for one candidate (higher is better). Port of
/// `fuzzy.ts`'s scoring until the real algorithm lands.
#[must_use]
pub fn fuzzy_score(needle: &str, haystack: &str) -> i64 {
    if needle.is_empty() {
        return 0;
    }
    let n = needle.to_lowercase();
    let h = haystack.to_lowercase();
    if h.contains(&n) {
        100 - i64::try_from(h.len()).unwrap_or(100)
    } else {
        -1
    }
}

/// Autocomplete popup state. Placeholder until the autocomplete
/// implementer lands.
#[derive(Debug, Clone, Default)]
pub struct Autocomplete {
    /// Whether the popup is open.
    pub open: bool,
    /// Raw candidates (display strings).
    pub candidates: Vec<String>,
    /// Selected row.
    pub selected: usize,
    /// Theme snapshot for rendering (unused in the stub).
    pub theme: Option<Theme>,
}

impl Autocomplete {
    /// Score all candidates against `query`, best first.
    #[must_use]
    pub fn ranked(&self, query: &str) -> Vec<(usize, i64)> {
        let mut rows: Vec<(usize, i64)> = self
            .candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, fuzzy_score(query, c)))
            .filter(|(_, s)| *s >= 0)
            .collect();
        rows.sort_by_key(|&(_, s)| std::cmp::Reverse(s));
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_prefers_exact_substring() {
        let a = Autocomplete {
            candidates: vec!["/model".into(), "/mode-diff".into(), "/compact".into()],
            ..Default::default()
        };
        let ranked = a.ranked("model");
        assert_eq!(ranked.first().map(|(i, _)| i), Some(&0));
    }
}
