//! Word-local fuzzy matcher — the port of `pi-tui/src/fuzzy.ts`.
//!
//! Matching is deliberately word-local for normal words: a query like
//! "image provider" must not match a long setting description only because
//! the letters i-m-a-g-e appear in order across unrelated words.
//!
//! **Lower score = better match.** All scores mirror the TS constants so
//! ranking stays identical to omp (TS).

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use parking_lot::Mutex;

/// Result of one fuzzy comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FuzzyMatch {
    /// Whether the query matched the text.
    pub matches: bool,
    /// Match quality; lower is better.
    pub score: f64,
}

/// One word of the normalized search text.
#[derive(Debug, Clone)]
struct SearchWord {
    text: String,
    /// Char offset of the word within the normalized string.
    index: usize,
    /// Word ordinal (0-based position in the word list).
    ordinal: usize,
}

/// Precomputed search index for one text (TS `SearchIndex`).
#[derive(Debug, Clone)]
struct SearchIndex {
    normalized: String,
    compact: String,
    /// Char offsets of each word start within `compact`.
    compact_word_starts: HashSet<usize>,
    words: Vec<SearchWord>,
}

const ALPHANUMERIC_SWAP_PENALTY: f64 = 5.0;
const COMPACT_PHRASE_BONUS: f64 = 1200.0;
const PHRASE_BONUS: f64 = 1000.0;

/// Shortest needle worth scanning past its leading occurrence for
/// (TS `MIN_SHADOW_RESCAN_LENGTH`).
const MIN_SHADOW_RESCAN_LENGTH: usize = 3;

const INDEX_CACHE_MAX: usize = 4096;
const MAX_CACHED_TEXT_LEN: usize = 4096;

/// Module-level memo of per-text search indexes (TS `indexCache`).
/// Admission stops at the cap instead of evicting.
static INDEX_CACHE: LazyLock<Mutex<HashMap<String, SearchIndex>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Normalize camelCase/ACRONYM splits and non-alphanumerics to words
/// (TS `normalizeForSearch`).
fn normalize_for_search(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    let mut out = String::with_capacity(value.len() + 8);
    for (i, &c) in chars.iter().enumerate() {
        if i > 0 && c.is_uppercase() {
            let prev = chars[i - 1];
            let splits_acronym_tail =
                prev.is_uppercase() && chars.get(i + 1).is_some_and(|c: &char| c.is_lowercase());
            if prev.is_lowercase() || prev.is_ascii_digit() || splits_acronym_tail {
                out.push(' ');
            }
        }
        out.extend(c.to_lowercase());
    }
    // Non Letter/Mark/Number runs collapse to a single space; trim.
    let mut cleaned = String::with_capacity(out.len());
    let mut pending_space = false;
    for c in out.chars() {
        if c.is_alphanumeric() {
            if pending_space && !cleaned.is_empty() {
                cleaned.push(' ');
            }
            pending_space = false;
            cleaned.push(c);
        } else {
            pending_space = true;
        }
    }
    cleaned
}

/// Build (or fetch) the search index for `text` (TS `buildSearchIndex`).
fn build_search_index(text: &str) -> SearchIndex {
    if text.chars().count() > MAX_CACHED_TEXT_LEN {
        return build_uncached_search_index(text);
    }
    let mut cache = INDEX_CACHE.lock();
    if let Some(cached) = cache.get(text) {
        return cached.clone();
    }
    let result = build_uncached_search_index(text);
    if cache.len() < INDEX_CACHE_MAX {
        cache.insert(text.to_string(), result.clone());
    }
    result
}

/// Uncached index construction (TS `buildUncachedSearchIndex`).
fn build_uncached_search_index(text: &str) -> SearchIndex {
    let normalized = normalize_for_search(text);
    if normalized.is_empty() {
        return SearchIndex {
            normalized,
            compact: String::new(),
            compact_word_starts: HashSet::new(),
            words: Vec::new(),
        };
    }
    let mut words = Vec::new();
    let mut compact_word_starts = HashSet::new();
    let mut index = 0usize;
    let mut compact_index = 0usize;
    for (ordinal, word) in normalized.split(' ').enumerate() {
        let len = word.chars().count();
        words.push(SearchWord {
            text: word.to_string(),
            index,
            ordinal,
        });
        compact_word_starts.insert(compact_index);
        index += len + 1;
        compact_index += len;
    }
    SearchIndex {
        compact: normalized.chars().filter(|&c| c != ' ').collect(),
        normalized,
        compact_word_starts,
        words,
    }
}

/// Character-level subsequence score within one word (TS `scoreCharacters`).
struct CharacterMatch {
    matches: bool,
    score: f64,
    span: usize,
}

fn score_characters(query_lower: &[char], text_lower: &[char]) -> CharacterMatch {
    if query_lower.is_empty() {
        return CharacterMatch {
            matches: true,
            score: 0.0,
            span: 0,
        };
    }
    if query_lower.len() > text_lower.len() {
        return CharacterMatch {
            matches: false,
            score: 0.0,
            span: 0,
        };
    }

    let mut query_index = 0usize;
    let mut score = 0.0f64;
    let mut first_match_index: isize = -1;
    let mut last_match_index: isize = -1;
    let mut consecutive_matches = 0usize;

    for (i, &c) in text_lower.iter().enumerate() {
        if query_index >= query_lower.len() {
            break;
        }
        if c == query_lower[query_index] {
            if first_match_index < 0 {
                first_match_index = i as isize;
            }
            if last_match_index == i as isize - 1 {
                consecutive_matches += 1;
                score -= (consecutive_matches * 5) as f64;
            } else {
                consecutive_matches = 0;
                if last_match_index >= 0 {
                    score += (i as isize - last_match_index - 1) as f64 * 2.0;
                }
            }
            score += i as f64 * 0.1;
            last_match_index = i as isize;
            query_index += 1;
        }
    }

    if query_index < query_lower.len() {
        return CharacterMatch {
            matches: false,
            score: 0.0,
            span: 0,
        };
    }

    CharacterMatch {
        matches: true,
        score,
        span: (last_match_index - first_match_index + 1) as usize,
    }
}

/// Single-swap alpha<->digit query variants (TS `buildAlphanumericSwapQueries`).
fn build_alphanumeric_swap_queries(query_lower: &[char]) -> Vec<String> {
    let mut variants: Vec<String> = Vec::new();
    for i in 0..query_lower.len().saturating_sub(1) {
        let current = query_lower[i];
        let next = query_lower[i + 1];
        let is_swap = (current.is_ascii_lowercase() && next.is_ascii_digit())
            || (current.is_ascii_digit() && next.is_ascii_lowercase());
        if !is_swap {
            continue;
        }
        let mut swapped: Vec<char> = query_lower.to_vec();
        swapped.swap(i, i + 1);
        let joined: String = swapped.into_iter().collect();
        if !variants.contains(&joined) {
            variants.push(joined);
        }
    }
    variants
}

fn with_position(score: f64, index: usize) -> f64 {
    score + index as f64 * 0.01
}

/// Whether `phrase` at char-offset `index` of `normalized` spans whole words.
fn is_word_boundary_phrase(normalized: &[char], index: usize, length: usize) -> bool {
    let before = index == 0 || normalized[index - 1] == ' ';
    let after_index = index + length;
    let after = after_index == normalized.len() || normalized[after_index] == ' ';
    before && after
}

fn find_subsequence(haystack: &[char], needle: &[char], from: usize) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() || from > haystack.len() {
        return None;
    }
    'outer: for start in from..=haystack.len() - needle.len() {
        for (k, &n) in needle.iter().enumerate() {
            if haystack[start + k] != n {
                continue 'outer;
            }
        }
        return Some(start);
    }
    None
}

/// Offset of the first whole-word occurrence of `phrase` in `normalized`, or
/// None (TS `findWordBoundaryPhrase` — rescans past shadowed hits only for
/// needles >= MIN_SHADOW_RESCAN_LENGTH).
fn find_word_boundary_phrase(normalized: &[char], phrase: &[char]) -> Option<usize> {
    if phrase.is_empty() {
        return None;
    }
    let first = find_subsequence(normalized, phrase, 0)?;
    if is_word_boundary_phrase(normalized, first, phrase.len()) {
        return Some(first);
    }
    if first == 0 || normalized[first - 1] == ' ' {
        return None;
    }
    if phrase.len() < MIN_SHADOW_RESCAN_LENGTH {
        return None;
    }
    let mut at = first + 1;
    while let Some(hit) = find_subsequence(normalized, phrase, at) {
        if is_word_boundary_phrase(normalized, hit, phrase.len()) {
            return Some(hit);
        }
        at = hit + 1;
    }
    None
}

/// Offset of the first `needle` occurrence starting a word in `index.compact`,
/// or None (TS `findCompactWordStart`).
fn find_compact_word_start(index: &SearchIndex, needle: &[char]) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    let compact: Vec<char> = index.compact.chars().collect();
    let first = find_subsequence(&compact, needle, 0)?;
    if index.compact_word_starts.contains(&first) {
        return Some(first);
    }
    if needle.len() < MIN_SHADOW_RESCAN_LENGTH {
        return None;
    }
    let mut at = first + 1;
    while let Some(hit) = find_subsequence(&compact, needle, at) {
        if index.compact_word_starts.contains(&hit) {
            return Some(hit);
        }
        at = hit + 1;
    }
    None
}

/// Score one query token against one word (TS `scoreTokenAgainstWord`).
fn score_token_against_word(token: &[char], word: &SearchWord) -> Option<FuzzyMatch> {
    let word_chars: Vec<char> = word.text.chars().collect();
    if word_chars == token {
        return Some(FuzzyMatch {
            matches: true,
            score: with_position(-200.0, word.index),
        });
    }
    if starts_with(&word_chars, token) {
        return Some(FuzzyMatch {
            matches: true,
            score: with_position(
                -170.0 + (word_chars.len() - token.len()) as f64 * 0.5,
                word.index,
            ),
        });
    }
    if starts_with(token, &word_chars) && token.len() - word_chars.len() <= 2 {
        return Some(FuzzyMatch {
            matches: true,
            score: with_position(-150.0 + (token.len() - word_chars.len()) as f64, word.index),
        });
    }
    if let Some(substring_index) = find_subsequence(&word_chars, token, 0) {
        return Some(FuzzyMatch {
            matches: true,
            score: with_position(-20.0 + substring_index as f64, word.index),
        });
    }
    let character_match = score_characters(token, &word_chars);
    if !character_match.matches {
        return None;
    }
    let max_span = usize::max(token.len() + 2, (token.len() as f64 * 1.8).ceil() as usize);
    if character_match.span > max_span {
        return None;
    }
    Some(FuzzyMatch {
        matches: true,
        score: with_position(-40.0 + character_match.score, word.index),
    })
}

fn starts_with(haystack: &[char], prefix: &[char]) -> bool {
    haystack.len() >= prefix.len() && haystack.iter().zip(prefix).all(|(a, b)| a == b)
}

/// Acronym match: token is the leading letters of consecutive words
/// (TS `scoreAcronym`).
fn score_acronym(token: &[char], index: &SearchIndex) -> Option<FuzzyMatch> {
    if !(2..=4).contains(&token.len()) || index.words.is_empty() {
        return None;
    }
    let mut query_index = 0usize;
    let mut first_ordinal: isize = -1;
    let mut last_ordinal: isize = -1;
    let mut first_text_index = 0usize;
    for word in &index.words {
        if word.text.chars().next() != token.get(query_index).copied() {
            continue;
        }
        if first_ordinal < 0 {
            first_ordinal = word.ordinal as isize;
            first_text_index = word.index;
        }
        last_ordinal = word.ordinal as isize;
        query_index += 1;
        if query_index == token.len() {
            break;
        }
    }
    if query_index < token.len() || first_ordinal < 0 || last_ordinal < 0 {
        return None;
    }
    let word_span = (last_ordinal - first_ordinal + 1) as f64;
    if word_span > token.len() as f64 + 2.0 {
        return None;
    }
    Some(FuzzyMatch {
        matches: true,
        score: with_position(
            -30.0 + word_span * 4.0 - token.len() as f64 * 2.0,
            first_text_index,
        ),
    })
}

fn score_token_direct(token: &[char], index: &SearchIndex) -> FuzzyMatch {
    if token.is_empty() {
        return FuzzyMatch {
            matches: true,
            score: 0.0,
        };
    }
    let mut best: Option<FuzzyMatch> = None;
    if let Some(compact_index) = find_compact_word_start(index, token) {
        best = Some(FuzzyMatch {
            matches: true,
            score: with_position(-140.0, compact_index),
        });
    }
    for word in &index.words {
        if let Some(m) = score_token_against_word(token, word) {
            if best.is_none_or(|b| m.score < b.score) {
                best = Some(m);
            }
        }
    }
    if let Some(acronym) = score_acronym(token, index) {
        if best.is_none_or(|b| acronym.score < b.score) {
            best = Some(acronym);
        }
    }
    best.unwrap_or(FuzzyMatch {
        matches: false,
        score: 0.0,
    })
}

/// Score one token, allowing a single alpha<->digit swap (TS `scoreToken`).
fn score_token(token: &[char], index: &SearchIndex) -> FuzzyMatch {
    let mut best = score_token_direct(token, index);
    if best.matches {
        return best;
    }
    for variant in build_alphanumeric_swap_queries(token) {
        let variant_chars: Vec<char> = variant.chars().collect();
        let m = score_token_direct(&variant_chars, index);
        if !m.matches {
            continue;
        }
        let score = m.score + ALPHANUMERIC_SWAP_PENALTY;
        if !best.matches || score < best.score {
            best = FuzzyMatch {
                matches: true,
                score,
            };
        }
    }
    best
}

/// A query normalized and split once (TS `PreparedQuery`).
#[derive(Debug, Clone)]
struct PreparedQuery {
    normalized: Vec<char>,
    tokens: Vec<Vec<char>>,
    compact: Vec<char>,
}

fn prepare_query(query: &str) -> Option<PreparedQuery> {
    let normalized = normalize_for_search(query);
    if normalized.is_empty() {
        return None;
    }
    let tokens: Vec<Vec<char>> = normalized.split(' ').map(|t| t.chars().collect()).collect();
    let compact: Vec<char> = normalized.chars().filter(|&c| c != ' ').collect();
    Some(PreparedQuery {
        normalized: normalized.chars().collect(),
        tokens,
        compact,
    })
}

fn fuzzy_match_core(pq: Option<&PreparedQuery>, index: &SearchIndex) -> FuzzyMatch {
    let Some(pq) = pq else {
        return FuzzyMatch {
            matches: true,
            score: 0.0,
        };
    };
    if index.words.is_empty() {
        return FuzzyMatch {
            matches: false,
            score: 0.0,
        };
    }
    let mut total_score = 0.0f64;
    let normalized_chars: Vec<char> = index.normalized.chars().collect();
    if let Some(phrase_index) = find_word_boundary_phrase(&normalized_chars, &pq.normalized) {
        total_score -= PHRASE_BONUS;
        total_score += phrase_index as f64 * 0.01;
    }
    if let Some(compact_index) = find_compact_word_start(index, &pq.compact) {
        total_score -= COMPACT_PHRASE_BONUS;
        total_score += compact_index as f64 * 0.01;
    }
    for token in &pq.tokens {
        let m = score_token(token, index);
        if !m.matches {
            return FuzzyMatch {
                matches: false,
                score: 0.0,
            };
        }
        total_score += m.score;
    }
    FuzzyMatch {
        matches: true,
        score: total_score,
    }
}

/// Match `query` against `text` (TS `fuzzyMatch`). Lower score is better.
#[must_use]
pub fn fuzzy_match(query: &str, text: &str) -> FuzzyMatch {
    let pq = prepare_query(query);
    fuzzy_match_core(pq.as_ref(), &build_search_index(text))
}

/// A text prepared once for repeated fuzzy matching (TS `FuzzyText`).
///
/// The search index lives exactly as long as the caller's reference, avoiding
/// the module cache's length cap for long corpora.
#[derive(Debug, Clone)]
pub struct FuzzyText {
    index: SearchIndex,
}

impl FuzzyText {
    /// Prepare `text` for repeated matching.
    #[must_use]
    pub fn new(text: &str) -> Self {
        Self {
            index: build_uncached_search_index(text),
        }
    }

    /// Match `query` (space-separated tokens; all must match).
    #[must_use]
    pub fn r#match(&self, query: &str) -> FuzzyMatch {
        fuzzy_match_core(prepare_query(query).as_ref(), &self.index)
    }
}

/// One ranked item (TS `FuzzyFilterResult`).
#[derive(Debug, Clone, PartialEq)]
pub struct FuzzyRankResult<T> {
    /// The matched item.
    pub item: T,
    /// Match quality; lower is better.
    pub score: f64,
}

/// Filter and sort items by fuzzy match quality, best first (TS `fuzzyRank`).
/// Supports space-separated tokens: all tokens must match.
#[must_use]
pub fn fuzzy_rank<T: Clone>(
    items: &[T],
    query: &str,
    get_text: impl Fn(&T) -> String,
) -> Vec<FuzzyRankResult<T>> {
    let pq = prepare_query(query);
    let mut results: Vec<FuzzyRankResult<T>> = Vec::new();
    for item in items {
        let text = get_text(item);
        // A non-blank query that normalizes to empty matches everything.
        let m = match &pq {
            None => FuzzyMatch {
                matches: true,
                score: 0.0,
            },
            Some(pq) => fuzzy_match_core(Some(pq), &build_search_index(&text)),
        };
        if m.matches {
            results.push(FuzzyRankResult {
                item: item.clone(),
                score: m.score,
            });
        }
    }
    results.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// Filter items by fuzzy match, best matches first (TS `fuzzyFilter`).
#[must_use]
pub fn fuzzy_filter<T: Clone>(items: &[T], query: &str, get_text: impl Fn(&T) -> String) -> Vec<T> {
    fuzzy_rank(items, query, get_text)
        .into_iter()
        .map(|r| r.item)
        .collect()
}

/// Clear the fuzzy search-index cache (TS `resetFuzzyIndexCache`). Intended
/// for tests/benchmarks.
pub fn reset_fuzzy_index_cache() {
    INDEX_CACHE.lock().clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches(query: &str, text: &str) -> bool {
        fuzzy_match(query, text).matches
    }

    #[test]
    fn empty_query_matches_everything() {
        assert!(matches("", "anything"));
        assert_eq!(fuzzy_match("", "anything").score, 0.0);
    }

    #[test]
    fn word_local_matching() {
        // "image provider" must not match across unrelated words.
        assert!(!matches("image provider", "experimental imbroglio"));
        assert!(matches("image provider", "the image provider setting"));
    }

    #[test]
    fn camel_case_is_split() {
        assert!(matches("provider", "ImageProviderConfig"));
        assert!(matches("ipc", "ImageProviderConfig")); // acronym
    }

    #[test]
    fn prefix_beats_substring_across_words() {
        let a = fuzzy_match("comp", "compact disk");
        let b = fuzzy_match("comp", "recompose");
        assert!(a.matches && b.matches);
        assert!(a.score < b.score);
    }

    #[test]
    fn acronym_scoring() {
        let m = fuzzy_match("img", "image magic glyph");
        assert!(m.matches);
        // worse than a direct word match
        assert!(m.score > -200.0);
    }

    #[test]
    fn all_tokens_must_match() {
        assert!(matches("foo bar", "foo the bar"));
        assert!(!matches("foo baz", "foo the bar"));
    }

    #[test]
    fn rank_sorts_best_first() {
        let items = vec!["reimage thing", "image provider", "imaginary"];
        let ranked = fuzzy_rank(&items, "image", |s| s.to_string());
        assert_eq!(ranked[0].item, "image provider");
    }

    #[test]
    fn fuzzy_text_reuses_index() {
        let ft = FuzzyText::new("ImageProviderConfig");
        assert!(ft.r#match("provider").matches);
        assert!(ft.r#match("ipc").matches);
    }

    #[test]
    fn phrase_shadowing_rescans() {
        // "image" shadowed inside "reimage", whole-word hit later.
        let m = fuzzy_match("image", "reimage image provider");
        assert!(m.matches);
        assert!(m.score <= -1000.0 + 12.0); // phrase bonus minus position offset
    }
}
