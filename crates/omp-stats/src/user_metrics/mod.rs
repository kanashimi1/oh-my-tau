//! Port of `stats/src/user-metrics.ts`: behavioral metrics extracted from a
//! single user message. Pure, side-effect free, designed for batch use during
//! session ingestion and standalone testing.
//!
//! JS `String.length` counts UTF-16 code units, so every length comparison
//! ([`compute_user_message_metrics`]'s `chars`, the [`PROSE_MAX_CHARS`] cap)
//! goes through `encode_utf16().count()`. The `regex` crate cannot express
//! look-behind or back-references, so the four patterns that need them
//! (`XML_TAG_PAIR_RE`, `DOTTED_TOKEN_RE`, `SAD_EMOTICON_RE`,
//! `BLAME_STOP_RE`) live in the sibling scanner modules and are applied in the
//! same order as the TS `replace` chain.

mod blame_stop;
mod dotted;
mod patterns;
mod scanners;
mod tables;

#[cfg(test)]
mod tests;

use regex::Regex;
use serde::{Deserialize, Serialize};

use blame_stop::{count_blame_stop, count_sad_emoticons};
use dotted::replace_dotted_tokens;
use patterns::*;
use scanners::{has_tripled_upper, is_js_space, js_trim, replace_xml_tag_pairs};

/// Behavioral metrics extracted from a single user message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessageMetrics {
    /// Total characters of analyzed text.
    pub chars: i64,
    /// Whitespace-delimited word count.
    pub words: i64,
    /// Sentences where more than half of the alphabetic characters are
    /// uppercase, with enough letters to make the ratio meaningful (short
    /// acronyms like "OK" don't count) and either two uppercase runs
    /// ("WHAT THE HELL") or one elongated run ("CMOOON") so a lone
    /// acronym / env var ("use JSON", "HOME=/tmp") never registers.
    pub yelling: i64,
    /// Profanity hits (word-boundary, case-insensitive).
    pub profanity: i64,
    /// Catch-all "obviously upset" signal computed on a *prose-only* body
    /// (code fences, XML/HTML tags, URLs, file mentions and quoted lines are
    /// stripped first; >= 3 remaining lines score zero because a formatted
    /// prompt isn't a tantrum). Sum of drama runs, elongated interjections,
    /// standalone `dude` and sad emoticons.
    pub anguish: i64,
    /// Corrective negation: the user is telling us we got it wrong.
    pub negation: i64,
    /// The user is repeating themselves.
    pub repetition: i64,
    /// Direct second-person reproach pinned on the agent.
    pub blame: i64,
}

/// Empty metrics constant for callers that need a default.
pub const EMPTY_USER_METRICS: UserMessageMetrics = UserMessageMetrics {
    chars: 0,
    words: 0,
    yelling: 0,
    profanity: 0,
    anguish: 0,
    negation: 0,
    repetition: 0,
    blame: 0,
};

/// Cap on the judge-facing prose stored per user message; longer prose is truncated.
pub const PROSE_MAX_CHARS: usize = 4000;

/// Sentences need this many letters before the caps ratio means anything.
const YELLING_MIN_LETTERS: i64 = 4;
/// Share of uppercase letters that must be exceeded.
const YELLING_THRESHOLD: f64 = 0.5;
/// Users don't really get angry with super detailed and formatted prompts - if
/// the remaining prose is this many lines or more, score zero.
const MAX_PROSE_LINES: i64 = 3;

/// JS `String.length`: UTF-16 code units, not Unicode scalar values.
fn js_len(text: &str) -> i64 {
    text.encode_utf16().count() as i64
}

/// Count regex hits without materializing the match array.
fn count_matches(text: &str, re: &Regex) -> i64 {
    re.find_iter(text).count() as i64
}

/// A sentence needs 2+ uppercase runs ("WHAT THE HELL"), or a single elongated
/// run with a tripled letter ("CMOOON"), before its caps ratio can count as
/// yelling. A lone acronym / identifier / env var never does.
fn is_shouted_sentence(sentence: &str) -> bool {
    let runs: Vec<&str> = UPPER_RUN_RE
        .find_iter(sentence)
        .map(|m| m.as_str())
        .collect();
    match runs.first() {
        None => false,
        Some(_) if runs.len() >= 2 => true,
        Some(first) => js_len(first) >= YELLING_MIN_LETTERS && has_tripled_upper(first),
    }
}

/// Count sentences whose share of uppercase letters exceeds the threshold.
/// Sentences below the letter floor are ignored, so "OK" / "WIP" / "TODO"
/// don't register.
fn count_yelling_sentences(text: &str) -> i64 {
    let mut count = 0;
    for sentence in SENTENCE_RE.find_iter(text) {
        let sentence = sentence.as_str();
        let letters = count_matches(sentence, &LETTER_RE);
        if letters >= YELLING_MIN_LETTERS {
            let upper = count_matches(sentence, &UPPER_LETTER_RE);
            if (upper as f64 / letters as f64) > YELLING_THRESHOLD && is_shouted_sentence(sentence)
            {
                count += 1;
            }
        }
    }
    count
}

/// Strip structured content so that pasted code, harness wrappers, file
/// mentions and quoted blocks don't dilute or fake behavior signals. Fenced
/// code and tag pairs collapse to a newline so line counting reflects what was
/// removed instead of merging neighbors; the rest become spaces.
fn strip_structured_content(text: &str) -> String {
    let out = FENCED_CODE_RE.replace_all(text, "\n");
    let out = replace_xml_tag_pairs(&out);
    let out = XML_TAG_BARE_RE.replace_all(&out, " ");
    let out = INLINE_CODE_RE.replace_all(&out, " ");
    let out = URL_RE.replace_all(&out, " ");
    let out = FILE_MENTION_RE.replace_all(&out, "${1} ");
    let out = replace_dotted_tokens(&out);
    let out = QUOTE_LINE_RE.replace_all(&out, "");
    let out = IMAGE_MARKER_RE.replace_all(&out, " ");
    ANSI_ESCAPE_RE.replace_all(&out, "").into_owned()
}

/// The prose a user message contributes to frustration judging: the same
/// structured-content strip the regex signals score on, trimmed and capped at
/// [`PROSE_MAX_CHARS`]. Empty when the message was nothing but markup.
#[must_use]
pub fn judge_prose(text: &str) -> String {
    let prose = js_trim(&strip_structured_content(&js_trim(text)));
    truncate_utf16(&prose, PROSE_MAX_CHARS)
}

/// `String.prototype.slice(0, max)` over UTF-16 code units. A surrogate pair
/// straddling the limit is dropped whole - `&str` cannot hold a lone surrogate,
/// and splitting one would corrupt the prose anyway.
fn truncate_utf16(text: &str, max: usize) -> String {
    let mut used = 0usize;
    let mut end = text.len();
    for (idx, c) in text.char_indices() {
        let width = c.len_utf16();
        if used + width > max {
            end = idx;
            break;
        }
        used += width;
    }
    text[..end].to_string()
}

fn count_non_empty_lines(text: &str) -> i64 {
    text.split('\n')
        .filter(|line| line.chars().any(|c| !is_js_space(c)))
        .count() as i64
}

/// Compute behavioral metrics for a user message.
///
/// `text` may be empty or whitespace; in that case every metric is 0.
#[must_use]
pub fn compute_user_message_metrics(text: &str) -> UserMessageMetrics {
    let trimmed = js_trim(text);
    if trimmed.is_empty() {
        return EMPTY_USER_METRICS;
    }
    let chars = js_len(&trimmed);
    let words = count_matches(&trimmed, &WORD_RE);

    // Behavior signals are computed on a stripped prose body; long /
    // well-formatted messages score zero because they are deliberate, not
    // emotional outbursts.
    let prose = js_trim(&strip_structured_content(&trimmed));
    if prose.is_empty() || count_non_empty_lines(&prose) >= MAX_PROSE_LINES {
        return UserMessageMetrics {
            chars,
            words,
            ..EMPTY_USER_METRICS
        };
    }

    let anguish = count_matches(&prose, &DRAMA_RE)
        + count_matches(&prose, &ANGUISH_RE)
        + count_matches(&prose, &DUDE_RE)
        + count_sad_emoticons(&prose) as i64;
    let negation = i64::from(NEGATION_LEAD_RE.is_match(&prose).unwrap_or(false))
        + count_matches(&prose, &NEGATION_PHRASE_RE);
    let repetition =
        count_matches(&prose, &REPETITION_RECALL_RE) + count_matches(&prose, &REPETITION_STILL_RE);
    let blame = count_matches(&prose, &BLAME_YOU_RE)
        + count_matches(&prose, &BLAME_WHY_RE)
        + count_blame_stop(&prose) as i64;

    UserMessageMetrics {
        chars,
        words,
        yelling: count_yelling_sentences(&prose),
        profanity: count_matches(&prose, &PROFANITY_RE),
        anguish,
        negation,
        repetition,
        blame,
    }
}
