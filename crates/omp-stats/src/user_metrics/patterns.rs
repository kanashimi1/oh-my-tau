//! Compiled regexes, ported 1:1 from the TS constants of the same name.
//!
//! JS `\b` / `\w` / `\d` are ASCII-only even with the `u` flag, while the
//! `regex` crate's default Unicode mode makes them Unicode-aware. Every
//! pattern leaning on those carries the inline `-u` flag so boundaries and
//! classes match JS exactly; patterns using `\p{L}` / `\p{Lu}` stay
//! Unicode-aware (JS does too with `u`).

use regex::Regex;
use std::sync::LazyLock;

use super::tables::{anguish_pattern, profanity_pattern};

fn compile(flags: &str, body: &str) -> Regex {
    // A global `(?-u)` makes `[^>]`-style classes able to match invalid UTF-8,
    // which a `&str` Regex rejects. So `-u` is applied only to the JS
    // ASCII-only escapes: `\b` -> `(?-u:\b)`, `\w`/`\W`/`\d`/`\D` -> ASCII classes.
    let (flags, body) = if flags.contains("-u") {
        (flags.replace("-u", ""), ascii_escapes(body))
    } else {
        (flags.to_string(), body.to_string())
    };
    let pat = if flags.is_empty() {
        body
    } else {
        format!("(?{flags}){body}")
    };
    Regex::new(&pat).unwrap_or_else(|e| panic!("user_metrics regex {pat:?}: {e}"))
}

/// Rewrite JS ASCII-only escapes for a Unicode-mode `regex`.
fn ascii_escapes(body: &str) -> String {
    let mut out = String::with_capacity(body.len() + 16);
    let mut chars = body.chars().peekable();
    let mut in_class = false;
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                let Some(n) = chars.next() else {
                    out.push(c);
                    break;
                };
                let rep = match (n, in_class) {
                    ('b', false) => "(?-u:\\b)",
                    ('B', false) => "(?-u:\\B)",
                    ('w', true) => "0-9A-Za-z_",
                    ('w', false) => "[0-9A-Za-z_]",
                    ('W', false) => "[^0-9A-Za-z_]",
                    ('d', true) => "0-9",
                    ('d', false) => "[0-9]",
                    ('D', false) => "[^0-9]",
                    _ => {
                        out.push('\\');
                        out.push(n);
                        continue;
                    }
                };
                out.push_str(rep);
            }
            '[' if !in_class => {
                in_class = true;
                out.push(c);
                if chars.peek() == Some(&'^') {
                    out.push(chars.next().unwrap_or('^'));
                }
                if chars.peek() == Some(&']') {
                    out.push(chars.next().unwrap_or(']'));
                }
            }
            ']' if in_class => {
                in_class = false;
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

pub(super) static SENTENCE_RE: LazyLock<Regex> = LazyLock::new(|| compile("", r"[^.!?\n]+"));
pub(super) static LETTER_RE: LazyLock<Regex> = LazyLock::new(|| compile("", r"\p{L}"));
pub(super) static UPPER_LETTER_RE: LazyLock<Regex> = LazyLock::new(|| compile("", r"\p{Lu}"));
pub(super) static UPPER_RUN_RE: LazyLock<Regex> = LazyLock::new(|| compile("", r"\p{Lu}{2,}"));
/// Runs starting with `!` or `?` followed by 2+ of `!?1`. The `1` is the
/// classic shift-key mishit ("!!!111" / "!?!??111") so those count as one burst.
pub(super) static DRAMA_RE: LazyLock<Regex> = LazyLock::new(|| compile("", r"[!?][!?1]{2,}"));
pub(super) static WORD_RE: LazyLock<Regex> = LazyLock::new(|| compile("", r"\S+"));
pub(super) static DUDE_RE: LazyLock<Regex> = LazyLock::new(|| compile("i-u", r"\bdude\b"));
pub(super) static PROFANITY_RE: LazyLock<Regex> =
    LazyLock::new(|| compile("i-u", &format!(r"\b(?:{})\b", profanity_pattern())));
pub(super) static ANGUISH_RE: LazyLock<Regex> =
    LazyLock::new(|| compile("i-u", &format!(r"\b(?:{})\b", anguish_pattern())));

/// Corrective negation, anchored at the very start of the trimmed prose body
/// (deliberately no `m` flag). Bare `no` only counts as an interjection -
/// punctuation, end-of-text or a discourse word must follow - so determiner
/// uses ("no extensions to the page") don't score. A hyphen separates only
/// when it doesn't glue a compound (`no - not that` yes, `no-op` no). The
/// bare-`no` branch has no trailing `\b`, so emphatic misspellings (`nou ...`)
/// count too.
const NEGATION_LEAD: &str = concat!(
    r"^[ \t]*(?:(?:nope|nah|nvm|wrong|incorrect)\b",
    r"|no(?=\s*(?:[,.!?;:\u{2013}\u{2014}]|-(?!\w)|$|",
    r"(?:i|im|u|you|ur|we|it|its|that|thats|this|the|they|theyre|he|she|man|dude|bro|",
    r"wait|dont|not|stop|just|again|please|plz|but|actually|literally|seriously|sorry|no|",
    r"never|nothing|wtf|why|what|wrong)\b)))"
);
const NEGATION_PHRASE: &str = concat!(
    r"\b(?:that['\u{2019}]?s\s+not\s+(?:what|right|it)",
    r"|not\s+what\s+i\s+(?:meant|asked|said|wanted)",
    r"|makes\s+(?:no|zero)\s+sense)\b"
);
/// Recall verbs, with the optional `like ` / `as ` prefix so "like i said"
/// doesn't double-count bare "i said". Bare `i asked` is too noisy (it is
/// overwhelmingly "i asked <some third party>"), so that variant requires
/// `i asked you`.
const REPETITION_RECALL: &str = concat!(
    r"\b(?:(?:like|as)\s+i\s+(?:said|told\s+you|asked)",
    r"|i\s+(?:meant|said|told\s+you|asked\s+you|already\s+(?:said|told|did|asked|wrote)))\b"
);
/// Bare `still` is ambiguous, so it counts only with a negative or sameness marker.
const REPETITION_STILL: &str = concat!(
    r"\bstill\s+(?:doesn['\u{2019}]?t|doesnt|isn['\u{2019}]?t|isnt|not|broken|wrong|fails|",
    r"failing|the\s+same|same)\b"
);
/// `you` alone is too generic, so it is anchored to accusatory verbs.
const BLAME_YOU: &str = r"\byou\s+(?:didn['\u{2019}]?t|did\s+not|broke|missed|forgot|keep|always|never|still|ignored)\b";
/// Only the past/conditional forms read as reproach; bare `why you` /
/// `why are you` are dominated by neutral how-does-this-work questions.
const BLAME_WHY: &str = r"\bwhy\s+(?:would|did)\s+(?:you|u)\b";

/// The only pattern with a look-ahead, which `regex` lacks: `fancy_regex`.
/// (JS `\b` is ASCII-only; the words here are ASCII, so Unicode `\b` agrees.)
pub(super) static NEGATION_LEAD_RE: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
    let pat = format!("(?i){NEGATION_LEAD}");
    fancy_regex::Regex::new(&pat).unwrap_or_else(|e| panic!("user_metrics regex {pat:?}: {e}"))
});
pub(super) static NEGATION_PHRASE_RE: LazyLock<Regex> =
    LazyLock::new(|| compile("i-u", NEGATION_PHRASE));
pub(super) static REPETITION_RECALL_RE: LazyLock<Regex> =
    LazyLock::new(|| compile("i-u", REPETITION_RECALL));
pub(super) static REPETITION_STILL_RE: LazyLock<Regex> =
    LazyLock::new(|| compile("i-u", REPETITION_STILL));
pub(super) static BLAME_YOU_RE: LazyLock<Regex> = LazyLock::new(|| compile("i-u", BLAME_YOU));
pub(super) static BLAME_WHY_RE: LazyLock<Regex> = LazyLock::new(|| compile("i-u", BLAME_WHY));

pub(super) static FENCED_CODE_RE: LazyLock<Regex> =
    LazyLock::new(|| compile("", r"```[\s\S]*?```"));
pub(super) static XML_TAG_BARE_RE: LazyLock<Regex> =
    LazyLock::new(|| compile("-u", r"</?[A-Za-z][\w-]*\b[^>]*\/?>"));
pub(super) static INLINE_CODE_RE: LazyLock<Regex> = LazyLock::new(|| compile("", r"`[^`\n]*`"));
pub(super) static URL_RE: LazyLock<Regex> = LazyLock::new(|| compile("i", r"\bhttps?://\S+"));
pub(super) static FILE_MENTION_RE: LazyLock<Regex> =
    LazyLock::new(|| compile("-u", r"(^|\s)@[\w./-]+"));
pub(super) static QUOTE_LINE_RE: LazyLock<Regex> = LazyLock::new(|| compile("m", r"^[ \t]*>.*$"));
pub(super) static IMAGE_MARKER_RE: LazyLock<Regex> =
    LazyLock::new(|| compile("", r"\[Image #[0-9]+\]"));
pub(super) static ANSI_ESCAPE_RE: LazyLock<Regex> =
    LazyLock::new(|| compile("", r"\x1b\[[0-9;]*[A-Za-z]"));
