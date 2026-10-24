//! Session title generation, ported from omp TS `utils/title-generator.ts` +
//! `tiny/text.ts` (system prompt verbatim, `<title>` marker parsing, low-signal
//! prefilter, `normalizeGeneratedTitle` caps and casing reconciliation) plus the
//! terminal title write (`\x1b]0;…\x07`, `π: <label>` shape).

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

use pa_types::ai::{
    AssistantContentBlock, Context, Message, Model, StopReason, UserContent, UserMessage,
};

/// Bundled `title-system.md`, verbatim (TS default path sends only this prompt).
pub const TITLE_SYSTEM_PROMPT: &str = "You name coding-agent sessions. Write a 3-6 word title for the task in the next message.\n\nOutput exactly one line: the title inside `<title>` tags, or `<title/>` when there is no task to name.\n\nRules:\n- Name the user's goal as a short imperative phrase built from the user's own key words (the feature, component, or error they name). Avoid vague filler like \"logic\", \"issues\", \"functionality\".\n- `[Image #N, WxH]` marks an image you cannot see. Never mention it or guess what it shows; title the task the surrounding words name.\n- Answer `<title/>` for greetings, acknowledgements, gibberish, or requests too vague to name without context you cannot see (\"help\", \"fix this\", \"what's wrong?\").\n- A `<chat>` block summarizes a longer session: its first `<user>` turn is the opening request, followed by recent turns (often the assistant's working notes). Title the goal the session is working on, using the opening request unless later turns clearly moved to a new task; never title the current micro-step, and ignore file names and symbols unless they are the subject.\n- No quotes, trailing punctuation, or explanations.\n\nExamples:\n- \"the retry queue drops jobs when redis restarts, can u look\" → `<title>Fix retry queue dropping jobs</title>`\n- \"can we get rid of the npm run build warning about peer deps\" → `<title>Remove peer dependency build warning</title>`\n- `<chat>` with only assistant notes like \"adding a backoff field to RetryPolicy… now wiring jitter into schedule()\" → `<title>Add retry backoff with jitter</title>`";

/// Ceiling that survives backends ignoring `disableReasoning` (TS `TITLE_MAX_TOKENS`).
const TITLE_MAX_TOKENS: u64 = 1024;

const MAX_TITLE_CHARS: usize = 80;
const MAX_TITLE_WORDS: usize = 12;

/// Sentinel a capable title model emits when a message carries no concrete task.
pub const NO_TITLE_SENTINEL: &str = "none";

/// Plain π, not the nerd-font `icon.omp` glyph: window/tab titles render in the
/// OS UI font, which has no nerd-font PUA coverage (TS `DEFAULT_TERMINAL_TITLE`).
pub const DEFAULT_TERMINAL_TITLE: &str = "π";

static MARKER_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"(?is)<title>([\s\S]*?)</title>|<title\s*/>|<title>\s*$").unwrap()
});

static THINKING_TAG_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(
        r"(?is)<(?:think|thinking|reasoning)>\s*[\s\S]*?</(?:think|thinking|reasoning)>",
    )
    .unwrap()
});

static THINKING_FENCE_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"(?is)```(?:thinking|reasoning)\b[\s\S]*?```").unwrap()
});

static LEADING_THINKING_TAG_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| {
        regex::Regex::new(
            r"(?is)^\s*<(?:think|thinking|reasoning)>\s*[\s\S]*?</(?:think|thinking|reasoning)>\s*",
        )
        .unwrap()
    });

static LEADING_THINKING_FENCE_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(?is)^\s*```(?:thinking|reasoning)\b[\s\S]*?```\s*").unwrap()
    });

static PROSE_THINKING_PREAMBLE_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(
    || {
        regex::Regex::new(
            r"(?i)^[ \t]*(?:(?:here(?:['\u{2019}]s| is)[ \t]+(?:a|the|my)[ \t]+)|my[ \t]+)?(?:thinking|thought|reasoning)[ \t]+process[ \t]*:?[ \t]*(?:\r?\n|$)",
        )
        .unwrap()
    },
);

static TITLE_TAG_FRAGMENT_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"(?i)</?title>").unwrap());

static WORD_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"[\p{L}\p{N}]+").unwrap());

static QUOTED_TITLE_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r#""title"\s*:\s*("(?:[^"\\]|\\.)*")"#).unwrap());

/// Greeting / acknowledgement / filler tokens (TS `FILLER_TITLE_TOKENS`).
const FILLER_TITLE_TOKENS: &[&str] = &[
    // greetings
    "hi",
    "hii",
    "hiii",
    "hiya",
    "hey",
    "heya",
    "hello",
    "helo",
    "hullo",
    "yo",
    "ya",
    "sup",
    "wassup",
    "whatsup",
    "howdy",
    "greetings",
    "hola",
    "ciao",
    "aloha",
    "gm",
    "gn",
    "good",
    "morning",
    "afternoon",
    "evening",
    "night",
    "day",
    // politeness / acknowledgement
    "thanks",
    "thank",
    "thx",
    "ty",
    "tysm",
    "cheers",
    "please",
    "pls",
    "plz",
    "ok",
    "okay",
    "okey",
    "k",
    "kk",
    "yep",
    "yes",
    "yeah",
    "yup",
    "nope",
    "no",
    "nah",
    "sure",
    "cool",
    "nice",
    "great",
    "awesome",
    "perfect",
    "lol",
    "lmao",
    "haha",
    "hehe",
    // poking the agent / fillers
    "test",
    "tests",
    "testing",
    "ping",
    "pong",
    "there",
    "you",
    "u",
    "hmm",
    "hmmm",
    "um",
    "uh",
    "so",
    "well",
    "anyway",
];

/// Common technical acronyms restored verbatim (TS `COMMON_TITLE_ACRONYMS`).
const COMMON_TITLE_ACRONYMS: &[&str] = &[
    "API", "CLI", "CPU", "CRUD", "CSS", "DNS", "ETL", "GPU", "HTML", "HTTP", "HTTPS", "ID", "JSON",
    "LLM", "REST", "SDK", "SSH", "TCP", "TLS", "TUI", "UI", "URI", "URL", "UX", "XML", "YAML",
];

/// True when a first user message is too low-signal to title (greeting, ack,
/// bare number, or empty): the caller defers titling to a later message instead
/// of latching onto "hi" (TS `isLowSignalTitleInput`, deterministic pre-filter).
#[must_use]
pub fn is_low_signal_title_input(message: &str) -> bool {
    let tokens: Vec<String> = WORD_RE
        .find_iter(message)
        .map(|m| m.as_str().to_lowercase())
        .collect();
    if tokens.is_empty() {
        return true;
    }
    tokens
        .iter()
        .all(|t| FILLER_TITLE_TOKENS.contains(&t.as_str()) || t.chars().all(|c| c.is_ascii_digit()))
}

/// `SessionManager.#cleanTitle`: control chars become spaces, runs collapse, trim.
#[must_use]
pub fn clean_title(raw: &str) -> String {
    let spaced: String = raw
        .chars()
        .map(|c| match c {
            '\u{0000}'..='\u{001f}' | '\u{007f}'..='\u{009f}' => ' ',
            _ => c,
        })
        .collect();
    spaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Normalize one extracted title candidate (TS `normalizeGeneratedTitle`).
/// Caps: 80 chars / 12 words; declines `<title/>`, `none`, and pure punctuation.
#[must_use]
pub fn normalize_generated_title(value: Option<&str>, source_text: Option<&str>) -> Option<String> {
    let value = value?;
    let first_line = value.lines().next().unwrap_or("").trim();
    if first_line.is_empty() {
        return None;
    }
    let unquoted = first_line.trim_matches(|c| c == '"' || c == '\'').trim();
    if unquoted.eq_ignore_ascii_case("<title/>") {
        return None;
    }
    let mut title = TITLE_TAG_FRAGMENT_RE.replace_all(unquoted, "").into_owned();
    title = title
        .trim()
        .trim_matches(|c| c == '"' || c == '\'')
        .to_string();
    if title.ends_with(['.', '!', '?']) {
        title.pop();
    }
    let title = title.trim().to_string();
    if title.is_empty() || title.eq_ignore_ascii_case(NO_TITLE_SENTINEL) {
        return None;
    }
    let words = WORD_RE.find_iter(&title).count();
    if words == 0 || title.chars().count() > MAX_TITLE_CHARS || words > MAX_TITLE_WORDS {
        return None;
    }
    source_text.map_or(Some(title.clone()), |src| {
        Some(reconcile_title_casing(&title, src))
    })
}

/// Reconcile a generated title's casing against the user's own message
/// (TS `reconcileTitleCasing`): keep verbatim tokens, restore distinctive mixed
/// casing (`TinyVMM`), restore ALL-CAPS acronyms from title-cased artifacts
/// (`Cnpg` → `CNPG`, unless the source is shouty), flatten stray camelCase
/// artifacts (`dAemon` → `daemon`), leave the rest.
fn reconcile_title_casing(title: &str, source_text: &str) -> String {
    let mut verbatim: HashSet<&str> = HashSet::new();
    let mut distinctive: BTreeMap<String, String> = BTreeMap::new();
    let mut acronyms: BTreeMap<String, String> = BTreeMap::new();
    let shouty = is_shouty_source(source_text);
    for m in WORD_RE.find_iter(source_text) {
        let token = m.as_str();
        verbatim.insert(token);
        if is_distinctive_casing(token) {
            distinctive
                .entry(token.to_lowercase())
                .or_insert_with(|| token.to_string());
        } else if !shouty && is_all_caps_acronym(token) {
            acronyms
                .entry(token.to_lowercase())
                .or_insert_with(|| token.to_string());
        }
    }
    WORD_RE
        .replace_all(title, |caps: &regex::Captures| {
            let token = caps.get(0).unwrap().as_str();
            if verbatim.contains(token) {
                return token.to_string();
            }
            let lower = token.to_lowercase();
            if let Some(restored) = distinctive.get(&lower) {
                return restored.clone();
            }
            if is_title_cased_artifact(token) {
                if let Some(acronym) = acronyms.get(&lower) {
                    return acronym.clone();
                }
            }
            if is_camel_artifact(token) {
                return lower;
            }
            token.to_string()
        })
        .into_owned()
}

/// Mixed-case identifier the user cased deliberately (`TinyVMM`, `iOS`, `IDs`).
fn is_distinctive_casing(token: &str) -> bool {
    let has_lower = token.chars().any(char::is_lowercase);
    let chars: Vec<char> = token.chars().collect();
    let interior_upper =
        chars.iter().skip(1).any(|c| c.is_uppercase()) || chars.iter().any(|c| c.is_uppercase());
    has_lower && interior_upper && token.chars().any(|c: char| c.is_uppercase())
}

/// Multi-letter ALL-CAPS token with a stronger acronym signal than a plain
/// emphasized word: consonant-only, digit-bearing, or a common acronym.
fn is_all_caps_acronym(token: &str) -> bool {
    if !is_all_caps_word(token) {
        return false;
    }
    let upper = token.to_uppercase();
    if COMMON_TITLE_ACRONYMS.contains(&upper.as_str()) {
        return true;
    }
    if token.chars().any(|c: char| c.is_numeric()) {
        return true;
    }
    !upper.chars().any(|c| "AEIOU".contains(c))
}

/// Multi-letter ALL-CAPS word (requires an actual uppercase letter so caseless
/// scripts like CJK never register).
fn is_all_caps_word(token: &str) -> bool {
    let letters: Vec<char> = token.chars().filter(|c: &char| c.is_alphabetic()).collect();
    if letters.len() < 2 {
        return false;
    }
    letters.iter().any(|c| c.is_uppercase()) && !letters.iter().any(|c| c.is_lowercase())
}

/// Plain title-cased word (`Cnpg`, `Etl`): starts uppercase, has lowercase
/// letters, no interior uppercase (`PascalCase` proper nouns excluded).
fn is_title_cased_artifact(token: &str) -> bool {
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_uppercase() {
        return false;
    }
    if !token.chars().any(|c: char| c.is_lowercase()) {
        return false;
    }
    !chars.any(|c: char| c.is_uppercase())
}

/// ≥2 consecutive multi-letter ALL-CAPS tokens in the source.
fn is_shouty_source(source_text: &str) -> bool {
    let mut run = 0;
    for m in WORD_RE.find_iter(source_text) {
        if is_all_caps_word(m.as_str()) {
            run += 1;
            if run >= 2 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// Lowercase word with a stray interior capital (`dAemon`, `cReate`).
fn is_camel_artifact(token: &str) -> bool {
    let mut chars = token.chars();
    matches!(chars.next(), Some(c) if c.is_lowercase())
        && token.chars().any(|c: char| c.is_uppercase())
}

/// Strip complete known thinking envelopes, iteratively (TS `stripLeakedThinkingMarkup`;
/// the complete-text case needs no streaming healer).
fn strip_thinking_markup(text: &str) -> String {
    let mut current = text.to_string();
    loop {
        let next = THINKING_FENCE_RE
            .replace_all(&THINKING_TAG_RE.replace_all(&current, ""), "")
            .into_owned();
        if next == current {
            return current;
        }
        current = next;
    }
}

/// Remaining thinking-open markers after complete envelopes are stripped: any
/// match swallows everything to the end (TS `StreamMarkupHealing` on the
/// complete-text path), so a marker after one is not visible.
static THINKING_OPEN_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"(?is)<(?:think|thinking|reasoning)>|```(?:thinking|reasoning)\b").unwrap()
});

/// True when the marker at `marker_index` sits in visible text: the prefix,
/// with complete thinking envelopes stripped, holds no unmatched opening tag
/// (TS `isVisibleTitleMarker` via `stripLeakedThinkingMarkup` + sentinel).
fn is_visible_marker(text: &str, marker_index: usize) -> bool {
    const SENTINEL: &str = "\u{e000}omp-title-visible\u{e000}";
    let stripped = strip_thinking_markup(&format!("{}{SENTINEL}", &text[..marker_index]));
    let Some(before) = stripped.strip_suffix(SENTINEL) else {
        return false;
    };
    !THINKING_OPEN_RE.is_match(before)
}

/// Prefer the first closed/visible `<title>` marker, else a plain sentence after
/// stripping only known leading leaked thinking envelopes plus stray title tags
/// (TS `extractGeneratedTitle`). Returns the unwrapped candidate (possibly empty).
#[must_use]
pub fn extract_generated_title(text: &str) -> String {
    let mut at = 0;
    while let Some(caps) = MARKER_RE.captures_at(text, at) {
        let m = caps.get(0).unwrap();
        if is_visible_marker(text, m.start()) {
            let content = caps.get(1).map_or("", |g| g.as_str()).trim();
            return unwrap_json_title(content);
        }
        at = m.end().max(m.start() + 1);
    }
    let cleaned = TITLE_TAG_FRAGMENT_RE
        .replace_all(&strip_leading_thinking_markup(text), "")
        .trim()
        .to_string();
    if PROSE_THINKING_PREAMBLE_RE.is_match(&cleaned) {
        return String::new();
    }
    unwrap_json_title(&cleaned)
}

fn strip_leading_thinking_markup(text: &str) -> String {
    let mut current = text.to_string();
    loop {
        let next = LEADING_THINKING_FENCE_RE
            .replace_all(&LEADING_THINKING_TAG_RE.replace_all(&current, ""), "")
            .into_owned();
        if next == current {
            return current;
        }
        current = next;
    }
}

/// Unwrap a JSON-shaped response (`{"title": "..."}`, optionally code-fenced)
/// into the bare title (TS `unwrapJsonTitle`).
fn unwrap_json_title(candidate: &str) -> String {
    let text = candidate
        .trim_start_matches('`')
        .trim_start_matches("json")
        .trim_end_matches('`')
        .trim();
    if !text.starts_with('{') {
        return candidate.to_string();
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(t) = v.get("title").and_then(serde_json::Value::as_str) {
            return t.trim().to_string();
        }
    }
    // Truncated/malformed JSON: salvage the quoted title value if present.
    if let Some(caps) = QUOTED_TITLE_RE.captures(text) {
        if let Ok(serde_json::Value::String(s)) = serde_json::from_str(&caps[1]) {
            return s.trim().to_string();
        }
    }
    candidate.to_string()
}

/// Wrap the user message for title generation (TS `formatTitleUserMessage`).
#[must_use]
pub fn format_title_user_message(message: &str) -> String {
    format!("<user>\n{message}\n</user>")
}

/// Generate a session title from the first user message with `model`
/// (TS `generateTitleOnlineWithModels`, single-candidate form): the bundled
/// title system prompt, `<user>`-wrapped message, maxTokens 1024,
/// temperature 0, reasoning disabled; the title is parsed from the `<title>`
/// marker in the text output.
///
/// Returns `None` on provider error, abort, or no acceptable title.
pub async fn generate_title_with_model(
    first_message: &str,
    model: &Model,
    api_key: Option<String>,
    headers: Option<BTreeMap<String, String>>,
) -> Option<String> {
    let context = Context {
        system_prompt: Some(TITLE_SYSTEM_PROMPT.to_string()),
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text(format_title_user_message(first_message)),
            timestamp: pa_agent::now_ms().cast_unsigned(),
            rest: serde_json::Map::default(),
        })],
        tools: None,
    };
    // `disableReasoning` like TS — except always-thinking models (zai glm)
    // reject the off flag outright; drop to their catalog thinking floor
    // (`always_thinking_floor`) so the request survives and the marker still
    // fits the 1024-token ceiling.
    let floor = model
        .reasoning
        .then(|| crate::model::always_thinking_floor(&model.provider, &model.id))
        .flatten();
    let (reasoning, disable_reasoning) = match floor {
        Some(level) => (Some(crate::adapter::model_thinking_level(level)), false),
        None => (None, true),
    };
    let options = pa_ai::types::SimpleStreamOptions {
        base: pa_ai::types::StreamOptions {
            max_tokens: Some(TITLE_MAX_TOKENS),
            temperature: Some(0.0),
            api_key,
            headers: headers.map(|h| h.into_iter().collect()),
            ..pa_ai::types::StreamOptions::default()
        },
        reasoning,
        thinking_budgets: None,
        disable_reasoning,
        tool_choice: None,
    };
    let response = match pa_ai::complete_simple(model, &context, Some(options)).await {
        Ok(r) => r,
        Err(e) => {
            if std::env::var_os("OMP_RS_TITLE_TRACE").is_some() {
                eprintln!(
                    "[title] {}/{} completion error: {e:?}",
                    model.provider, model.id
                );
            }
            return None;
        }
    };
    if matches!(
        response.stop_reason,
        StopReason::Aborted | StopReason::Error
    ) {
        if std::env::var_os("OMP_RS_TITLE_TRACE").is_some() {
            let v = serde_json::to_value(&response).unwrap_or_default();
            eprintln!(
                "[title] {}/{} stop_reason {:?}: {}",
                model.provider,
                model.id,
                response.stop_reason,
                serde_json::to_string(&v).unwrap_or_default()
            );
        }
        return None;
    }
    let text: String = response
        .content
        .iter()
        .filter_map(|b| match b {
            AssistantContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();
    let candidate = extract_generated_title(&text);
    normalize_generated_title(Some(&candidate), Some(first_message))
}

/// Remove control characters so model-generated titles cannot inject terminal
/// escapes (TS `sanitizeTerminalTitlePart`).
fn sanitize_terminal_title_part(value: Option<&str>) -> Option<String> {
    let value = value?;
    let sanitized: String = value
        .chars()
        .map(|c| match c {
            '\u{0000}'..='\u{001f}' | '\u{007f}'..='\u{009f}' => ' ',
            _ => c,
        })
        .collect();
    let sanitized = sanitized.trim();
    (!sanitized.is_empty()).then(|| sanitized.to_string())
}

fn fallback_terminal_title(cwd: Option<&Path>) -> Option<String> {
    let base = cwd?.canonicalize().ok()?;
    let base = base.file_name()?.to_string_lossy().into_owned();
    sanitize_terminal_title_part(Some(&base))
}

/// `π: <session name>` (or the cwd basename, or bare `π`) — TS `formatSessionTerminalTitle`.
#[must_use]
pub fn format_session_terminal_title(session_name: Option<&str>, cwd: Option<&Path>) -> String {
    let label = sanitize_terminal_title_part(session_name).or_else(|| fallback_terminal_title(cwd));
    match label {
        Some(l) => format!("{DEFAULT_TERMINAL_TITLE}: {l}"),
        None => DEFAULT_TERMINAL_TITLE.to_string(),
    }
}

/// Set the terminal title via OSC 0 (`\x1b]0;<title>\x07`), as TS `writeTerminalTitle`
/// does on POSIX. No-op when stdout is not a TTY or the title is unchanged.
static LAST_TERMINAL_TITLE: Mutex<Option<String>> = Mutex::new(None);

pub fn set_terminal_title(title: &str) {
    use std::io::IsTerminal;
    if !std::io::stdout().is_terminal() {
        return;
    }
    let next = sanitize_terminal_title_part(Some(title))
        .unwrap_or_else(|| DEFAULT_TERMINAL_TITLE.to_string());
    let mut last = LAST_TERMINAL_TITLE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if last.as_deref() == Some(next.as_str()) {
        return;
    }
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(format!("\x1b]0;{next}\x07").as_bytes());
    let _ = out.flush();
    *last = Some(next);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_signal_detects_greetings_and_acks() {
        assert!(is_low_signal_title_input("hi"));
        assert!(is_low_signal_title_input("Hey there!"));
        assert!(is_low_signal_title_input("ok thanks"));
        assert!(is_low_signal_title_input("42"));
        assert!(is_low_signal_title_input("..."));
        assert!(!is_low_signal_title_input(
            "fix the retry queue dropping jobs"
        ));
    }

    #[test]
    fn normalize_applies_ts_rules() {
        assert_eq!(
            normalize_generated_title(Some("<title>Fix retry queue</title>"), None).as_deref(),
            Some("Fix retry queue")
        );
        assert_eq!(
            normalize_generated_title(Some("\"Quoted title\""), None).as_deref(),
            Some("Quoted title")
        );
        assert_eq!(
            normalize_generated_title(Some("Title."), None).as_deref(),
            Some("Title")
        );
        assert_eq!(normalize_generated_title(Some("<title/>"), None), None);
        assert_eq!(normalize_generated_title(Some("none"), None), None);
        assert_eq!(normalize_generated_title(Some("None"), None), None);
        assert_eq!(normalize_generated_title(Some(".."), None), None);
        assert_eq!(
            normalize_generated_title(Some(&"x".repeat(81)), None),
            None,
            "over 80 chars"
        );
        assert_eq!(
            normalize_generated_title(
                Some("one two three four five six seven eight nine ten eleven twelve thirteen"),
                None
            ),
            None,
            "over 12 words"
        );
        assert_eq!(normalize_generated_title(Some(""), None), None);
        assert_eq!(normalize_generated_title(None, None), None);
    }

    #[test]
    fn casing_is_reconciled_against_source() {
        // Distinctive mixed casing restored.
        assert_eq!(
            normalize_generated_title(Some("Tinyvmm setup"), Some("set up TinyVMM now")).as_deref(),
            Some("TinyVMM setup")
        );
        // ALL-CAPS acronym restored from a title-cased artifact.
        assert_eq!(
            normalize_generated_title(Some("Cnpg backup"), Some("configure CNPG backups"))
                .as_deref(),
            Some("CNPG backup")
        );
        // Stray camelCase artifact flattened.
        assert_eq!(
            normalize_generated_title(Some("dAemon config"), Some("fix the daemon config"))
                .as_deref(),
            Some("daemon config")
        );
        // A plain title-cased word with no acronym match is left alone.
        assert_eq!(
            normalize_generated_title(Some("Daemon config"), Some("fix the daemon config"))
                .as_deref(),
            Some("Daemon config")
        );
        // Shouty source: no acronym restoration.
        assert_eq!(
            normalize_generated_title(Some("Error handling"), Some("ALL ERROR HANDLING IS BROKEN"))
                .as_deref(),
            Some("Error handling")
        );
        // Emphatic all-caps English word not restored.
        assert_eq!(
            normalize_generated_title(Some("Fix bug"), Some("FIX the bug please")).as_deref(),
            Some("Fix bug")
        );
        // Model-cased proper noun left alone.
        assert_eq!(
            normalize_generated_title(Some("GitHub OAuth setup"), Some("wire up github oauth"))
                .as_deref(),
            Some("GitHub OAuth setup")
        );
    }

    #[test]
    fn extracts_marked_title() {
        assert_eq!(
            extract_generated_title("<title>Fix retry queue</title>"),
            "Fix retry queue"
        );
        // Thinking envelope before the marker is stripped.
        assert_eq!(
            extract_generated_title("<thinking>let me think</thinking>\n<title>Title</title>"),
            "Title"
        );
        // Marker hidden inside a thinking envelope is ignored; fallback path used.
        assert_eq!(
            extract_generated_title("<thinking>use <title>Hidden</title></thinking> visible"),
            "visible"
        );
        // JSON-shaped output unwrapped.
        assert_eq!(
            extract_generated_title(r#"{"title": "Fix queue"}"#),
            "Fix queue"
        );
        assert_eq!(
            extract_generated_title("```json\n{\"title\": \"Fix queue\"}\n```"),
            "Fix queue"
        );
        // Truncated JSON salvaged (quoted value intact, object cut off).
        assert_eq!(
            extract_generated_title(r#"{"title": "Fix queue", "#),
            "Fix queue"
        );
        // Leading thinking preamble rejected on the markerless path.
        assert_eq!(
            extract_generated_title("Here is my thinking process:\nthe title"),
            ""
        );
    }

    #[test]
    fn terminal_title_shape_matches_ts() {
        assert_eq!(
            format_session_terminal_title(Some("My Session"), None),
            "π: My Session"
        );
        assert_eq!(format_session_terminal_title(None, None), "π");
        assert_eq!(
            format_session_terminal_title(Some("bad\u{0007}title"), None),
            "π: bad title"
        );
        let dir = std::env::temp_dir();
        assert_eq!(
            format_session_terminal_title(None, Some(&dir)),
            format!(
                "π: {}",
                dir.canonicalize()
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
            )
        );
    }

    #[test]
    fn clean_title_sanitizes_control_chars() {
        assert_eq!(clean_title("a\u{0007}b\u{001f}c   d"), "a b c d");
        assert_eq!(clean_title("  pad  "), "pad");
    }

    #[test]
    fn title_user_message_is_wrapped() {
        assert_eq!(format_title_user_message("hello"), "<user>\nhello\n</user>");
    }
}
