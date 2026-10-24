//! `web_search`: omp's web search tool, ported from pi-coding-agent's
//! `src/web/search/index.ts`.
//!
//! Provider selection: the user's default model `anthropic/claude-opus-5-5:medium`
//! carries `webSearch: "anthropic"` in the pi-catalog, so only the Anthropic
//! provider is implemented. The tool parses Google-style query operators
//! (`site:`, `inurl:`, `intitle:`, `filetype:`, `before:`/`after:`, phrases,
//! negation, `OR` groups), maps `site:` onto Anthropic's `allowed_domains`/
//! `blocked_domains`, passes the rest through as query text, and formats the
//! LLM-facing result.

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback, ToolResultContent};
use pa_agent::BoxFut;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ─── Constants ─────────────────────────────────────────────────────────────

/// omp `webSearchDescription` — first line compared by tests-tools.
pub const DESCRIPTION: &str = "Known URLs/programmatic data → `read`. Query: site: or -site:, after: or before: YYYY-MM-DD, inurl:, intitle:, filetype:, \"phrase\", -term, OR. Prefer primary sources; MUST link citations.";

/// omp `webSearchSystemPrompt` (sent as Anthropic `system` parameter).
const SYSTEM_PROMPT: &str =
    "Web research assistant: accurate, well-sourced, comprehensive answers.";

const WEB_SEARCH_TOOL_NAME: &str = "web_search";
const WEB_SEARCH_TOOL_TYPE: &str = "web_search_20250305";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const DEFAULT_TIMEOUT_SECS: u64 = 60;
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const ANTHROPIC_BETA: &str = "web-search-2025-03-05";

/// Anthropic query syntax: `site:` is mapped to `allowed_domains`/`blocked_domains`
/// instead, so it is absent from the syntax.
const ANTHROPIC_QUERY_SYNTAX: QuerySyntax = QuerySyntax {
    phrases: true,
    negation: true,
    or: true,
    site: false,
    in_url: true,
    in_title: true,
    in_text: false,
    filetype: true,
    date_range: true,
};

// ─── Compiled regexes ──────────────────────────────────────────────────────

static DIRECTIVE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^([+-]?)([a-z][a-z-]*):(.*)$").unwrap());

static PAGE_AGE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^(\d+)\s*(s|sec|second|m|min|minute|h|hour|d|day|w|week|mo|month|y|year)s?\s*(ago)?$",
    )
    .unwrap()
});

static RELATIVE_AGE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(\d+)\s*(minute|min|hour|hr|day|week|month|mo|year|yr|[mhdwy])s?\s*(ago)?$")
        .unwrap()
});

static ISO_DATE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d{4})-(\d{2})-(\d{2})").unwrap());

// ─── Schema ────────────────────────────────────────────────────────────────

#[must_use]
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "recency": {"type": "string", "enum": ["day", "week", "month", "year"]},
            "limit": {"type": "number"}
        },
        "required": ["query"],
        "additionalProperties": false
    })
}

// ─── Serialization types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchUsage {
    #[serde(rename = "inputTokens", skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(rename = "outputTokens", skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(rename = "searchRequests", skip_serializing_if = "Option::is_none")]
    pub search_requests: Option<u32>,
    #[serde(rename = "totalTokens", skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchSource {
    pub title: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    #[serde(rename = "publishedDate", skip_serializing_if = "Option::is_none")]
    pub published_date: Option<String>,
    #[serde(rename = "ageSeconds", skip_serializing_if = "Option::is_none")]
    pub age_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchCitation {
    pub url: String,
    pub title: String,
    #[serde(rename = "citedText", skip_serializing_if = "Option::is_none")]
    pub cited_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchResponse {
    pub provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    pub sources: Vec<SearchSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<SearchCitation>>,
    #[serde(rename = "searchQueries", skip_serializing_if = "Option::is_none")]
    pub search_queries: Option<Vec<String>>,
    #[serde(rename = "relatedQuestions", skip_serializing_if = "Option::is_none")]
    pub related_questions: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<SearchUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(rename = "requestId", skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(rename = "authMode", skip_serializing_if = "Option::is_none")]
    pub auth_mode: Option<String>,
}

// ─── Query parsing types ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
struct QueryTerm {
    text: String,
    phrase: bool,
    negated: bool,
    group: Option<usize>,
}

#[derive(Debug, Clone, Default)]
struct StructuredQuery {
    raw: String,
    text: String,
    terms: Vec<QueryTerm>,
    sites: Vec<String>,
    excluded_sites: Vec<String>,
    in_url: Vec<String>,
    excluded_in_url: Vec<String>,
    in_title: Vec<String>,
    excluded_in_title: Vec<String>,
    in_text: Vec<String>,
    excluded_in_text: Vec<String>,
    filetypes: Vec<String>,
    excluded_filetypes: Vec<String>,
    after: Option<String>,
    before: Option<String>,
    lang: Option<String>,
    has_directives: bool,
    has_constraints: bool,
}

#[derive(Debug, Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools)] // mirrors omp's QuerySyntax flags one-to-one
struct QuerySyntax {
    phrases: bool,
    negation: bool,
    or: bool,
    site: bool,
    in_url: bool,
    in_title: bool,
    in_text: bool,
    filetype: bool,
    date_range: bool,
}

struct RawToken {
    text: String,
    quoted: bool,
    quoted_value: bool,
}

// ─── Query parsing functions ────────────────────────────────────────────────

fn is_quote(ch: char) -> bool {
    ch == '"' || ch == '\u{201C}' || ch == '\u{201D}'
}

fn is_reserved_token(text: &str) -> bool {
    matches!(
        text,
        "(" | ")" | "OR" | "AND" | "NOT" | "|" | "||" | "&&" | "!"
    )
}

fn tokenize(raw: &str) -> Vec<RawToken> {
    let chars: Vec<char> = raw.chars().collect();
    let n = chars.len();
    let mut tokens: Vec<RawToken> = Vec::new();
    let mut i = 0;
    while i < n {
        let ch = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if is_quote(ch) {
            let start = i + 1;
            let mut j = start;
            while j < n && !is_quote(chars[j]) {
                j += 1;
            }
            let text: String = chars[start..j].iter().collect();
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                tokens.push(RawToken {
                    text: trimmed.to_string(),
                    quoted: true,
                    quoted_value: false,
                });
            }
            i = j + 1;
            continue;
        }
        let mut buf = String::new();
        let mut quoted_value = false;
        while i < n && !chars[i].is_whitespace() {
            let c = chars[i];
            if is_quote(c) && buf.ends_with(':') {
                let start = i + 1;
                let mut j = start;
                while j < n && !is_quote(chars[j]) {
                    j += 1;
                }
                buf.push_str(&chars[start..j].iter().collect::<String>());
                quoted_value = true;
                i = j + 1;
                continue;
            }
            if is_quote(c) {
                break;
            }
            buf.push(c);
            i += 1;
        }
        if !buf.is_empty() {
            tokens.push(RawToken {
                text: buf,
                quoted: false,
                quoted_value,
            });
        }
    }
    split_parens(tokens)
}

fn split_parens(tokens: Vec<RawToken>) -> Vec<RawToken> {
    let mut out: Vec<RawToken> = Vec::new();
    for tok in tokens {
        if tok.quoted || tok.quoted_value {
            out.push(tok);
            continue;
        }
        let mut text = tok.text;
        while let Some(rest) = text.strip_prefix('(') {
            out.push(RawToken {
                text: "(".to_string(),
                quoted: false,
                quoted_value: false,
            });
            text = rest.to_string();
        }
        let mut trailing = 0;
        loop {
            if !text.ends_with(')') {
                break;
            }
            let body = &text[..text.len() - 1];
            let depth: i32 = body
                .chars()
                .map(|c| match c {
                    '(' => 1,
                    ')' => -1,
                    _ => 0,
                })
                .sum();
            if depth > 0 {
                break;
            }
            text = body.to_string();
            trailing += 1;
        }
        if !text.is_empty() {
            out.push(RawToken {
                text,
                quoted: false,
                quoted_value: false,
            });
        }
        for _ in 0..trailing {
            out.push(RawToken {
                text: ")".to_string(),
                quoted: false,
                quoted_value: false,
            });
        }
    }
    out
}

fn iso_date(year: i32, month: u32, day: u32) -> Option<String> {
    if !(1000..=9999).contains(&year) || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some(format!("{year:04}-{month:02}-{day:02}"))
}

fn parse_date_value(value: &str) -> Option<String> {
    let t = value.trim();
    let re1 = Regex::new(r"^(\d{4})(?:[-/.](\d{1,2})(?:[-/.](\d{1,2}))?)?$").unwrap();
    if let Some(caps) = re1.captures(t) {
        let year: i32 = caps[1].parse().ok()?;
        let month: u32 = caps.get(2).map_or(1, |m| m.as_str().parse().unwrap_or(1));
        let day: u32 = caps.get(3).map_or(1, |m| m.as_str().parse().unwrap_or(1));
        return iso_date(year, month, day);
    }
    let re2 = Regex::new(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$").unwrap();
    if let Some(caps) = re2.captures(t) {
        let mut month: u32 = caps[1].parse().ok()?;
        let mut day: u32 = caps[2].parse().ok()?;
        if month > 12 && day <= 12 {
            std::mem::swap(&mut month, &mut day);
        }
        let year: i32 = caps[3].parse().ok()?;
        return iso_date(year, month, day);
    }
    None
}

fn normalize_site(value: &str) -> String {
    let mut site = value.trim().to_ascii_lowercase();
    if let Some(idx) = site.find("://") {
        site = site[idx + 3..].to_string();
    }
    if site.starts_with("*.") {
        site = site[2..].to_string();
    }
    while site.ends_with('/') || site.ends_with('.') {
        site.pop();
    }
    site
}

fn directive_field(name: &str) -> Option<&'static str> {
    let lname = name.to_ascii_lowercase();
    if ["site", "domain", "host"].contains(&lname.as_str()) {
        Some("site")
    } else if ["inurl", "url"].contains(&lname.as_str()) {
        Some("inUrl")
    } else if ["intitle", "title"].contains(&lname.as_str()) {
        Some("inTitle")
    } else if ["intext", "inbody", "inanchor", "allintext"].contains(&lname.as_str()) {
        Some("inText")
    } else if ["filetype", "ext"].contains(&lname.as_str()) {
        Some("filetype")
    } else if ["before", "until"].contains(&lname.as_str()) {
        Some("before")
    } else if ["after", "since"].contains(&lname.as_str()) {
        Some("after")
    } else if ["lang", "language"].contains(&lname.as_str()) {
        Some("lang")
    } else {
        None
    }
}

/// Convert days since epoch (1970-01-01) to (year, month, day) using
/// Howard Hinnant's `civil_from_days` algorithm.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Hinnant algorithm output is always in 1..=12 / 1..=31
fn days_to_ymd(days: u64) -> (i64, u32, u32) {
    let z = days.cast_signed() + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m as u32, d as u32)
}

/// Convert (year, month, day) to days since epoch using Howard Hinnant's
/// `days_from_civil` algorithm.
fn ymd_to_days(year: i64, month: i64, day: i64) -> i64 {
    let z = year - i64::from(month <= 2);
    let era = if z >= 0 { z } else { z - 399 } / 400;
    let yoe = z - era * 400;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn epoch_secs_to_iso_date(secs: u64) -> String {
    let days = secs / 86400;
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}")
}

fn iso_date_to_epoch_secs(date: &str) -> Option<u64> {
    let caps = ISO_DATE_RE.captures(date)?;
    let year: i64 = caps[1].parse().ok()?;
    let month: i64 = caps[2].parse().ok()?;
    let day: i64 = caps[3].parse().ok()?;
    let days = ymd_to_days(year, month, day);
    Some(days.cast_unsigned() * 86_400)
}

fn parse_page_age(page_age: &str) -> Option<f64> {
    let caps = PAGE_AGE_RE.captures(page_age)?;
    let value: f64 = caps[1].parse().ok()?;
    let unit = caps[2].to_lowercase();
    let multiplier = match unit.as_str() {
        "s" | "sec" | "second" => 1.0,
        "m" | "min" | "minute" => 60.0,
        "h" | "hr" | "hour" => 3600.0,
        "d" | "day" => 86400.0,
        "w" | "week" => 604_800.0,
        "mo" | "month" => 2_592_000.0,
        "y" | "yr" | "year" => 31_536_000.0,
        _ => return None,
    };
    Some(value * multiplier)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)] // TS number math: epoch-ms fits u64 for any realistic date
fn source_time(source: &SearchSource) -> Option<u64> {
    if let Some(age) = source.age_seconds {
        if age.is_finite() {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;
            let ms = now.as_secs() as f64 * 1000.0 - age * 1000.0;
            return Some((ms / 1000.0) as u64);
        }
    }
    let published = source.published_date.as_deref()?;
    let trimmed = published.trim();
    if let Some(caps) = RELATIVE_AGE_RE.captures(trimmed) {
        let value: f64 = caps[1].parse().ok()?;
        let unit = caps[2].to_lowercase();
        let multiplier = match unit.as_str() {
            "minute" | "min" | "m" => 60.0,
            "hour" | "hr" | "h" => 3600.0,
            "day" | "d" => 86400.0,
            "week" | "w" => 604_800.0,
            "month" | "mo" => 2_592_000.0,
            "year" | "yr" | "y" => 31_536_000.0,
            _ => return None,
        };
        let seconds = value * multiplier;
        if seconds > 0.0 {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;
            return Some((now.as_secs() as f64 - seconds) as u64);
        }
        return None;
    }
    // Try ISO date
    if let Some(epoch) = iso_date_to_epoch_secs(trimmed) {
        return Some(epoch);
    }
    None
}

// ─── Query parsing core ─────────────────────────────────────────────────────

fn quote_value(value: &str) -> String {
    if value.contains(char::is_whitespace) {
        format!("\"{value}\"")
    } else {
        value.to_string()
    }
}

fn render_term(term: &QueryTerm, syntax: &QuerySyntax) -> Option<String> {
    if term.negated && !syntax.negation {
        return None;
    }
    let body = if term.phrase && syntax.phrases {
        format!("\"{}\"", term.text)
    } else {
        term.text.clone()
    };
    if term.negated {
        Some(format!("-{body}"))
    } else {
        Some(body)
    }
}

fn render_terms(terms: &[QueryTerm], syntax: &QuerySyntax) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < terms.len() {
        let term = &terms[i];
        let mut handled_group = false;
        if syntax.or {
            if let Some(group) = term.group {
                let mut members: Vec<String> = Vec::new();
                let mut j = i;
                while j < terms.len() && terms[j].group == Some(group) {
                    if let Some(rendered) = render_term(&terms[j], syntax) {
                        members.push(rendered);
                    }
                    j += 1;
                }
                i = j;
                if members.len() > 1 {
                    parts.push(format!("({})", members.join(" OR ")));
                } else if members.len() == 1 {
                    parts.push(members[0].clone());
                }
                handled_group = true;
            }
        }
        if handled_group {
            continue;
        }
        if let Some(rendered) = render_term(term, syntax) {
            parts.push(rendered);
        }
        i += 1;
    }
    parts.join(" ")
}

fn format_query(q: &StructuredQuery, syntax: &QuerySyntax) -> String {
    let mut parts: Vec<String> = Vec::new();
    let text = render_terms(&q.terms, syntax);
    if !text.is_empty() {
        parts.push(text);
    }
    if syntax.site {
        if q.sites.len() > 1 && syntax.or {
            parts.push(format!(
                "({})",
                q.sites
                    .iter()
                    .map(|s| format!("site:{s}"))
                    .collect::<Vec<_>>()
                    .join(" OR ")
            ));
        } else {
            for s in &q.sites {
                parts.push(format!("site:{s}"));
            }
        }
        for s in &q.excluded_sites {
            parts.push(format!("-site:{s}"));
        }
    }
    if syntax.in_url {
        for v in &q.in_url {
            parts.push(format!("inurl:{}", quote_value(v)));
        }
        for v in &q.excluded_in_url {
            parts.push(format!("-inurl:{}", quote_value(v)));
        }
    }
    if syntax.in_title {
        for v in &q.in_title {
            parts.push(format!("intitle:{}", quote_value(v)));
        }
        for v in &q.excluded_in_title {
            parts.push(format!("-intitle:{}", quote_value(v)));
        }
    }
    if syntax.in_text {
        for v in &q.in_text {
            parts.push(format!("intext:{}", quote_value(v)));
        }
        for v in &q.excluded_in_text {
            parts.push(format!("-intext:{}", quote_value(v)));
        }
    }
    if syntax.filetype {
        if q.filetypes.len() > 1 && syntax.or {
            parts.push(format!(
                "({})",
                q.filetypes
                    .iter()
                    .map(|f| format!("filetype:{f}"))
                    .collect::<Vec<_>>()
                    .join(" OR ")
            ));
        } else {
            for f in &q.filetypes {
                parts.push(format!("filetype:{f}"));
            }
        }
        for f in &q.excluded_filetypes {
            parts.push(format!("-filetype:{f}"));
        }
    }
    if syntax.date_range {
        if let Some(after) = &q.after {
            parts.push(format!("after:{after}"));
        }
        if let Some(before) = &q.before {
            parts.push(format!("before:{before}"));
        }
    }
    let result = parts.join(" ").trim().to_string();
    if result.is_empty() {
        let fallback: Vec<&str> = q
            .sites
            .iter()
            .chain(&q.in_title)
            .chain(&q.in_url)
            .chain(&q.in_text)
            .chain(&q.filetypes)
            .map(std::string::String::as_str)
            .collect();
        let fb = fallback.join(" ").trim().to_string();
        return if fb.is_empty() {
            q.raw.trim().to_string()
        } else {
            fb
        };
    }
    result
}

/// Map parsed directives onto the Anthropic request: `site:` includes →
/// `allowed_domains`, `-site:` exclusions → `blocked_domains` (only when no
/// includes), remaining directives re-emitted as query syntax.
#[derive(Debug, Clone, Default)]
struct AnthropicQueryPlan {
    query: String,
    allowed_domains: Option<Vec<String>>,
    blocked_domains: Option<Vec<String>>,
}

fn plan_query(raw_query: &str, parsed: &StructuredQuery) -> AnthropicQueryPlan {
    if !parsed.has_directives {
        return AnthropicQueryPlan {
            query: raw_query.to_string(),
            allowed_domains: None,
            blocked_domains: None,
        };
    }
    let collect_hosts = |sites: &[String]| -> Vec<String> {
        let mut seen = HashSet::new();
        let mut hosts: Vec<String> = Vec::new();
        for site in sites {
            let host = site.split('/').next().unwrap_or(site);
            if !host.is_empty() && seen.insert(host.to_string()) {
                hosts.push(host.to_string());
            }
        }
        hosts
    };
    let allowed = collect_hosts(&parsed.sites);
    let blocked = if allowed.is_empty() {
        collect_hosts(&parsed.excluded_sites)
    } else {
        Vec::new()
    };
    let query = format_query(parsed, &ANTHROPIC_QUERY_SYNTAX);
    AnthropicQueryPlan {
        query,
        allowed_domains: if allowed.is_empty() {
            None
        } else {
            Some(allowed)
        },
        blocked_domains: if blocked.is_empty() {
            None
        } else {
            Some(blocked)
        },
    }
}

fn matches_site(url: &str, site: &str) -> bool {
    let parsed = match url.find("://") {
        Some(idx) => &url[idx + 3..],
        None => url,
    };
    let slash = site.find('/');
    let (site_host, site_path) = match slash {
        Some(pos) => (&site[..pos], &site[pos..]),
        None => (site, ""),
    };
    let parsed_host = parsed.split('/').next().unwrap_or(parsed).to_lowercase();
    if parsed_host != site_host && !parsed_host.ends_with(&format!(".{site_host}")) {
        return false;
    }
    if !site_path.is_empty() {
        let parsed_path = parsed[parsed_host.len()..].to_lowercase();
        if !parsed_path.starts_with(&site_path.to_lowercase()) {
            return false;
        }
    }
    true
}

fn matches_filetype(url: &str, ext: &str) -> bool {
    let path = url.split('/').next_back().unwrap_or("");
    path.to_lowercase().ends_with(&format!(".{ext}"))
}

/// Result of `apply_query_constraints` — sources surviving the lenient filter
/// plus the directive labels that were relaxed.
struct ConstraintFilterResult {
    sources: Vec<SearchSource>,
    dropped: Vec<String>,
}

struct ConstraintDimension {
    label: String,
    matches: Vec<String>,
    kind: ConstraintKind,
}

#[derive(Clone, Copy)]
enum ConstraintKind {
    Include, // any-of
    Exclude, // none-of
    DateRange,
}

fn constraint_dimensions(q: &StructuredQuery) -> Vec<ConstraintDimension> {
    let mut dims: Vec<ConstraintDimension> = Vec::new();
    if !q.sites.is_empty() {
        dims.push(ConstraintDimension {
            label: format!("site:{}", q.sites.join(" OR ")),
            matches: q.sites.clone(),
            kind: ConstraintKind::Include,
        });
    }
    if !q.excluded_sites.is_empty() {
        dims.push(ConstraintDimension {
            label: format!("-site:{}", q.excluded_sites.join(" ")),
            matches: q.excluded_sites.clone(),
            kind: ConstraintKind::Exclude,
        });
    }
    if !q.in_url.is_empty() {
        dims.push(ConstraintDimension {
            label: format!("inurl:{}", q.in_url.join(" ")),
            matches: q.in_url.clone(),
            kind: ConstraintKind::Include,
        });
    }
    if !q.excluded_in_url.is_empty() {
        dims.push(ConstraintDimension {
            label: format!("-inurl:{}", q.excluded_in_url.join(" ")),
            matches: q.excluded_in_url.clone(),
            kind: ConstraintKind::Exclude,
        });
    }
    if !q.in_title.is_empty() {
        dims.push(ConstraintDimension {
            label: format!("intitle:{}", q.in_title.join(" ")),
            matches: q.in_title.clone(),
            kind: ConstraintKind::Include,
        });
    }
    if !q.excluded_in_title.is_empty() {
        dims.push(ConstraintDimension {
            label: format!("-intitle:{}", q.excluded_in_title.join(" ")),
            matches: q.excluded_in_title.clone(),
            kind: ConstraintKind::Exclude,
        });
    }
    if !q.filetypes.is_empty() {
        dims.push(ConstraintDimension {
            label: format!("filetype:{}", q.filetypes.join(" OR ")),
            matches: q.filetypes.clone(),
            kind: ConstraintKind::Include,
        });
    }
    if !q.excluded_filetypes.is_empty() {
        dims.push(ConstraintDimension {
            label: format!("-filetype:{}", q.excluded_filetypes.join(" ")),
            matches: q.excluded_filetypes.clone(),
            kind: ConstraintKind::Exclude,
        });
    }
    if q.after.is_some() || q.before.is_some() {
        let mut parts: Vec<String> = Vec::new();
        if let Some(after) = &q.after {
            parts.push(format!("after:{after}"));
        }
        if let Some(before) = &q.before {
            parts.push(format!("before:{before}"));
        }
        dims.push(ConstraintDimension {
            label: parts.join(" "),
            matches: Vec::new(),
            kind: ConstraintKind::DateRange,
        });
    }
    dims
}

fn dimension_pred(dim: &ConstraintDimension, q: &StructuredQuery, source: &SearchSource) -> bool {
    match dim.kind {
        ConstraintKind::DateRange => {
            let time = source_time(source);
            if time.is_none() {
                return true;
            }
            let time = time.unwrap();
            let after = q.after.as_ref().and_then(|d| iso_date_to_epoch_secs(d));
            let before = q.before.as_ref().and_then(|d| iso_date_to_epoch_secs(d));
            if let Some(after) = after {
                if time < after {
                    return false;
                }
            }
            if let Some(before) = before {
                if time >= before {
                    return false;
                }
            }
            true
        }
        ConstraintKind::Include => match &dim.matches[0] {
            site if q.sites.contains(site) => q.sites.iter().any(|s| matches_site(&source.url, s)),
            v if q.in_url.contains(v) => q
                .in_url
                .iter()
                .all(|v| source.url.to_lowercase().contains(&v.to_lowercase())),
            v if q.in_title.contains(v) => q
                .in_title
                .iter()
                .all(|v| source.title.to_lowercase().contains(&v.to_lowercase())),
            ext if q.filetypes.contains(ext) => {
                q.filetypes.iter().any(|e| matches_filetype(&source.url, e))
            }
            _ => true,
        },
        ConstraintKind::Exclude => {
            // Check exclusions
            if q.excluded_sites
                .iter()
                .any(|s| matches_site(&source.url, s))
            {
                return false;
            }
            if q.excluded_in_url
                .iter()
                .any(|v| source.url.to_lowercase().contains(&v.to_lowercase()))
            {
                return false;
            }
            if q.excluded_in_title
                .iter()
                .any(|v| source.title.to_lowercase().contains(&v.to_lowercase()))
            {
                return false;
            }
            if q.excluded_filetypes
                .iter()
                .any(|e| matches_filetype(&source.url, e))
            {
                return false;
            }
            true
        }
    }
}

/// Strict per-source check: every constraint dimension must pass.
#[allow(dead_code)]
fn matches_query_constraints(source: &SearchSource, q: &StructuredQuery) -> bool {
    constraint_dimensions(q)
        .iter()
        .all(|dim| dimension_pred(dim, q, source))
}

/// Lenient post-filter: skips dimensions that would eliminate every source.
fn apply_query_constraints(
    sources: &[SearchSource],
    q: &StructuredQuery,
) -> ConstraintFilterResult {
    let mut current = sources.to_vec();
    let mut dropped: Vec<String> = Vec::new();
    if current.is_empty() {
        return ConstraintFilterResult {
            sources: current,
            dropped,
        };
    }
    for dim in constraint_dimensions(q) {
        let kept: Vec<SearchSource> = current
            .iter()
            .filter(|s| dimension_pred(&dim, q, s))
            .cloned()
            .collect();
        if kept.is_empty() {
            dropped.push(dim.label);
        } else {
            current = kept;
        }
    }
    ConstraintFilterResult {
        sources: current,
        dropped,
    }
}

/// Parse a raw query into a `StructuredQuery`.
#[allow(clippy::too_many_lines)] // single-pass tokenizer mirroring omp's parseSearchQuery
fn parse_search_query(raw: &str) -> StructuredQuery {
    let mut q = StructuredQuery {
        raw: raw.to_string(),
        ..Default::default()
    };

    let tokens = tokenize(raw);
    let mut state = ParseState::new();
    let mut all_mode: Option<&'static str> = None;
    let mut idx: usize = 0;

    while idx < tokens.len() {
        let tok = &tokens[idx];

        if tok.quoted {
            state.push_term(&mut q, &tok.text, true);
            idx += 1;
            continue;
        }

        match tok.text.as_str() {
            "(" | ")" => {
                idx += 1;
                continue;
            }
            "OR" | "|" | "||" => {
                state.or_pending = true;
                q.has_directives = true;
                idx += 1;
                continue;
            }
            "AND" | "&&" => {
                q.has_directives = true;
                idx += 1;
                continue;
            }
            "NOT" | "!" => {
                state.negate_next = true;
                q.has_directives = true;
                idx += 1;
                continue;
            }
            "-" | "+" => {
                if tok.text == "-" && tokens.get(idx + 1).is_some_and(|t| t.quoted) {
                    state.negate_next = true;
                }
                idx += 1;
                continue;
            }
            _ => {}
        }

        let caps = DIRECTIVE_RE.captures(&tok.text);
        let name = caps.as_ref().and_then(|c| c.get(2).map(|m| m.as_str()));
        let all_match = name.and_then(|n| {
            ALL_MODES
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(n))
                .map(|(_, v)| *v)
        });
        let field = name.and_then(directive_field);

        if let (Some(caps), Some(mode)) = (&caps, all_match) {
            q.has_directives = true;
            all_mode = Some(mode);
            let inline = caps.get(3).map_or("", |m| m.as_str()).trim();
            if !inline.is_empty() {
                let negated = caps.get(1).map(|m| m.as_str()) == Some("-");
                push_constraint_into(&mut q, mode, inline, negated);
            }
            state.or_pending = false;
            state.last_was_term = false;
            idx += 1;
            continue;
        }

        if let (Some(cap), Some(fld)) = (caps.as_ref(), field) {
            let mut value = cap.get(3).map_or("", |m| m.as_str()).trim().to_string();
            if value.is_empty() {
                if let Some(next) = tokens.get(idx + 1) {
                    if next.quoted || !is_reserved_token(&next.text) {
                        value.clone_from(&next.text);
                        idx += 1; // skip next token
                    }
                }
            }
            if value.is_empty() {
                q.has_directives = true;
                idx += 1;
                continue;
            }

            let negated = cap.get(1).map(|m| m.as_str()) == Some("-") || state.negate_next;
            state.negate_next = false;

            match fld {
                "before" | "after" => {
                    if let Some(iso) = parse_date_value(&value) {
                        q.has_directives = true;
                        state.or_pending = false;
                        state.last_was_term = false;
                        if fld == "before" {
                            q.before = Some(iso);
                        } else {
                            q.after = Some(iso);
                        }
                    } else {
                        state.push_term(&mut q, &tok.text, false);
                    }
                }
                "lang" => {
                    q.lang = Some(value.to_ascii_lowercase());
                    q.has_directives = true;
                    state.or_pending = false;
                    state.last_was_term = false;
                }
                _ => {
                    push_constraint_into(&mut q, fld, &value, negated);
                    state.or_pending = false;
                    state.last_was_term = false;
                }
            }
            idx += 1;
            continue;
        }

        // Plain term with optional +/- prefix
        let mut text = tok.text.clone();
        if text.starts_with('-') && text.len() > 1 {
            state.negate_next = true;
            q.has_directives = true;
            text = text.trim_start_matches('-').to_string();
            if text.is_empty() {
                idx += 1;
                continue;
            }
            if let Some(cap) = DIRECTIVE_RE.captures(&text) {
                let n = cap.get(2).map(|m| m.as_str());
                let nf = n.and_then(directive_field);
                if let Some(nf) = nf {
                    if nf != "before" && nf != "after" && nf != "lang" {
                        state.negate_next = false;
                        let val = cap.get(3).map_or("", |m| m.as_str());
                        push_constraint_into(&mut q, nf, val, true);
                        idx += 1;
                        continue;
                    }
                }
            }
            state.push_term(&mut q, &text, false);
            idx += 1;
            continue;
        }
        if text.starts_with('+') && text.len() > 1 {
            state.push_term(&mut q, &text[1..], true);
            q.has_directives = true;
            idx += 1;
            continue;
        }
        state.push_term(&mut q, &text, false);
        idx += 1;
    }

    q.text = render_terms(
        &q.terms,
        &QuerySyntax {
            phrases: true,
            negation: true,
            or: true,
            ..Default::default()
        },
    );
    q.has_constraints = !q.sites.is_empty()
        || !q.excluded_sites.is_empty()
        || !q.in_url.is_empty()
        || !q.excluded_in_url.is_empty()
        || !q.in_title.is_empty()
        || !q.excluded_in_title.is_empty()
        || !q.filetypes.is_empty()
        || !q.excluded_filetypes.is_empty()
        || q.before.is_some()
        || q.after.is_some();

    if all_mode.is_some() {
        q.has_directives = true;
    }

    q
}

fn push_constraint_into(q: &mut StructuredQuery, field: &str, value: &str, negated: bool) {
    let v = value.trim();
    if v.is_empty() {
        return;
    }
    q.has_directives = true;
    match field {
        "site" => {
            let site = normalize_site(v);
            if !site.is_empty() {
                if negated {
                    q.excluded_sites.push(site);
                } else {
                    q.sites.push(site);
                }
            }
        }
        "inUrl" => {
            if negated {
                q.excluded_in_url.push(v.to_string());
            } else {
                q.in_url.push(v.to_string());
            }
        }
        "inTitle" => {
            if negated {
                q.excluded_in_title.push(v.to_string());
            } else {
                q.in_title.push(v.to_string());
            }
        }
        "inText" => {
            if negated {
                q.excluded_in_text.push(v.to_string());
            } else {
                q.in_text.push(v.to_string());
            }
        }
        "filetype" => {
            let ext = v.trim_start_matches('.').to_ascii_lowercase();
            if !ext.is_empty() {
                if negated {
                    q.excluded_filetypes.push(ext);
                } else {
                    q.filetypes.push(ext);
                }
            }
        }
        _ => {}
    }
}

/// State carried during `parse_search_query`.
struct ParseState {
    negate_next: bool,
    or_pending: bool,
    last_was_term: bool,
    group_seq: usize,
}

impl ParseState {
    fn new() -> Self {
        ParseState {
            negate_next: false,
            or_pending: false,
            last_was_term: false,
            group_seq: 0,
        }
    }

    fn push_term(&mut self, q: &mut StructuredQuery, text: &str, phrase: bool) {
        let negated = self.negate_next;
        self.negate_next = false;
        q.has_directives = true;
        if self.or_pending && self.last_was_term {
            let prev_group = q.terms.last().and_then(|t| t.group);
            if let Some(group) = prev_group {
                q.terms.push(QueryTerm {
                    text: text.to_string(),
                    phrase,
                    negated,
                    group: Some(group),
                });
            } else {
                self.group_seq += 1;
                if let Some(prev) = q.terms.last_mut() {
                    prev.group = Some(self.group_seq);
                }
                q.terms.push(QueryTerm {
                    text: text.to_string(),
                    phrase,
                    negated,
                    group: Some(self.group_seq),
                });
            }
        } else {
            q.terms.push(QueryTerm {
                text: text.to_string(),
                phrase,
                negated,
                group: None,
            });
        }
        self.or_pending = false;
        self.last_was_term = true;
    }
}

const ALL_MODES: &[(&str, &str)] = &[
    ("allintitle", "inTitle"),
    ("allinurl", "inUrl"),
    ("allintext", "inText"),
];

// ─── Anthropic provider ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
struct AnthropicApiResponse {
    id: String,
    model: String,
    content: Vec<AnthropicContentBlock>,
    usage: AnthropicUsage,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    ServerToolUse {
        name: Option<String>,
        input: Option<AnthropicToolInput>,
    },
    WebSearchToolResult {
        content: Vec<serde_json::Value>,
    },
    Text {
        text: Option<String>,
        citations: Option<Vec<AnthropicCitation>>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicToolInput {
    query: String,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicCitation {
    url: String,
    title: String,
    #[serde(rename = "cited_text")]
    cited_text: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicUsage {
    #[serde(rename = "input_tokens")]
    input_tokens: Option<u32>,
    #[serde(rename = "output_tokens")]
    output_tokens: Option<u32>,
    #[serde(rename = "server_tool_use")]
    server_tool_use: Option<AnthropicServerToolUse>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicServerToolUse {
    #[serde(rename = "web_search_requests")]
    web_search_requests: Option<u32>,
}

/// Call Anthropic's Claude `web_search_20250305` tool.
async fn search_anthropic(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    plan: &AnthropicQueryPlan,
    signal: &AbortSignal,
) -> Result<AnthropicApiResponse> {
    let url = format!("{base_url}/v1/messages?beta=true");

    let tool_obj = {
        let mut t = serde_json::Map::new();
        t.insert("type".to_string(), json!(WEB_SEARCH_TOOL_TYPE));
        t.insert("name".to_string(), json!(WEB_SEARCH_TOOL_NAME));
        if let Some(domains) = &plan.allowed_domains {
            t.insert("allowed_domains".to_string(), json!(domains));
        }
        if let Some(domains) = &plan.blocked_domains {
            t.insert("blocked_domains".to_string(), json!(domains));
        }
        serde_json::Value::Object(t)
    };

    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), json!(model));
    body.insert("max_tokens".to_string(), json!(DEFAULT_MAX_TOKENS));
    body.insert(
        "messages".to_string(),
        json!([{"role": "user", "content": &plan.query}]),
    );
    body.insert("tools".to_string(), json!([tool_obj]));
    body.insert("system".to_string(), json!(SYSTEM_PROMPT));

    let body_val = serde_json::Value::Object(body);

    let response = tokio::select! {
        result = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("anthropic-beta", ANTHROPIC_BETA)
            .json(&body_val)
            .send() => result.map_err(|e| anyhow!("HTTP request failed: {e}"))?,
        () = signal.aborted() => {
            bail!("Web search was aborted");
        }
    };

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        bail!("Anthropic API error ({status}): {text}");
    }

    let api_response: AnthropicApiResponse = response
        .json()
        .await
        .map_err(|e| anyhow!("Failed to parse Anthropic response: {e}"))?;

    Ok(api_response)
}

/// Convert an Anthropic API response into a `SearchResponse`.
#[allow(clippy::too_many_lines)] // response fold: one pass over content blocks mirroring omp's parseResponse
fn parse_response(api_response: &AnthropicApiResponse) -> SearchResponse {
    let mut answer_parts: Vec<String> = Vec::new();
    let mut search_queries: Vec<String> = Vec::new();
    let mut sources: Vec<SearchSource> = Vec::new();
    let mut citations: Vec<SearchCitation> = Vec::new();

    for block in &api_response.content {
        match block {
            AnthropicContentBlock::ServerToolUse { name, input } => {
                let nm = name.as_deref().unwrap_or("");
                if nm == WEB_SEARCH_TOOL_NAME || nm.ends_with(&WEB_SEARCH_TOOL_NAME.to_string()) {
                    if let Some(inp) = input {
                        if !inp.query.is_empty() {
                            search_queries.push(inp.query.clone());
                        }
                    }
                }
            }
            AnthropicContentBlock::WebSearchToolResult { content } => {
                for item in content {
                    if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                        if item_type == "web_search_result" {
                            let title = item
                                .get("title")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let url = item
                                .get("url")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let page_age = item
                                .get("page_age")
                                .and_then(|v| v.as_str())
                                .map(std::string::ToString::to_string);
                            let age_seconds = page_age.as_deref().and_then(parse_page_age);
                            sources.push(SearchSource {
                                title,
                                url,
                                snippet: None,
                                published_date: page_age,
                                age_seconds,
                                author: None,
                            });
                        }
                    }
                }
            }
            AnthropicContentBlock::Text {
                text,
                citations: text_citations,
            } => {
                if let Some(t) = text {
                    if !t.is_empty() {
                        answer_parts.push(t.clone());
                    }
                }
                if let Some(tc) = text_citations {
                    for c in tc {
                        citations.push(SearchCitation {
                            url: c.url.clone(),
                            title: c.title.clone(),
                            cited_text: c.cited_text.clone(),
                        });
                    }
                }
            }
            AnthropicContentBlock::Other => {}
        }
    }

    SearchResponse {
        provider: "anthropic".to_string(),
        answer: if answer_parts.is_empty() {
            None
        } else {
            Some(answer_parts.join("\n\n"))
        },
        sources,
        citations: if citations.is_empty() {
            None
        } else {
            Some(citations)
        },
        search_queries: if search_queries.is_empty() {
            None
        } else {
            Some(search_queries)
        },
        related_questions: None,
        usage: Some(SearchUsage {
            input_tokens: api_response.usage.input_tokens,
            output_tokens: api_response.usage.output_tokens,
            search_requests: api_response
                .usage
                .server_tool_use
                .as_ref()
                .and_then(|s| s.web_search_requests),
            total_tokens: None,
        }),
        model: Some(api_response.model.clone()),
        request_id: Some(api_response.id.clone()),
        auth_mode: None,
    }
}

// ─── Formatting ─────────────────────────────────────────────────────────────

fn pluralize(label: &str, count: usize) -> String {
    if count == 1 {
        label.to_string()
    } else if label.ends_with('s')
        || label.ends_with('x')
        || label.ends_with('z')
        || label.ends_with("ch")
        || label.ends_with("sh")
    {
        format!("{label}es")
    } else if label.ends_with('y') && !label.ends_with("ay") && !label.ends_with("ey") {
        let stem = &label[..label.len() - 1];
        format!("{stem}ies")
    } else {
        format!("{label}s")
    }
}

fn format_count(label: &str, count: usize) -> String {
    format!("{count} {}", pluralize(label, count))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // caller passes finite positive ages; truncation toward zero matches TS
fn format_age(age_seconds: f64) -> String {
    if age_seconds <= 0.0 {
        return String::new();
    }
    let total = age_seconds as u64;
    let mins = total / 60;
    let hours = mins / 60;
    let days = hours / 24;
    let weeks = days / 7;
    let months = days / 30;

    if months > 0 {
        format!("{months}mo ago")
    } else if weeks > 0 {
        format!("{weeks}w ago")
    } else if days > 0 {
        format!("{days}d ago")
    } else if hours > 0 {
        format!("{hours}h ago")
    } else if mins > 0 {
        format!("{mins}m ago")
    } else {
        "just now".to_string()
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    let slice_len = max_len.saturating_sub(1);
    let truncated: String = s.chars().take(slice_len).collect();
    format!("{truncated}…")
}

fn has_renderable_search_content(response: &SearchResponse) -> bool {
    response
        .answer
        .as_ref()
        .is_some_and(|s| !s.trim().is_empty())
        || !response.sources.is_empty()
        || response.citations.as_ref().is_some_and(|c| !c.is_empty())
        || response
            .related_questions
            .as_ref()
            .is_some_and(|q| q.iter().any(|s| !s.trim().is_empty()))
        || response
            .search_queries
            .as_ref()
            .is_some_and(|q| q.iter().any(|s| !s.trim().is_empty()))
}

fn format_for_llm(response: &SearchResponse, notes: &[String]) -> String {
    let mut parts: Vec<String> = Vec::new();

    for note in notes {
        parts.push(format!("Note: {note}"));
    }

    if let Some(answer) = &response.answer {
        parts.push(answer.clone());
        if !response.sources.is_empty() {
            parts.push("\n## Sources".to_string());
            parts.push(format_count("source", response.sources.len()));
        }
    }

    for (i, src) in response.sources.iter().enumerate() {
        let age_str = src.age_seconds.map(format_age).unwrap_or_default();
        let age = if age_str.is_empty() {
            src.published_date.clone().unwrap_or_default()
        } else {
            age_str
        };
        let age_part = if age.is_empty() {
            String::new()
        } else {
            format!(" ({age})")
        };
        parts.push(format!(
            "[{}] {}{}\n    {}",
            i + 1,
            src.title,
            age_part,
            src.url
        ));
        if let Some(snippet) = &src.snippet {
            parts.push(format!("    {}", truncate(snippet, 240)));
        }
    }

    if let Some(citations) = &response.citations {
        if !citations.is_empty() {
            parts.push("\n## Citations".to_string());
            parts.push(format_count("citation", citations.len()));
            for (i, citation) in citations.iter().enumerate() {
                let title = if citation.title.is_empty() {
                    citation.url.as_str()
                } else {
                    citation.title.as_str()
                };
                parts.push(format!("[{}] {}\n    {}", i + 1, title, citation.url));
                if let Some(cited_text) = &citation.cited_text {
                    parts.push(format!("    {}", truncate(cited_text, 240)));
                }
            }
        }
    }

    if let Some(questions) = &response.related_questions {
        if !questions.is_empty() {
            parts.push("\n## Related".to_string());
            parts.push(format_count("question", questions.len()));
            for q in questions {
                parts.push(format!("- {q}"));
            }
        }
    }

    if let Some(sqs) = &response.search_queries {
        if !sqs.is_empty() {
            parts.push(format!("Search queries: {}", sqs.len()));
            for query in sqs.iter().take(3) {
                parts.push(format!("- {}", truncate(query, 120)));
            }
        }
    }

    parts.join("\n")
}

// ─── Tool struct ────────────────────────────────────────────────────────────

/// Web search tool — Anthropic provider only (user's configured model chain).
pub struct WebSearchTool {
    schema: Value,
}

impl AgentTool for WebSearchTool {
    fn name(&self) -> &'static str {
        "web_search"
    }
    fn description(&self) -> &str {
        DESCRIPTION
    }
    fn parameters(&self) -> &Value {
        &self.schema
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        params: Value,
        signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move { web_search(params, signal).await })
    }
}

/// Register the `web_search` tool (single-line registration in `builtin_tools`).
#[must_use]
pub fn tool() -> Arc<dyn AgentTool> {
    Arc::new(WebSearchTool { schema: schema() })
}

// ─── Tool logic ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct WebSearchParams {
    query: String,
    #[serde(default)]
    recency: Option<String>,
    #[serde(default)]
    limit: Option<u32>,
}

/// Convert recency string to an `after:` ISO date (days ago).
fn recency_to_after(recency: &str) -> Option<String> {
    let secs = match recency {
        "day" => 86400u64,
        "week" => 7 * 86400,
        "month" => 30 * 86400,
        "year" => 365 * 86400,
        _ => return None,
    };
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;
    let then = now.as_secs().saturating_sub(secs);
    Some(epoch_secs_to_iso_date(then))
}

/// Execute a web search through Anthropic's Claude `web_search_20250305` tool.
async fn web_search(params: Value, signal: AbortSignal) -> Result<AgentToolResult> {
    let parsed: WebSearchParams = serde_json::from_value(params.clone())
        .map_err(|e| anyhow!("Invalid web_search arguments: {e}"))?;

    if parsed.query.is_empty() {
        let resp = SearchResponse {
            provider: "anthropic".to_string(),
            answer: Some("Error: query is empty.".to_string()),
            ..Default::default()
        };
        return Ok(AgentToolResult {
            content: vec![ToolResultContent::text(format_for_llm(&resp, &[]))],
            details: serde_json::to_value(&resp)?,
            terminate: None,
            is_error: None,
        });
    }

    let api_key = std::env::var("ANTHROPIC_SEARCH_API_KEY")
        .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
        .map_err(|_| {
            anyhow!("Anthropic credentials not found. Set ANTHROPIC_SEARCH_API_KEY or ANTHROPIC_API_KEY.")
        })?;

    let base_url =
        std::env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());

    let parsed_query = parse_search_query(&parsed.query);
    let mut parsed_q = parsed_query;

    // Apply recency as an after: date filter.
    if let Some(recency) = &parsed.recency {
        if let Some(date) = recency_to_after(recency) {
            parsed_q.after = Some(date);
            parsed_q.has_constraints = true;
            parsed_q.has_directives = true;
        }
    }

    let plan = plan_query(&parsed.query, &parsed_q);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
        .build()
        .map_err(|e| anyhow!("Failed to build HTTP client: {e}"))?;

    let api_response = search_anthropic(
        &client,
        &base_url,
        &api_key,
        "claude-opus-5-5",
        &plan,
        &signal,
    )
    .await?;

    let mut response = parse_response(&api_response);

    // Apply constraint post-filter.
    let mut constraint_notes: Vec<String> = Vec::new();
    if parsed_q.has_constraints && !response.sources.is_empty() {
        let filtered = apply_query_constraints(&response.sources, &parsed_q);
        if filtered.sources.len() != response.sources.len() {
            response.sources = filtered.sources;
        }
        for label in &filtered.dropped {
            constraint_notes.push(format!(
                "no results matched `{label}`; the constraint was relaxed"
            ));
        }
    }

    // Limit sources.
    if let Some(n) = parsed.limit {
        if response.sources.len() > n as usize {
            response.sources.truncate(n as usize);
        }
    }

    if !has_renderable_search_content(&response) {
        let mut error_resp = response;
        error_resp.answer =
            Some("Error: Anthropic returned no renderable search content.".to_string());
        return Ok(AgentToolResult {
            content: vec![ToolResultContent::text(format_for_llm(&error_resp, &[]))],
            details: serde_json::to_value(&error_resp)?,
            terminate: None,
            is_error: None,
        });
    }

    let text = format_for_llm(&response, &constraint_notes);
    let details = serde_json::to_value(&response).unwrap_or(serde_json::Value::Null);

    Ok(AgentToolResult {
        content: vec![ToolResultContent::text(text)],
        details,
        terminate: None,
        is_error: None,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn schema_matches_ts_reference() {
        let s = schema();
        let props = s["properties"].as_object().unwrap();
        assert_eq!(props.len(), 3);
        assert_eq!(props["limit"]["type"], "number");
        assert_eq!(props["query"]["type"], "string");
        assert_eq!(
            props["recency"],
            json!({"type": "string", "enum": ["day", "week", "month", "year"]})
        );
        assert_eq!(s["required"], json!(["query"]));
        assert_eq!(
            s["properties"]["recency"],
            json!({"type": "string", "enum": ["day", "week", "month", "year"]})
        );
    }

    #[test]
    fn description_first_line_matches_ts() {
        assert_eq!(
            DESCRIPTION.lines().next().unwrap(),
            "Known URLs/programmatic data → `read`. Query: site: or -site:, after: or before: YYYY-MM-DD, inurl:, intitle:, filetype:, \"phrase\", -term, OR. Prefer primary sources; MUST link citations."
        );
    }

    #[test]
    fn parse_site_directive_maps_to_allowed_domains() {
        let parsed = parse_search_query("site:rust-lang.org rust programming");
        assert_eq!(parsed.sites, vec!["rust-lang.org".to_string()]);
        assert!(parsed.has_directives);
        assert!(parsed.has_constraints);
        assert_eq!(parsed.text, "rust programming");
    }

    #[test]
    fn parse_negated_site_maps_to_excluded() {
        let parsed = parse_search_query("-site:github.com rust tutorial");
        assert_eq!(parsed.excluded_sites, vec!["github.com".to_string()]);
        assert_eq!(parsed.text, "rust tutorial");
    }

    #[test]
    fn parse_phrase_and_exclusion() {
        let parsed = parse_search_query("\"exact phrase\" -excluded term");
        assert!(parsed.terms[0].phrase);
        assert_eq!(parsed.terms[0].text, "exact phrase");
        assert!(parsed.terms[1].negated);
        assert_eq!(parsed.terms[1].text, "excluded");
        assert_eq!(parsed.terms[2].text, "term");
        assert!(parsed.has_directives);
    }

    #[test]
    fn parse_or_group() {
        let parsed = parse_search_query("react OR vue svelte");
        // react and vue should be in the same group, svelte separate
        assert!(parsed.terms[0].group.is_some());
        assert_eq!(parsed.terms[0].group, parsed.terms[1].group);
        assert!(parsed.terms[2].group.is_none());
        assert!(parsed.has_directives);
    }

    #[test]
    fn parse_date_range() {
        let parsed = parse_search_query("after:2024-01-01 before:2024-12-31 rust");
        assert_eq!(parsed.after, Some("2024-01-01".to_string()));
        assert_eq!(parsed.before, Some("2024-12-31".to_string()));
        assert_eq!(parsed.text, "rust");
    }

    #[test]
    fn plan_query_maps_site_to_allowed_domains() {
        let parsed = parse_search_query("site:rust-lang.org rust");
        let plan = plan_query("site:rust-lang.org rust", &parsed);
        assert_eq!(
            plan.allowed_domains,
            Some(vec!["rust-lang.org".to_string()])
        );
        assert_eq!(plan.blocked_domains, None);
        assert_eq!(plan.query, "rust");
    }

    #[test]
    fn format_for_llm_renders_sources() {
        let resp = SearchResponse {
            provider: "anthropic".to_string(),
            answer: Some("Rust is a systems language.".to_string()),
            sources: vec![SearchSource {
                title: "Rust Docs".to_string(),
                url: "https://doc.rust-lang.org/".to_string(),
                snippet: None,
                published_date: Some("2 days ago".to_string()),
                age_seconds: Some(172_800.0),
                author: None,
            }],
            citations: None,
            search_queries: None,
            related_questions: None,
            usage: None,
            model: None,
            request_id: None,
            auth_mode: None,
        };
        let text = format_for_llm(&resp, &[]);
        assert!(text.contains("Rust Docs"));
        assert!(text.contains("2d ago"));
        assert!(text.contains("https://doc.rust-lang.org/"));
        assert!(text.contains("# Sources"));
    }

    #[test]
    fn parse_page_age_parses_human_string() {
        assert_eq!(parse_page_age("2 days ago"), Some(172_800.0));
        assert_eq!(parse_page_age("3h ago"), Some(10800.0));
        assert!(parse_page_age("not a date").is_none());
    }

    #[tokio::test]
    async fn mock_anthropic_response_parses() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");

        let mock_json = r#"{
            "id": "msg_abc123",
            "model": "claude-opus-5-5",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "tool_0",
                    "name": "web_search",
                    "input": {"query": "rust programming language"}
                },
                {
                    "type": "web_search_tool_result",
                    "content": [
                        {
                            "type": "web_search_result",
                            "title": "Rust Programming Language",
                            "url": "https://www.rust-lang.org/",
                            "encrypted_content": "ignored",
                            "page_age": "3d ago"
                        }
                    ]
                },
                {
                    "type": "text",
                    "text": "Rust is a systems programming language.",
                    "citations": [
                        {
                            "type": "web_search_result_location",
                            "url": "https://www.rust-lang.org/",
                            "title": "Rust Programming Language",
                            "cited_text": "systems programming language",
                            "encrypted_index": "1"
                        }
                    ]
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 200,
                "server_tool_use": {"web_search_requests": 1}
            }
        }"#;

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 8192];
            let _ = socket.read(&mut buf).await.unwrap();
            let body = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                mock_json.len(),
                mock_json
            );
            socket.write_all(body.as_bytes()).await.unwrap();
        });

        let client = reqwest::Client::new();
        let plan = AnthropicQueryPlan {
            query: "rust programming language".to_string(),
            allowed_domains: None,
            blocked_domains: None,
        };
        let controller = pa_agent::abort::AbortController::new();

        let api_response = search_anthropic(
            &client,
            &base_url,
            "test-key",
            "claude-opus-5-5",
            &plan,
            &controller.signal(),
        )
        .await
        .unwrap();

        let resp = parse_response(&api_response);
        assert_eq!(resp.provider, "anthropic");
        assert_eq!(resp.model, Some("claude-opus-5-5".to_string()));
        assert_eq!(resp.request_id, Some("msg_abc123".to_string()));
        assert_eq!(resp.sources.len(), 1);
        assert_eq!(resp.sources[0].title, "Rust Programming Language");
        assert_eq!(resp.sources[0].url, "https://www.rust-lang.org/");
        assert!(resp.sources[0].age_seconds.is_some());
        assert_eq!(
            resp.answer,
            Some("Rust is a systems programming language.".to_string())
        );
        assert_eq!(
            resp.search_queries.unwrap(),
            vec!["rust programming language".to_string()]
        );
        assert_eq!(resp.citations.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn mock_anthropic_error_status() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 8192];
            let _ = socket.read(&mut buf).await.unwrap();
            let body = "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n";
            socket.write_all(body.as_bytes()).await.unwrap();
        });

        let client = reqwest::Client::new();
        let plan = AnthropicQueryPlan {
            query: "test".to_string(),
            allowed_domains: None,
            blocked_domains: None,
        };
        let controller = pa_agent::abort::AbortController::new();

        let result = search_anthropic(
            &client,
            &base_url,
            "bad-key",
            "claude-opus-5-5",
            &plan,
            &controller.signal(),
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("401"));
    }
}
