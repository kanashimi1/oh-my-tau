//! Relevance-based inline view for spilled `bash`/`read` output.
//!
//! When a result exceeds `tools.artifactSpillThreshold`, the inline text is
//! chosen by BM25 over 5-unit chunks against the request (latest user message
//! plus the assistant text that issued the call) instead of a blind head+tail
//! window; the full text still goes to `artifact://N`. Policy and constants
//! (`bm25_keep`, `keep_hybrid`):
//! `bash` = BM25 5-line chunks, `read` = head 1/9 + tail 2/9 of the budget,
//! then BM25. `OMP_TRUNC_BM25=0` restores the previous head/tail behaviour.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;

use parking_lot::Mutex;

use pa_agent::agent_loop::BeforeToolCallFn;
use pa_agent::types::{
    AgentMessage, AssistantContent, BeforeToolCallContext, Message, UserContent, UserPart,
};

/// Long lines are split into units of this many bytes (analyze.py `SEG`).
const SEG: usize = 1024;
/// Units per BM25 chunk (analyze.py `CHUNK`).
pub const CHUNK_UNITS: usize = 5;
/// Pending per-call queries kept before the oldest is dropped (calls the
/// loop prepares but never executes, e.g. blocked by a hook).
const MAX_PENDING: usize = 64;

const STOP: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "have", "not", "are", "was", "were",
    "will", "would", "should", "could", "into", "onto", "over", "under", "about", "after",
    "before", "then", "than", "when", "what", "where", "which", "while", "true", "false", "none",
    "null", "self", "return", "import", "export", "const", "let", "var", "function", "def",
    "class", "struct", "impl", "pub", "use", "mod", "type", "enum", "match", "string", "number",
    "object", "array", "value", "values", "error", "errors", "warning", "info", "debug", "test",
    "tests", "main", "data", "file", "files", "path", "line", "lines", "name", "names", "text",
    "content", "result", "results", "output", "input", "default", "config",
];

/// Kill switch: `OMP_TRUNC_BM25=0` disables relevance selection.
#[must_use]
pub fn enabled() -> bool {
    std::env::var("OMP_TRUNC_BM25").map_or(true, |v| v.trim() != "0")
}

static PENDING: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

/// Record the query for tool call `id` (consumed by [`take`] when the tool runs).
pub fn remember(id: &str, query: String) {
    let mut p = PENDING.lock();
    p.retain(|(k, _)| k != id);
    if p.len() >= MAX_PENDING {
        p.remove(0);
    }
    p.push((id.to_owned(), query));
}

/// Take the query recorded for tool call `id`.
#[must_use]
pub fn take(id: &str) -> Option<String> {
    let mut p = PENDING.lock();
    let i = p.iter().position(|(k, _)| k == id)?;
    Some(p.remove(i).1)
}

fn assistant_text(content: &[AssistantContent]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            AssistantContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The request a tool call serves: the latest user message plus the text of
/// the assistant message issuing the call (or, when that has none, the last
/// assistant text before it).
#[must_use]
pub fn query_for(ctx: &BeforeToolCallContext) -> String {
    let mut user = String::new();
    let mut prev_asst = String::new();
    for m in ctx.context.messages.iter().rev() {
        match m {
            AgentMessage::Standard(Message::User(u)) if user.is_empty() => {
                user = match &u.content {
                    UserContent::Text(t) => t.clone(),
                    UserContent::Parts(parts) => parts
                        .iter()
                        .filter_map(|p| match p {
                            UserPart::Text(t) => Some(t.text.as_str()),
                            UserPart::Image(_) => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                };
            }
            AgentMessage::Standard(Message::Assistant(a)) if prev_asst.is_empty() => {
                let t = assistant_text(&a.content);
                if !t.trim().is_empty() {
                    prev_asst = t;
                }
            }
            _ => {}
        }
        if !user.is_empty() && !prev_asst.is_empty() {
            break;
        }
    }
    let own = assistant_text(&ctx.assistant_message.content);
    let asst = if own.trim().is_empty() {
        prev_asst
    } else {
        own
    };
    format!("{user}\n{asst}")
}

/// Wrap the agent's `before_tool_call` so every `bash`/`read` call records
/// its query (see [`query_for`]) before the inner hook runs.
#[must_use]
pub fn wrap_before(inner: Option<BeforeToolCallFn>) -> Option<BeforeToolCallFn> {
    if !enabled() {
        return inner;
    }
    Some(Arc::new(move |ctx: BeforeToolCallContext, signal| {
        if matches!(ctx.tool_call.name.as_str(), "bash" | "read") {
            let q = query_for(&ctx);
            if !q.trim().is_empty() {
                remember(&ctx.tool_call.id, q);
            }
        }
        match &inner {
            Some(f) => f(ctx, signal),
            None => Box::pin(async { Ok(None) }),
        }
    }))
}

/// analyze.py `stem`: split camelCase, lowercase, `[a-z0-9]+` runs longer
/// than one char, minus stop-words.
fn terms(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if cur.len() > 1 && !STOP.contains(&cur.as_str()) {
            out.push(std::mem::take(cur));
        } else {
            cur.clear();
        }
    };
    for c in s.chars() {
        if prev_lower && c.is_ascii_uppercase() {
            flush(&mut cur, &mut out);
        }
        prev_lower = c.is_ascii_lowercase();
        let l = c.to_ascii_lowercase();
        if l.is_ascii_lowercase() || l.is_ascii_digit() {
            cur.push(l);
        } else {
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// Text split into lines, long lines into [`SEG`]-byte units.
struct Units<'a> {
    text: Vec<&'a str>,
    line: Vec<usize>,
}

fn units_of(text: &str) -> Units<'_> {
    let mut u = Units {
        text: Vec::new(),
        line: Vec::new(),
    };
    for (li, line) in text.split('\n').enumerate() {
        let mut rest = line;
        loop {
            if rest.len() <= SEG {
                u.text.push(rest);
                u.line.push(li);
                break;
            }
            let mut cut = SEG;
            while !rest.is_char_boundary(cut) {
                cut -= 1;
            }
            u.text.push(&rest[..cut]);
            u.line.push(li);
            rest = &rest[cut..];
        }
    }
    u
}

/// Unit byte cost (text plus its separator).
fn ubytes(u: &str) -> usize {
    u.len() + 1
}

/// analyze.py `bm25_keep`: units in `pre` are kept first; then chunks of
/// `chunk` units are added best BM25 score first (ties by position) while
/// the kept bytes stay within `budget`. Returns the kept mask and whether
/// any chunk matched the query.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn select(
    units: &[&str],
    query: &str,
    chunk: usize,
    budget: usize,
    pre: &[bool],
) -> (Vec<bool>, bool) {
    let chunk = chunk.max(1);
    let n_units = units.len();
    let ch: Vec<std::ops::Range<usize>> = (0..n_units)
        .step_by(chunk)
        .map(|i| i..(i + chunk).min(n_units))
        .collect();
    let docs: Vec<Vec<String>> = ch
        .iter()
        .map(|r| terms(&units[r.clone()].join("\n")))
        .collect();
    let n = docs.len();
    let avg =
        (docs.iter().map(Vec::len).sum::<usize>() as f64 / n.max(1) as f64).max(f64::MIN_POSITIVE);
    let mut df: HashMap<&str, usize> = HashMap::new();
    for d in &docs {
        let set: HashSet<&str> = d.iter().map(String::as_str).collect();
        for w in set {
            *df.entry(w).or_default() += 1;
        }
    }
    let q: HashSet<String> = terms(query).into_iter().collect();
    let scores: Vec<f64> = docs
        .iter()
        .map(|d| {
            let mut tf: HashMap<&str, usize> = HashMap::new();
            for w in d {
                *tf.entry(w.as_str()).or_default() += 1;
            }
            let norm = 1.2 * (0.25 + 0.75 * d.len() as f64 / avg);
            q.iter()
                .filter_map(|w| {
                    let f = *tf.get(w.as_str())? as f64;
                    let dfw = df[w.as_str()] as f64;
                    let idf = (1.0 + (n as f64 - dfw + 0.5) / (dfw + 0.5)).ln();
                    Some(idf * f * 2.2 / (f + norm))
                })
                .sum()
        })
        .collect();
    let matched = scores.iter().any(|s| *s > 0.0);
    let mut kept = vec![false; n_units];
    let mut used = 0;
    for (i, k) in pre.iter().enumerate().take(n_units) {
        if *k {
            kept[i] = true;
            used += ubytes(units[i]);
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
    for k in order {
        let add: usize = ch[k]
            .clone()
            .filter(|i| !kept[*i])
            .map(|i| ubytes(units[i]))
            .sum();
        if used + add <= budget {
            for i in ch[k].clone() {
                kept[i] = true;
            }
            used += add;
        }
    }
    (kept, matched)
}

/// analyze.py `keep_headtail`: units whose midpoint lies in the first
/// `head` or last `tail` bytes.
#[must_use]
pub fn head_tail_mask(units: &[&str], head: usize, tail: usize) -> Vec<bool> {
    let total: usize = units.iter().map(|u| ubytes(u)).sum();
    let mut off = 0;
    units
        .iter()
        .map(|u| {
            let b = ubytes(u);
            let mid2 = 2 * off + b;
            off += b;
            mid2 < 2 * head || mid2 >= 2 * total.saturating_sub(tail)
        })
        .collect()
}

/// Push the `… [N lines omitted] …` marker for a non-empty gap and clear it.
fn flush_gap(out: &mut String, gap: &mut HashSet<usize>, started: bool) {
    if gap.is_empty() {
        return;
    }
    if started {
        out.push('\n');
    }
    let n = gap.len();
    let _ = write!(
        out,
        "… [{n} line{} omitted] …",
        if n == 1 { "" } else { "s" }
    );
    gap.clear();
}

/// Kept units in original order; each gap becomes `… [N lines omitted] …`.
fn render(u: &Units<'_>, kept: &[bool]) -> (String, usize) {
    let mut out = String::new();
    let mut prev: Option<usize> = None;
    let mut gap: HashSet<usize> = HashSet::new();
    let mut kept_lines: HashSet<usize> = HashSet::new();
    for (i, t) in u.text.iter().enumerate() {
        if !kept[i] {
            gap.insert(u.line[i]);
            continue;
        }
        let had_gap = !gap.is_empty();
        flush_gap(&mut out, &mut gap, prev.is_some());
        let joined = !had_gap && prev.is_some_and(|p| p + 1 == i && u.line[p] == u.line[i]);
        if !joined && (prev.is_some() || had_gap) {
            out.push('\n');
        }
        out.push_str(t);
        kept_lines.insert(u.line[i]);
        prev = Some(i);
    }
    flush_gap(&mut out, &mut gap, prev.is_some());
    (out, kept_lines.len())
}

/// Relevance view of over-budget text.
#[derive(Debug, Clone)]
pub struct Picked {
    pub content: String,
    /// Truncation notice without the artifact suffix.
    pub notice: String,
}

/// Inline view of `text` within `budget` bytes. `hybrid` = head/tail bytes
/// kept before BM25 (`read`); without it (`bash`) a query matching no chunk
/// falls back to the `fallback` head/tail window. `None` when `text` fits.
#[must_use]
pub fn pick(
    text: &str,
    query: &str,
    budget: usize,
    hybrid: Option<(usize, usize)>,
    fallback: (usize, usize),
) -> Option<Picked> {
    let u = units_of(text);
    let total_bytes: usize = u.text.iter().map(|t| ubytes(t)).sum();
    if total_bytes <= budget {
        return None;
    }
    let pre = hybrid.map_or_else(
        || vec![false; u.text.len()],
        |(h, t)| head_tail_mask(&u.text, h, t),
    );
    let (mut kept, matched) = select(&u.text, query, CHUNK_UNITS, budget, &pre);
    let mut policy = if hybrid.is_some() {
        "head/tail + relevance"
    } else {
        "relevance"
    };
    if !matched && hybrid.is_none() {
        let pre = head_tail_mask(&u.text, fallback.0, fallback.1);
        kept = select(&u.text, "", CHUNK_UNITS, budget, &pre).0;
        policy = "head/tail";
    }
    let (content, kept_lines) = render(&u, &kept);
    let total_lines = u.line.last().map_or(0, |l| l + 1);
    let notice = format!(
        "Showing {kept_lines} of {total_lines} lines ({policy} to the request); {} lines omitted",
        total_lines - kept_lines
    );
    Some(Picked { content, notice })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(n: usize, f: impl Fn(usize) -> String) -> String {
        (1..=n).map(f).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn terms_split_camel_case_and_drop_stop_words() {
        assert_eq!(
            terms("parseConfig the HTTPServer foo_bar x 42"),
            ["parse", "httpserver", "foo", "bar", "42"]
        );
    }

    #[test]
    fn selector_respects_budget_keeps_order_and_is_deterministic() {
        let text = lines(400, |i| format!("row {i} filler words alpha beta"));
        let text = text.replace("row 200 filler", "row 200 needleToken filler");
        let u = units_of(&text);
        let budget = 1000;
        let pre = vec![false; u.text.len()];
        let (kept, matched) = select(&u.text, "where is needleToken", CHUNK_UNITS, budget, &pre);
        assert!(matched);
        let used: usize = u
            .text
            .iter()
            .zip(&kept)
            .filter(|(_, k)| **k)
            .map(|(t, _)| ubytes(t))
            .sum();
        assert!(used <= budget, "{used}");
        assert!(kept[199], "needle unit kept");
        // Deterministic.
        assert_eq!(
            select(&u.text, "where is needleToken", CHUNK_UNITS, budget, &pre).0,
            kept
        );
        let p = pick(&text, "where is needleToken", budget, None, (300, 600)).unwrap();
        assert_eq!(
            p.content,
            pick(&text, "where is needleToken", budget, None, (300, 600))
                .unwrap()
                .content
        );
        // Original order: row numbers strictly increase.
        let nums: Vec<usize> = p
            .content
            .lines()
            .filter_map(|l| l.strip_prefix("row ")?.split(' ').next()?.parse().ok())
            .collect();
        assert!(nums.windows(2).all(|w| w[0] < w[1]), "{nums:?}");
        assert!(nums.contains(&200));
        assert!(p.content.contains("… ["), "{}", p.content);
    }

    #[test]
    fn markers_count_omitted_lines() {
        let text = lines(30, |i| format!("l{i}"));
        let mut kept = vec![false; 30];
        kept[0] = true;
        kept[10] = true;
        let u = units_of(&text);
        let (out, n) = render(&u, &kept);
        assert_eq!(
            out,
            "l1\n… [9 lines omitted] …\nl11\n… [19 lines omitted] …"
        );
        assert_eq!(n, 2);
        kept[29] = true;
        kept[28] = true;
        let (out, _) = render(&u, &kept);
        assert!(out.ends_with("… [17 lines omitted] …\nl29\nl30"), "{out}");
    }

    #[test]
    fn long_lines_split_into_units_and_rejoin() {
        let long = "x".repeat(2500);
        let text = format!("a\n{long}\nb");
        let u = units_of(&text);
        assert_eq!(u.text.len(), 5);
        let (out, n) = render(&u, &[true; 5]);
        assert_eq!(out, text);
        assert_eq!(n, 3);
    }

    #[test]
    fn unmatched_query_falls_back_to_head_tail_and_hybrid_keeps_edges() {
        let text = lines(1000, |i| format!("entry {i:04} payload"));
        let p = pick(&text, "zzzunrelated", 2000, None, (600, 1200)).unwrap();
        assert!(p.content.starts_with("entry 0001 payload"));
        assert!(p.content.ends_with("entry 1000 payload"));
        assert!(p.notice.contains("(head/tail"), "{}", p.notice);
        let text = text.replace("entry 0500 payload", "entry 0500 needle");
        let p = pick(&text, "needle", 2000, Some((200, 400)), (0, 0)).unwrap();
        assert!(p.content.starts_with("entry 0001 payload"));
        assert!(p.content.ends_with("entry 1000 payload"));
        assert!(p.content.contains("entry 0500 needle"));
        assert!(pick("short", "q", 2000, None, (1, 1)).is_none());
    }
}
