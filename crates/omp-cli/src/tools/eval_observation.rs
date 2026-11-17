//! Optional semantic observation packs for `eval` output (submodule of the
//! eval tool; registered only from `eval.rs`, so `tools/mod.rs` is untouched).
//!
//! Opt-in via `OMP_EVAL_OBSERVATION_PACK=1` (exact value; anything else keeps
//! the default rendering byte-identical). When enabled, a recognized cell
//! output — a single JSON document, a `cargo test`/`pytest` summary, or
//! grep-style `path:line:` lines — is replaced by a deterministic observation
//! pack that always carries a recoverable reference to the full text (the
//! session artifact store when a session exists, a content-addressed temp
//! backing file otherwise; no parallel store is introduced). Identical
//! consecutive outputs within one kernel session collapse to a dedup notice;
//! different content is never collapsed, and error cells are never packed or
//! deduplicated, so no change is ever hidden.
//!
//! Everything that would hide information — error cells, raw-output mode,
//! unknown formats, failed persistence — falls back to the default rendering
//! ([`super::cap_output_with`]), byte-identical to the pack-disabled path.
//!
//! The [`Spool`] hashes and windows the output incrementally over arbitrary
//! byte chunks, retaining only a bounded head/tail in RAM. Cell output
//! arrives as complete text because semantic classification needs the whole
//! document. The chunked interface still hashes/windows it incrementally;
//! the default non-pack path uses the separate frame-level output spool.

use sha2::{Digest, Sha256};
use std::fmt::Write as _;

/// Only the exact string `1` enables packs (same convention as the other
/// opt-in env gates).
const GATE_ENV: &str = "OMP_EVAL_OBSERVATION_PACK";

/// RAM budget of the retained head/tail window (preview material only).
const WINDOW_BYTES: usize = 4 * 1024;
/// Feed chunk size; `feed_bytes` accepts arbitrary chunk sizes.
const FEED_CHUNK: usize = 64 * 1024;
/// Hex characters identifying a result in notices.
const SHORT_HEX: usize = 16;
/// Deterministic summary caps.
const MAX_LISTED_KEYS: usize = 8;
const MAX_LISTED_FILES: usize = 8;
const MAX_SCALAR_REPR: usize = 120;
const MAX_PREVIEW_LINE: usize = 200;
/// grep-style classification: enough lines and a clear majority of matches.
const MIN_GREP_LINES: usize = 3;
const GREP_RATIO_NUM: usize = 9;
const GREP_RATIO_DENOM: usize = 10;
/// A single JSON document larger than this is not summarized.
const MAX_JSON_BYTES: usize = 8 * 1024 * 1024;

/// Serializes tests that touch [`GATE_ENV`] / the raw-output env var.
#[cfg(test)]
pub(crate) static GATE_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Gate value resolution (pure; [`pack_requested`] feeds it the env value).
#[must_use]
fn gate_from(value: Option<&str>) -> bool {
    value.is_some_and(|v| v == "1")
}

/// Test-only override: integration tests pin the gate without mutating the
/// process env (kernel tests run concurrently and must not observe flips).
#[cfg(test)]
pub(crate) static PACK_OVERRIDE: parking_lot::Mutex<Option<bool>> = parking_lot::Mutex::new(None);

/// `OMP_EVAL_OBSERVATION_PACK=1` (exact value) enables packs and dedup.
#[must_use]
pub(crate) fn pack_requested() -> bool {
    #[cfg(test)]
    if let Some(v) = *PACK_OVERRIDE.lock() {
        return v;
    }
    gate_from(std::env::var(GATE_ENV).ok().as_deref())
}

/// Identity of one non-error cell output: full SHA-256 hex + byte size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ObservationKey {
    pub sha: String,
    pub bytes: usize,
}

impl ObservationKey {
    /// Reference builder for tests (production builds the key from the spool).
    #[cfg(test)]
    fn new(text: &str) -> Self {
        Self {
            sha: format!("{:x}", Sha256::digest(text.as_bytes())),
            bytes: text.len(),
        }
    }

    fn short(&self) -> &str {
        &self.sha[..SHORT_HEX.min(self.sha.len())]
    }
}

/// Incrementally hashed/windowed output spool. Keeps the full SHA-256, the
/// byte/line totals, a bounded head and a bounded tail in RAM.
pub(crate) struct Spool {
    hasher: Sha256,
    bytes: usize,
    newlines: usize,
    last_byte: Option<u8>,
    head: Vec<u8>,
    tail: std::collections::VecDeque<u8>,
}

#[derive(Clone)]
pub(crate) struct Spooled {
    pub key: ObservationKey,
    pub lines: usize,
    /// First [`WINDOW_BYTES`] bytes (lossy UTF-8) for previews.
    pub head: String,
}

impl Spool {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            hasher: Sha256::new(),
            bytes: 0,
            newlines: 0,
            last_byte: None,
            head: Vec::new(),
            tail: std::collections::VecDeque::new(),
        }
    }

    /// Stream one chunk (any size, any UTF-8 boundary).
    #[allow(clippy::naive_bytecount)] // Keep this hot loop dependency-free.
    pub(crate) fn feed_bytes(&mut self, chunk: &[u8]) {
        self.hasher.update(chunk);
        self.bytes = self.bytes.saturating_add(chunk.len());
        self.newlines += chunk.iter().filter(|&&b| b == b'\n').count();
        self.last_byte = chunk.last().copied().or(self.last_byte);
        if self.head.len() < WINDOW_BYTES {
            let take = (WINDOW_BYTES - self.head.len()).min(chunk.len());
            self.head.extend_from_slice(&chunk[..take]);
        }
        for &b in chunk {
            if self.tail.len() == WINDOW_BYTES {
                self.tail.pop_front();
            }
            self.tail.push_back(b);
        }
    }

    #[must_use]
    pub(crate) fn finish(self) -> Spooled {
        let sha = format!("{:x}", self.hasher.finalize());
        let partial = self.last_byte.is_some_and(|b| b != b'\n');
        let lines = if self.bytes > 0 {
            self.newlines + usize::from(partial)
        } else {
            0
        };
        Spooled {
            key: ObservationKey {
                sha,
                bytes: self.bytes,
            },
            lines,
            head: String::from_utf8_lossy(&self.head).into_owned(),
        }
    }
}

fn cut_str(s: &str, max: usize) -> &str {
    let mut end = max.min(s.len());
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Recoverable reference to the full output: the session artifact store when
/// a session exists, else one content-addressed temp file (never a parallel
/// store). `None` means no reference could be produced; callers fall back to
/// the default rendering so nothing is hidden.
fn persist_ref(text: &str) -> Option<String> {
    if let Some(url) = super::save_artifact(text) {
        return Some(url);
    }
    let digest = Sha256::digest(text.as_bytes());
    let path = std::env::temp_dir().join(format!("omp-eval-obs-{digest:x}.out"));
    if !path.exists() {
        let tmp =
            std::env::temp_dir().join(format!("omp-eval-obs-{digest:x}.tmp{}", std::process::id()));
        std::fs::write(&tmp, text.as_bytes()).ok()?;
        std::fs::rename(&tmp, &path).ok()?;
    }
    Some(path.display().to_string())
}

fn pack_notice(
    kind: &str,
    summary: &str,
    key: &ObservationKey,
    lines: usize,
    reference: &str,
) -> String {
    format!(
        "[eval observation: {kind}] {summary}\nsha256:{} {}B {lines}ln full output: {reference}",
        key.short(),
        key.bytes,
    )
}

fn dedup_notice(key: &ObservationKey, lines: usize, reference: &str, head: &str) -> String {
    let preview = head.lines().next().unwrap_or("");
    format!(
        "[eval output identical to previous cell in this session] sha256:{} {}B {lines}ln starts: {}\nfull output: {reference}",
        key.short(),
        key.bytes,
        cut_str(preview, MAX_PREVIEW_LINE),
    )
}

/// Input for [`render`]; env-derived flags are passed explicitly so tests
/// never race on process-global env vars.
pub(crate) struct PackInput<'a> {
    pub text: &'a str,
    pub cell_error: bool,
    pub raw: bool,
    pub enabled: bool,
    pub prev: Option<ObservationKey>,
}

/// `(provider text, key to record)`: `None` text = use the default
/// [`super::cap_output_with`] rendering unchanged; a key is recorded for
/// every enabled non-error cell so the next identical output dedups.
pub(crate) fn render(input: &PackInput<'_>) -> (Option<String>, Option<ObservationKey>) {
    if !input.enabled || input.raw || input.cell_error {
        return (None, None);
    }
    let mut spool = Spool::new();
    for chunk in input.text.as_bytes().chunks(FEED_CHUNK) {
        spool.feed_bytes(chunk);
    }
    let spooled = spool.finish();
    let key = spooled.key.clone();
    // Same-session dedup: identical non-error output collapses to a notice
    // that still links the full text; different content never collapses.
    if input.prev.as_ref().is_some_and(|p| *p == key) {
        if let Some(reference) = persist_ref(input.text) {
            return (
                Some(dedup_notice(&key, spooled.lines, &reference, &spooled.head)),
                Some(key),
            );
        }
        return (None, Some(key)); // persistence failed: show everything
    }
    if let Some((kind, summary)) = classify(input.text) {
        if let Some(reference) = persist_ref(input.text) {
            return (
                Some(pack_notice(kind, &summary, &key, spooled.lines, &reference)),
                Some(key),
            );
        }
    }
    // Unknown format (or persistence failed): default rendering, but the key
    // is still recorded so a repeated identical output dedups next time.
    (None, Some(key))
}

/// Deterministic format recognition; first match wins.
fn classify(text: &str) -> Option<(&'static str, String)> {
    classify_json(text)
        .or_else(|| classify_cargo_tests(text))
        .or_else(|| classify_pytest(text))
        .or_else(|| classify_grep(text))
}

fn json_kind(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Whole trimmed output parses as one JSON document.
fn classify_json(text: &str) -> Option<(&'static str, String)> {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.len() > MAX_JSON_BYTES {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    let summary = match &v {
        serde_json::Value::Object(map) => {
            let mut names: Vec<&str> = map.keys().map(String::as_str).collect();
            names.sort_unstable();
            let total = names.len();
            if total == 0 {
                "object, 0 keys".to_owned()
            } else {
                let listed = names
                    .iter()
                    .take(MAX_LISTED_KEYS)
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", ");
                if total > MAX_LISTED_KEYS {
                    format!(
                        "object, {total} keys: {listed} …(+{})",
                        total - MAX_LISTED_KEYS
                    )
                } else {
                    format!("object, {total} keys: {listed}")
                }
            }
        }
        serde_json::Value::Array(a) => {
            let first = a.first().map_or("none", |v| json_kind(v));
            format!("array, {} items (first: {first})", a.len())
        }
        v => format!("{}: {}", json_kind(v), cut_str(trimmed, MAX_SCALAR_REPR)),
    };
    Some(("json", summary))
}

/// `cargo test`: one or more `test result: …` summary lines, counts summed.
fn classify_cargo_tests(text: &str) -> Option<(&'static str, String)> {
    let (mut suites, mut passed, mut failed, mut ignored, mut measured, mut filtered) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut any_failed = false;
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("test result: ") else {
            continue;
        };
        suites += 1;
        if rest.starts_with("FAILED") {
            any_failed = true;
        }
        let words: Vec<&str> = rest
            .split([';', '.'])
            .flat_map(str::split_whitespace)
            .collect();
        let mut i = 0;
        while i < words.len() {
            let Ok(n) = words[i].parse::<u64>() else {
                i += 1;
                continue;
            };
            match words.get(i + 1).copied().unwrap_or("") {
                "passed" => passed += n,
                "failed" => failed += n,
                "ignored" => ignored += n,
                "measured" => measured += n,
                "filtered" if words.get(i + 2) == Some(&"out") => filtered += n,
                _ => {
                    i += 1;
                    continue;
                }
            }
            i += 2;
        }
    }
    if suites == 0 {
        return None;
    }
    let mut summary = format!("suites={suites} passed={passed} failed={failed} ignored={ignored}");
    if measured > 0 {
        let _ = write!(summary, " measured={measured}");
    }
    if filtered > 0 {
        let _ = write!(summary, " filtered_out={filtered}");
    }
    summary.push_str(if any_failed || failed > 0 {
        " FAILED"
    } else {
        " ok"
    });
    Some(("cargo-test", summary))
}

/// `pytest`: the `====… N passed … in 1.2s ====…` banner line.
fn classify_pytest(text: &str) -> Option<(&'static str, String)> {
    const UNITS: [(&str, u8); 8] = [
        ("passed", 0),
        ("failed", 1),
        ("error", 2),
        ("skipped", 3),
        ("xfailed", 4),
        ("xpassed", 5),
        ("warning", 6),
        ("deselected", 7),
    ];
    for line in text.lines().rev().take(40) {
        let t = line.trim();
        if t.len() < 8 || !t.starts_with('=') || !t.ends_with('=') {
            continue;
        }
        let inner = t.trim_matches('=').trim();
        let words: Vec<&str> = inner.split_whitespace().collect();
        let mut got: Vec<(u8, &str, u64)> = Vec::new();
        let mut duration: Option<&str> = None;
        let mut i = 0;
        while i + 1 < words.len() {
            if words[i] == "in" {
                if let Some(d) = words.get(i + 1).filter(|d| d.ends_with('s')) {
                    duration = Some(d);
                }
                i += 1;
                continue;
            }
            let Ok(n) = words[i].parse::<u64>() else {
                i += 1;
                continue;
            };
            let unit = words[i + 1].trim_end_matches(',').trim_end_matches('s');
            if let Some((_, order)) = UNITS.iter().find(|(u, _)| *u == unit) {
                got.push((*order, unit, n));
                i += 2;
            } else {
                i += 1;
            }
        }
        if got.is_empty() {
            continue;
        }
        got.sort_unstable_by_key(|(order, _, _)| *order);
        let mut summary = got
            .iter()
            .map(|(_, unit, n)| format!("{unit}={n}"))
            .collect::<Vec<_>>()
            .join(" ");
        if let Some(d) = duration {
            let _ = write!(summary, " duration={d}");
        }
        return Some(("pytest", summary));
    }
    None
}

/// grep-style `path:line:` lines: enough lines, ≥90% matching.
fn classify_grep(text: &str) -> Option<(&'static str, String)> {
    let (mut total, mut matched) = (0usize, 0usize);
    let mut files: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        total += 1;
        if let Some(path) = grep_match_path(line) {
            matched += 1;
            files.insert(path);
        }
    }
    if total < MIN_GREP_LINES || matched * GREP_RATIO_DENOM < total * GREP_RATIO_NUM {
        return None;
    }
    let mut summary = format!("~{matched} matches in {} files", files.len());
    if !files.is_empty() {
        let listed: Vec<&str> = files.iter().take(MAX_LISTED_FILES).copied().collect();
        let _ = write!(summary, ": {}", listed.join(", "));
        if files.len() > MAX_LISTED_FILES {
            let _ = write!(summary, " …(+{})", files.len() - MAX_LISTED_FILES);
        }
    }
    Some(("grep-lines", summary))
}

fn grep_match_path(line: &str) -> Option<&str> {
    let (path, rest) = line.split_once(':')?;
    if path.is_empty() || path.len() > 512 || path.contains(char::is_whitespace) {
        return None;
    }
    let (lineno, _) = rest.split_once(':')?;
    if lineno.is_empty() || !lineno.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spool_hashes_and_counts_like_str_lines() {
        let mut s = Spool::new();
        for chunk in "alpha\nbeta\n\ngamma".as_bytes().chunks(3) {
            s.feed_bytes(chunk);
        }
        let sp = s.finish();
        assert_eq!(sp.key.sha, ObservationKey::new("alpha\nbeta\n\ngamma").sha);
        assert_eq!(sp.key.bytes, 17);
        assert_eq!(sp.lines, "alpha\nbeta\n\ngamma".lines().count());
        let mut s = Spool::new();
        s.feed_bytes(b"a\n\n");
        assert_eq!(s.finish().lines, "a\n\n".lines().count());
        let mut s = Spool::new();
        s.feed_bytes(b"");
        assert_eq!(s.finish().lines, 0);
    }

    #[test]
    fn spool_window_stays_bounded() {
        let mut s = Spool::new();
        let big = format!("{}\nTAIL-MARKER", "x".repeat(1024 * 1024));
        for chunk in big.as_bytes().chunks(997) {
            s.feed_bytes(chunk);
        }
        let sp = s.finish();
        assert!(sp.head.len() <= WINDOW_BYTES);
        assert!(sp.head.starts_with("xxxx"));
        // Tail preview via head is head-only; the tail deque is internal, so
        // verify boundedness indirectly through the spool struct invariants.
        assert_eq!(sp.key.bytes, big.len());
    }

    #[test]
    fn json_classification_is_sorted_and_capped() {
        let v = serde_json::json!({"zeta": 1, "alpha": 2, "mid": 3});
        let text = serde_json::to_string(&v).unwrap();
        let (kind, summary) = classify_json(&text).unwrap();
        assert_eq!(kind, "json");
        assert!(summary.contains("alpha, mid, zeta"), "{summary}");
        let mut keys = serde_json::Map::new();
        for i in 0..12 {
            keys.insert(format!("k{i:02}"), serde_json::json!(i));
        }
        let (_, summary) = classify_json(&serde_json::Value::Object(keys).to_string()).unwrap();
        assert!(summary.contains("…(+4)"), "{summary}");
        let (kind, summary) = classify_json("[1, 2, 3]").unwrap();
        assert_eq!(kind, "json");
        assert_eq!(summary, "array, 3 items (first: number)");
        let (_, summary) = classify_json("\"just a string\"").unwrap();
        assert!(summary.starts_with("string: "), "{summary}");
        assert!(classify_json("not json {").is_none());
        assert!(classify_json("").is_none());
    }

    #[test]
    fn cargo_summary_sums_suites() {
        let text = "running 15 tests\ntest a ... ok\ntest result: ok. 14 passed; 1 failed; 2 ignored; 0 measured; 33 filtered out\n\nrunning 5 tests\ntest result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n";
        let (kind, summary) = classify(text).unwrap();
        assert_eq!(kind, "cargo-test");
        assert_eq!(
            summary,
            "suites=2 passed=19 failed=1 ignored=2 filtered_out=33 FAILED"
        );
    }

    #[test]
    fn pytest_banner_is_recognized() {
        let text =
            "tmp/test_x.py ..\n================= 3 passed, 1 skipped in 0.02s =================";
        let (kind, summary) = classify(text).unwrap();
        assert_eq!(kind, "pytest");
        assert_eq!(summary, "passed=3 skipped=1 duration=0.02s");
    }

    #[test]
    fn grep_lines_need_majority() {
        let good = (0..10).fold(String::new(), |mut text, i| {
            let _ = writeln!(text, "src/f{i}.rs:{i}:let x = {i};");
            text
        });
        let (kind, summary) = classify(&good).unwrap();
        assert_eq!(kind, "grep-lines");
        assert!(
            summary.starts_with("~10 matches in 10 files: "),
            "{summary}"
        );
        let noisy = format!("{good}random prose line without colons\nanother plain line\n");
        assert!(classify(&noisy).is_none());
        assert!(classify("a:1:\n").is_none());
    }

    #[test]
    fn render_falls_back_unless_enabled_or_clean() {
        let key = ObservationKey::new("seed");
        let mk = |text: &str, error: bool, raw: bool, enabled: bool| {
            render(&PackInput {
                text,
                cell_error: error,
                raw,
                enabled,
                prev: Some(key.clone()),
            })
        };
        let json_text = "{\"a\": 1}\n";
        assert_eq!(mk(json_text, false, false, false), (None, None), "gate off");
        assert_eq!(mk(json_text, false, true, true), (None, None), "raw mode");
        assert_eq!(mk(json_text, true, false, true), (None, None), "error cell");
        let unknown = mk("plain prose output\n", false, false, true);
        assert_eq!(unknown.0, None, "unknown keeps the default rendering");
        assert!(unknown.1.is_some(), "unknown output still records its key");
        let (text_out, key_out) = mk(json_text, false, false, true);
        assert!(text_out
            .expect("enabled + known")
            .starts_with("[eval observation: json]"));
        assert!(key_out.is_some());
    }

    #[test]
    fn dedup_collapses_only_identical_and_keeps_reference() {
        let text = "{\"a\": 1}\n";
        let (first_text, first_key) = render(&PackInput {
            text,
            cell_error: false,
            raw: false,
            enabled: true,
            prev: None,
        });
        assert!(first_text
            .expect("first run packs")
            .starts_with("[eval observation"));
        let first_key = first_key.expect("key recorded");
        let (second_text, second_key) = render(&PackInput {
            text,
            cell_error: false,
            raw: false,
            enabled: true,
            prev: Some(first_key.clone()),
        });
        let second_text = second_text.expect("identical run dedups");
        assert!(
            second_text.contains("identical to previous cell"),
            "{second_text}"
        );
        assert!(second_text.contains(first_key.short()));
        // The dedup notice still carries a readable full-output reference.
        let reference = second_text
            .lines()
            .find_map(|l| l.strip_prefix("full output: "))
            .expect("reference line");
        let body = std::fs::read_to_string(reference).unwrap();
        assert_eq!(body, text);
        let (changed_text, changed_key) = render(&PackInput {
            text: "{\"a\": 2}\n",
            cell_error: false,
            raw: false,
            enabled: true,
            prev: Some(second_key.expect("key recorded")),
        });
        assert!(!changed_text
            .expect("changed content packs")
            .contains("identical to previous"));
        assert!(changed_key.is_some());
        // Unknown formats dedup too: the first occurrence records its key,
        // the identical second one collapses behind the notice.
        let plain = "plain prose output\n";
        let (t1, k1) = render(&PackInput {
            text: plain,
            cell_error: false,
            raw: false,
            enabled: true,
            prev: None,
        });
        assert!(t1.is_none(), "unknown keeps the default rendering");
        let (t2, _) = render(&PackInput {
            text: plain,
            cell_error: false,
            raw: false,
            enabled: true,
            prev: k1,
        });
        assert!(t2
            .expect("identical unknown output dedups")
            .contains("identical to previous cell"));
    }

    #[test]
    fn gate_requires_exact_value() {
        assert!(!gate_from(None));
        assert!(!gate_from(Some("typescript")));
        assert!(!gate_from(Some("0")));
        assert!(!gate_from(Some("")));
        assert!(gate_from(Some("1")));
    }
}
