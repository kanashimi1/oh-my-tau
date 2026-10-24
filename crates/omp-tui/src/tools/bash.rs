//! `bash` tool renderer — port of `pi-tui/src/tools/bash.ts`
//! (`bashToolRenderer = createShellRenderer({..., showHeader: false})`), plus
//! the live `!command` user bash execution card from
//! `pi-coding-agent/src/modes/interactive-mode/command-controller.ts` §4.3
//! (spec: `tools-a.md` §2, `interactive.md` §4.3).
//!
//! Expected `call` payload (superset of the raw tool call; the caller merges
//! in streaming/result fields until the contract grows a dedicated struct):
//!
//! ```json
//! {
//!   "name": "bash",
//!   "arguments": { "command": "...", "cwd": "...", "env": {"K":"V"},
//!                  "timeout": 30, "name": "optional label" },
//!   "__partialJson": "{\"command\": …",          // streamed raw args prefix
//!   "output": "live tail (renderContext.output)",
//!   "result": { "content": [{"type":"text","text":"..."}],
//!               "details": { "wallTimeMs": 1200, "exitCode": 1, … },
//!               "isError": false },
//!   "isPartial": true,
//!   "expanded": false,
//!   "previewLines": 10,
//!   "timeout": 30,
//!   "projectDir": "/repo"                        // for the `cd … &&` prefix
//! }
//! ```
//!
//! Visual (TS parity): framed card, muted border while pending/streaming,
//! error/warning border on failure/timeout. Body = dim `$ cd … && NAME="…"`
//! command lines (tail-capped to a viewport-sized window), an `Output`
//! section whose rows are tail-capped to `min(previewLines, viewport window)`,
//! a dim `⟨Wall: 1.20s | Timeout: 30s | …⟩` stats line (only applicable
//! parts), and a warning-colored `⟨Showing lines …⟩` truncation line.
//!
//! Deviations until sibling modules land: no syntax highlighting of the
//! command (`CodeHighlight` owns `highlightCode`), and output capping is a
//! hard line tail (TS also wraps visually at the box width via
//! `formatOutputPaneLines(visual: true)`).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use serde_json::Value;

use crate::theme::Theme;

use super::frame::{framed_card, CardSection};
use super::{error_detail_lines, ToolRenderer};

/// Default collapsed shell output preview height (TS `BASH_DEFAULT_PREVIEW_LINES`).
pub const BASH_DEFAULT_PREVIEW_LINES: usize = 10;
/// Display tab width (TS `pi-utils/DEFAULT_TAB_WIDTH`).
const TAB_SPACES: &str = "   ";
/// Unicode bracket pair (TS `theme.format.bracketLeft/Right`).
const BRACKET_LEFT: &str = "⟨";
const BRACKET_RIGHT: &str = "⟩";
/// Rows reserved below a collapsed preview window (TS `PREVIEW_WINDOW_RESERVED_ROWS`).
const PREVIEW_WINDOW_RESERVED_ROWS: usize = 20;
/// Floor for tiny/unknown viewports (TS `PREVIEW_WINDOW_MIN_LINES`).
const PREVIEW_WINDOW_MIN_LINES: usize = 6;
/// Assumed viewport when rows are unknown (TS `PREVIEW_WINDOW_FALLBACK_ROWS`).
const PREVIEW_WINDOW_FALLBACK_ROWS: usize = 30;

// ============================================================================
// Details (TS `BashToolDetails`)
// ============================================================================

/// Truncation metadata (TS `output-meta.ts/TruncationMeta`), bash-relevant subset.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TruncationMeta {
    pub direction: String,    // "head" | "tail" | "middle"
    pub truncated_by: String, // "lines" | "bytes" | "middle"
    pub total_lines: u64,
    pub total_bytes: u64,
    pub output_lines: u64,
    pub output_bytes: u64,
    pub max_bytes: Option<u64>,
    /// 1-indexed inclusive `(start, end)`; omitted for middle elision.
    pub shown_range: Option<(u64, u64)>,
    pub head_range: Option<(u64, u64)>,
    pub tail_range: Option<(u64, u64)>,
    pub elided_bytes: Option<u64>,
    pub elided_lines: Option<u64>,
    pub artifact_id: Option<String>,
    pub artifact_elided_bytes: Option<u64>,
    /// Next offset for pagination (head truncation only).
    pub next_offset: Option<u64>,
    /// The single shown line is a byte-capped preview of one oversized line.
    pub partial_line: bool,
}

/// Live service metadata (TS `BashToolDetails.service`).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ServiceDetails {
    pub name: String,
    pub state: String,
    pub ready: bool,
    pub timed_out: bool,
    pub pid: Option<u64>,
}

/// Background job metadata (TS `BashToolDetails.async`).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BackgroundJob {
    /// "running" | "completed" | "failed".
    pub state: String,
    pub job_id: String,
}

/// Shell execution metadata (TS `BashToolDetails`).
#[derive(Debug, Default, Clone)]
pub struct BashDetails {
    pub truncation: Option<TruncationMeta>,
    /// Artifact capture failure description.
    pub artifact_error: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub requested_timeout_seconds: Option<u64>,
    pub timeout_disabled: bool,
    pub wall_time_ms: Option<u64>,
    pub exit_code: Option<i64>,
    pub timed_out: bool,
    pub service: Option<ServiceDetails>,
    pub background: Option<BackgroundJob>,
}

fn vstr(v: &Value) -> Option<String> {
    v.as_str().map(str::to_owned)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // JSON numbers are non-negative here
fn vu64(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_f64().map(|f| f as u64))
}

#[allow(clippy::cast_possible_truncation)] // exit codes fit comfortably in i64
fn vi64(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))
}

fn vbool(v: &Value) -> Option<bool> {
    v.as_bool()
}

fn range_of(v: &Value) -> Option<(u64, u64)> {
    let s = v.get("start").and_then(vu64)?;
    let e = v.get("end").and_then(vu64)?;
    Some((s, e))
}

impl BashDetails {
    /// Parse from a JSON `details` object (unknown/extra keys ignored).
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let truncation = v.get("truncation").map(|t| {
            let mut tm = TruncationMeta {
                total_lines: t.get("totalLines").and_then(vu64).unwrap_or(0),
                total_bytes: t.get("totalBytes").and_then(vu64).unwrap_or(0),
                output_lines: t.get("outputLines").and_then(vu64).unwrap_or(0),
                output_bytes: t.get("outputBytes").and_then(vu64).unwrap_or(0),
                max_bytes: t.get("maxBytes").and_then(vu64),
                shown_range: t.get("shownRange").and_then(range_of),
                head_range: t.get("headRange").and_then(range_of),
                tail_range: t.get("tailRange").and_then(range_of),
                elided_bytes: t.get("elidedBytes").and_then(vu64),
                elided_lines: t.get("elidedLines").and_then(vu64),
                artifact_id: t.get("artifactId").and_then(vstr),
                artifact_elided_bytes: t.get("artifactElidedBytes").and_then(vu64),
                next_offset: t.get("nextOffset").and_then(vu64),
                partial_line: t.get("partialLine").and_then(vbool).unwrap_or(false),
                ..TruncationMeta::default()
            };
            if let Some(dir) = t.get("direction").and_then(vstr) {
                tm.direction = dir;
            }
            if let Some(by) = t.get("truncatedBy").and_then(vstr) {
                tm.truncated_by = by;
            }
            tm
        });
        Self {
            timeout_seconds: v.get("timeoutSeconds").and_then(vu64),
            requested_timeout_seconds: v.get("requestedTimeoutSeconds").and_then(vu64),
            timeout_disabled: v.get("timeoutDisabled").and_then(vbool).unwrap_or(false),
            wall_time_ms: v.get("wallTimeMs").and_then(vu64),
            exit_code: v.get("exitCode").and_then(vi64),
            timed_out: v.get("timedOut").and_then(vbool).unwrap_or(false),
            artifact_error: v.get("artifactError").and_then(vstr),
            truncation,
            service: v.get("service").map(|srv| ServiceDetails {
                name: srv.get("name").and_then(vstr).unwrap_or_default(),
                state: srv.get("state").and_then(vstr).unwrap_or_default(),
                ready: srv.get("ready").and_then(vbool).unwrap_or(false),
                timed_out: srv.get("timedOut").and_then(vbool).unwrap_or(false),
                pid: srv.get("pid").and_then(vu64),
            }),
            background: v.get("async").map(|a| BackgroundJob {
                state: a.get("state").and_then(vstr).unwrap_or_default(),
                job_id: a.get("jobId").and_then(vstr).unwrap_or_default(),
            }),
        }
    }

    /// The `truncation`/`artifactError` subset of `OutputMeta`.
    fn is_empty_meta(&self) -> bool {
        self.truncation.is_none() && self.artifact_error.is_none()
    }
}

// ============================================================================
// Model-facing notices (TS bash.ts:31-60) — these get stripped from display
// ============================================================================

/// LLM-facing footer appended when a call becomes a background job
/// (TS `formatBackgroundNotice`).
#[must_use]
pub fn format_background_notice(job_id: &str, timeout_sec: Option<u64>) -> String {
    let deadline = match timeout_sec {
        None => " (no deadline)".to_owned(),
        Some(sec) => {
            format!(" (deadline {sec}s)")
        }
    };
    format!(
        "Backgrounded as job {job_id}{deadline}; output auto-delivers on finish; don't poll, keep working or end the turn."
    )
}

/// Whether `line` is `format_background_notice(job_id, …)` for any deadline.
fn is_background_notice(line: &str, job_id: &str) -> bool {
    let prefix = format!("Backgrounded as job {job_id}");
    line.starts_with(&prefix) && line[prefix.len()..].starts_with([';', ' '])
}

#[allow(clippy::cast_precision_loss)] // millisecond wall times are far below 2^52
fn format_wall_time_seconds(wall_time_ms: u64) -> String {
    format!("{:.2}", wall_time_ms as f64 / 1000.0)
}

/// Model-facing command duration notice (TS `formatWallTimeNotice`).
#[must_use]
pub fn format_wall_time_notice(wall_time_ms: u64) -> String {
    format!(
        "Wall time: {} seconds",
        format_wall_time_seconds(wall_time_ms)
    )
}

/// Model-facing exit status notice (TS `formatExitCodeNotice`).
#[must_use]
pub fn format_exit_code_notice(exit_code: i64) -> String {
    format!("Command exited with code {exit_code}")
}

// ============================================================================
// Byte/line counting helpers
// ============================================================================

/// TS `pi-utils/formatBytes`.
#[must_use]
#[allow(clippy::cast_precision_loss)] // byte counts in notices fit comfortably in f64
pub fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Replace tabs with the fixed display tab width (TS `replaceTabs`).
#[must_use]
pub fn replace_tabs(text: &str) -> String {
    text.replace('\t', TAB_SPACES)
}

// ============================================================================
// Notice stripping (TS `output-meta.ts`)
// ============================================================================

/// Strip the last literal notice or a matching final line (TS `stripTrailingNotice`).
fn strip_trailing_notice(text: &str, notice: &str) -> String {
    let Some(mut start) = text.rfind(notice) else {
        return text.to_owned();
    };
    let mut end = start + notice.len();
    let bytes = text.as_bytes();
    if start > 0 && bytes[start - 1] == b'\n' {
        start -= 1;
    }
    if end < bytes.len() && bytes[end] == b'\n' {
        end += 1;
    }
    let stripped = format!("{}{}", &text[..start], &text[end..]);
    stripped.trim_end().to_owned()
}

/// Strip a trailing line matching `pred` (TS `stripTrailingNotice`, line form).
fn strip_trailing_line(text: &str, mut pred: impl FnMut(&str) -> bool) -> String {
    let trimmed = text.trim_end();
    let last_start = trimmed.rfind('\n').map_or(0, |i| i + 1);
    if !pred(&trimmed[last_start..]) {
        return text.to_owned();
    }
    // Consume the line plus its preceding newline and any trailing padding.
    let cut = if last_start > 0 {
        last_start - 1
    } else {
        last_start
    };
    text[..cut].trim_end().to_owned()
}

const RAW_OUTPUT_ARTIFACT_PREFIX: &str = "[raw output: artifact://";
const RAW_OUTPUT_ARTIFACT_SUFFIX: &str = "]";

/// Remove the trailing bash raw-output artifact footer, preserving its id
/// (TS `stripRawOutputArtifactNotice`).
#[must_use]
pub fn strip_raw_output_artifact_notice(text: &str) -> (String, Option<String>) {
    let mut artifact_id: Option<String> = None;
    let stripped = strip_trailing_line(text, |line| {
        if !line.starts_with(RAW_OUTPUT_ARTIFACT_PREFIX)
            || !line.ends_with(RAW_OUTPUT_ARTIFACT_SUFFIX)
        {
            return false;
        }
        let id =
            &line[RAW_OUTPUT_ARTIFACT_PREFIX.len()..line.len() - RAW_OUTPUT_ARTIFACT_SUFFIX.len()];
        if id.is_empty() || !id.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
        artifact_id = Some(id.to_owned());
        true
    });
    match artifact_id {
        None => (text.to_owned(), None),
        Some(id) => (stripped, Some(id)),
    }
}

/// Format the LLM-facing truncation/artifact notice text (TS
/// `formatTruncationMetaNotice` + `formatArtifactErrorNotice` parts of
/// `formatOutputNotice`; diagnostics are not produced by bash).
#[must_use]
pub fn format_truncation_meta_notice(t: &TruncationMeta, source: Option<&str>) -> String {
    let artifact_reference = t.artifact_id.as_ref().map(|id| {
        if source == Some("report") {
            format!("Read artifact://{id} for full report")
        } else if t.artifact_elided_bytes.unwrap_or(0) > 0 {
            format!(
                "Read artifact://{id} for a head/tail sample of the output; {} from its middle was not saved",
                format_bytes(t.artifact_elided_bytes.unwrap_or(0))
            )
        } else {
            format!("Read artifact://{id} for full output")
        }
    });

    if t.direction == "middle" {
        let elided_bytes = t
            .elided_bytes
            .unwrap_or_else(|| t.total_bytes.saturating_sub(t.output_bytes));
        let elided_lines = t
            .elided_lines
            .unwrap_or_else(|| t.total_lines.saturating_sub(t.output_lines));
        let head_part = t
            .head_range
            .map_or(String::new(), |(s, e)| format!("lines {s}-{e}"));
        let tail_part = t
            .tail_range
            .map_or(String::new(), |(s, e)| format!("{s}-{e}"));
        let mut notice = if !head_part.is_empty() && !tail_part.is_empty() {
            format!(
                "Showing {head_part} and {tail_part} of {}; {} middle line{} ({}) elided",
                t.total_lines,
                elided_lines,
                if elided_lines == 1 { "" } else { "s" },
                format_bytes(elided_bytes)
            )
        } else if elided_bytes > 0 {
            format!(
                "Showing head and tail bytes of {} line{}; {} elided",
                t.total_lines,
                if t.total_lines == 1 { "" } else { "s" },
                format_bytes(elided_bytes)
            )
        } else {
            format!(
                "Showing {} of {} lines; middle elided",
                t.output_lines.min(t.total_lines),
                t.total_lines
            )
        };
        if let Some(off) = t.next_offset {
            let _ = write!(notice, ". Use :{off} to continue");
        }
        if let Some(r) = artifact_reference {
            let _ = write!(notice, ". {r}");
        }
        return notice;
    }

    if t.partial_line {
        let line = t.shown_range.map_or(1, |(s, _)| s);
        let mut notice = format!(
            "Showing line {line} (partial, {} of {}) of {}",
            format_bytes(t.output_bytes),
            format_bytes(t.total_bytes),
            t.total_lines
        );
        if let Some(r) = artifact_reference {
            let _ = write!(notice, ". {r}");
        }
        return notice;
    }

    let mut notice = match t.shown_range {
        Some((s, e)) if e >= s => format!("Showing lines {s}-{e} of {}", t.total_lines),
        _ => format!("Showing {} of {} lines", t.output_lines, t.total_lines),
    };
    if t.truncated_by == "bytes" {
        let _ = write!(
            notice,
            " ({} limit)",
            format_bytes(t.max_bytes.unwrap_or(t.output_bytes))
        );
    }
    if let Some(off) = t.next_offset {
        let _ = write!(notice, ". Use :{off} to continue");
    }
    if let Some(r) = artifact_reference {
        let _ = write!(notice, ". {r}");
    }
    notice
}

/// LLM-facing notice baked into the output body (TS `formatOutputNotice`,
/// bash-relevant subset: truncation only, wrapped as `[…]`).
#[must_use]
pub fn format_output_notice(details: &BashDetails) -> String {
    let Some(t) = &details.truncation else {
        return String::new();
    };
    format!("\n\n[{}]", format_truncation_meta_notice(t, None))
}

/// Strip the trailing notice that `wrappedExecute` bakes into the LLM-facing
/// body so it is not double-printed next to the styled warning line
/// (TS `stripOutputNotice`).
#[must_use]
pub fn strip_output_notice(text: &str, details: &BashDetails) -> String {
    let notice = format_output_notice(details);
    if notice.is_empty() {
        return text.to_owned();
    }
    let trimmed_text = text.trim_end();
    let trimmed_notice = notice.trim_end();
    if trimmed_text.ends_with(trimmed_notice) {
        return strip_trailing_notice(trimmed_text, trimmed_notice);
    }
    text.to_owned()
}

/// Bash output with the LLM-facing notices removed (TS `stripBashNotices`).
/// Returns the clean text and the artifact id recovered from a raw-output footer.
#[must_use]
pub fn strip_bash_notices(raw_output: &str, details: &BashDetails) -> (String, Option<String>) {
    let without_background = match &details.background {
        Some(job) if job.state == "running" => {
            let job_id = job.job_id.clone();
            strip_trailing_line(raw_output, |line| is_background_notice(line, &job_id))
        }
        _ => raw_output.to_owned(),
    };
    let stripped = strip_output_notice(&without_background, details);
    let without_exit = match details.exit_code {
        Some(code) => strip_trailing_notice(&stripped, &format_exit_code_notice(code)),
        None => stripped,
    };
    let without_wall = match details.wall_time_ms {
        Some(ms) => strip_trailing_notice(&without_exit, &format_wall_time_notice(ms)),
        None => without_exit,
    };
    strip_raw_output_artifact_notice(&without_wall)
}

/// Styled truncation/artifact warning line text, without coloring
/// (TS `formatStyledTruncationWarning` minus `theme.fg`; the caller styles).
#[must_use]
pub fn format_truncation_warning(details: &BashDetails) -> Option<String> {
    if details.is_empty_meta() {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    if let Some(t) = &details.truncation {
        parts.push(format_truncation_meta_notice(t, None));
    }
    if let Some(err) = &details.artifact_error {
        parts.push(format!(
            "Full output was not saved completely (artifact {err} failed)"
        ));
    }
    Some(format!("{BRACKET_LEFT}{}{BRACKET_RIGHT}", parts.join(". ")))
}

// ============================================================================
// Env display + streamed `__partialJson` recovery (TS bash.ts:55-215)
// ============================================================================

fn escape_bash_env_value_for_display(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '"' => out.push_str("\\\""),
            '$' => out.push_str("\\$"),
            '`' => out.push_str("\\`"),
            _ => out.push(ch),
        }
    }
    out
}

/// Sorted `NAME="escaped"` assignments (TS `formatBashEnvAssignments`).
#[must_use]
pub fn format_bash_env_assignments(env: &BTreeMap<String, String>) -> String {
    env.iter()
        .map(|(k, v)| format!("{k}=\"{}\"", escape_bash_env_value_for_display(v)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn unescape_partial_json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }
        let Some(next) = chars.next() else {
            output.push('\\');
            break;
        };
        match next {
            '"' => output.push('"'),
            '\\' => output.push('\\'),
            '/' => output.push('/'),
            'b' => output.push('\u{0008}'),
            'f' => output.push('\u{000C}'),
            'n' => output.push('\n'),
            'r' => output.push('\r'),
            't' => output.push('\t'),
            'u' => {
                let hex: String = chars.by_ref().take(4).collect();
                if hex.len() == 4 && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                    let code = u32::from_str_radix(&hex, 16).unwrap_or(0xFFFD);
                    output.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                } else {
                    output.push_str("\\u");
                    output.push_str(&hex);
                }
            }
            other => output.push(other),
        }
    }
    output
}

// One-slot memo: every render path over one streamed args object asks for the
// same buffer (TS `lastPartialEnvJson`/`lastPartialEnv`).
type PartialEnvCache = Mutex<Option<(String, Option<BTreeMap<String, String>>)>>;
static LAST_PARTIAL_ENV: PartialEnvCache = Mutex::new(None);

/// Recover the `env` object from a partially streamed raw args JSON prefix
/// (TS `extractPartialBashEnv`). Returns `None` when no env object is present.
pub fn extract_partial_bash_env(partial_json: Option<&str>) -> Option<BTreeMap<String, String>> {
    let partial_json = partial_json?;
    {
        let guard = LAST_PARTIAL_ENV
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((last, env)) = guard.as_ref() {
            if last == partial_json {
                return env.clone();
            }
        }
    }
    let env = extract_partial_bash_env_uncached(partial_json);
    let mut guard = LAST_PARTIAL_ENV
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = Some((partial_json.to_owned(), env.clone()));
    env
}

fn extract_partial_bash_env_uncached(partial_json: &str) -> Option<BTreeMap<String, String>> {
    let env_start = partial_json.find("\"env\"")?;
    // Walk `":  {` on the ORIGINAL string so object_start stays a valid index.
    let bytes = partial_json.as_bytes();
    let mut p = env_start + 5;
    while p < bytes.len() && bytes[p].is_ascii_whitespace() {
        p += 1;
    }
    if p >= bytes.len() || bytes[p] != b':' {
        return None;
    }
    p += 1;
    while p < bytes.len() && bytes[p].is_ascii_whitespace() {
        p += 1;
    }
    if p >= bytes.len() || bytes[p] != b'{' {
        return None;
    }
    let object_start = p;
    // Scan the env object body, stopping at its closing brace outside strings.
    let mut depth: usize = 0;
    let mut object_end = partial_json.len();
    let mut scan = object_start + 1;
    while scan < bytes.len() {
        match bytes[scan] {
            b'"' => {
                scan += 1;
                while scan < bytes.len() {
                    match bytes[scan] {
                        b'\\' => scan += 1,
                        b'"' => break,
                        _ => {}
                    }
                    scan += 1;
                }
            }
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                if depth == 0 {
                    object_end = scan;
                    break;
                }
                depth -= 1;
            }
            _ => {}
        }
        scan += 1;
    }
    parse_env_entries(&partial_json[object_start + 1..object_end])
}

/// Parse `"KEY": "value"` entries from a (possibly truncated) env object body.
fn parse_env_entries(env_body: &str) -> Option<BTreeMap<String, String>> {
    let mut env = BTreeMap::new();
    let body = env_body.as_bytes();
    let mut cursor = 0;
    while cursor < body.len() {
        if body[cursor] != b'"' {
            cursor += 1;
            continue;
        }
        // Key: "([A-Za-z_][A-Za-z0-9_]*)"
        let mut key_end = cursor + 1;
        let key_start = key_end;
        while key_end < body.len()
            && (body[key_end].is_ascii_alphanumeric() || body[key_end] == b'_')
        {
            key_end += 1;
        }
        if key_end == key_start
            || !(body[key_start].is_ascii_alphabetic() || body[key_start] == b'_')
        {
            // Not a valid identifier key; skip this string.
            cursor += 1;
            continue;
        }
        if key_end >= body.len() || body[key_end] != b'"' {
            // Unterminated key string.
            cursor += 1;
            continue;
        }
        let key = &env_body[key_start..key_end];
        // Step past the key's closing quote.
        let mut colon = key_end + 1;
        while colon < body.len() && matches!(body[colon], b' ' | b'\t' | b'\n' | b'\r') {
            colon += 1;
        }
        if colon >= body.len() || body[colon] != b':' {
            cursor += 1;
            continue;
        }
        colon += 1;
        while colon < body.len() && matches!(body[colon], b' ' | b'\t' | b'\n' | b'\r') {
            colon += 1;
        }
        if colon >= body.len() || body[colon] != b'"' {
            cursor += 1;
            continue;
        }
        // Value: "((?:\\.|[^"\\])*)" with unterminated strings running to the end.
        let mut value = String::new();
        let mut value_end = colon + 1;
        let mut terminated = false;
        while value_end < body.len() {
            match body[value_end] {
                b'\\' if value_end + 1 < body.len() => {
                    value.push_str(&env_body[value_end..value_end + 2]);
                    value_end += 2;
                }
                b'"' => {
                    terminated = true;
                    value_end += 1;
                    break;
                }
                _ => {
                    value.push_str(&env_body[value_end..=value_end]);
                    value_end += 1;
                }
            }
        }
        if terminated || value_end >= body.len() {
            env.insert(key.to_owned(), unescape_partial_json_string(&value));
        }
        cursor = value_end.max(cursor + 1);
    }
    if env.is_empty() {
        None
    } else {
        Some(env)
    }
}

// ============================================================================
// Command preview (TS `formatBashCommandLines`)
// ============================================================================

/// Normalize a workdir for display: relative path inside the project, else the
/// shortened absolute path; `None` when it equals the project dir
/// (TS `formatToolWorkingDirectory`, Windows-drive special case N/A on Linux).
#[must_use]
pub fn format_tool_working_directory(workdir: Option<&str>, project_dir: &Path) -> Option<String> {
    let workdir = workdir?;
    let resolved_project = normalize_path(Path::new(project_dir));
    let raw = Path::new(workdir);
    let resolved_workdir = if raw.is_absolute() {
        normalize_path(raw)
    } else {
        normalize_path(&resolved_project.join(raw))
    };
    if resolved_workdir == resolved_project {
        return None;
    }
    let relative = pathdiff(&resolved_project, &resolved_workdir);
    let display = match relative {
        Some(rel) if !rel.starts_with("..") => rel,
        _ => shorten_path(&resolved_workdir),
    };
    Some(replace_tabs(&display))
}

/// Lexical `..`-aware path normalization (no symlink resolution).
fn normalize_path(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Relative path from `base` to `target`, or `None` when `target` is not
/// inside `base`.
fn pathdiff(base: &Path, target: &Path) -> Option<String> {
    let b: Vec<_> = base.components().collect();
    let t: Vec<_> = target.components().collect();
    if b.is_empty() || t.is_empty() || b[0] != t[0] {
        return None;
    }
    let mut common = 0;
    while common < b.len() && common < t.len() && b[common] == t[common] {
        common += 1;
    }
    if common < b.len() {
        return None; // target outside base
    }
    let rel: Vec<_> = t[common..]
        .iter()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    Some(if rel.is_empty() {
        ".".to_owned()
    } else {
        rel.join("/")
    })
}

/// `~/`-collapse a path for display (TS `shortenPath` subset).
fn shorten_path(p: &Path) -> String {
    if let Some(home) = std::env::var_os("HOME") {
        let home = Path::new(&home);
        if let Some(rest) = p
            .strip_prefix(home)
            .ok()
            .filter(|r| !r.as_os_str().is_empty())
        {
            return format!("~/{}", rest.to_string_lossy());
        }
    }
    p.to_string_lossy().into_owned()
}

/// Shell arguments used to build a command preview (TS `BashRenderArgs`).
#[derive(Debug, Default, Clone)]
pub struct BashRenderArgs {
    pub command: Option<String>,
    pub env: Option<BTreeMap<String, String>>,
    pub cwd: Option<String>,
    pub name: Option<String>,
}

impl BashRenderArgs {
    /// Parse from the tool call's `arguments` object plus the streamed
    /// `__partialJson` env recovery (TS `getBashEnvForDisplay`).
    #[must_use]
    pub fn from_value(call: &Value) -> Self {
        let args = call.get("arguments").unwrap_or(call);
        let mut env: Option<BTreeMap<String, String>> =
            args.get("env").and_then(Value::as_object).map(|obj| {
                obj.iter()
                    .map(|(k, v)| {
                        let val = v.as_str().map_or_else(|| v.to_string(), str::to_owned);
                        (k.clone(), val)
                    })
                    .collect()
            });
        // Merge streamed env under the parsed env (parsed wins).
        if let Some(partial_env) =
            extract_partial_bash_env(call.get("__partialJson").and_then(Value::as_str))
        {
            let mut merged = partial_env;
            if let Some(parsed) = env {
                merged.extend(parsed);
            }
            env = Some(merged);
        }
        Self {
            command: args.get("command").and_then(vstr),
            env,
            cwd: args.get("cwd").and_then(vstr),
            name: args.get("name").and_then(vstr),
        }
    }

    /// Plain command text for native views (TS `formatBashCommandText`).
    #[must_use]
    pub fn command_text(&self, project_dir: &Path) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(wd) = format_tool_working_directory(self.cwd.as_deref(), project_dir) {
            parts.push(format!("cd {wd} &&"));
        }
        if let Some(env) = &self.env {
            let assignments = format_bash_env_assignments(env);
            if !assignments.is_empty() {
                parts.push(assignments);
            }
        }
        parts.push(self.command.clone().unwrap_or_else(|| "…".to_owned()));
        parts.join(" ")
    }

    /// Dim `$ cd … && NAME="…" ` prefix spans (TS `formatBashCommandLines`,
    /// with the prefix only on the first line so SGR state cannot leak).
    fn prefix_spans(&self, project_dir: &Path) -> Vec<Span<'static>> {
        let mut prefix_parts: Vec<String> = vec!["$".to_owned()];
        if let Some(wd) = format_tool_working_directory(self.cwd.as_deref(), project_dir) {
            prefix_parts.push(format!("cd {wd} &&"));
        }
        if let Some(env) = &self.env {
            let assignments = format_bash_env_assignments(env);
            if !assignments.is_empty() {
                prefix_parts.push(assignments);
            }
        }
        vec![Span::styled(
            format!("{} ", prefix_parts.join(" ")),
            Style::default().fg(Color::DarkGray),
        )]
    }

    /// Command lines with the dim `$ cd … &&` prefix on the first line.
    #[must_use]
    pub fn command_lines(&self, project_dir: &Path, theme: &Theme) -> Vec<Line<'static>> {
        let _ = theme; // highlighting hook (CodeHighlight) lands later
        let command = replace_tabs(self.command.as_deref().unwrap_or("…"));
        let mut lines: Vec<Line<'static>> = Vec::new();
        let prefix = self.prefix_spans(project_dir);
        let command_lines: Vec<&str> = command.lines().collect();
        if command_lines.is_empty() {
            let mut spans = prefix;
            spans.pop();
            lines.push(Line::from(spans));
            return lines;
        }
        for (i, line) in command_lines.iter().enumerate() {
            if i == 0 {
                let mut spans = prefix.clone();
                spans.push(Span::raw((*line).to_owned()));
                lines.push(Line::from(spans));
            } else {
                lines.push(Line::from((*line).to_owned()));
            }
        }
        lines
    }
}

// ============================================================================
// Stats + viewport tail windows (TS `bashStatsParts`, `capPreviewLines`,
// `previewWindowRows`)
// ============================================================================

/// `Wall: 1.20s`, `Timeout: 30s`, … metadata parts (TS `bashStatsParts`).
#[must_use]
pub fn bash_stats_parts(
    details: &BashDetails,
    context_timeout: Option<u64>,
    artifact_id: Option<&str>,
    is_error: bool,
) -> Vec<String> {
    let timeout_disabled = details.timeout_disabled || context_timeout == Some(0);
    let timeout_seconds = if timeout_disabled {
        None
    } else {
        details.timeout_seconds.or(context_timeout)
    };
    let mut parts: Vec<String> = Vec::new();
    if details
        .background
        .as_ref()
        .is_some_and(|b| b.state == "running")
    {
        parts.push(format!(
            "Backgrounded: {}",
            details
                .background
                .as_ref()
                .map_or("", |b| b.job_id.as_str())
        ));
    }
    if let Some(service) = &details.service {
        parts.push(format!("Service: {}", service.name));
        parts.push(format!("State: {}", service.state));
        parts.push(format!(
            "Ready: {}",
            if service.ready {
                "yes"
            } else if service.timed_out {
                "timed out"
            } else {
                "no"
            }
        ));
        if let Some(pid) = service.pid {
            parts.push(format!("PID: {pid}"));
        }
    }
    if let Some(ms) = details.wall_time_ms {
        parts.push(format!("Wall: {}s", format_wall_time_seconds(ms)));
    }
    if timeout_disabled {
        parts.push("Timeout: disabled".to_owned());
    }
    if let Some(sec) = timeout_seconds {
        if details
            .requested_timeout_seconds
            .is_some_and(|r| Some(r) != timeout_seconds)
        {
            parts.push(format!(
                "Timeout: {sec}s (requested {}s clamped)",
                details.requested_timeout_seconds.unwrap_or(sec)
            ));
        } else {
            parts.push(format!("Timeout: {sec}s"));
        }
    }
    if let Some(id) = artifact_id {
        parts.push(format!("Artifact: {id}"));
    }
    if is_error && details.exit_code.is_some() {
        parts.push(format!("Exit: {}", details.exit_code.unwrap_or(0)));
    }
    parts
}

/// Tail-window height for collapsed previews (TS `previewWindowRows`).
#[must_use]
pub fn preview_window_rows(viewport_rows: Option<usize>) -> usize {
    let rows = viewport_rows.unwrap_or(PREVIEW_WINDOW_FALLBACK_ROWS);
    (rows.saturating_sub(PREVIEW_WINDOW_RESERVED_ROWS)).max(PREVIEW_WINDOW_MIN_LINES)
}

/// Cap pre-rendered lines to a viewport-sized tail window behind a
/// `… N earlier lines [ctrl+o: Expand]` marker (TS `capPreviewLines`).
/// `expanded` bypasses the cap.
#[must_use]
pub fn cap_preview_lines(
    mut lines: Vec<Line<'static>>,
    expanded: bool,
    max: Option<usize>,
    viewport_rows: Option<usize>,
) -> Vec<Line<'static>> {
    if expanded {
        return lines;
    }
    let max = max.unwrap_or_else(|| preview_window_rows(viewport_rows));
    if lines.len() <= max {
        return lines;
    }
    let visible = if max <= 1 {
        Vec::new()
    } else {
        lines.split_off(lines.len() - (max - 1))
    };
    // `lines` now holds exactly the hidden head.
    let hidden = lines.len();
    let mut out = vec![Line::from(Span::styled(
        format!(
            "… {hidden} earlier line{} [ctrl+o: Expand]",
            if hidden == 1 { "" } else { "s" }
        ),
        Style::default().fg(Color::DarkGray),
    ))];
    out.extend(visible);
    out
}

/// Tail-cap raw output lines to `min(preview_lines, preview window)` behind an
/// `… N earlier lines` marker (TS `formatOutputPaneLines` with
/// `edge: "tail"`, simplified to a hard line cap — see module docs).
#[must_use]
pub fn cap_output_tail(
    lines: Vec<Line<'static>>,
    preview_lines: usize,
    expanded: bool,
    viewport_rows: Option<usize>,
) -> Vec<Line<'static>> {
    cap_preview_lines(
        lines,
        expanded,
        Some(preview_lines.min(preview_window_rows(viewport_rows))),
        viewport_rows,
    )
}

// ============================================================================
// Theme helpers + card phases
// ============================================================================

/// Resolve a token, falling back when the theme does not define it.
fn themed(theme: &Theme, token: &str, fallback: &str) -> Color {
    let c = theme.color(token);
    if c == Color::Reset {
        theme.color(fallback)
    } else {
        c
    }
}

/// Visual phase of the framed card (TS `ToolCardSnapshot.phase`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BashPhase {
    /// Pending/streaming: muted border.
    Pending,
    /// Settled with an error: error border (yellow when timed out).
    Error,
    /// Settled by timeout deadline: warning border.
    Warning,
    /// Settled successfully.
    Success,
}

impl BashPhase {
    fn border_color(self, theme: &Theme) -> Color {
        match self {
            Self::Pending => themed(theme, "accent", "text_muted"),
            Self::Error => theme.color("error"),
            Self::Warning => theme.color("warning"),
            Self::Success => themed(theme, "dim", "text_muted"),
        }
    }
}

/// Everything needed to render a bash tool card.
struct BashCard<'a> {
    args: BashRenderArgs,
    details: BashDetails,
    /// Cleaned output text (notices stripped).
    output: String,
    artifact_id: Option<String>,
    is_partial: bool,
    is_error: bool,
    expanded: bool,
    preview_lines: usize,
    context_timeout: Option<u64>,
    project_dir: PathBuf,
    theme: &'a Theme,
}

impl<'a> BashCard<'a> {
    fn parse(call: &'a Value, theme: &'a Theme) -> Self {
        let args = BashRenderArgs::from_value(call);
        let result = call.get("result");
        let details = result
            .and_then(|r| r.get("details"))
            .map_or_else(BashDetails::default, BashDetails::from_value);
        let is_partial = call
            .get("isPartial")
            .and_then(Value::as_bool)
            .unwrap_or(result.is_none());
        let is_error = result
            .and_then(|r| r.get("isError"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let raw_output = call
            .get("output")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                result
                    .and_then(|r| r.get("content"))
                    .and_then(|c| c.as_array())
                    .and_then(|arr| {
                        arr.iter()
                            .find(|e| e.get("type").and_then(Value::as_str) == Some("text"))
                            .and_then(|e| e.get("text"))
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
            })
            .unwrap_or_default();
        let (output, artifact_id) = strip_bash_notices(&raw_output, &details);
        let project_dir = call
            .get("projectDir")
            .and_then(Value::as_str)
            .map_or_else(|| PathBuf::from("."), PathBuf::from);
        Self {
            args,
            details,
            output: output.trim_end().to_owned(),
            artifact_id,
            is_partial,
            is_error,
            expanded: call
                .get("expanded")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            preview_lines: call
                .get("previewLines")
                .and_then(Value::as_u64)
                .map_or(BASH_DEFAULT_PREVIEW_LINES, |n| {
                    usize::try_from(n).unwrap_or(BASH_DEFAULT_PREVIEW_LINES)
                }),
            context_timeout: call.get("timeout").and_then(Value::as_u64),
            project_dir,
            theme,
        }
    }

    fn phase(&self) -> BashPhase {
        if self.is_partial {
            BashPhase::Pending
        } else if self.is_error {
            if self.details.timed_out {
                BashPhase::Warning
            } else {
                BashPhase::Error
            }
        } else {
            BashPhase::Success
        }
    }

    /// Card sections: the command, then the labelled `Output` section with
    /// output rows, the dim stats line and the truncation warning.
    fn sections(&self, viewport_rows: Option<usize>) -> Vec<CardSection> {
        let theme = self.theme;

        // (1) Command lines, capped to the viewport tail window.
        let cmd_lines = cap_preview_lines(
            self.args.command_lines(&self.project_dir, theme),
            self.expanded,
            None,
            viewport_rows,
        );
        let mut out: Vec<Line<'static>> = Vec::new();

        // (2) Output pane. A settled error whose entire output is an
        //     `Error: ` failure notice renders as an error-detail block (TS
        //     `formatErrorDetail`) instead of a plain output pane.
        if self.is_error && !self.is_partial && self.output.starts_with("Error: ") {
            out.extend(error_detail_lines(&self.output, usize::MAX, theme));
        } else {
            let output_style = Style::default().fg(themed(theme, "toolOutput", "text"));
            let out_lines: Vec<Line<'static>> = if self.output.trim().is_empty() {
                Vec::new()
            } else {
                self.output
                    .split('\n')
                    .map(|l| Line::from(Span::styled(replace_tabs(l), output_style)))
                    .collect()
            };
            out.extend(cap_output_tail(
                out_lines,
                self.preview_lines,
                self.expanded,
                viewport_rows,
            ));
        }

        // (3) Dim stats line — only the parts that apply.
        let stats = bash_stats_parts(
            &self.details,
            self.context_timeout,
            self.artifact_id.as_deref(),
            self.is_error,
        );
        if !stats.is_empty() {
            out.push(Line::from(Span::styled(
                format!(
                    "{}{}{}",
                    theme.symbol("format.bracketLeft"),
                    stats.join(" | "),
                    theme.symbol("format.bracketRight")
                ),
                Style::default().fg(themed(theme, "dim", "text_muted")),
            )));
        }

        // (4) Truncation / artifact warning (suppressed when the expanded view
        //     already shows the full artifact output).
        let showing_full_output = self.expanded; // isFullOutput rides on `expanded` here
        if self.details.artifact_error.is_some()
            || (self.details.truncation.is_some() && !showing_full_output)
        {
            if let Some(warning) = format_truncation_warning(&self.details) {
                out.push(Line::from(Span::styled(
                    warning,
                    Style::default().fg(theme.color("warning")),
                )));
            }
        }
        vec![
            CardSection::plain(cmd_lines),
            CardSection::labelled(theme, "Output", out),
        ]
    }

    /// The framed card rows at `width`.
    fn card_lines(&self, width: usize, viewport_rows: Option<usize>) -> Vec<Line<'static>> {
        framed_card(
            None,
            self.sections(viewport_rows),
            width,
            self.phase().border_color(self.theme),
        )
    }
}

/// Framed bash card rows for a call payload (`output` / `result{details,
/// isError}` / `isPartial` as in [`BashCard::parse`]).
#[must_use]
pub fn card_lines(call: &Value, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let card = BashCard::parse(call, theme);
    card.card_lines(width, None)
}

/// TS `resolveTitle`: `Bash · {name}` or `Bash`. Exported for the native/TSP
/// tool head (`describeCall`/`describeResult`), which lives in the TSP emitter.
#[must_use]
pub fn resolve_title(args: &BashRenderArgs) -> String {
    match &args.name {
        Some(name) if !name.is_empty() => format!("Bash · {name}"),
        _ => "Bash".to_owned(),
    }
}

/// Renderer for the model's `bash` tool calls.
pub struct Bash;

impl ToolRenderer for Bash {
    fn summary(&self, call: &Value) -> String {
        let args = BashRenderArgs::from_value(call);
        let cmd = args.command.as_deref().unwrap_or("…");
        let cmd = cmd.lines().next().unwrap_or("…");
        let text: String = cmd.chars().take(80).collect();
        if let Some(name) = args.name.as_deref().filter(|n| !n.is_empty()) {
            format!("{name} · {text}")
        } else {
            text
        }
    }

    fn render(&self, frame: &mut Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
        let card = BashCard::parse(call, theme);
        let lines = card.card_lines(area.width as usize, Some(area.height as usize));
        frame.render_widget(Paragraph::new(lines), area);
    }
}

// ============================================================================
// `!command` user bash execution card (interactive.md §4.3)
// ============================================================================

/// State of a user-initiated `!command` execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserBashState {
    /// Running in the foreground; Esc aborts (`session.abortBash()`).
    Running,
    /// Finished with the given facts.
    Finished {
        exit_code: Option<i64>,
        wall_time_ms: Option<u64>,
    },
}

impl UserBashState {
    fn phase(&self) -> BashPhase {
        match self {
            Self::Running => BashPhase::Pending,
            Self::Finished { exit_code, .. } if exit_code.is_some_and(|c| c != 0) => {
                BashPhase::Error
            }
            Self::Finished { .. } => BashPhase::Success,
        }
    }
}

/// Render the live bash card shown in the pending-messages band for a
/// user-typed `!command` (interactive.md §4.3): framed card with a `❯ Bash`
/// header, the command, the streaming output tail, and — while running — a
/// dim `⟨esc: Cancel⟩` affordance. Height used: content + borders, capped at
/// `area.height`.
pub fn render_user_bash_card(
    frame: &mut Frame<'_>,
    area: Rect,
    command: &str,
    output: &str,
    state: &UserBashState,
    theme: &Theme,
) {
    let phase = state.phase();
    let mut lines: Vec<Line<'static>> = Vec::new();
    // Header: TS `renderStatusLine` with the `tool.bash` glyph and title.
    let header_glyph = theme.glyph("prompt");
    let header_style = Style::default().fg(if matches!(phase, BashPhase::Pending) {
        theme.color("accent")
    } else {
        theme.color("success")
    });
    lines.push(Line::from(vec![
        Span::styled(format!("{header_glyph} "), header_style),
        Span::styled(
            "Bash",
            Style::default()
                .fg(themed(theme, "toolTitle", "text"))
                .add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.extend(
        replace_tabs(command)
            .lines()
            .map(|l| Line::from(l.to_owned())),
    );
    let out: Vec<Line<'static>> = output
        .trim_end()
        .split('\n')
        .filter(|l| !l.is_empty())
        .map(|l| {
            Line::from(Span::styled(
                replace_tabs(l),
                Style::default().fg(themed(theme, "toolOutput", "text")),
            ))
        })
        .collect();
    lines.extend(cap_output_tail(
        out,
        BASH_DEFAULT_PREVIEW_LINES,
        false,
        Some(area.height as usize),
    ));
    match state {
        UserBashState::Running => {
            lines.push(Line::from(Span::styled(
                format!("{BRACKET_LEFT}esc: Cancel{BRACKET_RIGHT}"),
                Style::default().fg(themed(theme, "text_muted", "text_muted")),
            )));
        }
        UserBashState::Finished {
            exit_code,
            wall_time_ms,
        } => {
            let mut meta_parts: Vec<String> = Vec::new();
            if let Some(ms) = wall_time_ms {
                meta_parts.push(format!("Wall: {}s", format_wall_time_seconds(*ms)));
            }
            if let Some(code) = exit_code {
                meta_parts.push(format!("Exit: {code}"));
            }
            if !meta_parts.is_empty() {
                let color = if exit_code.is_some_and(|c| c != 0) {
                    theme.color("error")
                } else {
                    themed(theme, "text_muted", "text_muted")
                };
                lines.push(Line::from(Span::styled(
                    format!("{BRACKET_LEFT}{}{BRACKET_RIGHT}", meta_parts.join(" | ")),
                    Style::default().fg(color),
                )));
            }
        }
    }
    let border_color = match phase {
        BashPhase::Pending => theme.color("accent"),
        other => other.border_color(theme),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));
    let height = u16::try_from(lines.len().saturating_add(2))
        .unwrap_or(u16::MAX)
        .min(area.height.max(2));
    let area = Rect { height, ..area };
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn project_dir() -> PathBuf {
        PathBuf::from("/repo")
    }

    fn render_to_string(call: &Value, width: u16, height: u16) -> (String, BashPhase) {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        let theme = Theme::default();
        terminal
            .draw(|f| Bash.render(f, f.area(), call, &theme))
            .unwrap();
        let card = BashCard::parse(call, &theme);
        let phase = card.phase();
        let buf = terminal.backend().buffer();
        let text = (0..height)
            .map(|y| (0..width).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        (text, phase)
    }

    fn sample_result_call() -> Value {
        serde_json::json!({
            "name": "bash",
            "projectDir": "/repo",
            "arguments": {"command": "cargo test", "cwd": "/repo"},
            "result": {
                "content": [{"type": "text", "text": "ok\n"}],
                "details": {"wallTimeMs": 1200, "exitCode": 0, "timeoutSeconds": 30},
                "isError": false
            }
        })
    }

    #[test]
    fn summary_shows_first_command_line() {
        let call = serde_json::json!({"name":"bash","arguments":{"command":"cargo test
"}});
        assert_eq!(Bash.summary(&call), "cargo test");
    }

    #[test]
    fn summary_includes_name_label() {
        let call = serde_json::json!({"name":"bash","arguments":{"command":"npm run build","name":"build"}});
        assert_eq!(Bash.summary(&call), "build · npm run build");
    }

    #[test]
    fn background_notice_mentions_job_and_deadline() {
        let notice = format_background_notice("job-7", Some(30));
        assert!(notice.starts_with("Backgrounded as job job-7 (deadline 30s)"));
        let no_deadline = format_background_notice("job-7", None);
        assert!(no_deadline.starts_with("Backgrounded as job job-7 (no deadline)"));
        assert!(is_background_notice(
            "Backgrounded as job job-7; output auto-delivers",
            "job-7"
        ));
        assert!(!is_background_notice(
            "Backgrounded as job job-8; x",
            "job-7"
        ));
    }

    #[test]
    fn escapes_env_values_for_display() {
        let input = r#"a\b\nc\t"q\$x\`"#;
        let expected = r#"a\\b\\nc\\t\"q\\\$x\\\`"#;
        assert_eq!(escape_bash_env_value_for_display(input), expected);
        let mut env = BTreeMap::new();
        env.insert("B_KEY".to_owned(), "2".to_owned());
        env.insert("A_KEY".to_owned(), "1".to_owned());
        assert_eq!(format_bash_env_assignments(&env), "A_KEY=\"1\" B_KEY=\"2\"");
    }

    #[test]
    fn extracts_env_from_partial_json_stream() {
        let partial = r#"{"command": "ls", "env": {"FOO": "a\n", "BA_R": "b""#;
        let env = extract_partial_bash_env(Some(partial)).unwrap();
        assert_eq!(env.get("FOO").unwrap(), "a\n");
        assert_eq!(env.get("BA_R").unwrap(), "b");
        assert!(extract_partial_bash_env(Some("{\"command\": \"ls\"}")).is_none());
        assert!(extract_partial_bash_env(None).is_none());
    }

    #[test]
    fn command_lines_prefix_first_line_only() {
        let call = serde_json::json!({
            "arguments": {"command": "line1\nline2", "cwd": "/repo/sub", "env": {"K": "v"}}
        });
        let args = BashRenderArgs::from_value(&call);
        let theme = Theme::default();
        let lines = args.command_lines(&project_dir(), &theme);
        assert_eq!(lines.len(), 2);
        let first = lines[0]
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect::<String>();
        assert!(
            first.starts_with("$ cd sub && K=\"v\" line1"),
            "got: {first}"
        );
        let second = lines[1]
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect::<String>();
        assert_eq!(second, "line2");
    }

    #[test]
    fn command_lines_hide_workdir_inside_project_and_empty_command() {
        let call = serde_json::json!({"arguments": {"command": "pwd", "cwd": "/repo"}});
        let args = BashRenderArgs::from_value(&call);
        let theme = Theme::default();
        let lines = args.command_lines(&project_dir(), &theme);
        let first = lines[0]
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect::<String>();
        assert_eq!(first, "$ pwd");

        let empty = serde_json::json!({"arguments": {}});
        let args = BashRenderArgs::from_value(&empty);
        let lines = args.command_lines(&project_dir(), &theme);
        let first = lines[0]
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect::<String>();
        assert_eq!(first, "$ …");
    }

    #[test]
    fn strip_notices_removes_llm_facing_footers() {
        let details = BashDetails {
            wall_time_ms: Some(1200),
            exit_code: Some(1),
            background: Some(BackgroundJob {
                state: "running".into(),
                job_id: "job-9".into(),
            }),
            ..BashDetails::default()
        };
        let raw = format!(
            "some output\n[raw output: artifact://42]\n{}\n{}\n{}",
            format_wall_time_notice(1200),
            format_exit_code_notice(1),
            format_background_notice("job-9", Some(30))
        );
        let (text, artifact) = strip_bash_notices(&raw, &details);
        assert_eq!(text, "some output");
        assert_eq!(artifact.as_deref(), Some("42"));
    }

    #[test]
    fn strip_leaves_clean_output_untouched() {
        let (text, artifact) = strip_bash_notices("hello\nworld", &BashDetails::default());
        assert_eq!(text, "hello\nworld");
        assert_eq!(artifact, None);
    }

    #[test]
    fn stats_parts_cover_all_facts() {
        let details = BashDetails {
            background: Some(BackgroundJob {
                state: "running".into(),
                job_id: "j1".into(),
            }),
            service: Some(ServiceDetails {
                name: "srv".into(),
                state: "running".into(),
                ready: false,
                timed_out: true,
                pid: Some(4242),
            }),
            wall_time_ms: Some(500),
            timeout_seconds: Some(10),
            requested_timeout_seconds: Some(120),
            exit_code: Some(3),
            ..BashDetails::default()
        };
        let parts = bash_stats_parts(&details, None, Some("77"), true);
        let joined = parts.join(" | ");
        assert!(joined.contains("Backgrounded: j1"), "{joined}");
        assert!(joined.contains("Service: srv"), "{joined}");
        assert!(joined.contains("State: running"), "{joined}");
        assert!(joined.contains("Ready: timed out"), "{joined}");
        assert!(joined.contains("PID: 4242"), "{joined}");
        assert!(joined.contains("Wall: 0.50s"), "{joined}");
        assert!(
            joined.contains("Timeout: 10s (requested 120s clamped)"),
            "{joined}"
        );
        assert!(joined.contains("Artifact: 77"), "{joined}");
        assert!(joined.contains("Exit: 3"), "{joined}");

        let disabled = BashDetails {
            timeout_disabled: true,
            ..BashDetails::default()
        };
        assert_eq!(
            bash_stats_parts(&disabled, None, None, false),
            vec!["Timeout: disabled"]
        );
        assert!(
            bash_stats_parts(&BashDetails::default(), Some(0), None, false)
                .contains(&"Timeout: disabled".to_owned())
        );
    }

    #[test]
    fn truncation_notice_formats_ranges() {
        let t = TruncationMeta {
            direction: "tail".into(),
            truncated_by: "lines".into(),
            total_lines: 100,
            total_bytes: 5000,
            output_lines: 10,
            output_bytes: 500,
            shown_range: Some((91, 100)),
            ..TruncationMeta::default()
        };
        assert_eq!(
            format_truncation_meta_notice(&t, None),
            "Showing lines 91-100 of 100"
        );

        let bytes = TruncationMeta {
            truncated_by: "bytes".into(),
            max_bytes: Some(1024),
            ..t
        };
        assert_eq!(
            format_truncation_meta_notice(&bytes, None),
            "Showing lines 91-100 of 100 (1.0KB limit)"
        );

        let middle = TruncationMeta {
            direction: "middle".into(),
            total_lines: 1000,
            total_bytes: 100_000,
            output_lines: 20,
            output_bytes: 2000,
            elided_lines: Some(980),
            elided_bytes: Some(98_000),
            head_range: Some((1, 10)),
            tail_range: Some((991, 1000)),
            ..TruncationMeta::default()
        };
        let notice = format_truncation_meta_notice(&middle, None);
        assert!(
            notice.contains("Showing lines 1-10 and 991-1000 of 1000"),
            "{notice}"
        );
        assert!(
            notice.contains("980 middle lines (95.7KB) elided"),
            "{notice}"
        );

        let partial = TruncationMeta {
            partial_line: true,
            shown_range: Some((5, 5)),
            total_lines: 10,
            output_bytes: 100,
            total_bytes: 10_000,
            ..TruncationMeta::default()
        };
        assert_eq!(
            format_truncation_meta_notice(&partial, None),
            "Showing line 5 (partial, 100B of 9.8KB) of 10"
        );

        let with_artifact = TruncationMeta {
            artifact_id: Some("7".into()),
            shown_range: Some((1, 5)),
            total_lines: 50,
            ..TruncationMeta::default()
        };
        assert_eq!(
            format_truncation_meta_notice(&with_artifact, None),
            "Showing lines 1-5 of 50. Read artifact://7 for full output"
        );
    }

    #[test]
    fn truncation_warning_brackets_and_artifact_error() {
        let details = BashDetails {
            truncation: Some(TruncationMeta {
                direction: "tail".into(),
                shown_range: Some((1, 3)),
                total_lines: 9,
                ..TruncationMeta::default()
            }),
            ..BashDetails::default()
        };
        assert_eq!(
            format_truncation_warning(&details).unwrap(),
            "⟨Showing lines 1-3 of 9⟩"
        );
        let err = BashDetails {
            artifact_error: Some("disk full".into()),
            ..BashDetails::default()
        };
        assert_eq!(
            format_truncation_warning(&err).unwrap(),
            "⟨Full output was not saved completely (artifact disk full failed)⟩"
        );
        assert!(format_truncation_warning(&BashDetails::default()).is_none());
    }

    #[test]
    fn cap_preview_lines_tail_window() {
        let lines: Vec<Line<'static>> = (1..=12).map(|i| Line::from(format!("l{i}"))).collect();
        let capped = cap_preview_lines(lines.clone(), false, Some(5), None);
        assert_eq!(capped.len(), 5);
        assert!(
            capped[0].spans[0].content.starts_with("… 8 earlier lines"),
            "{:?}",
            capped[0]
        );
        assert_eq!(capped[1].spans[0].content, "l9");
        assert_eq!(capped[4].spans[0].content, "l12");
        // Expanded bypasses.
        assert_eq!(
            cap_preview_lines(lines.clone(), true, Some(5), None).len(),
            12
        );
        // Within the cap: unchanged.
        assert_eq!(
            cap_preview_lines(lines[..3].to_vec(), false, Some(5), None).len(),
            3
        );
    }

    #[test]
    fn preview_window_rows_math() {
        assert_eq!(preview_window_rows(Some(40)), 20);
        assert_eq!(preview_window_rows(Some(10)), 6);
        assert_eq!(preview_window_rows(None), 10);
    }

    #[test]
    fn format_bytes_matches_ts() {
        assert_eq!(format_bytes(512), "512B");
        assert_eq!(format_bytes(2048), "2.0KB");
        assert_eq!(format_bytes(3 * 1024 * 1024), "3.0MB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.0GB");
    }

    #[test]
    fn format_tool_working_directory_relative_and_absolute() {
        assert_eq!(
            format_tool_working_directory(Some("/repo"), &project_dir()),
            None
        );
        assert_eq!(
            format_tool_working_directory(Some("sub/dir"), &project_dir()).unwrap(),
            "sub/dir"
        );
        assert_eq!(
            format_tool_working_directory(Some("/repo/../repo/x"), &project_dir()).unwrap(),
            "x"
        );
        let outside = format_tool_working_directory(Some("/other/place"), &project_dir()).unwrap();
        assert!(outside.ends_with("other/place"), "{outside}");
    }

    #[test]
    fn renders_success_card_with_stats() {
        let (text, phase) = render_to_string(&sample_result_call(), 60, 12);
        assert_eq!(phase, BashPhase::Success);
        assert!(text.contains("$ cargo test"), "{text}");
        assert!(text.contains("Output"), "{text}");
        assert!(text.contains("ok"), "{text}");
        assert!(text.contains("⟦Wall: 1.20s | Timeout: 30s⟧"), "{text}");
    }

    #[test]
    fn renders_error_card_with_exit_and_warning_border() {
        let mut call = sample_result_call();
        call["result"]["isError"] = serde_json::json!(true);
        call["result"]["details"]["exitCode"] = serde_json::json!(2);
        call["result"]["details"]["timedOut"] = serde_json::json!(false);
        let (text, phase) = render_to_string(&call, 60, 12);
        assert_eq!(phase, BashPhase::Error);
        assert!(text.contains("Exit: 2"), "{text}");
    }

    #[test]
    fn timed_out_error_is_warning_phase() {
        let mut call = sample_result_call();
        call["result"]["isError"] = serde_json::json!(true);
        call["result"]["details"]["timedOut"] = serde_json::json!(true);
        let (_, phase) = render_to_string(&call, 60, 12);
        assert_eq!(phase, BashPhase::Warning);
    }

    #[test]
    fn error_notice_output_renders_error_detail_block() {
        let mut call = sample_result_call();
        call["result"]["isError"] = serde_json::json!(true);
        call["result"]["content"][0]["text"] = serde_json::json!("Error: timed out after 30s");
        let (text, phase) = render_to_string(&call, 60, 12);
        assert_eq!(phase, BashPhase::Error);
        assert!(text.contains("  timed out after 30s"), "{text}");
        assert!(!text.contains("Error: timed out"), "{text}");
    }

    #[test]
    fn partial_stream_is_pending_phase_and_skips_stats() {
        let call = serde_json::json!({
            "name": "bash",
            "arguments": {"command": "long-running"},
            "output": "chunk-1\nchunk-2",
            "isPartial": true
        });
        let (text, phase) = render_to_string(&call, 60, 12);
        assert_eq!(phase, BashPhase::Pending);
        assert!(text.contains("chunk-1"), "{text}");
        assert!(!text.contains("Wall:"), "{text}");
    }

    #[test]
    fn truncation_warning_line_renders() {
        let mut call = sample_result_call();
        call["result"]["details"]["truncation"] = serde_json::json!({
            "direction": "tail", "totalLines": 500, "outputLines": 10,
            "shownRange": {"start": 491, "end": 500}
        });
        let (text, _) = render_to_string(&call, 80, 12);
        assert!(text.contains("⟨Showing lines 491-500 of 500⟩"), "{text}");
    }

    #[test]
    fn llm_notices_are_stripped_from_visible_output() {
        let details = serde_json::json!({
            "wallTimeMs": 1200, "exitCode": 0, "timeoutSeconds": 30,
            "truncation": {"direction": "tail", "totalLines": 10, "outputLines": 5,
                           "shownRange": {"start": 6, "end": 10}}
        });
        let body = "real output\n\n[Showing lines 6-10 of 10]";
        let call = serde_json::json!({
            "name": "bash",
            "arguments": {"command": "f"},
            "result": {"content": [{"type": "text", "text": body}], "details": details, "isError": false}
        });
        let (text, _) = render_to_string(&call, 80, 12);
        assert!(text.contains("real output"), "{text}");
        assert!(!text.contains("[Showing lines 6-10 of 10]\n"), "{text}");
        assert!(text.contains("⟨Showing lines 6-10 of 10⟩"), "{text}");
    }

    #[test]
    fn collapsed_output_tail_capped_with_marker() {
        let big: String = (1..=30)
            .map(|i| format!("line-{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let call = serde_json::json!({
            "name": "bash",
            "arguments": {"command": "f"},
            "result": {"content": [{"type": "text", "text": big}], "isError": false}
        });
        let (text, _) = render_to_string(&call, 60, 14);
        assert!(text.contains("earlier line"), "{text}");
        assert!(text.contains("line-30"), "{text}");
        assert!(!text.contains("line-1"), "{text}");
    }

    #[test]
    fn user_bash_card_running_shows_header_and_cancel_hint() {
        let backend = TestBackend::new(50, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let theme = Theme::default();
        terminal
            .draw(|f| {
                render_user_bash_card(
                    f,
                    f.area(),
                    "git status",
                    "on branch main",
                    &UserBashState::Running,
                    &theme,
                );
            })
            .unwrap();
        let buf = terminal.backend().buffer();
        let text = (0..10)
            .map(|y| (0..50).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("❯ Bash"), "{text}");
        assert!(text.contains("git status"), "{text}");
        assert!(text.contains("on branch main"), "{text}");
        assert!(text.contains("⟨esc: Cancel⟩"), "{text}");
    }

    #[test]
    fn user_bash_card_finished_shows_exit_stats() {
        let backend = TestBackend::new(50, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let theme = Theme::default();
        let state = UserBashState::Finished {
            exit_code: Some(2),
            wall_time_ms: Some(150),
        };
        terminal
            .draw(|f| {
                render_user_bash_card(f, f.area(), "false", "boom", &state, &theme);
            })
            .unwrap();
        let buf = terminal.backend().buffer();
        let text = (0..10)
            .map(|y| (0..50).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("⟨Wall: 0.15s | Exit: 2⟩"), "{text}");
        assert!(!text.contains("esc: Cancel"), "{text}");
        // Non-zero exit: error-colored border top-left cell.
        assert_eq!(buf[(0, 0)].fg, theme.color("error"));
    }
}
