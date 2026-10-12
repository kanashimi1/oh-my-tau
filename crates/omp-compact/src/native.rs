//! Provider-native compaction payloads and the snapcompact frame-archive replay
//! (coding-agent `session/session-context.ts#getOpenAiRemoteCompactionPayload`,
//! pi-agent-core `compaction/anthropic.ts#getAnthropicCompactionPayload`,
//! snapcompact `getPreservedArchive` / `historyBlocks` + the replay-side helpers).
//!
//! Only the context-rebuild side is ported: the payloads are read from a
//! compaction entry's `preserveData` and replayed onto the summary message. The
//! commit side (running the remote / Anthropic-native compaction request) is
//! harness work and stays out of scope. Snapcompact frame payloads are
//! persisted as `blob:sha256:` references; without a blob-store resolver every
//! referenced frame replays as an "unavailable" gap marker, exactly like TS
//! called without `resolveFrameData`.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{json, Map, Value};

use crate::json::str_field;

/// `FRAME_DATA_BYTES_BUDGET`: hard cap on image base64 bytes attached to one
/// rebuilt provider request.
pub const FRAME_DATA_BYTES_BUDGET: usize = 3_000_000;

const LEGACY_FRAME_COUNT_GUARD: usize = 16;
const LEGACY_ARCHIVE_TEXT_GUARD: usize = 250_000;
const LEGACY_TRUNCATED_CHARS_GUARD: f64 = 1_000_000.0;
const DAMAGED_PAYLOAD_MIN_CHARS: usize = 40;
/// `NEWLINE_GLYPH`.
const NEWLINE_GLYPH: char = '\u{2588}';
const BLOB_REFERENCE_PREFIX: &str = "blob:sha256:";

/// `snapcompact.Archive` reduced to what a context rebuild reads.
#[derive(Debug, Clone, PartialEq)]
pub struct Archive {
    pub frames: Vec<Frame>,
    pub truncated_chars: f64,
    /// Full kept archive source (re-rendered each compaction in TS).
    pub text: Option<String>,
    pub text_head: Option<String>,
    pub text_tail: Option<String>,
}

/// `snapcompact.Frame` reduced to what a context rebuild reads.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub data: String,
    pub mime_type: String,
    /// Shape metadata (absent on legacy frames).
    pub font: Option<String>,
    pub variant: Option<String>,
    /// Resolution hint forwarded to the provider.
    pub detail: Option<String>,
}

/// Validated `OpenAI` Responses replacement history (`openaiResponsesHistory` payload).
#[must_use]
pub fn openai_remote_payload(compaction: &Value) -> Option<Value> {
    let candidate = compaction
        .get("preserveData")?
        .get("openaiRemoteCompaction")?
        .as_object()?;
    let provider = candidate.get("provider").and_then(Value::as_str)?;
    if provider.is_empty() {
        return None;
    }
    let items = candidate.get("replacementHistory")?.as_array()?;
    if !items.iter().all(Value::is_object) {
        return None;
    }
    Some(json!({"type": "openaiResponsesHistory", "provider": provider, "items": items}))
}

/// Validated Anthropic native compaction replay payload (`anthropicCompaction`).
#[must_use]
pub fn anthropic_compaction_payload(compaction: &Value) -> Option<Value> {
    let candidate = compaction.get("preserveData")?.get("anthropicCompaction")?;
    let provider = str_field(candidate, "provider").filter(|p| !p.is_empty())?;
    let content = str_field(candidate, "content").filter(|c| !c.is_empty())?;
    let opt_str = |key: &str| {
        str_field(candidate, key)
            .filter(|v| !v.is_empty())
            .map(str::to_owned)
    };
    let mut payload = Map::new();
    payload.insert("type".into(), "anthropicCompaction".into());
    payload.insert("provider".into(), provider.into());
    payload.insert("content".into(), content.into());
    if let Some(signature) = opt_str("signature") {
        payload.insert("signature".into(), signature.into());
    }
    if let Some(encrypted) = opt_str("encryptedContent") {
        payload.insert("encryptedContent".into(), encrypted.into());
    }
    if let Some(files) = opt_str("filesText") {
        payload.insert("filesText".into(), files.into());
    }
    Some(Value::Object(payload))
}

/// `getPreservedArchive`: validate and extract a persisted frame archive.
#[must_use]
pub fn get_preserved_archive(preserve_data: Option<&Value>) -> Option<Archive> {
    let candidate = preserve_data?.get("snapcompact")?;
    let frames = candidate
        .get("frames")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .filter_map(|frame| {
                    let data = str_field(frame, "data").filter(|d| !d.is_empty())?;
                    let mime = str_field(frame, "mimeType")?;
                    frame.get("cols").and_then(Value::as_f64)?;
                    frame.get("rows").and_then(Value::as_f64)?;
                    frame.get("chars").and_then(Value::as_f64)?;
                    Some(Frame {
                        data: data.to_owned(),
                        mime_type: mime.to_owned(),
                        font: str_field(frame, "font").map(str::to_owned),
                        variant: str_field(frame, "variant").map(str::to_owned),
                        detail: str_field(frame, "detail").map(str::to_owned),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let text = |key: &str| {
        str_field(candidate, key)
            .filter(|t| !t.is_empty())
            .map(str::to_owned)
    };
    let source_text = text("text");
    let text_head = text("textHead");
    let text_tail = text("textTail");
    if frames.is_empty() && source_text.is_none() && text_head.is_none() && text_tail.is_none() {
        return None;
    }
    Some(Archive {
        frames,
        truncated_chars: candidate
            .get("truncatedChars")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        text: source_text,
        text_head,
        text_tail,
    })
}

/// `hasLegacySnapcompactFrames`: frames without shape metadata.
#[must_use]
pub fn has_legacy_frames(archive: &Archive) -> bool {
    archive
        .frames
        .iter()
        .any(|f| f.font.is_none() && f.variant.is_none())
}

/// `snapcompactFrameDataBytes` (no blob resolver: raw payload lengths).
#[must_use]
pub fn frame_data_bytes(archive: &Archive) -> usize {
    archive.frames.iter().map(|f| f.data.len()).sum()
}

/// `archiveSourceText`: the archive's source text as plain text for LLM use.
#[must_use]
pub fn archive_source_text(archive: &Archive) -> Option<String> {
    let text = archive_text(archive)?;
    let elided = elide_data_urls(&to_plain_text(&text), true);
    if elided.is_empty() {
        None
    } else {
        Some(elided)
    }
}

fn archive_text(archive: &Archive) -> Option<String> {
    if let Some(text) = archive.text.as_deref().filter(|t| !t.is_empty()) {
        return Some(text.to_owned());
    }
    let mut parts: Vec<&str> = Vec::new();
    if let Some(head) = archive.text_head.as_deref().filter(|t| !t.is_empty()) {
        parts.push(head);
    }
    if let Some(tail) = archive.text_tail.as_deref().filter(|t| !t.is_empty()) {
        parts.push(tail);
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(&NEWLINE_GLYPH.to_string()))
    }
}

/// `isCrashRiskLegacySnapcompactArchive`: the conservative LLM-payload guard for
/// pre-shape-metadata frames.
#[must_use]
pub fn is_crash_risk_legacy_archive(archive: &Archive) -> bool {
    let source_len = archive_source_text(archive).map_or(0, |t| t.len());
    has_legacy_frames(archive)
        && frame_data_bytes(archive) >= FRAME_DATA_BYTES_BUDGET
        && (archive.frames.len() >= LEGACY_FRAME_COUNT_GUARD
            || archive.truncated_chars >= LEGACY_TRUNCATED_CHARS_GUARD
            || source_len >= LEGACY_ARCHIVE_TEXT_GUARD)
}

/// `snapcompactHistoryBlockOptions` + `historyBlocks` for the model context:
/// a legacy crash-risk archive loses its frame payloads entirely, everything
/// else keeps frames within [`FRAME_DATA_BYTES_BUDGET`].
#[must_use]
pub fn history_blocks_for_context(archive: &Archive) -> Vec<Value> {
    let max = if is_crash_risk_legacy_archive(archive) {
        0
    } else {
        FRAME_DATA_BYTES_BUDGET
    };
    history_blocks(archive, Some(max))
}

/// `historyBlocks`: ordered archive blocks, oldest to newest.
#[must_use]
pub fn history_blocks(archive: &Archive, max_frame_data_bytes: Option<usize>) -> Vec<Value> {
    let mut blocks = frame_blocks(&images_within_budget(archive, max_frame_data_bytes));
    let has_images = blocks.iter().any(|b| str_field(b, "type") == Some("image"));
    let has_omitted_images = blocks.iter().any(|b| str_field(b, "type") == Some("text"));
    if let Some(head) = &archive.text_head {
        let suffix = if has_images {
            "\n-------------- imaged middle below\n"
        } else {
            ""
        };
        let text = format!("{}{}", elide_data_urls(&to_plain_text(head), true), suffix);
        blocks.insert(0, json!({"type": "text", "text": text}));
    }
    if let Some(tail) = &archive.text_tail {
        let prefix = if has_images {
            "\n-------------- imaged middle above\n"
        } else if archive.truncated_chars > 0.0 || has_omitted_images {
            "\n-------------- middle history omitted above\n"
        } else {
            ""
        };
        let text = format!("{}{}", prefix, elide_data_urls(&to_plain_text(tail), true));
        match blocks
            .last_mut()
            .filter(|b| str_field(b, "type") == Some("text"))
        {
            Some(last) => {
                if let Some(obj) = last.as_object_mut() {
                    let prev = obj
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned();
                    obj.insert("text".into(), Value::String(format!("{prev}{text}")));
                }
            }
            None => blocks.push(json!({"type": "text", "text": text})),
        }
    }
    blocks
}

fn is_blob_reference(data: &str) -> bool {
    data.starts_with(BLOB_REFERENCE_PREFIX)
}

/// One reconstructed slot: a usable frame, an unavailable gap, or a byte-budget gap.
enum Slot {
    Frame(Frame),
    Unavailable,
    Omitted(usize),
}

fn images_within_budget(archive: &Archive, max_frame_data_bytes: Option<usize>) -> Vec<Slot> {
    let has_unresolved = archive.frames.iter().any(|f| is_blob_reference(&f.data));
    if max_frame_data_bytes.is_none() && !has_unresolved {
        return archive.frames.iter().cloned().map(Slot::Frame).collect();
    }
    let mut used_bytes = 0usize;
    let mut newest_first = Vec::new();
    for frame in archive.frames.iter().rev() {
        if is_blob_reference(&frame.data) {
            newest_first.push(Slot::Unavailable);
            continue;
        }
        let bytes = frame.data.len();
        if max_frame_data_bytes.is_some_and(|max| used_bytes + bytes > max) {
            newest_first.push(Slot::Omitted(bytes));
            continue;
        }
        used_bytes += bytes;
        newest_first.push(Slot::Frame(frame.clone()));
    }
    newest_first.reverse();
    newest_first
}

/// `toLocaleString()` (en-US) grouping for gap-notice counts.
fn locale_number(n: usize) -> String {
    let digits = n.to_string();
    let mut grouped = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(c);
    }
    grouped
}

fn format_frame_data_bytes(bytes: usize) -> String {
    if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{bytes} B")
    }
}

fn unavailable_frame_notice(count: usize) -> String {
    format!(
        "-------------- {} archived image frame{} unavailable here --------------",
        locale_number(count),
        if count == 1 { "" } else { "s" }
    )
}

fn omitted_frame_notice(omitted_frames: usize, omitted_bytes: usize) -> String {
    let budget_note = if omitted_bytes > 0 {
        format!(
            " {} of base64 exceeded the per-request snapcompact payload budget.",
            format_frame_data_bytes(omitted_bytes)
        )
    } else {
        String::new()
    };
    format!(
        "-------------- snapcompact image middle omitted\n{} archived image frame{} could not be included.{}\nThe compacted summary and visible text edges remain available.\n--------------",
        locale_number(omitted_frames),
        if omitted_frames == 1 { "" } else { "s" },
        budget_note
    )
}

/// `frameBlocks`: blocks for the imaged middle, gap runs merged per cause.
fn frame_blocks(slots: &[Slot]) -> Vec<Value> {
    fn flush(
        blocks: &mut Vec<Value>,
        pending_gap: &mut Option<&'static str>,
        pending_frames: &mut usize,
        pending_bytes: &mut usize,
    ) {
        match *pending_gap {
            Some("unavailable") => blocks.push(json!({"type": "text", "text": unavailable_frame_notice(*pending_frames)})),
            Some("budget") => blocks.push(json!({"type": "text", "text": omitted_frame_notice(*pending_frames, *pending_bytes)})),
            _ => {}
        }
        *pending_gap = None;
        *pending_frames = 0;
        *pending_bytes = 0;
    }
    let mut blocks = Vec::new();
    let mut pending_gap: Option<&'static str> = None;
    let mut pending_frames = 0usize;
    let mut pending_bytes = 0usize;
    for slot in slots {
        match slot {
            Slot::Frame(frame) => {
                flush(
                    &mut blocks,
                    &mut pending_gap,
                    &mut pending_frames,
                    &mut pending_bytes,
                );
                let mut image = Map::new();
                image.insert("type".into(), "image".into());
                image.insert("data".into(), frame.data.clone().into());
                image.insert("mimeType".into(), frame.mime_type.clone().into());
                if let Some(detail) = &frame.detail {
                    image.insert("detail".into(), detail.clone().into());
                }
                blocks.push(Value::Object(image));
            }
            Slot::Unavailable => {
                if pending_gap == Some("budget") {
                    flush(
                        &mut blocks,
                        &mut pending_gap,
                        &mut pending_frames,
                        &mut pending_bytes,
                    );
                }
                pending_gap = Some("unavailable");
                pending_frames += 1;
            }
            Slot::Omitted(bytes) => {
                if pending_gap == Some("unavailable") {
                    flush(
                        &mut blocks,
                        &mut pending_gap,
                        &mut pending_frames,
                        &mut pending_bytes,
                    );
                }
                pending_gap = Some("budget");
                pending_frames += 1;
                pending_bytes += *bytes;
            }
        }
    }
    flush(
        &mut blocks,
        &mut pending_gap,
        &mut pending_frames,
        &mut pending_bytes,
    );
    blocks
}

/// `toPlainText`: drop dim toggles, print newline glyphs as real newlines.
#[must_use]
pub fn to_plain_text(text: &str) -> String {
    text.replace(['\u{000e}', '\u{000f}'], "")
        .replace(NEWLINE_GLYPH, "\n")
}

static ELIDED_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\s*\[(?:…|\.{3})[0-9]+ch elided(?:…|\.{3})\]\s*").expect("static regex")
});

static CANONICAL_BASE64: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:[0-9A-Za-z+/]{4})*(?:[0-9A-Za-z+/]{4}|[0-9A-Za-z+/]{3}=|[0-9A-Za-z+/]{2}==)$")
        .expect("static regex")
});

static DATA_URL_ATOM: LazyLock<Regex> = LazyLock::new(|| {
    let media_token = r"[0-9A-Za-z_!#$%&'*+.^|~-]+";
    Regex::new(&format!(
        r"(?i)data:([0-9A-Za-z][0-9A-Za-z_.+-]*/[0-9A-Za-z_.+-]+(?:;{media_token}={media_token})*);base64,([0-9A-Za-z+/=]*(?:\s*\[(?:…|\.\.\.)[0-9]+ch elided(?:…|\.\.\.)\]\s*[0-9A-Za-z+/=]*)?)(\s*\))?"
    ))
    .expect("static regex")
});

/// Start of `!?[label](\s*` immediately before `data_index`, or `None`
/// (`adjacentMarkdownOpenerStart`); the opener must lie in `[cursor, data_index)`.
fn adjacent_markdown_opener_start(text: &str, data_index: usize, cursor: usize) -> Option<usize> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let char_before = |pos: usize| -> Option<(usize, char)> {
        chars
            .iter()
            .rev()
            .find(|(off, _)| *off < pos)
            .map(|(off, c)| (*off, *c))
    };
    let mut i = data_index;
    while i > cursor {
        let Some((off, c)) = char_before(i) else {
            break;
        };
        if !c.is_whitespace() {
            break;
        }
        i = off;
    }
    if i <= cursor {
        return None;
    }
    let Some((paren_off, '(')) = char_before(i) else {
        return None;
    };
    if paren_off <= cursor {
        return None;
    }
    let Some((bracket_off, ']')) = char_before(paren_off) else {
        return None;
    };
    if bracket_off < cursor {
        return None;
    }
    let mut opener: Option<usize> = None;
    for &(off, c) in chars.iter().rev() {
        if off >= paren_off - 1 || off < cursor {
            continue;
        }
        if c == ']' || c == '\n' {
            break;
        }
        if c == '[' {
            opener = Some(off);
        }
    }
    let opener = opener?;
    if opener > cursor {
        if let Some((bang_off, '!')) = char_before(opener) {
            if bang_off >= cursor {
                return Some(bang_off);
            }
        }
    }
    Some(opener)
}

/// `elideDataUrls`: replace every inline base64 data URL atom — Markdown
/// wrapper included — with its metadata placeholder. `archive_context` marks
/// archive text (sliced at arbitrary offsets), where every recognized prefix
/// is elided; intact source text keeps short prose mentions.
/// One elided data URL: byte span in the source, mime, payload, closing wrapper.
type MatchedDataUrl<'a> = (usize, usize, Option<&'a str>, String, Option<String>);

#[must_use]
pub fn elide_data_urls(text: &str, archive_context: bool) -> String {
    if !text.to_lowercase().contains(";base64,") {
        return text.to_owned();
    }
    let mut out: Vec<String> = Vec::new();
    let mut cursor = 0usize;
    let mut matched: Vec<MatchedDataUrl<'_>> = Vec::new();
    for caps in DATA_URL_ATOM.captures_iter(text) {
        let whole = caps.get(0).expect("match");
        matched.push((
            whole.start(),
            whole.end(),
            caps.get(1).map(|m| m.as_str()),
            caps.get(2).map_or("", |m| m.as_str()).to_owned(),
            caps.get(3).map(|m| m.as_str().to_owned()),
        ));
    }
    for (url_start, url_end, mime, payload, closer) in matched {
        let marker = ELIDED_MARKER_RE.find(&payload);
        let is_atom = archive_context
            || marker.is_some()
            || CANONICAL_BASE64.is_match(&payload)
            || payload.len() >= DAMAGED_PAYLOAD_MIN_CHARS;
        if !is_atom {
            out.push(text[cursor..url_end].to_owned());
            cursor = url_end;
            continue;
        }
        let b64_chars = match &marker {
            Some(m) => {
                let digits: String = m.as_str().chars().filter(char::is_ascii_digit).collect();
                payload.len() - m.as_str().len() + digits.parse::<usize>().unwrap_or(0)
            }
            None => payload.len(),
        };
        let placeholder = format!(
            "[data URL omitted: {}, {b64_chars} base64 chars]",
            mime.unwrap_or("")
        );
        let opener_start =
            adjacent_markdown_opener_start(text, url_start, cursor).filter(|o| *o >= cursor);
        let emit_start = opener_start.unwrap_or(url_start);
        out.push(text[cursor..emit_start].to_owned());
        if opener_start.is_some() && closer.is_some() {
            out.push(placeholder);
        } else {
            if let Some(opener_at) = opener_start {
                out.push(text[opener_at..url_start].to_owned());
            }
            out.push(placeholder);
            out.push(closer.unwrap_or_default());
        }
        cursor = url_end;
    }
    out.push(text[cursor..].to_owned());
    out.join("")
}
