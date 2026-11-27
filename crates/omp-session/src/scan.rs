//! One-pass session-file scanning: the file is read into a single buffer
//! (memory-mapped above [`MMAP_THRESHOLD`], plain read below) and split into
//! lines with `memchr`, so scans that need only a few fields per line
//! deserialize into borrowed structs instead of building a `serde_json::Value`
//! tree per entry.
//!
//! The line iteration reproduces `BufRead::split(b'\n')` exactly: segments
//! between newlines, a final segment only when the file does not end with a
//! newline (a truncated last line is still a line), an empty file yields
//! nothing, and blank segments are yielded (callers keep their own numbering).

use std::borrow::Cow;
use std::fs::File;
use std::ops::ControlFlow;
use std::path::Path;

use anyhow::{Context, Result};
use memchr::memchr_iter;

/// Files larger than this are memory-mapped instead of read into a `Vec<u8>`
/// (1 MiB).
pub const MMAP_THRESHOLD: u64 = 1024 * 1024;

/// Buffers larger than this are parsed with rayon (line spans first, then
/// parallel parse, results reassembled in file order).
pub const PAR_THRESHOLD: usize = 4 * 1024 * 1024;

/// A whole session file in memory: an owned buffer at or below
/// [`MMAP_THRESHOLD`], a read-only mmap of the file above it.
pub struct FileBuf {
    owned: Vec<u8>,
    mapped: Option<memmap2::Mmap>,
}

impl FileBuf {
    /// Read `path` whole, choosing the mmap path by file size.
    ///
    /// # Errors
    /// I/O or mmap failure.
    pub fn read(path: &Path) -> Result<Self> {
        Self::read_with(path, None)
    }

    fn read_with(path: &Path, force_mmap: Option<bool>) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let size = file
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        let use_mmap = force_mmap.unwrap_or(size > MMAP_THRESHOLD);
        if use_mmap {
            // SAFETY: read-only mapping of a file we just opened; no pointer
            // aliasing beyond the mmap itself, unmapped when `FileBuf` drops.
            let mapped = unsafe { memmap2::Mmap::map(&file) }
                .with_context(|| format!("mmap {}", path.display()))?;
            Ok(Self {
                owned: Vec::new(),
                mapped: Some(mapped),
            })
        } else {
            let owned = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
            Ok(Self {
                owned,
                mapped: None,
            })
        }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        match &self.mapped {
            Some(mapped) => &mapped[..],
            None => &self.owned,
        }
    }

    /// Whether this buffer is a memory map rather than an owned read.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        self.mapped.is_some()
    }
}

/// Call `f` for every line of `buf` (`BufRead::split(b'\n')` semantics):
/// `line_no` is 1-based and counts every segment, blank or not. `f` may stop
/// the scan with [`ControlFlow::Break`].
///
/// # Errors
/// The callback's error.
pub fn for_each_line<'a>(
    buf: &'a [u8],
    mut f: impl FnMut(usize, &'a [u8]) -> Result<ControlFlow<()>>,
) -> Result<()> {
    let mut start = 0usize;
    let mut line_no = 0usize;
    for at in memchr_iter(b'\n', buf) {
        line_no += 1;
        if f(line_no, &buf[start..at])?.is_break() {
            return Ok(());
        }
        start = at + 1;
    }
    if start < buf.len() {
        line_no += 1;
        let _ = f(line_no, &buf[start..])?;
    }
    Ok(())
}

/// Byte spans of every line of `buf` (`BufRead::split(b'\n')` semantics):
/// `(line_no, offset, len)` with `len` excluding the newline, blanks included,
/// a final truncated line included. Line `n` occupies `[offset, offset+len)`;
/// its `offset` is the sum of all prior `len + 1`.
#[must_use]
pub fn line_spans(buf: &[u8]) -> Vec<(usize, u32, u32)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut line_no = 0usize;
    for at in memchr_iter(b'\n', buf) {
        line_no += 1;
        out.push((line_no, start as u32, (at - start) as u32));
        start = at + 1;
    }
    if start < buf.len() {
        line_no += 1;
        out.push((line_no, start as u32, (buf.len() - start) as u32));
    }
    out
}

/// Decode one raw line like the streaming loader does: lossy UTF-8, trailing
/// `\r` stripped (CRLF files).
#[must_use]
pub fn decode_line(bytes: &[u8]) -> Cow<'_, str> {
    let text = String::from_utf8_lossy(bytes);
    if let Some(stripped) = text.strip_suffix('\r') {
        Cow::Owned(stripped.to_owned())
    } else {
        text
    }
}

/// The cheap borrowed per-line parse: only the top-level fields callers read
/// (`type` / `id` / `parentId` / `timestamp`). Strings are `Cow`: borrowed
/// from the line when the JSON string has no escapes, owned (decoded) when it
/// does, so escaped ids still scan.
#[derive(serde::Deserialize)]
pub struct HeadScan<'a> {
    /// The `type` field.
    #[serde(rename = "type", default, borrow)]
    pub kind: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    pub id: Option<Cow<'a, str>>,
    #[serde(rename = "parentId", default, borrow)]
    pub parent_id: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    pub timestamp: Option<Cow<'a, str>>,
}

/// [`HeadScan`] from raw line bytes: the fast path is a borrowed typed parse;
/// lines that are JSON objects with unusual field shapes (non-string fields
/// where a string is expected) fall back to a `Value` parse so they still scan
/// (with `None` for the fields a string type cannot produce). Returns `None`
/// when the line is not a JSON object at all.
#[must_use]
pub fn head_of(bytes: &[u8]) -> Option<HeadScan<'_>> {
    if let Ok(head) = serde_json::from_slice::<HeadScan>(bytes) {
        return Some(head);
    }
    let value = serde_json::from_slice::<serde_json::Value>(bytes).ok()?;
    let obj = value.as_object()?;
    let owned = |k: &str| {
        obj.get(k)
            .and_then(serde_json::Value::as_str)
            .map(|s| Cow::Owned(s.to_owned()))
    };
    Some(HeadScan {
        kind: owned("type"),
        id: owned("id"),
        parent_id: owned("parentId"),
        timestamp: owned("timestamp"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmpfile(name: &str, body: &[u8]) -> std::path::PathBuf {
        let p =
            std::env::temp_dir().join(format!("omp-session-scan-{name}-{}", std::process::id()));
        let mut f = File::create(&p).unwrap();
        f.write_all(body).unwrap();
        p
    }

    fn lines_of(buf: &[u8]) -> Vec<(usize, String)> {
        let mut out = Vec::new();
        for_each_line(buf, |n, b| {
            out.push((n, decode_line(b).into_owned()));
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
        out
    }

    #[test]
    fn line_splitting_matches_bufread_split() {
        assert!(lines_of(b"").is_empty(), "empty file");
        assert_eq!(
            lines_of(b"a\nb\nc"),
            vec![(1, "a".into()), (2, "b".into()), (3, "c".into())]
        );
        assert_eq!(lines_of(b"a\n"), vec![(1, "a".into())]);
        assert_eq!(
            lines_of(b"a\n\nb\n"),
            vec![(1, "a".into()), (2, "".into()), (3, "b".into())]
        );
        // CRLF: the trailing \r is stripped.
        assert_eq!(
            lines_of(b"a\r\nb\r\n"),
            vec![(1, "a".into()), (2, "b".into())]
        );
        // Truncated last line (no trailing newline) is still a line.
        assert_eq!(
            lines_of(b"{\"a\":1}\n{\"b\":2"),
            vec![(1, "{\"a\":1}".into()), (2, "{\"b\":2".into())]
        );
    }

    #[test]
    fn head_scan_borrows_fields() {
        let line = br#"{"type":"message","id":"a","parentId":"h","timestamp":"t"}"#;
        let head = head_of(line).unwrap();
        assert_eq!(head.kind.as_deref(), Some("message"));
        assert_eq!(head.id.as_deref(), Some("a"));
        assert_eq!(head.parent_id.as_deref(), Some("h"));
    }

    #[test]
    fn head_scan_rejects_non_objects_and_falls_back() {
        assert!(head_of(b"not json").is_none());
        assert!(head_of(b"[1,2]").is_none());
        // id is not a string: the borrowed parse fails, the Value fallback
        // still scans the object with a None id.
        let head = head_of(br#"{"type":"message","id":5}"#).unwrap();
        assert_eq!(head.kind.as_deref(), Some("message"));
        assert_eq!(head.id.as_deref(), None);
    }

    #[test]
    fn mmap_and_buffered_paths_agree() {
        let header = r#"{"type":"session","version":3,"id":"hdr","timestamp":"t","cwd":"/tmp"}"#;
        let body = format!("{header}\n");
        let _big_line = format!(
            r#"{{"type":"message","id":"m{}","parentId":"hdr","timestamp":"t","message":{{"role":"user","text":"{}"}}}}"#,
            0,
            "x".repeat(4096)
        );
        let mut body_bytes = body.into_bytes();
        for i in 0..400 {
            body_bytes.extend_from_slice(
                format!(
                    r#"{{"type":"message","id":"m{i}","parentId":"hdr","timestamp":"t","message":{{"role":"user","text":"{}"}}}}"#,
                    "x".repeat(4096)
                )
                .as_bytes(),
            );
            body_bytes.push(b'\n');
        }
        body_bytes.extend_from_slice(b"{\"type\":\"message\",\"id\":\"last\"");
        let path = tmpfile("mmap", &body_bytes);
        let size = std::fs::metadata(&path).unwrap().len();
        assert!(
            size > MMAP_THRESHOLD,
            "test file must exceed the mmap threshold"
        );

        let buffered = FileBuf::read_with(&path, Some(false)).unwrap();
        let mapped = FileBuf::read_with(&path, Some(true)).unwrap();
        assert!(!buffered.is_mapped());
        assert!(mapped.is_mapped());
        let auto = FileBuf::read(&path).unwrap();
        assert!(
            auto.is_mapped(),
            "auto path must pick mmap above the threshold"
        );

        let mut from_buffered = lines_of(buffered.as_slice());
        let from_mapped = lines_of(mapped.as_slice());
        let from_auto = lines_of(auto.as_slice());
        assert_eq!(from_buffered, from_mapped, "mmap == buffered");
        assert_eq!(from_buffered, from_auto, "auto == buffered");
        assert_eq!(from_buffered.len(), 402);
        let last = from_buffered.pop().unwrap();
        assert_eq!(last, (402, "{\"type\":\"message\",\"id\":\"last\"".into()));
        std::fs::remove_file(&path).ok();
    }
}
