//! Bounded eval-output buffering and streaming session artifacts.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

use super::super::util::{
    format_bytes, notice_with_artifact, truncate_head_lines, truncate_tail_lines, Elided,
    SpillConfig, ARTIFACT_MAX_BYTES, DEFAULT_MAX_BYTES,
};

/// Accumulates small cells normally; past the stream window it retains only
/// the head/tail needed for rendering and writes the full artifact incrementally.
pub(super) struct StreamingOutput {
    cfg: SpillConfig,
    state: State,
}

enum State {
    Inline(String),
    Streamed(Streamed),
}

struct Streamed {
    head: String,
    tail: String,
    tail_before: Option<u8>,
    total_bytes: usize,
    total_lines: usize,
    artifact: Option<StreamingArtifact>,
}

impl StreamingOutput {
    pub(super) fn new(cfg: SpillConfig) -> Self {
        Self {
            cfg,
            state: State::Inline(String::new()),
        }
    }

    pub(super) fn push(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match &mut self.state {
            State::Inline(inline) if inline.len() + text.len() <= DEFAULT_MAX_BYTES => {
                inline.push_str(text);
            }
            State::Inline(inline) => {
                let original = std::mem::take(inline);
                let head_at = floor_boundary(&original, DEFAULT_MAX_BYTES.min(original.len()));
                let mut head = original[..head_at].to_owned();
                let remaining = DEFAULT_MAX_BYTES.saturating_sub(head.len());
                head.push_str(&text[..floor_boundary(text, remaining)]);
                let mut tail = String::new();
                let mut tail_before = append_tail(&mut tail, &original);
                if let Some(byte) = append_tail(&mut tail, text) {
                    tail_before = Some(byte);
                }
                let total_lines =
                    count_lines(&original) + text.bytes().filter(|b| *b == b'\n').count();
                let mut artifact = StreamingArtifact::create();
                if let Some(file) = &mut artifact {
                    file.push(original.as_bytes());
                    file.push(text.as_bytes());
                }
                self.state = State::Streamed(Streamed {
                    head,
                    tail,
                    tail_before,
                    total_bytes: original.len() + text.len(),
                    total_lines,
                    artifact,
                });
            }
            State::Streamed(streamed) => {
                streamed.total_bytes += text.len();
                streamed.total_lines += text.bytes().filter(|b| *b == b'\n').count();
                if let Some(byte) = append_tail(&mut streamed.tail, text) {
                    streamed.tail_before = Some(byte);
                }
                if let Some(artifact) = &mut streamed.artifact {
                    artifact.push(text.as_bytes());
                }
            }
        }
    }

    /// Render exactly the existing inline/spill policy, without rebuilding the
    /// full string for outputs past the stream window.
    pub(super) fn finish(self) -> String {
        match self.state {
            State::Inline(text) => super::cap_output_with(&text, &self.cfg),
            State::Streamed(streamed) => {
                let elided = render_streamed(&streamed, self.cfg);
                let url = streamed
                    .artifact
                    .and_then(|artifact| artifact.finish(streamed.total_bytes));
                format!(
                    "{}\n\n[{}]",
                    elided.content,
                    notice_with_artifact(&elided.notice, url.as_deref())
                )
            }
        }
    }
}

fn count_lines(text: &str) -> usize {
    text.bytes().filter(|b| *b == b'\n').count() + 1
}

fn floor_boundary(text: &str, mut at: usize) -> usize {
    at = at.min(text.len());
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

fn ceil_boundary(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while at < text.len() && !text.is_char_boundary(at) {
        at += 1;
    }
    at
}

fn append_tail(tail: &mut String, text: &str) -> Option<u8> {
    tail.push_str(text);
    if tail.len() > DEFAULT_MAX_BYTES {
        let start = ceil_boundary(tail, tail.len() - DEFAULT_MAX_BYTES);
        let before = (start > 0).then(|| tail.as_bytes()[start - 1]);
        tail.drain(..start);
        before
    } else {
        None
    }
}

fn render_streamed(streamed: &Streamed, cfg: SpillConfig) -> Elided {
    let head_bytes = cfg.sink_head_bytes();
    if head_bytes == 0 {
        let tail = truncate_tail_lines(&streamed.tail, DEFAULT_MAX_BYTES, usize::MAX);
        let tail_start = streamed.tail.len() - tail.content.len();
        let partial = if tail_start > 0 {
            streamed.tail.as_bytes()[tail_start - 1] != b'\n'
        } else {
            streamed.tail_before.is_some_and(|byte| byte != b'\n')
        };
        let start = streamed.total_lines + 1 - tail.output_lines.min(streamed.total_lines);
        let notice = if partial {
            format!(
                "Showing line {start} (partial, {} of {}) of {}",
                format_bytes(tail.content.len()),
                format_bytes(streamed.total_bytes),
                streamed.total_lines
            )
        } else {
            format!(
                "Showing lines {start}-{} of {}",
                streamed.total_lines, streamed.total_lines
            )
        };
        return Elided {
            content: tail.content,
            notice,
        };
    }

    let head = truncate_head_lines(&streamed.head, head_bytes, usize::MAX);
    let byte_head = head.output_lines == 0;
    let head_text = if byte_head {
        &streamed.head[..floor_boundary(&streamed.head, head_bytes)]
    } else {
        head.content.as_str()
    };
    let tail_budget = (DEFAULT_MAX_BYTES - head_bytes).min(streamed.total_bytes - head_text.len());
    let tail = truncate_tail_lines(&streamed.tail, tail_budget, usize::MAX);
    let tail_start = streamed.tail.len() - tail.content.len();
    let partial_tail = tail_start > 0 && streamed.tail.as_bytes()[tail_start - 1] != b'\n';
    let head_kept = head.output_lines.max(1);
    let elided_bytes = streamed
        .total_bytes
        .saturating_sub(head_text.len() + tail.content.len());
    let elided_lines = streamed
        .total_lines
        .saturating_sub(head_kept + tail.output_lines);
    let marker = if elided_lines <= 1 {
        format!("[…{elided_bytes}B elided…]")
    } else {
        format!("[…{elided_lines}ln elided…]")
    };
    let notice = if byte_head || partial_tail {
        format!(
            "Showing head and tail bytes of {} line{}; {} elided",
            streamed.total_lines,
            if streamed.total_lines == 1 { "" } else { "s" },
            format_bytes(elided_bytes)
        )
    } else {
        format!(
            "Showing lines 1-{head_kept} and {}-{} of {}; {elided_lines} middle line{} ({}) elided",
            streamed.total_lines + 1 - tail.output_lines,
            streamed.total_lines,
            streamed.total_lines,
            if elided_lines == 1 { "" } else { "s" },
            format_bytes(elided_bytes)
        )
    };
    Elided {
        content: format!("{head_text}\n{marker}\n{}", tail.content),
        notice,
    }
}

struct StreamingArtifact {
    id: u64,
    file: File,
    len: usize,
    capped: bool,
    failed: bool,
    tail: Vec<u8>,
}

impl StreamingArtifact {
    fn create() -> Option<Self> {
        let dir = omp_urls::global_session_file()
            .as_ref()
            .and_then(|p| omp_urls::artifacts_directory_for(p))?;
        std::fs::create_dir_all(&dir).ok()?;
        for id in 0..=u64::MAX {
            let path = dir.join(format!("{id}.log"));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => {
                    return Some(Self {
                        id,
                        file,
                        len: 0,
                        capped: false,
                        failed: false,
                        tail: Vec::new(),
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(_) => return None,
            }
        }
        None
    }

    fn push(&mut self, mut bytes: &[u8]) {
        if self.failed {
            return;
        }
        if self.capped {
            append_tail_bytes(&mut self.tail, bytes);
            return;
        }
        let remaining = ARTIFACT_MAX_BYTES.saturating_sub(self.len);
        let inline = remaining.min(bytes.len());
        if self.file.write_all(&bytes[..inline]).is_err() {
            self.failed = true;
            return;
        }
        self.len += inline;
        bytes = &bytes[inline..];
        if bytes.is_empty() {
            return;
        }

        // The body cap keeps the first and last 32 MiB. The middle is already
        // on disk; retain its last window, truncate the middle away, then keep
        // only a rolling tail for subsequent frames.
        let half = ARTIFACT_MAX_BYTES / 2;
        let Some(head_len) = floor_file_boundary(&mut self.file, half) else {
            self.failed = true;
            return;
        };
        if self
            .file
            .seek(SeekFrom::Start((ARTIFACT_MAX_BYTES - half) as u64))
            .is_err()
        {
            self.failed = true;
            return;
        }
        self.tail.clear();
        let mut tail = Vec::new();
        if (&mut self.file)
            .take(half as u64)
            .read_to_end(&mut tail)
            .is_err()
        {
            self.failed = true;
            return;
        }
        self.tail = tail;
        append_tail_bytes(&mut self.tail, bytes);
        if self.file.set_len(head_len as u64).is_err() || self.file.seek(SeekFrom::End(0)).is_err()
        {
            self.failed = true;
            return;
        }
        self.len = head_len;
        self.capped = true;
    }

    fn finish(mut self, total_bytes: usize) -> Option<String> {
        if self.failed {
            return None;
        }
        if self.capped {
            let half = ARTIFACT_MAX_BYTES / 2;
            let tail_start = ceil_bytes_boundary(&self.tail, self.tail.len().saturating_sub(half));
            let tail = &self.tail[tail_start..];
            let elided = total_bytes.saturating_sub(self.len + tail.len());
            self.file
                .write_all(format!("\n[…{elided}B elided…]\n").as_bytes())
                .ok()?;
            self.file.write_all(tail).ok()?;
        }
        self.file.flush().ok()?;
        Some(format!("artifact://{}", self.id))
    }
}

fn append_tail_bytes(tail: &mut Vec<u8>, bytes: &[u8]) {
    tail.extend_from_slice(bytes);
    let half = ARTIFACT_MAX_BYTES / 2;
    if tail.len() > half {
        tail.drain(..tail.len() - half);
    }
}

fn floor_file_boundary(file: &mut File, at: usize) -> Option<usize> {
    let mut at = at;
    while at > 0 {
        let mut byte = [0];
        file.seek(SeekFrom::Start(at as u64)).ok()?;
        file.read_exact(&mut byte).ok()?;
        if byte[0] & 0b1100_0000 != 0b1000_0000 {
            return Some(at);
        }
        at -= 1;
    }
    Some(0)
}

fn ceil_bytes_boundary(bytes: &[u8], mut at: usize) -> usize {
    while at < bytes.len() && bytes[at] & 0b1100_0000 == 0b1000_0000 {
        at += 1;
    }
    at
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn check(full: &str, chunks: &[usize], cfg: SpillConfig) {
        let expected = crate::tools::util::sink_window(full, &cfg).expect("stream window");
        let mut output = StreamingOutput::new(cfg);
        let mut rest = full;
        for size in chunks {
            let at = floor_boundary(rest, (*size).min(rest.len()));
            output.push(&rest[..at]);
            rest = &rest[at..];
        }
        output.push(rest);
        let State::Streamed(streamed) = output.state else {
            panic!("test output must stream");
        };
        let actual = render_streamed(&streamed, cfg);
        assert_eq!(actual.content, expected.content);
        assert_eq!(actual.notice, expected.notice);
    }

    fn artifact_for_test(dir: &Path, id: u64) -> StreamingArtifact {
        StreamingArtifact {
            id,
            file: OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(dir.join(format!("{id}.log")))
                .expect("artifact"),
            len: 0,
            capped: false,
            failed: false,
            tail: Vec::new(),
        }
    }

    #[test]
    fn artifact_body_is_byte_identical_and_capped() {
        let dir = tempfile::tempdir().expect("tempdir");
        let full = "payload-0123456789\n".repeat(500_000);
        let mut artifact = artifact_for_test(dir.path(), 7);
        for chunk in full.as_bytes().chunks(31_337) {
            artifact.push(chunk);
        }
        assert_eq!(artifact.finish(full.len()).as_deref(), Some("artifact://7"));
        assert_eq!(
            std::fs::read(dir.path().join("7.log")).expect("body"),
            full.as_bytes()
        );

        let full = "x".repeat(ARTIFACT_MAX_BYTES + 12345);
        let mut artifact = artifact_for_test(dir.path(), 8);
        artifact.push(full.as_bytes());
        assert_eq!(artifact.finish(full.len()).as_deref(), Some("artifact://8"));
        let body = std::fs::read(dir.path().join("8.log")).expect("capped body");
        let half = ARTIFACT_MAX_BYTES / 2;
        let marker = format!("\n[…{}B elided…]\n", full.len() - ARTIFACT_MAX_BYTES);
        let mut expected = full.as_bytes()[..half].to_vec();
        expected.extend_from_slice(marker.as_bytes());
        expected.extend_from_slice(&full.as_bytes()[full.len() - half..]);
        assert_eq!(body, expected);
    }

    #[test]
    fn streamed_render_matches_full_string_across_chunks_and_unicode() {
        let line = "αβγ🙂payload-0123456789\n".repeat(5000);
        for size in [1, 7, 1024, 49_000, 70_000] {
            check(
                &line,
                &[size, size, size, size, size, size, size, size],
                SpillConfig::default(),
            );
        }
        check(
            &"x".repeat(8 * 1024 * 1024),
            &[8 * 1024 * 1024],
            SpillConfig::default(),
        );
        check(
            &"x".repeat(8 * 1024 * 1024),
            &[8 * 1024 * 1024],
            SpillConfig {
                head_bytes: 0,
                ..SpillConfig::default()
            },
        );
    }
}
