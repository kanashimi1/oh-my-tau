//! System clipboard reads — port of the read paths of
//! `coding-agent/src/utils/clipboard.ts` (`readTextFromClipboard` /
//! `readImageFromClipboard`).
//!
//! Like TS, reads shell out to the platform clipboard tools; there is no
//! in-process clipboard backend in omp-rs, so every platform branch is a
//! subprocess:
//!
//! - WSL: `powershell.exe` over interop (WSLg has no image transport and its
//!   Wayland clipboard only works when `wl-clipboard` happens to be
//!   installed) — the `Get-Clipboard -Raw` text one-liner and the
//!   clipboard-image→base64-PNG script, both verbatim from TS.
//! - Wayland: `wl-paste` (text via `--type text --no-newline`; images via
//!   `--list-types` + a typed read for the first supported MIME).
//! - X11: `xclip -selection clipboard -o`, falling back to `xsel`.
//! - macOS: `pbpaste`; Termux: `termux-clipboard-get`.
//!
//! Testability: `OMP_CLIPBOARD_TEXT_GET` / `OMP_CLIPBOARD_IMAGE_GET` name a
//! command (whitespace-split argv, resolved from `PATH`) whose stdout stands
//! in for the clipboard text / PNG image. They short-circuit every platform
//! branch, so the e2e harness can stub the clipboard without touching PATH.
//!
//! Reads are blocking (`capture` enforces the TS 2 s text / 8 s image
//! budgets); callers run them off the render thread (`App::poll_deadlines`
//! drains the result).

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::chrome::title::is_wsl;

/// Timeout for text reads (TS `spawnCapture` default).
const TEXT_TIMEOUT: Duration = Duration::from_secs(2);
/// Timeout for the PowerShell bridges (TS `POWERSHELL_TIMEOUT_MS`).
const IMAGE_TIMEOUT: Duration = Duration::from_secs(8);

/// Image MIME types the model layer accepts, in probe priority order
/// (TS `IMAGE_MIME_PRIORITY` / `SUPPORTED_IMAGE_MIME_TYPES`).
const IMAGE_MIME_TYPES: &[&str] = &["image/png", "image/jpeg", "image/webp", "image/gif"];

/// A clipboard image payload (always PNG in omp-rs: the only bridge that
/// produces images re-encodes to PNG, like TS's PowerShell bridge).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipboardImage {
    pub data: Vec<u8>,
    /// Always `image/png` (TS re-encodes the host bitmap the same way).
    pub mime: &'static str,
}

/// One outcome of a clipboard read, shipped back to the render loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClipboardOutcome {
    /// A clipboard bitmap (PNG bytes) for `ctrl+v` smart paste.
    Image(Vec<u8>),
    /// Clipboard text for smart paste (routed through the paste pipeline).
    Text(String),
    /// Clipboard text for `ctrl+shift+v`: inserted verbatim, no collapse.
    TextRaw(String),
    /// The clipboard is reachable but empty.
    Empty,
    /// Every bridge failed or timed out.
    Failed,
}

/// Which paste chord fired (TS `handleImagePaste` / `handleClipboardTextRawPaste`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PasteRequest {
    /// `ctrl+v`: image if present, else smart-paste the text.
    Smart,
    /// `ctrl+shift+v` / `alt+shift+v`: the clipboard text verbatim.
    TextRaw,
}

/// Run one clipboard read (blocking; call off the render thread).
#[must_use]
pub fn perform(request: PasteRequest) -> ClipboardOutcome {
    match request {
        PasteRequest::TextRaw => {
            let text = read_text();
            if text.is_empty() {
                ClipboardOutcome::Empty
            } else {
                ClipboardOutcome::TextRaw(text)
            }
        }
        PasteRequest::Smart => {
            if let Some(image) = read_image() {
                return ClipboardOutcome::Image(image.data);
            }
            let text = read_text();
            if text.is_empty() {
                ClipboardOutcome::Empty
            } else {
                ClipboardOutcome::Text(text)
            }
        }
    }
}

/// PowerShell one-liner emitting the clipboard text verbatim (TS
/// `POWERSHELL_TEXT_SCRIPT`): UTF-8 forced output encoding, `Get-Clipboard
/// -Raw` through `[Console]::Out.Write` (no trailing newline).
const POWERSHELL_TEXT_SCRIPT: &str = r#"
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
[Console]::Out.Write([string](Get-Clipboard -Raw))
"#;

/// PowerShell script emitting the clipboard image as base64 PNG, or nothing
/// when the clipboard holds no image (TS `POWERSHELL_IMAGE_SCRIPT`).
const POWERSHELL_IMAGE_SCRIPT: &str = r#"
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$img = [System.Windows.Forms.Clipboard]::GetImage()
if ($img -ne $null) {
	$ms = New-Object System.IO.MemoryStream
	$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
	[Console]::Out.Write([Convert]::ToBase64String($ms.ToArray()))
}
"#;

/// Run a stub command from an override spec (`OMP_CLIPBOARD_TEXT_GET` /
/// `OMP_CLIPBOARD_IMAGE_GET` value): whitespace-split argv, resolved from
/// `PATH`. Trailing newlines are stripped (a stub `printf`/`cat` pipeline
/// should behave like a clipboard read, not `echo`).
fn override_bytes(spec: &str) -> Option<Vec<u8>> {
    let mut parts = spec.split_whitespace();
    let prog = parts.next()?;
    capture(Command::new(prog).args(parts), TEXT_TIMEOUT).map(|mut out| {
        out.truncate(out.iter().rposition(|b| *b != b'\n').map_or(0, |p| p + 1));
        out
    })
}

fn stdout_lossy(mut bytes: Vec<u8>) -> String {
    let text = String::from_utf8_lossy(&bytes).replace("\r\n", "\n");
    bytes.clear();
    text
}

/// Read plain text from the system clipboard (TS `readTextFromClipboard`,
/// branch order preserved). Empty string = no text / unreachable clipboard.
#[must_use]
pub fn read_text() -> String {
    if let Some(bytes) = std::env::var("OMP_CLIPBOARD_TEXT_GET").ok().as_deref().and_then(override_bytes) {
        return stdout_lossy(bytes);
    }
    if cfg!(target_os = "macos") {
        if let Some(bytes) = capture(&mut Command::new("pbpaste"), TEXT_TIMEOUT) {
            return stdout_lossy(bytes);
        }
        return String::new();
    }
    if std::env::var_os("TERMUX_VERSION").is_some() {
        if let Some(bytes) = capture(&mut Command::new("termux-clipboard-get"), TEXT_TIMEOUT) {
            return stdout_lossy(bytes);
        }
        return String::new();
    }
    if is_wsl() {
        if let Some(text) = read_text_via_powershell() {
            return text;
        }
        // Bridge failed — fall through to wl-paste/xclip (TS does the same).
    }
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
    let x11 = std::env::var_os("DISPLAY").is_some();
    if wayland {
        if let Some(bytes) = capture(
            Command::new("wl-paste").args(["--type", "text", "--no-newline"]),
            TEXT_TIMEOUT,
        ) {
            return stdout_lossy(bytes);
        }
        if x11 {
            return read_text_from_x11();
        }
    } else if x11 {
        return read_text_from_x11();
    }
    String::new()
}

fn read_text_from_x11() -> String {
    if let Some(bytes) = capture(
        Command::new("xclip").args(["-selection", "clipboard", "-o"]),
        TEXT_TIMEOUT,
    ) {
        return stdout_lossy(bytes);
    }
    if let Some(bytes) = capture(
        Command::new("xsel").args(["--clipboard", "--output"]),
        TEXT_TIMEOUT,
    ) {
        return stdout_lossy(bytes);
    }
    String::new()
}

fn read_text_via_powershell() -> Option<String> {
    let bytes = capture(
        Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command", POWERSHELL_TEXT_SCRIPT]),
        IMAGE_TIMEOUT,
    )?;
    Some(stdout_lossy(bytes))
}

/// Read an image from the system clipboard (TS `readImageFromClipboard`).
/// `None` when no supported image payload is available.
#[must_use]
pub fn read_image() -> Option<ClipboardImage> {
    if std::env::var_os("TERMUX_VERSION").is_some() {
        return None;
    }
    if let Some(bytes) = std::env::var("OMP_CLIPBOARD_IMAGE_GET").ok().as_deref().and_then(override_bytes) {
        return Some(ClipboardImage { data: bytes, mime: "image/png" });
    }
    if is_wsl() || cfg!(target_os = "windows") {
        if let Some(image) = read_image_via_powershell() {
            return Some(image);
        }
    }
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        if let Some(image) = read_image_via_wl_paste() {
            return Some(image);
        }
    }
    None
}

fn read_image_via_powershell() -> Option<ClipboardImage> {
    let bytes = capture(
        Command::new("powershell.exe").args([
            "-NoProfile",
            "-NonInteractive",
            "-Sta",
            "-Command",
            POWERSHELL_IMAGE_SCRIPT,
        ]),
        IMAGE_TIMEOUT,
    )?;
    let b64 = String::from_utf8_lossy(&bytes);
    let b64 = b64.trim();
    if b64.is_empty() {
        return None;
    }
    let data = base64_decode(b64)?;
    if data.is_empty() {
        return None;
    }
    Some(ClipboardImage { data, mime: "image/png" })
}

fn read_image_via_wl_paste() -> Option<ClipboardImage> {
    let listing = capture(&mut Command::new("wl-paste").arg("--list-types"), TEXT_TIMEOUT)?;
    let offered: Vec<String> = stdout_lossy(listing)
        .lines()
        .map(str::to_owned)
        .collect();
    for mime in IMAGE_MIME_TYPES {
        if !offered.iter().any(|m| m == mime) {
            continue;
        }
        if let Some(data) = capture(
            Command::new("wl-paste").args(["--type", mime]),
            TEXT_TIMEOUT,
        ) {
            if !data.is_empty() {
                return Some(ClipboardImage { data, mime });
            }
        }
    }
    None
}

/// Run `cmd`, capture stdout; `None` on spawn failure, non-zero exit or
/// timeout (the child is killed, mirroring TS `spawnCapture`).
fn capture(cmd: &mut Command, timeout: Duration) -> Option<Vec<u8>> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdout = child.stdout.take()?;
    // Drain the pipe on a side thread so a large payload cannot deadlock the
    // timeout poller on a full pipe buffer.
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.read_to_end(&mut buf);
        let _ = tx.send(buf);
    });
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let bytes = rx.recv().ok()?;
                let _ = reader.join();
                if status.success() {
                    return Some(bytes);
                }
                return None;
            }
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(_) => return None,
        }
    }
}

/// Minimal standard base64 decoder (the PowerShell bridge emits base64).
fn base64_decode(input: &str) -> Option<Vec<u8>> {
    fn value(b: u8) -> Option<u32> {
        match b {
            b'A'..=b'Z' => Some(u32::from(b - b'A')),
            b'a'..=b'z' => Some(u32::from(b - b'a') + 26),
            b'0'..=b'9' => Some(u32::from(b - b'0') + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        if chunk.len() < 4 {
            return None;
        }
        let pad = chunk.iter().filter(|b| **b == b'=').count();
        if pad > 2 {
            return None;
        }
        let mut acc = 0u32;
        for (i, b) in chunk.iter().enumerate() {
            let v = if *b == b'=' {
                if i < 2 {
                    return None;
                }
                0
            } else {
                value(*b)?
            };
            acc = (acc << 6) | v;
        }
        out.push(u8::try_from((acc >> 16) & 0xFF).ok()?);
        if pad < 2 {
            out.push(u8::try_from((acc >> 8) & 0xFF).ok()?);
        }
        if pad < 1 {
            out.push(u8::try_from(acc & 0xFF).ok()?);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_text_specs() {
        assert_eq!(override_bytes("printf hello"), Some(b"hello".to_vec()));
        // argv split is literal — no shell quoting layer; coreutils printf
        // ignores extra operands beyond the format.
        assert_eq!(override_bytes("printf a b"), Some(b"a".to_vec()));
        assert_eq!(override_bytes("definitely-not-a-real-cmd-xyz"), None);
        assert_eq!(override_bytes(""), None);
    }

    #[test]
    fn override_image_specs() {
        assert_eq!(
            override_bytes("printf PNGDATA").as_deref(),
            Some(b"PNGDATA".as_slice())
        );
        assert_eq!(override_bytes("definitely-not-a-real-cmd-xyz"), None);
    }

    #[test]
    fn base64_roundtrip() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8h").unwrap(), b"hello!");
        assert_eq!(base64_decode("aGVs\nbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("!!!!"), None);
    }

    #[test]
    fn outcome_variants_are_comparable() {
        assert_eq!(ClipboardOutcome::Empty, ClipboardOutcome::Empty);
        assert_eq!(
            ClipboardOutcome::Text("x".into()),
            ClipboardOutcome::Text("x".into())
        );
    }
}
