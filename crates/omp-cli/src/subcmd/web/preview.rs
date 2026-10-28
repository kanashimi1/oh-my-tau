//! File preview for the web UI: `GET /preview/<abs path>` serves a file from
//! the agent host with its MIME type and HTTP Range support (video seeking),
//! so generated HTML pages load their relative assets, and media / PDFs play
//! inline. Office documents are converted to PDF through the installed
//! Microsoft Office (COM automation over WSL interop) and cached.
//!
//! HTML and SVG are served with `Content-Security-Policy: sandbox`, so a
//! previewed page runs in an opaque origin and cannot call the hub API.

use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{bail, Context as _, Result};
use hyper::header::{HeaderValue, CONTENT_TYPE};
use hyper::{Response, StatusCode};
use serde_json::{json, Value};

use super::Body;

/// Largest body of one response; bigger files are streamed as ranges.
const MAX_CHUNK: u64 = 16 << 20;
/// Largest file served without a Range request (text, html, images).
const MAX_WHOLE: u64 = 256 << 20;
const POWERSHELL: &str = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe";

/// `C:\x\y` / `C:/x/y` → `/mnt/c/x/y`, `~/x` → `$HOME/x`; others unchanged.
#[must_use]
pub fn host_path(raw: &str) -> PathBuf {
    let raw = raw.trim();
    let b = raw.as_bytes();
    if b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'\\' || b[2] == b'/')
    {
        let rest = raw[3..].replace('\\', "/");
        return PathBuf::from(format!(
            "/mnt/{}/{rest}",
            (b[0] as char).to_ascii_lowercase()
        ));
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(raw)
}

/// MIME type for `path` by extension.
#[must_use]
pub fn mime(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "svg" => "image/svg+xml",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "json" | "map" | "bbmodel" => "application/json; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "avif" => "image/avif",
        "bmp" => "image/bmp",
        "ico" => "image/x-icon",
        "mp4" | "m4v" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "ogv" => "video/ogg",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" | "oga" | "opus" => "audio/ogg",
        "m4a" => "audio/mp4",
        "flac" => "audio/flac",
        "pdf" => "application/pdf",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "otf" => "font/otf",
        "wasm" => "application/wasm",
        "glb" => "model/gltf-binary",
        "gltf" => "model/gltf+json",
        "md" | "markdown" | "txt" | "log" | "csv" | "tsv" | "py" | "rs" | "ts" | "tsx" | "jsx"
        | "toml" | "yaml" | "yml" | "sh" | "xml" | "ini" | "cfg" | "c" | "h" | "cpp" | "go"
        | "java" | "lua" | "sql" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// `bytes=a-b` / `bytes=a-` / `bytes=-n` → inclusive (start, end) within `len`.
fn parse_range(header: &str, len: u64) -> Option<(u64, u64)> {
    let spec = header
        .trim()
        .strip_prefix("bytes=")?
        .split(',')
        .next()?
        .trim();
    let (a, b) = spec.split_once('-')?;
    let (start, end) = if a.is_empty() {
        let n: u64 = b.parse().ok()?;
        (len.saturating_sub(n), len.checked_sub(1)?)
    } else {
        let start: u64 = a.parse().ok()?;
        let end = if b.is_empty() {
            len.checked_sub(1)?
        } else {
            b.parse::<u64>().ok()?.min(len.checked_sub(1)?)
        };
        (start, end)
    };
    (start <= end && end < len).then_some((start, end))
}

fn read_slice(path: &Path, start: u64, len: u64) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    f.seek(SeekFrom::Start(start))?;
    let mut buf = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
    f.take(len).read_to_end(&mut buf)?;
    Ok(buf)
}

fn plain(status: StatusCode, msg: &str) -> Response<Body> {
    let mut res = Response::new(Body::new(msg.to_owned().into()));
    *res.status_mut() = status;
    res.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    res
}

/// Script appended to HTML shown in the preview panel (`?embed=1`): reports
/// the scroll position to the panel and restores it after a live reload.
const EMBED_SCRIPT: &str = "<script>(()=>{const p=parent;if(p===window)return;\
addEventListener('message',e=>{const s=e.data&&e.data.tauScrollTo;if(s)scrollTo(s[0],s[1])});\
let t=0;addEventListener('scroll',()=>{clearTimeout(t);t=setTimeout(()=>p.postMessage({tauScroll:[scrollX,scrollY]},'*'),80)},{passive:true});\
addEventListener('load',()=>p.postMessage({tauLoaded:1},'*'))})()</script>";

/// Serve `path` (a host path), honouring a `Range` header. `embed` marks a
/// request from the preview panel (HTML gets [`EMBED_SCRIPT`]).
pub fn serve(path: &Path, range: Option<&str>, embed: bool) -> Response<Body> {
    let meta = match std::fs::metadata(path) {
        Ok(m) if m.is_file() => m,
        Ok(_) => return plain(StatusCode::NOT_FOUND, "not a file"),
        Err(e) => return plain(StatusCode::NOT_FOUND, &format!("{}: {e}", path.display())),
    };
    let len = meta.len();
    let ctype = mime(path);
    let (status, start, end) = match range.map(|r| parse_range(r, len)) {
        Some(Some((s, e))) => (StatusCode::PARTIAL_CONTENT, s, e.min(s + MAX_CHUNK - 1)),
        Some(None) if len > 0 => {
            let mut res = plain(StatusCode::RANGE_NOT_SATISFIABLE, "bad range");
            if let Ok(v) = HeaderValue::from_str(&format!("bytes */{len}")) {
                res.headers_mut().insert("content-range", v);
            }
            return res;
        }
        _ if len > MAX_WHOLE => (StatusCode::PARTIAL_CONTENT, 0, MAX_CHUNK - 1),
        _ => (StatusCode::OK, 0, len.saturating_sub(1)),
    };
    let mut body = if len == 0 {
        Vec::new()
    } else {
        match read_slice(path, start, end - start + 1) {
            Ok(b) => b,
            Err(e) => return plain(StatusCode::INTERNAL_SERVER_ERROR, &format!("read: {e}")),
        }
    };
    if embed && status == StatusCode::OK && ctype.starts_with("text/html") {
        body.extend_from_slice(EMBED_SCRIPT.as_bytes());
    }
    let mut res = Response::new(Body::new(body.into()));
    *res.status_mut() = status;
    let h = res.headers_mut();
    h.insert(CONTENT_TYPE, HeaderValue::from_static(ctype));
    h.insert("accept-ranges", HeaderValue::from_static("bytes"));
    h.insert("cache-control", HeaderValue::from_static("no-cache"));
    h.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    if status == StatusCode::PARTIAL_CONTENT {
        if let Ok(v) = HeaderValue::from_str(&format!("bytes {start}-{end}/{len}")) {
            h.insert("content-range", v);
        }
    }
    if ctype.starts_with("text/html") || ctype == "image/svg+xml" {
        // Opaque origin even when opened directly: no access to the hub API.
        h.insert(
            "content-security-policy",
            HeaderValue::from_static(
                "sandbox allow-scripts allow-forms allow-popups allow-modals allow-downloads",
            ),
        );
    }
    res
}

/// Extensions converted to PDF by [`office_pdf`].
#[must_use]
pub fn is_office(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some(
            "ppt"
                | "pptx"
                | "pps"
                | "ppsx"
                | "odp"
                | "doc"
                | "docx"
                | "odt"
                | "rtf"
                | "xls"
                | "xlsx"
                | "ods"
        )
    )
}

fn win_path(path: &Path) -> Result<String> {
    let out = std::process::Command::new("wslpath")
        .arg("-w")
        .arg(path)
        .output()
        .context("wslpath")?;
    if !out.status.success() {
        bail!("wslpath: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// PowerShell that exports `$src` to PDF `$dst` with the matching Office app.
const OFFICE_PS: &str = r"
$ErrorActionPreference = 'Stop'
$ext = [IO.Path]::GetExtension($src).ToLower()
if ($ext -in '.ppt','.pptx','.pps','.ppsx','.odp') {
  $app = New-Object -ComObject PowerPoint.Application
  $doc = $app.Presentations.Open($src, $true, $false, $false)
  try { $doc.SaveAs($dst, 32) } finally { $doc.Close(); if ($app.Presentations.Count -eq 0) { $app.Quit() } }
} elseif ($ext -in '.xls','.xlsx','.ods') {
  $app = New-Object -ComObject Excel.Application
  $app.DisplayAlerts = $false
  $doc = $app.Workbooks.Open($src, 0, $true)
  try { $doc.ExportAsFixedFormat(0, $dst) } finally { $doc.Close($false); if ($app.Workbooks.Count -eq 0) { $app.Quit() } }
} else {
  $app = New-Object -ComObject Word.Application
  $app.DisplayAlerts = 0
  $doc = $app.Documents.Open($src, $false, $true)
  try { $doc.ExportAsFixedFormat($dst, 17) } finally { $doc.Close(0); if ($app.Documents.Count -eq 0) { $app.Quit() } }
}
";

/// Convert an Office document to PDF (cached by path + mtime + size);
/// returns `{pdf}` with the host path of the PDF.
///
/// # Errors
/// Not on a WSL host with Office, or the export failed.
pub async fn office_pdf(path: &Path) -> Result<Value> {
    if !Path::new(POWERSHELL).exists() {
        bail!("Office preview needs Windows (WSL interop)");
    }
    let meta = std::fs::metadata(path).with_context(|| path.display().to_string())?;
    let stamp = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    let key = {
        use std::hash::{Hash as _, Hasher as _};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (path, stamp, meta.len()).hash(&mut h);
        h.finish()
    };
    // Windows-side cache: Office writes it directly (no \\wsl$ round trip).
    let cache = windows_temp()
        .context("cannot resolve the Windows %TEMP% dir")?
        .join("omt-preview");
    std::fs::create_dir_all(&cache).context("create preview cache")?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document");
    let pdf = cache.join(format!("{stem}-{key:016x}.pdf"));
    if !pdf.exists() {
        // Office opens files from a Windows path most reliably: copy WSL-side
        // documents next to the cache first.
        let src = if path.starts_with("/mnt/") {
            path.to_path_buf()
        } else {
            let copy = cache.join(format!(
                "{key:016x}-{}",
                path.file_name().and_then(|n| n.to_str()).unwrap_or("doc")
            ));
            std::fs::copy(path, &copy).context("stage document")?;
            copy
        };
        let script = format!(
            "$src = '{}'; $dst = '{}';{OFFICE_PS}",
            win_path(&src)?.replace('\'', "''"),
            win_path(&pdf)?.replace('\'', "''")
        );
        let out = tokio::time::timeout(
            std::time::Duration::from_mins(3),
            tokio::process::Command::new(POWERSHELL)
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    &script,
                ])
                .current_dir("/mnt/c/")
                .stdin(Stdio::null())
                .output(),
        )
        .await
        .context("Office export timed out")?
        .context("start PowerShell")?;
        if src != path {
            let _ = std::fs::remove_file(&src);
        }
        if !out.status.success() || !pdf.exists() {
            let err = String::from_utf8_lossy(&out.stderr);
            bail!(
                "Office export failed: {}",
                err.lines()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("no output")
            );
        }
    }
    Ok(json!({"pdf": pdf.display().to_string()}))
}

/// The Windows user's `%TEMP%` as a WSL path (resolved once).
fn windows_temp() -> Option<PathBuf> {
    static TEMP: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    TEMP.get_or_init(|| {
        let out = std::process::Command::new("/mnt/c/Windows/System32/cmd.exe")
            .args(["/c", "echo %TEMP%"])
            .current_dir("/mnt/c/")
            .output()
            .ok()?;
        let win = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        if win.is_empty() || win.contains('%') {
            return None;
        }
        let out = std::process::Command::new("wslpath")
            .arg("-u")
            .arg(&win)
            .output()
            .ok()?;
        let path = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        path.is_dir().then_some(path)
    })
    .clone()
}

/// Open `path` with its Windows default app, or (`reveal`) select it in
/// Explorer.
///
/// # Errors
/// Not a WSL host, or the path cannot be mapped.
pub fn open_in_windows(path: &Path, reveal: bool) -> Result<()> {
    if !Path::new(POWERSHELL).exists() {
        bail!("opening files needs Windows (WSL interop)");
    }
    if !path.exists() {
        bail!("{} does not exist", path.display());
    }
    let win = win_path(path)?;
    let mut cmd = std::process::Command::new("/mnt/c/Windows/explorer.exe");
    if reveal {
        cmd.arg(format!("/select,{win}"));
    } else {
        cmd.arg(&win);
    }
    let mut child = cmd
        .current_dir("/mnt/c/")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("start explorer.exe")?;
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_and_home_paths_map_to_host_paths() {
        assert_eq!(
            host_path(r"C:\Users\k\a b.html"),
            PathBuf::from("/mnt/c/Users/k/a b.html")
        );
        assert_eq!(host_path("D:/x/y.mp4"), PathBuf::from("/mnt/d/x/y.mp4"));
        assert_eq!(host_path("/tmp/x.pdf"), PathBuf::from("/tmp/x.pdf"));
    }

    #[test]
    fn ranges_clamp_to_the_file() {
        assert_eq!(parse_range("bytes=0-", 100), Some((0, 99)));
        assert_eq!(parse_range("bytes=10-19", 100), Some((10, 19)));
        assert_eq!(parse_range("bytes=90-500", 100), Some((90, 99)));
        assert_eq!(parse_range("bytes=-10", 100), Some((90, 99)));
        assert_eq!(parse_range("bytes=100-", 100), None);
        assert_eq!(parse_range("items=0-1", 100), None);
    }
}
