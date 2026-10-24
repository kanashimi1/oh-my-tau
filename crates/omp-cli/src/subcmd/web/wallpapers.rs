//! User wallpapers for the web UI: raster images uploaded from the browser,
//! kept in `~/.omp-rs/web/wallpapers/`. The bundled wallpapers ship with the
//! embedded assets (`assets/wallpapers/*.svg`).
//!
//! Uploads are sniffed by magic bytes and only raster formats are accepted:
//! an uploaded SVG could carry script and run same-origin when opened
//! directly.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _, Result};
use serde_json::{json, Value};

/// Largest accepted upload (4K PNGs are commonly 10–20 MB).
pub const MAX_UPLOAD: usize = 32 << 20;

fn dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".omp-rs/web/wallpapers"))
}

/// Image kind from the file's leading bytes: `(extension, mime)`.
fn sniff(bytes: &[u8]) -> Option<(&'static str, &'static str)> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some(("png", "image/png"))
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some(("jpg", "image/jpeg"))
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some(("gif", "image/gif"))
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some(("webp", "image/webp"))
    } else if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" && matches!(&bytes[8..12], b"avif" | b"avis") {
        Some(("avif", "image/avif"))
    } else {
        None
    }
}

fn mime_for(name: &str) -> Option<&'static str> {
    let ext = Path::new(name).extension()?.to_str()?;
    Some(match ext {
        "png" => "image/png",
        "jpg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "avif" => "image/avif",
        _ => return None,
    })
}

/// A stored file name is a plain `[A-Za-z0-9._-]` leaf with a raster extension.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && name.len() <= 128
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        && mime_for(name).is_some()
}

/// File stem from a client-supplied name, reduced to the safe alphabet.
fn safe_stem(name: &str) -> String {
    let stem = Path::new(name).file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let mut out: String = stem
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    out = out.trim_matches('-').chars().take(60).collect();
    if out.is_empty() {
        "wallpaper".to_owned()
    } else {
        out
    }
}

fn entry(name: &str) -> Value {
    json!({"name": name, "url": format!("/wallpapers/custom/{name}")})
}

/// `{"custom": [{name, url}]}`, newest first.
pub fn list() -> Value {
    let mut files: Vec<(std::time::SystemTime, String)> = dir()
        .and_then(|d| std::fs::read_dir(d).ok())
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let modified = e.metadata().ok()?.modified().ok()?;
            valid_name(&name).then_some((modified, name))
        })
        .collect();
    files.sort_by(|a, b| b.0.cmp(&a.0));
    json!({"custom": files.iter().map(|(_, n)| entry(n)).collect::<Vec<_>>()})
}

/// Store an upload under a fresh name derived from `name`.
///
/// # Errors
/// Not a supported raster image, or the write failed.
pub fn save(name: &str, bytes: &[u8]) -> Result<Value> {
    let Some((ext, _)) = sniff(bytes) else {
        bail!("unsupported image (PNG, JPEG, GIF, WebP or AVIF)");
    };
    let dir = dir().context("HOME is not set")?;
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let stem = safe_stem(name);
    let mut file = format!("{stem}.{ext}");
    let mut n = 2;
    while dir.join(&file).exists() {
        file = format!("{stem}-{n}.{ext}");
        n += 1;
    }
    std::fs::write(dir.join(&file), bytes).with_context(|| format!("write {file}"))?;
    Ok(entry(&file))
}

/// `(mime, bytes)` of a stored wallpaper.
pub fn read(name: &str) -> Option<(&'static str, Vec<u8>)> {
    if !valid_name(name) {
        return None;
    }
    let bytes = std::fs::read(dir()?.join(name)).ok()?;
    Some((mime_for(name)?, bytes))
}

/// Delete a stored wallpaper.
///
/// # Errors
/// Invalid name or the file could not be removed.
pub fn delete(name: &str) -> Result<()> {
    if !valid_name(name) {
        bail!("invalid wallpaper name");
    }
    let dir = dir().context("HOME is not set")?;
    std::fs::remove_file(dir.join(name)).with_context(|| format!("remove {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_raster_images_are_accepted() {
        assert_eq!(sniff(b"\x89PNG\r\n\x1a\nrest"), Some(("png", "image/png")));
        assert_eq!(sniff(&[0xFF, 0xD8, 0xFF, 0xE0]), Some(("jpg", "image/jpeg")));
        assert_eq!(sniff(b"RIFF\0\0\0\0WEBPVP8 "), Some(("webp", "image/webp")));
        assert_eq!(sniff(b"<svg xmlns=\"http://www.w3.org/2000/svg\"><script/>"), None);
        assert_eq!(sniff(b"<html>"), None);
    }

    #[test]
    fn stored_names_cannot_escape_the_directory() {
        assert!(valid_name("sunset-2.png"));
        assert!(!valid_name("../secret.png"));
        assert!(!valid_name("a/b.png"));
        assert!(!valid_name(".hidden.png"));
        assert!(!valid_name("evil.svg"));
        assert_eq!(safe_stem("Мой фон (1).PNG"), "1");
        assert_eq!(safe_stem("../../x y.png"), "x-y");
        assert_eq!(safe_stem("…"), "wallpaper");
    }
}
