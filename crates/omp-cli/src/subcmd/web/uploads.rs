//! Composer file attachments: `POST /api/uploads?name=` stores the raw body
//! under `~/.omp-rs/uploads/<stamp>/<name>` and returns the path, which the
//! composer adds to the prompt so the agent reads the file with its tools.

use std::path::PathBuf;

use anyhow::{Context as _, Result};
use serde_json::{json, Value};

/// Largest accepted attachment.
pub const MAX_UPLOAD: usize = 512 << 20;

/// `name` reduced to a safe single path component.
fn safe_name(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or_default();
    let clean: String = base
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, ':' | '*' | '?' | '"' | '<' | '>' | '|') {
                '_'
            } else {
                c
            }
        })
        .collect();
    let clean = clean.trim().trim_start_matches('.').to_owned();
    if clean.is_empty() {
        "file".to_owned()
    } else {
        clean.chars().take(120).collect()
    }
}

/// Save one attachment; returns `{path, name, size}`.
///
/// # Errors
/// The uploads dir cannot be created or written.
pub fn save(name: &str, bytes: &[u8]) -> Result<Value> {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    let stamp = chrono_stamp();
    let dir = home.join(".omp-rs/uploads").join(stamp);
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let name = safe_name(name);
    let path = dir.join(&name);
    std::fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(json!({"path": path.display().to_string(), "name": name, "size": bytes.len()}))
}

/// Unique per-upload dir name: millisecond clock plus a counter.
fn chrono_stamp() -> String {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    format!(
        "{}-{}",
        pa_agent::now_ms(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::safe_name;

    #[test]
    fn names_cannot_escape_the_upload_dir() {
        assert_eq!(safe_name("../../etc/passwd"), "passwd");
        assert_eq!(safe_name("C:\\Users\\x\\report.pdf"), "report.pdf");
        assert_eq!(safe_name(".."), "file");
        assert_eq!(safe_name("a:b?.txt"), "a_b_.txt");
        assert_eq!(safe_name("заметки.md"), "заметки.md");
    }
}
