//! `local://` provider (TS `internal-urls/local-protocol.ts`).

use std::path::{Path, PathBuf};

use crate::parse::InternalUrl;
use crate::spec::{
    InternalResource, SchemeBacking, SchemeSelectors, SchemeSpec, WritePolicy, WriteScope, WriteVia,
};
use crate::{UrlCtx, UrlProvider};

use crate::schemes::core::format_byte_size;

const LOCAL_TEXT_SNIFF_BYTES: u64 = 8 * 1024;
const LOCAL_TEXT_RESOURCE_MAX_BYTES: u64 = 1024 * 1024;
#[allow(dead_code)]
pub(crate) const LOCAL_WRITE_NOTE: &str =
    "Use write path local://<file> to persist large intermediate artifacts across turns.";

const BINARY_FILE_EXTENSIONS: &[&str] = &[
    "7z", "avi", "bmp", "bz2", "db", "doc", "docx", "gif", "gz", "ico", "jpeg", "jpg", "m4v",
    "mkv", "mov", "mp4", "pdf", "png", "ppt", "pptx", "rar", "sqlite", "tgz", "webm", "webp",
    "wmv", "xls", "xlsx", "xz", "zip",
];

/// Session-scoped scratch provider. State is per-call through [`UrlCtx`].
pub struct LocalProvider;

fn safe_session_id(ctx: &UrlCtx) -> String {
    let raw = ctx.session_id().unwrap_or_else(|| "session".to_string());
    let safe: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if safe.is_empty() {
        "session".to_string()
    } else {
        safe
    }
}

/// TS `resolveLocalRoot`: `<artifactsDir>/local`, else `$TMPDIR/omp-local/<safeSessionId>`.
pub fn resolve_local_root(ctx: &UrlCtx) -> PathBuf {
    if let Some(dir) = ctx.artifacts_dir() {
        return dir.join("local");
    }
    std::env::temp_dir()
        .join("omp-local")
        .join(safe_session_id(ctx))
}

/// TS `extractRelativePath`: authority counts as the first path segment.
fn extract_relative_path(url: &InternalUrl) -> anyhow::Result<String> {
    let combined = if !url.host.is_empty() {
        if !url.pathname.is_empty() && url.pathname != "/" {
            format!("{}{}", url.host, url.pathname)
        } else {
            url.host.clone()
        }
    } else if !url.pathname.is_empty() && url.pathname != "/" {
        url.pathname[1..].to_string()
    } else {
        String::new()
    };
    if combined.is_empty() {
        return Ok(String::new());
    }
    let decoded = crate::schemes::decode_component(&combined.replace('\\', "/"));
    // TS `validateRelativePath` with the scheme-prefixed messages.
    if std::path::Path::new(&decoded).is_absolute() {
        anyhow::bail!("Absolute paths are not allowed in local:// URLs");
    }
    let segments: Vec<&str> = decoded.split(['/', '\\']).collect();
    if segments.contains(&"..") {
        anyhow::bail!("Path traversal (..) is not allowed in local:// URLs");
    }
    Ok(decoded)
}

fn ensure_within_root(target: &Path, root: &Path, url: &str) -> anyhow::Result<()> {
    if target != root && !target.starts_with(root) {
        anyhow::bail!("local:// URL escapes local root: {url}");
    }
    Ok(())
}

enum LocalTarget {
    Listing(PathBuf),
    Directory(PathBuf),
    File { path: PathBuf, size: u64 },
}

/// TS `resolveLocalTarget`: realpath + containment on root, parent, target.
fn resolve_local_target(url: &InternalUrl, ctx: &UrlCtx) -> anyhow::Result<LocalTarget> {
    let root = resolve_local_root(ctx);
    std::fs::create_dir_all(&root)?;
    let resolved_root = std::fs::canonicalize(&root)
        .map_err(|_| anyhow::anyhow!("Unable to initialize local:// root"))?;
    let relative = extract_relative_path(url)?;
    let target = if relative.is_empty() {
        resolved_root.clone()
    } else {
        resolved_root.join(&relative)
    };
    ensure_within_root(&target, &resolved_root, &url.raw_href)?;
    if target == resolved_root {
        return Ok(LocalTarget::Listing(resolved_root));
    }
    // Missing or escaping targets fail closed.
    let real_target = std::fs::canonicalize(&target)
        .map_err(|_| anyhow::anyhow!("Local file not found: {}", url.raw_href))?;
    ensure_within_root(&real_target, &resolved_root, &url.raw_href)?;
    let meta = std::fs::metadata(&real_target)?;
    if meta.is_dir() {
        return Ok(LocalTarget::Directory(real_target));
    }
    if !meta.is_file() {
        anyhow::bail!(
            "local:// URL must resolve to a file or directory: {}",
            url.raw_href
        );
    }
    Ok(LocalTarget::File {
        path: real_target,
        size: meta.len(),
    })
}

fn non_text_resource(url: &InternalUrl, path: &Path, size: u64, reason: &str) -> InternalResource {
    let content = format!(
        "[Cannot read binary local:// file '{}' ({}): {reason}. This resource is not text. Use a metadata/key-frame/video-specific workflow instead.]",
        url.raw_href,
        format_byte_size(size)
    );
    let mut r = InternalResource::text(content);
    r.path = Some(path.to_string_lossy().into_owned());
    r
}

fn build_file_resource(
    url: &InternalUrl,
    path: &Path,
    size: u64,
) -> anyhow::Result<InternalResource> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if BINARY_FILE_EXTENSIONS.contains(&ext.as_str()) {
        return Ok(non_text_resource(
            url,
            path,
            size,
            "extension is a known binary/container type",
        ));
    }
    let mut sniff = vec![0u8; size.min(LOCAL_TEXT_SNIFF_BYTES) as usize];
    {
        use std::io::Read as _;
        std::fs::File::open(path)?.read_exact(&mut sniff)?;
    }
    if sniff.contains(&0) || std::str::from_utf8(&sniff).is_err() {
        return Ok(non_text_resource(
            url,
            path,
            size,
            "content is not valid UTF-8 text",
        ));
    }
    if size > LOCAL_TEXT_RESOURCE_MAX_BYTES {
        let content = format!(
            "[Cannot materialize local:// file '{}' as an internal text resource ({} exceeds {}). Use the read tool's filesystem path handling or a line selector so content is streamed with file-size safeguards.]",
            url.raw_href,
            format_byte_size(size),
            format_byte_size(LOCAL_TEXT_RESOURCE_MAX_BYTES)
        );
        let mut r = InternalResource::text(content);
        r.path = Some(path.to_string_lossy().into_owned());
        return Ok(r);
    }
    let content = std::fs::read_to_string(path)?;
    let mut r = InternalResource::text(content);
    r.path = Some(path.to_string_lossy().into_owned());
    Ok(r)
}

/// TS `listFilesRecursively`, sorted by name.
fn list_files_recursively(root: &Path) -> anyhow::Result<Vec<String>> {
    let mut files = Vec::new();
    let mut pending = vec![String::new()];
    while let Some(rel) = pending.pop() {
        let abs = root.join(&rel);
        for entry in std::fs::read_dir(&abs)?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let child = if rel.is_empty() {
                name.clone()
            } else {
                format!("{rel}/{name}")
            };
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                pending.push(child);
            } else if file_type.is_file() {
                files.push(child);
            }
        }
    }
    files.sort();
    Ok(files)
}

/// TS `buildListing`.
fn build_listing(url: &InternalUrl, root: &Path) -> anyhow::Result<InternalResource> {
    let files = list_files_recursively(root)?;
    let listing = if files.is_empty() {
        "(empty)".to_string()
    } else {
        files
            .iter()
            .map(|f| format!("- [{f}](local://{f})"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let content = format!(
        "# Local\n\nSession-scoped scratch space for large intermediate data, subagent handoffs, and reusable planning artifacts.\n\nRoot: {}\n\n{} file{} available:\n\n{listing}\n",
        root.display(),
        files.len(),
        if files.len() == 1 { "" } else { "s" }
    );
    let mut r = InternalResource::directory(content);
    r.path = Some(root.to_string_lossy().into_owned());
    let _ = url;
    Ok(r)
}

impl UrlProvider for LocalProvider {
    fn scheme(&self) -> &'static str {
        "local"
    }

    fn spec(&self) -> &SchemeSpec {
        const SPEC: SchemeSpec = SchemeSpec {
            backing: SchemeBacking::File,
            selectors: SchemeSelectors::Lines,
            immutable: false,
            single_slash_alias: true,
            linkable: true,
            port_authority: false,
            write: Some(WritePolicy {
                via: WriteVia::File,
                scope: WriteScope::Sandbox,
            }),
        };
        &SPEC
    }

    fn impl_locate(&self) -> bool {
        true
    }

    fn read(&self, url: &InternalUrl, ctx: &UrlCtx) -> anyhow::Result<InternalResource> {
        match resolve_local_target(url, ctx)? {
            LocalTarget::Listing(root) => build_listing(url, &root),
            LocalTarget::Directory(path) => {
                let entries: Vec<(String, bool)> = std::fs::read_dir(&path)?
                    .flatten()
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().into_owned();
                        e.file_type().ok().map(|t| (name, t.is_dir()))
                    })
                    .collect();
                let mut r =
                    InternalResource::directory(crate::schemes::format_directory_listing(entries));
                r.path = Some(path.to_string_lossy().into_owned());
                r.immutable = Some(true);
                Ok(r)
            }
            LocalTarget::File { path, size } => build_file_resource(url, &path, size),
        }
    }

    fn locate(&self, url: &InternalUrl, ctx: &UrlCtx) -> Option<PathBuf> {
        let root = resolve_local_root(ctx);
        let relative = extract_relative_path(url).ok()?;
        let target = if relative.is_empty() {
            root.clone()
        } else {
            root.join(relative)
        };
        ensure_within_root(&target, &root, &url.raw_href).ok()?;
        let real_root = std::fs::canonicalize(&root).ok()?;
        let real = std::fs::canonicalize(&target).ok()?;
        ensure_within_root(&real, &real_root, &url.raw_href).ok()?;
        Some(real)
    }

    fn locate_create(&self, url: &InternalUrl, ctx: &UrlCtx) -> Option<PathBuf> {
        let root = resolve_local_root(ctx);
        let relative = extract_relative_path(url).ok()?;
        let target = if relative.is_empty() {
            return None;
        } else {
            root.join(relative)
        };
        // Lexical containment; a realpath check would reject not-yet-created paths.
        ensure_within_root(&target, &root, &url.raw_href).ok()?;
        Some(target)
    }

    fn locate_sync(&self, url: &InternalUrl, ctx: &UrlCtx) -> Option<PathBuf> {
        let root = resolve_local_root(ctx);
        let relative = extract_relative_path(url).ok()?;
        let target = if relative.is_empty() {
            root
        } else {
            root.join(relative)
        };
        ensure_within_root(&target, &resolve_local_root(ctx), &url.raw_href).ok()?;
        Some(target)
    }

    fn complete(&self, _query: &str, ctx: &UrlCtx) -> Vec<crate::UrlCompletion> {
        let root = resolve_local_root(ctx);
        let Ok(files) = list_files_recursively(&root) else {
            return Vec::new();
        };
        files
            .into_iter()
            .map(|label| crate::UrlCompletion {
                label,
                detail: None,
            })
            .collect()
    }
}
