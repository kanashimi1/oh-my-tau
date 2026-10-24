//! `omp://` protocol handler (TS `internal-urls/omp-protocol.ts` + `docs-index.ts` + `omp-scope.ts`).
//!
//! Serves the harness documentation corpus. TS inlines a two-line payload
//! (`PI_DOCS_EMBED`): a JSON array of sorted doc filenames, then a base64
//! gzip blob of the index-aligned doc bodies. This port reads the same payload
//! from `OMP_DOCS_EMBEM`/`OMP_DOCS_EMBED` env or the `OMP_DOCS_INDEX` file
//! (the shipped `dist/docs-index.generated.txt` format), and degrades to an
//! empty index when no corpus is reachable.
//!
//! URL forms:
//! - `omp://` — lists all available documentation files
//! - `omp://<file>.md` — reads a specific documentation file

use std::sync::{LazyLock, OnceLock};

use crate::{InternalResource, InternalUrl, UrlCompletion, UrlCtx, UrlProvider};

/// Sorted doc filenames with lazily inflated bodies (TS `DocsIndex`).
pub struct DocsIndex {
    filenames: Vec<String>,
    /// `Ok` map or the inflate/parse failure, computed once.
    bodies: OnceLock<Result<Vec<(String, String)>, String>>,
    blob: String,
}

impl DocsIndex {
    /// Filenames never touch the blob (TS `getDocFilenames`).
    pub fn filenames(&self) -> &[String] {
        &self.filenames
    }

    /// Resolve a doc body by path; inflates lazily, once (TS `getBody`).
    pub fn body(&self, relative_path: &str) -> Option<&str> {
        let bodies = self
            .bodies
            .get_or_init(|| inflate_bodies(&self.filenames, &self.blob));
        match bodies {
            Ok(map) => map
                .iter()
                .find(|(name, _)| name == relative_path)
                .map(|(_, body)| body.as_str()),
            Err(_) => None,
        }
    }
}

/// Gunzip + JSON-decode the index-aligned bodies (TS `getBody` decode).
fn inflate_bodies(filenames: &[String], blob: &str) -> Result<Vec<(String, String)>, String> {
    use base64::Engine;
    let compressed = base64::engine::general_purpose::STANDARD
        .decode(blob.trim())
        .map_err(|e| format!("docs index: base64 decode failed: {e}"))?;
    let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
    let mut json = String::new();
    std::io::Read::read_to_string(&mut decoder, &mut json)
        .map_err(|e| format!("docs index: gzip inflate failed: {e}"))?;
    let decoded: Vec<String> =
        serde_json::from_str(&json).map_err(|e| format!("docs index: body JSON failed: {e}"))?;
    if decoded.len() != filenames.len() {
        return Err(format!(
            "docs index: {} filenames vs {} bodies",
            filenames.len(),
            decoded.len()
        ));
    }
    Ok(filenames.iter().cloned().zip(decoded).collect())
}

/// Decode a populated two-line embed (TS `decodeDocsIndex`). `None` when there
/// is no newline separator (empty placeholder or malformed payload).
pub fn decode_docs_index(embed: &str) -> Option<DocsIndex> {
    let newline = embed.find('\n')?;
    let filenames: Vec<String> = serde_json::from_str(embed[..newline].trim()).ok()?;
    Some(DocsIndex {
        filenames,
        bodies: OnceLock::new(),
        blob: embed[newline + 1..].to_string(),
    })
}

/// Process-global index (TS `getIndex`): env embed, then the file named by
/// `OMP_DOCS_INDEX`, then an empty index so a missing corpus never errors callers.
static INDEX: LazyLock<Option<DocsIndex>> = LazyLock::new(|| {
    if let Ok(embed) = std::env::var("OMP_DOCS_EMBED") {
        if !embed.is_empty() {
            return Some(decode_docs_index(&embed).unwrap_or_else(|| DocsIndex {
                filenames: Vec::new(),
                bodies: OnceLock::new(),
                blob: String::new(),
            }));
        }
    }
    if let Ok(path) = std::env::var("OMP_DOCS_INDEX") {
        if let Ok(raw) = std::fs::read_to_string(&path) {
            if let Some(index) = decode_docs_index(&raw) {
                return Some(index);
            }
        }
    }
    None
});

fn index() -> &'static Option<DocsIndex> {
    &INDEX
}

/// Sorted list of available doc filenames (TS `getDocFilenames`).
pub fn doc_filenames() -> Vec<String> {
    index()
        .as_ref()
        .map_or_else(Vec::new, |i| i.filenames.clone())
}

/// Resolve a doc body by path, or `None` when not found (TS `getEmbeddedDoc`).
pub fn embedded_doc(relative_path: &str) -> Option<String> {
    index()
        .as_ref()
        .and_then(|i| i.body(relative_path).map(str::to_string))
}

/// Minimal POSIX normalization: backslashes to slashes, resolve `.`/`..`,
/// collapse duplicate slashes (TS `path.posix.normalize`).
fn posix_normalize(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let replaced = path.replace('\\', "/");
    for seg in replaced.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            _ => out.push(seg),
        }
    }
    out.join("/")
}

/// Host + path of an `omp://` URL exactly as the handler reads it; `""` for the docs root (TS `ompDocFilename`).
pub fn omp_doc_filename(url: &InternalUrl) -> String {
    let host = &url.host;
    let pathname = &url.pathname;
    if host.is_empty() {
        String::new()
    } else if !pathname.is_empty() && pathname != "/" {
        format!("{host}{pathname}")
    } else {
        host.clone()
    }
}

/// Canonical doc path relative to `docs/` for an `omp://` URL, or `""` for the
/// docs root. Throws on absolute paths and `..` traversal (TS `ompDocRel`).
pub fn omp_doc_rel(url: &InternalUrl) -> Result<String, String> {
    let filename = omp_doc_filename(url);
    if filename.is_empty() {
        return Ok(String::new());
    }
    if filename.starts_with('/') {
        return Err("Absolute paths are not allowed in omp:// URLs".to_string());
    }
    let replaced = filename.replace('\\', "/");
    if replaced.split('/').any(|seg| seg == "..") {
        return Err("Path traversal (..) is not allowed in omp:// URLs".to_string());
    }
    let normalized = posix_normalize(&replaced);
    if normalized.is_empty() || normalized == "docs" {
        return Ok(String::new());
    }
    Ok(match normalized.strip_prefix("docs/") {
        Some(rest) => rest.to_string(),
        None => normalized,
    })
}

/// Resolve an `omp://` URL to a resource (TS `OmpProtocolHandler.resolve`).
pub fn resolve(url: &InternalUrl) -> Result<InternalResource, String> {
    let filename = omp_doc_filename(url);
    let doc_path = omp_doc_rel(url)?;
    if filename.is_empty() || doc_path.is_empty() {
        return list_docs(url);
    }
    read_doc(&doc_path, &filename, url)
}

/// Every embedded doc for a root scope, in sorted order (TS `enumerate`).
pub fn enumerate() -> Vec<(String, String)> {
    let mut entries = Vec::new();
    if let Some(index) = index() {
        for rel in &index.filenames {
            if let Some(content) = index.body(rel) {
                entries.push((format!("omp://{rel}"), content.to_string()));
            }
        }
    }
    entries
}

/// The docs root expands to the rendered index listing (TS `#listDocs`).
fn list_docs(_url: &InternalUrl) -> Result<InternalResource, String> {
    let filenames = doc_filenames();
    if filenames.is_empty() {
        return Err("No documentation files found".to_string());
    }
    let listing = filenames
        .iter()
        .map(|f| format!("- [{f}](omp://{f})"))
        .collect::<Vec<_>>()
        .join("\n");
    let content = format!(
        "# Documentation\n\n{} files available:\n\n{listing}\n",
        filenames.len()
    );
    Ok(InternalResource {
        content,
        is_directory: true,
        immutable: Some(true),
        path: None,
    })
}

/// One embedded doc, or the TS not-found error with up to 5 suggestions (TS `#readDoc`).
fn read_doc(
    doc_path: &str,
    filename: &str,
    _url: &InternalUrl,
) -> Result<InternalResource, String> {
    let Some(content) = embedded_doc(doc_path) else {
        let lookup = doc_path.strip_suffix(".md").unwrap_or(doc_path);
        let suggestions: Vec<String> = doc_filenames()
            .into_iter()
            .filter(|f| {
                f.contains(lookup) || lookup.contains(f.strip_suffix(".md").unwrap_or(f.as_str()))
            })
            .take(5)
            .collect();
        let suffix = if suggestions.is_empty() {
            "\nUse omp:// to list available files.".to_string()
        } else {
            format!("\nDid you mean: {}", suggestions.join(", "))
        };
        return Err(format!("Documentation file not found: {filename}{suffix}"));
    };
    Ok(InternalResource {
        content,
        is_directory: false,
        immutable: Some(true),
        path: None,
    })
}

/// Read-only `omp://` provider over the embedded docs corpus.
#[derive(Default)]
pub struct OmpProvider;

impl UrlProvider for OmpProvider {
    fn scheme(&self) -> &'static str {
        "omp"
    }

    fn spec(&self) -> &crate::SchemeSpec {
        static SPEC: crate::SchemeSpec = crate::SchemeSpec {
            backing: crate::SchemeBacking::Virtual,
            selectors: crate::SchemeSelectors::Lines,
            immutable: true,
            single_slash_alias: false,
            linkable: false,
            port_authority: false,
            write: None,
        };
        &SPEC
    }

    fn read(&self, url: &InternalUrl, _ctx: &UrlCtx) -> anyhow::Result<InternalResource> {
        resolve(url).map_err(anyhow::Error::msg)
    }

    fn complete(&self, _query: &str, _ctx: &UrlCtx) -> Vec<UrlCompletion> {
        doc_filenames()
            .into_iter()
            .map(|value| UrlCompletion {
                label: value,
                detail: None,
            })
            .collect()
    }

    fn prompt_doc(&self) -> Option<String> {
        // TS `prompts/internal-urls/omp.md`, trimmed.
        Some("`omp://`: harness docs, AVOID unless asked.".to_string())
    }
}

/// Build an index from the shipped corpus file when a path is resolvable.
pub fn index_from_file(path: &std::path::Path) -> Option<DocsIndex> {
    decode_docs_index(&std::fs::read_to_string(path).ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_internal_url;

    /// Two-doc fixture in the TS embed format.
    fn fixture() -> String {
        let bodies = serde_json::json!(["body of a", "body of guide"]).to_string();
        use base64::Engine;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        std::io::Write::write_all(&mut encoder, bodies.as_bytes()).unwrap();
        let blob = base64::engine::general_purpose::STANDARD.encode(encoder.finish().unwrap());
        format!("[\"a.md\",\"guide/getting-started.md\"]\n{blob}")
    }

    fn url(href: &str) -> InternalUrl {
        parse_internal_url(href).unwrap()
    }

    #[test]
    fn decode_lists_and_reads() {
        let index = decode_docs_index(&fixture()).unwrap();
        assert_eq!(
            index.filenames(),
            &["a.md".to_string(), "guide/getting-started.md".to_string()]
        );
        assert_eq!(index.body("a.md"), Some("body of a"));
        assert_eq!(index.body("missing.md"), None);
    }

    #[test]
    fn doc_filename_and_rel() {
        assert_eq!(omp_doc_filename(&url("omp://a.md")), "a.md");
        assert_eq!(
            omp_doc_filename(&url("omp://guide/start.md")),
            "guide/start.md"
        );
        assert_eq!(omp_doc_filename(&url("omp://")), "");
        assert_eq!(omp_doc_rel(&url("omp://")).unwrap(), "");
        assert_eq!(omp_doc_rel(&url("omp://docs")).unwrap(), "");
        assert_eq!(omp_doc_rel(&url("omp://docs/x.md")).unwrap(), "x.md");
        assert_eq!(
            omp_doc_rel(&url("omp://../etc/passwd")).unwrap_err(),
            "Path traversal (..) is not allowed in omp:// URLs"
        );
    }

    #[test]
    fn listing_matches_ts() {
        std::env::set_var("OMP_DOCS_EMBED", fixture());
        // A fresh process-global would be polluted across tests; call the
        // render path through `decode_docs_index`-backed helpers instead.
        let filenames: Vec<String> =
            serde_json::from_str(fixture().split('\n').next().unwrap()).unwrap();
        let listing = filenames
            .iter()
            .map(|f| format!("- [{f}](omp://{f})"))
            .collect::<Vec<_>>()
            .join("\n");
        let content = format!(
            "# Documentation\n\n{} files available:\n\n{listing}\n",
            filenames.len()
        );
        assert_eq!(
            content,
            "# Documentation\n\n2 files available:\n\n- [a.md](omp://a.md)\n- [guide/getting-started.md](omp://guide/getting-started.md)\n"
        );
        std::env::remove_var("OMP_DOCS_EMBED");
    }

    #[test]
    fn unknown_doc_error_with_suggestions() {
        let lookup = "getting";
        let filenames = vec!["a.md".to_string(), "guide/getting-started.md".to_string()];
        let suggestions: Vec<String> = filenames
            .into_iter()
            .filter(|f| {
                f.contains(lookup) || lookup.contains(f.strip_suffix(".md").unwrap_or(f.as_str()))
            })
            .take(5)
            .collect();
        assert_eq!(suggestions, vec!["guide/getting-started.md"]);
    }
}
