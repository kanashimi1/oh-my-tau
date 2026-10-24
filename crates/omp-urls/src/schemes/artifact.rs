//! `artifact://` provider (TS `internal-urls/artifact-protocol.ts`).

use std::path::PathBuf;

use crate::parse::InternalUrl;
use crate::spec::{InternalResource, SchemeBacking, SchemeSelectors, SchemeSpec};
use crate::{UrlCtx, UrlProvider};

use crate::schemes::core::artifacts_dirs;

const MAX_INLINE_ARTIFACT_BYTES: u64 = 8 * 1024 * 1024;

/// Numeric artifact id (TS `parseArtifactId`).
fn parse_artifact_id(url: &InternalUrl) -> anyhow::Result<&str> {
    let id = url.host.as_str();
    if id.is_empty() {
        anyhow::bail!("artifact:// URL requires a numeric ID: artifact://0");
    }
    if !id.chars().all(|c: char| c.is_ascii_digit()) {
        anyhow::bail!("artifact:// ID must be numeric, got: {id}");
    }
    Ok(id)
}

/// Resolve an `artifact://` URL to its backing file without reading bytes.
/// Returns `Err(MissingArtifact)` for unknown ids and `Err(String)`-free
/// anyhow errors for malformed ones and miss diagnostics.
pub(crate) fn resolve_artifact_file(
    url: &InternalUrl,
    ctx: &UrlCtx,
) -> anyhow::Result<PathBuf, ArtifactError> {
    let id = parse_artifact_id(url).map_err(ArtifactError::Message)?;
    let dirs = artifacts_dirs(ctx);
    if dirs.is_empty() {
        return Err(ArtifactError::Missing(
            "No session - artifacts unavailable".to_string(),
        ));
    }
    let mut any_dir_exists = false;
    let mut available: Vec<u64> = Vec::new();
    for dir in &dirs {
        let Ok(files) = std::fs::read_dir(dir) else {
            continue;
        };
        any_dir_exists = true;
        for entry in files.flatten() {
            let Some(name) = entry.file_name().to_str().map(ToOwned::to_owned) else {
                continue;
            };
            if let Some(rest) = name.strip_prefix(id).and_then(|r| r.strip_prefix('.')) {
                let _ = rest;
                return Ok(entry.path());
            }
            if let Some(num) = name.split('.').next().and_then(|p| p.parse::<u64>().ok()) {
                if !available.contains(&num) {
                    available.push(num);
                }
            }
        }
    }
    if !any_dir_exists {
        return Err(ArtifactError::Missing(
            "No artifacts directory found".to_string(),
        ));
    }
    available.sort_unstable();
    let available_str = if available.is_empty() {
        "none".to_string()
    } else {
        available
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    Err(ArtifactError::Missing(format!(
        "Artifact {id} not found. Available: {available_str}"
    )))
}

pub(crate) enum ArtifactError {
    Missing(String),
    Message(anyhow::Error),
}

impl From<ArtifactError> for anyhow::Error {
    fn from(e: ArtifactError) -> Self {
        match e {
            ArtifactError::Missing(m) => anyhow::Error::msg(m),
            ArtifactError::Message(err) => err,
        }
    }
}

/// `artifact://` provider: per-session counter files `<id>.*` in artifacts dirs.
pub struct ArtifactProvider;

impl UrlProvider for ArtifactProvider {
    fn scheme(&self) -> &'static str {
        "artifact"
    }

    fn spec(&self) -> &SchemeSpec {
        const SPEC: SchemeSpec = SchemeSpec {
            backing: SchemeBacking::File,
            selectors: SchemeSelectors::Lines,
            immutable: true,
            single_slash_alias: false,
            linkable: true,
            port_authority: false,
            write: None,
        };
        &SPEC
    }

    fn impl_locate(&self) -> bool {
        true
    }

    fn read(&self, url: &InternalUrl, ctx: &UrlCtx) -> anyhow::Result<InternalResource> {
        let path = resolve_artifact_file(url, ctx)?;
        let size = std::fs::metadata(&path)?.len();
        if size > MAX_INLINE_ARTIFACT_BYTES {
            let id = parse_artifact_id(url)?;
            anyhow::bail!(
                "Artifact {id} is {size} bytes; full internal resolution is blocked. Use read selectors such as artifact://{id}:1-3000 or artifact://{id}:raw:1-3000, and use the artifact file path for search/copy workflows: {}",
                path.display()
            );
        }
        let mut r = InternalResource::text(std::fs::read_to_string(&path)?);
        r.path = Some(path.to_string_lossy().into_owned());
        Ok(r)
    }

    fn locate(&self, url: &InternalUrl, ctx: &UrlCtx) -> Option<PathBuf> {
        match resolve_artifact_file(url, ctx) {
            Ok(path) => Some(path),
            Err(ArtifactError::Missing(_)) => None,
            Err(ArtifactError::Message(_)) => None,
        }
    }

    fn complete(&self, _query: &str, ctx: &UrlCtx) -> Vec<crate::UrlCompletion> {
        let mut ids: Vec<u64> = Vec::new();
        for dir in artifacts_dirs(ctx) {
            let Ok(files) = std::fs::read_dir(dir) else {
                continue;
            };
            for entry in files.flatten() {
                let Some(name) = entry.file_name().to_str().map(ToOwned::to_owned) else {
                    continue;
                };
                if let Some(num) = name.split('.').next().and_then(|p| p.parse::<u64>().ok()) {
                    if !ids.contains(&num) {
                        ids.push(num);
                    }
                }
            }
        }
        ids.sort_unstable();
        ids.into_iter()
            .map(|id| crate::UrlCompletion {
                label: id.to_string(),
                detail: None,
            })
            .collect()
    }
}
