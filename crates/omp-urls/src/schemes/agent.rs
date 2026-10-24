//! `agent://` provider (TS `internal-urls/agent-protocol.ts`).
//!
//! Resolves `<id>.md` outputs from artifacts dirs with optional JSON-path
//! extraction. The live-registry progress/superseded paths need the agent
//! registry omp-rs does not have yet; not-found errors match TS.

use std::path::PathBuf;

use crate::parse::InternalUrl;
use crate::spec::{InternalResource, SchemeBacking, SchemeSelectors, SchemeSpec};
use crate::{UrlCtx, UrlProvider};

use crate::schemes::core::{artifacts_dirs, fuzzy_suggest};

const MAX_ID_SUGGESTIONS: usize = 5;

fn output_id(url: &InternalUrl) -> anyhow::Result<&str> {
    let id = url.host.as_str();
    if id.is_empty() {
        anyhow::bail!("agent:// URL requires an output ID: agent://<id>");
    }
    Ok(id)
}

fn has_path_extraction(url: &InternalUrl) -> bool {
    !url.pathname.is_empty() && url.pathname != "/"
}

/// TS `extractJsonPath`: object segments index by key, array segments by index.
fn extract_json_path<'a>(
    mut value: &'a serde_json::Value,
    segments: &[String],
) -> Option<&'a serde_json::Value> {
    for segment in segments {
        match value {
            serde_json::Value::Array(items) => {
                let index: usize = segment.parse().ok()?;
                value = items.get(index)?;
            }
            serde_json::Value::Object(map) => value = map.get(segment)?,
            _ => return None,
        }
    }
    Some(value)
}

/// TS `#findOutput`: `<id>.md` first-hit per id, same-dir `<id>.json` sidecar.
struct OutputScan {
    found: Option<PathBuf>,
    json: Option<PathBuf>,
    any_dir_exists: bool,
    available: Vec<String>,
}

fn find_output(id: &str, ctx: &UrlCtx) -> OutputScan {
    let mut by_id: Vec<(String, PathBuf)> = Vec::new();
    let mut json_by_id: Vec<(String, PathBuf)> = Vec::new();
    let mut any_dir_exists = false;
    for dir in artifacts_dirs(ctx) {
        let Ok(files) = std::fs::read_dir(&dir) else {
            continue;
        };
        any_dir_exists = true;
        for entry in files.flatten() {
            let Some(name) = entry.file_name().to_str().map(ToOwned::to_owned) else {
                continue;
            };
            if let Some(json_id) = name.strip_suffix(".json") {
                if !json_by_id.iter().any(|(existing, _)| existing == json_id) {
                    json_by_id.push((json_id.to_string(), entry.path()));
                }
                continue;
            }
            if let Some(file_id) = name.strip_suffix(".md") {
                if !by_id.iter().any(|(existing, _)| existing == file_id) {
                    by_id.push((file_id.to_string(), entry.path()));
                }
            }
        }
    }
    let available = by_id.iter().map(|(id, _)| id.clone()).collect();
    let found = by_id
        .iter()
        .find(|(existing, _)| existing == id)
        .map(|(_, p)| p.clone());
    // Pair the sidecar only from the SAME dir as the matched `.md`.
    let json = found.as_ref().and_then(|found_path| {
        json_by_id
            .iter()
            .find(|(existing, _)| existing == id)
            .filter(|(_, p)| p.parent() == found_path.parent())
            .map(|(_, p)| p.clone())
    });
    OutputScan {
        found,
        json,
        any_dir_exists,
        available,
    }
}

/// TS `Output <id> is not valid JSON: <message>`.
fn parse_output_json(id: &str, raw: &str) -> anyhow::Result<serde_json::Value> {
    serde_json::from_str(raw).map_err(|err| anyhow::anyhow!("Output {id} is not valid JSON: {err}"))
}

/// TS `notFoundError`: closest few ids, else point at history://.
fn not_found(output_id: &str, candidates: &[String]) -> anyhow::Error {
    let mut unique: Vec<String> = Vec::new();
    for c in candidates {
        if c != output_id && !unique.contains(c) {
            unique.push(c.clone());
        }
    }
    let suggestions = fuzzy_suggest(&unique, output_id, MAX_ID_SUGGESTIONS);
    let hint = if suggestions.is_empty() {
        "List agents with history://".to_string()
    } else {
        format!("Did you mean: {}", suggestions.join(", "))
    };
    anyhow::Error::msg(format!("Not found: {output_id}\n{hint}"))
}

/// `agent://` provider.
pub struct AgentProvider;

impl UrlProvider for AgentProvider {
    fn scheme(&self) -> &'static str {
        "agent"
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
        let id = output_id(url)?;
        if id == "all" {
            anyhow::bail!("agent://all is write-only; use it to broadcast a message.");
        }
        let extraction = has_path_extraction(url);
        let scan = find_output(id, ctx);
        let Some(found) = scan.found else {
            if !scan.any_dir_exists && artifacts_dirs(ctx).is_empty() {
                anyhow::bail!("No session - agent outputs unavailable");
            }
            if !scan.any_dir_exists {
                anyhow::bail!("No artifacts directory found");
            }
            return Err(not_found(id, &scan.available));
        };

        if extraction {
            let segments: Vec<String> = url
                .pathname
                .split('/')
                .filter(|s| !s.is_empty())
                .map(crate::schemes::decode_component)
                .collect();
            // Sidecar `<id>.json` wins; else the `.md` content itself must parse.
            let (json_value, extracted_from) = if let Some(sidecar) = &scan.json {
                let raw = std::fs::read_to_string(sidecar)?;
                match serde_json::from_str(&raw) {
                    Ok(value) => (value, sidecar.clone()),
                    Err(_) => (
                        parse_output_json(id, &std::fs::read_to_string(&found)?)?,
                        found.clone(),
                    ),
                }
            } else {
                (
                    parse_output_json(id, &std::fs::read_to_string(&found)?)?,
                    found.clone(),
                )
            };
            let extracted = extract_json_path(&json_value, &segments);
            // A string field reads as prose; anything else pretty-printed JSON
            // (`undefined` extraction included — TS renders "null").
            let content = match extracted {
                Some(serde_json::Value::String(text)) => text.clone(),
                Some(value) => {
                    serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".to_string())
                }
                None => "null".to_string(),
            };
            let mut r = InternalResource::text(content);
            r.path = Some(extracted_from.to_string_lossy().into_owned());
            return Ok(r);
        }

        let content = std::fs::read_to_string(&found)?;
        let mut r = InternalResource::text(content);
        r.path = Some(found.to_string_lossy().into_owned());
        Ok(r)
    }

    fn locate(&self, url: &InternalUrl, ctx: &UrlCtx) -> Option<PathBuf> {
        let id = output_id(url).ok()?;
        if id == "all" || has_path_extraction(url) {
            return None;
        }
        find_output(id, ctx).found
    }

    fn complete(&self, _query: &str, ctx: &UrlCtx) -> Vec<crate::UrlCompletion> {
        let mut scan = find_output("\u{0}none", ctx);
        scan.available.sort();
        scan.available
            .into_iter()
            .map(|label| crate::UrlCompletion {
                label,
                detail: None,
            })
            .collect()
    }
}
