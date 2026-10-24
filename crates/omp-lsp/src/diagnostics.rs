//! Diagnostic waiting and location helpers (omp `lsp/diagnostics.ts`).
//!
//! `wait_for_diagnostics` gates on the publish channel (version-matched
//! publish, or a settled unversioned publish) and prefers an LSP 3.17
//! document-diagnostic pull when the server advertised support. A failed pull
//! never collapses into a clean result.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{json, Value};

use crate::client::LspClient;
use crate::types::{Diagnostic, Location, Position, Range};
use crate::utils::{format_location, read_location_context, uri_to_file};

pub const SINGLE_DIAGNOSTICS_WAIT_TIMEOUT_MS: u64 = 3000;
pub const PROJECT_DIAGNOSTICS_WAIT_TIMEOUT_MS: u64 = 10_000;
pub const BATCH_DIAGNOSTICS_WAIT_TIMEOUT_MS: u64 = 400;
pub const DEFERRED_DIAGNOSTICS_WAIT_TIMEOUT_MS: u64 = 12_000;
pub const MAX_GLOB_DIAGNOSTIC_TARGETS: usize = 20;
pub const WORKSPACE_SYMBOL_LIMIT: usize = 200;
pub const REFERENCE_CONTEXT_LIMIT: usize = 50;
pub const REFERENCES_RETRY_COUNT: usize = 2;
pub const REFERENCES_RETRY_DELAY_MS: u64 = 250;
const DIAGNOSTICS_POLL_MS: u64 = 100;
const DIAGNOSTICS_SETTLE_MS: u64 = 250;

/// Actions that need a fully indexed project before they return useful data.
pub const PROJECT_INDEXED_ACTIONS: [&str; 6] = [
    "definition",
    "type_definition",
    "implementation",
    "references",
    "rename",
    "hover",
];

const RUST_WORKSPACE_MARKERS: [&str; 2] = ["Cargo.toml", "rust-analyzer.toml"];

/// Whether any ancestor of `file_path` contains a Rust workspace marker.
pub fn has_rust_workspace_ancestor(file_path: &Path) -> bool {
    let mut dir = file_path.parent().map(Path::to_path_buf);
    while let Some(d) = dir {
        for marker in RUST_WORKSPACE_MARKERS {
            if d.join(marker).exists() {
                return true;
            }
        }
        let parent = d.parent().map(Path::to_path_buf);
        match parent {
            Some(p) if p != d => dir = Some(p),
            _ => return false,
        }
    }
    false
}

fn compare_position(a: &Position, b: &Position) -> std::cmp::Ordering {
    a.line
        .cmp(&b.line)
        .then_with(|| a.character.cmp(&b.character))
}

fn range_contains_position(range: &Range, position: &Position) -> bool {
    compare_position(&range.start, position) != std::cmp::Ordering::Greater
        && compare_position(position, &range.end) != std::cmp::Ordering::Greater
}

/// Single-location result containing exactly the queried declaration.
#[must_use]
pub fn is_only_queried_declaration(locations: &[Location], uri: &str, position: &Position) -> bool {
    locations.len() == 1
        && locations[0].uri == uri
        && range_contains_position(&locations[0].range, position)
}

/// Normalize `Location | Location[] | LocationLink | LocationLink[]` to `Location[]`.
///
/// # Panics
///
/// Panics if `result` is an array but its `as_array` view is unavailable
/// (impossible for JSON arrays).
#[must_use]
pub fn normalize_location_result(result: &Value) -> Vec<Location> {
    if result.is_null() {
        return Vec::new();
    }
    let raw: Vec<&Value> = if result.is_array() {
        result.as_array().unwrap().iter().collect()
    } else {
        vec![result]
    };
    let mut out = Vec::new();
    for loc in raw {
        if loc.get("uri").is_some() {
            if let Ok(location) = serde_json::from_value::<Location>(loc.clone()) {
                out.push(location);
            }
        } else if loc.get("targetUri").is_some() {
            if let Ok(link) = serde_json::from_value::<crate::types::LocationLink>(loc.clone()) {
                out.push(Location {
                    uri: link.target_uri,
                    range: link.target_selection_range.unwrap_or(link.target_range),
                });
            }
        }
    }
    out
}

const LOCATION_CONTEXT_LINES: usize = 1;

/// Format `  file:line:col` plus one line of context above and below.
pub async fn format_location_with_context(location: &Location, cwd: &Path) -> String {
    let header = format!("  {}", format_location(location, cwd));
    let context = read_location_context(
        &uri_to_file(&location.uri),
        location.range.start.line as usize + 1,
        LOCATION_CONTEXT_LINES,
    )
    .await;
    if context.is_empty() {
        return header;
    }
    let body = context
        .iter()
        .map(|line| format!("    {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{header}\n{body}")
}

struct PullOutcome {
    diagnostics: Option<Vec<Diagnostic>>,
    failed: Option<String>,
}

/// Wait for diagnostics for `uri` (omp `waitForDiagnostics`).
///
/// # Errors
///
/// Returns the pull failure when a document-diagnostic pull was attempted and
/// failed, so a timeout is never rendered as a clean file.
pub async fn wait_for_diagnostics(
    client: &LspClient,
    uri: &str,
    timeout_ms: u64,
    cancel: Option<&dyn crate::Cancel>,
    min_version: Option<u64>,
    expected_document_version: Option<i64>,
) -> Result<Vec<Diagnostic>, String> {
    let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);
    let mut pull_attempted = false;
    let mut pulled: Option<Vec<Diagnostic>> = None;
    let mut pull_failure: Option<String> = None;
    let mut settled_key: Option<PathBuf> = None;
    let mut settled_at = tokio::time::Instant::now();

    loop {
        if let Some(c) = cancel {
            if c.is_cancelled() {
                return Err("operation cancelled".to_string());
            }
        }
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        if !pull_attempted && client.supports_document_diagnostics() {
            pull_attempted = true;
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            match request_document_diagnostics(client, uri, remaining).await {
                PullOutcome {
                    diagnostics: Some(items),
                    ..
                } => {
                    pulled = Some(items);
                    break;
                }
                PullOutcome {
                    failed: Some(e), ..
                } => pull_failure = Some(e),
                _ => {}
            }
        }

        let version_ok = min_version.is_none_or(|v| client.diagnostics_version() > v);
        let published = client.published_diagnostics(uri);
        if let Some(published) = published.clone().filter(|_| version_ok) {
            if expected_document_version.is_some_and(|v| published.version == Some(v)) {
                return Ok(published.diagnostics);
            }
            let key = crate::utils::uri_key(uri);
            if settled_key.as_ref() != Some(&key) {
                settled_key = Some(key);
                settled_at = tokio::time::Instant::now();
            } else if settled_at.elapsed() >= Duration::from_millis(DIAGNOSTICS_SETTLE_MS)
                && !is_provisional_cold_publish(client, &published, expected_document_version)
            {
                return Ok(published.diagnostics);
            }
        }

        let poll = Duration::from_millis(DIAGNOSTICS_POLL_MS)
            .min(deadline.saturating_duration_since(tokio::time::Instant::now()));
        tokio::select! {
            () = tokio::time::sleep(poll) => {}
            () = async { match cancel {
                Some(c) => c.cancelled().await,
                None => std::future::pending().await,
            } } => {
                return Err("operation cancelled".to_string());
            }
        }
    }

    // Final acceptance pass on deadline.
    let version_ok = min_version.is_none_or(|v| client.diagnostics_version() > v);
    if let Some(published) = client.published_diagnostics(uri).filter(|_| version_ok) {
        if expected_document_version.is_some_and(|v| published.version == Some(v)) {
            return Ok(published.diagnostics);
        }
        if settled_key.as_ref() == Some(&crate::utils::uri_key(uri))
            && settled_at.elapsed() >= Duration::from_millis(DIAGNOSTICS_SETTLE_MS)
            && !is_provisional_cold_publish(client, &published, expected_document_version)
        {
            return Ok(published.diagnostics);
        }
    }
    if let Some(items) = pulled {
        client.store_pulled_diagnostics(uri, items.clone(), expected_document_version);
        return Ok(items);
    }
    if let Some(e) = pull_failure {
        return Err(e);
    }
    Ok(Vec::new())
}

/// An empty unversioned publish right after server start is often a placeholder.
fn is_provisional_cold_publish(
    client: &LspClient,
    published: &crate::types::PublishedDiagnostics,
    expected_document_version: Option<i64>,
) -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(0));
    let age = u64::try_from(now.saturating_sub(client.started_at())).unwrap_or(u64::MAX);
    expected_document_version.is_some()
        && published.version.is_none()
        && published.diagnostics.is_empty()
        && age < DEFERRED_DIAGNOSTICS_WAIT_TIMEOUT_MS
}

async fn request_document_diagnostics(
    client: &LspClient,
    uri: &str,
    timeout: Duration,
) -> PullOutcome {
    match client
        .send_request(
            "textDocument/diagnostic",
            json!({ "textDocument": { "uri": uri } }),
            None,
            Some(timeout),
        )
        .await
    {
        Ok(report) => {
            if report.get("kind").and_then(Value::as_str) != Some("full") {
                return PullOutcome {
                    diagnostics: None,
                    failed: None,
                };
            }
            let items = report
                .get("items")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| serde_json::from_value::<Diagnostic>(v.clone()).ok())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            PullOutcome {
                diagnostics: Some(items),
                failed: None,
            }
        }
        Err(e) => {
            tracing::debug!("LSP document diagnostic pull failed: {}", e.message);
            PullOutcome {
                diagnostics: None,
                failed: Some(e.message),
            }
        }
    }
}

// =============================================================================
// Diagnostic Targets (glob expansion)
// =============================================================================

const GLOB_PATTERN_CHARS: [char; 4] = ['*', '?', '[', '{'];

#[must_use]
pub fn has_glob_pattern(value: &str) -> bool {
    value.chars().any(|c| GLOB_PATTERN_CHARS.contains(&c))
}

/// Expand `pattern` against `cwd` honoring `max_matches` (omp `collectGlobMatches`).
///
/// # Errors
///
/// Filesystem errors from reading `cwd` and its descendants.
pub async fn collect_glob_matches(
    pattern: &str,
    cwd: &Path,
    max_matches: usize,
) -> std::io::Result<(Vec<String>, bool)> {
    let limit = max_matches.max(1);
    let mut matches = Vec::new();
    let mut entries = tokio::fs::read_dir(cwd).await?;
    let mut stack: Vec<PathBuf> = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        stack.push(entry.path());
    }
    // Walk recursively, matching relative paths against the glob.
    let mut results: Vec<String> = Vec::new();
    while let Some(path) = stack.pop() {
        let md = tokio::fs::metadata(&path).await?;
        let rel = path
            .strip_prefix(cwd)
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        if md.is_dir() {
            let mut sub = tokio::fs::read_dir(&path).await?;
            while let Some(entry) = sub.next_entry().await? {
                stack.push(entry.path());
            }
            continue;
        }
        if glob_match(pattern, &rel) && !results.contains(&rel) {
            results.push(rel);
        }
    }
    results.sort();
    let truncated = results.len() > limit;
    matches.extend(results.into_iter().take(limit));
    Ok((matches, truncated))
}

/// Minimal glob: `**` crosses directories, `*` within one segment, `?` one char.
fn glob_match(pattern: &str, name: &str) -> bool {
    fn seg_match(p: &[char], n: &[char]) -> bool {
        if p.is_empty() {
            return n.is_empty();
        }
        match p[0] {
            '*' => (0..=n.len()).any(|i| seg_match(&p[1..], &n[i..])),
            '?' => !n.is_empty() && seg_match(&p[1..], &n[1..]),
            c => !n.is_empty() && n[0] == c && seg_match(&p[1..], &n[1..]),
        }
    }
    let pseg: Vec<&str> = pattern.split('/').collect();
    let mut name_rest = name;
    let mut pi = 0;
    while pi < pseg.len() {
        let pat = pseg[pi];
        if pat == "**" {
            // Match zero or more whole segments.
            for skip in 0..=name_rest.matches('/').count() {
                let mut idx = 0;
                let mut rest = name_rest;
                for _ in 0..skip {
                    if let Some(i) = rest.find('/') {
                        idx += i + 1;
                        rest = &name_rest[idx..];
                    }
                }
                if glob_match(&pseg[pi + 1..].join("/"), rest) {
                    return true;
                }
            }
            return false;
        }
        let (seg, rest) = match name_rest.find('/') {
            Some(i) => (&name_rest[..i], &name_rest[i + 1..]),
            None => (name_rest, ""),
        };
        if !seg_match(
            &pat.chars().collect::<Vec<_>>(),
            &seg.chars().collect::<Vec<_>>(),
        ) {
            return false;
        }
        name_rest = rest;
        pi += 1;
    }
    name_rest.is_empty() && pi == pseg.len()
}

/// Resolve a diagnostics target: literal path, existing file, or glob
/// (omp `resolveDiagnosticTargets`). Returned matches are as the user spelled
/// them (relative to cwd) or the literal pattern.
///
/// # Errors
///
/// Filesystem errors while globbing the workspace.
pub async fn resolve_diagnostic_targets(
    file: &str,
    cwd: &Path,
    max_matches: usize,
) -> std::io::Result<(Vec<String>, bool)> {
    if !has_glob_pattern(file) {
        return Ok((vec![file.to_string()], false));
    }
    let resolved = crate::utils::resolve_to_cwd(file, cwd);
    if let Ok(md) = tokio::fs::metadata(&resolved).await {
        if md.is_file() {
            return Ok((vec![file.to_string()], false));
        }
    }
    collect_glob_matches(file, cwd, max_matches).await
}
