//! Snapcompact savings reader (port of the record schema, file cache and
//! project filtering in `gain-aggregator.ts`). The JSONL lives next to
//! `stats.db`; a missing file is zero records.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

use anyhow::Result;
use serde::Deserialize;

/// One snapcompact savings record (`snapcompact-savings.jsonl` line).
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub(crate) struct SnapcompactRecord {
    /// Epoch ms.
    pub ts: i64,
    pub session: String,
    #[serde(default)]
    pub tool_call_id: String,
    #[serde(default)]
    pub saved_tokens: i64,
}

pub(crate) struct SnapcompactSets {
    pub records: Vec<SnapcompactRecord>,
    pub projects: HashSet<String>,
}

struct SnapcompactCache {
    key: String,
    records: Vec<SnapcompactRecord>,
}

static SNAPCOMPACT_CACHE: Mutex<Option<SnapcompactCache>> = Mutex::new(None);

/// `<stats.db dir>/snapcompact-savings.jsonl`, i.e. `~/.omp/…` — resolved
/// relative to HOME exactly like the TS `getStatsDbPath` + `path.dirname`.
fn savings_file_path() -> PathBuf {
    crate::paths::stats_db_path()
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
        .join("snapcompact-savings.jsonl")
}

fn mtime_ms(meta: &fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .and_then(|d| i64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

/// Read, cache and filter the snapcompact records.
pub(crate) fn read_snapcompact_records(
    cutoff: Option<i64>,
    project: Option<&str>,
) -> Result<SnapcompactSets> {
    let file_path = savings_file_path();

    let meta = match fs::metadata(&file_path) {
        Ok(meta) => meta,
        // Missing file → zero records, never an error.
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(SnapcompactSets {
                records: Vec::new(),
                projects: HashSet::new(),
            });
        }
        Err(err) => {
            tracing::debug!(error = %err, "gain-aggregator: failed to stat snapcompact-savings.jsonl");
            return Ok(SnapcompactSets {
                records: Vec::new(),
                projects: HashSet::new(),
            });
        }
    };

    let cache_key = format!("{}:{}:{}", file_path.display(), mtime_ms(&meta), meta.len());
    let parsed: Vec<SnapcompactRecord> = {
        let mut cache = SNAPCOMPACT_CACHE.lock().expect("snapcompact cache lock");
        if cache.as_ref().is_some_and(|c| c.key == cache_key) {
            cache.as_ref().expect("checked").records.clone()
        } else {
            let text = match fs::read_to_string(&file_path) {
                Ok(text) => text,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(SnapcompactSets {
                        records: Vec::new(),
                        projects: HashSet::new(),
                    });
                }
                Err(err) => {
                    tracing::debug!(error = %err, "gain-aggregator: failed to read snapcompact-savings.jsonl");
                    return Ok(SnapcompactSets {
                        records: Vec::new(),
                        projects: HashSet::new(),
                    });
                }
            };
            let mut records = Vec::new();
            for line in text.split('\n') {
                if line.trim().is_empty() {
                    continue;
                }
                // Skip malformed lines.
                if let Ok(rec) = serde_json::from_str::<SnapcompactRecord>(line) {
                    records.push(rec);
                }
            }
            *cache = Some(SnapcompactCache {
                key: cache_key,
                records: records.clone(),
            });
            records
        }
    };

    let filtered: Vec<SnapcompactRecord> = match cutoff {
        None => parsed,
        Some(cutoff) => parsed.into_iter().filter(|rec| rec.ts >= cutoff).collect(),
    };

    // Dedup by session:toolCallId, keeping first occurrence.
    let mut seen: HashSet<String> = HashSet::new();
    let mut deduped: Vec<SnapcompactRecord> = Vec::new();
    for rec in filtered {
        let key = format!("{}:{}", rec.session, rec.tool_call_id);
        if seen.insert(key) {
            deduped.push(rec);
        }
    }

    let sessions: Vec<String> = deduped.iter().map(|rec| rec.session.clone()).collect();
    let projects_by_session = super::read_projects_by_session(&sessions)?;

    let mut projects: HashSet<String> = HashSet::new();
    let mut records: Vec<SnapcompactRecord> = Vec::new();
    for rec in deduped {
        let session_projects = projects_by_session.get(&rec.session);
        if let Some(session_projects) = session_projects {
            projects.extend(session_projects.iter().cloned());
        }
        if let Some(project) = project {
            let matched = session_projects
                .is_some_and(|sp| sp.iter().any(|p| super::matches_project(Some(p), project)));
            if !matched {
                continue;
            }
        }
        records.push(rec);
    }

    Ok(SnapcompactSets { records, projects })
}
