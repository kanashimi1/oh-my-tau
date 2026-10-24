//! Port of `stats/src/gain-aggregator.ts`: token-savings aggregation for the
//! Gain dashboard. Source of truth is the snapcompact savings JSONL colocated
//! with `stats.db` (`~/.omp/snapcompact-savings.jsonl`); per-session project
//! folders come from the ingested `messages` table. Missing files are zero
//! records — never an error.

mod snapcompact;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::LazyLock;

use anyhow::Result;
use regex::Regex;

use crate::types::{GainDashboardStats, GainSource, GainSourceTotals, GainTimeSeriesPoint};

const BYTES_PER_TOKEN_ESTIMATE: i64 = 4;
const DAY_MS: i64 = 86_400_000;

/// Paths that carry no dashboard signal — temp/internal locations.
static TEMP_PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?:^|/)(?:T|tmp|pi-bash-exec|omp-bash-exec|pi-bash-detach)(?:/|$)|^/var/folders(?:/|$)"#,
    )
    .expect("temp path regex")
});

/// Strip trailing separators and normalize backslashes; empty becomes "/".
#[must_use]
pub(crate) fn canonical_project_path(p: &str) -> String {
    let normalized = p.replace('\\', "/");
    let trimmed = normalized.trim_end_matches('/');
    if trimmed.is_empty() {
        "/".into()
    } else {
        trimmed.into()
    }
}

/// True when `candidate` exactly equals `parent` or is a separator-bounded
/// sub-path.
#[must_use]
pub(crate) fn is_same_or_sub_path(candidate: &str, parent: &str) -> bool {
    let c = canonical_project_path(candidate);
    let p = canonical_project_path(parent);
    c == p || c.starts_with(&format!("{p}/"))
}

/// Collapse conventional worktree sub-paths to their logical project root.
///
/// Returns `None` to drop temp/internal paths entirely.
#[must_use]
pub fn normalize_project_path(p: &str) -> Option<String> {
    let clean = canonical_project_path(p);
    if TEMP_PATH_RE.is_match(&clean) {
        return None;
    }
    if clean.contains("/.omp/wt/") {
        return None;
    }

    static WORKTREE_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
        [
            r#"^(.+)/\.wt/[^/]+(?:/.*)?$"#,
            r#"^(.+)/\.worktrees/[^/]+(?:/.*)?$"#,
            r#"^(.+)-wt/[^/]+(?:/.*)?$"#,
            r#"^(.+)-worktrees/[^/]+(?:/.*)?$"#,
            r#"^(.+)\.wt/[^/]+(?:/.*)?$"#,
        ]
        .iter()
        .map(|r| Regex::new(r).expect("worktree regex"))
        .collect()
    });
    for pattern in WORKTREE_PATTERNS.iter() {
        if let Some(caps) = pattern.captures(&clean) {
            if let Some(root) = caps.get(1) {
                return Some(canonical_project_path(root.as_str()));
            }
        }
    }

    Some(clean)
}

/// True when `cwd` (or its normalized project root) exactly equals `project`
/// or is a direct sub-path of it.
fn matches_project(cwd: Option<&str>, project: &str) -> bool {
    let Some(cwd) = cwd else { return false };
    let normalized_cwd = normalize_project_path(cwd).unwrap_or_else(|| canonical_project_path(cwd));
    let normalized_project =
        normalize_project_path(project).unwrap_or_else(|| canonical_project_path(project));
    is_same_or_sub_path(&normalized_cwd, &normalized_project)
        || is_same_or_sub_path(cwd, &normalized_project)
}

/// Normalize worktree paths and remove sub-paths already covered by a shorter
/// parent at depth >= 4. Returns a sorted, deduped list of project roots.
#[must_use]
pub fn dedupe_projects(raw_paths: &HashSet<String>) -> Vec<String> {
    let mut normalized: HashSet<String> = HashSet::new();
    for p in raw_paths {
        if let Some(n) = normalize_project_path(p) {
            normalized.insert(n);
        }
    }
    let mut sorted: Vec<String> = normalized.into_iter().collect();
    sorted.sort();
    let mut result: Vec<String> = Vec::new();
    for p in &sorted {
        let covered = sorted.iter().any(|other| {
            other != p
                && other.len() < p.len()
                && is_same_or_sub_path(p, other)
                && other.split('/').filter(|s| !s.is_empty()).count() >= 4
        });
        if !covered {
            result.push(p.clone());
        }
    }
    result
}

fn empty_totals() -> GainSourceTotals {
    GainSourceTotals::default()
}

/// ISO date string (UTC) from epoch ms, bucketed to the day.
fn to_date_bucket(epoch_ms: i64) -> String {
    let days = epoch_ms.div_euclid(DAY_MS);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe.div_euclid(1460) + doe.div_euclid(36_524) - doe.div_euclid(146_096))
        .div_euclid(365);
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe.div_euclid(4) - yoe.div_euclid(100));
    let mp = (5 * doy + 2).div_euclid(153);
    let d = doy - (153 * mp + 2).div_euclid(5) + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// Aggregate the Gain dashboard payload for `range` (optional `?range=` value)
/// and an optional project selector.
pub fn get_gain_dashboard_stats(
    range: Option<&str>,
    project: Option<&str>,
) -> Result<GainDashboardStats> {
    let window = crate::rollup::get_time_range_config(range);
    let effective_project = project.map(str::trim).filter(|s| !s.is_empty());

    let snapcompact::SnapcompactSets { records, projects } =
        snapcompact::read_snapcompact_records(window.cutoff, effective_project)?;

    let mut snapcompact_totals = empty_totals();
    let mut time_map: BTreeMap<String, f64> = BTreeMap::new();

    for rec in &records {
        snapcompact_totals.saved_tokens += rec.saved_tokens;
        snapcompact_totals.saved_bytes += rec.saved_tokens * BYTES_PER_TOKEN_ESTIMATE;
        snapcompact_totals.hits += 1;
        *time_map.entry(to_date_bucket(rec.ts)).or_default() += rec.saved_tokens as f64;
    }
    // No originalBytes for snapcompact — reductionPercent stays null.

    let overall = GainSourceTotals {
        saved_tokens: snapcompact_totals.saved_tokens,
        saved_bytes: snapcompact_totals.saved_bytes,
        hits: snapcompact_totals.hits,
        ..empty_totals()
    };

    let time_series: Vec<GainTimeSeriesPoint> = time_map
        .into_iter()
        .map(|(date, snapcompact)| GainTimeSeriesPoint {
            date,
            snapcompact,
            total: snapcompact,
        })
        .collect();

    let project_set: HashSet<String> = projects.into_iter().collect();
    let projects = dedupe_projects(&project_set);

    Ok(GainDashboardStats {
        overall,
        by_source: BTreeMap::from([(GainSource::Snapcompact, snapcompact_totals)]),
        time_series,
        project: effective_project.map(str::to_owned),
        projects,
    })
}

/// Distinct project folders for the given session files (chunked like the TS
/// SQLite variable limit).
pub(crate) fn read_projects_by_session(
    sessions: &[String],
) -> Result<HashMap<String, HashSet<String>>> {
    let mut unique: Vec<String> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for s in sessions {
        if !s.is_empty() && seen.insert(s) {
            unique.push(s.clone());
        }
    }
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    if unique.is_empty() {
        return Ok(out);
    }
    crate::db::with_db(|db| {
        for chunk in unique.chunks(500) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let sql = format!(
                "SELECT DISTINCT session_file, folder FROM messages WHERE session_file IN ({placeholders})"
            );
            let mut stmt = db.prepare(&sql)?;
            let mut rows = stmt.query(rusqlite::params_from_iter(chunk.iter()))?;
            while let Some(row) = rows.next()? {
                let session_file: String = row.get(0)?;
                let folder: String = row.get(1)?;
                if folder.is_empty() {
                    continue;
                }
                out.entry(session_file).or_default().insert(folder);
            }
        }
        Ok(())
    })?;
    Ok(out)
}
