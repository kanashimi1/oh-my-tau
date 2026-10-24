//! Frustration tallies and the judge-verdict cache (port of the frustration
//! section of `db.ts`). Each message is classified once: by its cached
//! verdict when `v` joined, else by the regex signals stored at ingest.

use anyhow::Result;
use rusqlite::{params, params_from_iter, Row};
use serde::{Deserialize, Serialize};

use crate::db::with_db;
use crate::types::FrustrationCounts;

/// Frustration tallies of one responding (model, provider) pair, before identity merging.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrustrationModelRow {
    pub model: String,
    pub provider: Option<String>,
    /// Earliest message timestamp (ms) in range.
    pub first_seen: i64,
    #[serde(flatten)]
    pub counts: FrustrationCounts,
}

/// One unique unjudged prose text: identical messages share a hash and one verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingProse {
    pub hash: String,
    pub prose: String,
}

/// Cached judge verdict for one prose hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrustrationVerdict {
    pub prose_hash: String,
    /// P(clearly annoyed) + P(angry) on the `annoyed` score question.
    pub p_annoyed: f64,
    /// P(angry, hostile, or swearing).
    pub p_angry: f64,
    /// Most likely `target` choice: `assistant`, `other`, or `none`.
    pub target: String,
    /// `provider/model` that produced the verdict.
    pub judge: String,
    pub judged_at: i64,
}

/// Count and total characters of the pending prose set.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingProseTotals {
    pub messages: i64,
    pub chars: i64,
}

const JUDGED_SQL: &str = "v.prose_hash IS NOT NULL";
const JUDGED_ANNOYED_SQL: &str = "v.p_annoyed >= 0.5";
const JUDGED_AT_ASSISTANT_SQL: &str = "v.p_annoyed >= 0.5 AND v.target = 'assistant'";
const REGEX_AT_ASSISTANT_SQL: &str = "u.negation + u.repetition + u.blame > 0";

/// TS `FRUSTRATION_COUNTS_SQL`, built once per query.
fn frustration_counts_sql() -> String {
    format!(
        "COUNT(*) AS messages,
	COALESCE(SUM({JUDGED_SQL}), 0) AS judged,
	COALESCE(SUM(CASE WHEN {JUDGED_SQL} THEN {JUDGED_ANNOYED_SQL}
		ELSE u.yelling + u.profanity + u.anguish + u.negation + u.repetition + u.blame > 0 END), 0) AS annoyed,
	COALESCE(SUM(CASE WHEN {JUDGED_SQL} THEN {JUDGED_AT_ASSISTANT_SQL}
		ELSE {REGEX_AT_ASSISTANT_SQL} END), 0) AS at_assistant,
	COALESCE(SUM(CASE WHEN {JUDGED_SQL} THEN {JUDGED_AT_ASSISTANT_SQL} AND v.p_angry >= 0.5
		ELSE {REGEX_AT_ASSISTANT_SQL} AND (u.profanity > 0 OR u.yelling > 0) END), 0) AS angry"
    )
}

/// TS `hasRangeCutoff`: only a positive cutoff narrows the range.
fn range_cutoff(cutoff: Option<i64>) -> Option<i64> {
    cutoff.filter(|c| *c > 0)
}

fn counts_from_row(row: &Row<'_>) -> rusqlite::Result<FrustrationCounts> {
    Ok(FrustrationCounts {
        messages: row.get("messages")?,
        judged: row.get("judged")?,
        annoyed: row.get("annoyed")?,
        at_assistant: row.get("at_assistant")?,
        angry: row.get("angry")?,
    })
}

/// TS `getFrustrationOverall(cutoff)`: every user message with prose in range,
/// linked to a model or not.
///
/// # Errors
/// Db open or query failure.
pub fn get_frustration_overall(cutoff: Option<i64>) -> Result<FrustrationCounts> {
    with_db(|conn| {
        let cutoff = range_cutoff(cutoff);
        let sql = format!(
            "SELECT {counts}
		FROM user_messages u
		LEFT JOIN frustration_verdicts v ON v.prose_hash = u.prose_hash
		WHERE u.prose != ''{range}",
            counts = frustration_counts_sql(),
            range = if cutoff.is_some() {
                " AND u.timestamp >= ?"
            } else {
                ""
            },
        );
        let mut stmt = conn.prepare(&sql)?;
        let binds: Vec<i64> = cutoff.into_iter().collect();
        Ok(stmt.query_row(params_from_iter(binds.iter()), counts_from_row)?)
    })
}

/// TS `getFrustrationByModel(cutoff)`: tallies per responding (model, provider).
/// User messages that never got a reply (null model) are only in the overall tally.
///
/// # Errors
/// Db open or query failure.
pub fn get_frustration_by_model(cutoff: Option<i64>) -> Result<Vec<FrustrationModelRow>> {
    with_db(|conn| {
        let cutoff = range_cutoff(cutoff);
        let sql = format!(
            "SELECT u.model AS model, u.provider AS provider, MIN(u.timestamp) AS first_seen, {counts}
		FROM user_messages u
		LEFT JOIN frustration_verdicts v ON v.prose_hash = u.prose_hash
		WHERE u.prose != '' AND u.model IS NOT NULL{range}
		GROUP BY u.model, u.provider",
            counts = frustration_counts_sql(),
            range = if cutoff.is_some() { " AND u.timestamp >= ?" } else { "" },
        );
        let mut stmt = conn.prepare(&sql)?;
        let binds: Vec<i64> = cutoff.into_iter().collect();
        let rows = stmt.query_map(params_from_iter(binds.iter()), |row| {
            Ok(FrustrationModelRow {
                model: row.get("model")?,
                provider: row.get("provider")?,
                first_seen: row.get("first_seen")?,
                counts: counts_from_row(row)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    })
}

/// TS `pendingProseSql(hasCutoff)`: unique unjudged prose in range, one row per hash.
fn pending_prose_sql(has_cutoff: bool) -> String {
    format!(
        "SELECT u.prose_hash AS hash, MIN(u.prose) AS prose
		FROM user_messages u
		WHERE u.prose != ''{}
		  AND NOT EXISTS (SELECT 1 FROM frustration_verdicts v WHERE v.prose_hash = u.prose_hash)
		GROUP BY u.prose_hash",
        if has_cutoff {
            " AND u.timestamp >= ?"
        } else {
            ""
        }
    )
}

/// TS `getPendingFrustrationProse(cutoff)`: unique prose texts without a verdict.
///
/// # Errors
/// Db open or query failure.
pub fn get_pending_frustration_prose(cutoff: Option<i64>) -> Result<Vec<PendingProse>> {
    with_db(|conn| {
        let cutoff = range_cutoff(cutoff);
        let mut stmt = conn.prepare(&pending_prose_sql(cutoff.is_some()))?;
        let binds: Vec<i64> = cutoff.into_iter().collect();
        let rows = stmt.query_map(params_from_iter(binds.iter()), |row| {
            Ok(PendingProse {
                hash: row.get("hash")?,
                prose: row.get("prose")?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    })
}

/// TS `getPendingFrustrationTotals(cutoff)`: count and characters of the
/// pending set, without loading the text.
///
/// # Errors
/// Db open or query failure.
pub fn get_pending_frustration_totals(cutoff: Option<i64>) -> Result<PendingProseTotals> {
    with_db(|conn| {
        let cutoff = range_cutoff(cutoff);
        let sql = format!(
            "SELECT COUNT(*) AS messages, COALESCE(SUM(LENGTH(prose)), 0) AS chars FROM ({})",
            pending_prose_sql(cutoff.is_some())
        );
        let mut stmt = conn.prepare(&sql)?;
        let binds: Vec<i64> = cutoff.into_iter().collect();
        Ok(stmt.query_row(params_from_iter(binds.iter()), |row| {
            Ok(PendingProseTotals {
                messages: row.get("messages")?,
                chars: row.get("chars")?,
            })
        })?)
    })
}

/// TS `upsertFrustrationVerdicts`: cache or replace verdicts in one
/// transaction (a judge run lands dozens per second).
///
/// # Errors
/// Db open, transaction or statement failure.
pub fn upsert_frustration_verdicts(verdicts: &[FrustrationVerdict]) -> Result<()> {
    if verdicts.is_empty() {
        return Ok(());
    }
    with_db(|conn| {
        let tx = conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO frustration_verdicts (prose_hash, p_annoyed, p_angry, target, judge, judged_at)
		 VALUES (?, ?, ?, ?, ?, ?)",
            )?;
            for v in verdicts {
                stmt.execute(params![
                    v.prose_hash,
                    v.p_annoyed,
                    v.p_angry,
                    v.target,
                    v.judge,
                    v.judged_at
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    })
}
