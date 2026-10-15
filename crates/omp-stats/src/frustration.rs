//! Frustration dashboard: per-model share of user messages that sound annoyed,
//! annoyed at the assistant, and angry at the assistant.
//!
//! Each message is classified by its cached judge verdict when one exists, and
//! otherwise by the regex signals stored at ingest (`user_metrics`); the SQL is
//! in `db_query::frustration`.
//!
//! Port note: the TS judge run needs the omp host's LLM judge, which this port
//! does not have (standalone `omp-stats` never calls `setStatsJudgeProvider`).
//! The judge-dependent surface therefore returns the TS "no judge" shapes:
//! `judgeAvailable: false`, an unavailable estimate, and a 503 start refusal.
//! The job state machine is kept so the status payload has the TS shape.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

use anyhow::{bail, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::db_query::{get_frustration_by_model, get_frustration_overall, FrustrationModelRow};
use crate::rollup::get_time_range_config;
use crate::taxonomy::{classify_model, compare_revision, parse_revision, ClassifyOptions, ModelIdentity};
use crate::types::{
    FrustrationCounts, FrustrationDashboardStats, FrustrationEstimate, FrustrationJobState, FrustrationJobStatus,
    FrustrationModelStats,
};

/// The judge questions, one judge request per prose text (TS `prompts/judge/frustration.json`).
const QUESTION_FILE: &str = include_str!("../data/frustration.json");

const NO_PROVIDER_REASON: &str =
    "This dashboard was started without a judge (standalone omp-stats). Run `omp stats` to classify.";

/// One question as the judge sees it. Kind follows the shape of `criteria`,
/// the same rule as TS `judgeQuestions` (`ai/src/judgment/question-file.ts`).
#[derive(Debug, Clone, PartialEq)]
pub enum Question {
    /// `criteria` is a list of levels, lowest first.
    Score { instructions: String, criteria: Vec<String> },
    /// Yes/no rubric (`true`/`false` labels only); `None` when the file has no criteria.
    Noul { instructions: String, criteria: Option<BTreeMap<String, String>> },
    /// Any other labels; `null` labels are kept as-is.
    Choice { instructions: String, criteria: BTreeMap<String, Option<String>> },
}

/// Every question of a question file, keyed by id.
pub type Questions = BTreeMap<String, Question>;

#[derive(Deserialize)]
struct QuestionSpec {
    instructions: String,
    #[serde(default)]
    criteria: Option<Value>,
}

/// TS `judgeQuestions(file)`: parse a JSON question file into typed questions.
///
/// # Errors
/// Malformed JSON, a score question with fewer than two levels, or criteria
/// that are neither a list nor an object.
pub fn judge_questions(text: &str) -> Result<Questions> {
    let file: BTreeMap<String, QuestionSpec> = serde_json::from_str(text)?;
    file.into_iter()
        .map(|(id, spec)| {
            let question = to_question(&id, spec)?;
            Ok((id, question))
        })
        .collect()
}

fn to_question(id: &str, spec: QuestionSpec) -> Result<Question> {
    let QuestionSpec { instructions, criteria } = spec;
    let criteria = match criteria {
        None | Some(Value::Null) => return Ok(Question::Noul { instructions, criteria: None }),
        Some(criteria) => criteria,
    };
    match criteria {
        Value::Array(_) => {
            let levels: Vec<String> = serde_json::from_value(criteria)?;
            if levels.len() < 2 {
                bail!("score question \"{id}\" needs at least two levels");
            }
            Ok(Question::Score { instructions, criteria: levels })
        }
        Value::Object(_) => {
            let labels: BTreeMap<String, Option<String>> = serde_json::from_value(criteria)?;
            // `{}` and any label besides true/false are a choice; only a true/false rubric is yes/no.
            let is_rubric = !labels.is_empty() && labels.keys().all(|k| k == "true" || k == "false");
            if is_rubric {
                let rubric = labels.into_iter().filter_map(|(k, v)| v.map(|v| (k, v))).collect();
                Ok(Question::Noul { instructions, criteria: Some(rubric) })
            } else {
                Ok(Question::Choice { instructions, criteria: labels })
            }
        }
        _ => bail!("question \"{id}\": criteria must be a list or an object"),
    }
}

/// `FRUSTRATION_QUESTIONS`: parsed once from the compiled-in question file.
pub static FRUSTRATION_QUESTIONS: LazyLock<Questions> =
    LazyLock::new(|| judge_questions(QUESTION_FILE).expect("bundled frustration.json is valid"));

/// Idle status: the value before any run and after a fresh process starts.
const fn idle_job() -> FrustrationJobStatus {
    FrustrationJobStatus {
        state: FrustrationJobState::Idle,
        total: 0,
        done: 0,
        failed: 0,
        cost: 0.0,
        judge: None,
        error: None,
        started_at: None,
        finished_at: None,
        concurrency: 0,
    }
}

/// Live status of the latest run (process-wide, like the TS module global).
static CURRENT_JOB: Mutex<FrustrationJobStatus> = Mutex::new(idle_job());

fn lock_job() -> MutexGuard<'static, FrustrationJobStatus> {
    CURRENT_JOB.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Snapshot of the current run's progress.
pub fn get_frustration_job_status() -> FrustrationJobStatus {
    lock_job().clone()
}

/// Merge per-(model, provider) rows into catalog identities (`class/family/revision`)
/// so provider and spelling variants share a row. Classified rows sort by class,
/// then revision ascending, then family; unclassified rows follow by first use.
pub fn merge_frustration_rows(rows: &[FrustrationModelRow]) -> Vec<FrustrationModelStats> {
    let mut merged: Vec<FrustrationModelStats> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for row in rows {
        let identity = identify(row);
        let classified = identity.class != "unknown" && identity.revision.is_some();
        let family = identity.family.clone();
        let key = match (&identity.revision, classified) {
            (Some(revision), true) => format!("{}/{}/{revision}", identity.class, family.as_deref().unwrap_or("")),
            _ => row.model.clone(),
        };
        if let Some(&slot) = index.get(&key) {
            let existing = &mut merged[slot];
            add_counts(&mut existing.counts, &row.counts);
            existing.first_seen = existing.first_seen.min(row.first_seen);
            if !existing.models.contains(&row.model) {
                existing.models.push(row.model.clone());
            }
            continue;
        }
        let revision = if classified { identity.revision.clone() } else { None };
        let label = match revision.as_deref().map(short_revision) {
            Some(short) => [family.as_deref(), Some(short)]
                .into_iter()
                .flatten()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
            None => row.model.clone(),
        };
        index.insert(key.clone(), merged.len());
        merged.push(FrustrationModelStats {
            counts: row.counts.clone(),
            key,
            label,
            model_class: identity.class,
            family,
            revision,
            models: vec![row.model.clone()],
            first_seen: row.first_seen,
        });
    }
    merged.sort_by(compare_model_stats);
    merged
}

/// Lenient classification never fails in practice; an error counts as unclassified.
fn identify(row: &FrustrationModelRow) -> ModelIdentity {
    let opts = ClassifyOptions { lenient: true, ..ClassifyOptions::default() };
    classify_model(row.provider.as_deref().unwrap_or(""), &row.model, Some(opts)).unwrap_or_else(|_| ModelIdentity {
        class: "unknown".to_owned(),
        ..ModelIdentity::default()
    })
}

/// `4.5.0` → `4.5`, `5.0.0` → `5` (TS `replace(/(?:\.0)+$/, "")`).
fn short_revision(revision: &str) -> &str {
    let mut out = revision;
    while let Some(stripped) = out.strip_suffix(".0") {
        out = stripped;
    }
    out
}

fn add_counts(into: &mut FrustrationCounts, from: &FrustrationCounts) {
    into.messages += from.messages;
    into.judged += from.judged;
    into.annoyed += from.annoyed;
    into.at_assistant += from.at_assistant;
    into.angry += from.angry;
}

/// Classified rows first (class, revision, family); unclassified after, by first use.
fn compare_model_stats(a: &FrustrationModelStats, b: &FrustrationModelStats) -> Ordering {
    let a_rev = a.revision.as_deref().and_then(parse_revision);
    let b_rev = b.revision.as_deref().and_then(parse_revision);
    let (Some(a_rev), Some(b_rev)) = (a_rev, b_rev) else {
        return match (a_rev.is_some(), b_rev.is_some()) {
            (true, _) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => a.first_seen.cmp(&b.first_seen),
        };
    };
    a.model_class
        .cmp(&b.model_class)
        .then_with(|| compare_revision(a_rev, b_rev).cmp(&0))
        .then_with(|| a.family.as_deref().unwrap_or("").cmp(b.family.as_deref().unwrap_or("")))
}

/// Payload of `GET /api/stats/frustration`.
///
/// # Errors
/// Db open or query failure.
pub fn get_frustration_dashboard_stats(range: Option<&str>) -> Result<FrustrationDashboardStats> {
    let cutoff = get_time_range_config(range).cutoff;
    Ok(FrustrationDashboardStats {
        overall: get_frustration_overall(cutoff)?,
        by_model: merge_frustration_rows(&get_frustration_by_model(cutoff)?),
        judge_available: false,
        job: get_frustration_job_status(),
    })
}

/// Pre-run cost quote. Without a judge provider the TS answer is unavailable
/// before any db read, so this never touches the db.
pub fn estimate_frustration_run(_range: Option<&str>) -> FrustrationEstimate {
    FrustrationEstimate::Unavailable { available: false, reason: NO_PROVIDER_REASON.to_owned() }
}

/// Outcome of [`start_frustration_run`]: the new job, or the HTTP status and reason it was refused.
#[derive(Debug, Clone)]
pub enum StartFrustrationRunResult {
    /// Unreachable in this port: a run needs a judge provider.
    Started { job: FrustrationJobStatus },
    Refused { status: u16, error: String },
}

/// Start the judge run. Same order as TS: a run in progress gives 409, then a
/// missing judge gives 503. This port has no judge, so it always refuses with 503.
pub fn start_frustration_run(_range: Option<&str>) -> StartFrustrationRunResult {
    if lock_job().state == FrustrationJobState::Running {
        return StartFrustrationRunResult::Refused {
            status: 409,
            error: "A frustration judge run is already in progress.".to_owned(),
        };
    }
    StartFrustrationRunResult::Refused { status: 503, error: NO_PROVIDER_REASON.to_owned() }
}

/// Abort the active run; a run that is not running is left untouched.
pub fn cancel_frustration_run() -> FrustrationJobStatus {
    let mut job = lock_job();
    if job.state == FrustrationJobState::Running {
        job.state = FrustrationJobState::Cancelled;
        job.finished_at = Some(now_ms());
    }
    job.clone()
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

#[cfg(test)]
mod tests;
