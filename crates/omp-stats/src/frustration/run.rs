//! The paid frustration judge run (TS `frustration.ts`: `setStatsJudgeProvider`,
//! `estimateFrustrationRun`, `startFrustrationRun`, `cancelFrustrationRun`, `runJob`).
//!
//! `omp-stats` holds no LLM client. The host (omp-cli) registers a
//! [`StatsJudgeProvider`]; the run asks it once for a [`StatsJudge`] and judges
//! every unjudged prose text with it, on a private tokio runtime so the
//! synchronous HTTP handlers can start a run and return immediately.

use std::collections::BTreeMap;
use std::future::Future;
use std::hash::{BuildHasher, Hasher};
use std::pin::Pin;
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::watch;
use tokio::task::JoinSet;

use super::{Questions, FRUSTRATION_QUESTIONS};
use crate::db_query::{
    get_pending_frustration_prose, get_pending_frustration_totals, upsert_frustration_verdicts,
    FrustrationVerdict, PendingProse,
};
use crate::rollup::get_time_range_config;
use crate::types::{FrustrationEstimate, FrustrationJobState, FrustrationJobStatus};

/// Per-request input-token overhead of the frustration questions, measured on jev-1.13.
const REQUEST_OVERHEAD_TOKENS: i64 = 561;
/// Prose characters per input token, measured on jev-1.13.
const CHARS_PER_TOKEN: f64 = 5.9;
/// Output tokens a prompted chat judge bills per request; native judges bill none (zero output price).
const OUTPUT_TOKENS_PER_REQUEST: i64 = 8;
/// Judge requests in flight at the start of a run; [`AdaptiveLimit`] tunes it from there.
const INITIAL_CONCURRENCY: f64 = 32.0;
const MIN_CONCURRENCY: f64 = 4.0;
const MAX_CONCURRENCY: f64 = 256.0;
/// Wait before the second attempt at a text; doubles for the third.
const RETRY_DELAY: Duration = Duration::from_millis(250);
/// Failures within this window after a backoff don't halve the limit again.
const BACKOFF_COOLDOWN: Duration = Duration::from_millis(2_000);
/// Verdicts buffered before a write; a partial batch is written after [`VERDICT_FLUSH`].
const VERDICT_BATCH: usize = 64;
const VERDICT_FLUSH: Duration = Duration::from_millis(500);
const ATTEMPTS_PER_TEXT: u32 = 3;
/// Stop the run when this many texts failed before any succeeded: the judge is not working.
const CIRCUIT_BREAKER_FAILURES: i64 = 25;

const NO_PROVIDER_REASON: &str =
    "This dashboard was started without a judge (standalone omp-stats). Run `omp stats` to classify.";
const NO_MODEL_REASON: &str = "No judge model is available. Configure the `judge` model role.";
const ALREADY_RUNNING: &str = "A frustration judge run is already in progress.";

/// The judge's primary model; prices the pre-run estimate (USD per million tokens).
#[derive(Debug, Clone, PartialEq)]
pub struct JudgeModel {
    pub provider: String,
    pub id: String,
    pub cost_input: f64,
    pub cost_output: f64,
}

/// A judge's answers to [`FRUSTRATION_QUESTIONS`] for one text.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgeAnswer {
    /// Provider that answered (verdicts are labelled `provider/model`).
    pub provider: String,
    pub model: String,
    /// Probabilities of the `annoyed` score question, keyed by level index (`"0"`..`"3"`).
    pub annoyed_levels: BTreeMap<String, f64>,
    /// The most likely `target` choice.
    pub target: String,
    /// Billed USD for this request.
    pub cost: f64,
}

/// Boxed future returned by [`StatsJudge::judge`].
pub type JudgeFuture<'a> = Pin<Box<dyn Future<Output = Result<JudgeAnswer>> + Send + 'a>>;

/// Judge supplied by the omp host process (TS `StatsJudge`).
pub trait StatsJudge: Send + Sync {
    /// The model that answers first; `None` when no judge model is configured.
    fn primary_model(&self) -> Option<JudgeModel>;
    /// Answer `questions` about one prose text. Dropping the future cancels the request.
    fn judge<'a>(&'a self, text: &'a str, questions: &'a Questions) -> JudgeFuture<'a>;
}

/// Lazily resolves the host judge on first estimate/run so `omp stats` startup stays fast.
pub type StatsJudgeProvider = Arc<dyn Fn() -> Result<Arc<dyn StatsJudge>> + Send + Sync>;

struct Slot {
    provider: Option<StatsJudgeProvider>,
    judge: Option<Arc<dyn StatsJudge>>,
}

static SLOT: Mutex<Slot> = Mutex::new(Slot {
    provider: None,
    judge: None,
});

fn lock_slot() -> MutexGuard<'static, Slot> {
    SLOT.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Register (or replace) the host judge; `None` removes it.
pub fn set_stats_judge_provider(provider: Option<StatsJudgeProvider>) {
    let mut slot = lock_slot();
    slot.provider = provider;
    slot.judge = None;
}

/// Whether a judge provider is registered (`judgeAvailable` in the dashboard payload).
pub fn judge_available() -> bool {
    lock_slot().provider.is_some()
}

/// Resolve the registered judge once; a failed resolution is forgotten so the next call retries.
/// `Err` carries the TS refusal reason.
fn ready_judge() -> Result<(Arc<dyn StatsJudge>, JudgeModel), String> {
    let judge = {
        let mut slot = lock_slot();
        let Some(provider) = slot.provider.clone() else {
            return Err(NO_PROVIDER_REASON.to_owned());
        };
        match &slot.judge {
            Some(judge) => Arc::clone(judge),
            None => {
                let judge =
                    provider().map_err(|e| format!("The judge could not be loaded: {e}"))?;
                slot.judge = Some(Arc::clone(&judge));
                judge
            }
        }
    };
    let model = judge
        .primary_model()
        .ok_or_else(|| NO_MODEL_REASON.to_owned())?;
    Ok((judge, model))
}

struct State {
    /// Live status of the latest run.
    job: FrustrationJobStatus,
    /// Run counter; a worker only touches `job` while its id is current, so a
    /// cancelled run's stragglers can't write into a newer run.
    run: u64,
    cancel: Option<watch::Sender<bool>>,
    /// Set while a start request resolves the judge / loads the queue, so a concurrent start gets 409.
    starting: bool,
    /// A worker of the latest run is still alive.
    active: bool,
}

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

static STATE: Mutex<State> = Mutex::new(State {
    job: idle_job(),
    run: 0,
    cancel: None,
    starting: false,
    active: false,
});
/// Notified when a worker finishes.
static FINISHED: Condvar = Condvar::new();

fn lock_state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Snapshot of the current run's progress.
pub fn get_frustration_job_status() -> FrustrationJobStatus {
    lock_state().job.clone()
}

/// Run `f` on the job of run `run`; `None` when a newer run owns the status.
fn with_job<R>(run: u64, f: impl FnOnce(&mut FrustrationJobStatus) -> R) -> Option<R> {
    let mut state = lock_state();
    (state.run == run).then(|| f(&mut state.job))
}

/// Block until the latest run's workers have stopped (done, cancelled or failed).
/// Returns `false` on timeout.
pub fn wait_frustration_run_finished(timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    let mut state = lock_state();
    while state.active {
        let Some(left) = deadline.checked_duration_since(Instant::now()) else {
            return false;
        };
        state = FINISHED
            .wait_timeout(state, left)
            .unwrap_or_else(PoisonError::into_inner)
            .0;
    }
    true
}

/// Back to idle with no judge registered (tests share the process-wide state).
#[cfg(test)]
pub(crate) fn reset_for_tests() {
    set_stats_judge_provider(None);
    let mut state = lock_state();
    if let Some(cancel) = state.cancel.take() {
        cancel.send_replace(true);
    }
    state.run += 1;
    state.job = idle_job();
    state.starting = false;
    state.active = false;
}

/// Pre-run cost quote for judging every unjudged prose text in range.
///
/// # Errors
/// Db open or query failure.
pub fn estimate_frustration_run(range: Option<&str>) -> Result<FrustrationEstimate> {
    let model = match ready_judge() {
        Ok((_, model)) => model,
        Err(reason) => {
            return Ok(FrustrationEstimate::Unavailable {
                available: false,
                reason,
            })
        }
    };
    let cutoff = get_time_range_config(range).cutoff;
    let totals = get_pending_frustration_totals(cutoff)?;
    let messages = totals.messages;
    let chars = totals.chars;
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let input_tokens =
        messages * REQUEST_OVERHEAD_TOKENS + (chars as f64 / CHARS_PER_TOKEN).ceil() as i64;
    #[allow(clippy::cast_precision_loss)]
    let cost = (input_tokens as f64 * model.cost_input) / 1e6
        + ((messages * OUTPUT_TOKENS_PER_REQUEST) as f64 * model.cost_output) / 1e6;
    Ok(FrustrationEstimate::Available {
        available: true,
        messages,
        chars,
        input_tokens,
        cost,
        judge: format!("{}/{}", model.provider, model.id),
    })
}

/// Outcome of [`start_frustration_run`]: the new job, or the HTTP status and reason it was refused.
#[derive(Debug, Clone)]
pub enum StartFrustrationRunResult {
    Started { job: FrustrationJobStatus },
    Refused { status: u16, error: String },
}

fn refused(status: u16, error: impl Into<String>) -> StartFrustrationRunResult {
    StartFrustrationRunResult::Refused {
        status,
        error: error.into(),
    }
}

/// Clears `starting` on every exit of a start request.
struct Starting;

impl Drop for Starting {
    fn drop(&mut self) {
        lock_state().starting = false;
    }
}

/// The private runtime judge runs execute on; the HTTP handlers are synchronous.
static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("omp-stats-judge")
        .enable_all()
        .build()
        .expect("judge runtime")
});

/// Start the process-wide judge run over the unique unjudged prose in range.
/// Returns immediately after queueing; progress is read via [`get_frustration_job_status`].
///
/// # Errors
/// Db open or query failure.
pub fn start_frustration_run(range: Option<&str>) -> Result<StartFrustrationRunResult> {
    {
        let mut state = lock_state();
        if state.starting || state.job.state == FrustrationJobState::Running {
            return Ok(refused(409, ALREADY_RUNNING));
        }
        if !judge_available() {
            return Ok(refused(503, NO_PROVIDER_REASON));
        }
        state.starting = true;
    }
    let _starting = Starting;
    let (judge, model) = match ready_judge() {
        Ok(ready) => ready,
        Err(reason) => return Ok(refused(503, reason)),
    };
    let cutoff = get_time_range_config(range).cutoff;
    let pending = shuffle(get_pending_frustration_prose(cutoff)?);
    let now = now_ms();
    let running = !pending.is_empty();
    let job = FrustrationJobStatus {
        state: if running {
            FrustrationJobState::Running
        } else {
            FrustrationJobState::Done
        },
        total: i64::try_from(pending.len()).unwrap_or(i64::MAX),
        done: 0,
        failed: 0,
        cost: 0.0,
        judge: Some(format!("{}/{}", model.provider, model.id)),
        error: None,
        started_at: Some(now),
        finished_at: if running { None } else { Some(now) },
        concurrency: 0,
    };
    let mut state = lock_state();
    state.run += 1;
    let run = state.run;
    state.job = job.clone();
    if !running {
        state.cancel = None;
        state.active = false;
        return Ok(StartFrustrationRunResult::Started { job });
    }
    let (cancel_tx, cancel_rx) = watch::channel(false);
    state.cancel = Some(cancel_tx);
    state.active = true;
    drop(state);
    RUNTIME.spawn(run_job(run, judge, pending, cancel_rx));
    Ok(StartFrustrationRunResult::Started { job })
}

/// Abort the active run; in-flight judge requests are cancelled and no new text is taken.
pub fn cancel_frustration_run() -> FrustrationJobStatus {
    let mut state = lock_state();
    if state.job.state == FrustrationJobState::Running {
        if let Some(cancel) = &state.cancel {
            cancel.send_replace(true);
        }
        state.job.state = FrustrationJobState::Cancelled;
        state.job.finished_at = Some(now_ms());
    }
    state.job.clone()
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// Fisher–Yates: a random order makes every model's rates converge evenly while verdicts stream in.
fn shuffle<T>(mut items: Vec<T>) -> Vec<T> {
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(u64::try_from(now_ms()).unwrap_or(0));
    let mut seed = hasher.finish() | 1;
    for i in (1..items.len()).rev() {
        // xorshift64*
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        let r = seed.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let j = usize::try_from(r % (i as u64 + 1)).unwrap_or(0);
        items.swap(i, j);
    }
    items
}

/// In-flight limit for judge requests, adapted like TCP congestion control:
/// it grows by one per success until the first failure (slow start), then by
/// about four per window of successes, and halves on failure — at most once
/// per [`BACKOFF_COOLDOWN`], so one burst of rate-limit errors from the same
/// window counts once.
pub(super) struct AdaptiveLimit {
    pub(super) limit: f64,
    slow_start: bool,
    last_backoff: Option<Instant>,
}

impl AdaptiveLimit {
    pub(super) const fn new() -> Self {
        Self {
            limit: INITIAL_CONCURRENCY,
            slow_start: true,
            last_backoff: None,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(super) fn value(&self) -> usize {
        self.limit.floor() as usize
    }

    pub(super) fn succeeded(&mut self) {
        let step = if self.slow_start {
            1.0
        } else {
            4.0 / self.limit
        };
        self.limit = (self.limit + step).min(MAX_CONCURRENCY);
    }

    pub(super) fn failed(&mut self) {
        self.slow_start = false;
        let now = Instant::now();
        if self
            .last_backoff
            .is_some_and(|last| now.duration_since(last) < BACKOFF_COOLDOWN)
        {
            return;
        }
        self.last_backoff = Some(now);
        self.limit = (self.limit / 2.0).max(MIN_CONCURRENCY);
    }
}

/// What one text's attempts came to.
struct TextOutcome {
    hash: String,
    /// Attempts that failed before the text settled.
    failed_attempts: u32,
    last_error: Option<String>,
    judged: Option<Judged>,
    /// Every attempt failed (not merely interrupted by a cancel).
    exhausted: bool,
}

struct Judged {
    verdict: FrustrationVerdict,
    cost: f64,
    label: String,
}

async fn cancelled(cancel: &mut watch::Receiver<bool>) {
    // Err: the sender is gone, which only happens once the run was replaced.
    let _ = cancel.wait_for(|stop| *stop).await;
}

async fn judge_text(
    judge: Arc<dyn StatsJudge>,
    item: PendingProse,
    mut cancel: watch::Receiver<bool>,
) -> TextOutcome {
    let mut outcome = TextOutcome {
        hash: item.hash.clone(),
        failed_attempts: 0,
        last_error: None,
        judged: None,
        exhausted: false,
    };
    for attempt in 1..=ATTEMPTS_PER_TEXT {
        if *cancel.borrow() {
            return outcome;
        }
        // Failures are mostly rate limits: give the judge a moment before retrying.
        if attempt > 1 {
            tokio::select! {
                () = tokio::time::sleep(RETRY_DELAY * (attempt - 1)) => {}
                () = cancelled(&mut cancel) => return outcome,
            }
        }
        let result = tokio::select! {
            result = judge.judge(&item.prose, &FRUSTRATION_QUESTIONS) => result,
            () = cancelled(&mut cancel) => return outcome,
        };
        match result {
            Ok(answer) => {
                let p_angry = answer.annoyed_levels.get("3").copied().unwrap_or(0.0);
                let p_annoyed = answer.annoyed_levels.get("2").copied().unwrap_or(0.0) + p_angry;
                let label = format!("{}/{}", answer.provider, answer.model);
                outcome.judged = Some(Judged {
                    verdict: FrustrationVerdict {
                        prose_hash: item.hash.clone(),
                        p_annoyed,
                        p_angry,
                        target: answer.target,
                        judge: label.clone(),
                        judged_at: now_ms(),
                    },
                    cost: answer.cost,
                    label,
                });
                return outcome;
            }
            Err(error) => {
                if *cancel.borrow() {
                    return outcome;
                }
                outcome.failed_attempts += 1;
                let message = error.to_string();
                tracing::warn!(hash = %item.hash, attempt, error = %message, "frustration judge attempt failed");
                outcome.last_error = Some(message);
            }
        }
    }
    outcome.exhausted = true;
    outcome
}

async fn run_job(
    run: u64,
    judge: Arc<dyn StatsJudge>,
    queue: Vec<PendingProse>,
    mut cancel: watch::Receiver<bool>,
) {
    let mut limit = AdaptiveLimit::new();
    let mut tasks: JoinSet<TextOutcome> = JoinSet::new();
    let mut queue = queue.into_iter();
    let mut remaining = true;
    let mut label_from_result = false;
    let mut last_error: Option<String> = None;
    let mut tripped = false;
    let mut stopped = false;
    // Verdicts are written in batches: one transaction per flush instead of one per text.
    let mut buffered: Vec<FrustrationVerdict> = Vec::new();
    let mut flush_at: Option<Instant> = None;

    // A write failure (e.g. SQLite) is not a judge failure: it ends the run.
    let flush = |buffered: &mut Vec<FrustrationVerdict>,
                 flush_at: &mut Option<Instant>,
                 tripped: &mut bool,
                 last_error: &mut Option<String>|
     -> bool {
        *flush_at = None;
        let batch = std::mem::take(buffered);
        match upsert_frustration_verdicts(&batch) {
            Ok(()) => false,
            Err(error) => {
                if !*tripped {
                    *tripped = true;
                    *last_error = Some(error.to_string());
                    tracing::error!(error = %error, "frustration judge run crashed");
                }
                true
            }
        }
    };

    loop {
        if tripped || *cancel.borrow() {
            if !stopped {
                stopped = true;
                tasks.abort_all();
            }
        } else {
            while remaining && tasks.len() < limit.value() {
                let Some(item) = queue.next() else {
                    remaining = false;
                    break;
                };
                tasks.spawn(judge_text(Arc::clone(&judge), item, cancel.clone()));
            }
        }
        let concurrency = i64::try_from(limit.value()).unwrap_or(i64::MAX);
        with_job(run, |job| job.concurrency = concurrency);
        if tasks.is_empty() {
            break;
        }
        let wait = flush_at.map(|at| at.saturating_duration_since(Instant::now()));
        tokio::select! {
            joined = tasks.join_next() => {
                let Some(Ok(outcome)) = joined else { continue };
                for _ in 0..outcome.failed_attempts {
                    limit.failed();
                }
                if outcome.last_error.is_some() {
                    last_error.clone_from(&outcome.last_error);
                }
                if let Some(judged) = outcome.judged {
                    limit.succeeded();
                    buffered.push(judged.verdict);
                    with_job(run, |job| {
                        job.done += 1;
                        job.cost += judged.cost;
                        if !label_from_result {
                            label_from_result = true;
                            job.judge = Some(judged.label);
                        }
                    });
                    if buffered.len() >= VERDICT_BATCH {
                        flush(&mut buffered, &mut flush_at, &mut tripped, &mut last_error);
                    } else if flush_at.is_none() {
                        flush_at = Some(Instant::now() + VERDICT_FLUSH);
                    }
                } else if outcome.exhausted && !stopped {
                    let breaker = with_job(run, |job| {
                        job.failed += 1;
                        job.failed >= CIRCUIT_BREAKER_FAILURES && job.done == 0
                    });
                    if breaker == Some(true) && !tripped {
                        tripped = true;
                        tracing::error!(hash = %outcome.hash, error = ?last_error, "frustration judge run stopped: no text succeeded");
                    }
                }
            }
            () = async { tokio::time::sleep(wait.unwrap_or_default()).await }, if wait.is_some() => {
                flush(&mut buffered, &mut flush_at, &mut tripped, &mut last_error);
            }
            changed = cancel.changed(), if !stopped => {
                if changed.is_err() {
                    // The run was replaced; nobody can cancel it any more.
                    stopped = true;
                    tasks.abort_all();
                }
            }
        }
    }

    // Verdicts already paid for are kept, even when the run was cancelled.
    flush(&mut buffered, &mut flush_at, &mut tripped, &mut last_error);
    let error = last_error;
    let mut state = lock_state();
    if state.run == run {
        if tripped {
            state.job.state = FrustrationJobState::Failed;
            state.job.error = error;
        } else if state.job.state == FrustrationJobState::Running {
            state.job.state = FrustrationJobState::Done;
        }
        state.job.finished_at.get_or_insert_with(now_ms);
        state.cancel = None;
        state.active = false;
    }
    drop(state);
    FINISHED.notify_all();
}
