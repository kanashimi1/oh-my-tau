//! `judge_batch()` op: batch parsing, fan-out workers and per-item retries.

use super::judge::{judgment_cache, judgment_result_value, parse_questions, parse_state};
use super::{
    json, mint_id, AbortSignal, AgentDir, Arc, AtomicU64, CancellationToken, Duration, Instant,
    JudgeRequest, Map, Mutex, Notify, OpCtx, Ordering, Value, EVAL_HANDLE_CONCURRENCY,
    JUDGE_BATCHES, REQUEST_SLOTS,
};

/// TS `DEFAULT_RETRIES` (`judgment-batch-bridge.ts`).
const DEFAULT_BATCH_RETRIES: u32 = 1;

/// TS `DEFAULT_INTENT`.
const DEFAULT_BATCH_INTENT: &str = "Judging";

/// TS `DRAIN_SETTLE_WINDOW_MS`: once a waiting drain sees its first item it
/// holds this long for more so near-simultaneous settles return together.
const DRAIN_SETTLE_WINDOW: Duration = Duration::from_millis(50);

struct BatchInput {
    key: Value,
    state: Value,
}

/// One settled item as the cell receives it from `drain` (TS
/// `JudgmentBatchItem`).
#[derive(Clone)]
struct BatchItem {
    key: Value,
    answers: Option<Map<String, Value>>,
    error: Option<String>,
    model: Option<String>,
}

#[derive(Default)]
struct BatchInner {
    settled: Vec<BatchItem>,
    /// Number of settled items handed out by `drain` so far (TS drain cursor).
    cursor: usize,
    failed: usize,
    cost: f64,
    last_model: Option<String>,
    running: bool,
    error: Option<String>,
}

pub(super) struct JudgmentBatchRun {
    id: String,
    intent: String,
    total: usize,
    min_ok: usize,
    inner: Mutex<BatchInner>,
    notify: Notify,
    cancel: CancellationToken,
    started: Instant,
}

impl JudgmentBatchRun {
    /// TS `JudgmentBatchStatus`.
    fn status(&self) -> Value {
        let inner = self.inner.lock();
        let mut status = json!({
            "id": self.id,
            "intent": self.intent,
            "total": self.total,
            "done": inner.settled.len(),
            "failed": inner.failed,
            "cost": inner.cost,
            "running": inner.running,
            "elapsedS": self.started.elapsed().as_secs(),
        });
        if let Some(model) = &inner.last_model {
            status["model"] = Value::String(model.clone());
        }
        if let Some(error) = &inner.error {
            status["error"] = Value::String(error.clone());
        }
        status
    }

    /// TS `JudgmentBatch.drain`: items settled since the previous drain; waits
    /// up to `timeout` for at least one, then holds a short settle window.
    async fn drain(&self, timeout: Option<Duration>) -> Vec<BatchItem> {
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        loop {
            {
                let inner = self.inner.lock();
                if inner.cursor < inner.settled.len() || !inner.running {
                    break;
                }
            }
            let wait = match deadline {
                Some(deadline) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return Vec::new();
                    }
                    remaining
                }
                None => Duration::from_hours(1),
            };
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            // Re-check after arming so a settle that raced the arm is seen.
            {
                let inner = self.inner.lock();
                if inner.cursor < inner.settled.len() || !inner.running {
                    break;
                }
            }
            let _ = tokio::time::timeout(wait, notified).await;
        }
        // First item is in: hold the settle window once for near-simultaneous
        // settles, then hand out everything past the cursor.
        let have = {
            let inner = self.inner.lock();
            inner.cursor < inner.settled.len()
        };
        if have {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let _ = tokio::time::timeout(DRAIN_SETTLE_WINDOW, notified).await;
        }
        let mut inner = self.inner.lock();
        let items: Vec<BatchItem> = inner.settled[inner.cursor..].to_vec();
        inner.cursor = inner.settled.len();
        items
    }

    /// Raise the wholesale run error once the cursor is exhausted (TS
    /// `drain()` rejects with `#error` after delivering every item).
    fn drain_error(&self) -> Option<String> {
        let inner = self.inner.lock();
        if inner.cursor >= inner.settled.len() && !inner.running {
            inner.error.clone()
        } else {
            None
        }
    }
}

fn batch_invalid(detail: &str) -> String {
    format!("judge_batch() received invalid arguments: {detail}")
}

fn parse_batch_items(value: Option<&Value>) -> Result<Vec<BatchInput>, String> {
    let Some(items) = value.and_then(Value::as_array) else {
        return Err(batch_invalid("items must be an array of { key, state }"));
    };
    if items.is_empty() {
        return Err(batch_invalid("items must not be empty"));
    }
    let mut seen: Vec<String> = Vec::new();
    let mut parsed = Vec::new();
    for (index, entry) in items.iter().enumerate() {
        let Some(entry) = entry.as_object() else {
            return Err(batch_invalid(&format!(
                "item {index} needs a string or number key"
            )));
        };
        let key = entry.get("key").cloned().unwrap_or(Value::Null);
        let valid_key = matches!(&key, Value::String(s) if !s.is_empty()) || key.is_number();
        if !valid_key {
            return Err(batch_invalid(&format!(
                "item {index} needs a string or number key"
            )));
        }
        let key_id = serde_json::to_string(&key).unwrap_or_default();
        if seen.contains(&key_id) {
            return Err(batch_invalid(&format!("duplicate item key {key_id}")));
        }
        seen.push(key_id);
        parsed.push(BatchInput {
            key,
            // TS reuses `judgment-bridge.ts` `parseState`/`parseQuestions`
            // verbatim, so their `judge()`-prefixed messages surface as-is.
            state: parse_state(entry.get("state").unwrap_or(&Value::Null))?,
        });
    }
    Ok(parsed)
}

fn parse_batch_count(value: Option<&Value>, name: &str, fallback: u64) -> Result<u64, String> {
    match value {
        None => Ok(fallback),
        Some(Value::Number(number)) => {
            let count = number
                .as_u64()
                .ok_or_else(|| batch_invalid(&format!("{name} must be a non-negative integer")))?;
            Ok(count.min(if name == "concurrency" {
                EVAL_HANDLE_CONCURRENCY as u64
            } else {
                u64::MAX
            }))
        }
        Some(_) => Err(batch_invalid(&format!(
            "{name} must be a non-negative integer"
        ))),
    }
}

#[allow(clippy::too_many_lines)] // judge-batch op: parse, fan out workers, collect, shape the reply
pub(super) async fn run_judge_batch_op(args: &Value, ctx: &OpCtx) -> Result<Value, String> {
    let Some(record) = args.as_object() else {
        return Err(batch_invalid("expected { op, … }"));
    };
    let op = record
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| batch_invalid("expected { op, … }"))?;

    match op {
        "create" => {
            let items = parse_batch_items(record.get("items"))?;
            let questions = parse_questions(record.get("questions"))?;
            let total = items.len();
            // TS `parseBatchOptions`: an explicit 0 falls back to the default
            // ceiling (`concurrency > 0 ? concurrency : EVAL_HANDLE_CONCURRENCY`).
            let concurrency = parse_batch_count(
                record.get("concurrency"),
                "concurrency",
                EVAL_HANDLE_CONCURRENCY as u64,
            )?;
            let concurrency = if concurrency == 0 {
                EVAL_HANDLE_CONCURRENCY
            } else {
                usize::try_from(concurrency).unwrap_or(usize::MAX)
            };
            let retries = u32::try_from(parse_batch_count(
                record.get("retries"),
                "retries",
                u64::from(DEFAULT_BATCH_RETRIES),
            )?)
            .unwrap_or(u32::MAX);
            let min_ok = record
                .get("minOk")
                .map(|_| parse_batch_count(record.get("minOk"), "minOk", 1))
                .transpose()?
                .map_or(1, |min_ok| usize::try_from(min_ok).unwrap_or(usize::MAX))
                .min(total);
            let intent = match record.get("intent") {
                None => DEFAULT_BATCH_INTENT.to_string(),
                Some(Value::String(intent)) if !intent.trim().is_empty() => intent.clone(),
                Some(_) => return Err(batch_invalid("intent must be a non-empty string")),
            };
            let batch = Arc::new(JudgmentBatchRun {
                id: mint_id("jb"),
                intent,
                total,
                min_ok,
                inner: Mutex::new(BatchInner {
                    running: true,
                    ..BatchInner::default()
                }),
                notify: Notify::new(),
                cancel: CancellationToken::new(),
                started: Instant::now(),
            });
            JUDGE_BATCHES.lock().insert(batch.id.clone(), batch.clone());
            start_batch(&batch, items, questions, concurrency, retries, &ctx.signal);
            // `create` answers with the status snapshot (TS returns
            // `batch.status()`; the prelude only requires id/total/intent).
            Ok(batch.status())
        }
        "attach" | "status" => {
            let batch = require_batch(record)?;
            Ok(batch.status())
        }
        "drain" => {
            let batch = require_batch(record)?;
            let timeout_ms = record.get("timeoutMs");
            if let Some(value) = timeout_ms {
                let Some(ms) = value.as_f64() else {
                    return Err(batch_invalid(
                        "timeoutMs must be a non-negative finite number",
                    ));
                };
                if !ms.is_finite() || ms < 0.0 {
                    return Err(batch_invalid(
                        "timeoutMs must be a non-negative finite number",
                    ));
                }
            }
            let timeout = timeout_ms
                .and_then(Value::as_f64)
                .map(|ms| Duration::from_secs_f64(ms / 1000.0));
            let items = batch.drain(timeout).await;
            if let Some(error) = batch.drain_error() {
                return Err(error);
            }
            Ok(json!({
                "items": items.iter().map(|item| {
                    let mut value = json!({"key": item.key});
                    if let Some(answers) = &item.answers {
                        value["answers"] = Value::Object(answers.clone());
                    }
                    if let Some(error) = &item.error {
                        value["error"] = Value::String(error.clone());
                    }
                    if let Some(model) = &item.model {
                        value["model"] = Value::String(model.clone());
                    }
                    value
                }).collect::<Vec<_>>()
            }))
        }
        "results" => {
            let batch = require_batch(record)?;
            let inner = batch.inner.lock();
            let mut results = Map::new();
            for item in &inner.settled {
                if let Some(answers) = &item.answers {
                    let key = serde_json::to_string(&item.key)
                        .ok()
                        .map(|key| key.trim_matches('"').to_string())
                        .unwrap_or_default();
                    results.insert(key, Value::Object(answers.clone()));
                }
            }
            Ok(json!({"results": Value::Object(results)}))
        }
        "failed" => {
            let batch = require_batch(record)?;
            let inner = batch.inner.lock();
            let mut failed = Map::new();
            for item in &inner.settled {
                if let Some(error) = &item.error {
                    let key = serde_json::to_string(&item.key)
                        .ok()
                        .map(|key| key.trim_matches('"').to_string())
                        .unwrap_or_default();
                    failed.insert(key, Value::String(error.clone()));
                }
            }
            Ok(json!({"failed": Value::Object(failed)}))
        }
        "cancel" => {
            let batch = require_batch(record)?;
            let was_running = batch.inner.lock().running;
            batch.cancel.cancel();
            Ok(json!({"cancelled": was_running}))
        }
        "close" => {
            let id = record
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| batch_invalid("op requires a batch id"))?;
            let batch = JUDGE_BATCHES
                .lock()
                .get(id)
                .cloned()
                .ok_or_else(|| format!("unknown judge_batch \"{id}\""))?;
            batch.cancel.cancel();
            JUDGE_BATCHES.lock().remove(id);
            Ok(json!({"closed": true}))
        }
        other => Err(batch_invalid(&format!("unknown op \"{other}\""))),
    }
}

fn require_batch(record: &Map<String, Value>) -> Result<Arc<JudgmentBatchRun>, String> {
    let id = record
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| batch_invalid("op requires a batch id"))?;
    JUDGE_BATCHES
        .lock()
        .get(id)
        .cloned()
        .ok_or_else(|| format!("unknown judge_batch \"{id}\""))
}

/// TS `JudgmentBatch.start()`: `min(concurrency, total)` workers pull items
/// and judge each with up to `retries` retries; per-item failures land in
/// `settled` with `error`, only a run that dies wholesale sets the batch error.
fn start_batch(
    batch: &Arc<JudgmentBatchRun>,
    items: Vec<BatchInput>,
    questions: Map<String, Value>,
    concurrency: usize,
    retries: u32,
    cell_signal: &AbortSignal,
) {
    let items = Arc::new(items);
    let questions = Arc::new(questions);
    let worker_count = concurrency.clamp(1, items.len());
    let next = Arc::new(AtomicU64::new(0));
    let finished = Arc::new(AtomicU64::new(0));
    for _ in 0..worker_count {
        let batch = batch.clone();
        let items = items.clone();
        let questions = questions.clone();
        let next = next.clone();
        let finished = finished.clone();
        let cell_signal = cell_signal.clone();
        tokio::spawn(async move {
            loop {
                if batch.cancel.is_cancelled() || cell_signal.is_aborted() {
                    break;
                }
                let index =
                    usize::try_from(next.fetch_add(1, Ordering::Relaxed)).unwrap_or(usize::MAX);
                let Some(input) = items.get(index) else {
                    break;
                };
                let settled = judge_one_batch_item(
                    &batch,
                    input,
                    &questions,
                    retries,
                    &AgentDir::default(),
                    &cell_signal,
                )
                .await;
                {
                    let mut inner = batch.inner.lock();
                    if let Some(model) = &settled.model {
                        inner.last_model = Some(model.clone());
                    }
                    if settled.error.is_some() {
                        inner.failed += 1;
                    }
                    inner.settled.push(settled);
                }
                batch.notify.notify_waiters();
            }
            // Last worker out closes the run and applies the minOk gate (TS
            // `#finish`): enough items judged ok, else the run dies wholesale.
            if finished.fetch_add(1, Ordering::Relaxed) + 1 == worker_count as u64 {
                let mut inner = batch.inner.lock();
                inner.running = false;
                let ok = inner.settled.len() - inner.failed;
                if ok < batch.min_ok {
                    let last = inner
                        .settled
                        .iter()
                        .rev()
                        .find_map(|item| item.error.as_deref())
                        .filter(|error| *error != "cancelled")
                        .unwrap_or("unknown error");
                    inner.error = Some(format!(
                        "judge_batch: only {ok}/{} item(s) judged (min_ok={}); last error: {last}",
                        batch.total, batch.min_ok
                    ));
                }
                drop(inner);
                batch.notify.notify_waiters();
            }
        });
    }
}

/// Judge one item with the resolved chain, up to `retries` extra attempts
/// (TS `#judgeItem`: `for (let attempt = 0; attempt <= retries; attempt++)`).
async fn judge_one_batch_item(
    batch: &JudgmentBatchRun,
    input: &BatchInput,
    questions: &Map<String, Value>,
    retries: u32,
    dir: &AgentDir,
    cell_signal: &AbortSignal,
) -> BatchItem {
    if batch.cancel.is_cancelled() || cell_signal.is_aborted() {
        return BatchItem {
            key: input.key.clone(),
            answers: None,
            error: Some("cancelled".to_string()),
            model: None,
        };
    }
    let request = JudgeRequest {
        state: input.state.clone(),
        questions: questions.clone(),
    };
    let chain = match omp_providers::resolve_judge_chain(dir) {
        Ok(chain) => chain,
        Err(error) => {
            return BatchItem {
                key: input.key.clone(),
                answers: None,
                error: Some(error.to_string()),
                model: None,
            };
        }
    };
    let cache = judgment_cache();
    let mut last_error: Option<String> = None;
    for attempt in 0..=retries {
        if batch.cancel.is_cancelled() || cell_signal.is_aborted() {
            return BatchItem {
                key: input.key.clone(),
                answers: None,
                error: Some("cancelled".to_string()),
                model: None,
            };
        }
        let Ok(_permit) = REQUEST_SLOTS.acquire().await else {
            last_error = Some("judge_batch: request aborted".to_string());
            break;
        };
        match omp_providers::judge(&chain, &request, cache.as_ref()).await {
            Ok(result) => {
                let value = judgment_result_value(&result);
                let answers = value
                    .get("answers")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                batch.inner.lock().cost += result.usage.cost;
                return BatchItem {
                    key: input.key.clone(),
                    answers: Some(answers),
                    error: None,
                    model: Some(format!("{}/{}", result.provider, result.model)),
                };
            }
            Err(error) => {
                last_error = Some(error.to_string());
                if attempt < retries {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }
    }
    BatchItem {
        key: input.key.clone(),
        answers: None,
        error: Some(last_error.unwrap_or_else(|| "judge_batch: judgment failed".to_string())),
        model: None,
    }
}

// ---------------------------------------------------------------------------
// `__workpool__` (TS `workpool-bridge.ts` + `task/workpool.ts`)
// ---------------------------------------------------------------------------
//
// TS pools keep subagents alive and feed them batches of queued items as
// follow-up turns; omp-rs subagents exit when their prompt settles, so pools
// run in the TS `eval.workpool.freshAgents` mode: every pushed item becomes
// one fresh worker (`<pool>-<n>`, an `agent://` handle that `status`/`cancel`/
// `write agent://` reach) running a one-item batch, at most `limit` at once,
// the rest queued FIFO. The aggregate pool job TS registers with the async
// job manager (final drained-pool delivery to the main agent) has no omp-rs
// counterpart; `status()`/`peek()` expose the same bookkeeping.
