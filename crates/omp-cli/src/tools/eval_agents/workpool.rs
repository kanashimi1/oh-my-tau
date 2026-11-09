//! `workpool()` op and the `wait`/`status`/`cancel` handle ops.

use super::judge::throw_if_aborted;
use super::{
    agent_id_taken, apply_child_policy, cancel_handle, json, launch_agent, lookup_handle,
    parse_handle_ref, settle_agent, Arc, Duration, Handle, HandleStatus, HashMap, Instant,
    Launcher, LazyLock, Map, Mutex, OpCtx, Ordering, PathBuf, Value, VecDeque, AGENT_HANDLES,
    EVAL_SUBAGENT_OUTPUT, LAUNCHER, subagent_concurrency, WAIT_TICK,
};

/// TS `WorkPoolItem.status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ItemStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl ItemStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

struct PoolItem {
    id: String,
    text: String,
    status: ItemStatus,
}

/// A live worker (fresh mode: dies with its single turn).
struct PoolWorker {
    id: String,
    /// Batch id of the running turn (TS `agent.jobId`).
    current: String,
}

struct PoolBatch {
    id: String,
    agent_id: String,
    items: Vec<usize>,
    status: ItemStatus,
    output: Option<String>,
}

/// One worker turn the pool wants launched.
#[derive(Debug, PartialEq, Eq)]
struct TurnStart {
    /// Index into `PoolState::batches`.
    batch: usize,
    worker_id: String,
    prompt: String,
}

/// Pure pool bookkeeping (TS `WorkPool` minus the job/session plumbing).
struct PoolState {
    name: String,
    agent: String,
    limit: usize,
    closed: bool,
    items: Vec<PoolItem>,
    workers: Vec<PoolWorker>,
    batches: Vec<PoolBatch>,
    fresh_queue: VecDeque<usize>,
    next_seq: u64,
    next_worker: u64,
}

impl PoolState {
    fn new(name: String, agent: String, limit: usize) -> Self {
        Self {
            name,
            agent,
            limit,
            closed: false,
            items: Vec::new(),
            workers: Vec::new(),
            batches: Vec::new(),
            fresh_queue: VecDeque::new(),
            next_seq: 1,
            next_worker: 1,
        }
    }

    /// TS `WorkPool.push`: queue items (ids `<pool>#<seq>`) and dispatch them.
    /// `taken` reports agent ids already in use (worker ids must be unique
    /// `agent://` handles).
    fn push(
        &mut self,
        texts: &[String],
        taken: &dyn Fn(&str) -> bool,
    ) -> Result<(Vec<String>, Vec<TurnStart>), String> {
        if self.closed {
            return Err(format!("workpool {} is closed", self.name));
        }
        let first = self.items.len();
        let mut ids = Vec::with_capacity(texts.len());
        for text in texts {
            let id = format!("{}#{}", self.name, self.next_seq);
            self.next_seq += 1;
            ids.push(id.clone());
            self.items.push(PoolItem {
                id,
                text: text.clone(),
                status: ItemStatus::Queued,
            });
        }
        let mut starts = Vec::new();
        for index in first..self.items.len() {
            self.dispatch(index, &mut starts, taken);
        }
        Ok((ids, starts))
    }

    /// TS `#dispatch` (fresh mode): spawn a worker while under the limit,
    /// otherwise wait in the FIFO queue.
    fn dispatch(
        &mut self,
        index: usize,
        starts: &mut Vec<TurnStart>,
        taken: &dyn Fn(&str) -> bool,
    ) {
        if self.closed || self.items[index].status != ItemStatus::Queued {
            return;
        }
        if self.workers.len() < self.limit {
            self.spawn_worker(index, starts, taken);
        } else {
            self.fresh_queue.push_back(index);
        }
    }

    /// TS `#spawn` + `#drain`: a new worker takes the item as its first batch.
    fn spawn_worker(
        &mut self,
        index: usize,
        starts: &mut Vec<TurnStart>,
        taken: &dyn Fn(&str) -> bool,
    ) {
        let base = format!("{}-{}", self.name, self.next_worker);
        self.next_worker += 1;
        let mut worker_id = base.clone();
        let mut suffix = 2;
        while taken(&worker_id) || self.workers.iter().any(|w| w.id == worker_id) {
            worker_id = format!("{base}-{suffix}");
            suffix += 1;
        }
        let batch_id = format!("{worker_id}-b1");
        self.items[index].status = ItemStatus::Running;
        self.workers.push(PoolWorker {
            id: worker_id.clone(),
            current: batch_id.clone(),
        });
        self.batches.push(PoolBatch {
            id: batch_id,
            agent_id: worker_id.clone(),
            items: vec![index],
            status: ItemStatus::Running,
            output: None,
        });
        let batch = self.batches.len() - 1;
        starts.push(TurnStart {
            batch,
            worker_id,
            prompt: self.batch_prompt(batch),
        });
    }

    /// TS `workpool-batch.md`, minus the keyed incremental `yield` contract
    /// (omp-rs children answer once, with their final text).
    // invariant: write!/writeln! to a String is infallible.
    #[allow(clippy::expect_used)]
    fn batch_prompt(&self, batch: usize) -> String {
        use std::fmt::Write as _;
        let batch = &self.batches[batch];
        let mut prompt = format!(
            "<workpool pool=\"{}\" batch=\"{}\">\nYou are a worker in pool `{}`. Complete every item below in order. Your final answer is the outcome: for each item, its number and a self-contained outcome/evidence value.\n",
            self.name, batch.id, self.name
        );
        for (n, &index) in batch.items.iter().enumerate() {
            writeln!(prompt, "## Item {}\n{}", n + 1, self.items[index].text)
                .expect("writing to a String is infallible");
        }
        prompt.push_str("</workpool>");
        prompt
    }

    /// TS `#finishTurn` (fresh mode): settle the batch and its items, retire
    /// the worker, and hand the next queued item a new worker.
    fn finish_turn(
        &mut self,
        batch: usize,
        status: ItemStatus,
        output: String,
        taken: &dyn Fn(&str) -> bool,
    ) -> Vec<TurnStart> {
        let Some(entry) = self.batches.get_mut(batch) else {
            return Vec::new();
        };
        if entry.status != ItemStatus::Running {
            return Vec::new();
        }
        entry.status = status;
        entry.output = Some(output);
        let agent_id = entry.agent_id.clone();
        for &index in &entry.items {
            self.items[index].status = status;
        }
        self.workers.retain(|w| w.id != agent_id);
        let mut starts = Vec::new();
        while let Some(next) = self.fresh_queue.pop_front() {
            if !self.closed && self.items[next].status == ItemStatus::Queued {
                self.dispatch(next, &mut starts, taken);
                break;
            }
        }
        starts
    }

    /// TS `WorkPool.status()`.
    fn status(&self) -> Value {
        let mut counts = Map::new();
        for status in [
            ItemStatus::Queued,
            ItemStatus::Running,
            ItemStatus::Completed,
            ItemStatus::Failed,
            ItemStatus::Cancelled,
        ] {
            let n = self
                .items
                .iter()
                .filter(|item| item.status == status)
                .count();
            counts.insert(status.as_str().to_string(), json!(n));
        }
        json!({
            "name": self.name,
            "agent": self.agent,
            "limit": self.limit,
            "closed": self.closed,
            "freshAgents": true,
            "agents": self.workers.iter().map(|w| json!({
                "id": w.id,
                "state": "running",
                "queued": 0,
                "turns": 0,
                "current": w.current,
            })).collect::<Vec<_>>(),
            "items": counts,
            "batches": self.batches.len(),
        })
    }

    /// TS `WorkPool.peek()`: batch results without consuming anything.
    fn peek(&self) -> Value {
        let batches: Vec<Value> = self
            .batches
            .iter()
            .map(|batch| {
                let mut entry = json!({
                    "id": batch.id,
                    "agent": batch.agent_id,
                    "items": batch.items.iter().map(|&i| self.items[i].id.as_str()).collect::<Vec<_>>(),
                    "status": batch.status.as_str(),
                });
                if let Some(output) = &batch.output {
                    entry["output"] = Value::String(output.clone());
                }
                entry
            })
            .collect();
        json!({"batches": batches, "pending": self.pending()})
    }

    fn pending(&self) -> usize {
        self.items
            .iter()
            .filter(|item| matches!(item.status, ItemStatus::Queued | ItemStatus::Running))
            .count()
    }

    /// TS `WorkPool.close()`: stop accepting work and cancel queued items;
    /// running turns finish.
    fn close(&mut self) -> Value {
        self.closed = true;
        let mut dropped = Vec::new();
        for item in &mut self.items {
            if item.status == ItemStatus::Queued {
                item.status = ItemStatus::Cancelled;
                dropped.push(item.id.clone());
            }
        }
        self.fresh_queue.clear();
        json!({"dropped": dropped})
    }
}

/// A registered pool: bookkeeping plus what launching its turns needs.
struct WorkPool {
    state: Mutex<PoolState>,
    context: Option<String>,
    cwd: PathBuf,
    launcher: Arc<dyn Launcher>,
}

/// Process-local pools by name (TS `WorkPoolRegistry`, main-agent owner).
static WORKPOOLS: LazyLock<Mutex<HashMap<String, Arc<WorkPool>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Launch the given worker turns; each one, once settled, settles its batch
/// and launches whatever the pool dispatches next.
fn start_turns(pool: &Arc<WorkPool>, starts: Vec<TurnStart>) {
    if starts.is_empty() {
        return;
    }
    let agent = pool.state.lock().agent.clone();
    for start in starts {
        let prompt = match &pool.context {
            Some(context) => format!("{context}\n\n{}", start.prompt),
            None => start.prompt,
        };
        let mut spec = omp_task::TaskSpec {
            id: Some(start.worker_id.clone()),
            agent: agent.clone(),
            prompt,
            cwd: pool.cwd.clone(),
            output_schema: None,
            schema_mode: omp_task::SchemaMode::Permissive,
            model: None,
            tools: None,
            depth: 0,
            extra_args: Vec::new(),
            env: Vec::new(),
            worktree: false,
        };
        apply_child_policy(&mut spec);
        super::super::task::apply_agent_definition(&mut spec);
        // Same channel as `task` children: a workpool worker can
        // `write agent://<sibling>` / `agent://parent` / `agent://all`.
        let lease = super::super::a2a::BusLease::acquire().ok();
        if let Some(lease) = &lease {
            spec.env.push((
                super::super::a2a::SOCK_ENV.to_string(),
                lease.sock().to_string_lossy().into_owned(),
            ));
            spec.env.push((
                super::super::a2a::SELF_ENV.to_string(),
                start.worker_id.clone(),
            ));
        }
        let handle = Handle::new("agent", start.worker_id.clone());
        // Reserve the worker id atomically: a concurrent registration of the
        // same id (another pool, an agent() call) must not have its handle
        // overwritten and orphaned from status/cancel.
        let worker_id = start.worker_id.clone();
        let inserted = {
            let mut registry = AGENT_HANDLES.lock();
            if registry.contains_key(&worker_id) {
                false
            } else {
                registry.insert(worker_id.clone(), handle.clone());
                true
            }
        };
        if !inserted {
            handle.settle(
                HandleStatus::Failed,
                None,
                None,
                Some(format!("worker id {worker_id} is already registered; refusing to overwrite its agent handle")),
            );
            let (status, text) = turn_outcome(&handle);
            let next = pool
                .state
                .lock()
                .finish_turn(start.batch, status, text, &agent_id_taken);
            start_turns(pool, next);
            continue;
        }
        let run = launch_agent(&handle, spec, &*pool.launcher);
        let pool = pool.clone();
        let batch = start.batch;
        tokio::spawn(async move {
            let output = run.await;
            drop(lease);
            settle_agent(&handle, &output, omp_task::SchemaMode::Permissive, false);
            handle.mark_run_finished();
            let (status, text) = turn_outcome(&handle);
            let next = pool
                .state
                .lock()
                .finish_turn(batch, status, text, &agent_id_taken);
            start_turns(&pool, next);
        });
    }
}

/// Batch status + output from a settled worker handle (TS `#finishTurn`
/// status rule; output falls back to the error text).
fn turn_outcome(handle: &Handle) -> (ItemStatus, String) {
    let inner = handle.inner.lock();
    let status = match inner.status {
        HandleStatus::Completed => ItemStatus::Completed,
        HandleStatus::Cancelled => ItemStatus::Cancelled,
        HandleStatus::Failed | HandleStatus::Running => ItemStatus::Failed,
    };
    let output = inner
        .text
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .or(inner.error.as_deref())
        .unwrap_or("(no output)")
        .to_string();
    (status, output)
}

fn workpool_optional_string(
    record: &Map<String, Value>,
    key: &str,
) -> Result<Option<String>, String> {
    match record.get(key) {
        None => Ok(None),
        Some(Value::String(value)) if !value.trim().is_empty() => {
            Ok(Some(value.trim().to_string()))
        }
        Some(_) => Err(format!("workpool {key} must be a non-empty string")),
    }
}

pub(super) fn run_workpool(args: &Value, ctx: &OpCtx) -> Result<Value, String> {
    run_workpool_with(args, &ctx.cwd, &LAUNCHER)
}

fn run_workpool_with(
    args: &Value,
    cwd: &std::path::Path,
    launcher: &Arc<dyn Launcher>,
) -> Result<Value, String> {
    let record = args
        .as_object()
        .ok_or_else(|| "workpool() arguments must be an object".to_string())?;
    let op = record
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| "workpool() requires an op".to_string())?;
    if op == "create" {
        return create_workpool(record, cwd, launcher);
    }
    let name = record
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "workpool operation requires a non-empty name".to_string())?;
    let pool = WORKPOOLS
        .lock()
        .get(name)
        .cloned()
        .ok_or_else(|| format!("unknown workpool \"{name}\""))?;
    match op {
        "push" => {
            let texts: Vec<String> = record
                .get("items")
                .and_then(Value::as_array)
                .and_then(|items| {
                    items
                        .iter()
                        .map(|item| item.as_str().map(str::to_string))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| "workpool push requires an items string array".to_string())?;
            let (ids, starts) = pool.state.lock().push(&texts, &agent_id_taken)?;
            start_turns(&pool, starts);
            Ok(json!({"ids": ids}))
        }
        "status" => Ok(pool.state.lock().status()),
        "peek" => Ok(pool.state.lock().peek()),
        "close" => Ok(pool.state.lock().close()),
        _ => Err(format!("unknown workpool operation \"{op}\"")),
    }
}

/// TS `runEvalWorkpool` `create`: `{name, agent, limit}`.
fn create_workpool(
    record: &Map<String, Value>,
    cwd: &std::path::Path,
    launcher: &Arc<dyn Launcher>,
) -> Result<Value, String> {
    let agent = workpool_optional_string(record, "agent")?;
    let requested = workpool_optional_string(record, "name")?;
    let context = workpool_optional_string(record, "context")?;
    match record.get("tools") {
        None => {}
        Some(Value::Array(tools))
            if tools
                .iter()
                .all(|tool| tool.as_str().is_some_and(|name| !name.is_empty())) =>
        {
            if !tools.is_empty() {
                return Err("Eval-defined tools are unavailable in this runtime.".to_string());
            }
        }
        Some(_) => return Err("workpool tools must be an array of non-empty strings".to_string()),
    }
    let agent = agent.unwrap_or_else(|| "task".to_string());
    if !omp_task::params::is_valid_agent_name(&agent) {
        return Err(format!(
            "workpool() received invalid arguments: `agent` must be a valid agent name, got {agent:?}"
        ));
    }
    // Every pushed item spawns a fresh worker subagent: same recursion gate
    // as agent() (TS `canSpawnAtDepth`).
    omp_task::recursion_gate()
        .map_err(|error| format!("workpool() cannot spawn workers: {error}"))?;
    let mut pools = WORKPOOLS.lock();
    let name = if let Some(name) = requested {
        if pools.contains_key(&name) {
            return Err(format!("workpool \"{name}\" already exists"));
        }
        name
    } else {
        let base = format!("{agent}-pool");
        let mut name = base.clone();
        let mut suffix = 2;
        while pools.contains_key(&name) || agent_id_taken(&name) {
            name = format!("{base}-{suffix}");
            suffix += 1;
        }
        name
    };
    let pool = WorkPool {
        state: Mutex::new(PoolState::new(
            name.clone(),
            agent.clone(),
            subagent_concurrency(),
        )),
        context,
        cwd: cwd.to_path_buf(),
        launcher: launcher.clone(),
    };
    pools.insert(name.clone(), Arc::new(pool));
    Ok(json!({"name": name, "agent": agent, "limit": subagent_concurrency()}))
}

// ---------------------------------------------------------------------------
// `__wait__` / `__status__` / `__cancel__` (TS `handle-bridge.ts`)
// ---------------------------------------------------------------------------

pub(super) async fn run_wait(args: &Value, ctx: &OpCtx) -> Result<Value, String> {
    let items = args
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| "wait() requires an items array".to_string())?;
    let refs: Vec<(&'static str, String)> = items
        .iter()
        .map(parse_handle_ref)
        .collect::<Result<_, _>>()?;
    let timeout_ms = args.get("timeoutMs").and_then(Value::as_f64);
    if let Some(ms) = timeout_ms {
        if !ms.is_finite() || ms < 0.0 {
            return Err("wait() timeoutMs must be a non-negative finite number".to_string());
        }
    }
    let handles: Vec<Arc<Handle>> = refs
        .iter()
        .map(|(kind, id)| lookup_handle(kind, id))
        .collect::<Result<_, _>>()?;
    let deadline = timeout_ms.map(|ms| Instant::now() + Duration::from_secs_f64(ms / 1000.0));

    loop {
        // TS `runEvalWait`: an aborted cell cancels every handle it waited on
        // and then awaits their settlement (`Promise.allSettled`) before
        // throwing, so an aborted caller never returns while the underlying
        // runs are still tearing down.
        if let Err(error) = throw_if_aborted(&ctx.signal) {
            for handle in &handles {
                cancel_handle(handle);
            }
            for handle in &handles {
                handle.wait_run_finished().await;
            }
            return Err(error);
        }
        let all_settled = handles.iter().all(|handle| handle.is_settled());
        if all_settled {
            break;
        }
        if let Some(deadline) = deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            tokio::time::sleep(WAIT_TICK.min(remaining)).await;
        } else {
            tokio::time::sleep(WAIT_TICK).await;
        }
    }
    Ok(json!({
        "items": handles.iter().map(|handle| handle.snapshot()).collect::<Vec<_>>()
    }))
}

pub(super) fn run_status(args: &Value) -> Result<Value, String> {
    let item = args
        .get("item")
        .ok_or_else(|| "handle request requires an item".to_string())?;
    let (kind, id) = parse_handle_ref(item)?;
    Ok(lookup_handle(kind, &id)?.snapshot())
}

pub(super) fn run_cancel(args: &Value) -> Result<Value, String> {
    let item = args
        .get("item")
        .ok_or_else(|| "handle request requires an item".to_string())?;
    let (kind, id) = parse_handle_ref(item)?;
    let handle = lookup_handle(kind, &id)?;
    Ok(json!({"cancelled": cancel_handle(&handle)}))
}

// ---------------------------------------------------------------------------
// `__budget__` (TS `budget-bridge.ts` `runEvalBudget`)
// ---------------------------------------------------------------------------

/// TS `runEvalBudget` fallback path when neither a per-turn directive nor Goal
/// Mode exists: no ceiling (`total: null`), advisory (`hard: false`), `spent`
/// = output tokens. omp-rs has no `+Nk` turn budget / Goal Mode ledger, and
/// the main session's usage is not reachable from the tool layer, so `spent`
/// counts what TS adds via `recordEvalSubagentUsage`: output tokens reported
/// by settled eval subagents (`agent()` + workpool workers), cumulative for
/// the process rather than per turn.
pub fn budget_snapshot() -> Value {
    json!({
        "total": Value::Null,
        "spent": EVAL_SUBAGENT_OUTPUT.load(Ordering::Relaxed),
        "hard": false,
    })
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod tests {
    use super::super::test_support::*;
    use super::super::{deliver_message, RunCtl, RunFuture, AGENT_CANCELLED_TEXT};
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    fn no_ids(_: &str) -> bool {
        false
    }

    #[test]
    fn pool_dispatches_up_to_limit_then_queues_fifo() {
        let mut pool = PoolState::new("p".to_string(), "task".to_string(), 2);
        let texts: Vec<String> = ["a", "b", "c"].iter().map(|s| (*s).to_string()).collect();
        let (ids, starts) = pool.push(&texts, &no_ids).unwrap();
        assert_eq!(ids, ["p#1", "p#2", "p#3"]);
        let workers: Vec<&str> = starts.iter().map(|s| s.worker_id.as_str()).collect();
        assert_eq!(workers, ["p-1", "p-2"]);
        assert!(starts[0].prompt.contains("batch=\"p-1-b1\""));
        assert!(starts[0].prompt.contains("## Item 1\na\n"));
        let status = pool.status();
        assert_eq!(status["items"]["running"], 2);
        assert_eq!(status["items"]["queued"], 1);
        assert_eq!(status["agents"].as_array().unwrap().len(), 2);
        assert_eq!(status["agents"][0]["current"], "p-1-b1");
        assert_eq!(status["limit"], 2);
        assert_eq!(status["freshAgents"], true);

        // A finished turn retires its worker and hands the queued item a new one.
        let next = pool.finish_turn(
            starts[0].batch,
            ItemStatus::Completed,
            "done a".to_string(),
            &no_ids,
        );
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].worker_id, "p-3");
        assert!(next[0].prompt.contains("## Item 1\nc\n"));
        let peek = pool.peek();
        assert_eq!(peek["pending"], 2);
        assert_eq!(peek["batches"][0]["status"], "completed");
        assert_eq!(peek["batches"][0]["output"], "done a");
        assert_eq!(peek["batches"][0]["items"], json!(["p#1"]));
        assert!(peek["batches"][1].get("output").is_none());

        // Settling is idempotent.
        assert!(pool
            .finish_turn(
                starts[0].batch,
                ItemStatus::Failed,
                "x".to_string(),
                &no_ids
            )
            .is_empty());
        assert_eq!(pool.peek()["batches"][0]["status"], "completed");

        pool.finish_turn(
            starts[1].batch,
            ItemStatus::Failed,
            "boom".to_string(),
            &no_ids,
        );
        pool.finish_turn(
            next[0].batch,
            ItemStatus::Cancelled,
            "stop".to_string(),
            &no_ids,
        );
        let status = pool.status();
        assert_eq!(status["items"]["completed"], 1);
        assert_eq!(status["items"]["failed"], 1);
        assert_eq!(status["items"]["cancelled"], 1);
        assert_eq!(status["batches"], 3);
        assert_eq!(pool.peek()["pending"], 0);
        assert!(status["agents"].as_array().unwrap().is_empty());
    }

    #[test]
    fn pool_close_drops_queued_and_rejects_pushes() {
        let mut pool = PoolState::new("q".to_string(), "task".to_string(), 1);
        let texts: Vec<String> = ["a", "b", "c"].iter().map(|s| (*s).to_string()).collect();
        let (_, starts) = pool.push(&texts, &no_ids).unwrap();
        assert_eq!(starts.len(), 1);
        assert_eq!(pool.close(), json!({"dropped": ["q#2", "q#3"]}));
        assert_eq!(
            pool.push(&["d".to_string()], &no_ids).unwrap_err(),
            "workpool q is closed"
        );
        // The running turn still finishes; nothing new is dispatched.
        let next = pool.finish_turn(
            starts[0].batch,
            ItemStatus::Completed,
            "ok".to_string(),
            &no_ids,
        );
        assert!(next.is_empty());
        let status = pool.status();
        assert_eq!(status["closed"], true);
        assert_eq!(status["items"]["cancelled"], 2);
        assert_eq!(status["items"]["completed"], 1);
        assert_eq!(pool.close(), json!({"dropped": []}));
    }

    #[test]
    fn pool_worker_ids_avoid_taken_agent_ids() {
        let mut pool = PoolState::new("r".to_string(), "task".to_string(), 4);
        let taken = |id: &str| id == "r-1";
        let (_, starts) = pool
            .push(&["a".to_string(), "b".to_string()], &taken)
            .unwrap();
        let workers: Vec<&str> = starts.iter().map(|s| s.worker_id.as_str()).collect();
        assert_eq!(workers, ["r-1-2", "r-2"]);
        assert!(pool.push(&[], &taken).unwrap().0.is_empty());
    }

    #[tokio::test]
    async fn workpool_ops_drive_worker_runs() {
        // Reads the recursion-depth env through `create_workpool`: serialize
        // with the depth-env tests.
        let _depth_env = DEPTH_ENV_LOCK.lock().await;
        let launcher = Arc::new(MockLauncher::default());
        let dyn_launcher: Arc<dyn Launcher> = launcher.clone();
        let cwd = PathBuf::from(".");
        let created = run_workpool_with(
            &json!({"op": "create", "name": "test-pool-e2e", "context": "Shared ctx"}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap();
        assert_eq!(
            created,
            json!({"name": "test-pool-e2e", "agent": "task", "limit": subagent_concurrency()})
        );
        let dup = run_workpool_with(
            &json!({"op": "create", "name": "test-pool-e2e"}),
            &cwd,
            &dyn_launcher,
        );
        assert_eq!(
            dup.unwrap_err(),
            "workpool \"test-pool-e2e\" already exists"
        );

        let pushed = run_workpool_with(
            &json!({"op": "push", "name": "test-pool-e2e", "items": ["one", "two"]}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap();
        assert_eq!(
            pushed,
            json!({"ids": ["test-pool-e2e#1", "test-pool-e2e#2"]})
        );
        assert_eq!(launcher.launched(), 2);
        {
            let runs = launcher.runs.lock();
            assert!(runs[0].0.prompt.starts_with("Shared ctx\n\n<workpool"));
            assert!(runs[0].0.prompt.contains("one"));
            assert_eq!(runs[1].0.id.as_deref(), Some("test-pool-e2e-2"));
        }
        // Workers are agent handles: messageable and cancellable.
        assert_eq!(
            deliver_message("test-pool-e2e-1", "hint").await.unwrap(),
            "Delivered to test-pool-e2e-1."
        );
        launcher
            .run(0)
            .finish(omp_task::RunStatus::Completed, "first done");
        let cancel =
            run_cancel(&json!({"item": {"kind": "agent", "id": "test-pool-e2e-2"}})).unwrap();
        assert_eq!(cancel, json!({"cancelled": true}));

        let peek_args = json!({"op": "peek", "name": "test-pool-e2e"});
        let mut peek = Value::Null;
        for _ in 0..200 {
            peek = run_workpool_with(&peek_args, &cwd, &dyn_launcher).unwrap();
            if peek["pending"] == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(peek["pending"], 0, "{peek}");
        assert_eq!(peek["batches"][0]["status"], "completed");
        assert_eq!(peek["batches"][0]["output"], "first done");
        assert_eq!(peek["batches"][1]["status"], "cancelled");
        assert_eq!(peek["batches"][1]["output"], AGENT_CANCELLED_TEXT);

        let closed = run_workpool_with(
            &json!({"op": "close", "name": "test-pool-e2e"}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap();
        assert_eq!(closed, json!({"dropped": []}));
        let err = run_workpool_with(
            &json!({"op": "push", "name": "test-pool-e2e", "items": ["x"]}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap_err();
        assert_eq!(err, "workpool test-pool-e2e is closed");
        assert_eq!(
            run_workpool_with(
                &json!({"op": "status", "name": "nope-pool"}),
                &cwd,
                &dyn_launcher
            )
            .unwrap_err(),
            "unknown workpool \"nope-pool\""
        );
        assert_eq!(
            run_workpool_with(
                &json!({"op": "create", "tools": ["bash"]}),
                &cwd,
                &dyn_launcher
            )
            .unwrap_err(),
            "Eval-defined tools are unavailable in this runtime."
        );
    }

    #[test]
    fn budget_is_the_ts_fallback_shape() {
        let snap = budget_snapshot();
        assert!(snap["total"].is_null());
        assert_eq!(snap["hard"], false);
        assert!(snap["spent"].is_u64());
    }

    /// Finding 5: registering a worker whose id is already taken (the race
    /// between the pool's taken-check and the handle insert, or any direct
    /// `start_turns` dispatch) must not overwrite the existing handle — the
    /// turn fails explicitly and the original handle keeps owning its run.
    #[tokio::test]
    async fn pool_worker_id_collision_fails_without_overwriting() {
        let _depth_env = DEPTH_ENV_LOCK.lock().await;
        let launcher = Arc::new(MockLauncher::default());
        let dyn_launcher: Arc<dyn Launcher> = launcher.clone();
        let cwd = PathBuf::from(".");
        run_workpool_with(
            &json!({"op": "create", "name": "test-pool-collision"}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap();
        let pool = WORKPOOLS
            .lock()
            .get("test-pool-collision")
            .cloned()
            .unwrap();

        // Occupy the id a turn will try to register.
        let existing = Handle::new("agent", "occupied-worker".to_string());
        AGENT_HANDLES
            .lock()
            .insert("occupied-worker".to_string(), existing.clone());

        start_turns(
            &pool,
            vec![TurnStart {
                batch: usize::MAX,
                worker_id: "occupied-worker".to_string(),
                prompt: "do work".to_string(),
            }],
        );
        assert_eq!(launcher.launched(), 0, "no run launches on a collision");

        // The existing handle still owns the id, untouched by the failed turn.
        let registered = AGENT_HANDLES.lock().get("occupied-worker").cloned();
        assert!(Arc::ptr_eq(&registered.unwrap(), &existing));
        assert_eq!(existing.snapshot()["status"], "running");
        AGENT_HANDLES.lock().remove("occupied-worker");
        WORKPOOLS.lock().remove("test-pool-collision");
    }

    /// A mock run whose cancel does *not* settle the run: the run resolves
    /// only when `release()` sends its output (like a real child that still
    /// has to be reaped after the kill).
    struct SlowMockRun {
        inner: MockRun,
        release: Mutex<Option<tokio::sync::oneshot::Sender<omp_task::RunOutput>>>,
    }

    impl RunCtl for SlowMockRun {
        fn steer<'a>(
            &'a self,
            message: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<(), omp_task::SteerError>> + Send + 'a>> {
            self.inner.steer(message)
        }

        fn cancel(&self) -> bool {
            // The kill is requested but the run stays unresolved until
            // `release()` (a real child still has to be reaped).
            if self.inner.cancelled.swap(true, Ordering::SeqCst) {
                return false;
            }
            true
        }

        fn output_tokens(&self) -> u64 {
            0
        }
    }

    struct SlowMockLauncher {
        runs: Mutex<Vec<Arc<SlowMockRun>>>,
    }

    impl Launcher for SlowMockLauncher {
        fn launch(&self, spec: omp_task::TaskSpec) -> (Arc<dyn RunCtl>, RunFuture) {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let id = spec.id.unwrap_or_default();
            let run = Arc::new(SlowMockRun {
                inner: MockRun {
                    id: id.clone(),
                    steered: Mutex::new(Vec::new()),
                    cancelled: AtomicBool::new(false),
                    done: Mutex::new(None),
                },
                release: Mutex::new(Some(tx)),
            });
            self.runs.lock().push(run.clone());
            let future: RunFuture = Box::pin(async move {
                rx.await
                    .unwrap_or_else(|_| run_output(&id, omp_task::RunStatus::Cancelled, ""))
            });
            (run, future)
        }
    }

    /// Finding 6: an aborted `wait()` cancels every handle and waits for them
    /// to settle before erroring (TS `handle-bridge.ts` `Promise.allSettled`).
    #[tokio::test]
    async fn aborted_wait_cancels_then_waits_for_settlement() {
        let _depth_env = DEPTH_ENV_LOCK.lock().await;
        let launcher = Arc::new(SlowMockLauncher {
            runs: Mutex::new(Vec::new()),
        });
        let dyn_launcher: Arc<dyn Launcher> = launcher.clone();
        let cwd = PathBuf::from(".");
        run_workpool_with(
            &json!({"op": "create", "name": "test-pool-abort"}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap();
        run_workpool_with(
            &json!({"op": "push", "name": "test-pool-abort", "items": ["one", "two"]}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap();
        assert_eq!(launcher.runs.lock().len(), 2);

        let controller = pa_agent::abort::AbortController::new();
        let signal = controller.signal();
        controller.abort();
        let ctx = OpCtx {
            cwd: cwd.clone(),
            signal,
            on_update: None,
            tool_call_id: "test-abort".to_string(),
        };
        let wait_args = json!({"items": [
            {"kind": "agent", "id": "test-pool-abort-1"},
            {"kind": "agent", "id": "test-pool-abort-2"},
        ]});
        let wait = run_wait(&wait_args, &ctx);
        tokio::pin!(wait);
        tokio::select! {
            _ = &mut wait => panic!("wait() must not return before the cancelled runs finish"),
            () = tokio::time::sleep(Duration::from_millis(200)) => {}
        }
        // The runs were asked to cancel but are not finished yet; the handles
        // already report cancelled (TS `manager.cancel` flips job status at once).
        assert!(launcher
            .runs
            .lock()
            .iter()
            .all(|run| run.inner.cancelled.load(Ordering::SeqCst)));
        let handle = lookup_handle("agent", "test-pool-abort-1").unwrap();
        assert_eq!(handle.snapshot()["status"], "cancelled");
        assert!(!handle
            .run_finished
            .load(std::sync::atomic::Ordering::SeqCst));

        for run in launcher.runs.lock().iter() {
            let tx = run.release.lock().take().unwrap();
            let _ = tx.send(run_output(
                &run.inner.id,
                omp_task::RunStatus::Cancelled,
                "",
            ));
        }
        let err = tokio::time::timeout(Duration::from_secs(5), &mut wait)
            .await
            .expect("wait() returns once the cancelled runs finish")
            .unwrap_err();
        assert_eq!(err, "aborted");
        for id in ["test-pool-abort-1", "test-pool-abort-2"] {
            let handle = lookup_handle("agent", id).unwrap();
            assert!(
                handle
                    .run_finished
                    .load(std::sync::atomic::Ordering::SeqCst),
                "{id} run finished before the abort error"
            );
        }
        WORKPOOLS.lock().remove("test-pool-abort");
        for id in ["test-pool-abort-1", "test-pool-abort-2"] {
            AGENT_HANDLES.lock().remove(id);
        }
    }

    #[tokio::test]
    async fn workpool_refuses_to_spawn_at_max_depth() {
        let _env = DEPTH_ENV_LOCK.lock().await;
        // Simulate a depth-2 session (the TS default ceiling): spawns from
        // here are refused with the TS message.
        std::env::set_var(omp_task::TASK_DEPTH_ENV, "2");
        let launcher = Arc::new(MockLauncher::default());
        let dyn_launcher: Arc<dyn Launcher> = launcher.clone();
        let cwd = PathBuf::from(".");
        let err = run_workpool_with(
            &json!({"op": "create", "name": "test-pool-depth"}),
            &cwd,
            &dyn_launcher,
        )
        .unwrap_err();
        assert_eq!(
            err,
            "workpool() cannot spawn workers: Cannot spawn another agent at task depth 2; maximum depth is 2."
        );
        assert_eq!(launcher.launched(), 0);
        std::env::remove_var(omp_task::TASK_DEPTH_ENV);
        WORKPOOLS.lock().remove("test-pool-depth");
    }
}
