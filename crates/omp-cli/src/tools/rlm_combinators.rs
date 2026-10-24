//! λ-RLM planner + SERA rubric host ops for eval cells.
//!
//! **λ-RLM** (arXiv 2603.20105): the eval preludes (`prelude_combinators.{py,js}`)
//! expose the paper's eight typed combinators — `split`, `peek`, `map`, `filter`,
//! `reduce`, `concat`, `cross` and the neural leaf `M` (= `llm_query`) — plus the
//! fixed-point executor `lam_rlm`. The model decides leaves only; control flow is
//! built by the deterministic planner here ([`plan`]): given `|P|`, the leaf
//! window `K` and the remaining call budget it fixes the fan-out `k*`, the leaf
//! threshold `τ*` and the depth `d`, or refuses before a single model call is
//! made. Termination is by construction: `k ≥ 2` and every split strictly
//! shrinks the chunk (`ceil(len/k) < len` for `len > τ ≥ 1`), so the recursion
//! rank is `d = #{shrinks until ≤ τ}`; leaves ≤ `k^d`. Leaf calls go through
//! `__llm_query__`, i.e. share the `OMP_RLM_MAX_*` ledger with plain `llm_query`.
//!
//! **SERA** (arXiv 2610.04902, rubric half; math in [`omp_task::rubric`]):
//! `__rubric_freeze__` commits a rubric before the child runs and returns its id;
//! `__rubric_score__` scores the child's output against *that* frozen rubric
//! (scorer call is charged to the same ledger), appends a row to
//! `OMP_RUBRIC_LOG` (JSONL) and returns the record; `__rubric_rank__` evaluates
//! the paper's ranking loss against verified outcomes. The `task` tool uses the
//! same pieces through [`SessionScorer`].

use std::collections::HashMap;
use std::future::Future;
use std::sync::LazyLock;

use parking_lot::Mutex;

use omp_task::rubric::{self, FrozenRubric, RubricRecord, RubricScorer};
use pa_agent::abort::AbortSignal;
use pa_agent::BoxFut;
use pa_types::ai::{Model, Usage};
use serde_json::{json, Value};

use super::eval_bridge::OpCtx;
use super::rlm::{Backend, Limits, Session, DEFAULT_MODEL, SESSION};

/// Env: hard cap on the characters a combinator program may chew (default 32 Mi).
pub const MAX_CHARS_ENV: &str = "OMP_LAM_MAX_CHARS";
/// Env: model that applies rubrics (default: the sub-call default model).
pub const RUBRIC_MODEL_ENV: &str = "OMP_RUBRIC_MODEL";
const DEFAULT_MAX_CHARS: u64 = 32 * 1024 * 1024;
/// Upper bound for the planner's fan-out.
pub const MAX_FANOUT: u64 = 64;
/// Hard recursion bound the executor asserts (planner depth is far below it).
pub const MAX_DEPTH: u64 = 32;

/// Planner input.
#[derive(Debug, Clone, Copy)]
pub struct PlanReq {
    /// `|P|` in characters.
    pub n: u64,
    /// Leaf window `K` in characters.
    pub window: u64,
    /// Leaf threshold `τ*` (clamped to `window`); default `window`.
    pub tau: Option<u64>,
    /// Explicit fan-out; default is the cost-minimal `ceil(n/τ)` (≤ [`MAX_FANOUT`]).
    pub fanout: Option<u64>,
    /// Model calls outside the leaves (task detection, synthesis).
    pub extra_calls: u64,
    /// Calls left in the shared ledger.
    pub remaining_calls: u64,
    pub max_chars: u64,
}

/// A concrete, bounded execution plan Φ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LamPlan {
    pub n: u64,
    pub tau: u64,
    pub k: u64,
    pub depth: u64,
    /// Upper bound on leaf `M` calls.
    pub max_leaves: u64,
    /// Upper bound on all model calls of the run.
    pub max_calls: u64,
    /// `n ≤ window`: one direct leaf, no decomposition.
    pub direct: bool,
}

impl LamPlan {
    #[must_use]
    pub fn to_json(self) -> Value {
        json!({
            "n": self.n, "tau": self.tau, "k": self.k, "depth": self.depth,
            "max_leaves": self.max_leaves, "max_calls": self.max_calls, "direct": self.direct,
        })
    }
}

/// Deterministic planner: pick `(k*, τ*, d)` or refuse before any model call.
///
/// # Errors
/// Empty input, input over `max_chars`, zero window/τ/fan-out, or a plan whose
/// worst-case call count exceeds the remaining budget.
pub fn plan(req: &PlanReq) -> Result<LamPlan, String> {
    if req.n == 0 {
        return Err("lam: input is empty".into());
    }
    if req.n > req.max_chars {
        return Err(format!(
            "lam: input of {} chars exceeds {MAX_CHARS_ENV}={}",
            req.n, req.max_chars
        ));
    }
    if req.window == 0 {
        return Err("lam: window must be ≥ 1".into());
    }
    let tau = req.tau.unwrap_or(req.window).min(req.window);
    if tau == 0 {
        return Err("lam: tau must be ≥ 1".into());
    }
    let finish = |p: LamPlan| {
        if p.max_calls > req.remaining_calls {
            Err(format!(
                "lam: plan needs up to {} model calls (n={}, tau={}, k={}, depth={}) but only {} remain in the \
                 shared llm_query budget; raise OMP_RLM_MAX_CALLS or shrink the input",
                p.max_calls, p.n, p.tau, p.k, p.depth, req.remaining_calls
            ))
        } else {
            Ok(p)
        }
    };
    if req.n <= req.window {
        return finish(LamPlan {
            n: req.n,
            tau,
            k: 1,
            depth: 0,
            max_leaves: 1,
            max_calls: 1,
            direct: true,
        });
    }
    let k = match req.fanout {
        Some(0) => return Err("lam: fan-out must be ≥ 2".into()),
        Some(f) => f.clamp(2, MAX_FANOUT),
        None => req.n.div_ceil(tau).clamp(2, MAX_FANOUT),
    };
    let (mut size, mut depth) = (req.n, 0u64);
    while size > tau {
        size = size.div_ceil(k);
        depth += 1;
        if depth > MAX_DEPTH {
            return Err("lam: plan exceeds the hard depth bound".into());
        }
    }
    let max_leaves = k
        .saturating_pow(u32::try_from(depth).unwrap_or(u32::MAX))
        .min(req.n);
    finish(LamPlan {
        n: req.n,
        tau,
        k,
        depth,
        max_leaves,
        max_calls: max_leaves.saturating_add(req.extra_calls),
        direct: false,
    })
}

fn arg_u64(args: &Value, key: &str) -> Option<u64> {
    args.get(key).and_then(Value::as_u64)
}

fn plan_op(args: &Value) -> Result<Value, String> {
    let limits = Limits::effective();
    let used = SESSION
        .stats(limits)
        .get("calls")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let max_chars = std::env::var(MAX_CHARS_ENV)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_MAX_CHARS);
    let req = PlanReq {
        n: arg_u64(args, "n").ok_or("lam: n must be a non-negative integer")?,
        window: arg_u64(args, "window").ok_or("lam: window must be a positive integer")?,
        tau: arg_u64(args, "tau"),
        fanout: arg_u64(args, "fanout"),
        extra_calls: arg_u64(args, "extra_calls").unwrap_or(0),
        remaining_calls: limits.max_calls.saturating_sub(used),
        max_chars,
    };
    plan(&req).map(|p| p.to_json())
}

// ---------------------------------------------------------------------------
// SERA rubrics
// ---------------------------------------------------------------------------

/// Rubrics committed by `__rubric_freeze__` (immutable once present).
static FROZEN: LazyLock<Mutex<HashMap<String, FrozenRubric>>> = LazyLock::new(Mutex::default);

/// Parse + freeze the optional `rubric` field of a wire item.
///
/// # Errors
/// The rubric validation message.
pub fn freeze_wire(raw: Option<&Value>) -> Result<Option<FrozenRubric>, String> {
    match raw {
        None | Some(Value::Null) => Ok(None),
        Some(v) => FrozenRubric::freeze(v).map(Some),
    }
}

fn freeze_op(args: &Value) -> Result<Value, String> {
    let rubric = FrozenRubric::freeze(
        args.get("rubric")
            .ok_or("rubric_freeze: missing `rubric`")?,
    )
    .map_err(|e| format!("rubric_freeze: {e}"))?;
    let mut map = FROZEN.lock();
    // Same content = same id: keep the first commit (its timestamp is the proof of order).
    let kept = map.entry(rubric.id.clone()).or_insert(rubric);
    Ok(json!(kept))
}

fn scorer_model(args: &Value) -> String {
    args.get("model")
        .and_then(Value::as_str)
        .filter(|m| !m.is_empty())
        .map(str::to_string)
        .or_else(|| {
            std::env::var(RUBRIC_MODEL_ENV)
                .ok()
                .filter(|m| !m.is_empty())
        })
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

/// Apply a frozen rubric to `output` through `session` (budgeted, memoized).
///
/// # Errors
/// Unknown rubric id, budget exhaustion, scorer failure or an unparsable verdict.
pub async fn score_with(
    session: &Session,
    backend: &dyn Backend,
    limits: Limits,
    args: &Value,
    signal: &AbortSignal,
) -> Result<Value, String> {
    let id = args
        .get("id")
        .and_then(Value::as_str)
        .ok_or("rubric_score: missing `id`")?;
    let rubric = FROZEN.lock().get(id).cloned().ok_or_else(|| {
        format!(
            "rubric_score: rubric {id} was never frozen (call rubric_freeze before the child runs)"
        )
    })?;
    let prompt = args
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let output = args
        .get("output")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let verified = args
        .get("verified")
        .and_then(Value::as_f64)
        .map(|v| v.clamp(0.0, 1.0));
    let run_id = args
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("llm_query");
    let source = args
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or("llm_query");
    let (gate, per) = if output.trim().is_empty() {
        (false, vec![0.0; rubric.criteria.len()])
    } else {
        let q =
            json!({"prompt": rubric.scorer_prompt(prompt, output), "model": scorer_model(args)});
        let reply = session.query(backend, limits, &q, signal).await?;
        let text = reply
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default();
        rubric::parse_scorer_reply(text, rubric.criteria.len())
            .map_err(|e| format!("rubric_score: {e}"))?
    };
    let rec = RubricRecord::from_verdict(
        source,
        &rubric,
        run_id,
        prompt.len(),
        output.len(),
        gate,
        per,
        verified,
    )?;
    rec.append_log();
    Ok(json!(rec))
}

/// Provider path for host ops: the live stack, or the unit-test backend.
struct HostBackend;

impl Backend for HostBackend {
    fn complete<'a>(
        &'a self,
        model: &'a str,
        prompt: &'a str,
        signal: &'a AbortSignal,
    ) -> BoxFut<'a, Result<(String, Usage, Model), String>> {
        Box::pin(async move {
            #[cfg(test)]
            {
                let test_backend = super::rlm::TEST_BACKEND
                    .lock()
                    .expect("test backend")
                    .clone();
                if let Some(b) = test_backend {
                    return b.complete(model, prompt, signal).await;
                }
            }
            super::eval_agents::query_once(model, prompt, None, signal).await
        })
    }
}

/// Scorer for the `task` tool: applies rubrics through the shared RLM session.
pub struct SessionScorer {
    pub model: String,
    pub signal: AbortSignal,
}

impl SessionScorer {
    #[must_use]
    pub fn new(signal: AbortSignal) -> Self {
        Self {
            model: scorer_model(&Value::Null),
            signal,
        }
    }
}

impl RubricScorer for SessionScorer {
    fn complete<'a>(
        &'a self,
        prompt: &'a str,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>> {
        Box::pin(async move {
            let q = json!({"prompt": prompt, "model": self.model});
            let reply = SESSION
                .query(&HostBackend, Limits::effective(), &q, &self.signal)
                .await?;
            Ok(reply
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string())
        })
    }
}

/// Host ops of this module; `None` = not ours.
pub async fn handle(op: &str, args: &Value, ctx: &OpCtx) -> Option<Result<Value, String>> {
    match op {
        "__lam_plan__" => Some(plan_op(args)),
        "__rubric_freeze__" => Some(freeze_op(args)),
        "__rubric_score__" => Some(
            score_with(
                &SESSION,
                &HostBackend,
                Limits::effective(),
                args,
                &ctx.signal,
            )
            .await,
        ),
        "__rubric_rank__" => Some(rank_op(args)),
        _ => None,
    }
}

fn rank_op(args: &Value) -> Result<Value, String> {
    let nums = |k: &str| -> Result<Vec<f64>, String> {
        args.get(k)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("rubric_rank: `{k}` must be an array of numbers"))?
            .iter()
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| format!("rubric_rank: `{k}` must contain only numbers"))
            })
            .collect()
    };
    let (scores, verified) = (nums("scores")?, nums("verified")?);
    if scores.len() != verified.len() {
        return Err("rubric_rank: scores and verified differ in length".into());
    }
    let margin = args.get("margin").and_then(Value::as_f64).unwrap_or(0.1);
    Ok(
        json!({"loss": rubric::rank_loss(&scores, &verified, margin), "pairs_skipped": rubric::rank_loss(&scores, &verified, margin).is_none()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn req(n: u64, window: u64, remaining: u64) -> PlanReq {
        PlanReq {
            n,
            window,
            tau: None,
            fanout: None,
            extra_calls: 0,
            remaining_calls: remaining,
            max_chars: 1 << 30,
        }
    }

    /// Reference execution of Φ's recursion: (rank reached, leaf count).
    fn simulate(len: u64, tau: u64, k: u64, depth: u64) -> (u64, u64) {
        if len <= tau {
            return (depth, 1);
        }
        assert!(depth < MAX_DEPTH, "recursion exceeded the hard bound");
        let parts = k.min(len);
        let (base, extra) = (len / parts, len % parts);
        (0..parts)
            .map(|i| {
                let child = base + u64::from(i < extra);
                assert!(child < len, "split must strictly shrink");
                simulate(child, tau, k, depth + 1)
            })
            .fold((0, 0), |(d, l), (cd, cl)| (d.max(cd), l + cl))
    }

    #[test]
    fn direct_when_it_fits() {
        let p = plan(&req(900, 1000, 10)).unwrap();
        assert!(p.direct && p.depth == 0 && p.max_leaves == 1);
    }

    #[test]
    fn plan_terminates_and_bounds_hold() {
        for (n, w, fan) in [
            (10_000, 700, None),
            (1_000_000, 4_000, None),
            (50_000, 100, Some(3)),
            (257, 2, Some(2)),
            (9_999, 10, Some(7)),
        ] {
            let mut r = req(n, w, u64::MAX);
            r.fanout = fan;
            let p = plan(&r).unwrap();
            let (rank, leaves) = simulate(n, p.tau, p.k, 0);
            assert!(rank <= p.depth, "rank {rank} > planned depth {}", p.depth);
            assert!(
                leaves <= p.max_leaves,
                "leaves {leaves} > bound {}",
                p.max_leaves
            );
            assert!(p.depth <= MAX_DEPTH);
        }
    }

    #[test]
    fn default_partition_is_leaf_minimal_single_level() {
        let p = plan(&req(10_000, 1_000, 100)).unwrap();
        assert_eq!((p.k, p.depth, p.max_leaves), (10, 1, 10));
    }

    #[test]
    fn budget_refused_before_any_call() {
        let e = plan(&req(10_000, 1_000, 9)).unwrap_err();
        assert!(e.contains("only 9 remain"), "{e}");
        let mut r = req(10_000, 1_000, 10);
        r.extra_calls = 1;
        assert!(plan(&r).is_err()); // 10 leaves + 1 synthesis > 10
    }

    #[test]
    fn size_and_argument_guards() {
        let mut r = req(2_000, 100, 1000);
        r.max_chars = 1_000;
        assert!(plan(&r).unwrap_err().contains("exceeds"));
        assert!(plan(&req(0, 10, 10)).is_err());
        assert!(plan(&req(10, 0, 10)).is_err());
        let mut z = req(100, 10, 100);
        z.fanout = Some(0);
        assert!(plan(&z).is_err());
    }

    struct Scorer(AtomicU64, &'static str);
    impl Backend for Scorer {
        fn complete<'a>(
            &'a self,
            _m: &'a str,
            _p: &'a str,
            _s: &'a AbortSignal,
        ) -> BoxFut<'a, Result<(String, Usage, Model), String>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            let text = self.1.to_string();
            Box::pin(async move {
                let usage = Usage {
                    input: 10,
                    output: 5,
                    ..Usage::default()
                };
                let model: Model = serde_json::from_value(json!({
                    "id": "mock-1", "name": "Mock", "api": "mock-api", "provider": "mock",
                    "baseUrl": "http://localhost", "reasoning": false, "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 128000, "maxTokens": 4096,
                }))
                .expect("mock model");
                Ok((text, usage, model))
            })
        }
    }

    #[tokio::test]
    async fn rubric_must_be_frozen_then_scores_and_shares_budget() {
        let session = Session::default();
        let backend = Scorer(
            AtomicU64::new(0),
            "{\"success\": true, \"scores\": [1, 0.5]}",
        );
        let limits = Limits {
            max_depth: 2,
            max_calls: 1,
            max_tokens: 1_000_000,
        };
        let signal = AbortSignal::never();
        // never frozen → refused
        let err = score_with(
            &session,
            &backend,
            limits,
            &json!({"id": "rb-deadbeef", "output": "x"}),
            &signal,
        )
        .await
        .unwrap_err();
        assert!(err.contains("never frozen"), "{err}");
        let frozen = freeze_op(&json!({"rubric": [{"criterion": "right", "weight": 1}, {"criterion": "short", "weight": 1}]})).unwrap();
        let id = frozen["id"].as_str().unwrap();
        let args = json!({"id": id, "prompt": "q", "output": "a", "verified": 1.0});
        let rec = score_with(&session, &backend, limits, &args, &signal)
            .await
            .unwrap();
        assert!((rec["score"].as_f64().unwrap() - 0.75).abs() < 1e-12);
        assert_eq!(rec["verified"], 1.0);
        assert!(rec["scored_ms"].as_u64() > rec["frozen_ms"].as_u64());
        // scorer call is charged to the shared ledger: the 1-call budget is now spent
        let again = json!({"id": id, "prompt": "q", "output": "different", "verified": 0.0});
        let err = score_with(&session, &backend, limits, &again, &signal)
            .await
            .unwrap_err();
        assert!(err.contains("call budget exhausted"), "{err}");
        assert_eq!(backend.0.load(Ordering::SeqCst), 1);
        // empty output: gate 0, no scorer call, score exactly 0
        let empty = score_with(
            &session,
            &backend,
            limits,
            &json!({"id": id, "output": "  "}),
            &signal,
        )
        .await
        .unwrap();
        assert_eq!(empty["score"], 0.0);
        assert_eq!(backend.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn freeze_keeps_first_commit() {
        let a = freeze_op(&json!({"rubric": [{"criterion": "unique-freeze-test"}]})).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(3));
        let b = freeze_op(&json!({"rubric": [{"criterion": "unique-freeze-test"}]})).unwrap();
        assert_eq!(a["frozen_ms"], b["frozen_ms"]);
        assert!(freeze_op(&json!({"rubric": []})).is_err());
    }

    #[test]
    fn rank_op_reports_loss() {
        let r = rank_op(&json!({"scores": [0.1, 0.9], "verified": [1, 0], "margin": 0.5})).unwrap();
        assert!((r["loss"].as_f64().unwrap() - 1.3).abs() < 1e-12);
        let skipped = rank_op(&json!({"scores": [0.1, 0.9], "verified": [1, 1]})).unwrap();
        assert_eq!(skipped["loss"], Value::Null);
        assert!(rank_op(&json!({"scores": [1], "verified": [1, 0]})).is_err());
    }
}
