//! RLM phase 1: `llm_query(prompt, context=None, model=None)` for eval cells.
//!
//! The kernels' `llm_query` helper (Python + JS preludes) calls the
//! `__llm_query__` host op, which lands in [`llm_query_op`]: a stateless,
//! toolless sub-LLM call through the omp-cli provider stack
//! ([`super::eval_agents::query_once`]; the op itself is policy-checked by the
//! eval bridge like every other host op). Every call is charged against hard
//! budgets (max depth / calls / tokens), memoized by
//! `hash(model, context, prompt)` and recorded in the process ledger, which
//! the kernels read back via `llm_stats()` (`__rlm_stats__`) and which can be
//! mirrored to a JSONL file (`OMP_RLM_LOG`) so a bench can count sub-call
//! tokens.
//!
//! Frame (TTIC, arXiv 2609.20831): a sub-call sees only its active frame —
//! `prompt`, the explicit `context` and optional `pointers` — never the parent
//! transcript or the `/rlm` context unless the caller passes it. `frame="only"`
//! is the default and the only accepted value.
//!
//! SRLM sampling (arXiv 2603.15653): `k=K` (+ `select='consistency'|'uncertainty'`,
//! `on_low='expand'|'escalate'|'return'`, `low=0.5`, `escalate_model`) draws K
//! memo-aware, budget-counted samples and returns the selected answer with
//! `agreement` and `candidates` (see [`Session::query_sampled`]). `OMP_RLM_LOG`
//! gets one `"kind":"select"` line per sampled query next to the per-call lines.
//!
//! Budgets (env, read per call): `OMP_RLM_MAX_DEPTH` (default 2),
//! `OMP_RLM_MAX_CALLS` (default 64), `OMP_RLM_MAX_TOKENS` (default 2 000 000,
//! input + output summed over all sub-calls). `/rlm` resets the ledger.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use pa_agent::abort::AbortSignal;
use pa_agent::BoxFut;
use pa_types::ai::{Model, Usage};
use serde_json::{json, Value};

use super::eval_bridge::OpCtx;

/// Default model for sub-calls: the session's smol role.
pub const DEFAULT_MODEL: &str = "smol";

/// Hard limits for one RLM session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_depth: u64,
    pub max_calls: u64,
    pub max_tokens: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_depth: 2,
            max_calls: 64,
            max_tokens: 2_000_000,
        }
    }
}

/// Per-run overrides set by `/rlm --max-*` flags (depth, calls, tokens).
static OVERRIDE: Mutex<[Option<u64>; 3]> = Mutex::new([None; 3]);

/// Install (or clear) the `/rlm` flag overrides.
pub fn set_override(depth: Option<u64>, calls: Option<u64>, tokens: Option<u64>) {
    *OVERRIDE.lock().expect("rlm override") = [depth, calls, tokens];
}

impl Limits {
    /// Defaults, overridden by `OMP_RLM_MAX_{DEPTH,CALLS,TOKENS}`, then by
    /// the active `/rlm` flags.
    #[must_use]
    pub fn effective() -> Self {
        let get = |key: &str, default: u64| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(default)
        };
        let d = Self::default();
        let o = *OVERRIDE.lock().expect("rlm override");
        Self {
            max_depth: o[0].unwrap_or_else(|| get("OMP_RLM_MAX_DEPTH", d.max_depth)),
            max_calls: o[1].unwrap_or_else(|| get("OMP_RLM_MAX_CALLS", d.max_calls)),
            max_tokens: o[2].unwrap_or_else(|| get("OMP_RLM_MAX_TOKENS", d.max_tokens)),
        }
    }
}

/// One accounted `llm_query` call.
#[derive(Debug, Clone)]
pub struct CallRecord {
    pub n: u64,
    /// Model as requested (tier name or selector).
    pub model: String,
    /// Model that actually answered (`provider/id`).
    pub resolved_model: String,
    pub api: String,
    pub provider: String,
    pub depth: u64,
    pub cached: bool,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub prompt_chars: usize,
    pub context_chars: usize,
    pub millis: u64,
    /// Samples requested by the query this call belongs to (1 = plain call).
    pub k: u64,
    /// Sample index within the query (0 = the plain-call memo key).
    pub sample: u64,
}

impl CallRecord {
    fn to_json(&self) -> Value {
        json!({
            "n": self.n, "model": self.model, "resolved_model": self.resolved_model,
            "api": self.api, "provider": self.provider,
            "depth": self.depth, "cached": self.cached,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "prompt_chars": self.prompt_chars, "context_chars": self.context_chars,
            "millis": self.millis, "k": self.k, "sample": self.sample,
        })
    }
}

/// Provider seam: one sub-LLM completion → `(text, usage, model)`.
pub trait Backend: Send + Sync {
    fn complete<'a>(
        &'a self,
        model: &'a str,
        prompt: &'a str,
        signal: &'a AbortSignal,
    ) -> BoxFut<'a, Result<(String, Usage, Model), String>>;
}

/// The real provider stack.
struct LiveBackend;

impl Backend for LiveBackend {
    fn complete<'a>(
        &'a self,
        model: &'a str,
        prompt: &'a str,
        signal: &'a AbortSignal,
    ) -> BoxFut<'a, Result<(String, Usage, Model), String>> {
        Box::pin(async move { super::eval_agents::query_once(model, prompt, None, signal).await })
    }
}

#[derive(Default)]
struct Ledger {
    calls: u64,
    memo_hits: u64,
    input_tokens: u64,
    output_tokens: u64,
    /// Estimated input tokens of in-flight calls (concurrent-call overshoot guard).
    reserved: u64,
    memo: HashMap<u128, String>,
    log: Vec<CallRecord>,
    context: Option<String>,
    /// SRLM sampled queries (`k>1`/`select`): count, summed agreement, low-agreement hits.
    selections: u64,
    agreement_sum: f64,
    low_agreement: u64,
    expansions: u64,
    escalations: u64,
    selection_log: Vec<Value>,
}

/// An RLM session: ledger + memo behind one lock.
#[derive(Default)]
pub struct Session {
    ledger: Mutex<Ledger>,
}

/// Cheap token estimate (≈4 chars/token) for pre-flight budget checks and for
/// providers that report no usage.
fn estimate_tokens(chars: usize) -> u64 {
    (chars as u64).div_ceil(4)
}

fn memo_key(model: &str, context: &str, prompt: &str) -> u128 {
    let hash = |salt: u8| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        salt.hash(&mut h);
        for part in [model, context, prompt] {
            part.len().hash(&mut h);
            part.hash(&mut h);
        }
        h.finish()
    };
    (u128::from(hash(1)) << 64) | u128::from(hash(2))
}

fn render_prompt(prompt: &str, context: &str) -> String {
    if context.is_empty() {
        prompt.to_string()
    } else {
        format!("<context>\n{context}\n</context>\n\n{prompt}")
    }
}

impl Session {
    /// Forget memo, counters and log (a fresh `/rlm` run); the stored
    /// `context` is kept for the caller to replace.
    pub fn reset(&self) {
        let mut ledger = self.ledger.lock().expect("rlm ledger");
        let context = ledger.context.take();
        *ledger = Ledger {
            context,
            ..Ledger::default()
        };
    }

    pub fn set_context(&self, context: Option<String>) {
        self.ledger.lock().expect("rlm ledger").context = context;
    }

    #[must_use]
    pub fn context(&self) -> Option<String> {
        self.ledger.lock().expect("rlm ledger").context.clone()
    }

    /// Ledger snapshot: totals, limits and the per-call log.
    #[must_use]
    pub fn stats(&self, limits: Limits) -> Value {
        let ledger = self.ledger.lock().expect("rlm ledger");
        json!({
            "calls": ledger.calls,
            "memo_hits": ledger.memo_hits,
            "input_tokens": ledger.input_tokens,
            "output_tokens": ledger.output_tokens,
            "total_tokens": ledger.input_tokens + ledger.output_tokens,
            "limits": {
                "max_depth": limits.max_depth,
                "max_calls": limits.max_calls,
                "max_tokens": limits.max_tokens,
            },
            "context_chars": ledger.context.as_ref().map(String::len),
            "frame": FRAME_ONLY,
            "srlm": {
                "queries": ledger.selections,
                "mean_agreement": if ledger.selections == 0 { Value::Null } else { json!(ledger.agreement_sum / ledger.selections as f64) },
                "low_agreement": ledger.low_agreement,
                "expansions": ledger.expansions,
                "escalations": ledger.escalations,
                "selections": ledger.selection_log,
            },
            "records": ledger.log.iter().map(CallRecord::to_json).collect::<Vec<_>>(),
        })
    }

    /// One budgeted, memoized sub-call. Errors are the exact text the cell
    /// sees (`llm_query: …`).
    ///
    /// Plain call (`k` absent or 1, no `select`): one completion, reply
    /// `{text, cached, usage, session}`. SRLM sampling (`k>1` or `select`):
    /// see [`Session::query_sampled`].
    ///
    /// Frame (TTIC): the sub-model sees only the active frame — `prompt`,
    /// the explicit `context`, and optional `pointers` — never the parent
    /// transcript or the `/rlm` context unless the caller passes it.
    ///
    /// # Errors
    /// Invalid arguments, an exceeded depth/call/token budget, or the
    /// backend's failure.
    pub async fn query(
        &self,
        backend: &dyn Backend,
        limits: Limits,
        args: &Value,
        signal: &AbortSignal,
    ) -> Result<Value, String> {
        let prompt = args
            .get("prompt")
            .and_then(Value::as_str)
            .filter(|p| !p.is_empty())
            .ok_or("llm_query: prompt must be a non-empty string")?;
        let context = match args.get("context") {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
        };
        let model = args
            .get("model")
            .and_then(Value::as_str)
            .filter(|m| !m.is_empty())
            .unwrap_or(DEFAULT_MODEL);
        let depth = args.get("depth").and_then(Value::as_u64).unwrap_or(1);
        if depth > limits.max_depth {
            return Err(format!(
                "llm_query: recursion depth budget exhausted (depth {depth} > max_depth {}); \
                 set OMP_RLM_MAX_DEPTH or restructure the query",
                limits.max_depth
            ));
        }
        match args.get("frame") {
            None | Some(Value::Null) => {}
            Some(Value::String(f)) if f == FRAME_ONLY => {}
            Some(other) => {
                return Err(format!(
                    "llm_query: frame must be \"{FRAME_ONLY}\" (the sub-call sees only prompt + context + pointers; \
                     the parent transcript is never forwarded), got {other}"
                ));
            }
        }
        let pointers = match args.get("pointers") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string))
                .collect::<Vec<_>>(),
            Some(_) => return Err("llm_query: pointers must be a list of strings".into()),
        };
        let prompt_frame = if pointers.is_empty() {
            prompt.to_string()
        } else {
            let list = pointers
                .iter()
                .map(|p| format!("- {p}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!("<pointers>\n{list}\n</pointers>\n\n{prompt}")
        };

        let k = match args.get("k") {
            None | Some(Value::Null) => 1,
            Some(v) => v
                .as_u64()
                .filter(|k| (1..=MAX_K).contains(k))
                .ok_or_else(|| format!("llm_query: k must be an integer in 1..={MAX_K}"))?,
        };
        let select = match args.get("select") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(Select::parse(s)?),
            Some(_) => {
                return Err("llm_query: select must be 'consistency' or 'uncertainty'".into())
            }
        };
        if k == 1 && select.is_none() {
            let call = self
                .call_one(
                    backend,
                    limits,
                    model,
                    &prompt_frame,
                    &context,
                    depth,
                    0,
                    1,
                    signal,
                )
                .await?;
            let ledger = self.ledger.lock().expect("rlm ledger");
            return Ok(reply(&call.text, &call.record, &ledger));
        }
        let select = select.unwrap_or(Select::Consistency);
        let on_low = match args.get("on_low") {
            None | Some(Value::Null) => OnLow::Return,
            Some(Value::String(s)) => OnLow::parse(s)?,
            Some(_) => {
                return Err("llm_query: on_low must be 'expand', 'escalate' or 'return'".into())
            }
        };
        let low = args
            .get("low")
            .and_then(Value::as_f64)
            .unwrap_or(DEFAULT_LOW);
        let escalate_model = args
            .get("escalate_model")
            .and_then(Value::as_str)
            .filter(|m| !m.is_empty())
            .unwrap_or(DEFAULT_ESCALATE_MODEL);
        let plan = Plan {
            model,
            prompt: &prompt_frame,
            context: &context,
            depth,
            k,
            select,
            on_low,
            low,
            escalate_model,
        };
        self.query_sampled(backend, limits, &plan, signal).await
    }

    /// SRLM-style sampled sub-call (arXiv 2603.15653 §2.2): draw `k`
    /// completions, group them by normalized answer, keep the plurality group
    /// (self-consistency; agreement = |group| / k) and pick one candidate from
    /// it — the earliest for `consistency`, or for `uncertainty` the one with
    /// the highest joint score `s = VC · Len` (`VC = ln(confidence/100)` from a
    /// verbalized confidence the model reports, `Len` = reply length in
    /// tokens; both ≤ 0 / > 0 so `s ≤ 0` and the best candidate is the one
    /// closest to 0: confident and short). Samples are memo-aware and charged
    /// against the call/token budgets; a budget error after the first sample
    /// just shrinks `k`. Low agreement (`< low`) applies `on_low`: `expand`
    /// draws `k` more samples and reselects, `escalate` asks `escalate_model`
    /// once and returns that, `return` flags it and returns the selection.
    async fn query_sampled(
        &self,
        backend: &dyn Backend,
        limits: Limits,
        plan: &Plan<'_>,
        signal: &AbortSignal,
    ) -> Result<Value, String> {
        let suffix = if plan.select == Select::Uncertainty {
            CONFIDENCE_SUFFIX
        } else {
            ""
        };
        let prompt_eff = format!("{}{suffix}", plan.prompt);
        let mut cands: Vec<Candidate> = Vec::new();
        let mut first_err: Option<String> = None;
        for sample in 0..plan.k {
            match self
                .call_one(
                    backend,
                    limits,
                    plan.model,
                    &prompt_eff,
                    plan.context,
                    plan.depth,
                    sample,
                    plan.k,
                    signal,
                )
                .await
            {
                Ok(call) => cands.push(Candidate::from_call(&call, plan.select, sample)),
                Err(e) => {
                    first_err.get_or_insert(e);
                    break;
                }
            }
        }
        if cands.is_empty() {
            return Err(first_err.unwrap_or_else(|| "llm_query: no samples".into()));
        }
        let mut sel = select_candidate(&cands, plan.select);
        let mut expanded = false;
        let mut escalated: Option<Candidate> = None;
        if sel.agreement < plan.low && first_err.is_none() {
            match plan.on_low {
                OnLow::Return => {}
                OnLow::Expand => {
                    expanded = true;
                    for sample in plan.k..plan.k * 2 {
                        match self
                            .call_one(
                                backend,
                                limits,
                                plan.model,
                                &prompt_eff,
                                plan.context,
                                plan.depth,
                                sample,
                                plan.k * 2,
                                signal,
                            )
                            .await
                        {
                            Ok(call) => {
                                cands.push(Candidate::from_call(&call, plan.select, sample))
                            }
                            Err(_) => break,
                        }
                    }
                    sel = select_candidate(&cands, plan.select);
                }
                OnLow::Escalate => {
                    if let Ok(call) = self
                        .call_one(
                            backend,
                            limits,
                            plan.escalate_model,
                            &prompt_eff,
                            plan.context,
                            plan.depth,
                            0,
                            1,
                            signal,
                        )
                        .await
                    {
                        escalated = Some(Candidate::from_call(&call, plan.select, 0));
                    }
                }
            }
        }
        let low_flag = sel.agreement < plan.low;
        let (text, chosen) = match &escalated {
            Some(c) => (c.answer.clone(), None),
            None => (cands[sel.index].answer.clone(), Some(sel.index)),
        };
        let usage_in: u64 = cands
            .iter()
            .chain(escalated.iter())
            .map(|c| c.input_tokens)
            .sum();
        let usage_out: u64 = cands
            .iter()
            .chain(escalated.iter())
            .map(|c| c.output_tokens)
            .sum();
        let all_cached = cands.iter().chain(escalated.iter()).all(|c| c.cached);
        let mut ledger = self.ledger.lock().expect("rlm ledger");
        ledger.selections += 1;
        ledger.agreement_sum += sel.agreement;
        ledger.low_agreement += u64::from(low_flag);
        ledger.expansions += u64::from(expanded);
        ledger.escalations += u64::from(escalated.is_some());
        let summary = json!({
            "kind": "select",
            "k": cands.len(),
            "select": plan.select.as_str(),
            "agreement": sel.agreement,
            "low": low_flag,
            "on_low": plan.on_low.as_str(),
            "expanded": expanded,
            "escalated": escalated.is_some(),
            "selected": chosen,
            "model": plan.model,
            "answers": cands.iter().map(|c| c.answer.clone()).collect::<Vec<_>>(),
            "scores": cands.iter().map(|c| c.score).collect::<Vec<_>>(),
            "input_tokens": 0, "output_tokens": 0,
        });
        append_json_log(&summary);
        ledger.selection_log.push(summary);
        let mut out = json!({
            "text": text,
            "cached": all_cached,
            "usage": {"input": usage_in, "output": usage_out},
            "session": {
                "calls": ledger.calls,
                "memo_hits": ledger.memo_hits,
                "total_tokens": ledger.input_tokens + ledger.output_tokens,
            },
            "k": cands.len(),
            "select": plan.select.as_str(),
            "agreement": sel.agreement,
            "low": low_flag,
            "expanded": expanded,
            "selected": chosen,
            "candidates": cands.iter().map(Candidate::to_json).collect::<Vec<_>>(),
        });
        if let Some(c) = escalated {
            out["escalated"] = json!({"model": plan.escalate_model, "answer": c.answer});
        }
        Ok(out)
    }

    /// One completion with memo + budget accounting: `sample` 0 shares its memo
    /// key with a plain `llm_query`, later samples are keyed per index so
    /// `k` draws of the same prompt stay distinct (and replay from the memo).
    #[allow(clippy::too_many_arguments)]
    async fn call_one(
        &self,
        backend: &dyn Backend,
        limits: Limits,
        model: &str,
        prompt: &str,
        context: &str,
        depth: u64,
        sample: u64,
        k: u64,
        signal: &AbortSignal,
    ) -> Result<Called, String> {
        let key = memo_key(
            model,
            context,
            &if sample == 0 {
                prompt.to_string()
            } else {
                format!("{prompt}\u{0}sample={sample}")
            },
        );
        let full = render_prompt(prompt, context);
        let estimate = estimate_tokens(full.len());
        let started = Instant::now();
        {
            let mut ledger = self.ledger.lock().expect("rlm ledger");
            if let Some(text) = ledger.memo.get(&key).cloned() {
                ledger.memo_hits += 1;
                let record = CallRecord {
                    n: ledger.log.len() as u64 + 1,
                    model: model.to_string(),
                    resolved_model: String::new(),
                    api: String::new(),
                    provider: String::new(),
                    depth,
                    cached: true,
                    input_tokens: 0,
                    output_tokens: 0,
                    prompt_chars: prompt.len(),
                    context_chars: context.len(),
                    millis: 0,
                    k,
                    sample,
                };
                append_log(&record);
                ledger.log.push(record.clone());
                return Ok(Called { text, record });
            }
            if ledger.calls >= limits.max_calls {
                return Err(format!(
                    "llm_query: call budget exhausted ({} of max_calls {} used); \
                     set OMP_RLM_MAX_CALLS to raise it",
                    ledger.calls, limits.max_calls
                ));
            }
            let spent = ledger.input_tokens + ledger.output_tokens + ledger.reserved;
            if spent + estimate > limits.max_tokens {
                return Err(format!(
                    "llm_query: token budget exhausted ({spent} spent/reserved + ~{estimate} for this call > \
                     max_tokens {}); set OMP_RLM_MAX_TOKENS to raise it",
                    limits.max_tokens
                ));
            }
            ledger.calls += 1;
            ledger.reserved += estimate;
        }

        let result = backend.complete(model, &full, signal).await;
        let mut ledger = self.ledger.lock().expect("rlm ledger");
        ledger.reserved = ledger.reserved.saturating_sub(estimate);
        let (text, mut usage, model_info) = result.map_err(|e| format!("llm_query: {e}"))?;
        if usage.input == 0 && usage.output == 0 && usage.cache_read == 0 && usage.cache_write == 0
        {
            // Provider reported no usage: charge the estimate rather than free calls.
            usage.input = estimate;
        }
        usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
        let input = usage.input + usage.cache_read + usage.cache_write;
        let output = usage.output;
        ledger.input_tokens += input;
        ledger.output_tokens += output;
        ledger.memo.insert(key, text.clone());
        let record = CallRecord {
            n: ledger.log.len() as u64 + 1,
            model: model.to_string(),
            resolved_model: format!("{}/{}", model_info.provider, model_info.id),
            api: model_info.api.clone(),
            provider: model_info.provider.clone(),
            depth,
            cached: false,
            input_tokens: input,
            output_tokens: output,
            prompt_chars: prompt.len(),
            context_chars: context.len(),
            millis: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            k,
            sample,
        };
        append_log(&record);
        append_session_entry(&record, &usage);
        ledger.log.push(record.clone());
        Ok(Called { text, record })
    }
}

/// A finished `call_one`.
struct Called {
    text: String,
    record: CallRecord,
}

/// The only frame mode: prompt + explicit context + pointers (TTIC).
pub const FRAME_ONLY: &str = "only";
/// Upper bound for `k` (samples per query).
const MAX_K: u64 = 16;
/// Agreement below this triggers `on_low`.
const DEFAULT_LOW: f64 = 0.5;
/// Model used by `on_low="escalate"` when `escalate_model` is not given.
const DEFAULT_ESCALATE_MODEL: &str = "default";
/// Appended under `select="uncertainty"` to elicit SRLM's verbalized confidence.
const CONFIDENCE_SUFFIX: &str =
    "\n\nAfter your answer, on a final separate line, report your confidence \
as JSON: {\"confidence\": N} where N is an integer in 1..=100.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Select {
    Consistency,
    Uncertainty,
}

impl Select {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "consistency" => Ok(Self::Consistency),
            "uncertainty" => Ok(Self::Uncertainty),
            other => Err(format!(
                "llm_query: select must be 'consistency' or 'uncertainty', got {other:?}"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Consistency => "consistency",
            Self::Uncertainty => "uncertainty",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnLow {
    Return,
    Expand,
    Escalate,
}

impl OnLow {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "return" => Ok(Self::Return),
            "expand" => Ok(Self::Expand),
            "escalate" => Ok(Self::Escalate),
            other => Err(format!(
                "llm_query: on_low must be 'expand', 'escalate' or 'return', got {other:?}"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Return => "return",
            Self::Expand => "expand",
            Self::Escalate => "escalate",
        }
    }
}

struct Plan<'a> {
    model: &'a str,
    prompt: &'a str,
    context: &'a str,
    depth: u64,
    k: u64,
    select: Select,
    on_low: OnLow,
    low: f64,
    escalate_model: &'a str,
}

/// One sampled answer.
#[derive(Debug, Clone)]
struct Candidate {
    sample: u64,
    /// Answer text with the confidence line removed.
    answer: String,
    /// Verbalized confidence 1..=100 (uncertainty mode only).
    confidence: Option<f64>,
    /// SRLM joint score `ln(confidence/100) · Len` (uncertainty mode only).
    score: Option<f64>,
    cached: bool,
    input_tokens: u64,
    output_tokens: u64,
}

impl Candidate {
    fn from_call(call: &Called, select: Select, sample: u64) -> Self {
        let (answer, confidence) = match select {
            Select::Consistency => (call.text.trim().to_string(), None),
            Select::Uncertainty => {
                let (a, c) = split_confidence(&call.text);
                (a, Some(c))
            }
        };
        let score = confidence.map(|c| (c / 100.0).ln() * estimate_tokens(call.text.len()) as f64);
        Self {
            sample,
            answer,
            confidence,
            score,
            cached: call.record.cached,
            input_tokens: call.record.input_tokens,
            output_tokens: call.record.output_tokens,
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "sample": self.sample, "text": self.answer, "confidence": self.confidence,
            "score": self.score, "cached": self.cached,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
        })
    }
}

/// Split a reply into `(answer, confidence)`: the trailing `{"confidence": N}`
/// line is removed and parsed (clamped to 1..=100); missing → 50.
fn split_confidence(text: &str) -> (String, f64) {
    const MISSING: f64 = 50.0;
    let Some(pos) = text.rfind("\"confidence\"") else {
        return (text.trim().to_string(), MISSING);
    };
    let tail = &text[pos + "\"confidence\"".len()..];
    let digits: String = tail
        .trim_start_matches(|c: char| c == ':' || c.is_whitespace() || c == '"')
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let conf = digits
        .parse::<f64>()
        .map_or(MISSING, |v| v.clamp(1.0, 100.0));
    let head = &text[..pos];
    let cut = head.rfind('{').unwrap_or(head.len());
    (head[..cut].trim().to_string(), conf)
}

/// Normalize an answer for agreement grouping: lowercase alphanumerics, single spaces.
fn normalize_answer(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_lowercase().collect::<String>()
            } else {
                " ".to_string()
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

struct Selection {
    index: usize,
    agreement: f64,
}

/// SRLM selection: plurality group by normalized answer (ties → the group
/// holding the earliest candidate), then within it the earliest candidate
/// (`consistency`) or the max joint score `VC·Len` (`uncertainty`; it also breaks
/// plurality size ties).
fn select_candidate(cands: &[Candidate], select: Select) -> Selection {
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, c) in cands.iter().enumerate() {
        let norm = normalize_answer(&c.answer);
        match groups.iter_mut().find(|(n, _)| *n == norm) {
            Some((_, members)) => members.push(i),
            None => groups.push((norm, vec![i])),
        }
    }
    // Plurality; size ties go to the group with the best joint score under
    // `uncertainty`, else to the group holding the earliest candidate.
    let top = |g: &(String, Vec<usize>)| {
        g.1.iter()
            .filter_map(|&i| cands[i].score)
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let best = groups
        .iter()
        .fold(None::<&(String, Vec<usize>)>, |acc, g| match acc {
            Some(a) if a.1.len() > g.1.len() => Some(a),
            Some(a)
                if a.1.len() == g.1.len()
                    && (select == Select::Consistency || top(a) >= top(g)) =>
            {
                Some(a)
            }
            _ => Some(g),
        });
    let members = &best.expect("at least one candidate").1;
    let index = match select {
        Select::Consistency => members[0],
        Select::Uncertainty => members.iter().copied().fold(members[0], |b, i| {
            if cands[i].score > cands[b].score {
                i
            } else {
                b
            }
        }),
    };
    Selection {
        index,
        agreement: members.len() as f64 / cands.len() as f64,
    }
}

fn reply(text: &str, record: &CallRecord, ledger: &Ledger) -> Value {
    json!({
        "text": text,
        "cached": record.cached,
        "usage": {"input": record.input_tokens, "output": record.output_tokens},
        "session": {
            "calls": ledger.calls,
            "memo_hits": ledger.memo_hits,
            "total_tokens": ledger.input_tokens + ledger.output_tokens,
        },
    })
}

/// Mirror one record to `OMP_RLM_LOG` (JSONL) when set; failures are ignored.
fn append_log(record: &CallRecord) {
    append_json_log(&record.to_json());
}

/// Append one JSON line to `OMP_RLM_LOG` when set; failures are ignored.
fn append_json_log(line: &Value) {
    use std::io::Write as _;
    let Some(path) = std::env::var_os("OMP_RLM_LOG") else {
        return;
    };
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{line}");
    }
}

/// Session-entry sink installed by the runtime ([`set_entry_sink`]): one
/// `model_usage` entry per provider call, so `omp stats` / bench can count
/// sub-call tokens straight from the session file.
static ENTRY_SINK: LazyLock<
    parking_lot::RwLock<Option<std::sync::Arc<dyn Fn(&str, Value) + Send + Sync>>>,
> = LazyLock::new(|| parking_lot::RwLock::new(None));

/// Install (or clear) the sink that records sub-call accounting in the session.
pub fn set_entry_sink(sink: Option<std::sync::Arc<dyn Fn(&str, Value) + Send + Sync>>) {
    *ENTRY_SINK.write() = sink;
}

/// Record one call as a `model_usage` session entry (no-op without a sink).
fn append_session_entry(record: &CallRecord, usage: &Usage) {
    let sink = ENTRY_SINK.read().clone();
    let Some(sink) = sink else {
        return;
    };
    let usage = serde_json::to_value(usage).unwrap_or_else(|_| json!({}));
    sink(
        "model_usage",
        json!({
            "purpose": "rlm",
            "role": "rlm",
            "api": record.api,
            "provider": record.provider,
            "model": record.resolved_model,
            "usage": usage,
            "stopReason": "stop",
        }),
    );
}

/// Process-wide session shared by every kernel (budgets are per session).
pub static SESSION: LazyLock<Session> = LazyLock::new(Session::default);

#[cfg(test)]
pub(crate) static TEST_BACKEND: Mutex<Option<std::sync::Arc<dyn Backend>>> = Mutex::new(None);

/// `__llm_query__` host op.
pub async fn llm_query_op(args: &Value, ctx: &OpCtx) -> Result<Value, String> {
    #[cfg(test)]
    if let Some(backend) = TEST_BACKEND.lock().expect("test backend").clone() {
        return SESSION
            .query(backend.as_ref(), Limits::effective(), args, &ctx.signal)
            .await;
    }
    SESSION
        .query(&LiveBackend, Limits::effective(), args, &ctx.signal)
        .await
}

/// `__rlm_stats__` host op.
#[must_use]
pub fn stats_snapshot() -> Value {
    SESSION.stats(Limits::effective())
}

/// `__rlm_context__` host op: the context `/rlm` loaded (null when none).
pub fn context_op() -> Result<Value, String> {
    Ok(SESSION.context().map_or(Value::Null, Value::String))
}

// ---------------------------------------------------------------------------
// `/rlm`
// ---------------------------------------------------------------------------

/// Parsed `/rlm` arguments.
#[derive(Debug, PartialEq, Eq)]
pub struct RlmArgs {
    pub question: String,
    pub context: ContextSource,
    pub max_depth: Option<u64>,
    pub max_calls: Option<u64>,
    pub max_tokens: Option<u64>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ContextSource {
    File(String),
    Text(String),
    /// No `@file` / `-- text`: the current conversation is the context.
    Session,
}

pub const USAGE: &str = "Usage: /rlm [--max-depth=N] [--max-calls=N] [--max-tokens=N] <question> @<file>\n       /rlm [flags] <question> -- <context text>\n       /rlm [flags] <question>   (context = this session's conversation)\n\
Loads the file/text into the eval kernel as `context` and lets the model inspect it with code, delegating chunks to `llm_query` (smol model by default). `/rlm stats` prints sub-call accounting.";

/// Parse `/rlm` arguments: leading `--max-*=N` flags, then the question, then
/// either a trailing `@file` token or ` -- ` followed by literal context text.
///
/// # Errors
/// Usage text when the question or the context source is missing.
pub fn parse_args(args: &str) -> Result<RlmArgs, String> {
    let mut rest = args.trim();
    let (mut depth, mut calls, mut tokens) = (None, None, None);
    while let Some(flag) = rest.strip_prefix("--") {
        let end = flag.find(char::is_whitespace).unwrap_or(flag.len());
        let (token, tail) = flag.split_at(end);
        let Some((name, value)) = token.split_once('=') else {
            break; // bare `--` delimiter or unknown token: part of the question
        };
        let parsed: u64 = value
            .parse()
            .map_err(|_| format!("/rlm: invalid value for --{name}: {value}"))?;
        match name {
            "max-depth" => depth = Some(parsed),
            "max-calls" => calls = Some(parsed),
            "max-tokens" => tokens = Some(parsed),
            _ => return Err(format!("/rlm: unknown flag --{name}\n{USAGE}")),
        }
        rest = tail.trim_start();
    }
    let (question, context) = if let Some((q, text)) = rest.split_once(" -- ") {
        (q.trim(), ContextSource::Text(text.to_string()))
    } else if let Some(at) = rest
        .rfind(" @")
        .map(|i| i + 1)
        .or_else(|| rest.starts_with('@').then_some(0))
    {
        let path = rest[at + 1..].trim();
        if path.is_empty() || path.contains(char::is_whitespace) {
            return Err(USAGE.to_string());
        }
        (rest[..at].trim(), ContextSource::File(path.to_string()))
    } else {
        (rest, ContextSource::Session)
    };
    if question.is_empty() {
        return Err(USAGE.to_string());
    }
    Ok(RlmArgs {
        question: question.to_string(),
        context,
        max_depth: depth,
        max_calls: calls,
        max_tokens: tokens,
    })
}

/// Load the stored context into the kernel as `context` (a real eval cell on
/// the session's per-cwd Python kernel, so the model's later cells see it).
///
/// # Errors
/// The cell's failure text.
pub async fn load_context_cell(cwd: &std::path::Path) -> Result<(), String> {
    let env = std::sync::Arc::new(super::Env::with_mode(cwd, pi_edit::EditMode::Replace));
    let ctx = super::eval::CallCtx {
        tool_call_id: "rlm-load".to_string(),
        signal: AbortSignal::never(),
        on_update: None,
    };
    let args = json!({"language": "py", "code": "context = rlm_context()", "timeout": 60});
    let result = super::eval::eval(env, args, ctx)
        .await
        .map_err(|e| format!("{e:#}"))?;
    if result.is_error == Some(true) {
        let text = result
            .content
            .iter()
            .filter_map(|c| match c {
                pa_agent::types::ToolResultContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Err(text);
    }
    Ok(())
}

/// The RLM loop prompt handed to the model.
#[must_use]
pub fn loop_prompt(question: &str, context_chars: usize, limits: Limits) -> String {
    format!(
        "RLM mode. A long input of {context_chars} characters is loaded in the Python `eval` kernel as the string variable `context` \
(if the name is missing after a kernel restart, run `context = rlm_context()`). It is NOT shown to you and must never be printed whole.\n\
Answer the question by writing code in `eval` (language `py`) that inspects `context`: `len(context)`, slices, `re` search, \
splitting into chunks, printing only short excerpts. For questions that need reading comprehension over a piece of text call \
`llm_query(prompt, context=piece)` (returns a string; cheap sub-model, memoized). Prefer exact search first, then delegate only \
the relevant chunks. `llm_stats()` shows sub-call accounting. Hard budgets: max {} sub-calls, {} tokens, depth {}; exceeding them raises an error in the cell.\n\
When done, reply with the final answer in plain text.\n\nQuestion: {question}",
        limits.max_calls, limits.max_tokens, limits.max_depth
    )
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    /// Serializes the query tests: the session-accounting sink is process-global,
    /// so the sink-collecting test must not run concurrently with other queries.
    static TEST_GUARD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// Mock provider: echoes a fixed text, counts calls, reports fake usage.
    struct Mock {
        calls: AtomicU64,
        input: u64,
        output: u64,
    }

    impl Mock {
        fn new(input: u64, output: u64) -> Self {
            Self {
                calls: AtomicU64::new(0),
                input,
                output,
            }
        }
    }

    fn fake_model() -> Model {
        serde_json::from_value(json!({
            "id": "mock-1", "name": "Mock", "api": "mock-api", "provider": "mock",
            "baseUrl": "http://localhost", "reasoning": false, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 128000, "maxTokens": 4096,
        }))
        .expect("mock model")
    }

    impl Backend for Mock {
        fn complete<'a>(
            &'a self,
            model: &'a str,
            prompt: &'a str,
            _signal: &'a AbortSignal,
        ) -> BoxFut<'a, Result<(String, Usage, Model), String>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let usage = Usage {
                    input: self.input,
                    output: self.output,
                    total_tokens: self.input + self.output,
                    ..Usage::default()
                };
                Ok((format!("{model}:{}", prompt.len()), usage, fake_model()))
            })
        }
    }

    fn q(prompt: &str, context: Option<&str>) -> Value {
        let mut args = json!({"prompt": prompt});
        if let Some(c) = context {
            args["context"] = json!(c);
        }
        args
    }

    fn limits(max_depth: u64, max_calls: u64, max_tokens: u64) -> Limits {
        Limits {
            max_depth,
            max_calls,
            max_tokens,
        }
    }

    #[tokio::test]
    async fn call_budget_exhaustion_is_a_clear_error() {
        let _guard = TEST_GUARD.lock().await;
        let (session, mock, sig) = (Session::default(), Mock::new(10, 5), AbortSignal::never());
        let lim = limits(2, 2, 1_000_000);
        for i in 0..2 {
            session
                .query(&mock, lim, &q(&format!("p{i}"), None), &sig)
                .await
                .unwrap();
        }
        let err = session
            .query(&mock, lim, &q("p2", None), &sig)
            .await
            .unwrap_err();
        assert!(
            err.contains("call budget exhausted") && err.contains("max_calls 2"),
            "{err}"
        );
        assert_eq!(
            mock.calls.load(Ordering::SeqCst),
            2,
            "no provider call after exhaustion"
        );
    }

    #[tokio::test]
    async fn token_budget_exhaustion_blocks_before_the_provider_call() {
        let _guard = TEST_GUARD.lock().await;
        let (session, mock, sig) = (Session::default(), Mock::new(80, 30), AbortSignal::never());
        let lim = limits(2, 10, 100);
        session
            .query(&mock, lim, &q("a", None), &sig)
            .await
            .unwrap(); // 110 tokens: budget now exceeded
        let err = session
            .query(&mock, lim, &q("b", None), &sig)
            .await
            .unwrap_err();
        assert!(err.contains("token budget exhausted"), "{err}");
        assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
        // A huge prompt is refused up front from the estimate alone.
        let fresh = Session::default();
        let big = "x".repeat(4_000);
        let err = fresh
            .query(&mock, limits(2, 10, 100), &q("p", Some(&big)), &sig)
            .await
            .unwrap_err();
        assert!(err.contains("token budget exhausted"), "{err}");
        assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn depth_budget_is_enforced() {
        let _guard = TEST_GUARD.lock().await;
        let (session, mock, sig) = (Session::default(), Mock::new(1, 1), AbortSignal::never());
        let mut args = q("p", None);
        args["depth"] = json!(3);
        let err = session
            .query(&mock, limits(2, 10, 1000), &args, &sig)
            .await
            .unwrap_err();
        assert!(err.contains("depth budget exhausted"), "{err}");
        args["depth"] = json!(2);
        session
            .query(&mock, limits(2, 10, 1000), &args, &sig)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn memo_hit_skips_provider_and_budget() {
        let _guard = TEST_GUARD.lock().await;
        let (session, mock, sig) = (Session::default(), Mock::new(40, 10), AbortSignal::never());
        let lim = limits(2, 1, 1_000_000);
        let first = session
            .query(&mock, lim, &q("what?", Some("ctx")), &sig)
            .await
            .unwrap();
        // max_calls is 1 and spent: only a memo hit can succeed now.
        let second = session
            .query(&mock, lim, &q("what?", Some("ctx")), &sig)
            .await
            .unwrap();
        assert_eq!(first["text"], second["text"]);
        assert_eq!(first["cached"], json!(false));
        assert_eq!(second["cached"], json!(true));
        assert_eq!(second["usage"], json!({"input": 0, "output": 0}));
        assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
        // Different context or model is a different key.
        assert!(session
            .query(&mock, lim, &q("what?", Some("other")), &sig)
            .await
            .is_err());
        let mut other_model = q("what?", Some("ctx"));
        other_model["model"] = json!("slow");
        assert!(session.query(&mock, lim, &other_model, &sig).await.is_err());
    }

    #[tokio::test]
    async fn accounting_records_every_call() {
        let _guard = TEST_GUARD.lock().await;
        let (session, mock, sig) = (Session::default(), Mock::new(40, 10), AbortSignal::never());
        let lim = limits(2, 10, 1_000_000);
        session
            .query(&mock, lim, &q("a", Some("one")), &sig)
            .await
            .unwrap();
        session
            .query(&mock, lim, &q("b", Some("two")), &sig)
            .await
            .unwrap();
        session
            .query(&mock, lim, &q("a", Some("one")), &sig)
            .await
            .unwrap(); // memo
        let stats = session.stats(lim);
        assert_eq!(stats["calls"], 2);
        assert_eq!(stats["memo_hits"], 1);
        assert_eq!(stats["input_tokens"], 80);
        assert_eq!(stats["output_tokens"], 20);
        assert_eq!(stats["total_tokens"], 100);
        let records = stats["records"].as_array().unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(
            records[0]["model"], "smol",
            "default model is the smol role"
        );
        assert_eq!(records[0]["context_chars"], 3);
        assert_eq!(records[2]["cached"], true);
        session.reset();
        assert_eq!(session.stats(lim)["calls"], 0);
    }

    #[tokio::test]
    async fn session_accounting_records_model_usage_entries() {
        let _guard = TEST_GUARD.lock().await;
        use std::sync::Arc;
        let entries: Arc<parking_lot::Mutex<Vec<(String, Value)>>> =
            Arc::new(parking_lot::Mutex::new(Vec::new()));
        let collected = Arc::clone(&entries);
        set_entry_sink(Some(Arc::new(move |kind: &str, payload: Value| {
            collected.lock().push((kind.to_string(), payload));
        })));
        let (session, mock, sig) = (Session::default(), Mock::new(40, 10), AbortSignal::never());
        let lim = limits(2, 10, 1_000_000);
        session
            .query(&mock, lim, &q("a", Some("one")), &sig)
            .await
            .unwrap();
        session
            .query(&mock, lim, &q("a", Some("one")), &sig)
            .await
            .unwrap(); // memo: no entry
        set_entry_sink(None);
        let entries = entries.lock();
        assert_eq!(
            entries.len(),
            1,
            "one model_usage entry per provider call; memo hits are free"
        );
        let (kind, payload) = &entries[0];
        assert_eq!(kind, "model_usage");
        assert_eq!(payload["purpose"], "rlm");
        assert_eq!(payload["provider"], "mock");
        assert_eq!(payload["model"], "mock/mock-1");
        assert_eq!(payload["api"], "mock-api");
        assert_eq!(payload["usage"]["input"], 40);
        assert_eq!(payload["usage"]["output"], 10);
        assert_eq!(payload["usage"]["totalTokens"], 50);
        assert_eq!(payload["stopReason"], "stop");
    }

    #[test]
    fn parse_rlm_args() {
        assert_eq!(
            parse_args("who is X? @docs/big.txt").unwrap(),
            RlmArgs {
                question: "who is X?".into(),
                context: ContextSource::File("docs/big.txt".into()),
                max_depth: None,
                max_calls: None,
                max_tokens: None,
            }
        );
        let a = parse_args("--max-calls=5 --max-tokens=900 find it -- some text here").unwrap();
        assert_eq!(a.question, "find it");
        assert_eq!(a.context, ContextSource::Text("some text here".into()));
        assert_eq!(
            (a.max_calls, a.max_tokens, a.max_depth),
            (Some(5), Some(900), None)
        );
        // No @file / `-- text`: the session conversation is the context.
        assert_eq!(
            parse_args("what did we decide?").unwrap().context,
            ContextSource::Session
        );
        assert!(parse_args("").is_err());
        assert!(parse_args("@only.txt").is_err());
        assert!(parse_args("--max-calls=x q @f").is_err());
    }

    /// Scripted provider: reply `n` is `script(model, n)`; records every prompt.
    struct Script {
        calls: AtomicU64,
        prompts: Mutex<Vec<String>>,
        script: Box<dyn Fn(&str, u64) -> String + Send + Sync>,
    }

    impl Script {
        fn new(script: impl Fn(&str, u64) -> String + Send + Sync + 'static) -> Self {
            Self {
                calls: AtomicU64::new(0),
                prompts: Mutex::new(Vec::new()),
                script: Box::new(script),
            }
        }
    }

    impl Backend for Script {
        fn complete<'a>(
            &'a self,
            model: &'a str,
            prompt: &'a str,
            _signal: &'a AbortSignal,
        ) -> BoxFut<'a, Result<(String, Usage, Model), String>> {
            Box::pin(async move {
                let n = self.calls.fetch_add(1, Ordering::SeqCst);
                self.prompts.lock().unwrap().push(prompt.to_string());
                let text = (self.script)(model, n);
                let usage = Usage {
                    input: 10,
                    output: 5,
                    total_tokens: 15,
                    ..Usage::default()
                };
                Ok((text, usage, fake_model()))
            })
        }
    }

    fn ks(prompt: &str, k: u64, extra: Value) -> Value {
        let mut args = json!({"prompt": prompt, "k": k});
        for (key, v) in extra.as_object().unwrap() {
            args[key] = v.clone();
        }
        args
    }

    #[tokio::test]
    async fn frame_only_never_forwards_parent_history() {
        let _guard = TEST_GUARD.lock().await;
        let (session, sig) = (Session::default(), AbortSignal::never());
        // The parent's loaded `/rlm` context must not ride along into sub-calls.
        session.set_context(Some("PARENT-TRANSCRIPT-SECRET".into()));
        let backend = Script::new(|_, _| "ok".into());
        let lim = limits(2, 10, 1_000_000);
        session
            .query(&backend, lim, &q("Q?", Some("chunk")), &sig)
            .await
            .unwrap();
        let mut args = q("Q2?", None);
        args["frame"] = json!("only");
        args["pointers"] = json!(["/tmp/a.txt", "memo#3"]);
        session.query(&backend, lim, &args, &sig).await.unwrap();
        let prompts = backend.prompts.lock().unwrap().clone();
        assert_eq!(prompts[0], "<context>\nchunk\n</context>\n\nQ?");
        assert_eq!(
            prompts[1],
            "<pointers>\n- /tmp/a.txt\n- memo#3\n</pointers>\n\nQ2?"
        );
        assert!(
            prompts
                .iter()
                .all(|p| !p.contains("PARENT-TRANSCRIPT-SECRET")),
            "{prompts:?}"
        );
        let mut bad = q("Q3?", None);
        bad["frame"] = json!("parent");
        let err = session.query(&backend, lim, &bad, &sig).await.unwrap_err();
        assert!(err.contains("frame must be"), "{err}");
        assert_eq!(backend.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn consistency_picks_plurality_and_replays_from_memo() {
        let _guard = TEST_GUARD.lock().await;
        let (session, sig) = (Session::default(), AbortSignal::never());
        let backend = Script::new(|_, n| ["Paris", "paris.", "London"][n as usize % 3].to_string());
        let lim = limits(2, 10, 1_000_000);
        let args = ks("capital?", 3, json!({"select": "consistency"}));
        let r = session.query(&backend, lim, &args, &sig).await.unwrap();
        assert_eq!(r["text"], "Paris");
        assert_eq!(r["k"], 3);
        assert!((r["agreement"].as_f64().unwrap() - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(r["candidates"].as_array().unwrap().len(), 3);
        assert_eq!(r["usage"], json!({"input": 30, "output": 15}));
        assert_eq!(backend.calls.load(Ordering::SeqCst), 3);
        // Same query again: every sample is a memo hit, budget untouched.
        let again = session
            .query(&backend, limits(2, 3, 1_000_000), &args, &sig)
            .await
            .unwrap();
        assert_eq!(again["text"], "Paris");
        assert_eq!(again["cached"], json!(true));
        assert_eq!(backend.calls.load(Ordering::SeqCst), 3);
        // A plain llm_query shares sample 0's memo entry.
        let plain = session
            .query(&backend, lim, &q("capital?", None), &sig)
            .await
            .unwrap();
        assert_eq!(plain["cached"], json!(true));
        let stats = session.stats(lim);
        assert_eq!(stats["srlm"]["queries"], 2);
        assert!(stats["srlm"]["mean_agreement"].as_f64().unwrap() > 0.6);
    }

    #[tokio::test]
    async fn uncertainty_selects_by_confidence_times_length_within_plurality() {
        let _guard = TEST_GUARD.lock().await;
        let (session, sig) = (Session::default(), AbortSignal::never());
        let long = "x".repeat(400);
        let replies = [
            "42\n{\"confidence\": 60}".to_string(),
            format!("42\n{long}\n{{\"confidence\": 99}}"),
            "7\n{\"confidence\": 100}".to_string(),
        ];
        // The long 99%-confident sample has s = ln(.99)*~105 ≈ -1.1; the short 60% one
        // s = ln(.6)*~6 ≈ -3.1 -> the long one wins; the lone 100% "7" is outvoted.
        let backend = Script::new(move |_, n| replies[n as usize % 3].clone());
        let args = ks("answer?", 3, json!({"select": "uncertainty"}));
        let r = session
            .query(&backend, limits(2, 10, 1_000_000), &args, &sig)
            .await
            .unwrap();
        // Different answer text -> three singleton groups; the size tie goes to the best
        // joint score: the 100%-confidence "7" has s = ln(1)*Len = 0.
        assert_eq!(r["selected"], 2);
        assert_eq!(r["candidates"][0]["confidence"], 60.0);
        assert!(
            r["candidates"][1]["score"].as_f64().unwrap()
                > r["candidates"][0]["score"].as_f64().unwrap()
        );
        assert!(backend.prompts.lock().unwrap()[0].contains("\"confidence\""));
    }

    #[test]
    fn uncertainty_score_prefers_confident_and_short_inside_the_plurality_group() {
        let mk = |answer: &str, conf: f64, tokens: u64| Candidate {
            sample: 0,
            answer: answer.into(),
            confidence: Some(conf),
            score: Some((conf / 100.0).ln() * tokens as f64),
            cached: false,
            input_tokens: 0,
            output_tokens: 0,
        };
        let cands = [mk("42", 60.0, 6), mk("42", 99.0, 100), mk("7", 100.0, 1)];
        let sel = select_candidate(&cands, Select::Uncertainty);
        assert_eq!(sel.index, 1, "ln(.99)*100 = -1.0 beats ln(.6)*6 = -3.1");
        assert!((sel.agreement - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(select_candidate(&cands, Select::Consistency).index, 0);
        // Size tie between "a" and "b": uncertainty prefers the group with the better score.
        let tie = [mk("a", 60.0, 6), mk("b", 99.0, 6)];
        assert_eq!(select_candidate(&tie, Select::Uncertainty).index, 1);
        assert_eq!(select_candidate(&tie, Select::Consistency).index, 0);
        let (answer, conf) = split_confidence("Paris\n{\"confidence\": 85}");
        assert_eq!((answer.as_str(), conf), ("Paris", 85.0));
        assert_eq!(split_confidence("no conf").1, 50.0);
    }

    #[tokio::test]
    async fn low_agreement_expand_and_escalate() {
        let _guard = TEST_GUARD.lock().await;
        let sig = AbortSignal::never();
        let lim = limits(2, 20, 1_000_000);
        // expand: first 3 all differ (1/3 < 0.5), 3 more draws, "a" ends with 3 of 6.
        let session = Session::default();
        let backend =
            Script::new(|_, n| ["a", "b", "c", "a", "a", "b"][n as usize % 6].to_string());
        let args = ks("q", 3, json!({"on_low": "expand"}));
        let r = session.query(&backend, lim, &args, &sig).await.unwrap();
        assert_eq!(
            (
                r["text"].as_str().unwrap(),
                r["k"].as_u64().unwrap(),
                r["expanded"].clone()
            ),
            ("a", 6, json!(true))
        );
        assert_eq!(backend.calls.load(Ordering::SeqCst), 6);
        // escalate: the escalation model's answer wins, and it is one extra call.
        let session = Session::default();
        let backend = Script::new(|model, n| {
            if model == "big" {
                "BIG".into()
            } else {
                format!("s{n}")
            }
        });
        let args = ks(
            "q",
            3,
            json!({"on_low": "escalate", "escalate_model": "big"}),
        );
        let r = session.query(&backend, lim, &args, &sig).await.unwrap();
        assert_eq!(r["text"], "BIG");
        assert_eq!(r["escalated"]["model"], "big");
        assert_eq!(backend.calls.load(Ordering::SeqCst), 4);
        // return: flagged, selection kept.
        let session = Session::default();
        let backend = Script::new(|_, n| format!("s{n}"));
        let r = session
            .query(
                &backend,
                lim,
                &ks("q", 3, json!({"on_low": "return"})),
                &sig,
            )
            .await
            .unwrap();
        assert_eq!(
            (r["low"].clone(), r["text"].clone()),
            (json!(true), json!("s0"))
        );
        assert_eq!(session.stats(lim)["srlm"]["low_agreement"], 1);
    }

    #[tokio::test]
    async fn sampling_counts_against_budget_and_degrades_gracefully() {
        let _guard = TEST_GUARD.lock().await;
        let sig = AbortSignal::never();
        let backend = Script::new(|_, _| "same".into());
        let session = Session::default();
        // max_calls 2: only two of the three samples can be drawn; still an answer.
        let r = session
            .query(
                &backend,
                limits(2, 2, 1_000_000),
                &ks("q", 3, json!({})),
                &sig,
            )
            .await
            .unwrap();
        assert_eq!(r["k"], 2);
        assert_eq!(r["agreement"], 1.0);
        assert_eq!(backend.calls.load(Ordering::SeqCst), 2);
        // Nothing left: the budget error surfaces.
        let err = session
            .query(
                &backend,
                limits(2, 2, 1_000_000),
                &ks("other", 3, json!({})),
                &sig,
            )
            .await
            .unwrap_err();
        assert!(err.contains("call budget exhausted"), "{err}");
        let err = session
            .query(
                &backend,
                limits(2, 9, 1_000_000),
                &ks("q", 99, json!({})),
                &sig,
            )
            .await
            .unwrap_err();
        assert!(err.contains("k must be"), "{err}");
    }
}
