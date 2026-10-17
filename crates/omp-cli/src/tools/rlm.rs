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
//! tokens. Design: `runs/omp-rust/DESIGN-RLM-CLM.md` §3.3.
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
        Self { max_depth: 2, max_calls: 64, max_tokens: 2_000_000 }
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
            std::env::var(key).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
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
}

impl CallRecord {
    fn to_json(&self) -> Value {
        json!({
            "n": self.n, "model": self.model, "resolved_model": self.resolved_model,
            "api": self.api, "provider": self.provider,
            "depth": self.depth, "cached": self.cached,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "prompt_chars": self.prompt_chars, "context_chars": self.context_chars,
            "millis": self.millis,
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
        *ledger = Ledger { context, ..Ledger::default() };
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
            "records": ledger.log.iter().map(CallRecord::to_json).collect::<Vec<_>>(),
        })
    }

    /// One budgeted, memoized sub-call. Errors are the exact text the cell
    /// sees (`llm_query: …`).
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

        let key = memo_key(model, &context, prompt);
        let full = render_prompt(prompt, &context);
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
                };
                let out = reply(&text, &record, &ledger);
                append_log(&record);
                ledger.log.push(record);
                return Ok(out);
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
        if usage.input == 0 && usage.output == 0 && usage.cache_read == 0 && usage.cache_write == 0 {
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
        };
        let out = reply(&text, &record, &ledger);
        append_log(&record);
        append_session_entry(&record, &usage);
        ledger.log.push(record);
        Ok(out)
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
    use std::io::Write as _;
    let Some(path) = std::env::var_os("OMP_RLM_LOG") else {
        return;
    };
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{}", record.to_json());
    }
}

/// Session-entry sink installed by the runtime ([`set_entry_sink`]): one
/// `model_usage` entry per provider call, so `omp stats` / bench can count
/// sub-call tokens straight from the session file.
static ENTRY_SINK: LazyLock<parking_lot::RwLock<Option<std::sync::Arc<dyn Fn(&str, Value) + Send + Sync>>>> =
    LazyLock::new(|| parking_lot::RwLock::new(None));

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
        return SESSION.query(backend.as_ref(), Limits::effective(), args, &ctx.signal).await;
    }
    SESSION.query(&LiveBackend, Limits::effective(), args, &ctx.signal).await
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
}

pub const USAGE: &str = "Usage: /rlm [--max-depth=N] [--max-calls=N] [--max-tokens=N] <question> @<file>\n       /rlm [flags] <question> -- <context text>\n\
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
        let parsed: u64 = value.parse().map_err(|_| format!("/rlm: invalid value for --{name}: {value}"))?;
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
    } else if let Some(at) = rest.rfind(" @").map(|i| i + 1).or_else(|| rest.starts_with('@').then_some(0)) {
        let path = rest[at + 1..].trim();
        if path.is_empty() || path.contains(char::is_whitespace) {
            return Err(USAGE.to_string());
        }
        (rest[..at].trim(), ContextSource::File(path.to_string()))
    } else {
        return Err(USAGE.to_string());
    };
    if question.is_empty() {
        return Err(USAGE.to_string());
    }
    Ok(RlmArgs { question: question.to_string(), context, max_depth: depth, max_calls: calls, max_tokens: tokens })
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
    let result = super::eval::eval(env, args, ctx).await.map_err(|e| format!("{e:#}"))?;
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
            Self { calls: AtomicU64::new(0), input, output }
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
        Limits { max_depth, max_calls, max_tokens }
    }

    #[tokio::test]
    async fn call_budget_exhaustion_is_a_clear_error() {
        let _guard = TEST_GUARD.lock().await;
        let (session, mock, sig) = (Session::default(), Mock::new(10, 5), AbortSignal::never());
        let lim = limits(2, 2, 1_000_000);
        for i in 0..2 {
            session.query(&mock, lim, &q(&format!("p{i}"), None), &sig).await.unwrap();
        }
        let err = session.query(&mock, lim, &q("p2", None), &sig).await.unwrap_err();
        assert!(err.contains("call budget exhausted") && err.contains("max_calls 2"), "{err}");
        assert_eq!(mock.calls.load(Ordering::SeqCst), 2, "no provider call after exhaustion");
    }

    #[tokio::test]
    async fn token_budget_exhaustion_blocks_before_the_provider_call() {
        let _guard = TEST_GUARD.lock().await;
        let (session, mock, sig) = (Session::default(), Mock::new(80, 30), AbortSignal::never());
        let lim = limits(2, 10, 100);
        session.query(&mock, lim, &q("a", None), &sig).await.unwrap(); // 110 tokens: budget now exceeded
        let err = session.query(&mock, lim, &q("b", None), &sig).await.unwrap_err();
        assert!(err.contains("token budget exhausted"), "{err}");
        assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
        // A huge prompt is refused up front from the estimate alone.
        let fresh = Session::default();
        let big = "x".repeat(4_000);
        let err = fresh.query(&mock, limits(2, 10, 100), &q("p", Some(&big)), &sig).await.unwrap_err();
        assert!(err.contains("token budget exhausted"), "{err}");
        assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn depth_budget_is_enforced() {
        let _guard = TEST_GUARD.lock().await;
        let (session, mock, sig) = (Session::default(), Mock::new(1, 1), AbortSignal::never());
        let mut args = q("p", None);
        args["depth"] = json!(3);
        let err = session.query(&mock, limits(2, 10, 1000), &args, &sig).await.unwrap_err();
        assert!(err.contains("depth budget exhausted"), "{err}");
        args["depth"] = json!(2);
        session.query(&mock, limits(2, 10, 1000), &args, &sig).await.unwrap();
    }

    #[tokio::test]
    async fn memo_hit_skips_provider_and_budget() {
        let _guard = TEST_GUARD.lock().await;
        let (session, mock, sig) = (Session::default(), Mock::new(40, 10), AbortSignal::never());
        let lim = limits(2, 1, 1_000_000);
        let first = session.query(&mock, lim, &q("what?", Some("ctx")), &sig).await.unwrap();
        // max_calls is 1 and spent: only a memo hit can succeed now.
        let second = session.query(&mock, lim, &q("what?", Some("ctx")), &sig).await.unwrap();
        assert_eq!(first["text"], second["text"]);
        assert_eq!(first["cached"], json!(false));
        assert_eq!(second["cached"], json!(true));
        assert_eq!(second["usage"], json!({"input": 0, "output": 0}));
        assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
        // Different context or model is a different key.
        assert!(session.query(&mock, lim, &q("what?", Some("other")), &sig).await.is_err());
        let mut other_model = q("what?", Some("ctx"));
        other_model["model"] = json!("slow");
        assert!(session.query(&mock, lim, &other_model, &sig).await.is_err());
    }

    #[tokio::test]
    async fn accounting_records_every_call() {
        let _guard = TEST_GUARD.lock().await;
        let (session, mock, sig) = (Session::default(), Mock::new(40, 10), AbortSignal::never());
        let lim = limits(2, 10, 1_000_000);
        session.query(&mock, lim, &q("a", Some("one")), &sig).await.unwrap();
        session.query(&mock, lim, &q("b", Some("two")), &sig).await.unwrap();
        session.query(&mock, lim, &q("a", Some("one")), &sig).await.unwrap(); // memo
        let stats = session.stats(lim);
        assert_eq!(stats["calls"], 2);
        assert_eq!(stats["memo_hits"], 1);
        assert_eq!(stats["input_tokens"], 80);
        assert_eq!(stats["output_tokens"], 20);
        assert_eq!(stats["total_tokens"], 100);
        let records = stats["records"].as_array().unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0]["model"], "smol", "default model is the smol role");
        assert_eq!(records[0]["context_chars"], 3);
        assert_eq!(records[2]["cached"], true);
        session.reset();
        assert_eq!(session.stats(lim)["calls"], 0);
    }

    #[tokio::test]
    async fn session_accounting_records_model_usage_entries() {
        let _guard = TEST_GUARD.lock().await;
        use std::sync::Arc;
        let entries: Arc<parking_lot::Mutex<Vec<(String, Value)>>> = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let collected = Arc::clone(&entries);
        set_entry_sink(Some(Arc::new(move |kind: &str, payload: Value| {
            collected.lock().push((kind.to_string(), payload));
        })));
        let (session, mock, sig) = (Session::default(), Mock::new(40, 10), AbortSignal::never());
        let lim = limits(2, 10, 1_000_000);
        session.query(&mock, lim, &q("a", Some("one")), &sig).await.unwrap();
        session.query(&mock, lim, &q("a", Some("one")), &sig).await.unwrap(); // memo: no entry
        set_entry_sink(None);
        let entries = entries.lock();
        assert_eq!(entries.len(), 1, "one model_usage entry per provider call; memo hits are free");
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
        assert_eq!((a.max_calls, a.max_tokens, a.max_depth), (Some(5), Some(900), None));
        assert!(parse_args("no context at all").is_err());
        assert!(parse_args("@only.txt").is_err());
        assert!(parse_args("--max-calls=x q @f").is_err());
    }
}
