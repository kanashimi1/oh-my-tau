//! The `judge` model role (`pi-coding-agent/src/judgment/index.ts`).
//!
//! Candidates = the `judge` role's model followed by `retry.fallbackChains.judge`,
//! restricted to credentialed models. From the first *native* candidate (an api
//! that answers System One judgments: `typesafe`, `openrouter-decisions`) on, only
//! native candidates remain - a prompted chat model never stands in for a failed
//! native judgment.
//!
//! This module also carries the System One HTTP client (`pi-ai/src/judgment/typesafe.ts`) and
//! the candidate fallback loop of `ChainJudge.withCandidate`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Map, Value};

use crate::{AgentDir, ResolvedModel};

/// How a judge candidate answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JudgeKind {
    /// System One wire format (`POST {base_url}{route}` with `{state, model, questions}`).
    Native,
    /// On-device keyword model.
    Local,
    /// Prompted chat model.
    Online,
}

/// One resolved judge backend.
#[derive(Debug, Clone)]
pub struct JudgeCandidate {
    pub kind: JudgeKind,
    pub model: ResolvedModel,
    /// Native judgments: full endpoint URL (`base_url` + route).
    pub url: Option<String>,
}

/// Route under the model's base URL, per native api.
#[must_use]
pub fn judgment_route(api: &str) -> Option<&'static str> {
    match api {
        "typesafe" => Some("/v1/systemone"),
        "openrouter-decisions" => Some("/decisions"),
        _ => None,
    }
}

fn classify(model: ResolvedModel) -> JudgeCandidate {
    if let Some(route) = judgment_route(&model.api) {
        let url = format!("{}{route}", model.base_url.trim_end_matches('/'));
        return JudgeCandidate { kind: JudgeKind::Native, model, url: Some(url) };
    }
    let kind = if model.api == "local-inference" { JudgeKind::Local } else { JudgeKind::Online };
    JudgeCandidate { kind, model, url: None }
}

/// The ordered judge chain. Empty candidates are an error (no judge configured / no credentials).
///
/// # Errors
/// No `judge` role, or no candidate resolves with credentials.
pub fn resolve_judge_chain(dir: &AgentDir) -> Result<Vec<JudgeCandidate>> {
    let primary = crate::resolve_in(dir, "judge")?;
    let mut selectors = vec![primary.selector.clone()];
    selectors.extend(primary.fallbacks.iter().cloned());
    let mut chain: Vec<JudgeCandidate> = Vec::new();
    let mut first_err = None;
    for (i, sel) in selectors.iter().enumerate() {
        let resolved = if i == 0 { Ok(primary.clone()) } else { crate::resolve_in(dir, sel) };
        match resolved {
            Ok(m) if m.api_key.is_some() || m.api == "local-inference" => chain.push(classify(m)),
            Ok(_) => {}
            Err(e) => first_err = first_err.or(Some(e)),
        }
    }
    if let Some(first_native) = chain.iter().position(|c| c.kind == JudgeKind::Native) {
        let mut i = 0;
        chain.retain(|c| {
            i += 1;
            i - 1 < first_native || c.kind == JudgeKind::Native
        });
    }
    if chain.is_empty() {
        return Err(first_err.unwrap_or_else(|| anyhow!("no credentialed model for the judge role")));
    }
    Ok(chain)
}

/// First candidate of [`resolve_judge_chain`].
///
/// # Errors
/// See [`resolve_judge_chain`].
pub fn resolve_judge(dir: &AgentDir) -> Result<JudgeCandidate> {
    Ok(resolve_judge_chain(dir)?.remove(0))
}

/// Per-attempt timeout (`DEFAULT_TIMEOUT_MS`).
const TIMEOUT: Duration = Duration::from_millis(10_000);
const MAX_ATTEMPTS: u32 = 3;
const BACKOFF_BASE_MS: u64 = 500;
const BACKOFF_MAX_MS: u64 = 5_000;

/// A typed judgment: `state` is a string, object or array; `questions` an object keyed by id
/// (`choice` / `noul` / `score` questions, forwarded verbatim).
#[derive(Debug, Clone)]
pub struct JudgeRequest {
    pub state: Value,
    pub questions: Map<String, Value>,
}

/// Token counts and billed USD (input-only pricing) reported by the endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct JudgeUsage {
    pub input: u64,
    pub output: u64,
    pub cost: f64,
}

/// Validated answers: one per question id, with the matching `type`.
#[derive(Debug, Clone)]
pub struct JudgeResult {
    pub api: String,
    pub provider: String,
    /// Model the server reports having used.
    pub model: String,
    pub answers: Map<String, Value>,
    pub usage: JudgeUsage,
}

/// Failure of one native judgment.
#[derive(Debug)]
struct JudgeError {
    message: String,
    status: Option<u16>,
    /// Timeout: the chain stops instead of trying the next candidate.
    timeout: bool,
}

impl std::fmt::Display for JudgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for JudgeError {}

fn backoff(attempt: u32, retry_after: Option<&reqwest::header::HeaderValue>) -> Duration {
    let hinted = retry_after
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0);
    if let Some(secs) = hinted {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        return Duration::from_millis(((secs * 1000.0) as u64).min(BACKOFF_MAX_MS));
    }
    Duration::from_millis((BACKOFF_BASE_MS << attempt).min(BACKOFF_MAX_MS))
}

fn label(c: &JudgeCandidate) -> String {
    format!("{}/{}", c.model.provider, c.model.id)
}

/// One System One judgment against a native candidate: `POST {url}` with
/// `{state, model, questions}`, bearer auth plus the model's headers, 10 s per attempt, up to 3
/// attempts on transport errors / 408 / 429 / 5xx (backoff 500 ms doubling, capped 5 s,
/// `Retry-After` honoured). Every question must come back with an answer of its own type.
///
/// # Errors
/// Non-native candidate, missing key, HTTP/transport failure, or an incomplete response.
pub async fn judge_native(candidate: &JudgeCandidate, request: &JudgeRequest) -> Result<JudgeResult> {
    judge_native_inner(candidate, request).await.map_err(anyhow::Error::new)
}

async fn judge_native_inner(
    candidate: &JudgeCandidate,
    request: &JudgeRequest,
) -> std::result::Result<JudgeResult, JudgeError> {
    let fail = |message: String| JudgeError { message, status: None, timeout: false };
    let (JudgeKind::Native, Some(url)) = (candidate.kind, candidate.url.as_deref()) else {
        return Err(fail(format!("{} is not a native judge", label(candidate))));
    };
    let key = candidate
        .model
        .api_key
        .as_deref()
        .ok_or_else(|| fail(format!("no API key for {}", label(candidate))))?;
    let body = json!({ "state": request.state, "model": candidate.model.id, "questions": request.questions });
    let client = reqwest::Client::builder().timeout(TIMEOUT).build().map_err(|e| fail(e.to_string()))?;
    let mut attempt = 0;
    let response: Value = loop {
        let mut req = client.post(url);
        for (k, v) in &candidate.model.headers {
            req = req.header(k, v);
        }
        req = req
            .bearer_auth(key)
            .header(reqwest::header::ACCEPT, "application/json")
            .json(&body);
        match req.send().await {
            Err(e) => {
                if attempt + 1 >= MAX_ATTEMPTS {
                    return Err(JudgeError {
                        message: format!("{} request failed: {e}", label(candidate)),
                        status: None,
                        timeout: e.is_timeout(),
                    });
                }
                tokio::time::sleep(backoff(attempt, None)).await;
            }
            Ok(resp) if resp.status().is_success() => {
                break resp.json().await.map_err(|e| fail(format!("{} bad response body: {e}", label(candidate))))?;
            }
            Ok(resp) => {
                let status = resp.status().as_u16();
                let retry_after = resp.headers().get(reqwest::header::RETRY_AFTER).cloned();
                let text = resp.text().await.unwrap_or_default();
                let transient = status == 408 || status == 429 || status >= 500;
                if !transient || attempt + 1 >= MAX_ATTEMPTS {
                    return Err(JudgeError {
                        message: format!("{} API error ({status}): {text}", label(candidate)),
                        status: Some(status),
                        timeout: false,
                    });
                }
                tokio::time::sleep(backoff(attempt, retry_after.as_ref())).await;
            }
        }
        attempt += 1;
    };
    let answers = response.get("answers").and_then(Value::as_object).cloned().unwrap_or_default();
    for (id, q) in &request.questions {
        let want = q.get("type").and_then(Value::as_str);
        if want.is_none() || answers.get(id).and_then(|a| a.get("type")).and_then(Value::as_str) != want {
            return Err(fail(format!(
                "{} response is missing a \"{}\" answer for question \"{id}\"",
                label(candidate),
                want.unwrap_or("?")
            )));
        }
    }
    let usage = response.get("usage");
    let count = |k: &str| usage.and_then(|u| u.get(k)).and_then(Value::as_u64).unwrap_or(0);
    let cost = usage
        .and_then(|u| u.get("cost"))
        .and_then(Value::as_f64)
        .filter(|c| c.is_finite())
        .unwrap_or(0.0);
    Ok(JudgeResult {
        api: candidate.model.api.clone(),
        provider: candidate.model.provider.clone(),
        model: response
            .get("model")
            .and_then(Value::as_str)
            .map_or_else(|| candidate.model.id.clone(), str::to_owned),
        answers,
        usage: JudgeUsage { input: count("input_tokens"), output: count("output_tokens"), cost },
    })
}

/// How long a candidate stays skipped after its account rejected a judgment (401/402/403).
const REJECTION_COOLDOWN: Duration = Duration::from_secs(5 * 60);

/// Skip-until instants keyed by candidate identity, process-wide (omp keeps them per registry).
static REJECTIONS: std::sync::Mutex<Option<HashMap<String, Instant>>> = std::sync::Mutex::new(None);

fn rejection_map<R>(f: impl FnOnce(&mut HashMap<String, Instant>) -> R) -> R {
    let mut guard = REJECTIONS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    f(guard.get_or_insert_with(HashMap::new))
}

/// `nativeJudge`: answers already cached for this state and model are served locally and only
/// the remaining questions reach the endpoint; fresh answers are recorded. A request with no
/// questions still makes one call.
async fn judge_candidate(
    candidate: &JudgeCandidate,
    request: &JudgeRequest,
    cache: Option<&JudgmentCache>,
) -> std::result::Result<JudgeResult, JudgeError> {
    let model_key = label(candidate);
    let cached = cache.map(|c| c.lookup(&model_key, request));
    let known = cached.as_ref().map(|c| &c.answers);
    let mut pending = Map::new();
    for (id, q) in &request.questions {
        if known.is_none_or(|k| !k.contains_key(id)) {
            pending.insert(id.clone(), q.clone());
        }
    }
    let mut fresh = None;
    if !pending.is_empty() || request.questions.is_empty() {
        let pending_request = JudgeRequest { state: request.state.clone(), questions: pending.clone() };
        let mut r = judge_native_inner(candidate, &pending_request).await?;
        if r.usage.cost == 0.0 {
            #[allow(clippy::cast_precision_loss)]
            let cost = (candidate.model.model.cost.input.as_f64() / 1_000_000.0) * r.usage.input as f64
                + (candidate.model.model.cost.output.as_f64() / 1_000_000.0) * r.usage.output as f64;
            r.usage.cost = cost;
        }
        if let (Some(cache), Some(cached)) = (cache, &cached) {
            cache.record(&model_key, cached, &pending, &r);
        }
        fresh = Some(r);
    }
    let mut answers = Map::new();
    for id in request.questions.keys() {
        let answer = known
            .and_then(|k| k.get(id))
            .or_else(|| fresh.as_ref().and_then(|f| f.answers.get(id)));
        if let Some(a) = answer {
            answers.insert(id.clone(), a.clone());
        }
    }
    Ok(JudgeResult {
        api: candidate.model.api.clone(),
        provider: candidate.model.provider.clone(),
        model: fresh.as_ref().map_or_else(|| candidate.model.id.clone(), |f| f.model.clone()),
        answers,
        usage: fresh.map(|f| f.usage).unwrap_or_default(),
    })
}

/// `ChainJudge.withCandidate`: try each candidate in order; any failure but a timeout falls
/// through to the next. A candidate whose account answered 401/402/403 is skipped for five
/// minutes. Non-native candidates are failures: `local-inference` needs omp's on-device worker
/// and prompted chat judgments need pa-ai's `disableReasoning` / forced `toolChoice`, which
/// `SimpleStreamOptions` lacks.
///
/// # Errors
/// Empty chain, a timeout, or every candidate failing.
pub async fn judge(
    chain: &[JudgeCandidate],
    request: &JudgeRequest,
    cache: Option<&JudgmentCache>,
) -> Result<JudgeResult> {
    if chain.is_empty() {
        bail!("judgment: no judge model available");
    }
    let mut last_failure = None;
    let mut last_unavailable = None;
    for candidate in chain {
        let identity = label(candidate);
        let skipped_until = rejection_map(|m| m.get(&identity).copied());
        if let Some(until) = skipped_until {
            if until > Instant::now() {
                last_unavailable = Some(format!("{identity} rejected the account recently"));
                continue;
            }
            rejection_map(|m| m.remove(&identity));
        }
        match judge_candidate(candidate, request, cache).await {
            Ok(r) => return Ok(r),
            Err(e) if e.timeout => return Err(anyhow!(e.message)),
            Err(e) => {
                if matches!(e.status, Some(401..=403)) {
                    rejection_map(|m| m.insert(identity, Instant::now() + REJECTION_COOLDOWN));
                }
                last_failure = Some(e.message);
            }
        }
    }
    Err(anyhow!(
        "judgment: every judge candidate failed: {}",
        last_failure.or(last_unavailable).unwrap_or_else(|| "unknown error".into())
    ))
}

const CACHE_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS states (id TEXT PRIMARY KEY, state TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS oracle (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state TEXT NOT NULL REFERENCES states(id),
    model TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    instruction TEXT NOT NULL,
    criteria TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE (state, model, name, type, instruction, criteria)
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state TEXT NOT NULL REFERENCES states(id),
    results TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    price REAL NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_state ON usage(state);
";

/// Answers already recorded for one request, plus the state identity [`JudgmentCache::record`] reuses.
#[derive(Debug, Clone)]
pub struct CachedAnswers {
    /// `sha256` hex of the canonical state JSON.
    pub state_id: String,
    /// Canonical state JSON.
    pub state: String,
    pub answers: Map<String, Value>,
}

/// Local answer cache for native judgments (`judgment/cache.ts`; same sqlite schema, so it shares
/// the database with omp). Best-effort: failures read as "no cached answers" / are ignored. An
/// answer is keyed by (state hash, model, question id, type, instructions, criteria); no TTL.
pub struct JudgmentCache {
    db: std::sync::Mutex<rusqlite::Connection>,
}

/// `$OMP_JUDGMENT_CACHE_DB`, else `<agent dir>/../cache/judgment-cache.db` (`~/.omp/cache/judgment-cache.db`).
#[must_use]
pub fn judgment_cache_path(dir: &AgentDir) -> PathBuf {
    if let Some(p) = std::env::var_os("OMP_JUDGMENT_CACHE_DB").filter(|p| !p.is_empty()) {
        return PathBuf::from(p);
    }
    dir.0.parent().unwrap_or(&dir.0).join("cache").join("judgment-cache.db")
}

/// `stableStringifyJson`: object keys sorted at every depth (array-index-like keys first,
/// ascending, as JS enumerates them), arrays in order, whole floats without `.0`.
#[must_use]
pub fn stable_json(v: &Value) -> String {
    fn write(v: &Value, out: &mut String) {
        match v {
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write(x, out);
                }
                out.push(']');
            }
            Value::Object(o) => {
                let index = |k: &str| k.parse::<u32>().ok().filter(|n| *n != u32::MAX && n.to_string() == k);
                let mut keys: Vec<&String> = o.keys().collect();
                keys.sort_by(|a, b| match (index(a), index(b)) {
                    (Some(x), Some(y)) => x.cmp(&y),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => a.encode_utf16().cmp(b.encode_utf16()),
                });
                out.push('{');
                for (i, k) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::String(k.clone()).to_string());
                    out.push(':');
                    write(&o[k], out);
                }
                out.push('}');
            }
            Value::Number(n) => match n.as_f64() {
                Some(f) if n.is_f64() && f.fract() == 0.0 && f.abs() < 1e21 => {
                    #[allow(clippy::cast_possible_truncation)]
                    out.push_str(&(f as i128).to_string());
                }
                _ => out.push_str(&n.to_string()),
            },
            other => out.push_str(&other.to_string()),
        }
    }
    let mut out = String::new();
    write(v, &mut out);
    out
}

fn unix_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

/// (name, type, instruction, criteria) of a question as stored in the `oracle` table.
fn question_key(name: &str, q: &Value) -> (String, String, String, String) {
    (
        name.to_owned(),
        q.get("type").and_then(Value::as_str).unwrap_or_default().to_owned(),
        q.get("instructions").and_then(Value::as_str).unwrap_or_default().to_owned(),
        stable_json(q.get("criteria").unwrap_or(&Value::Null)),
    )
}

impl JudgmentCache {
    /// Open (creating if needed) the cache at `path`.
    ///
    /// # Errors
    /// The directory or database cannot be created.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let db = rusqlite::Connection::open(path)?;
        db.pragma_update(None, "journal_mode", "WAL")?;
        db.pragma_update(None, "synchronous", "NORMAL")?;
        db.pragma_update(None, "foreign_keys", "ON")?;
        db.execute_batch(CACHE_SCHEMA)?;
        Ok(Self { db: std::sync::Mutex::new(db) })
    }

    /// Answers already recorded for `request`'s questions under `model` (`provider/id`).
    #[must_use]
    pub fn lookup(&self, model: &str, request: &JudgeRequest) -> CachedAnswers {
        let state = stable_json(&request.state);
        let state_id = sha256_hex(&state);
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut answers = Map::new();
        let found = (|| -> rusqlite::Result<()> {
            let mut stmt = db.prepare_cached(
                "SELECT answer FROM oracle WHERE state = ? AND model = ? AND name = ? AND type = ? AND instruction = ? AND criteria = ?",
            )?;
            for (name, q) in &request.questions {
                let (n, t, i, c) = question_key(name, q);
                let row: Option<String> = rusqlite::OptionalExtension::optional(
                    stmt.query_row(rusqlite::params![state_id, model, n, t, i, c], |r| r.get(0)),
                )?;
                if let Some(a) = row.and_then(|s| serde_json::from_str::<Value>(&s).ok()) {
                    answers.insert(name.clone(), a);
                }
            }
            Ok(())
        })();
        if found.is_err() {
            answers.clear();
        }
        CachedAnswers { state_id, state, answers }
    }

    /// Record one billed request: its state, one oracle row per answered question, and a usage row.
    pub fn record(&self, model: &str, cached: &CachedAnswers, questions: &Map<String, Value>, result: &JudgeResult) {
        let mut db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = unix_seconds();
        let _ = (|| -> rusqlite::Result<()> {
            let tx = db.transaction()?;
            tx.execute(
                "INSERT OR IGNORE INTO states (id, state, created_at) VALUES (?, ?, ?)",
                rusqlite::params![cached.state_id, cached.state, now],
            )?;
            let mut results = Vec::new();
            for (name, q) in questions {
                let (n, t, i, c) = question_key(name, q);
                let answer = result.answers.get(name).unwrap_or(&Value::Null).to_string();
                let id: i64 = tx.query_row(
                    "INSERT INTO oracle (state, model, name, type, instruction, criteria, answer, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                     ON CONFLICT (state, model, name, type, instruction, criteria)
                     DO UPDATE SET answer = excluded.answer, created_at = excluded.created_at
                     RETURNING id",
                    rusqlite::params![cached.state_id, model, n, t, i, c, answer, now],
                    |r| r.get(0),
                )?;
                results.push(id);
            }
            tx.execute(
                "INSERT INTO usage (state, results, provider, model, input_tokens, output_tokens, price, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rusqlite::params![
                    cached.state_id,
                    serde_json::to_string(&results).unwrap_or_default(),
                    result.provider,
                    result.model,
                    i64::try_from(result.usage.input).unwrap_or(i64::MAX),
                    i64::try_from(result.usage.output).unwrap_or(i64::MAX),
                    result.usage.cost,
                    now
                ],
            )?;
            tx.commit()
        })();
    }
}

fn sha256_hex(s: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(s.as_bytes()).iter().fold(String::with_capacity(64), |mut acc, b| {
        use std::fmt::Write;
        let _ = write!(acc, "{b:02x}");
        acc
    })
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    /// Serves one canned `(status, extra headers, body)` per connection; yields captured requests.
    async fn mock(responses: Vec<(u16, &'static str, String)>) -> (String, tokio::task::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let h = tokio::spawn(async move {
            let mut seen = Vec::new();
            for (status, extra, body) in responses {
                let (mut s, _) = listener.accept().await.unwrap();
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                loop {
                    let n = s.read(&mut tmp).await.unwrap();
                    buf.extend_from_slice(&tmp[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    if let Some(i) = text.find("\r\n\r\n") {
                        let len = text[..i]
                            .lines()
                            .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                            .unwrap_or(0);
                        if buf.len() >= i + 4 + len {
                            break;
                        }
                    }
                    if n == 0 {
                        break;
                    }
                }
                seen.push(String::from_utf8_lossy(&buf).to_string());
                let resp = format!(
                    "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\n{extra}content-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                s.write_all(resp.as_bytes()).await.unwrap();
            }
            seen
        });
        (format!("http://{addr}/decisions"), h)
    }

    fn cand(url: &str, id: &str) -> JudgeCandidate {
        let t = tempfile::tempdir().unwrap();
        std::fs::write(t.path().join("config.yml"), "{}\n").unwrap();
        let base = url.trim_end_matches("/decisions");
        std::fs::write(
            t.path().join("models.yml"),
            format!(
                "providers:\n  zz:\n    baseUrl: {base}\n    apiKey: sekret\n    api: openrouter-decisions\n    models:\n      - id: {id}\n"
            ),
        )
        .unwrap();
        let model = crate::resolve_in(&AgentDir(t.path().to_path_buf()), &format!("zz/{id}")).unwrap();
        classify(model)
    }

    fn request() -> JudgeRequest {
        let questions = json!({
            "c": {"type": "choice", "instructions": "pick", "criteria": {"a": null, "b": null}},
            "n": {"type": "noul", "instructions": "yes?"}
        });
        JudgeRequest { state: json!("hello"), questions: questions.as_object().unwrap().clone() }
    }

    fn ok_body() -> String {
        json!({
            "model": "srv-model",
            "answers": {
                "c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.1}, "confidence": 0.8},
                "n": {"type": "noul", "noul": 0.25}
            },
            "usage": {"input_tokens": 12, "output_tokens": 3, "cost": 0.0001}
        })
        .to_string()
    }

    #[tokio::test]
    async fn parses_answers_and_sends_wire_format() {
        let (url, h) = mock(vec![(200, "", ok_body())]).await;
        let r = judge_native(&cand(&url, "m1"), &request()).await.unwrap();
        assert_eq!(r.model, "srv-model");
        assert_eq!(r.answers["c"]["choice"], "a");
        assert_eq!(r.answers["n"]["noul"], 0.25);
        assert_eq!(r.usage, JudgeUsage { input: 12, output: 3, cost: 0.0001 });
        let seen = h.await.unwrap().remove(0);
        let lower = seen.to_ascii_lowercase();
        assert!(seen.starts_with("POST /decisions "));
        assert!(lower.contains("authorization: bearer sekret"));
        let body: Value = serde_json::from_str(seen.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body["model"], "m1");
        assert_eq!(body["state"], "hello");
        assert_eq!(body["questions"]["n"]["type"], "noul");
    }

    #[tokio::test]
    async fn answer_type_mismatch_is_an_error() {
        let body = json!({"model": "x", "answers": {"c": {"type": "noul", "noul": 0.5}, "n": {"type": "noul", "noul": 0.5}}, "usage": {}})
            .to_string();
        let (url, _h) = mock(vec![(200, "", body)]).await;
        let e = judge_native(&cand(&url, "m1"), &request()).await.unwrap_err();
        assert!(e.to_string().contains("missing a \"choice\" answer for question \"c\""), "{e}");
    }

    #[tokio::test]
    async fn chain_falls_back_after_hard_error_and_retries_transient() {
        let (bad, _h1) = mock(vec![(401, "", "{\"error\":\"nope\"}".into())]).await;
        let (good, h2) = mock(vec![(429, "retry-after: 0\r\n", "slow".into()), (200, "", ok_body())]).await;
        let chain = vec![cand(&bad, "fb-bad"), cand(&good, "fb-good")];
        let r = judge(&chain, &request(), None).await.unwrap();
        assert_eq!(r.model, "srv-model");
        assert_eq!(h2.await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn chain_reports_last_failure() {
        let (bad, _h) = mock(vec![(403, "", "denied".into())]).await;
        let e = judge(&[cand(&bad, "rej-403")], &request(), None).await.unwrap_err();
        let s = e.to_string();
        assert!(s.starts_with("judgment: every judge candidate failed:") && s.contains("(403)"), "{s}");
        assert!(judge(&[], &request(), None).await.is_err());
    }

    #[tokio::test]
    async fn rejected_account_is_skipped_for_the_cooldown() {
        let (bad, h) = mock(vec![(402, "", "no credits".into())]).await;
        let c = cand(&bad, "cooldown-402");
        let first = judge(std::slice::from_ref(&c), &request(), None).await.unwrap_err();
        assert!(first.to_string().contains("(402)"), "{first}");
        assert_eq!(h.await.unwrap().len(), 1);
        // The mock is gone: a second attempt would be a connection error, not this message.
        let second = judge(&[c], &request(), None).await.unwrap_err();
        assert!(second.to_string().ends_with("zz/cooldown-402 rejected the account recently"), "{second}");
    }

    #[tokio::test]
    async fn cache_serves_known_answers_and_sends_only_pending_questions() {
        let t = tempfile::tempdir().unwrap();
        let cache = JudgmentCache::open(&t.path().join("c/judgment.db")).unwrap();
        let extra = json!({"s": {"type": "score", "instructions": "how bad", "criteria": ["low", "high"]}});
        let mut wider = request();
        wider.questions.extend(extra.as_object().unwrap().clone());
        let score_body = json!({
            "model": "srv-model",
            "answers": {"s": {"type": "score", "score": 0.5, "probabilities": {"0": 0.5, "1": 0.5}, "confidence": 0.1}},
            "usage": {"input_tokens": 4, "output_tokens": 1}
        })
        .to_string();
        let (url, h) = mock(vec![(200, "", ok_body()), (200, "", score_body)]).await;
        let c = cand(&url, "cache-m");
        let first = judge(std::slice::from_ref(&c), &request(), Some(&cache)).await.unwrap();
        assert_eq!(first.usage.input, 12);
        // Same state + questions: served locally (the mock expects a second connection that never carries it).
        let again = judge(std::slice::from_ref(&c), &request(), Some(&cache)).await.unwrap();
        assert_eq!(again.answers["c"]["choice"], "a");
        assert_eq!(again.usage, JudgeUsage::default());
        // Superset: only the new question travels.
        let wide = judge(std::slice::from_ref(&c), &wider, Some(&cache)).await.unwrap();
        assert_eq!(wide.answers.len(), 3);
        assert_eq!(wide.answers["c"]["choice"], "a");
        assert_eq!(wide.answers["s"]["score"], 0.5);
        let seen = h.await.unwrap();
        assert_eq!(seen.len(), 2);
        let body: Value = serde_json::from_str(seen[1].split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body["questions"].as_object().unwrap().keys().collect::<Vec<_>>(), ["s"]);
        // Another state misses.
        let other = JudgeRequest { state: json!("different"), ..request() };
        assert!(cache.lookup("zz/cache-m", &other).answers.is_empty());
    }

    #[test]
    fn stable_json_sorts_keys_like_js() {
        let v = json!({"b": 1, "a": {"y": [1.0, 2.5], "x": null}, "10": 0, "9": 0});
        assert_eq!(stable_json(&v), r#"{"9":0,"10":0,"a":{"x":null,"y":[1,2.5]},"b":1}"#);
        assert_eq!(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }
}
