//! `omp-auth-broker serve` — Rust port of the TS `omp auth-broker serve` daemon
//! (`@oh-my-pi/pi-ai/src/auth-broker/server.ts` + `refresher.ts`, `pi-coding-agent`
//! `cli/auth-broker-cli.ts runServe`).
//!
//! A bearer-token REST vault over `agent.db`: omp clients pull credential snapshots (refresh
//! tokens redacted to `__remote__`), ask for forced OAuth refreshes, upload/disable
//! credentials, and read usage history; a background sweep re-mints OAuth rows that expire
//! within the refresh skew. Refreshes go through the omp-providers lease + compare-and-set path,
//! so the broker coexists with omp-rs processes refreshing the same `agent.db` locally.

mod store;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use bytes::Bytes;
use http_body_util::{combinators::BoxBody, BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::header::HeaderValue;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use omp_providers::auth::{force_refresh_row, ForceRefresh};
use serde_json::{json, Map, Value};
use tokio::sync::{mpsc, watch};

use store::{ClientReport, Entry, ObservedEntry, Store};

const DEFAULT_BIND: &str = "127.0.0.1:8765";
const REFRESH_SKEW_MS: i64 = 5 * 60_000;
const REFRESH_INTERVAL_MS: i64 = 60_000;
const STREAM_KEEPALIVE: Duration = Duration::from_secs(20);
const EXTERNAL_CHANGE_POLL: Duration = Duration::from_millis(250);
const MAX_SNAPSHOT_WAIT_MS: i64 = 30_000;
/// JS `Number.MAX_SAFE_INTEGER` (TS `DISABLED_NEXT_SWEEP_IN_MS`).
const DISABLED_NEXT_SWEEP_IN_MS: i64 = 9_007_199_254_740_991;
const CAPABILITIES_HEADER: &str = "OMP-Auth-Broker-Capabilities";
const CAPABILITY_CODEX_METER: &str = "codex-meter-block-scopes";
const CODEX_BLOCK_PROVIDER_KEY: &str = "openai-codex:oauth";
const DISABLE_REFRESHER_ENV: &str = "OMP_AUTH_BROKER_DISABLE_REFRESHER";
const VERSION: &str = env!("CARGO_PKG_VERSION");

type Body = BoxBody<Bytes, Infallible>;

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

/// One JSON log line on stdout (journald adds the wall clock).
fn log(level: &str, message: &str, fields: Value) {
    let mut line = Map::new();
    line.insert("ts".into(), json!(now_ms()));
    line.insert("level".into(), json!(level));
    line.insert("pid".into(), json!(std::process::id()));
    line.insert("message".into(), json!(message));
    if let Value::Object(extra) = fields {
        line.extend(extra);
    }
    println!("{}", Value::Object(line));
}

// ------------------------------------------------------------------ state

struct State {
    store: Store,
    tokens: Vec<String>,
    /// TS `CredentialPool.generation`: bumped once per observed store change.
    generation: watch::Sender<u64>,
    last_revision: Mutex<i64>,
    refresher_enabled: bool,
    next_sweep_at: AtomicI64,
}

impl State {
    /// TS `GenerationGate.#pollExternalChanges` / `pool.bump`: any `auth_change_revision` move
    /// (ours or a peer's) is one generation step.
    fn sync_generation(&self) {
        let Ok(revision) = self.store.revision() else {
            return;
        };
        let mut last = self
            .last_revision
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *last != revision {
            *last = revision;
            self.generation.send_modify(|g| *g += 1);
        }
    }

    fn generation(&self) -> u64 {
        *self.generation.borrow()
    }

    /// TS `resolveRefresherSchedule` → (wire, nextSweepAt).
    fn schedule(&self, now: i64) -> (Value, i64) {
        if !self.refresher_enabled {
            return (
                json!({"enabled": false, "intervalMs": 0, "skewMs": 0, "nextSweepInMs": DISABLED_NEXT_SWEEP_IN_MS}),
                DISABLED_NEXT_SWEEP_IN_MS,
            );
        }
        let next = self.next_sweep_at.load(Ordering::Relaxed);
        (
            json!({"enabled": true, "intervalMs": REFRESH_INTERVAL_MS, "skewMs": REFRESH_SKEW_MS,
                   "nextSweepInMs": (next - now).max(0)}),
            next,
        )
    }

    /// TS `buildSnapshot`.
    fn snapshot(&self, codex_meter_scopes: bool) -> Result<Map<String, Value>> {
        let now = now_ms();
        let entries = self.store.entries()?;
        let (wire, next_sweep_at) = self.schedule(now);
        let blocks = self.store.blocks(now)?;
        let credentials: Vec<Value> = entries
            .iter()
            .map(|e| {
                let mut out = e.wire();
                out.insert(
                    "rotatesInMs".into(),
                    rotates_in_ms(e, self.refresher_enabled, next_sweep_at, now),
                );
                let own: Vec<&store::Block> =
                    blocks.iter().filter(|b| b.credential_id == e.id).collect();
                let mut list = block_snapshots(&own, codex_meter_scopes);
                if !list.is_empty() {
                    list.sort_by(compare_blocks);
                    out.insert("blocks".into(), Value::Array(list));
                }
                Value::Object(out)
            })
            .collect();
        let mut body = Map::new();
        body.insert("generation".into(), json!(self.generation()));
        body.insert("generatedAt".into(), json!(now));
        body.insert("serverNowMs".into(), json!(now));
        body.insert("refresher".into(), wire);
        body.insert("credentials".into(), Value::Array(credentials));
        Ok(body)
    }
}

/// TS `computeRotatesInMs`.
fn rotates_in_ms(entry: &Entry, enabled: bool, next_sweep_at: i64, now: i64) -> Value {
    if !enabled || !entry.is_oauth() {
        return Value::Null;
    }
    let Some(expires) = entry.expires() else {
        return Value::Null;
    };
    let expires = expires as i64;
    let due_at = expires - REFRESH_SKEW_MS;
    let eligible_at = now.max(due_at);
    if due_at <= now && next_sweep_at <= now {
        return json!(0);
    }
    if next_sweep_at >= eligible_at {
        return json!((next_sweep_at - now).max(0));
    }
    let steps = (eligible_at - next_sweep_at + REFRESH_INTERVAL_MS - 1) / REFRESH_INTERVAL_MS;
    json!((next_sweep_at + steps * REFRESH_INTERVAL_MS - now).max(0))
}

/// TS `buildCredentialBlockGroups` for one credential: legacy clients get the Codex meter
/// scopes folded into one `shared` block.
fn block_snapshots(blocks: &[&store::Block], codex_meter_scopes: bool) -> Vec<Value> {
    let snap = |provider_key: &str, scope: &str, until: i64, updated: Option<i64>| {
        let mut m = Map::new();
        m.insert("providerKey".into(), json!(provider_key));
        m.insert("blockScope".into(), json!(scope));
        m.insert("blockedUntilMs".into(), json!(until));
        if let Some(u) = updated {
            m.insert("updatedAtMs".into(), json!(u));
        }
        Value::Object(m)
    };
    let mut out = Vec::new();
    let mut shared: Option<(i64, Option<i64>)> = None;
    for b in blocks {
        let legacy_fold = !codex_meter_scopes
            && b.provider_key == CODEX_BLOCK_PROVIDER_KEY
            && matches!(b.block_scope.as_str(), "chat" | "spark" | "shared");
        if !legacy_fold {
            out.push(snap(
                &b.provider_key,
                &b.block_scope,
                b.blocked_until_ms,
                Some(b.updated_at_ms),
            ));
            continue;
        }
        let (until, updated) = shared.unwrap_or((0, None));
        shared = Some((
            until.max(b.blocked_until_ms),
            Some(updated.map_or(b.updated_at_ms, |u| u.max(b.updated_at_ms))),
        ));
    }
    if let Some((until, updated)) = shared {
        out.push(snap(CODEX_BLOCK_PROVIDER_KEY, "shared", until, updated));
    }
    out
}

/// TS `compareCredentialBlockSnapshots`.
fn compare_blocks(a: &Value, b: &Value) -> std::cmp::Ordering {
    let s = |v: &Value, k: &str| {
        v.get(k)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let n = |v: &Value, k: &str| v.get(k).and_then(Value::as_i64).unwrap_or(0);
    s(a, "providerKey")
        .cmp(&s(b, "providerKey"))
        .then_with(|| s(a, "blockScope").cmp(&s(b, "blockScope")))
        .then_with(|| n(a, "blockedUntilMs").cmp(&n(b, "blockedUntilMs")))
        .then_with(|| n(a, "updatedAtMs").cmp(&n(b, "updatedAtMs")))
}

// ------------------------------------------------------------------ refresh

/// TS `isDefinitiveOAuthFailure` (`error/flags.ts isOAuthExpiry`).
fn is_definitive_oauth_failure(message: &str) -> bool {
    use std::sync::LazyLock;
    static DEFINITIVE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"(?i)invalid_grant|invalid_token|unauthorized_client|\brevoked\b|refresh[\s_]?token.*expired")
            .expect("static regex")
    });
    static TRANSIENT: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(
            r"(?i)timeout|network|fetch failed|ECONN(?:REFUSED|RESET)|ETIMEDOUT|EAI_AGAIN|socket hang up|\b(?:408|425|429|5\d{2})\b|rate.?limit|too many requests|temporar|unavailable|forbidden|permission_denied|cloudflare|captcha",
        )
        .expect("static regex")
    });
    static HTTP_AUTH: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"\b401\b").expect("static regex"));
    DEFINITIVE.is_match(message) || (HTTP_AUTH.is_match(message) && !TRANSIENT.is_match(message))
}

/// TS `oauth.refresh(id)` → the refreshed snapshot entry. A definitive provider failure
/// disables the row (TS `refreshStored` `disabledCause`).
async fn refresh_entry(state: &State, id: i64, auth_recovery: bool) -> Result<Map<String, Value>> {
    // The omp-providers refresh holds a (!Sync) rusqlite connection across awaits, so its future
    // is !Send: drive it on a blocking-pool thread instead of a runtime worker.
    let db = state.store.db.clone();
    let handle = tokio::runtime::Handle::current();
    let outcome = tokio::task::spawn_blocking(move || {
        handle.block_on(force_refresh_row(&db, id, auth_recovery))
    })
    .await
    .map_err(|e| anyhow!("refresh task failed: {e}"))
    .and_then(|r| r);
    state.sync_generation();
    match outcome {
        Ok(ForceRefresh::Refreshed | ForceRefresh::ReusedRecent) => {
            let entry = state
                .store
                .entry(id)?
                .ok_or_else(|| anyhow!("No credential with id={id}"))?;
            Ok(entry.wire())
        }
        Ok(ForceRefresh::Unsupported) => {
            let provider = state
                .store
                .entry(id)?
                .map(|e| e.provider)
                .unwrap_or_default();
            bail!("OAuth provider \"{provider}\" does not support token refresh")
        }
        Err(e) => {
            let message = format!("{e:#}");
            if is_definitive_oauth_failure(&message)
                && state
                    .store
                    .disable(id, &format!("oauth refresh failed: {message}"))?
            {
                state.sync_generation();
                log(
                    "warn",
                    "auth-broker refresh failed definitively",
                    json!({"id": id, "error": message}),
                );
            }
            Err(e)
        }
    }
}

/// TS `AuthBrokerRefresher.tick`: refresh every OAuth row expiring within the skew.
async fn refresher_tick(state: &State) {
    state.next_sweep_at.store(now_ms(), Ordering::Relaxed);
    let deadline = now_ms() + REFRESH_SKEW_MS;
    let targets: Vec<i64> = match state.store.entries() {
        Ok(entries) => entries
            .iter()
            .filter(|e| e.is_oauth() && e.expires().is_some_and(|x| x <= deadline as f64))
            .map(|e| e.id)
            .collect(),
        Err(e) => {
            log(
                "warn",
                "auth-broker refresher store read failed",
                json!({"error": format!("{e:#}")}),
            );
            Vec::new()
        }
    };
    let results =
        futures::future::join_all(targets.iter().map(|&id| refresh_entry(state, id, false))).await;
    for (id, result) in targets.iter().zip(results) {
        match result {
            Ok(entry) => log(
                "info",
                "auth-broker credential refreshed",
                json!({"id": id, "provider": entry.get("provider"), "source": "refresher",
                       "expires": entry.get("credential").and_then(|c| c.get("expires"))}),
            ),
            Err(e) => log(
                "debug",
                "auth-broker refresh failed",
                json!({"id": id, "error": format!("{e:#}")}),
            ),
        }
    }
    state
        .next_sweep_at
        .store(now_ms() + REFRESH_INTERVAL_MS, Ordering::Relaxed);
}

// ------------------------------------------------------------------ http helpers

fn full(bytes: impl Into<Bytes>) -> Body {
    Full::new(bytes.into()).boxed()
}

fn json_response(status: u16, body: &Value) -> Response<Body> {
    let mut res = Response::new(full(body.to_string()));
    *res.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    res.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    res
}

fn error(status: u16, message: &str) -> Response<Body> {
    json_response(status, &json!({"error": message}))
}

fn with_snapshot_headers(mut res: Response<Body>, generation: u64) -> Response<Body> {
    let h = res.headers_mut();
    if let Ok(v) = HeaderValue::from_str(&format!("\"{generation}\"")) {
        h.insert("etag", v);
    }
    h.insert("cache-control", HeaderValue::from_static("no-store"));
    h.insert("vary", HeaderValue::from_static(CAPABILITIES_HEADER));
    res
}

/// TS `parseGenerationTag`.
fn parse_generation_tag(header: Option<&str>) -> Option<u64> {
    let mut v = header?.trim();
    if let Some(rest) = v.strip_prefix("W/") {
        v = rest.trim();
    }
    if v.len() >= 2 && v.starts_with('"') && v.ends_with('"') {
        v = &v[1..v.len() - 1];
    }
    v.parse::<u64>().ok()
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                match std::str::from_utf8(&bytes[i + 1..i + 3])
                    .ok()
                    .and_then(|h| u8::from_str_radix(h, 16).ok())
                {
                    Some(b) => {
                        out.push(b);
                        i += 2;
                    }
                    None => out.push(b'%'),
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn query_param(query: Option<&str>, name: &str) -> Option<String> {
    query?.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        (percent_decode(k) == name).then(|| percent_decode(v))
    })
}

/// JS `Number.parseInt(raw, 10)`: leading integer prefix, else `None` (NaN).
fn parse_int_prefix(raw: &str) -> Option<i64> {
    let t = raw.trim_start();
    let (sign, digits) = match t.strip_prefix('-') {
        Some(rest) => (-1, rest),
        None => (1, t.strip_prefix('+').unwrap_or(t)),
    };
    let end = digits
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(digits.len());
    digits[..end].parse::<i64>().ok().map(|n| sign * n)
}

async fn read_json(
    req: Request<Incoming>,
    allow_empty: bool,
) -> std::result::Result<Value, Response<Body>> {
    let raw = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return Err(error(400, &format!("Invalid request body: {e}"))),
    };
    if raw.is_empty() {
        return if allow_empty {
            Ok(json!({}))
        } else {
            Err(error(400, "Request body required"))
        };
    }
    serde_json::from_slice(&raw).map_err(|e| error(400, &format!("Invalid JSON body: {e}")))
}

fn str_field<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::as_str)
}

/// TS `writableAuthCredentialSchema`.
fn validate_credential(c: &Value) -> std::result::Result<Map<String, Value>, String> {
    let Value::Object(map) = c else {
        return Err("credential must be an object".into());
    };
    match str_field(c, "type") {
        Some("oauth") => {
            let refresh = str_field(c, "refresh").ok_or("credential.refresh must be a string")?;
            if refresh == store::REMOTE_REFRESH_SENTINEL {
                return Err(format!(
                    "credential.refresh must be not equal to the remote sentinel ({})",
                    store::REMOTE_REFRESH_SENTINEL
                ));
            }
            if str_field(c, "access").is_none_or(str::is_empty) {
                return Err("credential.access must be a non-empty string".into());
            }
            if !c.get("expires").is_some_and(Value::is_number) {
                return Err("credential.expires must be a number".into());
            }
            Ok(map.clone())
        }
        Some("api_key") => {
            if str_field(c, "key").is_none_or(str::is_empty) {
                return Err("credential.key must be a non-empty string".into());
            }
            if let Some(extra) = map
                .keys()
                .find(|k| !matches!(k.as_str(), "type" | "key" | "source"))
            {
                return Err(format!("credential.{extra} must be removed"));
            }
            if map
                .get("source")
                .is_some_and(|s| s.as_str() != Some("login"))
            {
                return Err("credential.source must be \"login\"".into());
            }
            Ok(map.clone())
        }
        _ => Err("credential.type must be \"oauth\" or \"api_key\"".into()),
    }
}

fn client_report(v: &Value) -> std::result::Result<ClientReport, String> {
    let install_id = str_field(v, "installId")
        .ok_or("installId must be a string")?
        .to_string();
    let entries = v
        .get("entries")
        .and_then(Value::as_array)
        .ok_or("entries must be an array")?;
    let mut out = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let n = |k: &str| {
            e.get(k)
                .and_then(Value::as_f64)
                .ok_or(format!("entries[{i}].{k} must be a number"))
        };
        let s = |k: &str| {
            str_field(e, k)
                .map(str::to_string)
                .ok_or(format!("entries[{i}].{k} must be a string"))
        };
        out.push(ObservedEntry {
            at: n("at")? as i64,
            provider: s("provider")?,
            model: s("model")?,
            requests: n("requests")?,
            input: n("inputTokens")?,
            output: n("outputTokens")?,
            cache_read: n("cacheReadTokens")?,
            cache_write: n("cacheWriteTokens")?,
            cost: n("costUsd")?,
        });
    }
    Ok(ClientReport {
        install_id,
        hostname: str_field(v, "hostname").map(str::to_string),
        app: str_field(v, "app").map(str::to_string),
        entries: out,
    })
}

fn authorized(req: &Request<Incoming>, tokens: &[String]) -> bool {
    if tokens.is_empty() {
        return true;
    }
    let Some(header) = req
        .headers()
        .get("authorization")
        .and_then(|h| h.to_str().ok())
    else {
        return false;
    };
    let Some((scheme, rest)) = header.split_once(char::is_whitespace) else {
        return false;
    };
    scheme.eq_ignore_ascii_case("bearer") && tokens.iter().any(|t| t == rest.trim())
}

// ------------------------------------------------------------------ routes

async fn serve_snapshot(
    state: &State,
    req: &Request<Incoming>,
    codex: bool,
) -> Result<Response<Body>> {
    state.sync_generation();
    let current = state.generation();
    let client = parse_generation_tag(
        req.headers()
            .get("if-none-match")
            .and_then(|h| h.to_str().ok()),
    );
    let wait_ms = query_param(req.uri().query(), "wait")
        .and_then(|w| w.trim().parse::<f64>().ok())
        .filter(|w| w.is_finite())
        .map_or(0, |w| (w.trunc() as i64).clamp(0, MAX_SNAPSHOT_WAIT_MS));
    let Some(client) = client.filter(|c| *c == current && wait_ms > 0) else {
        let body = state.snapshot(codex)?;
        let generation = body
            .get("generation")
            .and_then(Value::as_u64)
            .unwrap_or(current);
        return Ok(with_snapshot_headers(
            json_response(200, &Value::Object(body)),
            generation,
        ));
    };
    let mut rx = state.generation.subscribe();
    let _ = tokio::time::timeout(Duration::from_millis(wait_ms as u64), async {
        while *rx.borrow_and_update() == client {
            if rx.changed().await.is_err() {
                break;
            }
        }
    })
    .await;
    state.sync_generation();
    let now = state.generation();
    if now != client {
        let body = state.snapshot(codex)?;
        return Ok(with_snapshot_headers(
            json_response(200, &Value::Object(body)),
            now,
        ));
    }
    let mut res = Response::new(full(Bytes::new()));
    *res.status_mut() = StatusCode::NOT_MODIFIED;
    Ok(with_snapshot_headers(res, now))
}

fn sse_event(event: &str, body: &Value) -> Bytes {
    Bytes::from(format!("event: {event}\ndata: {body}\n\n"))
}

/// TS `fingerprintEntry`.
fn fingerprint(entry: &Value) -> String {
    json!([
        entry.get("id"),
        entry.get("provider"),
        entry.get("identityKey"),
        entry.get("rotatesInMs"),
        entry.get("credential"),
        entry.get("blocks").cloned().unwrap_or_else(|| json!([]))
    ])
    .to_string()
}

/// TS `serveSnapshotStream`: one `snapshot` event, then `entry`/`removed` deltas per generation
/// bump, `: keepalive` comments in between.
fn serve_stream(state: Arc<State>, codex: bool) -> Result<Response<Body>> {
    state.sync_generation();
    let initial = state.snapshot(codex)?;
    let (tx, rx) = mpsc::channel::<Bytes>(64);
    let mut seen: Vec<(i64, String)> = initial
        .get("credentials")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .map(|e| {
                    (
                        e.get("id").and_then(Value::as_i64).unwrap_or(0),
                        fingerprint(e),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let mut first = Map::new();
    first.insert("kind".into(), json!("snapshot"));
    first.extend(initial);
    let first = sse_event("snapshot", &Value::Object(first));
    tokio::spawn(async move {
        if tx.send(first).await.is_err() {
            return;
        }
        let mut gen_rx = state.generation.subscribe();
        gen_rx.borrow_and_update();
        let mut keepalive = tokio::time::interval(STREAM_KEEPALIVE);
        keepalive.tick().await;
        loop {
            tokio::select! {
                _ = keepalive.tick() => {
                    if tx.send(Bytes::from_static(b": keepalive\n\n")).await.is_err() { return; }
                }
                changed = gen_rx.changed() => {
                    if changed.is_err() { return; }
                    gen_rx.borrow_and_update();
                    let Ok(snap) = state.snapshot(codex) else { continue };
                    let head = |kind: &str| {
                        let mut m = Map::new();
                        m.insert("kind".into(), json!(kind));
                        m.insert("generation".into(), snap["generation"].clone());
                        m.insert("serverNowMs".into(), snap["serverNowMs"].clone());
                        m.insert("refresher".into(), snap["refresher"].clone());
                        m
                    };
                    let list = snap.get("credentials").and_then(Value::as_array).cloned().unwrap_or_default();
                    let mut ids = Vec::new();
                    for entry in list {
                        let id = entry.get("id").and_then(Value::as_i64).unwrap_or(0);
                        ids.push(id);
                        let fp = fingerprint(&entry);
                        match seen.iter_mut().find(|(i, _)| *i == id) {
                            Some((_, old)) if *old == fp => continue,
                            Some((_, old)) => *old = fp,
                            None => seen.push((id, fp)),
                        }
                        let mut m = head("entry");
                        m.insert("entry".into(), entry);
                        if tx.send(sse_event("entry", &Value::Object(m))).await.is_err() { return; }
                    }
                    let removed: Vec<i64> = seen.iter().map(|(i, _)| *i).filter(|i| !ids.contains(i)).collect();
                    seen.retain(|(i, _)| ids.contains(i));
                    for id in removed {
                        let mut m = head("removed");
                        m.insert("id".into(), json!(id));
                        if tx.send(sse_event("removed", &Value::Object(m))).await.is_err() { return; }
                    }
                }
            }
        }
    });
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|b| (Ok::<_, Infallible>(Frame::data(b)), rx))
    });
    let mut res = Response::new(StreamBody::new(stream).boxed());
    let h = res.headers_mut();
    h.insert(
        "content-type",
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    h.insert("cache-control", HeaderValue::from_static("no-cache"));
    h.insert("connection", HeaderValue::from_static("keep-alive"));
    h.insert("x-accel-buffering", HeaderValue::from_static("no"));
    h.insert("vary", HeaderValue::from_static(CAPABILITIES_HEADER));
    Ok(res)
}

fn credential_route(path: &str) -> Option<(i64, &str)> {
    let rest = path.strip_prefix("/v1/credential/")?;
    let (id, action) = rest.split_once('/')?;
    if id.is_empty() || !id.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((id.parse().ok()?, action))
}

async fn route(state: Arc<State>, req: Request<Incoming>) -> Result<Response<Body>> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(str::to_string);
    if method == Method::GET && path == "/v1/healthz" {
        return Ok(json_response(200, &json!({"ok": true, "version": VERSION})));
    }
    if !authorized(&req, &state.tokens) {
        log(
            "info",
            "auth-broker request unauthorized",
            json!({"method": method.as_str(), "path": path}),
        );
        return Ok(error(401, "unauthorized"));
    }
    let codex = req
        .headers()
        .get(CAPABILITIES_HEADER)
        .and_then(|h| h.to_str().ok())
        .is_some_and(|caps| caps.split(',').any(|c| c.trim() == CAPABILITY_CODEX_METER));
    match (&method, path.as_str()) {
        (&Method::GET, "/v1/snapshot/stream") => return serve_stream(state, codex),
        (&Method::GET, "/v1/snapshot") => return serve_snapshot(&state, &req, codex).await,
        (&Method::GET, "/v1/usage") => {
            return Ok(error(501, "omp-auth-broker does not serve live usage reports; clients fetch provider usage directly"))
        }
        (&Method::GET, "/v1/usage/history") => {
            let since = query_param(query.as_deref(), "sinceMs").and_then(|s| parse_int_prefix(&s)).unwrap_or(0);
            let provider = query_param(query.as_deref(), "provider");
            let entries = state.store.usage_history(since, provider.as_deref());
            return Ok(json_response(200, &json!({"generatedAt": now_ms(), "entries": entries})));
        }
        (&Method::POST, "/v1/usage/observed") => {
            let body = match read_json(req, false).await {
                Ok(b) => b,
                Err(res) => return Ok(res),
            };
            let report = match client_report(&body) {
                Ok(r) => r,
                Err(e) => return Ok(error(400, &e)),
            };
            return Ok(match state.store.record_client(&report, now_ms()) {
                Ok(()) => json_response(200, &json!({"ok": true})),
                Err(e) => error(500, &format!("{e:#}")),
            });
        }
        (&Method::GET, "/v1/usage/clients") => {
            let since = query_param(query.as_deref(), "sinceMs").and_then(|s| parse_int_prefix(&s)).unwrap_or(0);
            let clients = state.store.client_summary(since)?;
            return Ok(json_response(200, &json!({"generatedAt": now_ms(), "clients": clients})));
        }
        // No usage cache to drop: live reports are not served (see `/v1/usage`).
        (&Method::POST, "/v1/usage/stale") => return Ok(json_response(200, &json!({"ok": true}))),
        (&Method::GET, "/v1/credentials/disabled") => {
            let provider = query_param(query.as_deref(), "provider");
            let disabled = state.store.disabled(provider.as_deref())?;
            return Ok(json_response(200, &json!({"generatedAt": now_ms(), "disabled": disabled})));
        }
        (&Method::POST, "/v1/credential") => {
            let body = match read_json(req, false).await {
                Ok(b) => b,
                Err(res) => return Ok(res),
            };
            let provider = str_field(&body, "provider").filter(|p| !p.is_empty());
            let Some(provider) = provider else { return Ok(error(400, "provider must be a non-empty string")) };
            let credential = match body.get("credential").map(validate_credential) {
                Some(Ok(c)) => c,
                Some(Err(e)) => return Ok(error(400, &e)),
                None => return Ok(error(400, "credential must be an object")),
            };
            return Ok(match state.store.upsert(provider, &credential) {
                Ok(entries) => {
                    state.sync_generation();
                    log("info", "auth-broker credential upserted", json!({"provider": provider, "providerTotal": entries.len()}));
                    let entries: Vec<Value> = entries.iter().map(|e| Value::Object(e.wire())).collect();
                    json_response(200, &json!({"entries": entries}))
                }
                Err(e) => error(500, &format!("{e:#}")),
            });
        }
        _ => {}
    }
    let Some((id, action)) = credential_route(&path) else {
        return Ok(error(404, &format!("No route: {method} {path}")));
    };
    match (&method, action) {
        (&Method::POST, "refresh") => {
            let recovery =
                query_param(query.as_deref(), "reason").as_deref() == Some("auth-recovery");
            Ok(match refresh_entry(&state, id, recovery).await {
                Ok(entry) => {
                    log(
                        "info",
                        "auth-broker credential refreshed",
                        json!({"id": id, "provider": entry.get("provider"),
                               "expires": entry.get("credential").and_then(|c| c.get("expires"))}),
                    );
                    json_response(200, &json!({"entry": entry}))
                }
                Err(e) => {
                    let message = format!("{e:#}");
                    error(
                        if message.contains("No credential with id") {
                            404
                        } else {
                            500
                        },
                        &message,
                    )
                }
            })
        }
        (&Method::POST, "disable") => {
            let body = match read_json(req, true).await {
                Ok(b) => b,
                Err(res) => return Ok(res),
            };
            if body.get("cause").is_some_and(|c| !c.is_string()) {
                return Ok(error(400, "cause must be a string"));
            }
            let cause = str_field(&body, "cause")
                .filter(|c| !c.is_empty())
                .unwrap_or("disabled via auth-broker");
            if !state.store.disable(id, cause)? {
                return Ok(error(404, &format!("No credential with id={id}")));
            }
            state.sync_generation();
            log(
                "info",
                "auth-broker credential disabled",
                json!({"id": id, "cause": cause}),
            );
            Ok(json_response(200, &json!({"ok": true})))
        }
        (&Method::POST, "block") => {
            let body = match read_json(req, false).await {
                Ok(b) => b,
                Err(res) => return Ok(res),
            };
            let (Some(key), Some(scope), Some(until)) = (
                str_field(&body, "providerKey").filter(|k| !k.is_empty()),
                str_field(&body, "blockScope"),
                body.get("blockedUntilMs").and_then(Value::as_f64),
            ) else {
                return Ok(error(
                    400,
                    "providerKey, blockScope and blockedUntilMs are required",
                ));
            };
            if state.store.entry(id)?.is_none() {
                return Ok(error(404, &format!("No credential with id={id}")));
            }
            state.store.block_upsert(id, key, scope, until as i64)?;
            state.sync_generation();
            Ok(json_response(200, &json!({"ok": true})))
        }
        (&Method::DELETE, "block") => {
            let body = match read_json(req, false).await {
                Ok(b) => b,
                Err(res) => return Ok(res),
            };
            let (Some(key), Some(scope)) = (
                str_field(&body, "providerKey").filter(|k| !k.is_empty()),
                str_field(&body, "blockScope"),
            ) else {
                return Ok(error(400, "providerKey and blockScope are required"));
            };
            if state.store.entry(id)?.is_none() {
                return Ok(error(404, &format!("No credential with id={id}")));
            }
            state.store.block_delete(id, key, scope)?;
            state.sync_generation();
            Ok(json_response(200, &json!({"ok": true})))
        }
        (&Method::DELETE, "blocks") => {
            if state.store.entry(id)?.is_none() {
                return Ok(error(404, &format!("No credential with id={id}")));
            }
            state.store.blocks_delete_all(id)?;
            state.sync_generation();
            Ok(json_response(200, &json!({"ok": true})))
        }
        _ => Ok(error(404, &format!("No route: {method} {path}"))),
    }
}

// ------------------------------------------------------------------ main

fn config_root() -> PathBuf {
    let agent = omp_providers::AgentDir::default().0;
    agent
        .parent()
        .map_or_else(|| agent.clone(), std::path::Path::to_path_buf)
}

/// TS `ensureToken`: the existing `<config root>/auth-broker.token`, else a fresh 0600 one.
fn ensure_token(path: &std::path::Path) -> Result<String> {
    if let Ok(raw) = std::fs::read_to_string(path) {
        let t = raw.trim();
        if !t.is_empty() {
            return Ok(t.to_string());
        }
    }
    let mut buf = [0u8; 32];
    std::io::Read::read_exact(&mut std::fs::File::open("/dev/urandom")?, &mut buf)?;
    let token: String = buf.iter().map(|b| format!("{b:02x}")).collect();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, format!("{token}\n"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(token)
}

const HELP: &str =
    "omp-auth-broker — credential vault daemon (Rust port of `omp auth-broker serve`)

Usage: omp-auth-broker serve [--bind=HOST:PORT]

  --bind   listen address (default 127.0.0.1:8765)

Files: <config root>/auth-broker.token (bearer token, created 0600 when missing),
       <agent dir>/agent.db (credential store). Agent dir: $OMP_AGENT_DIR or ~/.omp/agent.
Env:   OMP_AUTH_BROKER_DISABLE_REFRESHER=1  no background OAuth refresh sweep
       HTTPS_PROXY / HTTP_PROXY / NO_PROXY  egress for provider token endpoints
       KIMI_CODE_OAUTH_HOST                 kimi-code token endpoint base";

fn parse_args() -> Result<String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let help = args.iter().any(|a| a == "-h" || a == "--help");
    if help || args.first().map(String::as_str) != Some("serve") {
        println!("{HELP}");
        std::process::exit(if help { 0 } else { 2 });
    }
    let mut bind = DEFAULT_BIND.to_string();
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if let Some(v) = a.strip_prefix("--bind=") {
            bind = v.to_string();
        } else if a == "--bind" {
            bind = it.next().context("--bind needs a value")?.clone();
        } else {
            bail!("unknown argument: {a}");
        }
    }
    Ok(bind)
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let bind = parse_args()?;
    let addr: SocketAddr = bind
        .parse()
        .with_context(|| format!("invalid --bind {bind}"))?;
    let token_path = config_root().join("auth-broker.token");
    let token = ensure_token(&token_path)?;
    let db = omp_providers::AgentDir::default().0.join("agent.db");
    let store = Store::open(&db)?;
    let refresher_enabled =
        std::env::var(DISABLE_REFRESHER_ENV).map_or(true, |v| v.is_empty() || v == "0");
    let state = Arc::new(State {
        last_revision: Mutex::new(store.revision().unwrap_or(0)),
        store,
        tokens: vec![token],
        generation: watch::channel(1).0,
        refresher_enabled,
        next_sweep_at: AtomicI64::new(now_ms()),
    });
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    log(
        "info",
        "auth-broker listening",
        json!({"url": format!("http://{addr}"), "db": db.display().to_string()}),
    );
    log(
        "info",
        "auth-broker bearer token loaded",
        json!({"path": token_path.display().to_string(), "mode": "0600"}),
    );

    let poll = state.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(EXTERNAL_CHANGE_POLL);
        loop {
            tick.tick().await;
            poll.sync_generation();
        }
    });
    if refresher_enabled {
        let r = state.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_millis(REFRESH_INTERVAL_MS as u64));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                refresher_tick(&r).await;
            }
        });
    }

    #[cfg(unix)]
    let shutdown = async {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => "SIGINT",
            _ = term.recv() => "SIGTERM",
        }
    };
    #[cfg(not(unix))]
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        "SIGINT"
    };
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            signal = &mut shutdown => {
                log("info", "auth-broker shutting down", json!({"signal": signal}));
                return Ok(());
            }
            accepted = listener.accept() => {
                let Ok((stream, _)) = accepted else { continue };
                let state = state.clone();
                tokio::spawn(async move {
                    let service = service_fn(move |req| {
                        let state = state.clone();
                        async move {
                            let method = req.method().to_string();
                            let path = req.uri().path().to_string();
                            Ok::<_, Infallible>(match route(state, req).await {
                                Ok(res) => res,
                                Err(e) => {
                                    log("error", "auth-broker request failed",
                                        json!({"method": method, "path": path, "error": format!("{e:#}")}));
                                    error(500, "internal error")
                                }
                            })
                        }
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .keep_alive(true)
                        .serve_connection(TokioIo::new(stream), service)
                        .await;
                });
            }
        }
    }
}
