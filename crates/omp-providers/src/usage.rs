//! OAuth/API-key rate-limit usage polling and reset credits — a Rust port of the
//! TS `packages/ai/src/usage/` providers. Covers anthropic (`claude.ts` +
//! `claude-reset.ts`), kimi-code (`kimi.ts`), zai (`zai.ts`), openai-codex
//! (`openai-codex.ts` + `openai-codex-reset.ts`), google-antigravity and
//! google-gemini-cli. Each provider fills a unified [`UsageReport`] of
//! [`UsageLine`]s; only anthropic and openai-codex expose saved reset credits.

use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::Value;

/// `user-agent` Claude's account routes gate on (TS `getClaudeCodeUserAgent`).
pub const CLAUDE_CLI_VERSION: &str = "2.1.281";
/// OAuth protocol beta required by Claude's account routes.
pub const CLAUDE_OAUTH_BETA: &str = "claude-code-20250219,oauth-2025-04-20";
/// Canonical subscription usage/profile endpoint root.
pub const DEFAULT_CLAUDE_OAUTH_BASE_URL: &str = "https://api.anthropic.com/api/oauth";

const KIMI_DEFAULT_BASE_URL: &str = "https://api.kimi.com/coding/v1";
const ZAI_DEFAULT_ENDPOINT: &str = "https://api.z.ai";
const CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
const ANTIGRAVITY_DEFAULT_ENDPOINT: &str = "https://daily-cloudcode-pa.googleapis.com";
const ANTIGRAVITY_SANDBOX_ENDPOINT: &str = "https://daily-cloudcode-pa.sandbox.googleapis.com";
const GEMINI_DEFAULT_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";

const CEDAR_PROGRAM: &str = "cedar_ember";
const JUNIPER_PROGRAM: &str = "juniper_tide";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

// ---------------------------------------------------------------------------
// Unified report
// ---------------------------------------------------------------------------

/// One rendered usage row: a window, a per-model limit, or a quota counter.
#[derive(Debug, Clone, Default)]
pub struct UsageLine {
    pub label: String,
    /// 0..=100.
    pub used_percent: Option<f64>,
    /// Provider-specific remaining units (tokens, requests, credits).
    pub remaining: Option<f64>,
    /// Epoch milliseconds.
    pub resets_at: Option<i64>,
    /// Suffix note such as `(active)` or a plan/tier tag.
    pub note: Option<String>,
}

/// A provider usage report, provider-agnostic.
#[derive(Debug, Clone)]
pub struct UsageReport {
    pub provider: String,
    pub fetched_at_ms: i64,
    pub lines: Vec<UsageLine>,
    /// Account-level notes: plan/tier, extra usage, spend, credit overage...
    pub notes: Vec<String>,
}

/// Credentials handed to [`fetch_provider_usage`]. `data` is the stored
/// credential row JSON (orgId/projectId/accountId/email/expires live there).
#[derive(Debug, Clone, Copy, Default)]
pub struct UsageCredential<'a> {
    /// OAuth access token (anthropic, kimi-code, openai-codex, google-*).
    pub access_token: Option<&'a str>,
    /// Raw API key used verbatim (zai). For oauth-capable providers this is the
    /// same value as `access_token`.
    pub raw_key: Option<&'a str>,
    /// Parsed `auth_credentials.data` of the resolved row.
    pub data: Option<&'a Value>,
}

impl UsageCredential<'_> {
    fn data_str(&self, key: &str) -> Option<&str> {
        self.data.and_then(|d| d.get(key)).and_then(Value::as_str)
    }

    /// True when the stored oauth token is already expired (callers refresh first).
    fn token_expired(&self) -> bool {
        self.data
            .and_then(|d| d.get("expires"))
            .and_then(|e| e.as_i64().or_else(|| e.as_f64().map(|f| f as i64)))
            .is_some_and(|expires| expires <= now_ms())
    }
}

/// Dispatch on the provider id; `Err` carries a human-readable reason.
pub async fn fetch_provider_usage(
    provider: &str,
    cred: &UsageCredential<'_>,
    base_url: Option<&str>,
) -> Result<UsageReport> {
    match provider {
        "anthropic" => fetch_anthropic_usage(cred, base_url).await,
        "kimi-code" => fetch_kimi_usage(cred, base_url).await,
        "zai" => fetch_zai_usage(cred, base_url).await,
        "openai-codex" => fetch_codex_usage(cred, base_url).await,
        "google-antigravity" => fetch_antigravity_usage(cred, base_url).await,
        "google-gemini-cli" => fetch_gemini_usage(cred, base_url).await,
        other => bail!("no usage endpoint implemented for provider `{other}`"),
    }
}

/// Providers with saved rate-limit reset credits.
pub fn supports_reset_credits(provider: &str) -> bool {
    matches!(provider, "anthropic" | "openai-codex")
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

fn as_f64(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str()?.parse().ok())
}

fn as_i64(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))
}

/// Seconds-or-millis epoch normalization (provider payloads mix both).
fn epoch_ms(v: Option<&Value>) -> Option<i64> {
    let n = v.and_then(as_i64)?;
    Some(if n > 1_000_000_000_000 { n } else { n * 1000 })
}

fn parse_iso_ms(s: &str) -> Option<i64> {
    // chrono-free ISO-8601 parse for the formats the APIs emit.
    let s = s.trim();
    let date = s.get(..10)?;
    let (y, mo, d): (i64, i64, i64) = (
        date.get(..4)?.parse().ok()?,
        date.get(5..7)?.parse().ok()?,
        date.get(8..10)?.parse().ok()?,
    );
    let time_end = s[11..]
        .find(|c: char| c == '+' || c == 'Z' || c == '-')
        .map(|i| 11 + i)
        .unwrap_or(s.len());
    let time = s.get(11..time_end)?;
    let h: i64 = time.get(..2)?.parse().ok()?;
    let mi: i64 = time.get(3..5)?.parse().ok()?;
    let sec: i64 = time.get(6..8).and_then(|v| v.parse().ok()).unwrap_or(0);
    // days-from-civil (Howard Hinnant's algorithm).
    let y = if mo <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if mo > 2 { mo - 3 } else { mo + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Some(days * 86_400_000 + h * 3_600_000 + mi * 60_000 + sec * 1000)
}

fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("usage http client")
}

fn header_map(pairs: &[(&'static str, String)]) -> reqwest::header::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    for (name, value) in pairs {
        headers.insert(
            reqwest::header::HeaderName::from_bytes(name.as_bytes()).expect("valid header name"),
            value.parse().expect("valid header value"),
        );
    }
    headers
}

fn bearer_headers(token: &str, extra: &[(&'static str, String)]) -> reqwest::header::HeaderMap {
    let mut pairs = vec![
        ("authorization", format!("Bearer {token}")),
        ("accept", "application/json".to_string()),
    ];
    pairs.extend_from_slice(extra);
    header_map(&pairs)
}

async fn get_json(
    client: &reqwest::Client,
    url: &str,
    headers: reqwest::header::HeaderMap,
) -> Result<Value> {
    let response = client.get(url).headers(headers).send().await?;
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("{status}: {}", text.chars().take(200).collect::<String>());
    }
    serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("malformed JSON: {e}"))
}

async fn post_json(
    client: &reqwest::Client,
    url: &str,
    headers: reqwest::header::HeaderMap,
    body: Value,
) -> Result<Value> {
    let response = client.post(url).headers(headers).json(&body).send().await?;
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("{status}: {}", text.chars().take(200).collect::<String>());
    }
    serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("malformed JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Anthropic (TS usage/claude.ts + claude-reset.ts)
// ---------------------------------------------------------------------------

fn claude_headers(token: &str) -> reqwest::header::HeaderMap {
    header_map(&[
        ("authorization", format!("Bearer {token}")),
        ("anthropic-beta", CLAUDE_OAUTH_BETA.to_string()),
        ("anthropic-version", "2023-06-01".to_string()),
        ("user-agent", format!("claude-cli/{CLAUDE_CLI_VERSION}")),
        ("accept", "application/json".to_string()),
        ("x-app", "cli".to_string()),
    ])
}

/// Candidate OAuth base URLs: the configured one (normalized), then the canonical
/// default — TS `claudeOAuthBaseUrls`.
pub fn oauth_base_urls(configured: Option<&str>) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(raw) = configured.map(str::trim).filter(|s| !s.is_empty()) {
        if let Ok(mut url) = reqwest::Url::parse(raw) {
            let mut path = url.path().trim_end_matches('/').to_string();
            for suffix in ["/api/oauth", "/v1"] {
                if path.to_lowercase().ends_with(suffix) {
                    path = path[..path.len() - suffix.len()].to_string();
                    break;
                }
            }
            url.set_path(&path);
            url.set_query(None);
            url.set_fragment(None);
            out.push(format!("{url}/api/oauth"));
        }
    }
    if out.iter().all(|u| u != DEFAULT_CLAUDE_OAUTH_BASE_URL) {
        out.push(DEFAULT_CLAUDE_OAUTH_BASE_URL.to_string());
    }
    out
}

fn claude_api_root(base: &str) -> String {
    base.trim_end_matches("/api/oauth").to_string()
}

async fn fetch_anthropic_usage(
    cred: &UsageCredential<'_>,
    base_url: Option<&str>,
) -> Result<UsageReport> {
    let token = cred.access_token.filter(|t| t.contains("sk-ant-oat"));
    let Some(token) = token else {
        bail!("no anthropic oauth access token");
    };
    if cred.token_expired() {
        bail!("anthropic oauth token expired (refresh first)");
    }
    let client = client()?;
    let mut last_err = anyhow::anyhow!("no oauth base url");
    for base in oauth_base_urls(base_url) {
        match get_json(&client, &format!("{base}/usage"), claude_headers(token)).await {
            Ok(payload) => {
                if !anthropic_has_usage_data(&payload) {
                    last_err = anyhow::anyhow!("usage endpoint returned no usage data");
                    continue;
                }
                return Ok(parse_anthropic_report(&payload, token));
            }
            Err(e) => {
                if e.to_string().starts_with("429") {
                    bail!("usage endpoint rate-limited (429); retry in a minute");
                }
                last_err = e;
            }
        }
    }
    Err(last_err)
}

fn anthropic_has_usage_data(payload: &Value) -> bool {
    [
        "five_hour",
        "seven_day",
        "limits",
        "extra_usage",
        "spend",
        CEDAR_PROGRAM,
        JUNIPER_PROGRAM,
    ]
    .iter()
    .any(|k| payload.get(*k).is_some())
}

fn anthropic_bucket(v: Option<&Value>) -> Option<UsageBucket> {
    let v = v?.as_object()?;
    let utilization = v.get("utilization").and_then(as_f64);
    let resets_at = v
        .get("resets_at")
        .and_then(Value::as_str)
        .and_then(parse_iso_ms);
    (utilization.is_some() || resets_at.is_some()).then_some(UsageBucket {
        utilization,
        resets_at,
    })
}

#[derive(Debug, Clone, Default)]
struct UsageBucket {
    utilization: Option<f64>,
    resets_at: Option<i64>,
}

fn parse_anthropic_report(payload: &Value, _token: &str) -> UsageReport {
    let mut lines = Vec::new();
    for (key, label) in [
        ("five_hour", "5h"),
        ("seven_day", "7d"),
        ("seven_day_opus", "7d opus"),
        ("seven_day_sonnet", "7d sonnet"),
    ] {
        if let Some(b) = anthropic_bucket(payload.get(key)) {
            lines.push(UsageLine {
                label: label.to_string(),
                used_percent: b.utilization,
                remaining: None,
                resets_at: b.resets_at,
                note: None,
            });
        }
    }
    if let Some(entries) = payload.get("limits").and_then(Value::as_array) {
        for e in entries {
            let kind = e
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("limit")
                .to_string();
            let display = e
                .pointer("/scope/model/display_name")
                .and_then(Value::as_str)
                .map(str::to_string);
            let is_active = e.get("is_active").and_then(Value::as_bool).unwrap_or(false);
            let mut label = kind;
            if let Some(display) = display {
                label = format!("{label} {display}");
            }
            lines.push(UsageLine {
                label,
                used_percent: e.get("percent").and_then(as_f64),
                remaining: None,
                resets_at: e
                    .get("resets_at")
                    .and_then(Value::as_str)
                    .and_then(parse_iso_ms),
                note: is_active.then_some("active".to_string()),
            });
        }
    }
    let mut notes = Vec::new();
    if let Some(extra) = payload.get("extra_usage").and_then(|e| e.as_object()) {
        if extra
            .get("is_enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let used = extra
                .get("used_credits")
                .and_then(as_f64)
                .map(|u| u.to_string())
                .unwrap_or_else(|| "?".into());
            let limit = extra
                .get("monthly_limit")
                .and_then(as_f64)
                .map(|u| u.to_string())
                .unwrap_or_else(|| "?".into());
            let currency = extra.get("currency").and_then(Value::as_str).unwrap_or("");
            notes.push(format!("extra usage: {used}/{limit} credits ({currency})"));
        }
    }
    if let Some(spend) = payload.get("spend").and_then(|s| s.as_object()) {
        if spend
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let minor = |k: &str| {
                spend
                    .get(k)
                    .and_then(|m| m.as_object())
                    .and_then(|m| m.get("amount_minor"))
                    .and_then(as_i64)
            };
            let used = minor("used")
                .map(|u| format!("{:.2}", u as f64 / 100.0))
                .unwrap_or_else(|| "?".into());
            let limit = minor("limit")
                .map(|u| format!("{:.2}", u as f64 / 100.0))
                .unwrap_or_else(|| "?".into());
            let currency = spend
                .get("used")
                .and_then(|m| m.as_object())
                .and_then(|m| m.get("currency"))
                .and_then(Value::as_str)
                .unwrap_or("");
            notes.push(format!("spend: {used}/{limit} {currency}"));
        }
    }
    UsageReport {
        provider: "anthropic".to_string(),
        fetched_at_ms: now_ms(),
        lines,
        notes,
    }
}

/// A saved rate-limit reset credit (`cedar_ember` grant or `juniper_tide` reset).
#[derive(Debug, Clone)]
pub struct ResetCredit {
    pub id: String,
    /// `cedar_ember` or `juniper_tide`.
    pub program: String,
    pub usable: bool,
    pub remaining_count: Option<i64>,
    /// Window ids the credit clears (e.g. `["anthropic:5h"]`).
    pub clears: Vec<String>,
    pub expires_at_ms: Option<i64>,
    pub next_available_at_ms: Option<i64>,
    pub weekly_resets_at_ms: Option<i64>,
    pub ineligible_reason: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ResetCreditList {
    pub credits: Vec<ResetCredit>,
    pub available_count: usize,
    /// The credit the server selected as next.
    pub next_credit_id: Option<String>,
    pub eligible: bool,
}

/// Outcome of consuming a reset credit.
#[derive(Debug, Clone)]
pub struct ConsumeOutcome {
    pub ok: bool,
    /// `reset`, `already_redeemed`, `nothing_to_reset`, or an error code
    /// (`auth_error`, `rate_limited`, `http_*`, `network_error`, ...).
    pub code: String,
    pub reason: Option<String>,
    pub cleared: Vec<String>,
    pub http_status: u16,
}

fn parse_window_list(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn parse_cedar_credits(block: &Value) -> ResetCreditList {
    let mut list = ResetCreditList::default();
    let now = now_ms();
    if let Some(grants) = block.get("grants").and_then(Value::as_array) {
        for g in grants {
            let id = g
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if id.is_empty() {
                continue;
            }
            let remaining = g.get("remaining_count").and_then(as_i64);
            list.credits.push(ResetCredit {
                id,
                program: CEDAR_PROGRAM.to_string(),
                usable: remaining.is_some_and(|r| r > 0),
                remaining_count: remaining,
                clears: parse_window_list(g.get("clears")),
                expires_at_ms: g
                    .get("ends_at")
                    .and_then(Value::as_str)
                    .and_then(parse_iso_ms),
                next_available_at_ms: None,
                weekly_resets_at_ms: None,
                ineligible_reason: None,
            });
        }
    }
    list.available_count = list
        .credits
        .iter()
        .filter(|c| c.usable && c.expires_at_ms.is_none_or(|e| e > now))
        .count();
    list.next_credit_id = block
        .get("next_grant_id")
        .and_then(Value::as_str)
        .map(str::to_string);
    if let Some(next) = list.next_credit_id.clone() {
        if !list.credits.iter().any(|c| c.id == next) {
            list.credits.push(ResetCredit {
                id: next,
                program: CEDAR_PROGRAM.to_string(),
                usable: list.available_count > 0,
                remaining_count: None,
                clears: parse_window_list(block.get("clears")),
                expires_at_ms: None,
                next_available_at_ms: None,
                weekly_resets_at_ms: None,
                ineligible_reason: None,
            });
        }
    }
    list.eligible = list.next_credit_id.is_some() || list.available_count > 0;
    list
}

fn parse_juniper_credits(block: &Value) -> ResetCreditList {
    let now = now_ms();
    let next_available_at_ms = block
        .get("next_available_at")
        .and_then(Value::as_str)
        .and_then(parse_iso_ms);
    let weekly_resets_at_ms = block
        .get("weekly_resets_at")
        .and_then(Value::as_str)
        .and_then(parse_iso_ms);
    let cooldown_until_ms = block
        .get("cooldown_until")
        .and_then(Value::as_str)
        .and_then(parse_iso_ms);
    let available = block
        .get("available")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && cooldown_until_ms.is_none_or(|c| c <= now);
    let ineligible_reason = block
        .get("ineligible_reason")
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut list = ResetCreditList::default();
    list.credits.push(ResetCredit {
        id: JUNIPER_PROGRAM.to_string(),
        program: JUNIPER_PROGRAM.to_string(),
        usable: available,
        remaining_count: Some(if available { 1 } else { 0 }),
        clears: if available {
            vec!["anthropic:5h".to_string()]
        } else {
            Vec::new()
        },
        expires_at_ms: None,
        next_available_at_ms,
        weekly_resets_at_ms,
        ineligible_reason,
    });
    if available {
        list.available_count = 1;
        list.next_credit_id = Some(JUNIPER_PROGRAM.to_string());
    }
    list.eligible = available;
    list
}

/// Reset credits embedded in a plain usage payload (cedar/juniper keys on the body).
pub fn parse_reset_credits_from_usage_payload(payload: &Value) -> Option<ResetCreditList> {
    if let Some(cedar) = payload.get(CEDAR_PROGRAM) {
        let list = parse_cedar_credits(cedar);
        if list.available_count > 0 || list.next_credit_id.is_some() {
            return Some(list);
        }
    }
    if let Some(juniper) = payload.get(JUNIPER_PROGRAM) {
        let list = parse_juniper_credits(juniper);
        if list.eligible || !list.credits.is_empty() {
            return Some(list);
        }
    }
    None
}

async fn claude_usage_query(token: &str, base_url: Option<&str>, query: &str) -> Option<Value> {
    let client = client().ok()?;
    for base in oauth_base_urls(base_url) {
        let url = format!("{base}/usage?{query}");
        if let Ok(payload) = get_json(&client, &url, claude_headers(token)).await {
            if payload.as_object().is_some_and(|o| !o.is_empty()) {
                return Some(payload);
            }
        }
    }
    None
}

/// Discover saved anthropic reset credits: cedar grants first, then the juniper
/// session reset. `Ok(None)` means discovery failed; a zero-count list is an
/// authoritative answer.
pub async fn anthropic_reset_credits(
    token: &str,
    base_url: Option<&str>,
) -> Result<Option<ResetCreditList>> {
    if let Some(payload) = claude_usage_query(token, base_url, "cedar_ember=1&skip_spend=1").await {
        if let Some(cedar) = payload.get(CEDAR_PROGRAM) {
            let list = parse_cedar_credits(cedar);
            if list.available_count > 0 || list.next_credit_id.is_some() {
                return Ok(Some(list));
            }
        }
    }
    if let Some(payload) = claude_usage_query(token, base_url, "at_wall=1&skip_spend=1").await {
        if let Some(juniper) = payload.get(JUNIPER_PROGRAM) {
            return Ok(Some(parse_juniper_credits(juniper)));
        }
        if anthropic_has_usage_data(&payload) {
            return Ok(Some(ResetCreditList::default()));
        }
    }
    Ok(None)
}

/// GET `/api/oauth/profile` and extract the current organization id.
pub async fn resolve_organization(token: &str, base_url: Option<&str>) -> Option<String> {
    let client = client().ok()?;
    for base in oauth_base_urls(base_url) {
        if let Ok(payload) =
            get_json(&client, &format!("{base}/profile"), claude_headers(token)).await
        {
            if let Some(org) = payload
                .get("organizations")
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .and_then(|o| o.get("id").or_else(|| o.get("org_id")))
                .and_then(Value::as_str)
            {
                return Some(org.to_string());
            }
        }
    }
    None
}

/// POST `/api/organizations/{orgId}/reset_rate_limits`, spending exactly one
/// anthropic credit. Mutation requests are never retried (TS parity).
pub async fn consume_anthropic_reset_credit(
    token: &str,
    base_url: Option<&str>,
    org_id: &str,
    credit: &ResetCredit,
) -> ConsumeOutcome {
    consume_program_reset(token, base_url, org_id, credit, "reset_rate_limits").await
}

/// Shared mutation path: anthropic hits `/api/organizations/{org}/reset_rate_limits`,
/// codex hits `/wham/rate-limit-reset-credits/consume` with no org in the path.
async fn consume_program_reset(
    token: &str,
    base_url: Option<&str>,
    org_id: &str,
    credit: &ResetCredit,
    anthropic_path: &str,
) -> ConsumeOutcome {
    let fail = |code: &str, status: u16| ConsumeOutcome {
        ok: false,
        code: code.to_string(),
        reason: None,
        cleared: Vec::new(),
        http_status: status,
    };
    if !credit.usable || credit.remaining_count.is_none_or(|r| r <= 0) {
        return fail("credit_not_usable", 0);
    }
    let client = match client() {
        Ok(c) => c,
        Err(_) => return fail("network_error", 0),
    };
    let (url, body, headers) = if credit.program == CEDAR_PROGRAM
        || credit.program == JUNIPER_PROGRAM
    {
        let base = oauth_base_urls(base_url)
            .into_iter()
            .next()
            .unwrap_or_else(|| DEFAULT_CLAUDE_OAUTH_BASE_URL.to_string());
        let body = if credit.program == CEDAR_PROGRAM {
            serde_json::json!({"program": CEDAR_PROGRAM, "grant_id": credit.id, "request_id": uuid_v4()})
        } else {
            serde_json::json!({"program": JUNIPER_PROGRAM})
        };
        let mut headers = claude_headers(token);
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().expect("json ct"),
        );
        (
            format!(
                "{}/api/organizations/{org_id}/{anthropic_path}",
                claude_api_root(&base)
            ),
            body,
            headers,
        )
    } else if credit.program == "codex_reset" {
        let base = codex_base_url(base_url);
        let mut headers = codex_headers(token, None);
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().expect("json ct"),
        );
        (
            format!("{base}/wham/rate-limit-reset-credits/consume"),
            serde_json::json!({}),
            headers,
        )
    } else {
        return fail("unknown_program", 0);
    };
    let response = match client
        .post(url)
        .headers(headers)
        .body(body.to_string())
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => return fail("network_error", 0),
    };
    let status = response.status();
    let payload: Value =
        serde_json::from_str(&response.text().await.unwrap_or_default()).unwrap_or(Value::Null);
    if !status.is_success() {
        let code = match status.as_u16() {
            401 | 403 => "auth_error",
            429 => "rate_limited",
            s => return fail(&format!("http_{s}"), status.as_u16()),
        };
        let mut out = fail(code, status.as_u16());
        out.reason = payload
            .pointer("/error/message")
            .and_then(Value::as_str)
            .map(str::to_string);
        return out;
    }
    let result = payload.get("result").and_then(Value::as_str).unwrap_or("");
    if result.is_empty() {
        return fail("malformed_response", status.as_u16());
    }
    let reason = payload
        .get("reason")
        .and_then(Value::as_str)
        .map(str::to_string);
    let cleared = parse_window_list(payload.get("cleared"));
    let code = match result {
        "reset" => "reset",
        "already_used" => "already_redeemed",
        "not_limited" => "nothing_to_reset",
        other => other,
    };
    ConsumeOutcome {
        ok: result == "reset",
        code: code.to_string(),
        reason,
        cleared,
        http_status: status.as_u16(),
    }
}

// ---------------------------------------------------------------------------
// Kimi (TS usage/kimi.ts)
// ---------------------------------------------------------------------------

async fn fetch_kimi_usage(
    cred: &UsageCredential<'_>,
    base_url: Option<&str>,
) -> Result<UsageReport> {
    let Some(token) = cred.access_token else {
        bail!("no kimi-code oauth access token");
    };
    if cred.token_expired() {
        bail!("kimi-code oauth token expired (refresh first)");
    }
    let env_base = std::env::var("KIMI_CODE_BASE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let base = base_url
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or(env_base.as_deref())
        .unwrap_or(KIMI_DEFAULT_BASE_URL)
        .trim_end_matches('/');
    let client = client()?;
    let mut headers = reqwest::header::HeaderMap::new();
    for (name, value) in pa_ai::kimi_common_headers() {
        headers.insert(
            reqwest::header::HeaderName::from_bytes(name.as_bytes()).expect("kimi header name"),
            value.parse().expect("kimi header value"),
        );
    }
    headers.insert(
        reqwest::header::AUTHORIZATION,
        format!("Bearer {token}").parse().expect("bearer"),
    );
    let payload = get_json(&client, &format!("{base}/usages"), headers).await?;
    let report = parse_kimi_report(&payload);
    if report.lines.is_empty() {
        bail!("kimi usage response had no rows");
    }
    Ok(report)
}

fn kimi_reset_time(data: &serde_json::Map<String, Value>, now: i64) -> Option<i64> {
    for key in ["reset_at", "resetAt", "reset_time", "resetTime"] {
        if let Some(v) = data.get(key) {
            if let Some(s) = v.as_str() {
                if let Some(ms) = parse_iso_ms(s) {
                    return Some(ms);
                }
            }
            if let Some(n) = as_i64(v) {
                return Some(if n > 1_000_000_000_000 { n } else { n * 1000 });
            }
        }
    }
    for key in ["reset_in", "resetIn", "ttl", "window"] {
        if let Some(secs) = data.get(key).and_then(as_f64) {
            return Some(now + (secs * 1000.0) as i64);
        }
    }
    None
}

fn kimi_window(data: &serde_json::Map<String, Value>, now: i64) -> (String, Option<i64>) {
    let duration = data.get("duration").and_then(as_f64);
    let unit = data
        .get("timeUnit")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_uppercase();
    let resets_at = kimi_reset_time(data, now);
    let duration_ms = duration.map(|d| match unit.as_str() {
        u if u.contains("MINUTE") => d * 60_000.0,
        u if u.contains("HOUR") => d * 3_600_000.0,
        u if u.contains("DAY") => d * 86_400_000.0,
        u if u.contains("WEEK") => d * 604_800_000.0,
        u if u.contains("SECOND") => d * 1000.0,
        _ => d,
    });
    let id = match duration_ms {
        Some(ms) if ms > 0.0 && (ms % 86_400_000.0) == 0.0 => {
            format!("{}d", (ms / 86_400_000.0) as i64)
        }
        Some(ms) if ms > 0.0 && (ms % 3_600_000.0) == 0.0 => {
            format!("{}h", (ms / 3_600_000.0) as i64)
        }
        Some(ms) if ms > 0.0 => format!("{}m", (ms / 60_000.0).round() as i64),
        _ => "default".to_string(),
    };
    (id, resets_at)
}

fn kimi_row(data: &serde_json::Map<String, Value>, label: &str, now: i64) -> Option<UsageLine> {
    let limit = data.get("limit").and_then(as_f64);
    let mut used = data.get("used").and_then(as_f64);
    let remaining = data.get("remaining").and_then(as_f64);
    if used.is_none() {
        used = limit.zip(remaining).map(|(l, r)| l - r);
    }
    if used.is_none() && limit.is_none() {
        return None;
    }
    // Kimi nests the window descriptor (duration/timeUnit) under a `window`
    // object on the row (TS `buildWindow(windowData)`); row-level reset keys
    // (resetTime/reset_in) ride on the row itself.
    let (window_id, win_reset) = match data.get("window").and_then(|w| w.as_object()) {
        Some(window_data) => kimi_window(window_data, now),
        None => ("default".to_string(), None),
    };
    let resets_at = kimi_reset_time(data, now).or(win_reset);
    let name = data
        .get("name")
        .or_else(|| data.get("title"))
        .and_then(Value::as_str)
        .unwrap_or(label);
    let used_percent = limit.zip(used).map(|(l, u)| {
        if l > 0.0 {
            (u / l * 100.0).min(100.0)
        } else {
            0.0
        }
    });
    Some(UsageLine {
        label: format!("{name} ({window_id})"),
        used_percent,
        remaining,
        resets_at,
        note: limit.map(|l| format!("of {l:.0}")),
    })
}

fn parse_kimi_report(payload: &Value) -> UsageReport {
    let now = now_ms();
    let mut lines = Vec::new();
    if let Some(usage) = payload.get("usage").and_then(|u| u.as_object()) {
        if let Some(mut row) = kimi_row(usage, "Weekly limit", now) {
            row.label = "Weekly limit (7d)".to_string();
            lines.push(row);
        }
    }
    if let Some(total) = payload.get("totalQuota").and_then(|t| t.as_object()) {
        if let Some(row) = kimi_row(total, "Total quota", now) {
            lines.push(row);
        }
    }
    if let Some(limits) = payload.get("limits").and_then(Value::as_array) {
        for l in limits {
            if let Some(obj) = l.as_object() {
                if let Some(row) = kimi_row(obj, "Limit", now) {
                    lines.push(row);
                }
            }
        }
    }
    UsageReport {
        provider: "kimi-code".to_string(),
        fetched_at_ms: now,
        lines,
        notes: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Z.AI (TS usage/zai.ts)
// ---------------------------------------------------------------------------

async fn fetch_zai_usage(
    cred: &UsageCredential<'_>,
    base_url: Option<&str>,
) -> Result<UsageReport> {
    // Sign-in (oauth) stores the minted id.secret key in access; the paste path in
    // apiKey. Both are used verbatim as the Authorization header (no Bearer).
    let token = cred.access_token.or(cred.raw_key);
    let Some(token) = token.filter(|t| !t.is_empty()) else {
        bail!("no zai key (oauth or api key)");
    };
    let endpoint = base_url
        .and_then(|u| reqwest::Url::parse(u.trim()).ok())
        .map(|u| u.origin().ascii_serialization())
        .unwrap_or_else(|| ZAI_DEFAULT_ENDPOINT.to_string());
    let client = client()?;
    let payload = get_json(
        &client,
        &format!("{endpoint}/api/monitor/usage/quota/limit"),
        header_map(&[
            ("authorization", token.to_string()),
            ("content-type", "application/json".to_string()),
        ]),
    )
    .await?;
    if payload.get("success").and_then(Value::as_bool) != Some(true) {
        let msg = payload
            .get("msg")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        bail!("zai usage rejected: {msg}");
    }
    let report = parse_zai_report(&payload);
    if report.lines.is_empty() {
        bail!("zai usage response had no limits");
    }
    Ok(report)
}

fn zai_window(item: &serde_json::Map<String, Value>) -> (String, String) {
    let count = item
        .get("number")
        .and_then(as_f64)
        .filter(|n| *n > 0.0)
        .unwrap_or(1.0);
    match item.get("unit").and_then(as_i64) {
        Some(3) => (
            format!("{}h", count as i64),
            format!("{} Hour", count as i64),
        ),
        Some(4) => (
            format!("{}d", count as i64),
            format!("{} Day", count as i64),
        ),
        Some(5) => (format!("{}mo", count as i64), "Monthly".to_string()),
        Some(6) => ("1w".to_string(), "Weekly".to_string()),
        Some(u) => (format!("{}u{u}", count as i64), "Quota".to_string()),
        None => ("quota".to_string(), "Quota".to_string()),
    }
}

fn parse_zai_report(payload: &Value) -> UsageReport {
    let mut lines = Vec::new();
    let mut notes = Vec::new();
    if let Some(level) = payload.pointer("/data/level").and_then(Value::as_str) {
        notes.push(format!("plan: {level}"));
    }
    let limits = payload
        .pointer("/data/limits")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for item in limits.iter().filter_map(|l| l.as_object()) {
        let Some(ltype) = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        let (window_id, window_label) = zai_window(item);
        let used_percent = item
            .get("percentage")
            .and_then(as_f64)
            .map(|p| p.clamp(0.0, 100.0));
        let used = item.get("currentValue").and_then(as_f64);
        let limit = item.get("usage").and_then(as_f64);
        let remaining = item.get("remaining").and_then(as_f64);
        let resets_at = epoch_ms(item.get("nextResetTime"));
        let (label, unit) = match ltype {
            "TOKENS_LIMIT" => (
                format!("ZAI {window_label} Token Quota ({window_id})"),
                "tokens",
            ),
            "TIME_LIMIT" => (
                format!("ZAI {window_label} Request Quota ({window_id})"),
                "requests",
            ),
            other => (format!("ZAI {window_label} {other} ({window_id})"), other),
        };
        lines.push(UsageLine {
            label,
            used_percent,
            remaining,
            resets_at,
            note: Some(format!(
                "{unit}: {}/{}",
                used.map(|u| u.to_string()).unwrap_or_else(|| "?".into()),
                limit.map(|u| u.to_string()).unwrap_or_else(|| "?".into())
            )),
        });
    }
    UsageReport {
        provider: "zai".to_string(),
        fetched_at_ms: now_ms(),
        lines,
        notes,
    }
}

// ---------------------------------------------------------------------------
// OpenAI Codex (TS usage/openai-codex.ts + openai-codex-reset.ts)
// ---------------------------------------------------------------------------

fn codex_base_url(base_url: Option<&str>) -> String {
    let trimmed = base_url.map(str::trim).unwrap_or("").trim_end_matches('/');
    match trimmed.is_empty() {
        true => CODEX_BASE_URL.to_string(),
        false => match reqwest::Url::parse(trimmed) {
            Ok(url) => {
                let host = url.host_str().unwrap_or("").to_lowercase();
                if host == "chatgpt.com" || host == "chat.openai.com" {
                    format!(
                        "{}://{}",
                        url.scheme(),
                        url.host_str().unwrap_or("chatgpt.com")
                    )
                } else {
                    CODEX_BASE_URL.to_string()
                }
            }
            Err(_) => CODEX_BASE_URL.to_string(),
        },
    }
}

/// Decode a JWT payload segment without external crates.
fn parse_jwt(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let mut bytes = Vec::with_capacity(payload.len() * 3 / 4 + 3);
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    for ch in payload.bytes() {
        let v: u64 = match ch {
            b'A'..=b'Z' => u64::from(ch - b'A'),
            b'a'..=b'z' => u64::from(ch - b'a' + 26),
            b'0'..=b'9' => u64::from(ch - b'0' + 52),
            b'-' => 62,
            b'_' => 63,
            _ => continue,
        };
        acc = (acc << 6) | v;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            bytes.push((acc >> nbits) as u8);
            acc &= (1u64 << nbits) - 1;
        }
    }
    serde_json::from_slice(&bytes).ok()
}

fn codex_headers(token: &str, account_id: Option<&str>) -> reqwest::header::HeaderMap {
    let mut pairs = vec![
        ("authorization", format!("Bearer {token}")),
        ("user-agent", format!("claude-cli/{CLAUDE_CLI_VERSION}")),
        ("accept", "application/json".to_string()),
    ];
    if let Some(id) = account_id.filter(|s| !s.is_empty()) {
        pairs.push(("chatgpt-account-id", id.to_string()));
    }
    header_map(&pairs)
}

async fn fetch_codex_usage(
    cred: &UsageCredential<'_>,
    base_url: Option<&str>,
) -> Result<UsageReport> {
    let Some(token) = cred.access_token else {
        bail!("no openai-codex oauth access token");
    };
    if cred.token_expired() {
        bail!("openai-codex oauth token expired (refresh first)");
    }
    let jwt = parse_jwt(token);
    let account_id = cred.data_str("accountId").map(str::to_string).or_else(|| {
        jwt.as_ref()
            .and_then(|j| j.get("https://api.openai.com/auth"))
            .and_then(|a| a.get("chatgpt_account_id"))
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    let base = codex_base_url(base_url);
    let client = client()?;
    let payload = get_json(
        &client,
        &format!("{base}/wham/usage"),
        codex_headers(token, account_id.as_deref()),
    )
    .await?;
    let report = parse_codex_report(&payload);
    if report.lines.is_empty() {
        bail!("codex usage response had no windows");
    }
    Ok(report)
}

fn codex_window_line(key: &str, w: Option<&Value>, plan_type: Option<&str>) -> Option<UsageLine> {
    let w = w?.as_object()?;
    let used_percent = w.get("used_percent").and_then(as_f64);
    let window_secs = w.get("limit_window_seconds").and_then(as_f64);
    let reset_at = epoch_ms(w.get("reset_at")).or_else(|| {
        w.get("reset_after_seconds")
            .and_then(as_f64)
            .map(|s| now_ms() + (s * 1000.0) as i64)
    });
    if used_percent.is_none() && window_secs.is_none() && reset_at.is_none() {
        return None;
    }
    let label = match window_secs {
        Some(s) if (s - 18_000.0).abs() < 1.0 => format!("{key} 5h"),
        Some(s) if (s - 604_800.0).abs() < 1.0 => format!("{key} 7d"),
        Some(s) => format!("{key} {}s", s as i64),
        None => key.to_string(),
    };
    let note = plan_type.map(|p| format!("plan {p}"));
    Some(UsageLine {
        label,
        used_percent,
        remaining: None,
        resets_at: reset_at,
        note,
    })
}

fn parse_codex_report(payload: &Value) -> UsageReport {
    let mut lines = Vec::new();
    let mut notes = Vec::new();
    let plan_type = payload
        .get("plan_type")
        .and_then(Value::as_str)
        .map(str::to_string);
    if let Some(plan_type) = &plan_type {
        notes.push(format!("plan: {plan_type}"));
    }
    let rl = payload.get("rate_limit").and_then(|r| r.as_object());
    if let Some(rl) = rl {
        if let Some(allowed) = rl.get("allowed").and_then(Value::as_bool) {
            notes.push(format!("requests allowed: {allowed}"));
        }
        if rl.get("limit_reached").and_then(Value::as_bool) == Some(true) {
            notes.push("plan window exhausted".to_string());
            let credits = payload.get("credits").and_then(|c| c.as_object());
            let funded = credits.is_some_and(|c| {
                (c.get("unlimited") == Some(&Value::Bool(true))
                    || c.get("has_credits") == Some(&Value::Bool(true)))
                    && c.get("overage_limit_reached") != Some(&Value::Bool(true))
            });
            if funded {
                notes.push("credit overage still funds requests".to_string());
            }
        }
        if let Some(line) =
            codex_window_line("primary", rl.get("primary_window"), plan_type.as_deref())
        {
            lines.push(line);
        }
        if let Some(line) = codex_window_line("secondary", rl.get("secondary_window"), None) {
            lines.push(line);
        }
    }
    if let Some(additional) = payload
        .get("additional_rate_limits")
        .and_then(Value::as_array)
    {
        for entry in additional {
            let name = entry
                .get("limit_name")
                .or_else(|| entry.get("metered_feature"))
                .and_then(Value::as_str)
                .unwrap_or("feature");
            let rl = entry.get("rate_limit");
            if let Some(line) =
                codex_window_line(name, rl.and_then(|r| r.get("primary_window")), None)
            {
                lines.push(line);
            }
            if let Some(line) =
                codex_window_line(name, rl.and_then(|r| r.get("secondary_window")), None)
            {
                lines.push(line);
            }
        }
    }
    // Saved reset credits count rides on /wham/usage.
    if let Some(credits) = payload
        .pointer("/rate_limit_reset_credits/available_count")
        .and_then(as_i64)
    {
        notes.push(format!("reset credits available: {credits}"));
    }
    UsageReport {
        provider: "openai-codex".to_string(),
        fetched_at_ms: now_ms(),
        lines,
        notes,
    }
}

/// List codex saved reset credits (`GET /wham/rate-limit-reset-credits`).
pub async fn codex_reset_credits(
    token: &str,
    base_url: Option<&str>,
) -> Result<Option<ResetCreditList>> {
    let base = codex_base_url(base_url);
    let client = client()?;
    let payload = match get_json(
        &client,
        &format!("{base}/wham/rate-limit-reset-credits"),
        codex_headers(token, None),
    )
    .await
    {
        Ok(p) => p,
        Err(e) => {
            if e.to_string().starts_with("404") || e.to_string().starts_with("405") {
                return Ok(Some(ResetCreditList::default()));
            }
            return Err(e);
        }
    };
    let available = payload.get("available_count").and_then(as_i64).unwrap_or(0);
    let credits = payload
        .get("credits")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|c| {
                    Some(ResetCredit {
                        id: c.get("id").and_then(Value::as_str)?.to_string(),
                        program: "codex_reset".to_string(),
                        usable: c.get("usable").and_then(Value::as_bool).unwrap_or(true),
                        remaining_count: c.get("remaining_count").and_then(as_i64).or(Some(1)),
                        clears: parse_window_list(c.get("clears")),
                        expires_at_ms: epoch_ms(c.get("expires_at")),
                        next_available_at_ms: epoch_ms(c.get("next_available_at")),
                        weekly_resets_at_ms: None,
                        ineligible_reason: c
                            .get("ineligible_reason")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let usable = credits.iter().filter(|c| c.usable).count();
    Ok(Some(ResetCreditList {
        available_count: if available > 0 {
            available as usize
        } else {
            usable
        },
        next_credit_id: credits.first().map(|c| c.id.clone()),
        eligible: available > 0 || usable > 0,
        credits,
    }))
}

/// POST `/wham/rate-limit-reset-credits/consume`, spending one codex credit.
pub async fn consume_codex_reset_credit(
    token: &str,
    base_url: Option<&str>,
    credit: &ResetCredit,
) -> ConsumeOutcome {
    consume_program_reset(token, base_url, "", credit, "").await
}

// ---------------------------------------------------------------------------
// Google Antigravity (TS usage/google-antigravity.ts, quota-summary path)
// ---------------------------------------------------------------------------

async fn fetch_antigravity_usage(
    cred: &UsageCredential<'_>,
    base_url: Option<&str>,
) -> Result<UsageReport> {
    let Some(token) = cred.access_token else {
        bail!("no google-antigravity oauth access token");
    };
    if cred.token_expired() {
        bail!("google-antigravity oauth token expired (refresh first)");
    }
    let project_id = cred
        .data_str("projectId")
        .or_else(|| cred.data_str("project_id"))
        .map(str::to_string);
    let Some(project_id) = project_id else {
        bail!("google-antigravity credential has no projectId");
    };
    let endpoints: Vec<String> = match base_url.map(str::trim).filter(|s| !s.is_empty()) {
        Some(u) => vec![u.trim_end_matches('/').to_string()],
        None => vec![
            ANTIGRAVITY_DEFAULT_ENDPOINT.to_string(),
            ANTIGRAVITY_SANDBOX_ENDPOINT.to_string(),
        ],
    };
    let client = client()?;
    let body = serde_json::json!({"project": project_id});
    for endpoint in &endpoints {
        let result = post_json(
            &client,
            &format!("{endpoint}/v1internal:retrieveUserQuotaSummary"),
            bearer_headers(token, &[("content-type", "application/json".to_string())]),
            body.clone(),
        )
        .await;
        if let Ok(payload) = result {
            if let Some(report) = parse_quota_summary(&payload) {
                return Ok(report);
            }
        }
    }
    bail!("antigravity quota summary unavailable on all endpoints")
}

fn classify_window(id: Option<&str>, label: Option<&str>) -> Option<(&'static str, &'static str)> {
    let source = format!("{} {}", id.unwrap_or(""), label.unwrap_or("")).to_lowercase();
    if source.contains("week") || source.contains("7d") {
        Some(("weekly", "Weekly"))
    } else if source.contains("5h") || source.contains("five hour") {
        Some(("5h", "5 Hour"))
    } else if source.contains("day") || source.contains("24h") {
        Some(("daily", "Daily"))
    } else {
        None
    }
}

fn parse_quota_summary(payload: &Value) -> Option<UsageReport> {
    let mut lines = Vec::new();
    let mut push_bucket = |bucket: &Value, group_name: Option<&str>| {
        if bucket.get("disabled").and_then(Value::as_bool) == Some(true) {
            return;
        }
        let window_id = bucket.get("window").and_then(Value::as_str);
        let display = bucket.get("displayName").and_then(Value::as_str);
        let (wid, wlabel) = classify_window(window_id, display)
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .unwrap_or_else(|| {
                (
                    window_id.unwrap_or("default").to_string(),
                    display.unwrap_or("Usage").to_string(),
                )
            });
        let remaining_fraction = bucket
            .get("remainingFraction")
            .and_then(as_f64)
            .map(|f| f.clamp(0.0, 1.0));
        let remaining_amount = bucket
            .get("remainingAmount")
            .and_then(|v| as_f64(v).or_else(|| v.as_str()?.parse().ok()));
        let used_percent = remaining_fraction.map(|f| (1.0 - f) * 100.0);
        let resets_at = bucket
            .get("resetTime")
            .and_then(Value::as_str)
            .and_then(parse_iso_ms);
        let bucket_id = bucket.get("bucketId").and_then(Value::as_str).unwrap_or("");
        let gname = group_name.unwrap_or("").to_lowercase();
        let counter = if bucket_id.to_lowercase().starts_with("gemini-") || gname.contains("gemini")
        {
            "Gemini"
        } else if bucket_id.to_lowercase().starts_with("3p-")
            || gname.contains("claude")
            || gname.contains("gpt")
        {
            "Claude & GPT (shared)"
        } else {
            wlabel.as_str()
        };
        lines.push(UsageLine {
            label: format!("{counter} {wlabel}"),
            used_percent,
            remaining: remaining_amount,
            resets_at,
            note: Some(format!("window {wid}")),
        });
    };
    let groups = payload.get("groups").and_then(Value::as_array);
    let has_groups = groups.is_some_and(|g| {
        g.iter().any(|gr| {
            gr.get("buckets")
                .and_then(Value::as_array)
                .is_some_and(|b| !b.is_empty())
        })
    });
    if has_groups {
        for group in groups.unwrap() {
            let gname = group.get("displayName").and_then(Value::as_str);
            if let Some(buckets) = group.get("buckets").and_then(Value::as_array) {
                for bucket in buckets {
                    push_bucket(bucket, gname);
                }
            }
        }
    } else if let Some(buckets) = payload.get("buckets").and_then(Value::as_array) {
        for bucket in buckets {
            push_bucket(bucket, None);
        }
    }
    if lines.is_empty() {
        return None;
    }
    Some(UsageReport {
        provider: "google-antigravity".to_string(),
        fetched_at_ms: now_ms(),
        lines,
        notes: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Google Gemini CLI (TS usage/gemini.ts)
// ---------------------------------------------------------------------------

async fn fetch_gemini_usage(
    cred: &UsageCredential<'_>,
    base_url: Option<&str>,
) -> Result<UsageReport> {
    let Some(token) = cred.access_token else {
        bail!("no google-gemini-cli oauth access token");
    };
    if cred.token_expired() {
        bail!("google-gemini-cli oauth token expired (refresh first)");
    }
    let base = base_url
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(GEMINI_DEFAULT_ENDPOINT)
        .trim_end_matches('/');
    let project_id = cred
        .data_str("projectId")
        .or_else(|| cred.data_str("project_id"))
        .map(str::to_string);
    let client = client()?;
    let headers = bearer_headers(token, &[("content-type", "application/json".to_string())]);
    // loadCodeAssist provisions the project when the credential has none.
    let project_id = match project_id {
        Some(p) => Some(p),
        None => {
            let body = serde_json::json!({});
            post_json(
                &client,
                &format!("{base}/v1internal:loadCodeAssist"),
                headers.clone(),
                body,
            )
            .await
            .ok()
            .and_then(|r| {
                r.get("cloudaicompanionProject")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
        }
    };
    let Some(project_id) = project_id else {
        bail!("google-gemini-cli: no project id (loadCodeAssist failed)");
    };
    let quota = post_json(
        &client,
        &format!("{base}/v1internal:retrieveUserQuota"),
        headers,
        serde_json::json!({"project": project_id}),
    )
    .await?;
    let mut lines = Vec::new();
    if let Some(buckets) = quota.get("buckets").and_then(Value::as_array) {
        for bucket in buckets {
            let model_id = bucket.get("modelId").and_then(Value::as_str);
            let remaining_fraction = bucket
                .get("remainingFraction")
                .and_then(as_f64)
                .map(|f| f.clamp(0.0, 1.0));
            let window = bucket
                .get("resetTime")
                .and_then(Value::as_str)
                .and_then(parse_iso_ms);
            let label = model_id
                .map(|m| format!("Gemini {m}"))
                .unwrap_or_else(|| "Gemini quota".to_string());
            lines.push(UsageLine {
                label,
                used_percent: remaining_fraction.map(|f| (1.0 - f) * 100.0),
                remaining: None,
                resets_at: window,
                note: model_id.map(|m| format!("model {m}")),
            });
        }
    }
    if lines.is_empty() {
        bail!("gemini retrieveUserQuota returned no buckets");
    }
    Ok(UsageReport {
        provider: "google-gemini-cli".to_string(),
        fetched_at_ms: now_ms(),
        lines,
        notes: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

/// Random UUIDv4 without pulling in a uuid crate.
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let pid = std::process::id() as u128;
    let mut x: u128 = nanos ^ (pid << 64) ^ 0x9E37_79B9_7F4A_7C15;
    let mut bytes = [0u8; 16];
    for b in &mut bytes {
        x ^= x << 7;
        x ^= x >> 9;
        x = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        *b = (x >> 64) as u8;
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex = bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn anthropic_parses_windows_limits_extra_and_spend() {
        let payload = json!({
            "five_hour": {"utilization": 42.5, "resets_at": "2026-10-21T18:00:00+00:00"},
            "seven_day": {"utilization": 33.0, "resets_at": "2026-10-25T17:59:59.674220+00:00"},
            "limits": [
                {"kind": "weekly", "percent": 33,
                 "scope": {"model": {"display_name": "Opus"}}, "is_active": true}
            ],
            "extra_usage": {"is_enabled": true, "monthly_limit": 100, "used_credits": 12, "currency": "USD"},
            "spend": {"used": {"amount_minor": 500, "currency": "USD"}, "enabled": true}
        });
        let report = parse_anthropic_report(&payload, "t");
        assert_eq!(report.lines.len(), 3);
        assert!((report.lines[0].used_percent.unwrap() - 42.5).abs() < f64::EPSILON);
        assert!(report.lines[0].resets_at.is_some());
        assert_eq!(report.lines[2].note.as_deref(), Some("active"));
        assert!(report.notes.iter().any(|n| n.starts_with("extra usage")));
        assert!(report.notes.iter().any(|n| n.starts_with("spend")));
    }

    #[test]
    fn parses_cedar_grants_from_probe_payload() {
        let payload = json!({
            "cedar_ember": {
                "grants": [
                    {"id": "ce_abc123", "remaining_count": 2, "clears": ["anthropic:5h"],
                     "ends_at": "2026-11-01T00:00:00+00:00"}
                ],
                "next_grant_id": "ce_abc123"
            }
        });
        let list = parse_reset_credits_from_usage_payload(&payload).expect("cedar list");
        assert_eq!(list.available_count, 1);
        assert_eq!(list.next_credit_id.as_deref(), Some("ce_abc123"));
        assert_eq!(list.credits[0].clears, vec!["anthropic:5h".to_string()]);
    }

    #[test]
    fn parses_juniper_block() {
        let payload = json!({"juniper_tide": {"available": true}});
        let list = parse_reset_credits_from_usage_payload(&payload).expect("juniper list");
        assert!(list.eligible);
        assert_eq!(list.credits[0].clears, vec!["anthropic:5h".to_string()]);
        let cooled = json!({"juniper_tide": {"available": true, "cooldown_until": "2099-01-01T00:00:00+00:00"}});
        let list = parse_reset_credits_from_usage_payload(&cooled).expect("cooled");
        assert!(!list.eligible);
        assert!(!list.credits[0].usable);
    }

    #[test]
    fn kimi_parses_weekly_summary_and_limits() {
        let payload = json!({
            "usage": {"limit": 100, "used": 40, "resetTime": "2026-10-25T17:59:59+00:00"},
            "totalQuota": {"limit": 500, "remaining": 300,
                           "window": {"duration": 300, "timeUnit": "minutes"}},
            "limits": [{"name": "RPM", "limit": 60, "used": 6,
                        "window": {"duration": 1, "timeUnit": "minutes"}, "reset_in": 30}]
        });
        let report = parse_kimi_report(&payload);
        assert_eq!(report.lines.len(), 3);
        assert_eq!(report.lines[0].label, "Weekly limit (7d)");
        assert!((report.lines[0].used_percent.unwrap() - 40.0).abs() < f64::EPSILON);
        assert!(
            report.lines[1].label.contains("5h"),
            "{}",
            report.lines[1].label
        );
        assert!(report.lines[2].resets_at.is_some());
    }

    #[test]
    fn zai_parses_quota_limits() {
        let payload = json!({
            "success": true,
            "data": {
                "level": "pro",
                "limits": [
                    {"type": "TOKENS_LIMIT", "usage": 1000, "currentValue": 250,
                     "percentage": 25, "remaining": 750, "nextResetTime": 1793000000,
                     "unit": 3, "number": 5},
                    {"type": "TIME_LIMIT", "usage": 100, "currentValue": 10,
                     "percentage": 10, "nextResetTime": 1793000000, "unit": 6}
                ]
            }
        });
        let report = parse_zai_report(&payload);
        assert_eq!(report.lines.len(), 2);
        assert!(report.lines[0].label.contains("5 Hour"));
        assert!((report.lines[0].used_percent.unwrap() - 25.0).abs() < f64::EPSILON);
        assert!(report.lines[1].label.contains("Weekly"));
        assert_eq!(report.notes, vec!["plan: pro".to_string()]);
    }

    #[test]
    fn codex_parses_plan_windows_and_credits() {
        let payload = json!({
            "plan_type": "plus",
            "rate_limit": {
                "allowed": true, "limit_reached": false,
                "primary_window": {"used_percent": 55, "limit_window_seconds": 18000, "reset_after_seconds": 3600},
                "secondary_window": {"used_percent": 10, "limit_window_seconds": 604800, "reset_at": 1793500000}
            },
            "rate_limit_reset_credits": {"available_count": 2}
        });
        let report = parse_codex_report(&payload);
        assert!(report.lines.iter().any(|l| l.label == "primary 5h"));
        assert!(report.lines.iter().any(|l| l.label == "secondary 7d"));
        assert!((report.lines[0].used_percent.unwrap() - 55.0).abs() < f64::EPSILON);
        assert!(report.notes.iter().any(|n| n == "plan: plus"));
        assert!(report
            .notes
            .iter()
            .any(|n| n == "reset credits available: 2"));
    }

    #[test]
    fn antigravity_quota_summary_groups_and_shared_counter() {
        let payload = json!({
            "groups": [
                {"displayName": "Gemini models", "buckets": [
                    {"bucketId": "gemini-3", "window": "5h", "remainingFraction": 0.75,
                     "resetTime": "2026-10-21T18:00:00+00:00"}]},
                {"displayName": "Third-party", "buckets": [
                    {"bucketId": "3p-claude", "window": "weekly", "remainingFraction": 0.9}]}
            ]
        });
        let report = parse_quota_summary(&payload).expect("report");
        assert_eq!(report.lines.len(), 2);
        assert!(report.lines[0].label.starts_with("Gemini"));
        assert!((report.lines[0].used_percent.unwrap() - 25.0).abs() < f64::EPSILON);
        assert!(report.lines[1].label.contains("shared"));
    }

    #[test]
    fn base_url_normalization_strips_v1_and_oauth_suffixes() {
        let urls = oauth_base_urls(Some("https://gateway.example.com/prefix/v1"));
        assert_eq!(urls[0], "https://gateway.example.com/prefix/api/oauth");
        assert_eq!(urls[1], DEFAULT_CLAUDE_OAUTH_BASE_URL);
    }

    #[test]
    fn iso_parse_handles_offset_and_fraction() {
        let a = parse_iso_ms("2026-10-21T18:00:00+00:00").expect("a");
        let b = parse_iso_ms("2026-10-22T18:00:00+00:00").expect("b");
        assert_eq!(b - a, 86_400_000);
        assert!(parse_iso_ms("2026-10-25T17:59:59.674220+00:00").expect("frac") > 0);
    }

    #[test]
    fn jwt_payload_decodes_account_claim() {
        // {"https://api.openai.com/auth":{"chatgpt_account_id":"acc-1"}}
        let payload =
            "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjLTEifX0";
        let token = format!("eyJhbGciOiJIUzI1NiJ9.{payload}.sig");
        let claims = parse_jwt(&token).expect("jwt");
        assert_eq!(
            claims
                .get("https://api.openai.com/auth")
                .and_then(|a| a.get("chatgpt_account_id"))
                .and_then(Value::as_str),
            Some("acc-1")
        );
        assert!(parse_jwt("not-a-jwt").is_none());
    }
}
