//! OAuth rate-limit usage polling and reset credits, TS `packages/ai/src/usage/claude.ts`
//! and `claude-reset.ts` ported for the anthropic provider. Fetches the account's
//! `/api/oauth/usage` report (5h/7d windows, per-model limits, extra usage, spend),
//! discovers saved reset credits (`cedar_ember` grants, `juniper_tide` session reset),
//! and consumes one via `POST /api/organizations/{orgId}/reset_rate_limits`.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// `user-agent` Claude's account routes gate on (TS `getClaudeCodeUserAgent`).
pub const CLAUDE_CLI_VERSION: &str = "2.1.281";
/// OAuth protocol beta required by the account routes.
pub const CLAUDE_OAUTH_BETA: &str = "claude-code-20250219,oauth-2025-04-20";
/// Canonical subscription usage/profile endpoint root.
pub const DEFAULT_CLAUDE_OAUTH_BASE_URL: &str = "https://api.anthropic.com/api/oauth";

const CEDAR_PROGRAM: &str = "cedar_ember";
const JUNIPER_PROGRAM: &str = "juniper_tide";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// One usage window (`five_hour`, `seven_day`, per-model variants).
#[derive(Debug, Clone, Default)]
pub struct UsageBucket {
    /// 0..100 when the provider reports it.
    pub utilization: Option<f64>,
    /// Epoch milliseconds.
    pub resets_at: Option<i64>,
}

/// An entry of the provider's `limits` array.
#[derive(Debug, Clone, Default)]
pub struct ApiLimit {
    pub kind: String,
    pub percent: Option<f64>,
    pub resets_at: Option<i64>,
    /// `scope.model.display_name` when present (e.g. "Opus", "Sonnet").
    pub display_name: Option<String>,
    pub is_active: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ExtraUsage {
    pub is_enabled: bool,
    pub monthly_limit: Option<f64>,
    pub used_credits: Option<f64>,
    pub currency: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Spend {
    pub enabled: bool,
    pub used_minor: Option<i64>,
    pub limit_minor: Option<i64>,
    pub currency: Option<String>,
}

/// Parsed `/api/oauth/usage` report.
#[derive(Debug, Clone)]
pub struct UsageReport {
    pub fetched_at_ms: i64,
    pub buckets: Vec<(String, UsageBucket)>,
    pub limits: Vec<ApiLimit>,
    pub extra_usage: Option<ExtraUsage>,
    pub spend: Option<Spend>,
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
    /// The credit Claude's server selected as next.
    pub next_credit_id: Option<String>,
    pub eligible: bool,
}

/// Outcome of consuming a reset credit (`result`/`reason`/`cleared` from the API).
#[derive(Debug, Clone)]
pub struct ConsumeOutcome {
    pub ok: bool,
    /// Normalized result code: `reset`, `already_redeemed`, `nothing_to_reset`,
    /// or an error code (`auth_error`, `rate_limited`, `http_*`, `network_error`, ...).
    pub code: String,
    pub reason: Option<String>,
    /// Window ids the server cleared.
    pub cleared: Vec<String>,
    pub http_status: u16,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn as_f64(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str()?.parse().ok())
}

fn as_i64(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))
}

fn parse_iso_ms(s: &str) -> Option<i64> {
    // chrono-free ISO-8601 parse for the formats the API emits
    // ("2026-10-25T17:59:59.674220+00:00").
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

fn parse_bucket(v: Option<&Value>) -> Option<UsageBucket> {
    let v = v?.as_object()?;
    let utilization = v.get("utilization").and_then(as_f64);
    let resets_at = v.get("resets_at").and_then(Value::as_str).and_then(parse_iso_ms);
    (utilization.is_some() || resets_at.is_some()).then_some(UsageBucket { utilization, resets_at })
}

fn parse_usage_payload(payload: &Value) -> UsageReport {
    let fetched_at_ms = now_ms();
    let mut buckets = Vec::new();
    for (key, label) in [
        ("five_hour", "5h"),
        ("seven_day", "7d"),
        ("seven_day_opus", "7d opus"),
        ("seven_day_sonnet", "7d sonnet"),
    ] {
        if let Some(b) = parse_bucket(payload.get(key)) {
            buckets.push((label.to_string(), b));
        }
    }
    let mut limits = Vec::new();
    if let Some(entries) = payload.get("limits").and_then(Value::as_array) {
        for e in entries {
            let kind = e.get("kind").and_then(Value::as_str).unwrap_or("").to_string();
            let percent = e.get("percent").and_then(as_f64);
            let resets_at = e.get("resets_at").and_then(Value::as_str).and_then(parse_iso_ms);
            let display_name = e
                .pointer("/scope/model/display_name")
                .and_then(Value::as_str)
                .map(str::to_string);
            let is_active = e.get("is_active").and_then(Value::as_bool).unwrap_or(false);
            if kind.is_empty() && percent.is_none() {
                continue;
            }
            limits.push(ApiLimit { kind, percent, resets_at, display_name, is_active });
        }
    }
    let extra_usage = payload.get("extra_usage").and_then(|e| {
        let o = e.as_object()?;
        Some(ExtraUsage {
            is_enabled: o.get("is_enabled").and_then(Value::as_bool).unwrap_or(false),
            monthly_limit: o.get("monthly_limit").and_then(as_f64),
            used_credits: o.get("used_credits").and_then(as_f64),
            currency: o.get("currency").and_then(Value::as_str).map(str::to_string),
        })
    });
    let spend = payload.get("spend").and_then(|s| {
        let o = s.as_object()?;
        let money = |k: &str| o.get(k).and_then(|m| m.as_object());
        let minor = |k: &str| money(k).and_then(|m| m.get("amount_minor")).and_then(as_i64);
        Some(Spend {
            enabled: o.get("enabled").and_then(Value::as_bool).unwrap_or(false),
            used_minor: minor("used"),
            limit_minor: minor("limit"),
            currency: money("used")
                .and_then(|m| m.get("currency"))
                .and_then(Value::as_str)
                .map(str::to_string),
        })
    });
    UsageReport { fetched_at_ms, buckets, limits, extra_usage, spend }
}

/// Whether a payload looks like a usage envelope at all.
pub fn has_usage_data(payload: &Value) -> bool {
    ["five_hour", "seven_day", "limits", "extra_usage", "spend", CEDAR_PROGRAM, JUNIPER_PROGRAM]
        .iter()
        .any(|k| payload.get(*k).is_some())
}

fn parse_window_list(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

fn parse_cedar_credits(block: &Value) -> ResetCreditList {
    let mut list = ResetCreditList::default();
    let now = now_ms();
    if let Some(grants) = block.get("grants").and_then(Value::as_array) {
        for g in grants {
            let id = g.get("id").and_then(Value::as_str).unwrap_or("").to_string();
            if id.is_empty() {
                continue;
            }
            let remaining = g.get("remaining_count").and_then(as_i64);
            let usable = remaining.is_some_and(|r| r > 0);
            list.credits.push(ResetCredit {
                id,
                program: CEDAR_PROGRAM.to_string(),
                usable,
                remaining_count: remaining,
                clears: parse_window_list(g.get("clears")),
                expires_at_ms: g.get("ends_at").and_then(Value::as_str).and_then(parse_iso_ms),
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
    list.next_credit_id = block.get("next_grant_id").and_then(Value::as_str).map(str::to_string);
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
    let available = block.get("available").and_then(Value::as_bool).unwrap_or(false)
        && cooldown_until_ms.is_none_or(|c| c <= now);
    let ineligible_reason = block.get("ineligible_reason").and_then(Value::as_str).map(str::to_string);
    let mut list = ResetCreditList::default();
    list.credits.push(ResetCredit {
        id: JUNIPER_PROGRAM.to_string(),
        program: JUNIPER_PROGRAM.to_string(),
        usable: available,
        remaining_count: Some(if available { 1 } else { 0 }),
        clears: if available { vec!["anthropic:5h".to_string()] } else { Vec::new() },
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

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder().timeout(REQUEST_TIMEOUT).build().context("usage http client")
}

fn oauth_headers(access_token: &str) -> Vec<(&'static str, String)> {
    vec![
        ("authorization", format!("Bearer {access_token}")),
        ("anthropic-beta", CLAUDE_OAUTH_BETA.to_string()),
        ("anthropic-version", "2023-06-01".to_string()),
        ("user-agent", format!("claude-cli/{CLAUDE_CLI_VERSION}")),
        ("accept", "application/json".to_string()),
        ("x-app", "cli".to_string()),
    ]
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

fn header_map(access_token: &str) -> reqwest::header::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    for (name, value) in oauth_headers(access_token) {
        headers.insert(
            reqwest::header::HeaderName::from_bytes(name.as_bytes()).expect("valid header name"),
            value.parse().expect("valid header value"),
        );
    }
    headers
}

/// GET `/api/oauth/usage` and parse the report. Tries the configured base URL,
/// falling back to the canonical one when the configured host has no endpoint.
pub async fn fetch_usage(access_token: &str, base_url: Option<&str>) -> Result<UsageReport> {
    let client = client()?;
    let mut last_err = anyhow::anyhow!("no oauth base url");
    for base in oauth_base_urls(base_url) {
        let request = client.get(format!("{base}/usage")).headers(header_map(access_token));
        match request.send().await {
            Ok(response) => {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                if status.as_u16() == 429 {
                    bail!("usage endpoint rate-limited (429); retry in a minute");
                }
                if !status.is_success() {
                    last_err = anyhow::anyhow!(
                        "usage endpoint {status}: {}",
                        text.chars().take(200).collect::<String>()
                    );
                    continue;
                }
                let payload: Value = serde_json::from_str(&text)
                    .map_err(|e| anyhow::anyhow!("usage endpoint returned malformed JSON: {e}"))?;
                if !has_usage_data(&payload) {
                    last_err = anyhow::anyhow!("usage endpoint returned no usage data");
                    continue;
                }
                return Ok(parse_usage_payload(&payload));
            }
            Err(e) => {
                last_err = anyhow::anyhow!("usage request failed: {e}");
            }
        }
    }
    Err(last_err)
}

async fn fetch_usage_query(access_token: &str, base_url: Option<&str>, query: &str) -> Option<Value> {
    let client = client().ok()?;
    for base in oauth_base_urls(base_url) {
        let url = format!("{base}/usage?{query}");
        let request = client.get(&url).headers(header_map(access_token));
        let response = match request.send().await {
            Ok(r) => r,
            Err(_) => return None,
        };
        if !response.status().is_success() {
            continue;
        }
        let Ok(payload) = serde_json::from_str::<Value>(&response.text().await.unwrap_or_default()) else {
            continue;
        };
        if payload.as_object().is_some_and(|o| !o.is_empty()) {
            return Some(payload);
        }
    }
    None
}

/// Discover saved reset credits: cedar grants first, then the juniper session reset.
/// `Ok(None)` means discovery failed; a zero-count list is an authoritative answer.
pub async fn list_reset_credits(access_token: &str, base_url: Option<&str>) -> Result<Option<ResetCreditList>> {
    // Cedar grants live under `?cedar_ember=1&skip_spend=1`.
    if let Some(payload) = fetch_usage_query(access_token, base_url, "cedar_ember=1&skip_spend=1").await {
        if let Some(cedar) = payload.get(CEDAR_PROGRAM) {
            let list = parse_cedar_credits(cedar);
            if list.available_count > 0 || list.next_credit_id.is_some() {
                return Ok(Some(list));
            }
        }
    }
    // Juniper session reset: `?at_wall=1&skip_spend=1`.
    if let Some(payload) = fetch_usage_query(access_token, base_url, "at_wall=1&skip_spend=1").await {
        if let Some(juniper) = payload.get(JUNIPER_PROGRAM) {
            return Ok(Some(parse_juniper_credits(juniper)));
        }
        // The probe answered but carries no juniper block: authoritative "no credits".
        if has_usage_data(&payload) {
            return Ok(Some(ResetCreditList::default()));
        }
    }
    Ok(None)
}

/// GET `/api/oauth/profile` and extract the current organization id.
pub async fn resolve_organization(access_token: &str, base_url: Option<&str>) -> Option<String> {
    let client = client().ok()?;
    for base in oauth_base_urls(base_url) {
        let request = client.get(format!("{base}/profile")).headers(header_map(access_token));
        let Ok(response) = request.send().await else { continue };
        if !response.status().is_success() {
            continue;
        }
        let Ok(payload) = serde_json::from_str::<Value>(&response.text().await.unwrap_or_default()) else {
            continue;
        };
        // Claude's profile payload lists organizations; take the first id.
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
    None
}

/// POST `/api/organizations/{orgId}/reset_rate_limits`, spending exactly one credit.
/// Mutation requests are never retried (TS parity).
pub async fn consume_reset_credit(
    access_token: &str,
    base_url: Option<&str>,
    org_id: &str,
    credit: &ResetCredit,
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
    let body = if credit.program == CEDAR_PROGRAM {
        serde_json::json!({
            "program": CEDAR_PROGRAM,
            "grant_id": credit.id,
            "request_id": uuid_v4(),
        })
    } else if credit.program == JUNIPER_PROGRAM {
        serde_json::json!({"program": JUNIPER_PROGRAM})
    } else {
        return fail("unknown_program", 0);
    };
    let Ok(client) = client() else { return fail("network_error", 0) };
    let base = oauth_base_urls(base_url)
        .into_iter()
        .next()
        .unwrap_or_else(|| DEFAULT_CLAUDE_OAUTH_BASE_URL.to_string());
    let api_root = base.trim_end_matches("/api/oauth");
    let url = format!("{api_root}/api/organizations/{org_id}/reset_rate_limits");
    let mut headers = header_map(access_token);
    headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse().expect("json ct"));
    let response = match client.post(url).headers(headers).body(body.to_string()).send().await {
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
    let reason = payload.get("reason").and_then(Value::as_str).map(str::to_string);
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

/// Random UUIDv4 without pulling in a uuid crate.
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
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
    format!("{}-{}-{}-{}-{}", &hex[..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..32])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_windows_limits_extra_and_spend() {
        let payload = json!({
            "five_hour": {"utilization": 42.5, "resets_at": "2026-10-21T18:00:00+00:00"},
            "seven_day": {"utilization": 33.0, "resets_at": "2026-10-25T17:59:59.674220+00:00"},
            "limits": [
                {"kind": "weekly", "percent": 33, "resets_at": null,
                 "scope": {"model": {"display_name": "Opus"}}, "is_active": true}
            ],
            "extra_usage": {"is_enabled": true, "monthly_limit": 100, "used_credits": 12, "currency": "USD"},
            "spend": {"used": {"amount_minor": 500, "currency": "USD"}, "limit": null, "enabled": true}
        });
        let report = parse_usage_payload(&payload);
        assert_eq!(report.buckets.len(), 2);
        assert!((report.buckets[0].1.utilization.unwrap() - 42.5).abs() < f64::EPSILON);
        assert!(report.buckets[0].1.resets_at.is_some());
        assert_eq!(report.limits.len(), 1);
        assert_eq!(report.limits[0].display_name.as_deref(), Some("Opus"));
        assert!(report.limits[0].is_active);
        let extra = report.extra_usage.expect("extra usage");
        assert!(extra.is_enabled);
        assert!((extra.used_credits.unwrap() - 12.0).abs() < f64::EPSILON);
        let spend = report.spend.expect("spend");
        assert_eq!(spend.used_minor, Some(500));
    }

    #[test]
    fn parses_cedar_grants_from_probe_payload() {
        let payload = json!({
            "cedar_ember": {
                "grants": [
                    {"id": "ce_abc123", "remaining_count": 2, "clears": ["anthropic:5h"],
                     "starts_at": "2026-10-01T00:00:00+00:00", "ends_at": "2026-11-01T00:00:00+00:00"}
                ],
                "next_grant_id": "ce_abc123"
            }
        });
        let list = parse_reset_credits_from_usage_payload(&payload).expect("cedar list");
        assert_eq!(list.available_count, 1);
        assert_eq!(list.next_credit_id.as_deref(), Some("ce_abc123"));
        assert_eq!(list.credits[0].clears, vec!["anthropic:5h".to_string()]);
        assert!(list.credits[0].usable);
    }

    #[test]
    fn parses_juniper_block() {
        let payload = json!({
            "juniper_tide": {"available": true, "next_available_at": null, "weekly_resets_at": null}
        });
        let list = parse_reset_credits_from_usage_payload(&payload).expect("juniper list");
        assert!(list.eligible);
        assert_eq!(list.credits[0].program, "juniper_tide");
        assert_eq!(list.credits[0].clears, vec!["anthropic:5h".to_string()]);
    }

    #[test]
    fn juniper_cooldown_makes_credit_unusable() {
        let payload =
            json!({"juniper_tide": {"available": true, "cooldown_until": "2099-01-01T00:00:00+00:00"}});
        let list = parse_reset_credits_from_usage_payload(&payload).expect("juniper list");
        assert!(!list.eligible);
        assert!(!list.credits[0].usable);
    }

    #[test]
    fn base_url_normalization_strips_v1_and_oauth_suffixes() {
        let urls = oauth_base_urls(Some("https://gateway.example.com/prefix/v1"));
        assert_eq!(urls[0], "https://gateway.example.com/prefix/api/oauth");
        assert_eq!(urls[1], DEFAULT_CLAUDE_OAUTH_BASE_URL);
        let urls = oauth_base_urls(None);
        assert_eq!(urls, vec![DEFAULT_CLAUDE_OAUTH_BASE_URL.to_string()]);
    }

    #[test]
    fn iso_parse_handles_offset_and_fraction() {
        let a = parse_iso_ms("2026-10-21T18:00:00+00:00").expect("a");
        let b = parse_iso_ms("2026-10-22T18:00:00+00:00").expect("b");
        assert_eq!(b - a, 86_400_000);
        assert!(parse_iso_ms("2026-10-25T17:59:59.674220+00:00").expect("frac") > 0);
    }
}
