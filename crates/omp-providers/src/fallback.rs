//! `retry.fallbackChains` from `config.yml`, resolved the way omp does
//! (`pi-coding-agent/src/session/retry-fallback-chains.ts`):
//!
//! * chain keys are model selectors (`provider/id[:level]`), provider wildcards
//!   (`provider/*`, `provider/prefix/*`) or role names;
//! * the chain for a live model is picked by specificity: exact model key,
//!   suffixless model key, longest wildcard, hinted role, role keys whose assigned
//!   model matches (`default` first), then `default`;
//! * the `default` chain is shared by every non-kind role without its own chain;
//! * candidates are the chain entries after the live model, wildcard entries
//!   re-targeted at the failing model's id.
//!
//! State (`FallbackState`): per-selector cooldowns, the active fallback with its
//! `served` flag and revert-to-primary after cooldown expiry; `wrapAround`, `normalized`
//! effort matching and the aggregator -> direct id rewrite use [`ModelLookup`].
//!
//! Not ported: subagent role installation and provider-specific rate-limit patterns.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use crate::{ModelThinkingLevel, Sel};

/// Roles that never inherit the `default` chain (`KIND_ROLE_IDS`).
const KIND_ROLES: [&str; 5] = ["image", "web", "speech", "dictation", "judge"];

/// `retry.*` settings with omp's defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryConfig {
    pub enabled: bool,
    pub max_retries: u32,
    pub base_delay_ms: u64,
    /// Longest wait honoured before failing over / giving up; `0` = no ceiling.
    pub max_delay_ms: u64,
    /// `cooldown-expiry` (default) or `never`: whether to return to the primary model later.
    pub revert_policy: String,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 10,
            base_delay_ms: 500,
            max_delay_ms: 300_000,
            revert_policy: "cooldown-expiry".into(),
        }
    }
}

impl RetryConfig {
    /// Legacy YAML reader, kept for its unit test; production uses
    /// [`Self::from_settings`].
    #[cfg(test)]
    fn from_yaml(retry: Option<&serde_yaml_ng::Value>) -> Self {
        let mut c = Self::default();
        let Some(r) = retry else { return c };
        if let Some(v) = r.get("enabled").and_then(serde_yaml_ng::Value::as_bool) {
            c.enabled = v;
        }
        if let Some(v) = r.get("maxRetries").and_then(serde_yaml_ng::Value::as_u64) {
            c.max_retries = u32::try_from(v).unwrap_or(u32::MAX);
        }
        if let Some(v) = r.get("baseDelayMs").and_then(serde_yaml_ng::Value::as_u64) {
            c.base_delay_ms = v;
        }
        if let Some(v) = r.get("maxDelayMs").and_then(serde_yaml_ng::Value::as_u64) {
            c.max_delay_ms = v;
        }
        if r.get("fallbackRevertPolicy")
            .and_then(serde_yaml_ng::Value::as_str)
            == Some("never")
        {
            c.revert_policy = "never".into();
        }
        c
    }

    /// Settings-backed construction: the `retry.*` schema defaults equal
    /// [`Self::default`], so only user-supplied (non-Default-provenance) keys
    /// override — replacing the old hand-rolled YAML read.
    pub(crate) fn from_settings(settings: &omp_settings::Settings) -> Self {
        let mut c = Self::default();
        let set = |id: &str| {
            settings
                .provenance(id)
                .is_some_and(|p| p != omp_settings::Provenance::Default)
        };
        if set("retry.enabled") {
            if let Some(v) = settings.get_bool("retry.enabled") {
                c.enabled = v;
            }
        }
        if set("retry.maxRetries") {
            if let Some(v) = settings
                .get_number("retry.maxRetries")
                .filter(|n| n.fract() == 0.0 && *n >= 0.0 && *n <= f64::from(u32::MAX))
            {
                c.max_retries = v as u32;
            }
        }
        if set("retry.baseDelayMs") {
            if let Some(v) = settings
                .get_number("retry.baseDelayMs")
                .filter(|n| n.fract() == 0.0 && *n >= 0.0 && *n <= u64::MAX as f64)
            {
                c.base_delay_ms = v as u64;
            }
        }
        if set("retry.maxDelayMs") {
            if let Some(v) = settings
                .get_number("retry.maxDelayMs")
                .filter(|n| n.fract() == 0.0 && *n >= 0.0 && *n <= u64::MAX as f64)
            {
                c.max_delay_ms = v as u64;
            }
        }
        if set("retry.fallbackRevertPolicy")
            && settings.get_string("retry.fallbackRevertPolicy").as_deref() == Some("never")
        {
            c.revert_policy = "never".into();
        }
        c
    }

    /// Capped exponential delay (`calculateRetryBackoffDelayMs`): `base * 2^(attempt-1)`,
    /// capped at 8 s, times `1 - jitter * 0.25` with `jitter` in `[0, 1)`.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn backoff_ms(&self, attempt: u32, jitter: f64) -> u64 {
        let capped = (self.base_delay_ms as f64
            * 2f64.powi(i32::try_from(attempt.saturating_sub(1).min(30)).unwrap_or(30)))
        .min(8_000.0);
        (capped * (1.0 - jitter.clamp(0.0, 1.0) * 0.25)) as u64
    }
}

/// Heuristic for omp's `AIError.retriable` on a failed stream: transient HTTP
/// statuses and transport errors retry (and may fail over); context overflow and
/// auth/validation errors do not. `status` is the provider's HTTP status when known.
#[must_use]
pub fn is_retryable_error(status: Option<u16>, message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    if [
        "context length",
        "context window",
        "prompt is too long",
        "maximum context",
        "too many tokens",
    ]
    .iter()
    .any(|p| m.contains(p))
    {
        return false;
    }
    let status = status.or_else(|| status_from_message(&m));
    if let Some(s) = status {
        return matches!(s, 408 | 409 | 425 | 429) || s >= 500;
    }
    [
        "rate limit",
        "overloaded",
        "timeout",
        "timed out",
        "connection reset",
        "connection closed",
        "connection refused",
        "broken pipe",
        "unexpected eof",
        "socket",
        "temporarily unavailable",
        "usage limit",
        "quota",
        "network",
        "stream ended",
    ]
    .iter()
    .any(|p| m.contains(p))
}

/// Guess an HTTP status from an error message, but only from explicit status
/// shapes ("http 429", "status: 500", "error code 503") or a provider-style
/// reason phrase ("503 Service Unavailable"). Bare digit runs are never a
/// status: ":443" is a port and "try again in 480 seconds" is a retry hint.
fn status_from_message(m: &str) -> Option<u16> {
    let in_range = |s: u16| (400..600).contains(&s);
    for (start, _) in m.match_indices("http") {
        let after = &m[start + 4..];
        // A URL scheme ("https://…") is never a status line.
        if after.starts_with("://") || after.starts_with("s://") {
            continue;
        }
        if let Some(s) = take_status(after).filter(|&s| in_range(s)) {
            return Some(s);
        }
    }
    for marker in ["status", "code"] {
        for (start, _) in m.match_indices(marker) {
            if let Some(s) = take_status(&m[start + marker.len()..]).filter(|&s| in_range(s)) {
                return Some(s);
            }
        }
    }
    // "<code> <Reason Phrase>" as HTTP servers and SDKs print it.
    const REASONS: [&str; 7] = [
        "too many requests",
        "internal server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "request timeout",
        "http/2 5",
    ];
    for reason in REASONS {
        if let Some(pos) = m.find(reason) {
            // Last digit run before the phrase; must be exactly 3 digits.
            let run: String = m[..pos]
                .trim_end_matches(|c: char| !c.is_ascii_digit())
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_digit())
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            if run.len() == 3 {
                if let Ok(s) = run.parse::<u16>() {
                    if in_range(s) {
                        return Some(s);
                    }
                }
            }
        }
    }
    None
}

/// First digit run of exactly 3 characters in `rest` (skipping separators).
fn take_status(rest: &str) -> Option<u16> {
    rest.split(|c: char| !c.is_ascii_digit())
        .filter(|tok| tok.len() == 3)
        .find_map(|tok| tok.parse().ok())
}

/// A parsed `provider/id[:level]` selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Selector {
    pub raw: String,
    pub provider: String,
    pub id: String,
    pub level: Option<ModelThinkingLevel>,
}

pub(crate) fn parse_selector(s: &str) -> Option<Selector> {
    let raw = s.trim();
    if raw.is_empty() {
        return None;
    }
    let (base, level) = crate::split_level(raw);
    let (provider, id) = base.split_once('/')?;
    if provider.is_empty() || id.is_empty() {
        return None;
    }
    let level = match level {
        Some(Sel::Level(l)) => Some(l),
        _ => None,
    };
    Some(Selector {
        raw: raw.to_string(),
        provider: provider.into(),
        id: id.into(),
        level,
    })
}

fn is_model_key(k: &str) -> bool {
    k.contains('/')
}

fn is_wildcard(k: &str) -> bool {
    k.ends_with("/*")
}

type ExistsFn = Arc<dyn Fn(&str, &str) -> bool + Send + Sync>;
type ClampFn =
    Arc<dyn Fn(&str, &str, Option<ModelThinkingLevel>) -> Option<ModelThinkingLevel> + Send + Sync>;

/// Model lookups the resolution needs from the catalog: whether a model exists
/// (aggregator -> direct id rewrite) and how a thinking level clamps on a model
/// (`normalized` effort matching, `resolveThinkingLevelForModel`).
#[derive(Clone)]
pub struct ModelLookup {
    exists: ExistsFn,
    clamp: ClampFn,
}

impl fmt::Debug for ModelLookup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ModelLookup")
    }
}

impl ModelLookup {
    /// Lookup with an explicit clamp (`provider`, `id`, requested level -> effective level).
    pub fn new(
        exists: impl Fn(&str, &str) -> bool + Send + Sync + 'static,
        clamp: impl Fn(&str, &str, Option<ModelThinkingLevel>) -> Option<ModelThinkingLevel>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            exists: Arc::new(exists),
            clamp: Arc::new(clamp),
        }
    }

    /// Existence-only lookup: levels never clamp, so only identical efforts match.
    pub fn exists_only(exists: impl Fn(&str, &str) -> bool + Send + Sync + 'static) -> Self {
        Self::new(exists, |_, _, l| l)
    }

    /// Lookup backed by the resolver's model sources for agent dir `dir`
    /// (models.yml, models.db cache, built-in catalog); the config loads lazily.
    #[must_use]
    pub(crate) fn catalog(dir: &Path) -> Self {
        let dir = dir.to_path_buf();
        let cfg: Arc<OnceLock<Option<crate::Config>>> = Arc::new(OnceLock::new());
        let find = {
            let (dir, cfg) = (dir, cfg);
            move |p: &str, i: &str| {
                let cfg = cfg
                    .get_or_init(|| crate::Config::load(&dir).ok())
                    .as_ref()?;
                crate::find_model(&dir, cfg, p, i).ok()
            }
        };
        let find = Arc::new(find);
        let exists = Arc::clone(&find);
        Self::new(
            move |p, i| exists(p, i).is_some(),
            move |p, i, l| {
                let level = l?;
                let (model, meta) = find(p, i)?;
                if !model.reasoning {
                    return None;
                }
                let default = meta
                    .as_ref()
                    .and_then(|m| m.get("defaultLevel"))
                    .and_then(serde_json::Value::as_str)
                    .and_then(crate::effort);
                crate::clamp_level(level, meta.as_ref(), default)
            },
        )
    }
}

/// How a chain key's primary selector matches the live selector
/// (`selectorMatchKind`), strongest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Match {
    /// Same provider/model and effort.
    Exact,
    /// Same provider/model; both efforts clamp to the same level on the model.
    Normalized,
    /// Suffixless key naming the model: applies at any effort.
    Base,
    /// Same model at a distinct explicit effort (role keys only, weakest tier).
    Effort,
    None,
}

/// Knobs of candidate selection (`findRetryFallbackCandidates` options plus the
/// turn-recovery filters).
#[derive(Debug, Clone, Copy, Default)]
pub struct CandidateOptions<'a> {
    /// Chain role without an assigned model still walks from the live selector.
    pub allow_missing_primary: bool,
    /// Also return the entries before the live one (never the live one itself).
    pub wrap_around: bool,
    /// Drop candidates of this provider (long usage-limit fallback).
    pub exclude_provider: Option<&'a str>,
}

/// Everything fallback resolution needs, detached from the filesystem.
#[derive(Debug, Clone, Default)]
pub struct FallbackConfig {
    /// Configured chains in file order (order is meaningful: model keys are searched in order).
    pub chains: Vec<(String, Vec<String>)>,
    pub roles: BTreeMap<String, String>,
    pub known_providers: BTreeSet<String>,
    /// Catalog access for the aggregator -> direct id rewrite and `normalized`
    /// effort matching; `None` = unknown (no rewrite, no normalization).
    pub model_exists: Option<ModelLookup>,
}

impl FallbackConfig {
    /// Chains with the `default` chain applied to every non-kind role lacking its own.
    fn expanded(&self) -> Vec<(String, Vec<String>)> {
        let mut out = self.chains.clone();
        let Some(default) = self
            .chains
            .iter()
            .find(|(k, _)| k == "default")
            .map(|(_, v)| v.clone())
        else {
            return out;
        };
        for role in self.roles.keys() {
            if role != "default"
                && !KIND_ROLES.contains(&role.as_str())
                && !out.iter().any(|(k, _)| k == role)
            {
                out.push((role.clone(), default.clone()));
            }
        }
        out
    }

    fn wildcard(&self, key: &str) -> (String, Option<String>) {
        let template = &key[..key.len() - 2];
        match template.split_once('/') {
            Some((p, rest)) if !self.known_providers.contains(template) => {
                (p.to_string(), Some(rest.to_string()))
            }
            _ => (template.to_string(), None),
        }
    }

    fn primary(&self, key: &str) -> Option<Selector> {
        if is_wildcard(key) {
            return None;
        }
        if is_model_key(key) {
            return parse_selector(key);
        }
        parse_selector(self.roles.get(key)?)
    }

    fn match_kind(&self, primary: Option<&Selector>, cur: &Selector) -> Match {
        let Some(p) = primary else { return Match::None };
        if p.provider != cur.provider || p.id != cur.id {
            return Match::None;
        }
        if p.level == cur.level {
            return Match::Exact;
        }
        if p.level.is_none() {
            return Match::Base;
        }
        let normalized = self.model_exists.as_ref().is_some_and(|m| {
            (m.clamp)(&cur.provider, &cur.id, p.level)
                == (m.clamp)(&cur.provider, &cur.id, cur.level)
        });
        if normalized {
            Match::Normalized
        } else {
            Match::Effort
        }
    }

    /// Which chain key serves `current` (a `provider/id[:level]` selector).
    #[must_use]
    pub fn chain_key(&self, current: &str, role_hint: Option<&str>) -> Option<String> {
        let chains = self.expanded();
        let has = |k: &str| chains.iter().any(|(ck, _)| ck == k);
        let Some(cur) = parse_selector(current) else {
            return role_hint.filter(|r| has(r)).map(str::to_string);
        };
        // 1. model-selector keys: exact effort beats model-normalized effort beats a
        //    suffixless key, regardless of order; distinct efforts never match.
        let (mut normalized_key, mut base_key) = (None, None);
        for (key, _) in &chains {
            if !is_model_key(key) || is_wildcard(key) {
                continue;
            }
            match self.match_kind(self.primary(key).as_ref(), &cur) {
                Match::Exact => return Some(key.clone()),
                Match::Normalized => {
                    normalized_key.get_or_insert_with(|| key.clone());
                }
                Match::Base => {
                    base_key.get_or_insert_with(|| key.clone());
                }
                Match::Effort | Match::None => {}
            }
        }
        if normalized_key.is_some() {
            return normalized_key;
        }
        if base_key.is_some() {
            return base_key;
        }
        // 2. provider wildcards, longest id prefix wins.
        let mut best: Option<(&String, usize)> = None;
        for (key, _) in &chains {
            if !is_wildcard(key) {
                continue;
            }
            let (provider, prefix) = self.wildcard(key);
            if provider != cur.provider
                || prefix
                    .as_deref()
                    .is_some_and(|p| !cur.id.starts_with(&format!("{p}/")))
            {
                continue;
            }
            let len = prefix.map_or(0, |p| p.len());
            if best.is_none_or(|(_, l)| len > l) {
                best = Some((key, len));
            }
        }
        if let Some((k, _)) = best {
            return Some(k.clone());
        }
        // 3. hinted role, then roles assigned this model (`default` first, effort-mismatch last).
        if let Some(h) = role_hint.filter(|r| has(r)) {
            return Some(h.to_string());
        }
        let (mut matched, mut effort_only) = (None, None);
        for (key, _) in &chains {
            if is_model_key(key) {
                continue;
            }
            match self.match_kind(self.primary(key).as_ref(), &cur) {
                Match::None => {}
                Match::Effort => {
                    if key == "default" || effort_only.is_none() {
                        effort_only = Some(key.clone());
                    }
                }
                Match::Exact | Match::Normalized | Match::Base => {
                    if key == "default" {
                        return Some("default".into());
                    }
                    matched.get_or_insert_with(|| key.clone());
                }
            }
        }
        if matched.is_some() {
            return matched;
        }
        if effort_only.is_some() {
            return effort_only;
        }
        // 4. default chain, even when its primary is another model.
        chains
            .iter()
            .find(|(k, v)| k == "default" && !v.is_empty())
            .map(|(k, _)| k.clone())
    }

    fn chain_entry(&self, entry: &str, cur: Option<&Selector>) -> Option<Selector> {
        if !is_wildcard(entry) {
            return parse_selector(entry);
        }
        let cur = cur?;
        let (provider, prefix) = self.wildcard(entry);
        let bare = cur.id.rsplit('/').next().unwrap_or(&cur.id).to_string();
        let exists =
            |p: &str, i: &str| self.model_exists.as_ref().is_some_and(|m| (m.exists)(p, i));
        let id = match prefix {
            Some(p) => format!("{p}/{bare}"),
            // Aggregator -> direct: the failing id carries a vendor prefix the target provider
            // does not use (openrouter/google/x -> google-vertex/x).
            None if bare != cur.id && !exists(&provider, &cur.id) && exists(&provider, &bare) => {
                bare
            }
            None => cur.id.clone(),
        };
        Some(Selector {
            raw: format!("{provider}/{id}"),
            provider,
            id,
            level: None,
        })
    }

    /// Selectors to try after `current` fails, in order. Empty when no chain applies.
    #[must_use]
    pub fn candidates(&self, current: &str, role_hint: Option<&str>) -> Vec<String> {
        let Some(key) = self.chain_key(current, role_hint) else {
            return Vec::new();
        };
        self.candidates_for_key(&key, current, CandidateOptions::default())
    }

    /// Candidates of one chain `key` after `current` (`findRetryFallbackCandidates`).
    #[must_use]
    pub fn candidates_for_key(
        &self,
        key: &str,
        current: &str,
        opts: CandidateOptions<'_>,
    ) -> Vec<String> {
        let chains = self.expanded();
        let entries = chains
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_slice())
            .unwrap_or_default();
        let cur = parse_selector(current);
        let mut seen = BTreeSet::new();
        let mut chain: Vec<Selector> = Vec::new();
        if is_wildcard(key) {
            if let Some(c) = &cur {
                seen.insert(c.raw.clone());
                chain.push(c.clone());
            }
        } else if let Some(p) = self.primary(key) {
            seen.insert(p.raw.clone());
            chain.push(p);
        } else if (key == "default" || opts.allow_missing_primary) && cur.is_some() {
            if let Some(c) = &cur {
                seen.insert(c.raw.clone());
                chain.push(c.clone());
            }
        } else if !opts.allow_missing_primary {
            return Vec::new();
        }
        for e in entries {
            if let Some(p) = self.chain_entry(e, cur.as_ref()) {
                if seen.insert(p.raw.clone()) {
                    chain.push(p);
                }
            }
        }
        let Some(cur) = cur else {
            return chain.into_iter().map(|s| s.raw).collect();
        };
        if chain.len() <= 1 {
            return Vec::new();
        }
        let idx = chain.iter().position(|s| s.raw == cur.raw).or_else(|| {
            chain
                .iter()
                .position(|s| s.provider == cur.provider && s.id == cur.id)
        });
        match idx {
            Some(i) => {
                let mut out: Vec<String> = chain[i + 1..].iter().map(|s| s.raw.clone()).collect();
                if opts.wrap_around {
                    out.extend(chain[..i].iter().map(|s| s.raw.clone()));
                }
                out
            }
            None => chain.into_iter().map(|s| s.raw).collect(),
        }
    }
}

/// Iterator-style walker over the fallback candidates of one model. Call
/// [`FallbackWalker::next_model`] each time a turn fails with a retryable error;
/// unresolvable (unknown model, missing credentials) candidates are skipped and
/// recorded in `skipped`.
#[derive(Debug)]
pub struct FallbackWalker {
    dir: crate::AgentDir,
    remaining: VecDeque<String>,
    /// `(selector, reason)` of candidates passed over.
    pub skipped: Vec<(String, String)>,
}

impl FallbackWalker {
    #[must_use]
    pub(crate) fn new(dir: crate::AgentDir, selectors: Vec<String>) -> Self {
        Self {
            dir,
            remaining: selectors.into(),
            skipped: Vec::new(),
        }
    }

    /// Remaining candidate selectors (not yet resolved).
    #[must_use]
    pub fn pending(&self) -> Vec<String> {
        self.remaining.iter().cloned().collect()
    }

    /// Next usable fallback model, or `None` when the chain is exhausted.
    pub fn next_model(&mut self) -> Option<crate::ResolvedModel> {
        while let Some(sel) = self.remaining.pop_front() {
            match crate::resolve_in(&self.dir, &sel) {
                Ok(m) if m.api_key.is_some() || m.api == "local-inference" => return Some(m),
                Ok(_) => self.skipped.push((sel, "no credentials".into())),
                Err(e) => self.skipped.push((sel, e.to_string())),
            }
        }
        None
    }
}

/// Why a provider refused a request (`parseRateLimitReason`); drives the cooldown
/// a failing selector sits out before it may be restored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitReason {
    QuotaExhausted,
    RateLimitExceeded,
    ConcurrentLimit,
    ModelCapacityExhausted,
    ServerError,
    Unknown,
}

/// Classify an error message (the phrase families of omp's `parseRateLimitReason`
/// that are provider-independent; Google RPC / CN / provider-specific patterns
/// are not ported).
#[must_use]
pub fn parse_rate_limit_reason(message: &str) -> RateLimitReason {
    let lower = message.to_lowercase();
    let has = |needles: &[&str]| needles.iter().any(|n| lower.contains(n));
    if has(&["quota will reset", "exhausted your capacity"]) {
        return RateLimitReason::QuotaExhausted;
    }
    if has(&["concurrent", "too many concurrent"]) {
        return RateLimitReason::ConcurrentLimit;
    }
    if has(&["capacity", "overloaded", "529", "503"]) {
        return RateLimitReason::ModelCapacityExhausted;
    }
    if has(&[
        "per minute",
        "rate limit",
        "rate_limit",
        "too many requests",
        "tpm",
        "rpm",
    ]) {
        return RateLimitReason::RateLimitExceeded;
    }
    if has(&[
        "exhausted",
        "quota",
        "usage limit",
        "usage_limit",
        "out of credits",
        "run out of credits",
        "spending limit",
        "spending-limit",
        "insufficient balance",
        "insufficient_balance",
        "access_terminated_error",
    ]) {
        return RateLimitReason::QuotaExhausted;
    }
    if has(&["500", "internal error", "internal server error"]) {
        return RateLimitReason::ServerError;
    }
    RateLimitReason::Unknown
}

/// `calculateRateLimitBackoffMs`; `rand01` (in `[0, 1)`) drives the capacity jitter.
#[must_use]
pub fn rate_limit_backoff_ms(reason: RateLimitReason, rand01: f64) -> u64 {
    match reason {
        RateLimitReason::RateLimitExceeded => 30_000,
        RateLimitReason::ConcurrentLimit => 5_000,
        // 45 s base + up to 30 s jitter
        RateLimitReason::ModelCapacityExhausted => {
            45_000 + (rand01.clamp(0.0, 1.0) * 30_000.0) as u64
        }
        RateLimitReason::ServerError => 20_000,
        RateLimitReason::QuotaExhausted | RateLimitReason::Unknown => 30 * 60_000,
    }
}

/// Cooldown for a failed selector: the provider's retry hint if positive, else 5 min
/// for unclassified errors, else the reason's backoff (`noteRetryFallbackCooldown`).
#[must_use]
pub fn fallback_cooldown_ms(retry_after_ms: Option<u64>, message: &str, rand01: f64) -> u64 {
    if let Some(ms) = retry_after_ms.filter(|ms| *ms > 0) {
        return ms;
    }
    match parse_rate_limit_reason(message) {
        RateLimitReason::Unknown => 5 * 60_000,
        r => rate_limit_backoff_ms(r, rand01),
    }
}

/// `retry.fallbackRevertPolicy`: whether the primary is restored once its cooldown expires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RevertPolicy {
    Never,
    #[default]
    CooldownExpiry,
}

impl RevertPolicy {
    /// omp reads anything but `never` as `cooldown-expiry`.
    #[must_use]
    pub fn from_setting(value: Option<&str>) -> Self {
        if value == Some("never") {
            Self::Never
        } else {
            Self::CooldownExpiry
        }
    }
}

/// A fallback switch that stays on record until the primary is restored
/// (`ActiveRetryFallbackState`). Thinking levels are configured-level tokens
/// (`"high"`, `"auto"`, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveFallback {
    /// Chain key that produced the switch (role name or model-selector key).
    pub role: String,
    pub original_selector: String,
    pub original_thinking: Option<String>,
    pub last_applied_thinking: Option<String>,
    pub pinned: bool,
    /// Set once a turn on the fallback target settled successfully; until then the
    /// switch is only a routing decision.
    pub served: bool,
}

/// What to switch back to when the primary's cooldown has expired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestorePlan {
    pub selector: String,
    pub thinking: Option<String>,
}

/// Runtime fallback state of one session: selector cooldowns (omp's
/// `suppressSelector`), the active fallback and its served flag, and the
/// revert-to-primary decision. Time is passed in (`now_ms`, epoch ms).
#[derive(Debug, Default)]
pub struct FallbackState {
    policy: RevertPolicy,
    suppressed: HashMap<String, u64>,
    active: Option<ActiveFallback>,
}

/// Cooldown key of a selector: `provider/id` without thinking suffix.
fn suppression_key(selector: &str) -> String {
    parse_selector(selector).map_or_else(
        || selector.trim().to_string(),
        |s| format!("{}/{}", s.provider, s.id),
    )
}

impl FallbackState {
    #[must_use]
    pub fn new(policy: RevertPolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn active(&self) -> Option<&ActiveFallback> {
        self.active.as_ref()
    }

    /// Drop fallback ownership after an explicit model change or a restore.
    pub fn clear_active(&mut self) {
        self.active = None;
    }

    /// Suppress `selector` until `until_ms`.
    pub fn suppress(&mut self, selector: &str, until_ms: u64) {
        self.suppressed.insert(suppression_key(selector), until_ms);
    }

    #[must_use]
    pub fn is_suppressed(&self, selector: &str, now_ms: u64) -> bool {
        self.suppressed
            .get(&suppression_key(selector))
            .is_some_and(|until| *until > now_ms)
    }

    /// Clear one selector's cooldown after an explicit user selection.
    pub fn clear_suppressed(&mut self, selector: &str) {
        self.suppressed.remove(&suppression_key(selector));
    }

    pub fn clear_all_suppressed(&mut self) {
        self.suppressed.clear();
    }

    /// Record the cooldown of a failing selector; returns its length in ms.
    pub fn note_cooldown(
        &mut self,
        current_selector: &str,
        retry_after_ms: Option<u64>,
        message: &str,
        now_ms: u64,
        rand01: f64,
    ) -> u64 {
        let ms = fallback_cooldown_ms(retry_after_ms, message, rand01);
        self.suppress(current_selector, now_ms.saturating_add(ms));
        ms
    }

    /// Chain keys to consult for `current`, most specific walk first: the chain that
    /// owns the active walk, then the chain the current model owns when different
    /// (`retryFallbackChainKeys`).
    #[must_use]
    pub fn chain_keys(
        &self,
        cfg: &FallbackConfig,
        current: &str,
        role_hint: Option<&str>,
    ) -> Vec<String> {
        let pinned = self.active.as_ref().map(|a| a.role.clone());
        let live = cfg.chain_key(current, role_hint);
        match (pinned, live) {
            (None, live) => live.into_iter().collect(),
            (Some(p), Some(l)) if l != p => vec![p, l],
            (Some(p), _) => vec![p],
        }
    }

    /// Fallback selectors to try after `current` failed, skipping cooled-down
    /// selectors and `opts.exclude_provider`; walk order follows [`Self::chain_keys`].
    #[must_use]
    pub fn candidates(
        &self,
        cfg: &FallbackConfig,
        current: &str,
        role_hint: Option<&str>,
        opts: CandidateOptions<'_>,
        now_ms: u64,
    ) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for key in self.chain_keys(cfg, current, role_hint) {
            for sel in cfg.candidates_for_key(&key, current, opts) {
                let excluded = opts
                    .exclude_provider
                    .is_some_and(|p| parse_selector(&sel).is_some_and(|s| s.provider == p));
                if !excluded && !self.is_suppressed(&sel, now_ms) && !out.contains(&sel) {
                    out.push(sel);
                }
            }
        }
        out
    }

    /// Record a committed switch onto a fallback. `role` is the chain key walked,
    /// `from_selector`/`from_thinking` describe the model being left. The first switch
    /// pins the original; later hops only update the applied level. The new target is
    /// not `served` until [`Self::mark_served`].
    pub fn apply(
        &mut self,
        role: &str,
        from_selector: &str,
        from_thinking: Option<String>,
        applied_thinking: Option<String>,
        pin: bool,
    ) {
        match &mut self.active {
            None => {
                self.active = Some(ActiveFallback {
                    role: role.to_string(),
                    original_selector: from_selector.to_string(),
                    original_thinking: from_thinking,
                    last_applied_thinking: applied_thinking,
                    pinned: pin,
                    served: false,
                });
            }
            Some(a) => {
                a.last_applied_thinking = applied_thinking;
                a.pinned = a.pinned || pin;
                a.served = false;
            }
        }
    }

    /// A turn on the fallback target settled. `true` exactly once per switch (the
    /// one-shot `retry_fallback_succeeded` moment).
    pub fn mark_served(&mut self) -> bool {
        match &mut self.active {
            Some(a) if !a.served => {
                a.served = true;
                true
            }
            _ => false,
        }
    }

    /// Whether the session is running on a fallback that has not been restored.
    #[must_use]
    pub fn is_fallback_active(&self) -> bool {
        self.active.is_some()
    }

    /// Decide whether to switch back to the primary (`maybeRestoreRetryFallbackPrimary`,
    /// first half): policy and pin checks, then the cooldown of the original selector.
    /// The caller resolves the primary model and its credentials, switches, and calls
    /// [`Self::clear_active`] (before the swap, so observers never see the restored
    /// primary tagged as fallback-served). `current_thinking` is the configured level now.
    pub fn restore_plan(
        &mut self,
        current_selector: &str,
        current_thinking: Option<&str>,
        now_ms: u64,
    ) -> Option<RestorePlan> {
        let a = self.active.as_ref()?;
        if a.pinned || self.policy != RevertPolicy::CooldownExpiry {
            return None;
        }
        let Some(original) = parse_selector(&a.original_selector) else {
            self.active = None;
            return None;
        };
        let suppressed = self.is_suppressed(&original.raw, now_ms);
        if current_selector == original.raw {
            if !suppressed {
                self.active = None;
            }
            return None;
        }
        if suppressed {
            return None;
        }
        let thinking = if current_thinking == a.last_applied_thinking.as_deref() {
            a.original_thinking.clone()
        } else {
            current_thinking.map(str::to_string)
        };
        Some(RestorePlan {
            selector: original.raw,
            thinking,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(chains: &[(&str, &[&str])], roles: &[(&str, &str)]) -> FallbackConfig {
        FallbackConfig {
            chains: chains
                .iter()
                .map(|(k, v)| ((*k).into(), v.iter().map(|s| (*s).into()).collect()))
                .collect(),
            roles: roles
                .iter()
                .map(|(k, v)| ((*k).into(), (*v).into()))
                .collect(),
            known_providers: ["zai", "openrouter", "anthropic"]
                .iter()
                .map(|s| (*s).into())
                .collect(),
            model_exists: None,
        }
    }

    #[test]
    fn role_chain_candidates_follow_current() {
        let c = cfg(
            &[("judge", &["commandcode/typesafe/jev", "zai/glm-5.3-flash"])],
            &[("judge", "openrouter/typesafe/jev-1.13")],
        );
        assert_eq!(
            c.chain_key("openrouter/typesafe/jev-1.13", None).as_deref(),
            Some("judge")
        );
        assert_eq!(
            c.candidates("openrouter/typesafe/jev-1.13", None),
            ["commandcode/typesafe/jev", "zai/glm-5.3-flash"]
        );
        assert_eq!(
            c.candidates("commandcode/typesafe/jev", Some("judge")),
            ["zai/glm-5.3-flash"]
        );
        assert!(c.candidates("zai/glm-5.3-flash", Some("judge")).is_empty());
    }

    #[test]
    fn default_chain_expands_to_roles_but_not_kind_roles() {
        let c = cfg(
            &[("default", &["zai/a", "zai/b"])],
            &[
                ("default", "anthropic/x:medium"),
                ("smol", "zai/s"),
                ("judge", "zai/j"),
            ],
        );
        assert_eq!(c.chain_key("zai/s:minimal", None).as_deref(), Some("smol"));
        assert_eq!(c.candidates("zai/s", None), ["zai/a", "zai/b"]);
        // judge is a kind role: no inherited chain; unknown model falls through to default chain
        assert_eq!(c.chain_key("zai/j", None).as_deref(), Some("default"));
        assert_eq!(c.candidates("anthropic/x", None), ["zai/a", "zai/b"]);
    }

    #[test]
    fn model_key_and_wildcards() {
        let c = cfg(
            &[
                ("zai/*", &["openrouter/z-ai/*"]),
                ("zai/g:high", &["anthropic/c"]),
                ("default", &["anthropic/d"]),
            ],
            &[],
        );
        assert_eq!(
            c.chain_key("zai/g:high", None).as_deref(),
            Some("zai/g:high")
        );
        assert_eq!(c.chain_key("zai/g:low", None).as_deref(), Some("zai/*"));
        assert_eq!(c.candidates("zai/g:low", None), ["openrouter/z-ai/g"]);
        assert_eq!(c.candidates("zai/g:high", None), ["anthropic/c"]);
    }

    #[test]
    fn retry_config_and_backoff() {
        let y: serde_yaml_ng::Value =
            serde_yaml_ng::from_str("maxRetries: 3\nmaxDelayMs: 10000").unwrap();
        let r = RetryConfig::from_yaml(Some(&y));
        assert_eq!(
            (r.max_retries, r.base_delay_ms, r.max_delay_ms),
            (3, 500, 10_000)
        );
        assert_eq!(r.backoff_ms(1, 0.0), 500);
        assert_eq!(r.backoff_ms(3, 0.0), 2000);
        assert_eq!(r.backoff_ms(10, 0.0), 8000);
    }

    #[test]
    fn retryable_classification() {
        assert!(is_retryable_error(Some(429), "x"));
        assert!(is_retryable_error(None, "503 Service Unavailable"));
        assert!(is_retryable_error(None, "connection reset by peer"));
        assert!(!is_retryable_error(Some(401), "bad key"));
        assert!(!is_retryable_error(
            Some(400),
            "maximum context length exceeded"
        ));
    }

    /// Bare digit runs are not HTTP statuses: a port or a "N seconds" retry
    /// hint must not preempt the transport-keyword classification.
    #[test]
    fn retryable_status_guess_never_uses_bare_digits() {
        // ":443" is a port; the socket keyword makes this retryable.
        assert!(is_retryable_error(
            None,
            "fetch failed: socket hang up connecting to api.example.com:443"
        ));
        // "480 seconds" is a retry hint, not a status; no keyword -> not retryable
        // (previously misread as status 480, same outcome but for the wrong reason;
        // the important part is the guess must not fire).
        assert!(!status_guess_is_bare_digit("try again in 480 seconds"));
        assert!(!status_guess_is_bare_digit(
            "connection to api.example.com:443 failed"
        ));
        assert!(!status_guess_is_bare_digit(
            "via https://api.example.com:443/v1"
        ));
        // Explicit status shapes still classify.
        assert!(is_retryable_error(
            None,
            "provider returned http status 429, retry later"
        ));
        assert!(is_retryable_error(None, "error code: 503, backend down"));
        assert!(is_retryable_error(None, "500 Internal Server Error"));
        assert!(!is_retryable_error(
            None,
            "400 Bad Request: malformed tool call"
        ));
    }

    fn status_guess_is_bare_digit(message: &str) -> bool {
        super::status_from_message(&message.to_ascii_lowercase()).is_some()
    }

    fn lookup(
        models: &'static [(&'static str, &'static str)],
        cap: ModelThinkingLevel,
    ) -> ModelLookup {
        ModelLookup::new(
            move |p, i| models.iter().any(|(mp, mi)| *mp == p && *mi == i),
            move |_, _, l| l.map(|l| l.min(cap)),
        )
    }

    #[test]
    fn normalized_effort_beats_suffixless_key_and_matches_clamped_levels() {
        use ModelThinkingLevel as L;
        let mut c = cfg(
            &[
                ("zai/g", &["anthropic/base"]),
                ("zai/g:max", &["anthropic/norm"]),
            ],
            &[],
        );
        // no catalog: `max` and `high` stay distinct, the suffixless key serves
        assert_eq!(c.chain_key("zai/g:high", None).as_deref(), Some("zai/g"));
        // high-capped model: `max` normalizes to `high`, beating the suffixless key
        c.model_exists = Some(lookup(&[("zai", "g")], L::High));
        assert_eq!(
            c.chain_key("zai/g:high", None).as_deref(),
            Some("zai/g:max")
        );
        assert_eq!(c.candidates("zai/g:high", None), ["anthropic/norm"]);
        // an effort that clamps differently stays distinct
        assert_eq!(c.chain_key("zai/g:low", None).as_deref(), Some("zai/g"));
    }

    #[test]
    fn role_with_distinct_effort_is_weakest_tier() {
        let c = cfg(
            &[("a", &["zai/x"]), ("b", &["zai/y"])],
            &[("a", "zai/m:high"), ("b", "zai/m")],
        );
        // `b` is suffixless (base) and beats `a` whose explicit effort differs
        assert_eq!(c.chain_key("zai/m:low", None).as_deref(), Some("b"));
        let only_effort = cfg(&[("a", &["zai/x"])], &[("a", "zai/m:high")]);
        assert_eq!(
            only_effort.chain_key("zai/m:low", None).as_deref(),
            Some("a")
        );
    }

    #[test]
    fn wrap_around_appends_entries_before_current() {
        let c = cfg(&[("default", &["zai/b", "zai/c"])], &[("default", "zai/a")]);
        let wrap = CandidateOptions {
            wrap_around: true,
            ..CandidateOptions::default()
        };
        assert_eq!(
            c.candidates_for_key("default", "zai/b", CandidateOptions::default()),
            ["zai/c"]
        );
        assert_eq!(
            c.candidates_for_key("default", "zai/b", wrap),
            ["zai/c", "zai/a"]
        );
        assert_eq!(
            c.candidates_for_key("default", "zai/c", wrap),
            ["zai/a", "zai/b"]
        );
        // current is never returned
        assert!(!c
            .candidates_for_key("default", "zai/a", wrap)
            .contains(&"zai/a".to_string()));
    }

    #[test]
    fn missing_primary_needs_default_or_allow_flag() {
        let c = cfg(&[("smol", &["zai/b"])], &[]);
        assert!(c
            .candidates_for_key("smol", "zai/a", CandidateOptions::default())
            .is_empty());
        let allow = CandidateOptions {
            allow_missing_primary: true,
            ..CandidateOptions::default()
        };
        assert_eq!(c.candidates_for_key("smol", "zai/a", allow), ["zai/b"]);
    }

    #[test]
    fn aggregator_to_direct_id_rewrite_uses_catalog() {
        let mut c = cfg(&[("openrouter/*", &["zai/*", "anthropic/*"])], &[]);
        // without a catalog the id is kept verbatim
        assert_eq!(
            c.candidates("openrouter/google/gem", None),
            ["zai/google/gem", "anthropic/google/gem"]
        );
        c.model_exists = Some(ModelLookup::exists_only(|p, i| (p, i) == ("zai", "gem")));
        // zai lacks `google/gem` but has `gem`: rewritten; anthropic has neither: kept
        assert_eq!(
            c.candidates("openrouter/google/gem", None),
            ["zai/gem", "anthropic/google/gem"]
        );
        // a target that has the prefixed id keeps it
        c.model_exists = Some(ModelLookup::exists_only(|p, i| {
            p == "zai" && (i == "gem" || i == "google/gem")
        }));
        assert_eq!(
            c.candidates("openrouter/google/gem", None)[0],
            "zai/google/gem"
        );
    }

    #[test]
    fn cooldown_durations_follow_rate_limit_reason() {
        assert_eq!(fallback_cooldown_ms(Some(1234), "anything", 0.0), 1234);
        assert_eq!(fallback_cooldown_ms(Some(0), "weird failure", 0.0), 300_000);
        assert_eq!(
            fallback_cooldown_ms(None, "429 rate limit exceeded", 0.0),
            30_000
        );
        assert_eq!(fallback_cooldown_ms(None, "quota exceeded", 0.0), 1_800_000);
        assert_eq!(fallback_cooldown_ms(None, "model overloaded", 0.0), 45_000);
        assert_eq!(fallback_cooldown_ms(None, "model overloaded", 0.5), 60_000);
        assert_eq!(
            fallback_cooldown_ms(None, "internal server error", 0.0),
            20_000
        );
        assert_eq!(
            parse_rate_limit_reason("Your quota will reset after 3h"),
            RateLimitReason::QuotaExhausted
        );
    }

    #[test]
    fn state_cooldown_filters_candidates_and_excluded_provider() {
        let c = cfg(
            &[("default", &["zai/b", "anthropic/c", "zai/d"])],
            &[("default", "zai/a")],
        );
        let mut st = FallbackState::new(RevertPolicy::CooldownExpiry);
        st.note_cooldown("zai/a:high", Some(1000), "x", 10, 0.0);
        assert!(st.is_suppressed("zai/a", 500));
        assert!(!st.is_suppressed("zai/a", 1010));
        st.suppress("zai/b", 100);
        let got = st.candidates(&c, "zai/a", None, CandidateOptions::default(), 50);
        assert_eq!(got, ["anthropic/c", "zai/d"]);
        let excl = CandidateOptions {
            exclude_provider: Some("zai"),
            ..CandidateOptions::default()
        };
        assert_eq!(st.candidates(&c, "zai/a", None, excl, 50), ["anthropic/c"]);
        // cooldown expired: b is back
        assert_eq!(
            st.candidates(&c, "zai/a", None, CandidateOptions::default(), 200)[0],
            "zai/b"
        );
    }

    #[test]
    fn active_walk_chain_is_pinned_and_current_chain_consulted() {
        let c = cfg(
            &[("default", &["zai/b"]), ("zai/b", &["anthropic/c"])],
            &[("default", "zai/a")],
        );
        let mut st = FallbackState::new(RevertPolicy::default());
        assert_eq!(st.chain_keys(&c, "zai/a", None), ["default"]);
        st.apply(
            "default",
            "zai/a",
            Some("high".into()),
            Some("high".into()),
            false,
        );
        // landed on zai/b, which owns its own chain: walk pinned chain then b's chain
        assert_eq!(st.chain_keys(&c, "zai/b", None), ["default", "zai/b"]);
        assert_eq!(
            st.candidates(&c, "zai/b", None, CandidateOptions::default(), 0),
            ["anthropic/c"]
        );
    }

    #[test]
    fn served_flag_is_one_shot_per_switch() {
        let mut st = FallbackState::new(RevertPolicy::default());
        assert!(!st.mark_served());
        st.apply("default", "zai/a", None, None, false);
        assert!(!st.active().unwrap().served);
        assert!(st.mark_served());
        assert!(!st.mark_served());
        // next hop re-arms; pin is sticky; original stays
        st.apply(
            "default",
            "zai/b",
            Some("low".into()),
            Some("low".into()),
            true,
        );
        let a = st.active().unwrap();
        assert_eq!(
            (a.original_selector.as_str(), a.pinned, a.served),
            ("zai/a", true, false)
        );
        assert!(st.mark_served());
    }

    #[test]
    fn revert_to_primary_after_cooldown_expiry() {
        let mut st = FallbackState::new(RevertPolicy::CooldownExpiry);
        assert!(st.restore_plan("zai/b", None, 0).is_none()); // nothing active
        st.apply(
            "default",
            "zai/a:high",
            Some("high".into()),
            Some("low".into()),
            false,
        );
        st.suppress("zai/a:high", 1000);
        assert!(st.restore_plan("zai/b:low", Some("low"), 500).is_none()); // still cooling
                                                                           // cooldown over; level unchanged since the switch -> original level comes back
        let plan = st.restore_plan("zai/b:low", Some("low"), 1000).unwrap();
        assert_eq!(
            (plan.selector.as_str(), plan.thinking.as_deref()),
            ("zai/a:high", Some("high"))
        );
        assert!(st.is_fallback_active());
        st.clear_active();
        assert!(!st.is_fallback_active());
        // user changed effort after the switch -> kept
        st.apply(
            "default",
            "zai/a",
            Some("high".into()),
            Some("low".into()),
            false,
        );
        let plan = st.restore_plan("zai/b", Some("max"), 1000).unwrap();
        assert_eq!(plan.thinking.as_deref(), Some("max"));
    }

    #[test]
    fn revert_respects_policy_pin_and_already_on_primary() {
        let mut never = FallbackState::new(RevertPolicy::from_setting(Some("never")));
        never.apply("default", "zai/a", None, None, false);
        assert!(never.restore_plan("zai/b", None, 0).is_none());
        let mut pinned = FallbackState::new(RevertPolicy::from_setting(None));
        pinned.apply("default", "zai/a", None, None, true);
        assert!(pinned.restore_plan("zai/b", None, 0).is_none());
        // already on the primary: record is dropped once it is no longer cooling down
        let mut st = FallbackState::new(RevertPolicy::CooldownExpiry);
        st.apply("default", "zai/a", None, None, false);
        st.suppress("zai/a", 100);
        assert!(st.restore_plan("zai/a", None, 50).is_none());
        assert!(st.is_fallback_active());
        assert!(st.restore_plan("zai/a", None, 100).is_none());
        assert!(!st.is_fallback_active());
    }
}
