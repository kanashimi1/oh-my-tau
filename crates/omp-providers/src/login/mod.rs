//! `omp login` / `omp logout` / `/login` / `/logout`: provider flows plus credential storage.
//!
//! Mirrors TS `AuthStorage.oauth.login` (`pi-ai/src/auth/oauth.ts`): run the provider's flow,
//! then `upsertOAuth` (OAuth grants, stored under `storeCredentialsAs ?? id`, `authorizedAt`
//! stamped) or `storeLoginApiKey` (`{key, source:"login"}`) into the local `agent.db`, or upload
//! to the auth broker when one is configured. Logout soft-deletes rows (`disabled_cause`).
//!
//! Flows: the OAuth flows are `pa_ai::oauth` (anthropic, openai-codex, github-copilot, xai-oauth,
//! kimi-code); every `login "api-key"` provider of the TS rules runs through [`api_key`]; the
//! endpoint-choice key prompts ([`custom`]) cover `alibaba-coding-plan` / `alibaba-token-plan`.
//! The flows in this file cover cursor, muse-code, perplexity, the Google
//! loopback flows (gemini-cli, antigravity, devin, stencil) and the Codex
//! device flow. Every TS login flow is ported; unknown ids fail with a clear message.

pub mod api_key;
pub mod cloudflare;
pub mod codex_device;
pub mod cursor;
pub mod custom;
pub mod factory;
pub mod gitlab;
pub mod kilo;
pub mod muse_code;
pub mod oauth_code;
pub mod perplexity;
pub mod providers;
pub mod store;
pub mod xiaomi;

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use pa_ai::oauth::{
    CodexHttp, CodexLoginUi, OAuthLoginUi, OAuthPrompt, ProviderHttp, ProviderHttpMethod,
    ProviderHttpRequest, ReqwestCodexHttp, ReqwestProviderHttp,
};
use serde_json::{Map, Value};

pub use store::{
    broker_client_usage_summary, broker_usage_history, BrokerClientProviderUsage,
    BrokerClientUsageClientSummary, BrokerUsageHistoryEntry, Credential, LocalStore, StoredRow,
};

use crate::auth::{resolve_broker_config, BrokerConfig, Hosts};
use crate::AgentDir;

/// Where credentials live: the local `agent.db` or the configured auth broker.
#[derive(Debug, Clone)]
pub enum Target {
    Local(PathBuf),
    Broker(BrokerConfig),
}

impl Target {
    /// TS `discoverAuthStorage`: broker config (env/config/token file) wins over the local db.
    ///
    /// # Errors
    /// A broker URL without a token.
    pub fn resolve(dir: &AgentDir) -> Result<Self> {
        match resolve_broker_config(dir, &|k| std::env::var(k).ok())? {
            Some(cfg) => Ok(Self::Broker(cfg)),
            None => Ok(Self::Local(dir.0.join("agent.db"))),
        }
    }

    /// The `Credentials saved to …` location line (TS `getAgentDbPath()` / broker url).
    #[must_use]
    pub fn saved_line(&self) -> String {
        match self {
            Self::Local(db) => format!("Credentials saved to {}", db.display()),
            Self::Broker(cfg) => format!("Credentials saved to auth broker {}", cfg.url),
        }
    }

    /// The `Credential removed from …` location (TUI `/logout`).
    #[must_use]
    pub fn removed_line(&self) -> String {
        match self {
            Self::Local(db) => format!("Credential removed from {}", db.display()),
            Self::Broker(cfg) => format!("Credential removed from auth broker {}", cfg.url),
        }
    }

    /// Active rows (optionally one provider's).
    ///
    /// # Errors
    /// Store/broker failure.
    pub async fn list(&self, provider: Option<&str>) -> Result<Vec<StoredRow>> {
        match self {
            Self::Local(db) => {
                let (db, provider) = (db.clone(), provider.map(str::to_string));
                tokio::task::spawn_blocking(move || {
                    if !db.exists() {
                        return Ok(Vec::new());
                    }
                    LocalStore::open(&db)?.list(provider.as_deref())
                })
                .await?
            }
            Self::Broker(cfg) => store::broker_list(cfg, provider).await,
        }
    }

    /// Providers with an active credential, first-seen order.
    ///
    /// # Errors
    /// Store/broker failure.
    pub async fn providers(&self) -> Result<Vec<String>> {
        let mut out: Vec<String> = Vec::new();
        for row in self.list(None).await? {
            if !out.contains(&row.provider) {
                out.push(row.provider);
            }
        }
        Ok(out)
    }

    async fn persist(&self, provider: &str, credential: Credential) -> Result<()> {
        match self {
            Self::Local(db) => {
                let (db, provider) = (db.clone(), provider.to_string());
                tokio::task::spawn_blocking(move || {
                    LocalStore::open(&db)?
                        .upsert(&provider, &credential)
                        .map(|_| ())
                })
                .await?
            }
            Self::Broker(cfg) => store::broker_upload(cfg, provider, &credential).await,
        }
    }

    /// TS `credentials.removeById`: soft-delete one row with cause `deleted by user`;
    /// `false` when it is no longer stored.
    ///
    /// # Errors
    /// Store/broker failure.
    pub async fn remove_by_id(&self, provider: &str, id: i64) -> Result<bool> {
        if self.list(Some(provider)).await?.iter().all(|r| r.id != id) {
            return Ok(false);
        }
        match self {
            Self::Local(db) => {
                let db = db.clone();
                tokio::task::spawn_blocking(move || {
                    LocalStore::open(&db)?.disable(id, "deleted by user")
                })
                .await?
            }
            Self::Broker(cfg) => store::broker_disable(cfg, id, "deleted by user")
                .await
                .map(|()| true),
        }
    }

    /// TS `omp auth-broker logout`: soft-delete every active row of `provider`.
    ///
    /// # Errors
    /// Store/broker failure.
    pub async fn logout_provider(&self, provider: &str, cause: &str) -> Result<()> {
        match self {
            Self::Local(db) => {
                let (db, provider, cause) = (db.clone(), provider.to_string(), cause.to_string());
                tokio::task::spawn_blocking(move || {
                    LocalStore::open(&db)?.disable_provider(&provider, &cause)
                })
                .await?
            }
            Self::Broker(cfg) => {
                for row in store::broker_list(cfg, Some(provider)).await? {
                    store::broker_disable(cfg, row.id, cause).await?;
                }
                Ok(())
            }
        }
    }
}

/// The account a login stored (TS `OAuthLoginIdentity`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Identity {
    ApiKey,
    OAuth {
        email: Option<String>,
        account_id: Option<String>,
        org_id: Option<String>,
        org_name: Option<String>,
    },
}

impl Identity {
    /// TS `formatLoginIdentity`: `email (org)`, `email`, `org`; `None` for api-key logins.
    #[must_use]
    pub fn describe(&self) -> Option<String> {
        let Self::OAuth {
            email,
            account_id,
            org_id,
            org_name,
        } = self
        else {
            return None;
        };
        let base = email.clone().or_else(|| account_id.clone());
        let org = org_name.clone().or_else(|| org_id.clone());
        match (base, org) {
            (Some(base), Some(org)) => Some(format!("{base} ({org})")),
            (Some(base), None) => Some(base),
            (None, org) => org,
        }
    }
}

type EnvFn = Box<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// Everything a flow reaches out to; tests plug in scripted transports and mock hosts.
pub struct Services {
    pub http: Box<dyn ProviderHttp>,
    pub codex_http: Box<dyn CodexHttp>,
    pub hosts: Hosts,
    pub env: EnvFn,
}

impl Services {
    /// Real network, real environment (`KIMI_CODE_OAUTH_HOST` etc.).
    #[must_use]
    pub fn production() -> Self {
        Self {
            http: Box::new(ReqwestProviderHttp::new()),
            codex_http: Box::new(ReqwestCodexHttp::new()),
            hosts: Hosts::from_env(),
            env: Box::new(|k| std::env::var(k).ok()),
        }
    }
}

/// The originator TS puts on the Codex authorize URL (`OPENAI_HEADER_VALUES.ORIGINATOR_CODEX`).
const CODEX_ORIGINATOR: &str = "omp";
/// TS `FAR_FUTURE_MS` of the Copilot hook: 10 years.
const COPILOT_VALIDITY_MS: i64 = 315_576_000_000;

#[allow(clippy::cast_possible_truncation)]
fn now_ms_i64() -> i64 {
    crate::now_ms() as i64
}

/// Whether `id` can be logged into by this build.
#[must_use]
pub fn is_supported(id: &str) -> bool {
    matches!(
        id,
        "anthropic"
            | "openai-codex"
            | "github-copilot"
            | "xai-oauth"
            | "kimi-code"
            | "alibaba-coding-plan"
            | "alibaba-token-plan"
            | "cloudflare-ai-gateway"
            | "cursor"
            | "devin"
            | "factory-droid"
            | "gitlab-duo"
            | "gitlab-duo-agent"
            | "google-antigravity"
            | "google-gemini-cli"
            | "kilo"
            | "muse-code"
            | "openai-codex-device"
            | "perplexity"
            | "stencil"
            | "xiaomi"
    ) || api_key::provider(id).is_some()
}

struct CodexUi(Arc<dyn OAuthLoginUi>);

impl CodexLoginUi for CodexUi {
    fn on_auth(&self, url: &str, instructions: &str) {
        self.0.on_auth(url, Some(instructions));
    }
    fn on_manual_code_input(&self) -> Option<Pin<Box<dyn Future<Output = Option<String>> + Send>>> {
        // Probe synchronously so a surface without a paste keeps Codex's callback-only wait.
        drop(self.0.on_manual_code_input()?);
        let ui = self.0.clone();
        Some(Box::pin(async move { ui.on_manual_code_input()?.await }))
    }
    fn on_prompt(&self, message: &str) -> Pin<Box<dyn Future<Output = Option<String>> + Send>> {
        let ui = self.0.clone();
        let message = message.to_string();
        Box::pin(async move {
            ui.on_prompt(&OAuthPrompt {
                message,
                placeholder: None,
                allow_empty: false,
            })
            .await
        })
    }
    fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

fn put(map: &mut Map<String, Value>, key: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|v| !v.is_empty()) {
        map.insert(key.to_string(), Value::String(value.to_string()));
    }
}

fn base_oauth(access: &str, refresh: &str, expires: i64) -> Map<String, Value> {
    let mut data = Map::new();
    data.insert("access".into(), Value::String(access.to_string()));
    data.insert("refresh".into(), Value::String(refresh.to_string()));
    data.insert("expires".into(), Value::from(expires));
    data
}

fn identity_of(data: &Map<String, Value>) -> Identity {
    let get = |k: &str| data.get(k).and_then(Value::as_str).map(str::to_string);
    Identity::OAuth {
        email: get("email"),
        account_id: get("accountId"),
        org_id: get("orgId"),
        org_name: get("orgName"),
    }
}

/// The Codex identity projection (TS `getTokenProfile`): the ChatGPT
/// workspace id is both `accountId` and `orgId`, the plan type the `orgName`.
fn codex_data(access: &str, refresh: &str, expires: i64, account_id: &str) -> Map<String, Value> {
    let payload = store::jwt_payload(access).unwrap_or_default();
    let email = payload
        .get("https://api.openai.com/profile")
        .and_then(|p| p.get("email"))
        .and_then(Value::as_str)
        .map(|e| e.trim().to_lowercase());
    let plan = payload
        .get("https://api.openai.com/auth")
        .and_then(|a| a.get("chatgpt_plan_type"))
        .and_then(Value::as_str)
        .map(|p| p.trim().to_lowercase());
    let mut data = base_oauth(access, refresh, expires);
    put(&mut data, "accountId", Some(account_id));
    put(&mut data, "email", email.as_deref());
    put(&mut data, "orgId", Some(account_id));
    put(&mut data, "orgName", plan.as_deref());
    data
}

/// xAI `userinfo` enrichment (`email`/`sub`), best effort like TS `applyUserinfo`.
async fn xai_userinfo(http: &dyn ProviderHttp, access: &str) -> (Option<String>, Option<String>) {
    let request = ProviderHttpRequest {
        method: ProviderHttpMethod::Get,
        url: "https://auth.x.ai/oauth2/userinfo".to_string(),
        headers: vec![("Authorization".to_string(), format!("Bearer {access}"))],
        body: None,
        follow_redirects: true,
    };
    let Ok(response) = http.request(request, 30_000).await else {
        return (None, None);
    };
    if !response.ok() {
        return (None, None);
    }
    let Ok(body) = serde_json::from_str::<Value>(&response.body) else {
        return (None, None);
    };
    let scalar = |k: &str| match body.get(k) {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Number(n)) => Some(n.to_string()),
        _ => None,
    };
    (scalar("email"), scalar("sub"))
}

/// Run `provider`'s login and store the result (TS `AuthStorage.oauth.login`).
///
/// `Ok(None)` = the flow stored nothing (empty optional key).
///
/// # Errors
/// The flow failed or was cancelled (`Login cancelled`), the provider is unknown, or storing
/// failed. The message is what TS prints after `Login failed: `.
pub async fn login(
    target: &Target,
    provider: &str,
    ui: Arc<dyn OAuthLoginUi>,
    services: &Services,
) -> Result<Option<Identity>, String> {
    let store_provider = providers::store_as(provider).to_string();
    let credential = match provider {
        "anthropic" => {
            let c = pa_ai::oauth::login_anthropic(services.http.as_ref(), ui.as_ref()).await?;
            let mut data = base_oauth(&c.access, &c.refresh, c.expires);
            put(&mut data, "email", c.email.as_deref());
            put(&mut data, "accountId", c.account_id.as_deref());
            put(&mut data, "orgId", c.org_id.as_deref());
            put(&mut data, "orgName", c.org_name.as_deref());
            data
        }
        "openai-codex" => {
            let c = pa_ai::oauth::login_openai_codex(
                services.codex_http.as_ref(),
                &CodexUi(ui.clone()),
                CODEX_ORIGINATOR,
            )
            .await?;
            codex_data(&c.access, &c.refresh, c.expires, &c.account_id)
        }
        "openai-codex-device" => {
            let c =
                codex_device::login(ui.as_ref(), services.http.as_ref(), now_ms_i64(), |token| {
                    token.get("expires_in").and_then(Value::as_i64)
                })
                .await?;
            codex_data(&c.access, &c.refresh, c.expires, &c.account_id)
        }
        "cursor" => {
            let c = cursor::login(ui.as_ref(), services.http.as_ref(), now_ms_i64()).await?;
            base_oauth(&c.access, &c.refresh, c.expires)
        }
        "muse-code" => {
            let (access, account_id, email) =
                muse_code::login(ui.as_ref(), services.http.as_ref()).await?;
            let mut data = base_oauth(&access, "", muse_code::NEVER_EXPIRES);
            put(&mut data, "accountId", account_id.as_deref());
            put(&mut data, "email", email.as_deref());
            data
        }
        "perplexity" => {
            let c = perplexity::login(ui.as_ref(), services.http.as_ref()).await?;
            let mut data = base_oauth(&c.access, &c.refresh, c.expires);
            put(&mut data, "email", Some(&c.email));
            data
        }
        "google-gemini-cli" => {
            let outcome = oauth_code::login_google_gemini_cli(
                services.http.as_ref(),
                ui.as_ref(),
                services.env.as_ref(),
            )
            .await?;
            let mut data = base_oauth(&outcome.access, &outcome.refresh, outcome.expires);
            put(&mut data, "email", outcome.email.as_deref());
            put(&mut data, "projectId", outcome.project.as_deref());
            data
        }
        "google-antigravity" => {
            let outcome =
                oauth_code::login_google_antigravity(services.http.as_ref(), ui.as_ref()).await?;
            let mut data = base_oauth(&outcome.access, &outcome.refresh, outcome.expires);
            put(&mut data, "email", outcome.email.as_deref());
            put(&mut data, "projectId", outcome.project.as_deref());
            data
        }
        "devin" => {
            let outcome = oauth_code::login_devin(services.http.as_ref(), ui.as_ref()).await?;
            let expires = oauth_code::jwt_exp_ms(&outcome.access, 0)
                .unwrap_or_else(|| now_ms_i64() + 31_536_000_000);
            let mut data = base_oauth(&outcome.access, &outcome.refresh, expires);
            put(&mut data, "apiEndpoint", Some("https://api.devin.ai"));
            put(&mut data, "enterpriseUrl", Some("https://app.devin.ai"));
            data
        }
        "stencil" => {
            let outcome = oauth_code::login_stencil(
                services.http.as_ref(),
                ui.as_ref(),
                services.env.as_ref(),
            )
            .await?;
            let mut data = base_oauth(&outcome.access, &outcome.refresh, outcome.expires);
            put(&mut data, "email", outcome.email.as_deref());
            put(&mut data, "accountId", outcome.account_id.as_deref());
            data
        }
        "kimi-code" => {
            let base = services
                .hosts
                .kimi
                .clone()
                .unwrap_or_else(|| pa_ai::oauth::KIMI_DEFAULT_HOST.to_string());
            let headers = pa_ai::kimi_common_headers();
            let c = pa_ai::oauth::login_kimi(services.http.as_ref(), ui.as_ref(), &base, &headers)
                .await?;
            let mut data = base_oauth(&c.access, &c.refresh, c.expires);
            put(&mut data, "accountId", c.account_id.as_deref());
            data
        }
        "xai-oauth" => {
            let c = pa_ai::oauth::login_xai(services.http.as_ref(), ui.as_ref()).await?;
            let mut data = base_oauth(&c.access, &c.refresh, c.expires);
            let claim = store::jwt_payload(&c.access)
                .and_then(|p| p.get("sub").and_then(Value::as_str).map(str::to_string));
            put(&mut data, "accountId", claim.as_deref());
            let (email, account) = xai_userinfo(services.http.as_ref(), &c.access).await;
            put(&mut data, "email", email.as_deref());
            put(&mut data, "accountId", account.as_deref());
            data
        }
        "github-copilot" => {
            let c = pa_ai::oauth::login_github_copilot(services.http.as_ref(), ui.as_ref()).await?;
            let mut data = Map::new();
            data.insert("refresh".into(), Value::String(c.refresh.clone()));
            data.insert("access".into(), Value::String(c.refresh));
            data.insert(
                "expires".into(),
                Value::from(now_ms_i64().saturating_add(COPILOT_VALIDITY_MS)),
            );
            put(&mut data, "enterpriseUrl", c.enterprise_url.as_deref());
            data
        }
        "gitlab-duo" => {
            let c = gitlab::gitlab_duo(ui.as_ref(), services.http.as_ref(), services.env.as_ref())
                .await?;
            base_oauth(&c.access, &c.refresh, c.expires)
        }
        "gitlab-duo-agent" => {
            let c = gitlab::gitlab_duo_agent(ui.as_ref(), services.http.as_ref()).await?;
            base_oauth(&c.access, &c.refresh, c.expires)
        }
        "kilo" => {
            let c = kilo::login_kilo(services.http.as_ref(), ui.as_ref()).await?;
            base_oauth(&c.access, "", c.expires)
        }
        "factory-droid" => {
            let c = factory::login_factory(services.http.as_ref(), ui.as_ref()).await?;
            let mut data = base_oauth(&c.access, &c.refresh, c.expires);
            put(&mut data, "email", c.email.as_deref());
            put(&mut data, "accountId", c.account_id.as_deref());
            put(&mut data, "orgId", c.org_id.as_deref());
            put(&mut data, "region", c.region.as_deref());
            put(&mut data, "inferenceRegion", c.inference_region.as_deref());
            data
        }
        "cloudflare-ai-gateway" => {
            let key = cloudflare::login_cloudflare_ai_gateway(ui.as_ref()).await?;
            target
                .persist(&store_provider, Credential::ApiKey(key))
                .await
                .map_err(|e| format!("{e:#}"))?;
            return Ok(Some(Identity::ApiKey));
        }
        "xiaomi" => {
            let key = xiaomi::login_xiaomi(ui.as_ref(), services.http.as_ref()).await?;
            target
                .persist(&store_provider, Credential::ApiKey(key))
                .await
                .map_err(|e| format!("{e:#}"))?;
            return Ok(Some(Identity::ApiKey));
        }
        "alibaba-coding-plan" => {
            let key = custom::alibaba_coding_plan(
                ui.as_ref(),
                services.http.as_ref(),
                services.env.as_ref(),
            )
            .await?;
            if key.is_empty() {
                return Ok(None);
            }
            target
                .persist(&store_provider, Credential::ApiKey(key))
                .await
                .map_err(|e| format!("{e:#}"))?;
            return Ok(Some(Identity::ApiKey));
        }
        "alibaba-token-plan" => {
            let key = custom::alibaba_token_plan(
                ui.as_ref(),
                services.http.as_ref(),
                services.env.as_ref(),
            )
            .await?;
            if key.is_empty() {
                return Ok(None);
            }
            target
                .persist(&store_provider, Credential::ApiKey(key))
                .await
                .map_err(|e| format!("{e:#}"))?;
            return Ok(Some(Identity::ApiKey));
        }
        other => {
            let Some(rule) = api_key::provider(other) else {
                return Err(format!("Unknown OAuth provider: {other}"));
            };
            let key = api_key::login(
                rule,
                ui.as_ref(),
                services.http.as_ref(),
                services.env.as_ref(),
            )
            .await?;
            if key.is_empty() {
                return Ok(None);
            }
            target
                .persist(&store_provider, Credential::ApiKey(key))
                .await
                .map_err(|e| format!("{e:#}"))?;
            return Ok(Some(Identity::ApiKey));
        }
    };
    let mut data = credential;
    data.insert("authorizedAt".into(), Value::from(now_ms_i64()));
    let identity = identity_of(&data);
    target
        .persist(&store_provider, Credential::OAuth(data))
        .await
        .map_err(|e| format!("{e:#}"))?;
    Ok(Some(identity))
}

#[cfg(test)]
mod tests;
