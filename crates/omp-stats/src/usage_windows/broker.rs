//! Pluggable auth-broker usage source (TS `usage-windows.ts` broker mode:
//! `fetchUsageData`, `fetchFleetTokens`, `sumFleetTokens`).
//!
//! `omp-stats` holds no HTTP client (and must not depend on `omp-providers`,
//! which depends on it). The host (omp-cli) registers a [`UsageBroker`]
//! resolved from the real agent dir; [`super::fetch_usage_data`] asks it for
//! the broker's durable `usage_history` and per-client token burn, falling
//! back to the local agent.db read when no broker is installed or a fetch
//! fails (TS degrades the same way).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use anyhow::Result;

use super::UsageSnapshotRow;

/// Per-provider aggregate of one client's recorded usage (TS
/// `ClientProviderUsage`); only the token fields `sumFleetTokens` folds.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClientProviderUsage {
    pub provider: String,
    pub input_tokens: f64,
    pub output_tokens: f64,
    pub cache_read_tokens: f64,
    pub cache_write_tokens: f64,
}

/// One known client with its usage aggregates over the queried window (TS
/// `ClientUsageClientSummary`); only the fields the fleet-token fold uses.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClientUsageClientSummary {
    pub install_id: String,
    pub providers: Vec<ClientProviderUsage>,
}

/// Host-supplied auth-broker usage source (the broker half of TS
/// `AuthBrokerClient.fetchUsageHistory` / `fetchClientUsageSummary`).
///
/// Implementations run their own HTTP on their own runtime; `omp-stats` calls
/// these from synchronous handlers.
pub trait UsageBroker: Send + Sync {
    /// Recorded usage-limit snapshots from the broker host, oldest first.
    ///
    /// # Errors
    /// Transport, HTTP or schema failure.
    fn usage_history(&self, since_ms: i64) -> Result<Vec<UsageSnapshotRow>>;
    /// Per-client token burn aggregates over the window.
    ///
    /// # Errors
    /// Transport, HTTP or schema failure.
    fn client_usage_summary(&self, since_ms: i64) -> Result<Vec<ClientUsageClientSummary>>;
}

static BROKER: Mutex<Option<Arc<dyn UsageBroker>>> = Mutex::new(None);

/// Register (or replace) the host usage broker; `None` removes it (the TS
/// "no broker configured" path: local agent.db, no fleet tokens).
pub fn set_usage_broker(broker: Option<Arc<dyn UsageBroker>>) {
    *BROKER.lock().unwrap_or_else(PoisonError::into_inner) = broker;
}

pub(crate) fn installed_broker() -> Option<Arc<dyn UsageBroker>> {
    BROKER
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// TS `fetchFleetTokens`: sum broker-recorded client token burn per provider;
/// a summary failure or an empty report is `None`, not an error, so callers
/// fall back to local message stats instead of zeroing estimates.
pub(crate) fn fetch_fleet_tokens(
    broker: &Arc<dyn UsageBroker>,
    since_ms: i64,
) -> Option<HashMap<String, f64>> {
    match broker.client_usage_summary(since_ms) {
        Ok(clients) => sum_fleet_tokens(&clients),
        Err(err) => {
            tracing::debug!(error = %err, "broker client usage summary unavailable");
            None
        }
    }
}

/// TS `sumFleetTokens`: fold per-client provider aggregates into total tokens
/// per provider (input + output + cache read/write, matching message-stat
/// `totalTokens`). Empty input is `None`: "no data", not "zero burn".
#[must_use]
pub fn sum_fleet_tokens(clients: &[ClientUsageClientSummary]) -> Option<HashMap<String, f64>> {
    let mut tokens: HashMap<String, f64> = HashMap::new();
    for client in clients {
        for usage in &client.providers {
            let total = usage.input_tokens
                + usage.output_tokens
                + usage.cache_read_tokens
                + usage.cache_write_tokens;
            *tokens.entry(usage.provider.clone()).or_default() += total;
        }
    }
    if tokens.is_empty() {
        None
    } else {
        Some(tokens)
    }
}
