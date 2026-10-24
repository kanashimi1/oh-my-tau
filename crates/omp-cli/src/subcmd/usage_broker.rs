//! The dashboard's auth-broker usage source (TS `fetchUsageData` broker mode):
//! answers `omp-stats`'s [`UsageBroker`] with the broker's `/v1/usage/history`
//! and `/v1/usage/clients` reads. Registered with `omp-stats`, which holds no
//! HTTP client itself.

use std::future::Future;
use std::sync::{mpsc, Arc, LazyLock};

use anyhow::{anyhow, Context, Result};
use omp_providers::auth::{resolve_broker_config, BrokerConfig};
use omp_providers::login::{broker_client_usage_summary, broker_usage_history};
use omp_providers::AgentDir;
use omp_stats::usage_windows::UsageSnapshotRow;
use omp_stats::usage_windows::{
    set_usage_broker, ClientProviderUsage, ClientUsageClientSummary, UsageBroker,
};

/// Install the broker usage source. Idempotent; every dashboard entry point
/// (`omp-rs stats`, `/stats`, `/trace`) calls it before starting the server.
pub fn register() {
    set_usage_broker(Some(Arc::new(BrokerSource) as Arc<dyn UsageBroker>));
}

/// Stateless: like TS `fetchUsageData`, the broker config is resolved on every
/// fetch, so a broker configured after the dashboard started is picked up. No
/// broker or a resolution error is an `Err`, which `omp-stats` turns into the
/// local agent.db fallback (TS catch path).
struct BrokerSource;

/// TS `getAgentDir()` honours `PI_CONFIG_DIR`, the same dir the local fallback
/// reads `agent.db` from; `OMP_AGENT_DIR` stays the omp-rs override.
fn broker_config() -> Result<BrokerConfig> {
    let dir = std::env::var_os("OMP_AGENT_DIR")
        .map_or_else(omp_stats::paths::agent_dir, std::path::PathBuf::from);
    let env = |key: &str| std::env::var(key).ok();
    resolve_broker_config(&AgentDir(dir), &env)?.ok_or_else(|| anyhow!("no auth broker configured"))
}

/// Private runtime for the broker HTTP calls.
static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("omp-stats-broker")
        .enable_all()
        .build()
        .expect("usage broker runtime")
});

/// Run `fut` on the private runtime and wait for it on this thread. Spawn +
/// channel instead of `block_on`, so it is safe from any context (a
/// `spawn_blocking` thread, a plain thread, or inside another runtime);
/// `broker_call`'s request timeout bounds the wait.
fn run<T: Send + 'static>(fut: impl Future<Output = Result<T>> + Send + 'static) -> Result<T> {
    let (tx, rx) = mpsc::sync_channel(1);
    RUNTIME.spawn(async move {
        let _ = tx.send(fut.await);
    });
    rx.recv().context("usage broker task dropped")?
}

impl UsageBroker for BrokerSource {
    fn usage_history(&self, since_ms: i64) -> Result<Vec<UsageSnapshotRow>> {
        let cfg = broker_config()?;
        let entries = run(async move { broker_usage_history(&cfg, since_ms).await })?;
        Ok(entries
            .into_iter()
            .map(|entry| UsageSnapshotRow {
                recorded_at: entry.recorded_at,
                provider: entry.provider,
                account_key: entry.account_key,
                email: entry.email,
                account_id: entry.account_id,
                limit_id: entry.limit_id,
                label: entry.label,
                window_label: entry.window_label,
                used_fraction: entry.used_fraction,
                status: entry.status,
            })
            .collect())
    }

    fn client_usage_summary(&self, since_ms: i64) -> Result<Vec<ClientUsageClientSummary>> {
        let cfg = broker_config()?;
        let clients = run(async move { broker_client_usage_summary(&cfg, since_ms).await })?;
        Ok(clients
            .into_iter()
            .map(|client| ClientUsageClientSummary {
                install_id: client.install_id,
                providers: client
                    .providers
                    .into_iter()
                    .map(|usage| ClientProviderUsage {
                        provider: usage.provider,
                        input_tokens: usage.input_tokens,
                        output_tokens: usage.output_tokens,
                        cache_read_tokens: usage.cache_read_tokens,
                        cache_write_tokens: usage.cache_write_tokens,
                    })
                    .collect(),
            })
            .collect())
    }
}
