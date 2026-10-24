//! TS `usage-windows.ts` memoization (`cachedUsageData`, 60s TTL).

use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::{fetch_usage_data, UsageDataSnapshot, HOUR_MS};
use crate::paths::agent_db_path;

/// TS `USAGE_CACHE_TTL_MS`.
const USAGE_CACHE_TTL: Duration = Duration::from_secs(60);

struct CachedUsage {
    key: String,
    at: Instant,
    data: UsageDataSnapshot,
}

static USAGE_CACHE: Mutex<Option<CachedUsage>> = Mutex::new(None);

/// TS `cachedUsageData`: [`fetch_usage_data`] memoized for a minute per
/// hour-aligned cutoff and agent DB path.
#[must_use]
pub fn cached_usage_data(since_ms: i64) -> UsageDataSnapshot {
    let since_hour = since_ms.div_euclid(HOUR_MS) * HOUR_MS;
    let key = format!("{}:{since_hour}", agent_db_path().display());
    let mut cache = USAGE_CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(cached) = cache.as_ref() {
        if cached.key == key && cached.at.elapsed() < USAGE_CACHE_TTL {
            return cached.data.clone();
        }
    }
    let data = fetch_usage_data(since_hour);
    *cache = Some(CachedUsage {
        key,
        at: Instant::now(),
        data: data.clone(),
    });
    data
}
