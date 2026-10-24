//! Rust port of `usage-windows.ts`: provider subscription-window analytics.
//!
//! The auth layer appends one row to `usage_history` in agent.db every time a
//! provider usage report is fetched. This module reads those snapshots
//! read-only and derives utilization series per account and limit window, plus
//! per-window insights (window-equivalents consumed, estimated tokens per
//! window, peak concurrent utilization and the account count it implies).
//!
//! Where the data comes from (TS `fetchUsageData`): when the host installs a
//! [`broker::UsageBroker`] — a configured auth broker performs every upstream
//! usage fetch, so the local `usage_history` stays frozen — rows and the
//! fleet-wide token totals come from the broker; broker errors degrade to the
//! local agent.db read with no fleet tokens, like TS. `omp-stats` registers no
//! broker itself (it holds no HTTP client); omp-cli does.
//!
//! `localeCompare` ordering (used by the TS sorts) is approximated by
//! [`locale_compare`]; see its docs for the exact scope.

mod broker;
mod cache;
mod collate;

pub use broker::{
    set_usage_broker, sum_fleet_tokens, ClientProviderUsage, ClientUsageClientSummary, UsageBroker,
};
pub use cache::cached_usage_data;
#[cfg(test)]
mod tests;

pub use collate::locale_compare;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags};

use crate::paths::agent_db_path;
use crate::types::{ProviderWindowInsight, UsageWindowPoint, UsageWindowSeries};

/// A used-fraction drop smaller than this is jitter, not a window reset.
const RESET_DROP_THRESHOLD: f64 = 0.05;
/// Minimum window-equivalents consumed before extrapolating tokens/window.
const MIN_EXTRAPOLATION_FRACTION: f64 = 0.1;
/// Fleet-capacity headroom target: peak demand should stay under 90%.
const TARGET_PEAK_UTILIZATION: f64 = 0.9;
/// Used fraction at or above this counts as exhausted even without a status.
const EXHAUSTED_FRACTION: f64 = 0.999;
/// Utilization series are downsampled (peak per bucket) to at most this many points.
const MAX_SERIES_POINTS: usize = 400;
pub(crate) const HOUR_MS: i64 = 3_600_000;

/// Subset of a `usage_history` row consumed by the window analytics.
#[derive(Debug, Clone, PartialEq)]
pub struct UsageSnapshotRow {
    /// Epoch ms the report was fetched.
    pub recorded_at: i64,
    pub provider: String,
    /// Stable credential identity key.
    pub account_key: String,
    pub email: Option<String>,
    pub account_id: Option<String>,
    pub limit_id: String,
    pub label: String,
    pub window_label: Option<String>,
    /// Used fraction (0..1, >1 = overage) when the provider reported one.
    pub used_fraction: Option<f64>,
    pub status: Option<String>,
}

/// TS `UsageWindowStats`.
#[derive(Debug, Clone, Default)]
pub struct UsageWindowStats {
    pub usage_series: Vec<UsageWindowSeries>,
    pub window_insights: Vec<ProviderWindowInsight>,
}

/// TS `UsageDataSnapshot`. `fleet_tokens_by_provider` is `Some` only in broker
/// mode (an installed [`UsageBroker`] whose summary fetch succeeded); callers
/// then use local message-stat tokens instead.
#[derive(Debug, Clone, Default)]
pub struct UsageDataSnapshot {
    pub rows: Vec<UsageSnapshotRow>,
    pub fleet_tokens_by_provider: Option<HashMap<String, f64>>,
}

/// Read usage-limit snapshots recorded at or after `since_ms`, oldest first.
/// Opens the DB read-only; returns no rows when the DB or table is absent or
/// unreadable (TS logs at debug and returns `[]`).
#[must_use]
pub fn read_usage_snapshots(since_ms: i64, db_path: &Path) -> Vec<UsageSnapshotRow> {
    match query_usage_snapshots(since_ms, db_path) {
        Ok(rows) => rows,
        Err(err) => {
            tracing::debug!(db = %db_path.display(), error = %err, "usage_history unavailable for provider stats");
            Vec::new()
        }
    }
}

fn query_usage_snapshots(since_ms: i64, db_path: &Path) -> rusqlite::Result<Vec<UsageSnapshotRow>> {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    conn.busy_timeout(Duration::from_millis(5000))?;
    let mut stmt = conn.prepare(
        "SELECT recorded_at, provider, account_key, email, account_id, limit_id, label, window_label, used_fraction, status
         FROM usage_history
         WHERE recorded_at >= ?1
         ORDER BY recorded_at ASC",
    )?;
    let rows = stmt.query_map([since_ms], |row| {
        Ok(UsageSnapshotRow {
            recorded_at: row.get(0)?,
            provider: row.get(1)?,
            account_key: row.get(2)?,
            email: row.get(3)?,
            account_id: row.get(4)?,
            limit_id: row.get(5)?,
            label: row.get(6)?,
            window_label: row.get(7)?,
            used_fraction: row.get(8)?,
            status: row.get(9)?,
        })
    })?;
    rows.collect()
}

/// TS `fetchUsageData`: snapshots and, in broker mode, fleet-wide token burn
/// per provider from the installed [`UsageBroker`]; a broker fetch failure
/// (or no broker) falls back to the local agent.db read with no fleet tokens.
#[must_use]
pub fn fetch_usage_data(since_ms: i64) -> UsageDataSnapshot {
    if let Some(broker) = broker::installed_broker() {
        match broker.usage_history(since_ms) {
            Ok(rows) => {
                let fleet_tokens_by_provider = broker::fetch_fleet_tokens(&broker, since_ms);
                return UsageDataSnapshot {
                    rows,
                    fleet_tokens_by_provider,
                };
            }
            Err(err) => {
                tracing::debug!(error = %err, "broker usage history unavailable, falling back to local");
            }
        }
    }
    UsageDataSnapshot {
        rows: read_usage_snapshots(since_ms, &agent_db_path()),
        fleet_tokens_by_provider: None,
    }
}

/// True when a snapshot reports an exhausted window, by status or by fraction.
fn is_exhausted(fraction: Option<f64>, status: Option<&str>) -> bool {
    status == Some("exhausted") || fraction.is_some_and(|f| f >= EXHAUSTED_FRACTION)
}

/// Reduce a point list to at most [`MAX_SERIES_POINTS`] by keeping the
/// peak-fraction point per time bucket, so utilization peaks survive downsampling.
fn downsample_points(points: Vec<UsageWindowPoint>) -> Vec<UsageWindowPoint> {
    if points.len() <= MAX_SERIES_POINTS {
        return points;
    }
    let first = points[0].timestamp;
    let span = points[points.len() - 1].timestamp - first;
    let bucket_ms = ((span as f64 / MAX_SERIES_POINTS as f64).ceil() as i64).max(1);
    let mut out: Vec<UsageWindowPoint> = Vec::new();
    let mut bucket = -1_i64;
    for point in points {
        let b = ((point.timestamp - first) as f64 / bucket_ms as f64).floor() as i64;
        if b != bucket {
            out.push(point);
            bucket = b;
            continue;
        }
        // `bucket == -1` only before the first push, so `out` is non-empty here.
        if let Some(last) = out.last_mut() {
            if point.used_fraction.unwrap_or(-1.0) >= last.used_fraction.unwrap_or(-1.0) {
                *last = point;
            }
        }
    }
    out
}

/// One account's snapshots within a window group.
struct AccountSeries<'a> {
    account_key: &'a str,
    account_label: String,
    rows: Vec<&'a UsageSnapshotRow>,
}

/// One `(provider, limitId)` window across accounts, in first-seen order.
struct WindowGroup<'a> {
    provider: &'a str,
    window_key: &'a str,
    window_label: String,
    accounts: Vec<AccountSeries<'a>>,
}

/// Display label for one limit window: the limit label, plus the window label
/// when it adds information ("Usage (Google) · Daily").
fn window_display_label(row: &UsageSnapshotRow) -> String {
    match row.window_label.as_deref() {
        None | Some("") => row.label.clone(),
        Some(window_label) => {
            if row
                .label
                .to_lowercase()
                .contains(&window_label.to_lowercase())
            {
                row.label.clone()
            } else {
                format!("{} · {}", row.label, window_label)
            }
        }
    }
}

/// JS truthiness of an optional string (`null`, `undefined` and `""` are falsy).
fn truthy(value: Option<&str>) -> bool {
    value.is_some_and(|s| !s.is_empty())
}

/// Derive utilization series and per-window insights from raw snapshots.
///
/// Windows are grouped by `(provider, limitId)`, never by display label:
/// distinct limits can share a duration label, and merging them inflates
/// consumption. `tokens_by_provider` gives each provider's token burn over the
/// same range and converts consumed window fraction into an estimated token
/// capacity per window.
#[must_use]
pub fn compute_usage_window_stats(
    rows: &[UsageSnapshotRow],
    tokens_by_provider: &HashMap<String, f64>,
) -> UsageWindowStats {
    let mut groups: Vec<WindowGroup<'_>> = Vec::new();
    let mut group_index: HashMap<(&str, &str), usize> = HashMap::new();
    for row in rows {
        let gi = *group_index
            .entry((row.provider.as_str(), row.limit_id.as_str()))
            .or_insert_with(|| {
                groups.push(WindowGroup {
                    provider: &row.provider,
                    window_key: &row.limit_id,
                    window_label: String::new(),
                    accounts: Vec::new(),
                });
                groups.len() - 1
            });
        let group = &mut groups[gi];
        // Labels can change across snapshots (provider renames); latest wins.
        group.window_label = window_display_label(row);
        let ai = match group
            .accounts
            .iter()
            .position(|a| a.account_key == row.account_key)
        {
            Some(ai) => ai,
            None => {
                group.accounts.push(AccountSeries {
                    account_key: &row.account_key,
                    account_label: row
                        .email
                        .clone()
                        .or_else(|| row.account_id.clone())
                        .unwrap_or_else(|| row.account_key.clone()),
                    rows: Vec::new(),
                });
                group.accounts.len() - 1
            }
        };
        let account = &mut group.accounts[ai];
        if truthy(row.email.as_deref()) || truthy(row.account_id.as_deref()) {
            account.account_label = row
                .email
                .clone()
                .or_else(|| row.account_id.clone())
                .unwrap_or_else(|| row.account_key.clone());
        }
        account.rows.push(row);
    }

    let mut usage_series = Vec::new();
    let mut window_insights = Vec::new();

    for group in &groups {
        let mut fraction_consumed = 0.0_f64;
        let mut cycles = 0_i64;
        let mut exhausted_events = 0_i64;

        for account in &group.accounts {
            let points: Vec<UsageWindowPoint> = account
                .rows
                .iter()
                .map(|row| UsageWindowPoint {
                    timestamp: row.recorded_at,
                    used_fraction: row.used_fraction,
                    exhausted: is_exhausted(row.used_fraction, row.status.as_deref()),
                })
                .collect();
            usage_series.push(UsageWindowSeries {
                provider: group.provider.to_owned(),
                account_key: account.account_key.to_owned(),
                account_label: account.account_label.clone(),
                window_key: group.window_key.to_owned(),
                window_label: group.window_label.clone(),
                points: downsample_points(points),
            });

            let mut prev_fraction: Option<f64> = None;
            let mut prev_exhausted = false;
            for row in &account.rows {
                let exhausted = is_exhausted(row.used_fraction, row.status.as_deref());
                if exhausted && !prev_exhausted {
                    exhausted_events += 1;
                }
                prev_exhausted = exhausted;
                let Some(fraction) = row.used_fraction else {
                    continue;
                };
                if let Some(prev) = prev_fraction {
                    let delta = fraction - prev;
                    if delta > 0.0 {
                        fraction_consumed += delta;
                    } else if delta < -RESET_DROP_THRESHOLD {
                        cycles += 1;
                    }
                }
                prev_fraction = Some(fraction);
            }
        }

        let provider_tokens = tokens_by_provider
            .get(group.provider)
            .copied()
            .unwrap_or(0.0);
        let peak = peak_concurrent_fraction(group);
        window_insights.push(ProviderWindowInsight {
            provider: group.provider.to_owned(),
            window_key: group.window_key.to_owned(),
            window_label: group.window_label.clone(),
            accounts: group.accounts.len() as i64,
            cycles,
            fraction_consumed,
            // Positive operands only, so `f64::round` is JS `Math.round` here.
            est_tokens_per_window: (provider_tokens > 0.0
                && fraction_consumed >= MIN_EXTRAPOLATION_FRACTION)
                .then(|| (provider_tokens / fraction_consumed).round()),
            peak_concurrent_fraction: peak,
            ideal_accounts: (peak / TARGET_PEAK_UTILIZATION).ceil().max(1.0) as i64,
            exhausted_events,
        });
    }

    usage_series.sort_by(|a, b| {
        locale_compare(&a.provider, &b.provider)
            .then_with(|| locale_compare(&a.window_key, &b.window_key))
            .then_with(|| locale_compare(&a.account_label, &b.account_label))
    });
    window_insights.sort_by(|a, b| {
        locale_compare(&a.provider, &b.provider).then_with(|| {
            b.fraction_consumed
                .partial_cmp(&a.fraction_consumed)
                .unwrap_or(Ordering::Equal)
        })
    });
    UsageWindowStats {
        usage_series,
        window_insights,
    }
}

/// Peak of sum-across-accounts used fraction at any sampled instant: sweep all
/// snapshot times, forward-filling each account's last known fraction.
fn peak_concurrent_fraction(group: &WindowGroup<'_>) -> f64 {
    let mut events: Vec<(i64, &str, f64)> = Vec::new();
    for account in &group.accounts {
        for row in &account.rows {
            if let Some(fraction) = row.used_fraction {
                events.push((row.recorded_at, account.account_key, fraction));
            }
        }
    }
    events.sort_by_key(|event| event.0);

    let mut current: HashMap<&str, f64> = HashMap::new();
    let mut sum = 0.0_f64;
    let mut peak = 0.0_f64;
    for (_, account, fraction) in events {
        sum += fraction - current.get(account).copied().unwrap_or(0.0);
        current.insert(account, fraction);
        if sum > peak {
            peak = sum;
        }
    }
    peak
}
