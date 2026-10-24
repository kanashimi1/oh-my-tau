//! Backfill bodies + db.ts pricing helpers (part of `schema`; see schema.rs).
use anyhow::Result;
use rusqlite::Connection;

use super::{
    meta_value, reset_offsets, set_meta, should_reset_backfill, AGENT_TYPE_BACKFILL_KEY,
    BACKFILL_COMPLETE, BACKFILL_PENDING, CATALOG_BACKFILL_VERSION, CATALOG_COST_BACKFILL_KEY,
    COST_REINGEST_BACKFILL_KEY, COST_UNPRICED_BACKFILL_KEY, FORK_DEDUPE_KEY,
    NO_CACHE_INPUT_BACKFILL_KEY, PRIORITY_PREMIUM_REQUESTS_BACKFILL_KEY, TOOL_CALLS_BACKFILL_KEY,
    USER_MESSAGES_BACKFILL_KEY, USER_MESSAGE_LINKS_REPAIR_KEY,
};
use crate::catalog::{
    calculate_uncached_input_cost, calculate_usage_cost, get_catalog_cost, Usage,
};
use crate::types::AgentType;
#[cfg(feature = "db_ingest")]
use crate::types::{MessageStatsInput, UsageInputCost};

pub(crate) fn backfill_user_messages(conn: &Connection) -> Result<()> {
    if !should_reset_backfill(meta_value(conn, USER_MESSAGES_BACKFILL_KEY)?.as_deref()) {
        return Ok(());
    }
    conn.execute("DELETE FROM user_messages", [])?;
    conn.execute("DELETE FROM file_offsets", [])?;
    set_meta(conn, USER_MESSAGES_BACKFILL_KEY, BACKFILL_PENDING)?;
    Ok(())
}

pub(crate) fn backfill_tool_calls(conn: &Connection) -> Result<()> {
    if !should_reset_backfill(meta_value(conn, TOOL_CALLS_BACKFILL_KEY)?.as_deref()) {
        return Ok(());
    }
    conn.execute("DELETE FROM tool_calls", [])?;
    conn.execute("DELETE FROM file_offsets", [])?;
    set_meta(conn, TOOL_CALLS_BACKFILL_KEY, BACKFILL_PENDING)?;
    Ok(())
}

pub(crate) fn backfill_reingest_costs(conn: &Connection) -> Result<()> {
    reset_offsets(conn, COST_REINGEST_BACKFILL_KEY)
}

pub(crate) fn backfill_unpriced_costs(conn: &Connection) -> Result<()> {
    reset_offsets(conn, COST_UNPRICED_BACKFILL_KEY)
}

pub(crate) fn repair_user_message_links(conn: &Connection) -> Result<()> {
    reset_offsets(conn, USER_MESSAGE_LINKS_REPAIR_KEY)
}

pub(crate) fn backfill_priority_premium_requests(conn: &Connection) -> Result<()> {
    reset_offsets(conn, PRIORITY_PREMIUM_REQUESTS_BACKFILL_KEY)
}

/// Path-based main / subagent / advisor classification (TS `classifyAgentType`
/// in parser.ts). Single implementation lives in `crate::paths` (both
/// feature trees can see it); this typed adapter maps the wire string.
fn classify_agent_type(session_path: &str) -> AgentType {
    match crate::paths::classify_agent_type_str(session_path) {
        "advisor" => AgentType::Advisor,
        "subagent" => AgentType::Subagent,
        _ => AgentType::Main,
    }
}

pub(crate) fn backfill_agent_type(conn: &Connection) -> Result<()> {
    if meta_value(conn, AGENT_TYPE_BACKFILL_KEY)?.as_deref() != Some(BACKFILL_PENDING) {
        return Ok(());
    }
    let mut stmt = conn.prepare("SELECT DISTINCT session_file FROM messages")?;
    let files: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    drop(stmt);
    let tx = conn.unchecked_transaction()?;
    {
        let mut update =
            tx.prepare("UPDATE messages SET agent_type = ?1 WHERE session_file = ?2")?;
        for file in &files {
            let agent_type = classify_agent_type(file);
            // Rows already default to 'main'; only the nested transcripts move.
            if agent_type != AgentType::Main {
                update.execute(rusqlite::params![agent_type_str(agent_type), file])?;
            }
        }
    }
    tx.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params![AGENT_TYPE_BACKFILL_KEY, BACKFILL_COMPLETE],
    )?;
    tx.commit()?;
    Ok(())
}

pub(crate) fn agent_type_str(t: AgentType) -> &'static str {
    match t {
        AgentType::Main => "main",
        AgentType::Subagent => "subagent",
        AgentType::Advisor => "advisor",
    }
}

/// Zero-cost rows are repriced from the bundled catalog once per release (the
/// stored sentinel value is the TS VERSION that ran it), since only a catalog
/// update can price a model it could not price before.
pub(crate) fn backfill_missing_catalog_costs(conn: &Connection) -> Result<()> {
    if meta_value(conn, CATALOG_COST_BACKFILL_KEY)?.as_deref() == Some(CATALOG_BACKFILL_VERSION) {
        return Ok(());
    }
    let mut stmt = conn.prepare(
        "SELECT id, provider, model, timestamp, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
         FROM messages WHERE cost_total = 0 AND total_tokens > 0",
    )?;
    let rows = {
        stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, i64>(7)?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?
    };
    let tx = conn.unchecked_transaction()?;
    {
        let mut update = tx.prepare(
            "UPDATE messages SET cost_input = ?1, cost_output = ?2, cost_cache_read = ?3,
             cost_cache_write = ?4, cost_total = ?5 WHERE id = ?6",
        )?;
        for (id, provider, model, timestamp, input, output, cache_read, cache_write) in rows {
            // Never reprice recorded scheduled usage: a zero ingested for a
            // scheduled card is permanent. Re-parsing the session file is the
            // only repair.
            if get_catalog_cost(&provider, &model)
                .and_then(|c| c.time_based)
                .is_some()
            {
                continue;
            }
            let usage = Usage {
                input,
                output,
                cache_read,
                cache_write,
                ..Usage::default()
            };
            let Some(cost) = calculate_catalog_cost(&provider, &model, &usage, timestamp) else {
                continue;
            };
            if cost.total == 0.0 {
                continue;
            }
            update.execute(rusqlite::params![
                cost.input,
                cost.output,
                cost.cache_read,
                cost.cache_write,
                cost.total,
                id
            ])?;
        }
    }
    tx.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params![CATALOG_COST_BACKFILL_KEY, CATALOG_BACKFILL_VERSION],
    )?;
    tx.commit()?;
    Ok(())
}

/// Rows predating the `cost_no_cache_input` column are NULL; one pass settles
/// the column for good.
pub(crate) fn backfill_no_cache_input_costs(conn: &Connection) -> Result<()> {
    if meta_value(conn, NO_CACHE_INPUT_BACKFILL_KEY)?.as_deref() == Some(BACKFILL_COMPLETE) {
        return Ok(());
    }
    let mut stmt = conn.prepare(
        "SELECT id, provider, model, timestamp, input_tokens, cache_read_tokens, cache_write_tokens
         FROM messages WHERE cost_no_cache_input IS NULL",
    )?;
    let rows = {
        stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?
    };
    let tx = conn.unchecked_transaction()?;
    {
        let mut update =
            tx.prepare("UPDATE messages SET cost_no_cache_input = ?1 WHERE id = ?2")?;
        for (id, provider, model, timestamp, input, cache_read, cache_write) in rows {
            let usage = Usage {
                input,
                output: 0,
                cache_read,
                cache_write,
                ..Usage::default()
            };
            let cost =
                calculate_no_cache_input_cost(&provider, &model, &usage, timestamp).unwrap_or(0.0);
            update.execute(rusqlite::params![cost, id])?;
        }
    }
    tx.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params![NO_CACHE_INPUT_BACKFILL_KEY, BACKFILL_COMPLETE],
    )?;
    tx.commit()?;
    Ok(())
}

/// One-time collapse of forked-session duplicates that landed under the old
/// `UNIQUE(session_file, entry_id)`-only invariant: keep the lowest-`id` row
/// per `(entry_id, timestamp)` (almost always the parent) and drop copies.
pub(crate) fn backfill_fork_duplicates(conn: &Connection) -> Result<()> {
    if meta_value(conn, FORK_DEDUPE_KEY)?.as_deref() == Some(BACKFILL_COMPLETE) {
        return Ok(());
    }
    let tx = conn.unchecked_transaction()?;
    tx.execute_batch(
        "DELETE FROM messages WHERE id NOT IN (SELECT MIN(id) FROM messages GROUP BY entry_id, timestamp);
         DELETE FROM user_messages WHERE id NOT IN (SELECT MIN(id) FROM user_messages GROUP BY entry_id, timestamp);",
    )?;
    tx.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params![FORK_DEDUPE_KEY, BACKFILL_COMPLETE],
    )?;
    tx.commit()?;
    Ok(())
}

/// TS `calculateCatalogCost`: price the four stored buckets (plus
/// orchestration, when the input carries it) at the request timestamp.
pub(crate) fn calculate_catalog_cost(
    provider: &str,
    model_id: &str,
    tokens: &Usage,
    timestamp: i64,
) -> Option<crate::catalog::UsageCost> {
    let cost = get_catalog_cost(provider, model_id)?;
    // catalog::calculate_usage_cost already folds orchestration into the
    // priced buckets (mirrors the TS totalTokens rebuild).
    let mut usage = *tokens;
    Some(calculate_usage_cost(&cost, &mut usage, Some(timestamp)))
}

#[cfg(feature = "db_ingest")]
fn normalize_usage_cost(cost: &UsageInputCost) -> crate::catalog::UsageCost {
    let input = cost.input.unwrap_or(0.0);
    let output = cost.output.unwrap_or(0.0);
    let cache_read = cost.cache_read.unwrap_or(0.0);
    let cache_write = cost.cache_write.unwrap_or(0.0);
    crate::catalog::UsageCost {
        input,
        output,
        cache_read,
        cache_write,
        total: cost
            .total
            .unwrap_or(input + output + cache_read + cache_write),
    }
}

/// The parser records no timestamp for an entry that carries neither a numeric
/// message timestamp nor a parseable entry timestamp, leaving this sentinel.
fn has_request_timestamp(timestamp: i64) -> bool {
    timestamp > 0
}

#[cfg(feature = "db_ingest")]
pub(crate) struct ResolvedCost {
    pub cost: crate::catalog::UsageCost,
    /// The stored zero is unknown spend rather than a price: a scheduled card
    /// with no recoverable request timestamp to select a tariff from.
    pub unpriced: bool,
}

/// TS `resolveStoredCost`: scheduled prices are frozen per request, including
/// explicitly free usage; preserve legacy zero-cost subscription correction
/// for unscheduled models.
#[cfg(feature = "db_ingest")]
pub(crate) fn resolve_stored_cost(stats: &MessageStatsInput) -> ResolvedCost {
    let zero = crate::catalog::UsageCost::default();
    // `usage.cost` is absent when the session entry recorded no price at all,
    // and legacy payloads can carry a partially-populated cost object.
    let stored_cost = stats.usage.cost.as_ref().map(normalize_usage_cost);
    let catalog = get_catalog_cost(&stats.provider, &stats.model);

    if let Some(stored) = stored_cost {
        if stored.total != 0.0
            || catalog
                .as_ref()
                .map(|c| c.time_based.is_some())
                .unwrap_or(false)
        {
            return ResolvedCost {
                cost: stored,
                unpriced: false,
            };
        }
    }
    // Without a request timestamp a scheduled card has no tariff to select,
    // and the epoch would silently become one.
    if !has_request_timestamp(stats.timestamp)
        && catalog
            .as_ref()
            .map(|c| c.time_based.is_some())
            .unwrap_or(false)
    {
        return ResolvedCost {
            cost: stored_cost.unwrap_or(zero),
            unpriced: true,
        };
    }
    let cost = calculate_catalog_cost(
        &stats.provider,
        &stats.model,
        &usage_from_input(&stats.usage),
        stats.timestamp,
    )
    .or(stored_cost)
    .unwrap_or(zero);
    ResolvedCost {
        cost,
        unpriced: false,
    }
}

/// TS `calculateNoCacheInputCost`: price the prompt as if fully uncached.
/// An unpriced scheduled request must not report its whole prompt as cache
/// savings just because the clock could be read.
pub(crate) fn calculate_no_cache_input_cost(
    provider: &str,
    model_id: &str,
    tokens: &Usage,
    timestamp: i64,
) -> Option<f64> {
    let cost = get_catalog_cost(provider, model_id)?;
    if !has_request_timestamp(timestamp) && cost.time_based.is_some() {
        return None;
    }
    let orch = tokens.orchestration.unwrap_or_default();
    let prompt_input_tokens = tokens.input
        + tokens.cache_read
        + tokens.cache_write
        + orch.input.unwrap_or(0)
        + orch.cache_read.unwrap_or(0);
    Some(calculate_uncached_input_cost(
        &cost,
        prompt_input_tokens,
        Some(timestamp),
    ))
}

/// Map the wire `UsageInput` onto the pricing `Usage` buckets.
/// DEVIATION: `UsageInput` (types.rs) has no orchestration/cttl fields, so
/// orchestration-billed subscription models price from the conversation
/// buckets only (see report).
#[cfg(feature = "db_ingest")]
pub(crate) fn usage_from_input(input: &crate::types::UsageInput) -> Usage {
    Usage {
        input: input.input,
        output: input.output,
        cache_read: input.cache_read,
        cache_write: input.cache_write,
        orchestration: input
            .orchestration
            .map(|o| crate::catalog::OrchestrationUsage {
                input: o.input,
                cache_read: o.cache_read,
                output: o.output,
            }),
        cttl: input.cttl.map(|c| crate::catalog::CttlUsage {
            ephemeral5m: c.ephemeral5m,
            ephemeral1h: c.ephemeral1h,
        }),
        ..Usage::default()
    }
}

#[cfg(all(test, feature = "db_ingest"))]
mod orchestration_tests {
    use super::{calculate_catalog_cost, usage_from_input};

    /// Review regression (RevCatalog #1): orchestration and cttl from the
    /// wire usage must reach pricing. Expected values from TS
    /// `calculateUsageCost(getBundledModel("anthropic","claude-sonnet-5-5").cost, …)`.
    #[test]
    fn orchestration_and_cttl_are_priced_like_ts() {
        let input: crate::types::UsageInput = serde_json::from_value(serde_json::json!({
            "input": 1000, "output": 500, "cacheRead": 2000, "cacheWrite": 3000, "totalTokens": 0,
            "orchestration": {"input": 400, "output": 100, "cacheRead": 50},
            "cttl": {"ephemeral5m": 1000, "ephemeral1h": 2000}
        }))
        .expect("wire usage");
        let cost = calculate_catalog_cost(
            "anthropic",
            "claude-sonnet-5-5",
            &usage_from_input(&input),
            1_792_000_000_000,
        )
        .expect("priced");
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12;
        assert!(close(cost.input, 0.0028), "{cost:?}");
        assert!(close(cost.output, 0.006), "{cost:?}");
        assert!(close(cost.cache_read, 0.000_205), "{cost:?}");
        assert!(close(cost.cache_write, 0.0105), "{cost:?}");
        assert!(close(cost.total, 0.019_505), "{cost:?}");
    }
}
