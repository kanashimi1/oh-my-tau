//! Range sources: the `UNION ALL` that merges clean rollup hours, the partial
//! hour at the range start and dirty hours, plus the `queryMessages` /
//! `queryTools` wrappers every read goes through.

use anyhow::Result;
use rusqlite::types::Value;
use rusqlite::{params_from_iter, Connection, Row};

use super::{
    hour_ceil, hour_of, message_facts, tool_facts, with_rollup_db, RangeWindow, EXACT_DIRTY_LIMIT,
    HOUR_MS, MESSAGE_DIMENSIONS, MESSAGE_ROLLUP_COLUMNS, TOOL_JOIN, TOOL_ROLLUP_COLUMNS,
};

pub(crate) struct Source {
    pub sql: String,
    pub params: Vec<Value>,
}

/// Above this many dirty hours, reads use stale rollup rows instead of scanning
/// dirty hours raw.
pub(crate) fn dirty_is_small(conn: &Connection) -> Result<bool> {
    let n: i64 = conn.query_row(
        &format!(
            "SELECT COUNT(*) AS n FROM (SELECT 1 FROM rollup_dirty LIMIT {})",
            EXACT_DIRTY_LIMIT + 1
        ),
        [],
        |row| row.get(0),
    )?;
    Ok(n <= EXACT_DIRTY_LIMIT)
}

/// Message facts covering `[cutoff, now]`, at hour granularity (5-minute raw
/// buckets when `fine`). See the module docs for the union's parts.
fn message_source(conn: &Connection, cutoff: Option<i64>, fine: bool) -> Result<Source> {
    let group_raw = format!("GROUP BY 1, {MESSAGE_DIMENSIONS}");
    if fine {
        return Ok(Source {
            sql: format!(
                "SELECT {facts} FROM messages WHERE timestamp >= ? {group_raw}",
                facts = message_facts("(timestamp / 300000) * 300000", "")
            ),
            params: vec![Value::Integer(cutoff.unwrap_or(0))],
        });
    }
    let hi = cutoff.map(hour_ceil);
    let exact = dirty_is_small(conn)?;
    let mut parts = Vec::new();
    let mut params = Vec::new();
    parts.push(format!(
        "SELECT {MESSAGE_ROLLUP_COLUMNS} FROM message_rollup WHERE {clean}{hi_filter}",
        clean = if exact {
            "bucket NOT IN (SELECT bucket FROM rollup_dirty)"
        } else {
            "1"
        },
        hi_filter = if hi.is_some() { " AND bucket >= ?" } else { "" }
    ));
    if let Some(hi) = hi {
        params.push(Value::Integer(hi));
    }
    if let Some(cutoff) = cutoff {
        if let Some(hi) = hi {
            if cutoff < hi {
                parts.push(format!(
                    "SELECT {facts} FROM messages WHERE timestamp >= ? AND timestamp < ? {group_raw}",
                    facts = message_facts(&hour_of("timestamp"), "")
                ));
                params.push(Value::Integer(cutoff));
                params.push(Value::Integer(hi));
            }
        }
    }
    if exact {
        parts.push(format!(
            "SELECT {facts} FROM rollup_dirty d JOIN messages m ON m.timestamp >= d.bucket AND m.timestamp < d.bucket + {HOUR_MS} {where} GROUP BY d.bucket, m.model, m.provider, m.folder, m.agent_type",
            facts = message_facts("d.bucket", "m."),
            where = if hi.is_some() { "WHERE d.bucket >= ?" } else { "" }
        ));
        if let Some(hi) = hi {
            params.push(Value::Integer(hi));
        }
    }
    Ok(Source {
        sql: parts.join(" UNION ALL "),
        params,
    })
}

/// Tool facts covering `[cutoff, now]`; same union shape as [`message_source`].
fn tool_source(conn: &Connection, cutoff: Option<i64>, fine: bool) -> Result<Source> {
    let group_raw = "GROUP BY 1, t.tool_name, t.model, t.provider";
    if fine {
        return Ok(Source {
            sql: format!(
                "SELECT {facts} FROM tool_calls t {TOOL_JOIN} WHERE t.timestamp >= ? {group_raw}",
                facts = tool_facts("(t.timestamp / 300000) * 300000")
            ),
            params: vec![Value::Integer(cutoff.unwrap_or(0))],
        });
    }
    let hi = cutoff.map(hour_ceil);
    let exact = dirty_is_small(conn)?;
    let mut parts = Vec::new();
    let mut params = Vec::new();
    parts.push(format!(
        "SELECT {TOOL_ROLLUP_COLUMNS} FROM tool_rollup WHERE {clean}{hi_filter}",
        clean = if exact {
            "bucket NOT IN (SELECT bucket FROM rollup_dirty)"
        } else {
            "1"
        },
        hi_filter = if hi.is_some() { " AND bucket >= ?" } else { "" }
    ));
    if let Some(hi) = hi {
        params.push(Value::Integer(hi));
    }
    if let Some(cutoff) = cutoff {
        if let Some(hi) = hi {
            if cutoff < hi {
                parts.push(format!(
                    "SELECT {facts} FROM tool_calls t {TOOL_JOIN} WHERE t.timestamp >= ? AND t.timestamp < ? {group_raw}",
                    facts = tool_facts(&hour_of("t.timestamp"))
                ));
                params.push(Value::Integer(cutoff));
                params.push(Value::Integer(hi));
            }
        }
    }
    if exact {
        parts.push(format!(
            "SELECT {facts} FROM rollup_dirty d JOIN tool_calls t ON t.timestamp >= d.bucket AND t.timestamp < d.bucket + {HOUR_MS} {TOOL_JOIN} {where} GROUP BY d.bucket, t.tool_name, t.model, t.provider",
            facts = tool_facts("d.bucket"),
            where = if hi.is_some() { "WHERE d.bucket >= ?" } else { "" }
        ));
        if let Some(hi) = hi {
            params.push(Value::Integer(hi));
        }
    }
    Ok(Source {
        sql: parts.join(" UNION ALL "),
        params,
    })
}

fn normalize_cutoff(cutoff: Option<i64>) -> Option<i64> {
    cutoff.filter(|cutoff| *cutoff > 0)
}

pub(crate) fn series_bucket(bucket_ms: i64) -> String {
    format!("(f.bucket / {bucket_ms}) * {bucket_ms}")
}

/// Run `select … FROM (<facts>) f <tail>` over the message facts for a window.
pub(crate) fn query_messages<T>(
    window: &RangeWindow,
    select: &str,
    tail: &str,
    mut map: impl FnMut(&Row) -> rusqlite::Result<T>,
) -> Result<Vec<T>> {
    let fine = window.fine();
    let cutoff = normalize_cutoff(window.cutoff);
    with_rollup_db(|conn| {
        let source = message_source(conn, cutoff, fine)?;
        let sql = format!("SELECT {select} FROM ({}) f {tail}", source.sql);
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(source.params.iter()), &mut map)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    })
}

/// Tool-side twin of [`query_messages`].
pub(crate) fn query_tools<T>(
    window: &RangeWindow,
    select: &str,
    tail: &str,
    mut map: impl FnMut(&Row) -> rusqlite::Result<T>,
) -> Result<Vec<T>> {
    let fine = window.fine();
    let cutoff = normalize_cutoff(window.cutoff);
    with_rollup_db(|conn| {
        let source = tool_source(conn, cutoff, fine)?;
        let sql = format!("SELECT {select} FROM ({}) f {tail}", source.sql);
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(source.params.iter()), &mut map)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    })
}
