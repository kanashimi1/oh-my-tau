//! SQLite-WAL store (`SPEC.md`): envelopes with per-recipient seq and a
//! timestamped state log, dedup on `message_id`, claims with monotonic
//! fencing tokens, and endpoint bookkeeping. Survives hub restarts.

use parking_lot::Mutex;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{Map, Value};

use crate::envelope::{Envelope, PayloadRef, Priority, State};

/// One envelope row as stored.
#[derive(Debug, Clone)]
pub struct EnvelopeRow {
    pub envelope: Envelope,
    pub reply_expected: bool,
    pub created_ms: u64,
    pub updated_ms: u64,
}

/// A live claim.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClaimRow {
    pub resource: String,
    pub owner: String,
    pub fencing_token: u64,
    pub expires_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    pub acquired_ms: u64,
    /// Computed at read time: `expires_ms <= now`.
    pub expired: bool,
}

/// Result of a `claim` attempt.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClaimOutcome {
    pub granted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fencing_token: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holder: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_ms: Option<u64>,
}

/// One reply-expected envelope still waiting.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PendingReply {
    pub message_id: String,
    pub recipient: String,
    pub kind: String,
    pub age_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<u64>,
}

/// Per-recipient or global status aggregation.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StatusReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipient: Option<String>,
    /// Unread envelopes (state `sent`/`delivered`), oldest first.
    pub unread: Vec<Envelope>,
    /// Reply-expected envelopes still unanswered, with age.
    pub pending_replies: Vec<PendingReply>,
    /// Active (unexpired) claims.
    pub active_claims: Vec<ClaimRow>,
    /// Recipients with unread mail, `recipient -> count`.
    pub unread_by_recipient: Vec<(String, u64)>,
}

/// Result of inserting an envelope: `dedup` when the `message_id` existed.
#[derive(Debug, Clone)]
pub struct InsertOutcome {
    pub seq: u64,
    pub state: String,
    pub dedup: bool,
}

const SCHEMA: &str = r"
CREATE TABLE IF NOT EXISTS envelopes (
  message_id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL,
  run_id TEXT, parent_id TEXT, correlation_id TEXT, causation_id TEXT,
  sender TEXT NOT NULL, recipient TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'agent.message', priority INTEGER NOT NULL DEFAULT 1,
  expected TEXT, reply_schema TEXT, deadline_ms INTEGER,
  payload TEXT, payload_path TEXT, payload_sha256 TEXT, payload_bytes INTEGER,
  reply_expected INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL, state_log TEXT NOT NULL,
  reply_message_id TEXT,
  created_ms INTEGER NOT NULL, updated_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_env_recipient_seq ON envelopes(recipient, seq);
CREATE INDEX IF NOT EXISTS idx_env_state_deadline ON envelopes(state, deadline_ms);
CREATE INDEX IF NOT EXISTS idx_env_sender ON envelopes(sender);
CREATE TABLE IF NOT EXISTS seqs (
  recipient TEXT PRIMARY KEY, next INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS claims (
  resource TEXT PRIMARY KEY, owner TEXT NOT NULL, fencing_token INTEGER NOT NULL,
  expires_ms INTEGER NOT NULL, intent TEXT, acquired_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS endpoints (
  agent TEXT PRIMARY KEY, last_hello_ms INTEGER NOT NULL DEFAULT 0,
  last_seq INTEGER NOT NULL DEFAULT 0, registered_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY, value INTEGER NOT NULL
);
";

fn i64_of(v: u64) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}
fn i64_of_sz(v: usize) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}

/// Thread-safe wrapper over one connection. All mutations hold the mutex;
/// `BEGIN IMMEDIATE` guards the seq assignment + insert pair.
pub struct Store {
    db_path: PathBuf,
    conn: Mutex<Connection>,
}

impl Store {
    /// Open (creating if needed) `<dir>/hub.db` in WAL mode.
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        let db_path = dir.join("hub.db");
        let conn =
            Connection::open(&db_path).with_context(|| format!("open {}", db_path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        conn.execute_batch(SCHEMA).context("hub schema")?;
        Ok(Self {
            db_path,
            conn: Mutex::new(conn),
        })
    }

    /// Path of the underlying db (for diagnostics).
    #[must_use]
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    fn lock(&self) -> parking_lot::MutexGuard<'_, Connection> {
        self.conn.lock()
    }

    /// Store an envelope: dedup on `message_id`, assign the per-recipient
    /// seq, set `sent` + state log, default the deadline when
    /// reply-expected. On dedup returns the stored seq/state.
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    #[allow(clippy::too_many_lines)] // one transaction: seq upsert + insert.
    pub fn insert(&self, env: &mut Envelope, now: u64) -> Result<InsertOutcome> {
        if let Some(row) = self.get(&env.message_id)? {
            return Ok(InsertOutcome {
                seq: row.envelope.seq.unwrap_or(0),
                state: row
                    .envelope
                    .state
                    .unwrap_or_else(|| State::SENT.to_string()),
                dedup: true,
            });
        }
        let c = self.lock();
        let reply_expected = env.is_reply_expected();
        let deadline = env.effective_deadline(now);
        let state_log = {
            let mut m = Map::new();
            m.insert(State::SENT.to_string(), Value::from(now));
            m
        };
        c.execute("BEGIN IMMEDIATE", [])?;
        let result = (|| -> Result<u64> {
            c.execute(
                "INSERT INTO seqs(recipient, next) VALUES(?1, 2)
                 ON CONFLICT(recipient) DO UPDATE SET next = next + 1",
                params![env.recipient],
            )?;
            let seq: i64 = c.query_row(
                "SELECT next - 1 FROM seqs WHERE recipient = ?1",
                params![env.recipient],
                |r| r.get(0),
            )?;
            let seq = u64::try_from(seq).unwrap_or(0);
            let reply_schema = env
                .reply_schema
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?;
            let state_log_s = serde_json::to_string(&state_log)?;
            let (p_path, p_sha, p_bytes) =
                env.payload_ref
                    .as_ref()
                    .map_or((None, None, None), |r: &PayloadRef| {
                        (
                            Some(r.path.clone()),
                            Some(r.sha256.clone()),
                            Some(i64_of(r.bytes)),
                        )
                    });
            c.execute(
                "INSERT INTO envelopes(
                   message_id, seq, run_id, parent_id, correlation_id, causation_id,
                   sender, recipient, kind, priority, expected, reply_schema, deadline_ms,
                   payload, payload_path, payload_sha256, payload_bytes,
                   reply_expected, state, state_log, created_ms, updated_ms)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?21)",
                params![
                    env.message_id,
                    i64_of(seq),
                    env.run_id,
                    env.parent_id,
                    env.correlation_id,
                    env.causation_id,
                    env.sender,
                    env.recipient,
                    env.kind,
                    i64::from(env.priority.0),
                    env.expected,
                    reply_schema,
                    deadline.map(i64_of),
                    env.payload,
                    p_path,
                    p_sha,
                    p_bytes,
                    i64::from(u8::from(reply_expected)),
                    State::SENT,
                    state_log_s,
                    i64_of(now),
                ],
            )?;
            Ok(seq)
        })();
        match result {
            Ok(seq) => {
                c.execute("COMMIT", []).ok();
                env.seq = Some(seq);
                env.state = Some(State::SENT.to_string());
                env.deadline = deadline;
                env.state_log = Some(state_log);
                Ok(InsertOutcome {
                    seq,
                    state: State::SENT.to_string(),
                    dedup: false,
                })
            }
            Err(e) => {
                c.execute("ROLLBACK", []).ok();
                // Concurrent send of the same message_id: the loser reports
                // the winner's row instead of a constraint error.
                if let Some(row) = self.get(&env.message_id)? {
                    return Ok(InsertOutcome {
                        seq: row.envelope.seq.unwrap_or(0),
                        state: row
                            .envelope
                            .state
                            .unwrap_or_else(|| State::SENT.to_string()),
                        dedup: true,
                    });
                }
                Err(e).context("insert envelope")
            }
        }
    }

    /// Fetch one envelope by id.
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn get(&self, message_id: &str) -> Result<Option<EnvelopeRow>> {
        let c = self.lock();
        row_by_id(&c, message_id).context("get envelope")
    }

    /// Move an envelope to a new state, appending to its state log.
    /// Idempotent (same state twice = ok); refuses moves out of terminal
    /// states or backward in the chain. Returns false when refused.
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn transition(&self, message_id: &str, new_state: &str, now: u64) -> Result<bool> {
        let Some(row) = self.get(message_id)? else {
            return Ok(false);
        };
        let cur = row.envelope.state.as_deref().unwrap_or(State::SENT);
        if cur == new_state {
            return Ok(true);
        }
        let allowed = match new_state {
            State::DELIVERED => cur == State::SENT,
            State::READ => matches!(cur, State::SENT | State::DELIVERED),
            State::ANSWERED => matches!(cur, State::SENT | State::DELIVERED | State::READ),
            State::EXPIRED | State::FAILED | State::CANCELLED => {
                matches!(cur, State::SENT | State::DELIVERED | State::READ)
            }
            _ => false,
        };
        if !allowed {
            return Ok(false);
        }
        let mut log = row.envelope.state_log.unwrap_or_default();
        log.insert(new_state.to_string(), Value::from(now));
        let log_s = serde_json::to_string(&log)?;
        let c = self.lock();
        c.execute(
            "UPDATE envelopes SET state = ?1, state_log = ?2, updated_ms = ?3
             WHERE message_id = ?4",
            params![new_state, log_s, i64_of(now), message_id],
        )?;
        Ok(true)
    }

    /// Record which message answered this one.
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn set_reply(&self, message_id: &str, reply_message_id: &str) -> Result<()> {
        let c = self.lock();
        c.execute(
            "UPDATE envelopes SET reply_message_id = ?1, updated_ms = ?2 WHERE message_id = ?3",
            params![reply_message_id, i64_of(crate::now_ms()), message_id],
        )?;
        Ok(())
    }

    /// Envelopes for a recipient with `seq > since_seq`, urgent first,
    /// then by seq. `unread_only` filters to states `sent`/`delivered`
    /// (the replay contract from `hello.last_seq`).
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn inbox(
        &self,
        recipient: &str,
        since_seq: u64,
        limit: usize,
        unread_only: bool,
    ) -> Result<Vec<EnvelopeRow>> {
        let c = self.lock();
        let mut stmt = c.prepare(&format!(
            "SELECT {COLUMNS} FROM envelopes WHERE recipient = ?1 AND seq > ?2
             ORDER BY priority ASC, seq ASC LIMIT ?3",
        ))?;
        let rows = stmt
            .query_map(
                params![recipient, i64_of(since_seq), i64_of_sz(limit)],
                row_from_sql,
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows
            .into_iter()
            .filter(|r| {
                !unread_only
                    || r.envelope
                        .state
                        .as_deref()
                        .is_some_and(|s| State::UNREAD.contains(&s))
            })
            .collect())
    }

    /// Reply-expected envelopes whose deadline passed and which are still
    /// `sent`/`delivered`: to be marked `expired` by the caller.
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn due_expired(&self, now: u64) -> Result<Vec<EnvelopeRow>> {
        let c = self.lock();
        let mut stmt = c.prepare(&format!(
            "SELECT {COLUMNS} FROM envelopes
             WHERE deadline_ms IS NOT NULL AND deadline_ms <= ?1
               AND state IN ('sent', 'delivered')
             ORDER BY priority ASC, seq ASC LIMIT 500",
        ))?;
        let mapped = stmt.query_map(params![i64_of(now)], row_from_sql)?;
        let rows = mapped.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Aggregate status for one recipient.
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn status_recipient(&self, recipient: &str, now: u64) -> Result<StatusReport> {
        let c = self.lock();
        let mut report = StatusReport {
            recipient: Some(recipient.to_string()),
            ..StatusReport::default()
        };
        let mut stmt = c.prepare(&format!(
            "SELECT {COLUMNS} FROM envelopes WHERE recipient = ?1
             AND state IN ('sent','delivered')
             ORDER BY priority ASC, seq ASC LIMIT 50",
        ))?;
        let mapped = stmt.query_map(params![recipient], row_from_sql)?;
        let rows = mapped.collect::<std::result::Result<Vec<_>, _>>()?;
        report.unread = rows.into_iter().map(|r| r.envelope).collect();
        report.pending_replies = pending_rows(&c, Some(recipient), now)?;
        report.active_claims = claim_rows(&c, None, Some(recipient), now)?;
        Ok(report)
    }

    /// Aggregate status across all recipients.
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn status_all(&self, now: u64) -> Result<StatusReport> {
        let c = self.lock();
        let mut stmt = c.prepare(
            "SELECT recipient, COUNT(*) FROM envelopes
             WHERE state IN ('sent','delivered') GROUP BY recipient ORDER BY recipient",
        )?;
        let mapped = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                u64::try_from(r.get::<_, i64>(1)?).unwrap_or(0),
            ))
        })?;
        let unread_by_recipient = mapped.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(StatusReport {
            unread_by_recipient,
            pending_replies: pending_rows(&c, None, now)?,
            active_claims: claim_rows(&c, None, None, now)?,
            ..StatusReport::default()
        })
    }

    /// Try to claim a resource. An existing claim blocks only while
    /// unexpired and held by someone else; the outcome then names the
    /// current holder and expiry. Every grant gets a fresh monotonic
    /// fencing token.
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn claim(
        &self,
        resource: &str,
        owner: &str,
        ttl_s: u64,
        intent: Option<&str>,
        now: u64,
    ) -> Result<ClaimOutcome> {
        let expires = now + ttl_s.saturating_mul(1000);
        let c = self.lock();
        if let Some(row) = claim_row(&c, resource)? {
            let stale = row.expires_ms <= now;
            if !stale && row.owner != owner {
                return Ok(ClaimOutcome {
                    granted: false,
                    fencing_token: None,
                    holder: Some(row.owner),
                    expires_ms: Some(row.expires_ms),
                });
            }
        }
        let token: i64 = {
            c.execute(
                "INSERT INTO meta(key, value) VALUES('fencing', 1)
                 ON CONFLICT(key) DO UPDATE SET value = value + 1",
                [],
            )?;
            c.query_row("SELECT value FROM meta WHERE key = 'fencing'", [], |r| {
                r.get(0)
            })?
        };
        c.execute(
            "INSERT INTO claims(resource, owner, fencing_token, expires_ms, intent, acquired_ms)
             VALUES (?1,?2,?3,?4,?5,?6)
             ON CONFLICT(resource) DO UPDATE SET
               owner = excluded.owner, fencing_token = excluded.fencing_token,
               expires_ms = excluded.expires_ms, intent = excluded.intent,
               acquired_ms = excluded.acquired_ms",
            params![resource, owner, token, i64_of(expires), intent, i64_of(now)],
        )?;
        Ok(ClaimOutcome {
            granted: true,
            fencing_token: Some(u64::try_from(token).unwrap_or(0)),
            holder: Some(owner.to_string()),
            expires_ms: Some(expires),
        })
    }

    /// Release a claim; requires the exact owner and fencing token and an
    /// unexpired lease.
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn release(
        &self,
        resource: &str,
        owner: &str,
        fencing_token: u64,
        now: u64,
    ) -> Result<bool> {
        let c = self.lock();
        let Some(row) = claim_row(&c, resource)? else {
            return Ok(false);
        };
        if row.owner != owner || row.fencing_token != fencing_token || row.expires_ms <= now {
            return Ok(false);
        }
        let n = c.execute("DELETE FROM claims WHERE resource = ?1", params![resource])?;
        Ok(n > 0)
    }

    /// All active (unexpired) claims, optionally filtered by resource prefix
    /// or owner.
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn claims(
        &self,
        prefix: Option<&str>,
        owner: Option<&str>,
        now: u64,
    ) -> Result<Vec<ClaimRow>> {
        let c = self.lock();
        claim_rows(&c, prefix, owner, now)
    }

    /// Who holds this resource right now, if anyone. Expired leases are
    /// reported as free (`expiry frees it`).
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn check(&self, resource: &str, now: u64) -> Result<Option<ClaimRow>> {
        let c = self.lock();
        Ok(claim_row(&c, resource)?.filter(|r| r.expires_ms > now))
    }

    /// Record a `hello` from a live endpoint.
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn hello(&self, agent: &str, last_seq: u64, now: u64) -> Result<()> {
        let c = self.lock();
        c.execute(
            "INSERT INTO endpoints(agent, last_hello_ms, last_seq, registered_ms)
             VALUES (?1,?2,?3,?3)
             ON CONFLICT(agent) DO UPDATE SET
               last_hello_ms = excluded.last_hello_ms,
               last_seq = MAX(endpoints.last_seq, excluded.last_seq)",
            params![agent, i64_of(now), i64_of(last_seq)],
        )?;
        Ok(())
    }

    /// Highest seq the hub has handed this recipient (for replay checks).
    ///
    /// # Errors
    /// Errors from the `SQLite` store.
    pub fn last_seq_for(&self, recipient: &str) -> Result<u64> {
        let c = self.lock();
        Ok(c.query_row(
            "SELECT COALESCE(MAX(seq), 0) FROM envelopes WHERE recipient = ?1",
            params![recipient],
            |r| r.get::<_, i64>(0),
        )
        .map(|v| u64::try_from(v).unwrap_or(0))?)
    }
}

fn pending_rows(c: &Connection, recipient: Option<&str>, now: u64) -> Result<Vec<PendingReply>> {
    let sql = format!(
        "SELECT message_id, recipient, kind, created_ms, deadline_ms FROM envelopes
         WHERE reply_expected = 1 AND state IN ('sent','delivered'){} 
         ORDER BY deadline_ms IS NULL, deadline_ms ASC LIMIT 100",
        if recipient.is_some() {
            " AND recipient = ?1"
        } else {
            ""
        }
    );
    let mut stmt = c.prepare(&sql)?;
    let map = |r: &rusqlite::Row<'_>| {
        let created: i64 = r.get(3)?;
        let deadline: Option<i64> = r.get(4)?;
        Ok(PendingReply {
            message_id: r.get(0)?,
            recipient: r.get(1)?,
            kind: r.get(2)?,
            age_ms: now.saturating_sub(u64::try_from(created).unwrap_or(now)),
            deadline_ms: deadline.and_then(|d| u64::try_from(d).ok()),
        })
    };
    let rows: Vec<PendingReply> = if let Some(rec) = recipient {
        stmt.query_map(params![rec], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        stmt.query_map([], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?
    };
    Ok(rows)
}

fn claim_row(c: &Connection, resource: &str) -> Result<Option<ClaimRow>> {
    c.prepare(
        "SELECT resource, owner, fencing_token, expires_ms, intent, acquired_ms
         FROM claims WHERE resource = ?1",
    )?
    .query_row(params![resource], |r| {
        let expires: i64 = r.get(3)?;
        Ok(ClaimRow {
            resource: r.get(0)?,
            owner: r.get(1)?,
            fencing_token: u64::try_from(r.get::<_, i64>(2)?).unwrap_or(0),
            expires_ms: u64::try_from(expires).unwrap_or(0),
            intent: r.get(4)?,
            acquired_ms: u64::try_from(r.get::<_, i64>(5)?).unwrap_or(0),
            expired: false,
        })
    })
    .optional()
    .map_err(Into::into)
}

fn claim_rows(
    c: &Connection,
    prefix: Option<&str>,
    owner: Option<&str>,
    now: u64,
) -> Result<Vec<ClaimRow>> {
    let mut sql = String::from(
        "SELECT resource, owner, fencing_token, expires_ms, intent, acquired_ms
         FROM claims WHERE 1=1",
    );
    if prefix.is_some() {
        sql.push_str(" AND resource LIKE ?1 || '%'");
    }
    if owner.is_some() {
        sql.push_str(if prefix.is_some() {
            " AND owner = ?2"
        } else {
            " AND owner = ?1"
        });
    }
    sql.push_str(" ORDER BY resource");
    let mut stmt = c.prepare(&sql)?;
    let map = |r: &rusqlite::Row<'_>| {
        let expires: i64 = r.get(3)?;
        Ok(ClaimRow {
            resource: r.get(0)?,
            owner: r.get(1)?,
            fencing_token: u64::try_from(r.get::<_, i64>(2)?).unwrap_or(0),
            expires_ms: u64::try_from(expires).unwrap_or(0),
            intent: r.get(4)?,
            acquired_ms: u64::try_from(r.get::<_, i64>(5)?).unwrap_or(0),
            expired: false,
        })
    };
    let mut rows: Vec<ClaimRow> = match (prefix, owner) {
        (Some(p), Some(o)) => stmt
            .query_map(params![p, o], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?,
        (Some(p), None) => stmt
            .query_map(params![p], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?,
        (None, Some(o)) => stmt
            .query_map(params![o], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?,
        (None, None) => stmt
            .query_map([], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?,
    };
    for r in &mut rows {
        r.expired = r.expires_ms <= now;
    }
    rows.retain(|r| !r.expired);
    Ok(rows)
}

const COLUMNS: &str = "message_id, seq, run_id, parent_id, correlation_id, causation_id, \
    sender, recipient, kind, priority, expected, reply_schema, deadline_ms, \
    payload, payload_path, payload_sha256, payload_bytes, \
    reply_expected, state, state_log, reply_message_id, created_ms, updated_ms";

fn row_by_id(c: &Connection, message_id: &str) -> Result<Option<EnvelopeRow>> {
    let sql = format!("SELECT {COLUMNS} FROM envelopes WHERE message_id = ?1");
    c.prepare(&sql)?
        .query_row(params![message_id], row_from_sql)
        .optional()
        .map_err(Into::into)
}

fn row_from_sql(r: &rusqlite::Row<'_>) -> rusqlite::Result<EnvelopeRow> {
    let priority: i64 = r.get(9)?;
    let reply_schema: Option<String> = r.get(11)?;
    let deadline: Option<i64> = r.get(12)?;
    let payload_bytes: Option<i64> = r.get(16)?;
    let reply_expected: i64 = r.get(17)?;
    let state_log: String = r.get(19)?;
    let p_path: Option<String> = r.get(14)?;
    let p_sha: Option<String> = r.get(15)?;
    let payload_ref = p_path.map(|path| PayloadRef {
        path,
        sha256: p_sha.unwrap_or_default(),
        bytes: u64::try_from(payload_bytes.unwrap_or(0)).unwrap_or(0),
    });
    let state_log: Map<String, Value> = serde_json::from_str(&state_log).unwrap_or_default();
    let created: i64 = r.get(21)?;
    let updated: i64 = r.get(22)?;
    let envelope = Envelope {
        message_id: r.get(0)?,
        run_id: r.get(2)?,
        parent_id: r.get(3)?,
        correlation_id: r.get(4)?,
        causation_id: r.get(5)?,
        sender: r.get(6)?,
        recipient: r.get(7)?,
        kind: r.get(8)?,
        priority: Priority(u8::try_from(priority).unwrap_or(1)),
        expected: r.get(10)?,
        reply_schema: reply_schema
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok()),
        deadline: deadline.and_then(|d| u64::try_from(d).ok()),
        payload: r.get(13)?,
        payload_ref,
        seq: Some(u64::try_from(r.get::<_, i64>(1)?).unwrap_or(0)),
        state: Some(r.get(18)?),
        state_log: Some(state_log),
        reply_message_id: r.get(20)?,
    };
    Ok(EnvelopeRow {
        envelope,
        reply_expected: reply_expected != 0,
        created_ms: u64::try_from(created).unwrap_or(0),
        updated_ms: u64::try_from(updated).unwrap_or(0),
    })
}
