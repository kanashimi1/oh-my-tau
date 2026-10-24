//! Port of the connection half of `db.ts`: the process-global sqlite handle
//! the TS keeps in a module `let db`. Schema/migrations/backfill flags live in
//! `schema.rs`, the write path (file offsets, parse-result application,
//! message/tool/user-message inserts) in `ingest.rs`; reads are `db_query`.
//!
//! Every module touches sqlite only through [`with_db`]. Transactions on the
//! shared `&Connection` use `conn.unchecked_transaction()`.

#[cfg(feature = "db_ingest")]
mod ingest;
#[cfg(any(feature = "db_ingest", feature = "db_query"))]
pub(crate) mod rows;
mod schema;

use std::sync::Mutex;

use anyhow::{Context as _, Result};
use rusqlite::Connection;

#[cfg(feature = "db_ingest")]
pub use ingest::*;
pub use schema::*;

static DB: Mutex<Option<Connection>> = Mutex::new(None);

fn lock() -> std::sync::MutexGuard<'static, Option<Connection>> {
    DB.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// TS `initDb`: open `~/.omp/stats.db` once, set the pragmas, create/migrate
/// the schema. Idempotent; later calls are no-ops while the handle is open.
///
/// # Errors
/// Directory creation, open, pragma or schema failure.
pub fn init_db() -> Result<()> {
    let mut guard = lock();
    if guard.is_some() {
        return Ok(());
    }
    let root = crate::paths::config_root_dir();
    std::fs::create_dir_all(&root).with_context(|| format!("create {}", root.display()))?;
    let path = crate::paths::stats_db_path();
    let conn = Connection::open(&path).with_context(|| format!("open {}", path.display()))?;
    // Busy handler BEFORE any lock-taking statement (TS issue #2421).
    conn.busy_timeout(std::time::Duration::from_millis(5000))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    schema::ensure_schema(&conn)?;
    *guard = Some(conn);
    Ok(())
}

/// Run `f` on the shared connection, opening it first if needed.
///
/// # Errors
/// Whatever `init_db` or `f` returns.
pub fn with_db<T>(f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
    init_db()?;
    let guard = lock();
    let conn = guard.as_ref().context("stats db closed concurrently")?;
    f(conn)
}

/// TS `currentDb() !== null`.
#[must_use]
pub fn current_db_open() -> bool {
    lock().is_some()
}

/// TS `closeDb`.
pub fn close_db() {
    if let Some(conn) = lock().take() {
        let _ = conn.close();
    }
}
