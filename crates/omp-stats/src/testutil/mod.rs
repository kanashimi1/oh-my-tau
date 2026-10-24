//! Test-only helpers shared by every module that touches the global stats db.
//! `cargo test` runs all lib tests in one process; the db handle and `HOME`
//! are process-global, so db tests must hold [`TestHome`] for their duration.

#[cfg(feature = "db_ingest")]
pub(crate) mod golden;

use std::sync::{Mutex, MutexGuard};

static DB_TEST_LOCK: Mutex<()> = Mutex::new(());

/// An isolated `HOME` (fresh temp dir) with the global db closed; the db
/// opens lazily inside it on the next `with_db`. Drop closes the db again.
pub(crate) struct TestHome {
    _guard: MutexGuard<'static, ()>,
    /// The temp home; `<dir>/.omp/stats.db` is the db file.
    dir: tempfile::TempDir,
}

impl TestHome {
    pub(crate) fn new() -> Self {
        let guard = DB_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        #[cfg(feature = "db")]
        crate::db::close_db();
        let dir = tempfile::tempdir().expect("temp home");
        // SAFETY-free: tests serialize on DB_TEST_LOCK, so no concurrent env readers here.
        std::env::set_var("HOME", dir.path());
        Self { _guard: guard, dir }
    }

    /// The temp `HOME` path.
    #[allow(dead_code)] // only some feature subsets' tests read it
    pub(crate) fn path(&self) -> &std::path::Path {
        self.dir.path()
    }
}

impl Drop for TestHome {
    fn drop(&mut self) {
        #[cfg(feature = "db")]
        crate::db::close_db();
    }
}
