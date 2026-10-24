//! TS `withStatsSyncLock` over pi-utils `withFileLock` (pi-natives `FileLock`):
//! an OS-owned exclusive lease on `<stats.db>.sync.lock`. The kernel releases it
//! when the owner exits, so an interrupted owner never leaves a stale lock and a
//! live owner is never displaced. Same primitive as the native TS lock so a Rust
//! and a TS omp sharing one stats db exclude each other:
//! - Linux: bind an abstract Unix datagram socket named
//!   `omp-file-lock-<xxh64(path, HIGH)><xxh64(path, LOW)>`;
//! - other Unix: `flock(LOCK_EX|LOCK_NB)` on the `.lock` file;
//! - Windows (TS: named mutex): an exclusive file lock on the `.lock` file.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context as _, Result};

const STATS_SYNC_LOCK_RETRY_MS: u64 = 25;
const STATS_SYNC_LOCK_WAIT_MS: u64 = 60 * 60 * 1000;

/// Serialize stats ingestion and archive reconciliation across processes (and
/// threads of this one). The lock covers file discovery, parsing, and the final
/// SQLite write so a parse result for a session moved by GC can never commit
/// after cleanup.
///
/// # Errors
/// Directory creation, lock acquisition (after waiting an hour), or `f`.
pub fn with_stats_sync_lock<T>(db_path: &Path, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if let Some(dir) = db_path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    let mut target = db_path.as_os_str().to_owned();
    target.push(".sync");
    let retries = STATS_SYNC_LOCK_WAIT_MS.div_ceil(STATS_SYNC_LOCK_RETRY_MS);
    let lock = acquire(
        Path::new(&target),
        retries,
        Duration::from_millis(STATS_SYNC_LOCK_RETRY_MS),
    )?;
    let out = f();
    drop(lock);
    out
}

/// pi-utils `getLockPath`: `${path.resolve(filePath)}.lock`. `path.resolve`
/// also folds `.`/`..` lexically (`std::path::absolute` keeps `..`), and the
/// Linux lock name hashes this string, so it must match TS byte for byte.
fn lock_path(file_path: &Path) -> PathBuf {
    use std::path::Component;
    let absolute = std::path::absolute(file_path).unwrap_or_else(|_| file_path.to_path_buf());
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            // `/..` stays `/`, like path.resolve.
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                resolved.push(component)
            }
        }
    }
    let mut path = resolved.into_os_string();
    path.push(".lock");
    PathBuf::from(path)
}

/// pi-utils `acquireFileLock`: `retries` attempts (including the first), `delay` apart.
fn acquire(file_path: &Path, retries: u64, delay: Duration) -> Result<platform::Lock> {
    let path = lock_path(file_path);
    for attempt in 0..retries {
        if let Some(lock) = platform::try_acquire(&path)? {
            return Ok(lock);
        }
        if attempt + 1 < retries {
            std::thread::sleep(delay);
        }
    }
    bail!(
        "Failed to acquire lock for {} after {retries} attempts",
        file_path.display()
    )
}

#[cfg(target_os = "linux")]
mod platform {
    use std::os::linux::net::SocketAddrExt as _;
    use std::os::unix::net::{SocketAddr, UnixDatagram};
    use std::path::Path;

    use anyhow::{Context as _, Result};

    /// Held while the abstract socket stays bound.
    pub(super) struct Lock(#[allow(dead_code)] UnixDatagram);

    /// pi-natives `memory_lock_name`.
    fn memory_lock_name(path: &str) -> String {
        const HIGH_SEED: u64 = 0x4f4d_502d_4c4f_434b;
        const LOW_SEED: u64 = 0x5049_2d46_494c_454c;
        let high = xxhash_rust::xxh64::xxh64(path.as_bytes(), HIGH_SEED);
        let low = xxhash_rust::xxh64::xxh64(path.as_bytes(), LOW_SEED);
        format!("omp-file-lock-{high:016x}{low:016x}")
    }

    pub(super) fn try_acquire(path: &Path) -> Result<Option<Lock>> {
        let path = path.to_str().context("lock path is not valid UTF-8")?;
        let address = SocketAddr::from_abstract_name(memory_lock_name(path).as_bytes())?;
        match UnixDatagram::bind_addr(&address) {
            Ok(socket) => Ok(Some(Lock(socket))),
            Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => Ok(None),
            Err(error) => Err(error).context("bind stats sync lock"),
        }
    }

    #[cfg(test)]
    pub(super) fn lock_name(path: &str) -> String {
        memory_lock_name(path)
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use std::fs::{File, OpenOptions, TryLockError};
    use std::path::Path;

    use anyhow::{Context as _, Result};

    /// Held while the open file description keeps its exclusive lock.
    pub(super) struct Lock(#[allow(dead_code)] File);

    pub(super) fn try_acquire(path: &Path) -> Result<Option<Lock>> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("open {}", path.display()))?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Lock(file))),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(error)) => Err(error).context("lock stats sync file"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_excludes_second_holder_until_released() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("stats.db.sync");
        let first = acquire(&target, 1, Duration::ZERO).expect("first acquire");
        let second = acquire(&target, 2, Duration::from_millis(1));
        assert!(
            second.is_err(),
            "second holder must wait while the first is alive"
        );
        drop(first);
        assert!(
            acquire(&target, 1, Duration::ZERO).is_ok(),
            "released lock is free again"
        );
    }

    #[test]
    fn with_lock_runs_body_and_releases() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("nested/stats.db");
        let value = with_stats_sync_lock(&db, || Ok(7)).expect("locked body");
        assert_eq!(value, 7);
        assert!(db.parent().is_some_and(Path::is_dir), "db dir is created");
        let again = with_stats_sync_lock(&db, || Ok(8)).expect("lock released after body");
        assert_eq!(again, 8);
    }

    #[test]
    fn lock_path_normalizes_like_path_resolve() {
        let lock = |p: &str| lock_path(Path::new(p));
        assert_eq!(lock("/a/b/../stats.db.sync"), lock("/a/stats.db.sync"));
        assert_eq!(
            lock("/a/stats.db.sync"),
            PathBuf::from("/a/stats.db.sync.lock")
        );
        assert_eq!(
            lock("/a/./b//c/../../stats.db.sync"),
            PathBuf::from("/a/stats.db.sync.lock")
        );
        assert_eq!(lock("/../x.sync"), PathBuf::from("/x.sync.lock"));
        let cwd = std::env::current_dir().expect("cwd");
        assert_eq!(lock("x/../y.sync"), cwd.join("y.sync.lock"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn lock_name_matches_pi_natives_format() {
        let name = platform::lock_name("/tmp/x.lock");
        assert!(name.starts_with("omp-file-lock-"));
        assert_eq!(name.len(), "omp-file-lock-".len() + 32);
    }
}
