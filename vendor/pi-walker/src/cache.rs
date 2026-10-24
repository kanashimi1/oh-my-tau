//! Shared walker scan cache used by owned-entry collection.

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt,
    path::{Component, Path, PathBuf, Prefix},
    sync::{Arc, LazyLock},
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use pi_vfs::BlockingFs;
use rayon::{ThreadPool, prelude::*};

use crate::{CollectedEntries, CollectedEntry, FileType, WalkError, WalkOptions};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CacheKey {
    root: PathBuf,
    options: WalkOptions,
}

#[derive(Clone)]
struct CacheEntry {
    created_at: Instant,
    entries: Arc<Vec<CollectedEntry>>,
    bytes: usize,
}

struct ScanCache {
    entries: HashMap<CacheKey, CacheEntry>,
    bytes: usize,
    generation: u64,
    ttl: Duration,
    max_entries: usize,
    max_bytes: usize,
}

impl ScanCache {
    fn new(ttl: Duration, max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            bytes: 0,
            generation: 0,
            ttl,
            max_entries,
            max_bytes,
        }
    }

    fn remove(&mut self, key: &CacheKey) {
        if let Some(entry) = self.entries.remove(key) {
            self.bytes -= entry.bytes;
        }
    }

    fn expire(&mut self, now: Instant) {
        self.entries.retain(|_, entry| {
            if now.saturating_duration_since(entry.created_at) < self.ttl {
                true
            } else {
                self.bytes -= entry.bytes;
                false
            }
        });
    }

    fn get(&mut self, key: &CacheKey, now: Instant) -> Option<CacheEntry> {
        self.expire(now);
        self.entries.get(key).cloned()
    }

    fn insert(&mut self, key: CacheKey, entry: CacheEntry, generation: u64, now: Instant) {
        self.expire(now);
        if generation != self.generation
            || self.max_entries == 0
            || entry.bytes > self.max_bytes
            || self.max_bytes == 0
            || now.saturating_duration_since(entry.created_at) >= self.ttl
            || self
                .entries
                .get(&key)
                .is_some_and(|cached| cached.created_at > entry.created_at)
        {
            return;
        }
        self.remove(&key);
        while self.entries.len() >= self.max_entries || self.bytes > self.max_bytes - entry.bytes {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, value)| value.created_at)
                .map(|(key, _)| key.clone());
            if let Some(oldest) = oldest {
                self.remove(&oldest);
            } else {
                break;
            }
        }
        self.bytes += entry.bytes;
        self.entries.insert(key, entry);
    }

    /// Drops entries under `target` (all when `None`) and, in the same pass,
    /// every TTL-expired entry: invalidation runs after each agent write, so
    /// this releases stale snapshots (up to the byte cap) without waiting for
    /// the next cache lookup.
    fn invalidate(&mut self, target: Option<&Path>, now: Instant) {
        self.generation = self.generation.wrapping_add(1);
        self.entries.retain(|key, entry| {
            if target.is_none_or(|target| target.starts_with(&key.root))
                || now.saturating_duration_since(entry.created_at) >= self.ttl
            {
                self.bytes -= entry.bytes;
                false
            } else {
                true
            }
        });
    }
}

fn entry_bytes(entries: &[CollectedEntry], capacity: usize) -> usize {
    entries.iter().fold(
        capacity.saturating_mul(size_of::<CollectedEntry>()),
        |bytes, entry| bytes.saturating_add(entry.path.capacity()),
    )
}

static CACHE_TTL_MS: LazyLock<u64> =
    LazyLock::new(|| env_uint("FS_SCAN_CACHE_TTL_MS", 1_000, 0, u64::MAX));
static EMPTY_RECHECK_MS: LazyLock<u64> =
    LazyLock::new(|| env_uint("FS_SCAN_EMPTY_RECHECK_MS", 200, 0, u64::MAX));
static MAX_CACHE_ENTRIES: LazyLock<usize> =
    LazyLock::new(|| env_uint("FS_SCAN_CACHE_MAX_ENTRIES", 16, 0, usize::MAX));
static MAX_CACHE_BYTES: LazyLock<usize> =
    LazyLock::new(|| env_uint("FS_SCAN_CACHE_MAX_BYTES", 64 * 1024 * 1024, 0, usize::MAX));
const DEFAULT_WALK_WORKERS: usize = 4;

static WALK_WORKERS: LazyLock<usize> = LazyLock::new(|| {
    normalize_worker_count(env_uint(
        "PI_WALK_WORKERS",
        DEFAULT_WALK_WORKERS,
        0,
        usize::MAX,
    ))
});
static WALK_POOL: LazyLock<Option<ThreadPool>> = LazyLock::new(|| {
    let workers = walk_workers();
    if workers <= 1 {
        return None;
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("pi-walker-{index}"))
        .build()
        .ok()
});
static SCAN_CACHE: LazyLock<Mutex<ScanCache>> = LazyLock::new(|| {
    Mutex::new(ScanCache::new(
        Duration::from_millis(*CACHE_TTL_MS),
        *MAX_CACHE_ENTRIES,
        *MAX_CACHE_BYTES,
    ))
});

fn env_uint<T>(name: &str, default: T, min: T, max: T) -> T
where
    T: Copy + Ord + std::str::FromStr,
{
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
        .clamp(min, max)
}

fn normalize_worker_count_with_available(configured: usize, available: usize) -> usize {
    if configured == 0 {
        available.max(1)
    } else {
        configured.max(1)
    }
}

fn available_worker_count() -> usize {
    std::thread::available_parallelism().map_or(DEFAULT_WALK_WORKERS, usize::from)
}

fn normalize_worker_count(configured: usize) -> usize {
    normalize_worker_count_with_available(configured, available_worker_count())
}

/// Configured cache TTL in milliseconds.
pub fn cache_ttl_ms() -> u64 {
    *CACHE_TTL_MS
}

/// Configured empty-result recheck threshold in milliseconds.
pub fn empty_recheck_ms() -> u64 {
    *EMPTY_RECHECK_MS
}

/// Configured maximum number of cache entries.
pub fn max_cache_entries() -> usize {
    *MAX_CACHE_ENTRIES
}

/// Effective worker count for filesystem traversal and related parallel work.
///
/// `PI_WALK_WORKERS=0` means auto-detect; `PI_WALK_WORKERS=1` forces serial
/// work.
pub fn walk_workers() -> usize {
    *WALK_WORKERS
}

/// Run parallel traversal-adjacent work on the centralized walker pool.
pub fn with_walk_pool<R>(operation: impl FnOnce() -> R + Send) -> R
where
    R: Send,
{
    if let Some(pool) = WALK_POOL.as_ref() {
        pool.install(operation)
    } else {
        operation()
    }
}

const PARALLEL_MIN_FILES: usize = 256;

/// Return whether traversal-adjacent work should run in parallel.
pub fn should_parallelize(item_count: usize) -> bool {
    walk_workers() > 1 && item_count >= PARALLEL_MIN_FILES
}

/// Run traversal-adjacent work serially or on the centralized walker pool.
pub fn parallel_for_each<T, E>(
    items: &[T],
    operation: impl Fn(&T) -> std::result::Result<(), E> + Send + Sync,
) -> std::result::Result<(), E>
where
    T: Sync,
    E: Send,
{
    if !should_parallelize(items.len()) {
        return items.iter().try_for_each(operation);
    }
    with_walk_pool(|| items.par_iter().try_for_each(operation))
}

/// Run traversal-adjacent work with per-worker state on the centralized walker
/// pool.
pub fn parallel_for_each_init<T, S, E>(
    items: &[T],
    init: impl Fn() -> S + Send + Sync,
    operation: impl Fn(&mut S, &T) -> std::result::Result<(), E> + Send + Sync,
) -> std::result::Result<(), E>
where
    T: Sync,
    S: Send,
    E: Send,
{
    if !should_parallelize(items.len()) {
        let mut state = init();
        return items
            .iter()
            .try_for_each(|item| operation(&mut state, item));
    }
    with_walk_pool(|| items.par_iter().try_for_each_init(init, operation))
}

fn cache_key(root: &Path, mut options: WalkOptions) -> CacheKey {
    options.cache = false;
    CacheKey {
        root: root.to_path_buf(),
        options,
    }
}

/// Normalize a filesystem path to a forward-slash string on Windows.
///
/// Verbatim (`\\?\`) and device (`\\.\`) paths are returned unchanged: Windows
/// only honors those prefixes with backslash separators.
pub fn normalize_path(path: &Path) -> Cow<'_, str> {
    let text = path.to_string_lossy();
    if cfg!(windows) && text.contains('\\') && !has_literal_prefix(path) {
        Cow::Owned(text.replace('\\', "/"))
    } else {
        text
    }
}

fn has_literal_prefix(path: &Path) -> bool {
    matches!(
        path.components().next(),
        Some(Component::Prefix(prefix))
            if prefix.kind().is_verbatim() || matches!(prefix.kind(), Prefix::DeviceNS(_))
    )
}

/// Normalize a filesystem path to a forward-slash relative string.
pub fn normalize_relative_path<'a>(root: &Path, path: &'a Path) -> Cow<'a, str> {
    normalize_path(path.strip_prefix(root).unwrap_or(path))
}

/// Return whether a path contains the exact component name.
pub fn contains_component(path: &Path, target: &str) -> bool {
    path.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(|value| value == target)
    })
}

/// Return whether user-facing discovery should skip a relative path.
pub fn should_skip_path(path: &Path, mentions_node_modules: bool) -> bool {
    if contains_component(path, ".git") {
        return true;
    }
    if !mentions_node_modules && contains_component(path, "node_modules") {
        return true;
    }
    false
}

fn file_type_from_std(file_type: std::fs::FileType) -> Option<FileType> {
    if file_type.is_symlink() {
        Some(FileType::Symlink)
    } else if file_type.is_dir() {
        Some(FileType::Dir)
    } else if file_type.is_file() {
        Some(FileType::File)
    } else {
        None
    }
}

fn mtime_ms(metadata: &std::fs::Metadata) -> Option<f64> {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis() as f64)
}

/// Classify an existing filesystem path, skipping unsupported special files.
pub fn classify_file_type(path: &Path) -> Option<(FileType, Option<f64>, Option<u64>)> {
    let metadata = std::fs::symlink_metadata(path).ok()?;
    let file_type = file_type_from_std(metadata.file_type())?;
    let size = if file_type == FileType::File {
        Some(metadata.len())
    } else {
        None
    };
    Some((file_type, mtime_ms(&metadata), size))
}

/// Resolve a search path string to a canonical directory path.
pub fn resolve_search_path(path: &str) -> Result<PathBuf, WalkError<String>> {
    let candidate = PathBuf::from(path);
    let root = if candidate.is_absolute() {
        candidate
    } else {
        let cwd = std::env::current_dir().map_err(|err| WalkError::InvalidData {
            path: PathBuf::from(path),
            message: format!("Failed to resolve cwd: {err}"),
        })?;
        cwd.join(candidate)
    };
    let metadata = std::fs::metadata(&root).map_err(|err| WalkError::InvalidData {
        path: root.clone(),
        message: format!("Path not found: {err}"),
    })?;
    if !metadata.is_dir() {
        return Err(WalkError::InvalidData {
            path: root,
            message: "Search path must be a directory".to_string(),
        });
    }
    Ok(std::fs::canonicalize(&root).unwrap_or(root))
}

fn collect_entries_uncached<H, E>(
    fs: &BlockingFs,
    root: &Path,
    mut options: WalkOptions,
    heartbeat: &H,
) -> Result<CollectedEntries, WalkError<String>>
where
    H: Fn() -> std::result::Result<(), E> + Sync,
    E: fmt::Display,
{
    options.cache = false;
    let scan = crate::scan_entries(fs, root, options, || {
        heartbeat().map_err(|err| err.to_string())
    })?;
    Ok(scan)
}

/// Entries a scan produced, shared with the scan cache rather than copied out
/// of it.
pub(crate) struct SharedEntries {
    pub(crate) entries: Arc<Vec<CollectedEntry>>,
    /// Age of the cache entry in milliseconds; zero means freshly scanned.
    pub(crate) cache_age_ms: u64,
}

impl From<CollectedEntries> for SharedEntries {
    fn from(scan: CollectedEntries) -> Self {
        Self {
            entries: Arc::new(scan.entries),
            cache_age_ms: scan.cache_age_ms,
        }
    }
}

fn get_or_scan<H, E>(
    fs: &BlockingFs,
    root: &Path,
    options: WalkOptions,
    heartbeat: &H,
) -> Result<SharedEntries, WalkError<String>>
where
    H: Fn() -> std::result::Result<(), E> + Sync,
    E: fmt::Display,
{
    if *CACHE_TTL_MS == 0 || *MAX_CACHE_ENTRIES == 0 || *MAX_CACHE_BYTES == 0 {
        return collect_entries_uncached(fs, root, options, heartbeat).map(SharedEntries::from);
    }

    heartbeat().map_err(|err| WalkError::Interrupted(err.to_string()))?;
    let key = cache_key(root, options);
    let now = Instant::now();
    let (cached, generation) = {
        let mut cache = SCAN_CACHE.lock();
        (cache.get(&key, now), cache.generation)
    };
    if let Some(entry) = cached {
        return Ok(SharedEntries {
            entries: entry.entries,
            cache_age_ms: now.saturating_duration_since(entry.created_at).as_millis() as u64,
        });
    }

    let scan = collect_entries_uncached(fs, root, options, heartbeat)?;
    let bytes = entry_bytes(&scan.entries, scan.entries.capacity());
    if bytes > *MAX_CACHE_BYTES {
        return Ok(scan.into());
    }
    let entries = Arc::new(scan.entries);
    SCAN_CACHE.lock().insert(
        key,
        CacheEntry {
            created_at: now,
            entries: Arc::clone(&entries),
            bytes,
        },
        generation,
        Instant::now(),
    );
    Ok(SharedEntries {
        entries,
        cache_age_ms: 0,
    })
}

/// Return whether a scan of `root` through `fs` may use the shared cache.
///
/// Cached results are keyed by host path and options alone, yet a scan also
/// reads ancestor repository markers, ancestor and global ignore files, and
/// symlink targets outside `root`. Only a filesystem without a provider sees
/// all of those exactly as the host does, so any provider disables sharing
/// even when it serves `root` natively.
pub(crate) fn shares_scan_cache(fs: &BlockingFs, root: &Path) -> bool {
    fs.is_native() && fs.is_native_local(root)
}

/// Collect entries from `fs`, consulting the shared scan cache only when
/// `options.cache` is set and [`shares_scan_cache`] allows it.
pub fn collect_entries_in<H, E>(
    fs: &BlockingFs,
    root: &Path,
    options: WalkOptions,
    heartbeat: H,
) -> Result<CollectedEntries, WalkError<String>>
where
    H: Fn() -> std::result::Result<(), E> + Sync,
    E: fmt::Display,
{
    let shared = collect_shared_entries_in(fs, root, options, &heartbeat)?;
    let entries = Arc::try_unwrap(shared.entries).or_else(|cached| {
        // Copying a large cached snapshot takes a while; honor a cancellation
        // that arrived meanwhile.
        let entries = cached.as_ref().clone();
        heartbeat().map_err(|err| WalkError::Interrupted(err.to_string()))?;
        Ok(entries)
    })?;
    Ok(CollectedEntries {
        entries,
        cache_age_ms: shared.cache_age_ms,
    })
}

/// [`collect_entries_in`] without copying a cached scan out of the cache.
pub(crate) fn collect_shared_entries_in<H, E>(
    fs: &BlockingFs,
    root: &Path,
    options: WalkOptions,
    heartbeat: &H,
) -> Result<SharedEntries, WalkError<String>>
where
    H: Fn() -> std::result::Result<(), E> + Sync,
    E: fmt::Display,
{
    if options.cache && shares_scan_cache(fs, root) {
        get_or_scan(fs, root, options, heartbeat)
    } else {
        collect_entries_uncached(fs, root, options, heartbeat).map(SharedEntries::from)
    }
}

/// Invalidate cache entries whose root contains `target`.
pub fn invalidate_path(target: &Path) {
    SCAN_CACHE.lock().invalidate(Some(target), Instant::now());
}

/// Resolve a possibly relative path and invalidate matching cache roots.
pub fn invalidate_path_string(path: &str) {
    let candidate = PathBuf::from(path);
    let absolute = if candidate.is_absolute() {
        candidate
    } else if let Ok(cwd) = std::env::current_dir() {
        cwd.join(candidate)
    } else {
        PathBuf::from(path)
    };
    let target = std::fs::canonicalize(&absolute)
        .or_else(|_| {
            absolute
                .parent()
                .and_then(|parent| std::fs::canonicalize(parent).ok())
                .and_then(|parent| absolute.file_name().map(|name| parent.join(name)))
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
        })
        .unwrap_or(absolute);
    invalidate_path(&target);
}

/// Clear the entire scan cache.
pub fn invalidate_all() {
    SCAN_CACHE.lock().invalidate(None, Instant::now());
}
