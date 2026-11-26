//! The store: two-tier object storage, promotion, packs, eviction.
//!
//! Layout under `root`:
//! ```text
//! scratch/objects/<2 hex>/<64 hex>   (ephemeral, LRU-evicted)
//! durable/objects/<2 hex>/<64 hex>   (permanent until explicit remove)
//! packs/<pack object id>.pack        (built packs, themselves durable objects)
//! tmp/                               (staging for atomic renames)
//! ```
//!
//! All writes stage into `tmp/` and rename into place — a concurrent reader
//! never sees a partial object. A write is accepted only after the staged
//! bytes are re-hashed from disk; a mismatch refuses the object instead of
//! persisting wrong content.
//!
//! Lifetime rules (the production slice this crate exists for):
//! * [`Store::put_scratch`] never touches `durable/` and never promotes.
//! * [`Store::get`] consults scratch, then durable; a scratch hit refreshes
//!   recency but changes no tier.
//! * [`Store::promote`] is the only scratch → durable path, and it is
//!   explicit.
//! * Durable objects are never evicted; [`Store::remove`] is the only
//!   deletion.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::cache::{CachePolicy, ScratchClock, ScratchStats};
use crate::depgraph::DepGraph;
use crate::digest::Digest;
use crate::error::{io_err, Error, Result};
use crate::object::{ObjectId, Tier};
use crate::pack;

/// Content-addressed object store with scratch/durable tiers.
#[derive(Debug)]
pub struct Store {
    root: PathBuf,
    clock: ScratchClock,
    policy: parking_lot::Mutex<CachePolicy>,
    /// Count of scratch objects evicted by the byte budget since open.
    evictions: AtomicU64,
    /// Uniqueness component for tmp file names.
    tmp_seq: AtomicU64,
}

/// On-disk policy record (`root/scratch-policy.json`). Persisted so the
/// scratch budget stays in force across reopen — an eviction policy that
/// silently lapses on reopen would let scratch grow unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct PolicyFile {
    max_bytes: Option<u64>,
    max_object_bytes: Option<u64>,
}

impl From<CachePolicy> for PolicyFile {
    fn from(p: CachePolicy) -> Self {
        Self {
            max_bytes: p.max_bytes,
            max_object_bytes: p.max_object_bytes,
        }
    }
}

impl From<PolicyFile> for CachePolicy {
    fn from(f: PolicyFile) -> Self {
        Self {
            max_bytes: f.max_bytes,
            max_object_bytes: f.max_object_bytes,
        }
    }
}

/// Name of the persisted policy record inside the store root.
const POLICY_FILE: &str = "scratch-policy.json";

impl Store {
    /// Open (creating if needed) a store at `root`.
    ///
    /// The scratch tier's LRU accounting is seeded from disk (file names
    /// must parse as digests; sizes come from `stat`), so byte-budget
    /// eviction still works when a session-scoped store is reopened
    /// instead of living in one process. Read verification is unchanged —
    /// seeded sizes are bookkeeping only.
    ///
    /// # Errors
    /// Directory creation failed.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        for dir in ["scratch/objects", "durable/objects", "packs", "tmp"] {
            fs::create_dir_all(root.join(dir)).map_err(|e| io_err(&root.join(dir), e))?;
        }
        let policy = Self::load_policy(&root);
        let store = Self {
            root,
            clock: ScratchClock::default(),
            policy: parking_lot::Mutex::new(policy),
            evictions: AtomicU64::new(0),
            tmp_seq: AtomicU64::new(0),
        };
        store.seed_scratch_from_disk()?;
        Ok(store)
    }

    /// Load a persisted scratch policy; absent/corrupt file falls back to
    /// unbounded (the store never guesses a budget it was not given).
    fn load_policy(root: &Path) -> CachePolicy {
        let path = root.join(POLICY_FILE);
        std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<PolicyFile>(&bytes).ok())
            .map_or(CachePolicy::unbounded(), CachePolicy::from)
    }

    /// Seed scratch LRU accounting from the on-disk scratch tier. Names
    /// that do not parse as digests (foreign files) are ignored, not
    /// deleted — the store never removes what it did not write.
    fn seed_scratch_from_disk(&self) -> Result<()> {
        let objects_root = self.root.join(Tier::Scratch.dir()).join("objects");
        let mut entries: Vec<(Digest, u64)> = Vec::new();
        let prefixes = match fs::read_dir(&objects_root) {
            Ok(read_dir) => read_dir,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(io_err(&objects_root, e)),
        };
        for prefix in prefixes {
            let prefix = prefix.map_err(|e| io_err(&objects_root, e))?;
            let prefix_path = prefix.path();
            let shard = match fs::read_dir(&prefix_path) {
                Ok(shard) => shard,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(io_err(&prefix_path, e)),
            };
            for file in shard {
                let file = file.map_err(|e| io_err(&prefix_path, e))?;
                let name = file.file_name();
                let Some(name) = name.to_str() else {
                    continue;
                };
                let Ok(digest) = Digest::parse(name) else {
                    continue;
                };
                let Ok(meta) = file.metadata() else {
                    continue;
                };
                if meta.is_file() {
                    entries.push((digest, meta.len()));
                }
            }
        }
        self.clock.seed(&entries);
        Ok(())
    }

    /// Store root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Replace the scratch policy (byte budget / per-object cap). The new
    /// policy is persisted so reopen keeps it in force, and the budget is
    /// enforced immediately (existing scratch may be evicted right away).
    ///
    /// # Errors
    /// Persisting the policy failed.
    pub fn set_scratch_policy(&self, policy: CachePolicy) -> Result<()> {
        let file = PolicyFile::from(policy);
        let bytes = serde_json::to_vec(&file)
            .map_err(|_| Error::Format("policy serialize failed"))?;
        let path = self.root.join(POLICY_FILE);
        let tmp = self.tmp_path("policy");
        {
            let f = File::create(&tmp).map_err(|e| io_err(&tmp, e))?;
            let mut writer = BufWriter::new(f);
            writer.write_all(&bytes).map_err(|e| io_err(&tmp, e))?;
            writer.flush().map_err(|e| io_err(&tmp, e))?;
        }
        fs::rename(&tmp, &path).map_err(|e| io_err(&path, e))?;
        *self.policy.lock() = policy;
        self.evict_scratch()
    }

    /// Current scratch policy.
    #[must_use]
    pub fn scratch_policy(&self) -> CachePolicy {
        *self.policy.lock()
    }

    /// Scratch accounting snapshot.
    #[must_use]
    pub fn scratch_stats(&self) -> ScratchStats {
        let (objects, bytes) = self.clock.tracked();
        ScratchStats {
            bytes,
            objects,
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    /// On-disk path of an object in a given tier (whether or not it exists).
    #[must_use]
    pub fn tier_path(&self, tier: Tier, digest: &Digest) -> PathBuf {
        self.root.join(tier.dir()).join(store_rel_path(digest))
    }

    /// Write `bytes` into the scratch tier. Subject to `max_object_bytes`
    /// rejection and byte-budget eviction; recency is stamped on success.
    ///
    /// # Errors
    /// I/O failure, or the object exceeds the configured per-object cap.
    pub fn put_scratch(&self, bytes: &[u8]) -> Result<ObjectId> {
        let policy = *self.policy.lock();
        if let Some(cap) = policy.max_object_bytes {
            if (bytes.len() as u64) > cap {
                return Err(Error::Limit("scratch object exceeds max_object_bytes"));
            }
        }
        let id = ObjectId::of(bytes);
        let inserted = self.write_verified(Tier::Scratch, id, bytes)?;
        if inserted {
            self.clock.record_put(id.size());
        }
        self.clock.stamp_now(&id.digest());
        self.evict_scratch()?;
        Ok(id)
    }

    /// Write `bytes` into the durable tier. Never evicted, never demoted.
    ///
    /// # Errors
    /// I/O failure.
    pub fn put_durable(&self, bytes: &[u8]) -> Result<ObjectId> {
        let id = ObjectId::of(bytes);
        self.write_verified(Tier::Durable, id, bytes)?;
        Ok(id)
    }

    /// The one explicit scratch → durable promotion. Returns `false` when
    /// the digest is not in the scratch tier (durable-only digests are not
    /// promoted; probing deliberately is not an error). Promotion copies,
    /// so scratch recency is unchanged and eviction still governs the
    /// scratch copy.
    ///
    /// # Errors
    /// I/O failure, or the scratch bytes fail verification — corruption is
    /// reported, never silently promoted.
    pub fn promote(&self, digest: &Digest) -> Result<bool> {
        let src = self.tier_path(Tier::Scratch, digest);
        let bytes = match fs::read(&src) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(io_err(&src, e)),
        };
        let id = ObjectId::of(&bytes);
        if id.digest() != *digest {
            return Err(Error::Corrupt {
                digest: digest.hex(),
                reason: "scratch bytes do not hash to digest at promote".into(),
            });
        }
        self.write_verified(Tier::Durable, id, &bytes)?;
        Ok(true)
    }

    /// Fetch verified bytes: scratch first, then durable. A scratch hit
    /// refreshes LRU recency; no tier changes.
    ///
    /// # Errors
    /// [`Error::MissingObject`] when neither tier holds the digest;
    /// [`Error::Corrupt`] when stored bytes fail verification.
    pub fn get(&self, digest: &Digest) -> Result<Vec<u8>> {
        if let Some(bytes) = self.get_verified_if_present(Tier::Scratch, digest)? {
            self.clock.touch_if_present(digest);
            return Ok(bytes);
        }
        if let Some(bytes) = self.get_verified_if_present(Tier::Durable, digest)? {
            return Ok(bytes);
        }
        Err(Error::MissingObject {
            digest: digest.hex(),
        })
    }

    /// Whether the digest is readable from either tier (verified).
    ///
    /// # Errors
    /// I/O failure; verification failure surfaces as [`Error::Corrupt`].
    pub fn contains(&self, digest: &Digest) -> Result<bool> {
        for tier in [Tier::Scratch, Tier::Durable] {
            let path = self.tier_path(tier, digest);
            if !path.is_file() {
                continue;
            }
            return Ok(self.get_verified_if_present(tier, digest)?.is_some());
        }
        Ok(false)
    }

    /// Write a scratch object's verified bytes to `dest` (streaming sink
    /// over [`Store::get`]). Refreshes recency on scratch hits.
    ///
    /// # Errors
    /// [`Store::get`] errors, or the sink write fails.
    pub fn read_into(&self, digest: &Digest, dest: &mut impl std::io::Write) -> Result<()> {
        let bytes = self.get(digest)?;
        let sink = PathBuf::from("<sink>");
        dest.write_all(&bytes).map_err(|e| io_err(&sink, e))?;
        Ok(())
    }

    /// Delete one object from a tier. Returns whether it existed. Scratch
    /// bookkeeping stays consistent.
    ///
    /// # Errors
    /// I/O failure other than not-found.
    pub fn remove(&self, tier: Tier, digest: &Digest) -> Result<bool> {
        let path = self.tier_path(tier, digest);
        if tier == Tier::Scratch {
            // Subtract the real size before dropping the stamp.
            let len = match fs::metadata(&path) {
                Ok(meta) => Some(meta.len()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => return Err(io_err(&path, e)),
            };
            match fs::remove_file(&path) {
                Ok(()) => {
                    if let Some(len) = len {
                        self.clock.record_remove(len);
                    }
                    self.clock.forget(digest);
                    Ok(true)
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    self.clock.forget(digest);
                    Ok(false)
                }
                Err(e) => Err(io_err(&path, e)),
            }
        } else {
            match fs::remove_file(&path) {
                Ok(()) => Ok(true),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
                Err(e) => Err(io_err(&path, e)),
            }
        }
    }

    /// Build a pack over `roots` plus their transitive dependencies
    /// (per `graph`) — every object must be present in either tier — and
    /// write it to `writer`. The pack bytes are also stored as a durable
    /// object; its id and total written bytes are returned.
    ///
    /// Packs are complete by construction: a missing object anywhere in the
    /// closure is [`Error::MissingObject`], never a partial pack.
    ///
    /// # Errors
    /// Missing dependency, graph bound exceeded, sink failure.
    pub fn build_pack(
        &self,
        roots: &[Digest],
        graph: &DepGraph,
        max_nodes: usize,
        writer: &mut impl Write,
    ) -> Result<(ObjectId, u64)> {
        let order = graph.transitive(roots, max_nodes)?;
        let mut objects = BTreeMap::new();
        for digest in &order {
            let bytes = self.get(digest).map_err(|err| match err {
                Error::MissingObject { .. } => Error::MissingObject {
                    digest: digest.hex(),
                },
                other => other,
            })?;
            objects.insert(*digest, bytes);
        }
        let mut pack_bytes = Vec::new();
        let written = pack::build_pack(&objects, &mut pack_bytes)?;
        writer
            .write_all(&pack_bytes)
            .map_err(|e| io_err(&pack::sink_label(), e))?;
        let id = self.put_durable(&pack_bytes)?;
        Ok((id, written))
    }

    /// Build a pack and persist it at `root/packs/<id>.pack` (staged via
    /// `tmp/`, atomic rename). Returns the pack object id and file path.
    ///
    /// # Errors
    /// See [`Store::build_pack`]; plus file write/rename failures.
    pub fn build_pack_file(
        &self,
        roots: &[Digest],
        graph: &DepGraph,
        max_nodes: usize,
    ) -> Result<(ObjectId, PathBuf)> {
        let mut bytes = Vec::new();
        let (id, _) = self.build_pack(roots, graph, max_nodes, &mut bytes)?;
        let path = self.root.join("packs").join(format!("{}.pack", id.hex()));
        let tmp = self.tmp_path("pack");
        {
            let file = File::create(&tmp).map_err(|e| io_err(&tmp, e))?;
            let mut writer = BufWriter::new(file);
            writer.write_all(&bytes).map_err(|e| io_err(&tmp, e))?;
            writer.flush().map_err(|e| io_err(&tmp, e))?;
        }
        fs::rename(&tmp, &path).map_err(|e| io_err(&path, e))?;
        Ok((id, path))
    }

    /// Import a pack file into `tier`, digest-verifying every object as it
    /// is written. Import is bounded by the pack container caps (see the
    /// [`crate::pack`] module docs): fixed-offset parsing, `checked_add`
    /// bounds, decompression-bomb ceiling — no unbounded recursion. Objects
    /// land only at digest-derived names, so a pack cannot direct a write
    /// anywhere else (no path traversal by construction).
    ///
    /// Scratch imports participate in policy accounting/eviction like any
    /// scratch write; durable imports do not.
    ///
    /// # Errors
    /// Format corruption, digest mismatch, I/O failure, policy rejection.
    pub fn import_pack(&self, path: &Path, tier: Tier) -> Result<Vec<ObjectId>> {
        let ids = pack::apply_pack_file(path, |obj| {
            let inserted = self.write_verified(tier, obj.id, &obj.bytes)?;
            if tier == Tier::Scratch {
                if inserted {
                    self.clock.record_put(obj.id.size());
                }
                self.clock.stamp_now(&obj.id.digest());
            }
            Ok(())
        })?;
        if tier == Tier::Scratch {
            self.evict_scratch()?;
        }
        Ok(ids)
    }

    // ── internals ───────────────────────────────────────────────────────

    /// Stage bytes in `tmp/`, verify them from disk, then atomically rename
    /// into the tier. Returns `true` only for a newly inserted object, so
    /// scratch accounting counts an on-disk object once; re-puts still refresh
    /// recency at the caller. Scratch accounting is the caller's responsibility.
    fn write_verified(&self, tier: Tier, id: ObjectId, bytes: &[u8]) -> Result<bool> {
        let dest = self.tier_path(tier, &id.digest());
        if dest.is_file() {
            // Already present: verify the existing bytes anyway so a
            // clobbered file can't masquerade as a successful write.
            let existing = fs::read(&dest).map_err(|e| io_err(&dest, e))?;
            if ObjectId::of(&existing).digest() != id.digest() {
                return Err(Error::Corrupt {
                    digest: id.hex(),
                    reason: "existing file does not match digest".into(),
                });
            }
            return Ok(false);
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| io_err(parent, e))?;
        }
        let tmp = self.tmp_path("obj");
        {
            let file = File::create(&tmp).map_err(|e| io_err(&tmp, e))?;
            let mut writer = BufWriter::new(file);
            writer.write_all(bytes).map_err(|e| io_err(&tmp, e))?;
            writer.flush().map_err(|e| io_err(&tmp, e))?;
        }
        // Verify what landed on disk, not what we handed to the kernel.
        let staged = fs::read(&tmp).map_err(|e| io_err(&tmp, e))?;
        if ObjectId::of(&staged).digest() != id.digest() {
            let _ = fs::remove_file(&tmp);
            return Err(Error::Corrupt {
                digest: id.hex(),
                reason: "staged bytes do not hash to digest".into(),
            });
        }
        fs::rename(&tmp, &dest).map_err(|e| io_err(&dest, e))?;
        Ok(true)
    }

    /// Read and verify an object if present in `tier`; `None` when absent.
    fn get_verified_if_present(&self, tier: Tier, digest: &Digest) -> Result<Option<Vec<u8>>> {
        let path = self.tier_path(tier, digest);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io_err(&path, e)),
        };
        if ObjectId::of(&bytes).digest() != *digest {
            return Err(Error::Corrupt {
                digest: digest.hex(),
                reason: "stored bytes do not hash to digest".into(),
            });
        }
        Ok(Some(bytes))
    }

    /// Enforce the scratch byte budget: evict least-recently-touched
    /// objects until tracked bytes fit. Never touches durable.
    fn evict_scratch(&self) -> Result<()> {
        let budget = self.policy.lock().max_bytes;
        let Some(budget) = budget else {
            return Ok(());
        };
        if self.clock.tracked().1 <= budget {
            return Ok(());
        }
        for (digest, _) in self.clock.lru_order() {
            if self.clock.tracked().1 <= budget {
                break;
            }
            if self.remove(Tier::Scratch, &digest)? {
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    fn tmp_path(&self, label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let seq = self.tmp_seq.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        self.root
            .join("tmp")
            .join(format!("{label}-{pid}-{seq}-{nonce}"))
    }
}

/// Store-relative path for a digest: `objects/<2 hex>/<64 hex>`.
fn store_rel_path(digest: &Digest) -> PathBuf {
    let hex = digest.hex();
    PathBuf::from("objects").join(&hex[..2]).join(hex)
}
