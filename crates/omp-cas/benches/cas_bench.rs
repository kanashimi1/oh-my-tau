//! Real benchmark: packed size and cold/warm restore/read time vs the
//! uncompressed baseline (loose per-object files, which is what the store
//! itself does without packs).
//!
//! Measures, over a synthetic but shape-realistic corpus (source-like text
//! files, binary blobs, tiny JSON objects):
//!
//! * `loose_bytes` / `pack_bytes` — on-disk size of the same object set as
//!   loose scratch objects vs one transitive pack (zstd-3).
//! * `cold_restore` — importing the pack into an empty store (disk I/O +
//!   verify + write-out).
//! * `warm_read` — re-reading every object from the warm store (page-cache
//!   hits, verified reads).
//! * `baseline_warm_read` — unverified sequential read of the loose files,
//!   the floor any CAS must beat or approach.
//!
//! Run: `cargo bench -p omp-cas` (prints a table; harness = false).
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly


use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use omp_cas::{DepGraph, Digest, Store};

/// Compression level used by the pack codec (`pack.rs`), redeclared here so
/// the benchmark measures the shipped configuration.
const BENCH_ZSTD_LEVEL: i32 = 3;

/// Deterministic pseudo-random bytes (xorshift64*), so numbers are
/// reproducible run to run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn fill(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let v = self.next().to_le_bytes();
            let n = chunk.len();
            chunk.copy_from_slice(&v[..n]);
        }
    }

    fn word(&mut self, len: usize) -> String {
        const ALPHABET: &[u8] = b"abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ_0123456789";
        // Index into the alphabet by a modulo of its true length.
        (0..len)
            .map(|_| {
                let idx = usize::try_from(self.next() % (ALPHABET.len() as u64)).unwrap_or(0);
                ALPHABET[idx] as char
            })
            .collect()
    }
}

/// Shape-realistic corpus: source-like text (high redundancy), binary
/// blobs (incompressible), tiny JSON metadata objects.
struct Corpus {
    objects: BTreeMap<Digest, Vec<u8>>,
    /// digest -> dependency digests (a module imports the objects of the
    /// files it references; leaves have none).
    graph: DepGraph,
    /// Ordered digest list for read loops.
    order: Vec<Digest>,
    raw_total: usize,
}

fn build_corpus(
    text_files: usize,
    binaries: usize,
    jsons: usize,
    text_kib: usize,
    bin_kib: usize,
    seed: u64,
) -> Corpus {
    let mut rng = Rng(seed);
    let mut objects = BTreeMap::new();
    let mut graph = DepGraph::new();
    let mut order = Vec::new();

    for i in 0..text_files {
        // Source-like: lines of identifiers with shared prefixes.
        let mut body = String::new();
        for line in 0..text_kib * 8 {
            body.push_str("fn ");
            body.push_str(&rng.word(12));
            body.push('_');
            body.push_str(&line.to_string());
            body.push('(');
            body.push_str(&rng.word(6));
            body.push_str(") -> ");
            body.push_str(&rng.word(8));
            body.push_str(" { ok(");
            body.push_str(&rng.word(10));
            body.push_str(") }\n");
        }
        let bytes = body.into_bytes();
        let digest = Digest::of(&bytes);
        objects.insert(digest, bytes.clone());
        order.push(digest);
        // Each text object depends on two earlier ones (chained closure);
        // the first two are roots of their chains.
        if i >= 2 {
            let a = order[i - 1];
            let b = order[i - 2];
            graph.declare(digest, [a, b]);
        }
    }
    for _ in 0..binaries {
        let mut bytes = vec![0_u8; bin_kib * 1024];
        rng.fill(&mut bytes);
        let digest = Digest::of(&bytes);
        // Blobs attach to the previous object, keeping the corpus one
        // connected dependency chain (a full-snapshot shape: a single root
        // closes over everything).
        if let Some(prev) = order.last() {
            graph.declare(digest, [*prev]);
        }
        objects.insert(digest, bytes);
        order.push(digest);
    }
    for _ in 0..jsons {
        let bytes = format!(
            r#"{{"kind":"meta","seq":{},"tags":["{}","{}"],"ok":false,"n":null,"score":0.5}}"#,
            rng.next(),
            rng.word(8),
            rng.word(8)
        )
        .into_bytes();
        let digest = Digest::of(&bytes);
        if let Some(prev) = order.last() {
            graph.declare(digest, [*prev]);
        }
        objects.insert(digest, bytes);
        order.push(digest);
    }
    let raw_total = objects.values().map(Vec::len).sum();
    Corpus {
        objects,
        graph,
        order,
        raw_total,
    }
}

fn timed<F: FnOnce()>(f: F) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64()
}

/// Sum file sizes under `root` (objects/prefix/file — the prefix shards
/// recurse; directory inodes are not counted).
fn dir_bytes(root: &std::path::Path) -> u64 {
    fn walk(dir: &std::path::Path, total: &mut u64) {
        for entry in fs::read_dir(dir).expect("read dir").flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, total);
            } else if let Ok(meta) = entry.metadata() {
                *total += meta.len();
            }
        }
    }
    let mut total = 0;
    walk(root, &mut total);
    total
}

#[allow(clippy::cast_precision_loss)] // ratio display on MiB-scale counts, f64 mantissa is ample
fn main() {
    let corpus = build_corpus(120, 40, 200, 16, 256, 0xC0FFE);
    println!(
        "corpus: {} objects, {} files-like text (16KiB ea), {} binaries (256KiB ea), {} tiny JSON; raw {} MiB",
        corpus.order.len(),
        120,
        40,
        200,
        corpus.raw_total / (1024 * 1024)
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(dir.path()).expect("store");

    // ── loose baseline: plain per-object files, no pack ──────────────────
    let loose_start = Instant::now();
    for bytes in corpus.objects.values() {
        store.put_durable(bytes).expect("put");
    }
    let loose_put = loose_start.elapsed().as_secs_f64();
    let loose_bytes = dir_bytes(&dir.path().join("durable"));
    println!("loose: put {loose_put:.3}s, file bytes {loose_bytes}");

    // ── pack build + size ────────────────────────────────────────────────
    // One root (the newest object) closes over the whole corpus: objects
    // depend on their predecessors, so the transitive closure is total.
    let roots = [*corpus.order.last().expect("nonempty corpus")];
    let closure = corpus.graph.transitive(&roots, 10_000).expect("closure");
    assert_eq!(closure.len(), corpus.order.len(), "closure must be total");
    let build_start = Instant::now();
    let (pack_id, path) = store.build_pack_file(&roots, &corpus.graph, 10_000).expect("pack");
    let build_secs = build_start.elapsed().as_secs_f64();
    let pack_bytes = fs::metadata(&path).expect("pack stat").len();
    // Reference point: plain zstd-3 over the raw content, the best a naive
    // "compress everything" pipeline achieves without CAS structure.
    let raw_concat: Vec<u8> = corpus.objects.values().flatten().copied().collect();
    let zstd_only = zstd::stream::encode_all(&raw_concat[..], BENCH_ZSTD_LEVEL)
        .expect("reference compress")
        .len();
    println!(
        "pack: {} objects, {} bytes ({:.2}x vs raw content, {:.2}x vs loose files, {:.2}x vs plain zstd-3), built in {build_secs:.3}s",
        closure.len(),
        pack_bytes,
        corpus.raw_total as f64 / pack_bytes.max(1) as f64,
        loose_bytes as f64 / pack_bytes.max(1) as f64,
        zstd_only as f64 / pack_bytes.max(1) as f64,
    );
    black_box(pack_id);

    // ── cold restore: import the pack into an empty store ────────────────
    let cold_dir = tempfile::tempdir().expect("tempdir");
    let cold_store = Store::open(cold_dir.path()).expect("store");
    let cold = timed(|| {
        cold_store
            .import_pack(&path, omp_cas::Tier::Durable)
            .expect("cold import");
    });
    println!("cold restore: import_pack {cold:.3}s");

    // ── warm read: every object, verified, from the warm store ───────────
    let warm = timed(|| {
        for digest in &corpus.order {
            let bytes = cold_store.get(digest).expect("warm get");
            black_box(bytes.len());
        }
    });
    let baseline_warm = timed(|| {
        for digest in &corpus.order {
            let hex = digest.hex();
            let p = cold_dir
                .path()
                .join(format!("durable/objects/{}/{}", &hex[..2], hex));
            let bytes = fs::read(&p).expect("loose read");
            black_box(bytes.len());
        }
    });
    println!("warm read: verified CAS {warm:.3}s vs unverified loose {baseline_warm:.3}s");
    println!("done");
}
