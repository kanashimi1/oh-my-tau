//! Small real-corpus measurement: cold vs warm parse over this
//! crate's own sources. Prints numbers with `--nocapture`; asserts
//! only structural facts (item counts, cache reuse), never timings.

use std::time::Instant;

use omp_ladder::{CstCache, CfgContext, SemanticCache};

fn corpus() -> Vec<(String, String)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for sub in ["src", "tests"] {
        let Ok(rd) = std::fs::read_dir(dir.join(sub)) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "rs") {
                if let Ok(text) = std::fs::read_to_string(&p) {
                    files.push((p.display().to_string(), text));
                }
            }
        }
    }
    files.sort();
    files
}

#[test]
fn real_corpus_cold_warm_measurement() {
    let files = corpus();
    assert!(!files.is_empty());
    let total_bytes: usize = files.iter().map(|(_, t)| t.len()).sum();

    let mut csts = CstCache::new(64 << 20);
    let mut sems = SemanticCache::new(1024);
    let ctx = CfgContext::empty();

    // Cold: parse + project every file.
    let t0 = Instant::now();
    let mut total_items = 0usize;
    for (_, text) in &files {
        let cst = csts.get_or_parse(text);
        let idx = sems.get_or_project(&cst, &ctx);
        total_items += idx.items.len();
    }
    let cold = t0.elapsed();
    let cold_stats = csts.stats();
    assert_eq!(cold_stats.hits, 0);
    assert_eq!(usize::try_from(cold_stats.misses).unwrap_or(usize::MAX), files.len());
    assert!(total_items > 0);

    // Warm: identical content, both layers must hit; no parsing.
    let t1 = Instant::now();
    let mut warm_items = 0usize;
    for (_, text) in &files {
        let cst = csts.get_or_parse(text);
        let idx = sems.get_or_project(&cst, &ctx);
        warm_items += idx.items.len();
    }
    let warm = t1.elapsed();
    assert_eq!(warm_items, total_items);
    assert_eq!(usize::try_from(csts.stats().hits).unwrap_or(usize::MAX), files.len());
    assert_eq!(csts.stats().misses, cold_stats.misses);

    let bytes_per_item = total_bytes.checked_div(total_items).unwrap_or(0);
    eprintln!(
        "corpus: {} files, {} bytes, {} items ({} bytes/item)",
        files.len(),
        total_bytes,
        total_items,
        bytes_per_item
    );
    let bytes_per_sec = (total_bytes as u128)
        .saturating_mul(1_000_000)
        .checked_div(cold.as_micros().max(1))
        .unwrap_or(0);
    eprintln!(
        "cold parse+project: {:?} (~{} MB/s); warm (all cache hits): {:?}",
        cold,
        bytes_per_sec / 1_000_000,
        warm
    );
    eprintln!(
        "cache stats cold={:?} after-warm cst={:?} sem={:?}",
        cold_stats,
        csts.stats(),
        sems.stats()
    );
    assert!(bytes_per_item > 0);
}
