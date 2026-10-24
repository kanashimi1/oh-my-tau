//! Round-trip every line of real omp sessions through `FileEntry` and report how each survives.
//! Usage: cargo run --release -p omp-session --example roundtrip -- <dir-or-file>... [--show N]
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use omp_session::{diff_paths, load, reserialize, round_trip, unknown_reason, RoundTrip};

fn files(p: &Path, out: &mut Vec<PathBuf>) {
    if p.is_dir() {
        for e in std::fs::read_dir(p).into_iter().flatten().flatten() {
            files(&e.path(), out);
        }
    } else if p.extension().is_some_and(|e| e == "jsonl") {
        out.push(p.to_path_buf());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let show: usize = args
        .iter()
        .position(|a| a == "--show")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let mut paths = Vec::new();
    for a in args
        .iter()
        .filter(|a| !a.starts_with("--") && a.parse::<usize>().is_err())
    {
        files(Path::new(a), &mut paths);
    }
    type Counts = (
        BTreeMap<String, u64>,
        BTreeMap<(String, String), u64>,
        BTreeMap<String, u64>,
        BTreeMap<String, u64>,
    );
    let (mut counts, mut by_kind, mut reasons, mut changed): Counts = Default::default();
    let mut examples: Vec<String> = Vec::new();
    let (mut bad_files, mut bytes, mut malformed_lines, mut malformed_files, mut bad_header) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    for p in &paths {
        bytes += std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        let lines = match load(p) {
            Ok(l) => {
                malformed_lines += l.malformed.len() as u64;
                malformed_files += u64::from(!l.malformed.is_empty());
                bad_header += u64::from(l.invalid_header);
                l.lines
            }
            Err(e) => {
                bad_files += 1;
                if examples.len() < show {
                    examples.push(format!("READ FAIL {e:#}"));
                }
                continue;
            }
        };
        for l in &lines {
            let kind = serde_json::from_str::<serde_json::Value>(&l.raw)
                .ok()
                .and_then(|v| v.get("type").and_then(|t| t.as_str()).map(str::to_owned))
                .unwrap_or_default();
            let rt = round_trip(l);
            *counts.entry(format!("{rt:?}")).or_default() += 1;
            *by_kind
                .entry((kind.clone(), format!("{rt:?}")))
                .or_default() += 1;
            match rt {
                RoundTrip::Unknown => {
                    if let Some(r) = unknown_reason(&l.raw) {
                        let key: String = r.chars().take(140).collect();
                        *reasons.entry(key).or_default() += 1;
                    }
                }
                RoundTrip::Changed => {
                    let a: serde_json::Value = serde_json::from_str(&l.raw).unwrap_or_default();
                    let b: serde_json::Value = reserialize(l)
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_default();
                    for d in diff_paths(&a, &b, 3) {
                        let key: String = d.chars().take(120).collect();
                        *changed.entry(format!("{kind}: {key}")).or_default() += 1;
                    }
                }
                _ => {}
            }
        }
    }
    println!("files {} ({:.1} GB), unreadable {bad_files}, files with malformed lines {malformed_files} ({malformed_lines} lines skipped), invalid header {bad_header}", paths.len(), bytes as f64 / 1e9);
    println!("lines by outcome: {counts:?}");
    println!("\nby entry type:");
    for ((k, o), n) in &by_kind {
        println!("  {k:24} {o:10} {n}");
    }
    let top = |m: &BTreeMap<String, u64>, title: &str| {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        println!("\n{title}:");
        for (k, n) in v.into_iter().take(15) {
            println!("  {n:8}  {k}");
        }
    };
    top(
        &reasons,
        "Unknown reasons (modeled type that failed typed parse)",
    );
    top(&changed, "Changed paths (typed parse altered the value)");
    for e in examples {
        println!("{e}");
    }
}
