//! Every TS golden response body must deserialize into its Rust type and
//! serialize back to the same JSON (key set, nulls vs absent, numbers by
//! value). Goldens: `OMP_STATS_GOLDEN` or runs/omp-rust/webui-port/golden;
//! skipped when absent.
#![cfg(feature = "types")]

use std::path::PathBuf;

use omp_stats::types::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

fn golden_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("OMP_STATS_GOLDEN").map_or_else(
        || PathBuf::from("/home/user/omrs-fixtures/stats/golden"),
        PathBuf::from,
    );
    dir.join("_index.json").is_file().then_some(dir)
}

fn same(a: &Value, b: &Value, path: &str, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (
                x.as_f64().unwrap_or(f64::NAN),
                y.as_f64().unwrap_or(f64::NAN),
            );
            if (x - y).abs() > 1e-9 * x.abs().max(y.abs()).max(1.0) {
                out.push(format!("{path}: {x} != {y}"));
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().filter(|k| !y.contains_key(*k)) {
                out.push(format!("{path}.{k}: dropped by the Rust type"));
            }
            for k in y.keys().filter(|k| !x.contains_key(*k)) {
                out.push(format!("{path}.{k}: added by the Rust type ({})", y[k]));
            }
            for (k, v) in x {
                if let Some(w) = y.get(k) {
                    same(v, w, &format!("{path}.{k}"), out);
                }
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                out.push(format!("{path}: len {} != {}", x.len(), y.len()));
            }
            for (i, (v, w)) in x.iter().zip(y).enumerate() {
                same(v, w, &format!("{path}[{i}]"), out);
            }
        }
        _ if a == b => {}
        _ => out.push(format!("{path}: {a} != {b}")),
    }
}

fn check<T: DeserializeOwned + Serialize>(prefix: &str, failures: &mut Vec<String>) -> usize {
    let Some(dir) = golden_dir() else { return 0 };
    let index: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("_index.json")).unwrap()).unwrap();
    let mut n = 0;
    for (req, meta) in index.as_object().unwrap() {
        let path = req.split('?').next().unwrap();
        if path != prefix || meta["status"] != 200 {
            continue;
        }
        let g: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join(meta["file"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let body = &g["body"];
        n += 1;
        match serde_json::from_value::<T>(body.clone()) {
            Err(e) => failures.push(format!("{req}: deserialize: {e}")),
            Ok(v) => {
                let back = serde_json::to_value(&v).unwrap();
                let mut out = Vec::new();
                same(body, &back, "$", &mut out);
                failures.extend(out.into_iter().take(8).map(|l| format!("{req}: {l}")));
            }
        }
    }
    n
}

#[test]
fn golden_bodies_round_trip_through_rust_types() {
    if golden_dir().is_none() {
        eprintln!("goldens absent; skipped");
        return;
    }
    let mut f = Vec::new();
    let mut n = 0;
    n += check::<DashboardStats>("/api/stats", &mut f);
    n += check::<ToolDashboardStats>("/api/stats/tools", &mut f);
    n += check::<ProviderDashboardStats>("/api/stats/providers", &mut f);
    n += check::<FrustrationDashboardStats>("/api/stats/frustration", &mut f);
    n += check::<GainDashboardStats>("/api/stats/gain", &mut f);
    n += check::<Vec<FolderStats>>("/api/stats/folders", &mut f);
    n += check::<Vec<ModelStats>>("/api/stats/models", &mut f);
    n += check::<Vec<TimeSeriesPoint>>("/api/stats/timeseries", &mut f);
    n += check::<Vec<MessageStats>>("/api/stats/recent", &mut f);
    n += check::<Vec<MessageStats>>("/api/stats/errors", &mut f);
    n += check::<Vec<SessionSummary>>("/api/sessions", &mut f);
    n += check::<SessionTrace>("/api/session/trace", &mut f);
    n += check::<RequestDetails>("/api/request/1", &mut f);
    n += check::<RequestDetails>("/api/request/2", &mut f);
    assert!(n > 40, "only {n} goldens matched a type");
    assert!(f.is_empty(), "{} mismatches:\n{}", f.len(), f.join("\n"));
}
