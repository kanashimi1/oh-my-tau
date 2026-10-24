//! Oracle test: the Rust resolver must reproduce TS `resolveModelPolicy` for every (provider,
//! model) in a copy of `~/.omp/agent/models.db` plus the bundled generated catalog. The dump is
//! produced by `scripts/compat-oracle/oracle.ts` (re-run it to refresh).

use serde_json::Value;

use super::resolve::{resolve_model_policy, ModelSpec};

/// Path of a compat-oracle fixture: `$OMP_COMPAT_ORACLE_DIR` if set, else `<repo
/// root>/target/compat-oracle/` (where `scripts/compat-oracle/*.ts` writes), else the legacy
/// `<repo root>/runs/omp-rust/compat-oracle/`. `None` when the file exists in none of them.
fn oracle_file(name: &str) -> Option<std::path::PathBuf> {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("pa-ai lives at <repo>/vendor/pa-ai");
    std::env::var_os("OMP_COMPAT_ORACLE_DIR")
        .map(std::path::PathBuf::from)
        .into_iter()
        .chain([repo.join("target/compat-oracle"), repo.join("runs/omp-rust/compat-oracle")])
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
}

fn spec_of(v: &Value) -> ModelSpec {
    let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
    ModelSpec {
        provider: s("provider").expect("provider"),
        id: s("id").expect("id"),
        api: s("api").expect("api"),
        base_url: s("baseUrl").unwrap_or_default(),
        provider_type: s("providerType"),
        reasoning: v.get("reasoning").and_then(Value::as_bool).unwrap_or(false),
        thinking: v.get("thinking").cloned(),
        compat: v.get("compat").and_then(Value::as_object).cloned(),
        lenient: false,
    }
}

/// Keys whose values differ between two JSON objects (one level), for readable mismatch reports.
fn diff(a: &Value, b: &Value) -> String {
    match (a.as_object(), b.as_object()) {
        (Some(a), Some(b)) => {
            let mut out = Vec::new();
            let keys: std::collections::BTreeSet<_> = a.keys().chain(b.keys()).collect();
            for k in keys {
                if a.get(k) != b.get(k) {
                    out.push(format!("{k}: rust={:?} ts={:?}", a.get(k), b.get(k)));
                }
            }
            out.join("; ")
        }
        _ => format!("rust={a:?} ts={b:?}"),
    }
}

#[test]
fn resolver_matches_ts_for_every_catalog_and_cache_row() {
    let Some(dump) = oracle_file("dump.jsonl") else {
        eprintln!(
            "skipping resolver_matches_ts_for_every_catalog_and_cache_row: dump.jsonl not found \
             (generate with bun scripts/compat-oracle/oracle.ts, see scripts/compat-oracle/README.md)"
        );
        return;
    };
    let text = std::fs::read_to_string(&dump).unwrap_or_else(|e| {
        panic!(
            "{}: {e} — regenerate with `bun scripts/compat-oracle/oracle.ts`",
            dump.display()
        )
    });
    let (mut total, mut mismatches) = (0usize, Vec::<String>::new());
    for line in text.lines().filter(|l| !l.is_empty()) {
        let case: Value = serde_json::from_str(line).expect("dump line parses");
        let spec = spec_of(&case["spec"]);
        let label = format!(
            "{}:{} [{}/{}]",
            case["source"].as_str().unwrap_or("?"),
            case["variant"],
            spec.provider,
            spec.id
        );
        total += 1;
        let rust = resolve_model_policy(&spec, None);
        match (case.get("error").and_then(Value::as_str), rust) {
            (Some(_), Err(_)) => {}
            (Some(e), Ok(_)) => {
                mismatches.push(format!("{label}: TS threw ({e}) but Rust resolved"))
            }
            (None, Err(e)) => mismatches.push(format!("{label}: Rust error {e}")),
            (None, Ok(r)) => {
                let want = &case["result"];
                let got = r.to_json();
                // TS JSON drops `undefined`; absent optional sections are absent on both sides.
                let mut want = want.clone();
                if let Some(o) = want.as_object_mut() {
                    for k in ["compat", "thinking"] {
                        if o.get(k) == Some(&Value::Null) {
                            o.remove(k);
                        }
                    }
                }
                if got != want {
                    mismatches.push(format!("{label}: {}", diff(&got, &want)));
                }
            }
        }
    }
    eprintln!(
        "compat oracle: {total} cases, {} mismatches",
        mismatches.len()
    );
    for m in mismatches.iter().take(40) {
        eprintln!("  MISMATCH {m}");
    }
    assert!(
        mismatches.is_empty(),
        "{} of {total} cases differ from TS",
        mismatches.len()
    );
}
