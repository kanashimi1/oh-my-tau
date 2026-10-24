//! Build-time constants the discovery writer shares with TS `catalog/src/model-cache.ts`:
//!
//! ```text
//! materializationPolicy() = `app-${VERSION}:builder-${MODEL_MATERIALIZATION_VERSION}:rules-${RULES.version}-`
//!     + Bun.hash(JSON.stringify(RULES)).toString(36)
//! ```
//!
//! `VERSION` is the oh-my-pi `coding-agent` package version and `RULES` is
//! `catalog/src/compat/rules.json` from the vendored TS checkout. `Bun.hash`
//! is wyhash with seed 0 (`omp_stats::parser::wyhash`), and `toString(36)` on
//! its `bigint` result is the plain base-36 rendering of the unsigned 64-bit
//! value. TS readers reject any `model_cache` row whose
//! `materialization_policy` differs, so these values must stay in lockstep
//! (checked by `scripts/verify-models-db.ts`).

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    // A local oh-my-pi checkout (`vendor-src/`, not in git) wins so a TS bump is picked up;
    // a clean clone builds from the in-repo snapshot (same rules.json pa-ai vendors).
    let ts = manifest.join("../../vendor-src/oh-my-pi/packages");
    let (rules_path, pkg_path) = if ts.join("catalog/src/compat/rules.json").is_file() {
        (ts.join("catalog/src/compat/rules.json"), ts.join("coding-agent/package.json"))
    } else {
        (
            manifest.join("../../vendor/pa-ai/data/compat/rules.json"),
            manifest.join("ts-coding-agent.json"),
        )
    };
    println!("cargo:rerun-if-changed={}", rules_path.display());
    println!("cargo:rerun-if-changed={}", pkg_path.display());

    let rules = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(&rules_path).expect("reading rules.json"),
    )
    .expect("parsing rules.json");
    let rules_version = rules
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .expect("rules.json version");
    // `JSON.stringify(RULES)`: compact, insertion (file) key order.
    let serialized = serde_json::to_string(&rules).expect("serializing rules.json");
    let hash = omp_stats::parser::wyhash::wyhash(serialized.as_bytes());
    let version = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(&pkg_path).expect("reading coding-agent package.json"),
    )
    .expect("parsing package.json")["version"]
        .as_str()
        .expect("package.json version")
        .to_owned();

    println!(
        "cargo:rustc-env=OMP_MODEL_MATERIALIZATION_POLICY=\
         app-{version}:builder-1:rules-{rules_version}-{}",
        base36(hash),
    );
}

/// `bigint.toString(36)` for an unsigned 64-bit value.
fn base36(mut v: u64) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if v == 0 {
        return "0".to_owned();
    }
    let mut out = Vec::new();
    while v > 0 {
        out.push(DIGITS[usize::try_from(v % 36).expect("fits")]);
        v /= 36;
    }
    out.reverse();
    String::from_utf8(out).expect("ascii")
}
