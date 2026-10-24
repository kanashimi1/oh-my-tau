//! Request-body parity with TS `streamSimple`: for every oracle case the Rust transport must
//! resolve the same wire api and build the same body. Cases come from the transport section of
//! `scripts/compat-oracle/bodies.ts` (`bodies-transport.jsonl`).

use serde_json::{json, Value};

use crate::compat::materialize;
use crate::transport::{plan_simple_request, TransportEnv};
use crate::types::{Context, Model, ModelThinkingLevel, SimpleStreamOptions, SimpleToolChoice};

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
/// The TS oracle stubs `crypto.randomUUID` to this value.
const FIXED_ID: &str = "00000000-0000-4000-8000-000000000000";

fn options_of(case: &Value) -> SimpleStreamOptions {
    let o = &case["options"];
    let mut opts = SimpleStreamOptions::default();
    opts.base.api_key = Some(case["apiKey"].as_str().unwrap_or("test-key").to_string());
    opts.base.max_tokens = o.get("maxTokens").and_then(Value::as_u64);
    opts.base.temperature = o.get("temperature").and_then(Value::as_f64);
    opts.base.session_id = o
        .get("sessionId")
        .and_then(Value::as_str)
        .map(str::to_string);
    opts.reasoning = o
        .get("reasoning")
        .and_then(Value::as_str)
        .and_then(crate::models::thinking_level_from_str);
    if o.get("disableReasoning") == Some(&json!(true)) {
        opts.disable_reasoning = true;
    }
    if let Some(name) = o
        .pointer("/toolChoice/function/name")
        .and_then(Value::as_str)
    {
        opts.tool_choice = Some(SimpleToolChoice::Function {
            name: name.to_string(),
        });
    }
    let _ = ModelThinkingLevel::Off;
    opts
}

fn env_of(case: &Value) -> TransportEnv {
    TransportEnv {
        openrouter_responses: case["env"]
            .get("PI_OPENROUTER_RESPONSES")
            .and_then(Value::as_str)
            != Some("0"),
        azure_deployment_name_map: case["env"]
            .get("AZURE_OPENAI_DEPLOYMENT_NAME_MAP")
            .and_then(Value::as_str)
            .map(str::to_string),
    }
}

/// Mask UUID-shaped strings (the facade's per-inference session id) in `value`.
fn mask_uuids(value: &mut Value, re: &regex::Regex) {
    match value {
        Value::String(s) => {
            if re.is_match(s) {
                *s = re.replace_all(s, "<uuid>").into_owned();
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|v| mask_uuids(v, re)),
        Value::Object(o) => o.values_mut().for_each(|v| mask_uuids(v, re)),
        _ => {}
    }
}

fn diff(got: &Value, want: &Value) -> String {
    let (Some(g), Some(w)) = (got.as_object(), want.as_object()) else {
        return format!("rust={got} ts={want}");
    };
    let keys: std::collections::BTreeSet<_> = g.keys().chain(w.keys()).collect();
    keys.into_iter()
        .filter(|k| g.get(*k) != w.get(*k))
        .map(|k| {
            format!(
                "{k}: rust={} ts={}",
                g.get(k).map_or("<absent>".into(), ToString::to_string),
                w.get(k).map_or("<absent>".into(), ToString::to_string)
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

#[test]
fn rust_transport_matches_ts_stream_simple_for_every_case() {
    let Some(bodies) = oracle_file("bodies-transport.jsonl") else {
        eprintln!(
            "skipping rust_transport_matches_ts_stream_simple_for_every_case: \
             bodies-transport.jsonl not found (generate with bun scripts/compat-oracle/bodies.ts, \
             see scripts/compat-oracle/README.md)"
        );
        return;
    };
    let text = std::fs::read_to_string(&bodies).unwrap_or_else(|e| {
        panic!(
            "{}: {e} — regenerate with `bun scripts/compat-oracle/bodies.ts`",
            bodies.display()
        )
    });
    let filter = std::env::var("TRANSPORT_FILTER").ok();
    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut apis = std::collections::BTreeMap::<String, usize>::new();
    for line in text.lines().filter(|l| !l.is_empty()) {
        let case: Value = serde_json::from_str(line).expect("case parses");
        if case.get("error").is_some_and(|e| !e.is_null()) {
            continue;
        }
        let name = case["case"].as_str().unwrap_or("?").to_string();
        if filter.as_ref().is_some_and(|f| !name.contains(f.as_str())) {
            continue;
        }
        total += 1;
        let mut model: Model =
            serde_json::from_value(case["spec"].clone()).expect("spec parses as a Model");
        materialize(&mut model, None).expect("policy resolves");
        let context: Context =
            serde_json::from_value(case["context"].clone()).expect("context parses");
        let want = case["body"].clone();
        let want = &want;
        let want_api = case["payloadApi"].as_str().unwrap_or("?");
        let planned = plan_simple_request(
            &model,
            &context,
            Some(&options_of(&case)),
            &env_of(&case),
            &|| FIXED_ID.to_string(),
        );
        *apis.entry(want_api.to_string()).or_default() += 1;
        match planned {
            Err(e) => failures.push(format!("{name}: rust error: {e}")),
            Ok(mut p) => {
                let mut want = want.clone();
                if case["options"].get("sessionId").is_none() {
                    let re = regex::Regex::new(
                        "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    )
                    .expect("regex");
                    mask_uuids(&mut p.body, &re);
                    mask_uuids(&mut want, &re);
                }
                if p.api != want_api {
                    failures.push(format!("{name}: api rust={} ts={want_api}", p.api));
                } else if p.body != want {
                    failures.push(format!("{name}: {}", diff(&p.body, &want)));
                }
            }
        }
    }
    eprintln!(
        "transport parity: {total} cases {apis:?}, {} mismatches",
        failures.len()
    );
    let mut by_kind = std::collections::BTreeMap::<String, usize>::new();
    for f in &failures {
        let key = f
            .split(':')
            .next()
            .unwrap_or("")
            .rsplitn(3, '/')
            .nth(2)
            .unwrap_or("?")
            .to_string();
        *by_kind.entry(key).or_default() += 1;
    }
    eprintln!("  by model: {by_kind:?}");
    for f in failures
        .iter()
        .filter(|f| !f.contains("no planned transport"))
        .take(30)
    {
        eprintln!("  MISMATCH {}", &f[..f.len().min(600)]);
    }
    assert!(
        failures.is_empty(),
        "{} of {total} transport requests differ from TS",
        failures.len()
    );
}
