//! `dialects.expected.json` was produced by omp's TypeScript under bun:
//! `preferredDialect(id)` for every model id in the catalog (`models.json`) plus extras, and
//! `serializeConversationForSummary(messages, dialect)` for all ten dialects over three inputs
//! (the converted fixture session, a synthetic set exercising thinking / images / errors / useless
//! results / control tokens / nested arguments, and the empty conversation).

use omp_compact::dialect::{preferred_dialect, serialize_conversation_for_summary, Dialect};
use serde_json::Value;

fn expected() -> Value {
    let raw = std::fs::read_to_string(format!(
        "{}/tests/fixtures/dialects.expected.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    serde_json::from_str(&raw).unwrap()
}

fn parse(name: &str) -> Dialect {
    match name {
        "xml" => Dialect::Xml,
        "anthropic" => Dialect::Anthropic,
        "glm" => Dialect::Glm,
        "gemini" => Dialect::Gemini,
        "gemma" => Dialect::Gemma,
        "kimi" => Dialect::Kimi,
        "qwen3" => Dialect::Qwen3,
        "deepseek" => Dialect::Deepseek,
        "minimax" => Dialect::Minimax,
        "harmony" => Dialect::Harmony,
        other => panic!("unknown dialect {other}"),
    }
}

#[test]
fn preferred_dialect_matches_ts_for_catalog_models() {
    let exp = expected();
    let mut bad = Vec::new();
    for (id, d) in exp["dialects"].as_object().unwrap() {
        let want = parse(d.as_str().unwrap());
        let got = preferred_dialect(id);
        if got != want {
            bad.push(format!("{id}: want {want:?} got {got:?}"));
        }
    }
    assert!(
        bad.is_empty(),
        "{} mismatches, e.g. {:?}",
        bad.len(),
        &bad[..bad.len().min(10)]
    );
}

#[test]
fn serialized_conversation_matches_ts_for_every_dialect() {
    let exp = expected();
    for (dialect, per_input) in exp["outputs"].as_object().unwrap() {
        for (input, want) in per_input.as_object().unwrap() {
            let messages = exp["inputs"][input].as_array().unwrap();
            let got = serialize_conversation_for_summary(messages, parse(dialect));
            assert_eq!(
                got,
                want.as_str().unwrap(),
                "dialect={dialect} input={input}"
            );
        }
    }
}
