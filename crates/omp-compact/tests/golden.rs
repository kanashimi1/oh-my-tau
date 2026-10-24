//! Golden tests: expectations in `fixtures/compacted-session.expected.json` were produced by
//! running omp's own TypeScript (`buildSessionContext` + `convertToLlm` + `prepareCompaction`,
//! pi-coding-agent 18.6.1) over the redacted real session `fixtures/compacted-session.jsonl`.

use omp_compact::context::{branch_path, Leaf};
use omp_compact::settings::CompactionSettings;
use omp_compact::{build_session_context, convert_to_llm, plan_compaction, ByteEstimate};
use serde_json::Value;

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

fn entries() -> Vec<Value> {
    fixture("compacted-session.jsonl")
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .filter(|e| !matches!(e["type"].as_str(), Some("session" | "title")))
        .collect()
}

fn expected() -> Value {
    serde_json::from_str(&fixture("compacted-session.expected.json")).unwrap()
}

#[test]
fn rebuilt_context_matches_ts() {
    let ctx = build_session_context(&entries(), Leaf::Last).unwrap();
    let exp = expected();
    assert_eq!(Value::Array(ctx.messages.clone()), exp["agent"]);
    assert_eq!(Value::Array(convert_to_llm(&ctx.messages)), exp["llm"]);
}

#[test]
fn summary_leads_and_pre_cut_history_is_gone() {
    let ctx = build_session_context(&entries(), Leaf::Last).unwrap();
    assert_eq!(ctx.messages[0]["role"], "compactionSummary");
    let llm = convert_to_llm(&ctx.messages);
    let text = llm[0]["content"][0]["text"].as_str().unwrap();
    assert!(text.starts_with("Prior model work/tool state available.\nMUST build on prior work; NEVER duplicate prior work.\n\n<summary>\n"));
    // the original request (entry 58f45d66) and the first assistant turn precede the cut
    assert!(!ctx.messages.iter().any(|m| m["role"] == "user"));
}

#[test]
fn cut_points_match_ts() {
    let all = entries();
    let ci = all.iter().position(|e| e["type"] == "compaction").unwrap();
    let before = &all[..ci];
    for case in expected()["plans"].as_array().unwrap() {
        let keep = case["keep"].as_u64().unwrap();
        let settings = CompactionSettings {
            keep_recent_tokens: keep,
            ..CompactionSettings::default()
        };
        let before_refs = before.iter().collect::<Vec<_>>();
        let plan = plan_compaction(&before_refs, &settings, &ByteEstimate).unwrap();
        let exp = &case["plan"];
        let plan = plan.unwrap_or_else(|| panic!("keep={keep}: expected a plan"));
        assert_eq!(
            plan.first_kept_entry_id,
            exp["firstKeptEntryId"].as_str().unwrap(),
            "keep={keep}"
        );
        assert_eq!(
            plan.is_split_turn,
            exp["isSplitTurn"].as_bool().unwrap(),
            "keep={keep}"
        );
        assert_eq!(
            plan.tokens_before,
            exp["tokensBefore"].as_u64().unwrap(),
            "keep={keep}"
        );
        assert_eq!(
            plan.messages_to_summarize.len() as u64,
            exp["summarize"].as_u64().unwrap(),
            "keep={keep}"
        );
        assert_eq!(
            plan.turn_prefix_messages.len() as u64,
            exp["prefix"].as_u64().unwrap(),
            "keep={keep}"
        );
        assert_eq!(
            plan.recent_messages.len() as u64,
            exp["recent"].as_u64().unwrap(),
            "keep={keep}"
        );
        let ids: Vec<&str> = exp["recentEntryIds"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(plan.recent_entry_ids, ids, "keep={keep}");
        let sorted = |s: &std::collections::BTreeSet<String>| s.iter().cloned().collect::<Vec<_>>();
        let list = |k: &str| {
            exp[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            sorted(&plan.file_ops.read),
            list("readFiles"),
            "keep={keep}"
        );
        assert_eq!(
            sorted(&plan.file_ops.written),
            list("written"),
            "keep={keep}"
        );
        assert_eq!(sorted(&plan.file_ops.edited), list("edited"), "keep={keep}");
    }
}

#[test]
fn branch_path_follows_parents() {
    let all = entries();
    let path = branch_path(&all, Leaf::Last);
    assert_eq!(path.len(), all.len());
}
