#![allow(clippy::float_cmp)]

use std::collections::BTreeSet;

use omp_compact::apply::{compaction_entry, ApplyOptions};
use omp_compact::files::format_file_operations;
use omp_compact::messages::{
    render_branch_summary_context, render_compaction_summary_context,
    render_handoff_summary_context,
};
use omp_compact::prompts;
use omp_compact::settings::{resolve_threshold_tokens, should_compact, CompactionSettings};
use omp_compact::summary::{compact, SummaryKind, SummaryParams, SummaryRequest};
use omp_compact::{plan_compaction, ByteEstimate, CompactionResult};
use pa_types::ai::{AssistantMessage, Message, UserContent, UserContentBlock};
use serde_json::{json, Value};

fn golden(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/tests/fixtures/prompts/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

#[test]
fn static_prompts_match_ts_render() {
    for (name, tpl) in [
        ("compaction-summary", prompts::SUMMARIZATION),
        ("compaction-update-summary", prompts::UPDATE_SUMMARIZATION),
        ("compaction-short-summary", prompts::SHORT_SUMMARY),
        ("compaction-turn-prefix", prompts::TURN_PREFIX),
        ("summarization-system", prompts::SUMMARIZATION_SYSTEM),
        ("branch-summary", prompts::BRANCH_SUMMARY),
        ("branch-summary-preamble", prompts::BRANCH_SUMMARY_PREAMBLE),
    ] {
        assert_eq!(prompts::render_static(tpl), golden(name), "{name}");
    }
}

// Expected strings captured from omp's `prompt.render` (bun) for this probe text.
const PROBE: &str = "A <b> & \"q\" 's' {{x}} `c`\n\n\n\nz  \n";

#[test]
fn context_wrappers_match_ts_render() {
    assert_eq!(
        render_compaction_summary_context(PROBE),
        "Prior model work/tool state available.\nMUST build on prior work; NEVER duplicate prior work.\n\n<summary>\nA <b> & \"q\" 's' {{x}} `c`\nz\n</summary>"
    );
    assert_eq!(
        render_branch_summary_context(PROBE),
        "Branch-return summary:\n\n<summary>\nA <b> & \"q\" 's' {{x}} `c`\nz\n</summary>"
    );
    assert!(render_handoff_summary_context(PROBE)
        .ends_with("<handoff>\nA <b> & \"q\" 's' {{x}} `c`\nz\n</handoff>"));
}

#[test]
fn file_operations_block_matches_ts() {
    let read = vec!["a/b.ts".to_owned(), "a/c.ts".to_owned()];
    let modified = vec!["a/d.ts".to_owned(), "z.md".to_owned()];
    let set: BTreeSet<String> = ["a/d.ts".to_owned()].into();
    assert_eq!(
        format_file_operations(&read, &modified, Some(&set)),
        "<files>\nz.md (Write)\n# a/\nb.ts (Read)\nc.ts (Read)\nd.ts (RW)\n</files>"
    );
}

#[test]
fn threshold_arithmetic() {
    let s = CompactionSettings::default();
    // 200k window, defaulted reserve 16384 < 15% (30000) -> reserve 30000 -> threshold 170000
    assert_eq!(resolve_threshold_tokens(200_000, &s), 170_000.0);
    assert!(should_compact(170_001, 200_000, &s));
    assert!(!should_compact(170_000, 200_000, &s));
    let pct = CompactionSettings {
        threshold_percent: Some(50.0),
        ..s.clone()
    };
    assert_eq!(resolve_threshold_tokens(200_000, &pct), 100_000.0);
    let fixed = CompactionSettings {
        threshold_tokens: Some(5_000_000.0),
        ..s.clone()
    };
    assert_eq!(resolve_threshold_tokens(200_000, &fixed), 199_999.0);
    // small window: defaulted reserve is impossible -> proportional 15%
    assert_eq!(resolve_threshold_tokens(16_000, &s), 13_600.0);
    assert!(!should_compact(
        1_000_000,
        200_000,
        &CompactionSettings {
            enabled: false,
            ..s
        }
    ));
}

fn reply(text: &str) -> AssistantMessage {
    serde_json::from_value(json!({
        "role": "assistant", "content": [{"type": "text", "text": text}], "api": "anthropic-messages", "provider": "anthropic",
        "model": "m", "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
        "stopReason": "stop", "timestamp": 1
    }))
    .unwrap()
}

fn session() -> Vec<Value> {
    let text = |t: &str| json!([{"type":"text","text":t}]);
    vec![
        json!({"type":"message","id":"u1","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z","message":{"role":"user","content":text("first request"),"timestamp":1}}),
        json!({"type":"message","id":"a1","parentId":"u1","timestamp":"2026-01-01T00:00:01.000Z","message":{"role":"assistant","content":[{"type":"toolCall","id":"c1","name":"read","arguments":{"path":"src/a.rs:1-20"}},{"type":"toolCall","id":"c2","name":"edit","arguments":{"path":"src/b.rs"}}],"stopReason":"toolUse","timestamp":2,"usage":{"input":100,"output":10,"cacheRead":0,"cacheWrite":0,"totalTokens":110}}}),
        json!({"type":"message","id":"t1","parentId":"a1","timestamp":"2026-01-01T00:00:02.000Z","message":{"role":"toolResult","toolCallId":"c1","toolName":"read","content":text("file body"),"isError":false,"timestamp":3}}),
        json!({"type":"message","id":"t2","parentId":"t1","timestamp":"2026-01-01T00:00:02.100Z","message":{"role":"toolResult","toolCallId":"c2","toolName":"edit","content":text("ok"),"isError":false,"timestamp":4}}),
        json!({"type":"message","id":"u2","parentId":"t2","timestamp":"2026-01-01T00:00:03.000Z","message":{"role":"user","content":text("second request"),"timestamp":5}}),
        json!({"type":"message","id":"a2","parentId":"u2","timestamp":"2026-01-01T00:00:04.000Z","message":{"role":"assistant","content":text("done"),"stopReason":"stop","timestamp":6}}),
    ]
}

/// Bench finding (s5): a local summary must not re-stamp the previous
/// compaction's provider-native payload onto the new entry. TS writes only
/// the new result's own `preserveData`; carrying `openaiRemoteCompaction`
/// forward made the context rebuild trust the stale payload for the kept
/// tail and replay nothing — the just-finished turn vanished ("no assistant
/// response"). Only the cumulative snapcompact archive slot carries forward.
#[tokio::test]
async fn local_compact_does_not_carry_remote_preserve_data_forward() {
    let mut entries = session();
    // A previous provider-native compaction boundary, followed by a new turn.
    entries.push(json!({
        "type": "compaction", "id": "c9", "parentId": "a2",
        "timestamp": "2026-01-01T00:00:05.000Z",
        "summary": "old summary", "shortSummary": "old",
        "firstKeptEntryId": "u2", "tokensBefore": "110", "tokensAfter": "10",
        "method": "remote",
        "preserveData": {
            "openaiRemoteCompaction": {"version": "v2", "replacementHistory": [{"id": "cmp_x"}]},
            "snapcompact": {"historyBlocks": [{"id": "f1"}], "frameDataBytes": 3}
        }
    }));
    entries.push(json!({"type":"message","id":"u3","parentId":"c9","timestamp":"2026-01-01T00:00:06.000Z","message":{"role":"user","content":[{"type":"text","text":"third request"}],"timestamp":7}}));
    entries.push(json!({"type":"message","id":"a3","parentId":"u3","timestamp":"2026-01-01T00:00:07.000Z","message":{"role":"assistant","content":[{"type":"text","text":"third reply"}],"stopReason":"stop","timestamp":8}}));

    let settings = CompactionSettings {
        keep_recent_tokens: 30,
        ..CompactionSettings::default()
    };
    let prep = plan_compaction(
        &entries.iter().collect::<Vec<_>>(),
        &settings,
        &ByteEstimate,
    )
    .unwrap()
    .unwrap();
    assert!(prep.previous_preserve_data.is_some());

    let completer = |req: SummaryRequest| {
        let text = if req.kind == SummaryKind::CompactionShortSummary {
            "short"
        } else {
            "## Goal\nthe goal"
        };
        async move { Ok(reply(text)) }
    };
    let result = compact(&completer, &prep, &SummaryParams::new(42))
        .await
        .unwrap();
    let preserve = result
        .preserve_data
        .as_ref()
        .expect("snapcompact archive carries forward");
    assert!(
        preserve.get("openaiRemoteCompaction").is_none(),
        "stale remote payload must not be re-stamped: {preserve}"
    );
    assert!(preserve.get("snapcompact").is_some(), "{preserve}");
}

#[tokio::test]
async fn compact_runs_summary_flow_through_injected_completer() {
    let entries = session();
    let settings = CompactionSettings {
        keep_recent_tokens: 30,
        ..CompactionSettings::default()
    };
    let prep = plan_compaction(
        &entries.iter().collect::<Vec<_>>(),
        &settings,
        &ByteEstimate,
    )
    .unwrap()
    .unwrap();
    assert_eq!(prep.first_kept_entry_id, "u2");
    assert!(!prep.is_split_turn);
    assert_eq!(prep.tokens_before, 110);

    let seen = std::sync::Mutex::new(Vec::<SummaryRequest>::new());
    let completer = |req: SummaryRequest| {
        let text = if req.kind == SummaryKind::CompactionShortSummary {
            "I did things."
        } else {
            "## Goal\nthe goal"
        };
        seen.lock().unwrap().push(req);
        async move { Ok(reply(text)) }
    };
    let params = SummaryParams::new(42);
    let result = compact(&completer, &prep, &params).await.unwrap();

    let seen = seen.into_inner().unwrap();
    assert_eq!(
        seen.iter().map(|r| r.kind).collect::<Vec<_>>(),
        [
            SummaryKind::CompactionSummary,
            SummaryKind::CompactionShortSummary
        ]
    );
    assert_eq!(seen[0].max_tokens, 13_107);
    assert_eq!(seen[1].max_tokens, 512);
    let Message::User(user) = &seen[0].context.messages[0] else {
        panic!("user request expected")
    };
    let UserContent::Blocks(blocks) = &user.content else {
        panic!("blocks expected")
    };
    let UserContentBlock::Text(t) = &blocks[0] else {
        panic!("text expected")
    };
    assert!(t.text.starts_with("<conversation>\n\n\nHuman: first request\n\nAssistant: <invoke name=\"read\"><parameter name=\"path\">\"src/a.rs:1-20\"</parameter></invoke>\n<invoke name=\"edit\">"));
    assert!(t
        .text
        .contains("Human: <tool_response>\nfile body\n</tool_response>"));
    assert!(t
        .text
        .ends_with(&prompts::render_static(prompts::SUMMARIZATION)));
    assert_eq!(user.timestamp, 42);

    assert_eq!(
        result.summary,
        "## Goal\nthe goal\n\n<files>\n# src/\na.rs (Read)\nb.rs (Write)\n</files>"
    );
    assert_eq!(result.short_summary, "I did things.");
    assert_eq!(result.first_kept_entry_id, "u2");
    assert_eq!(
        result.details,
        json!({"readFiles": ["src/a.rs"], "modifiedFiles": ["src/b.rs"]})
    );
}

#[test]
fn compaction_entry_serializes_in_omp_order() {
    let result = CompactionResult {
        summary: "S".into(),
        short_summary: String::new(),
        first_kept_entry_id: "u2".into(),
        tokens_before: 110,
        details: json!({"readFiles": [], "modifiedFiles": []}),
        preserve_data: None,
    };
    let opts = ApplyOptions {
        method: Some("soft".into()),
        tokens_after: Some(50),
        from_extension: Some(false),
    };
    let line = compaction_entry(
        &result,
        "d5daf5a9",
        Some("8a644e3e"),
        "2026-01-01T00:00:00.000Z",
        &opts,
    );
    let line = omp_session::OmpEntry::Compaction(Box::new(line))
        .to_line()
        .unwrap();
    assert_eq!(
        line,
        r#"{"type":"compaction","id":"d5daf5a9","parentId":"8a644e3e","timestamp":"2026-01-01T00:00:00.000Z","summary":"S","shortSummary":"","firstKeptEntryId":"u2","tokensBefore":110,"tokensAfter":50,"method":"soft","details":{"readFiles":[],"modifiedFiles":[]},"fromExtension":false}"#
    );
}

#[test]
fn context_after_new_compaction_keeps_tail_then_new_messages() {
    let mut entries = session();
    entries.push(json!({"type":"compaction","id":"k1","parentId":"a2","timestamp":"2026-01-01T00:00:05.000Z","summary":"SUM","firstKeptEntryId":"u2","tokensBefore":110}));
    entries.push(json!({"type":"message","id":"u3","parentId":"k1","timestamp":"2026-01-01T00:00:06.000Z","message":{"role":"user","content":"third","timestamp":7}}));
    let ctx = omp_compact::build_session_context(&entries, omp_compact::Leaf::Last).unwrap();
    let roles: Vec<&str> = ctx
        .messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["compactionSummary", "user", "assistant", "user"]);
    let llm = omp_compact::convert_to_llm(&ctx.messages);
    assert_eq!(llm[0]["historyRewriteAt"], llm[0]["timestamp"]);
    assert_eq!(llm[3]["attribution"], "user");
}

#[test]
fn invalid_native_payload_replays_like_local_compaction() {
    // A malformed `openaiRemoteCompaction` payload yields no provider payload
    // (`getOpenAiRemoteCompactionPayload` returns undefined), so the rebuild
    // falls back to the plain kept-tail layout without a providerPayload.
    let mut entries = session();
    entries.push(json!({"type":"compaction","id":"k1","parentId":"a2","timestamp":"2026-01-01T00:00:05.000Z","summary":"S","firstKeptEntryId":"u2","tokensBefore":1,"preserveData":{"openaiRemoteCompaction":{}}}));
    let ctx = omp_compact::build_session_context(&entries, omp_compact::Leaf::Last).unwrap();
    assert_eq!(ctx.messages[0]["role"], "compactionSummary");
    assert!(ctx.messages[0].get("providerPayload").is_none());
    assert_eq!(ctx.messages.len(), 3);
}

/// The owned resume rebuild must produce the exact context the borrowed
/// `build_session_context` builds — across the local, remote, reset and
/// branch-summary layouts (shims on the resume path change nothing here).
#[test]
fn owned_rebuild_matches_borrowed() {
    for fixture in [
        session(),
        {
            let mut e = session();
            e.push(json!({"type":"compaction","id":"k1","parentId":"a2","timestamp":"2026-01-01T00:00:05.000Z","summary":"S","firstKeptEntryId":"u2","tokensBefore":1,"preserveData":{"openaiRemoteCompaction":{"items":[{"role":"user"}]}},"providerReplayThroughEntryId":"a1"}));
            e
        },
        {
            let mut e = session();
            e.push(json!({"type":"reset_boundary","id":"rb","parentId":"a2","timestamp":"2026-01-01T00:00:04.000Z"}));
            e.push(json!({"type":"message","id":"u3","parentId":"rb","timestamp":"2026-01-01T00:00:05.000Z","message":{"role":"user","content":[{"type":"text","text":"after reset"}]}}));
            e
        },
        {
            let mut e = session();
            e.push(json!({"type":"branch_summary","id":"bs","parentId":"a2","timestamp":"2026-01-01T00:00:04.000Z","summary":"branched","fromId":"a1"}));
            e
        },
        {
            let mut e = session();
            e.push(json!({"type":"custom","id":"n1","parentId":"a2","timestamp":"2026-01-01T00:00:03.500Z","customType":"experimental_context_notes","data":{"version":1,"text":"notes"}}));
            e
        },
    ] {
        let borrowed =
            omp_compact::build_session_context(&fixture, omp_compact::Leaf::Last).unwrap();
        let owned =
            omp_compact::build_session_context_owned(fixture.clone(), omp_compact::Leaf::Last)
                .unwrap();
        assert_eq!(borrowed, owned, "owned rebuild diverged: {fixture:?}");
    }
}

#[test]
fn rollover_compaction_is_still_rejected() {
    let mut entries = session();
    entries.push(json!({"type":"compaction","id":"k1","parentId":"a2","timestamp":"2026-01-01T00:00:05.000Z","summary":"S","firstKeptEntryId":"u2","tokensBefore":1,"details":{"kind":"experimental-context-rollover"}}));
    assert!(omp_compact::build_session_context(&entries, omp_compact::Leaf::Last).is_err());
}
