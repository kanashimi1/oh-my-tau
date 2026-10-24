//! Handoff lane smoke tests: the document oneshot, the `<files>` upsert, the
//! committed `method: "handoff"` entry, and its context replay. Real sessions
//! contain 62 handoff compactions (the most common non-soft method), so this
//! lane is exercised end-to-end here with an injected completer.

use omp_compact::build_session_context;
use omp_compact::context::Leaf;
use omp_compact::files::FileOperations;
use omp_compact::handoff::{
    build_handoff_request, generate_handoff_from_context, handoff_compaction_entry,
    render_handoff_prompt, HandoffCommit,
};
use omp_compact::summary::{SummaryKind, SummaryRequest};
use pa_types::ai::{AssistantMessage, Context, Message};
use serde_json::json;

fn reply(text: &str) -> AssistantMessage {
    serde_json::from_value(json!({
        "role": "assistant", "content": [{"type": "text", "text": text}], "api": "anthropic-messages", "provider": "anthropic",
        "model": "m", "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
        "stopReason": "stop", "timestamp": 1
    }))
    .unwrap()
}

fn golden(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/tests/fixtures/prompts/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

#[test]
fn render_handoff_prompt_matches_ts_handlebars_render() {
    // Expectations captured from `renderHandoffPrompt()` in pi-agent-core
    // 18.6.1 under bun (handlebars `{{#if additionalFocus}}` semantics).
    assert_eq!(
        render_handoff_prompt(None),
        golden("handoff-document-none.txt")
    );
    assert_eq!(
        render_handoff_prompt(Some("pay attention to X")),
        golden("handoff-document-focus.txt")
    );
    assert_eq!(
        render_handoff_prompt(Some("")),
        golden("handoff-document-none.txt")
    );
}

#[tokio::test]
async fn handoff_document_commits_as_handoff_compaction_and_replays() {
    // 1. The oneshot request carries the caller context + handoff prompt.
    let prompt = render_handoff_prompt(None);
    let mut context: Context = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "handoff please", "timestamp": 1}]
    }))
    .unwrap();
    context.messages.push(
        serde_json::from_value::<Message>(
            json!({"role": "user", "content": prompt.as_str(), "timestamp": 2}),
        )
        .unwrap(),
    );
    let request = build_handoff_request(context, 4096);
    assert_eq!(request.kind, SummaryKind::Handoff);
    assert_eq!(request.max_tokens, 4096);

    // 2. Generate the document through an injected completer.
    let completer = |req: SummaryRequest| {
        assert_eq!(req.kind, SummaryKind::Handoff);
        async move { Ok(reply("## Goal\ndo the thing")) }
    };
    let document = generate_handoff_from_context(&completer, request)
        .await
        .unwrap();
    assert_eq!(document, "## Goal\ndo the thing");

    // 3. Commit payload: `method: "handoff"`, cumulative `<files>` upsert, details lists.
    let mut file_ops = FileOperations::default();
    file_ops.read.insert("src/lib.rs".into());
    file_ops.read.insert("src/main.rs".into());
    file_ops.written.insert("src/main.rs".into());
    let commit = HandoffCommit {
        first_kept_entry_id: "a2",
        tokens_before: 500,
        tokens_after: Some(80),
        file_ops: &file_ops,
    };
    let entry = handoff_compaction_entry(
        "h1",
        Some("a2"),
        "2026-01-01T00:00:10.000Z",
        &document,
        &commit,
    );
    let wire = serde_json::to_value(&entry).unwrap();
    assert_eq!(wire["method"], "handoff");
    assert_eq!(wire["firstKeptEntryId"], "a2");
    assert_eq!(wire["tokensBefore"], 500);
    assert_eq!(wire["tokensAfter"], 80);
    assert!(wire.get("shortSummary").is_none() || wire["shortSummary"].is_null());
    // A file both read and written lands only in modifiedFiles (TS semantics).
    assert_eq!(wire["details"]["readFiles"], json!(["src/lib.rs"]));
    assert_eq!(wire["details"]["modifiedFiles"], json!(["src/main.rs"]));
    assert!(wire["summary"]
        .as_str()
        .unwrap()
        .contains("## Goal\ndo the thing"));
    assert!(wire["summary"].as_str().unwrap().contains("<files>"));

    // 4. Replay: the committed entry rebuilds like a local compaction
    // (summary first, kept tail from firstKeptEntryId, then post-compaction).
    let mut entries: Vec<serde_json::Value> = vec![
        json!({"type":"message","id":"u1","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z","message":{"role":"user","content":"first","timestamp":1}}),
        json!({"type":"message","id":"a2","parentId":"u1","timestamp":"2026-01-01T00:00:01.000Z","message":{"role":"assistant","content":"answer","timestamp":2}}),
    ];
    let mut entry_value = serde_json::to_value(&entry).unwrap();
    entry_value["type"] = "compaction".into();
    entries.push(entry_value);
    entries.push(json!({"type":"message","id":"u3","parentId":"h1","timestamp":"2026-01-01T00:00:11.000Z","message":{"role":"user","content":"after","timestamp":3}}));
    let ctx = build_session_context(&entries, Leaf::Last).unwrap();
    let roles: Vec<&str> = ctx
        .messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["compactionSummary", "assistant", "user"]);
}
