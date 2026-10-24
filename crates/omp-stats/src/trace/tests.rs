//! Tests for the trace module. Golden tests replay the recorded TS
//! `/api/session/trace` and `/api/sessions` responses against the port
//! (needs `db_ingest` for the golden fixture; the rest run under `trace`
//! alone). Plus the two behavioral guarantees: path escape → TracePathError,
//! and appending to a transcript changes its fingerprint.

use super::*;

#[cfg(feature = "db_ingest")]
mod golden_tests {
    use super::*;
    use crate::testutil::golden;

    /// Percent-decode the `file=` query param (the goldens only use %2F).
    fn percent_decode(s: &str) -> String {
        let bytes = s.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 3 <= bytes.len() {
                if let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                    if let Ok(v) = u8::from_str_radix(hex, 16) {
                        out.push(v);
                        i += 3;
                        continue;
                    }
                }
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    fn query_param<'a>(request: &'a str, name: &str) -> Option<&'a str> {
        let query = request.split('?').nth(1)?;
        query.split('&').find_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            (key == name).then_some(value)
        })
    }

    /// Map the golden's absolute fixture path into this test's temp home.
    fn map_file(home: &crate::testutil::TestHome, file_param: &str) -> String {
        let decoded = percent_decode(file_param);
        decoded.replace(
            crate::testutil::golden::FIXTURE_HOME,
            &home.path().to_string_lossy(),
        )
    }

    #[test]
    fn trace_requests_match_golden() {
        let Some(home) = golden::fixture_home() else {
            return;
        };
        let requests = golden::golden_requests("/api/session/trace");
        assert_eq!(requests.len(), 6, "expected 6 recorded trace requests");
        for request in requests {
            let file_param = query_param(&request, "file").expect("file param");
            let trace =
                build_session_trace(&map_file(&home, file_param), None).expect("build trace");
            let actual = serde_json::to_value(&trace).expect("serialize trace");
            golden::assert_golden(&request, &actual, &home);
        }
    }

    #[test]
    fn sessions_requests_match_golden() {
        let Some(home) = golden::fixture_home() else {
            return;
        };
        let requests = golden::golden_requests("/api/sessions");
        assert_eq!(requests.len(), 2, "expected 2 recorded sessions requests");
        for request in requests {
            let limit = query_param(&request, "limit")
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or(100);
            let q = query_param(&request, "q");
            let rows = list_session_summaries(limit, q).expect("list summaries");
            let actual = serde_json::to_value(&rows).expect("serialize summaries");
            golden::assert_golden(&request, &actual, &home);
        }
    }
}

#[test]
fn path_outside_sessions_dir_is_trace_path_error() {
    let home = crate::testutil::TestHome::new();
    let sessions = home.path().join(".omp/agent/sessions");
    std::fs::create_dir_all(&sessions).expect("create sessions dir");
    let err = build_session_trace("/etc/passwd", None).expect_err("outside path must fail");
    let path_err = err
        .downcast_ref::<TracePathError>()
        .expect("must downcast to TracePathError");
    assert!(path_err
        .to_string()
        .contains("outside the sessions directory"));
}

#[test]
fn non_transcript_name_is_trace_path_error() {
    let home = crate::testutil::TestHome::new();
    let sessions = home.path().join(".omp/agent/sessions/proj");
    std::fs::create_dir_all(&sessions).expect("create sessions dir");
    let not_transcript = sessions.join("notes.txt");
    std::fs::write(&not_transcript, b"{}").expect("write file");
    let err = build_session_trace(not_transcript.to_str().unwrap(), None)
        .expect_err("non-transcript must fail");
    assert!(
        err.downcast_ref::<TracePathError>().is_some(),
        "must be TracePathError"
    );
}

#[test]
fn fingerprint_changes_after_append() {
    let home = crate::testutil::TestHome::new();
    let sessions = home.path().join(".omp/agent/sessions/proj");
    std::fs::create_dir_all(&sessions).expect("create sessions dir");
    let file = sessions.join("2026-10-15T10-24-49-741Z_01a2-fingerprint.jsonl");
    let line =
        r#"{"type":"session","cwd":"/tmp","title":"t","timestamp":"2026-10-15T10:24:49.741Z"}"#;
    std::fs::write(&file, format!("{line}\n")).expect("write session");

    let before = trace_fingerprint_for_etag(file.to_str().unwrap()).expect("fingerprint");
    // Same mtime granularity can hide an append on coarse-timestamp
    // filesystems; the memo/fingerprint contract only requires the key to
    // change, and mtime is the load-bearing half of it.
    std::fs::write(&file, format!("{line}\n{line}\n")).expect("append session");
    let after = trace_fingerprint_for_etag(file.to_str().unwrap()).expect("fingerprint");
    assert_ne!(
        before, after,
        "fingerprint must change after appending to the file"
    );
}

#[test]
fn get_trace_entry_roundtrip() {
    let home = crate::testutil::TestHome::new();
    let sessions = home.path().join(".omp/agent/sessions/proj");
    std::fs::create_dir_all(&sessions).expect("create sessions dir");
    let file = sessions.join("2026-10-15T10-24-49-741Z_01a2-entry.jsonl");
    std::fs::write(
        &file,
        "{\"type\":\"title\",\"title\":\"hello\"}\n{\"type\":\"message\",\"id\":\"e1\",\"message\":{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":1}}\n",
    )
    .expect("write session");
    let entry = get_trace_entry(file.to_str().unwrap(), "e1").expect("get entry");
    assert!(entry.is_some(), "entry e1 must be found");
    assert!(get_trace_entry(file.to_str().unwrap(), "missing")
        .expect("get entry")
        .is_none());
}

#[test]
fn get_trace_entry_keeps_compaction_payload() {
    // m2: the raw entry JSON must come back unchanged (TS returns the parsed
    // line verbatim), not a re-typed SessionEntry with dropped fields.
    let home = crate::testutil::TestHome::new();
    let sessions = home.path().join(".omp/agent/sessions/proj");
    std::fs::create_dir_all(&sessions).expect("create sessions dir");
    let file = sessions.join("2026-10-15T10-24-49-741Z_01a2-compact.jsonl");
    let line = "{\"type\":\"compaction\",\"id\":\"c9\",\"timestamp\":\"2026-10-15T10:25:00.000Z\",\"tokensBefore\":142000,\"tokensAfter\":42000,\"summary\":\"shrunk context\"}";
    std::fs::write(&file, format!("{line}\n")).expect("write session");
    let entry = get_trace_entry(file.to_str().unwrap(), "c9")
        .expect("get entry")
        .expect("entry c9");
    let value = serde_json::to_value(&entry).expect("serialize entry");
    assert_eq!(value["type"], "compaction");
    assert_eq!(value["tokensBefore"], 142000);
    assert_eq!(value["summary"], "shrunk context");
}

/// Write a minimal transcript with one model call so it has activity bounds.
fn write_transcript(path: &std::path::Path, agent: Option<&str>, start_ms: i64) {
    let mut lines = vec![
        "{\"type\":\"session\",\"cwd\":\"/tmp\",\"timestamp\":\"2026-10-15T10:24:49.741Z\"}"
            .to_string(),
    ];
    if let Some(agent) = agent {
        lines.push(format!(
            "{{\"type\":\"session_init\",\"id\":\"init\",\"agent\":\"{agent}\",\"resolvedModel\":\"m1\"}}"
        ));
    }
    lines.push(format!(
        "{{\"type\":\"message\",\"id\":\"a1\",\"timestamp\":\"2026-10-15T10:24:50.000Z\",\"message\":{{\"role\":\"assistant\",\"model\":\"m1\",\"timestamp\":{start_ms},\"completedAt\":{},\"usage\":{{\"totalTokens\":10,\"cost\":{{\"total\":0.001}}}}}}}}",
        start_ms + 1000
    ));
    std::fs::write(path, lines.join("\n") + "\n").expect("write transcript");
}

#[test]
fn grandchild_track_labels_and_advisor_span() {
    // M1: with root R → child C → grandchild G, C's label must be
    // "explore C" (not overwritten by G's), and an unlinked advisor with no
    // agent must produce an "Advisor" Agents-lane span.
    let home = crate::testutil::TestHome::new();
    let sessions = home.path().join(".omp/agent/sessions/proj");
    let artifacts_c = sessions.join("2026-10-15T10-24-49-741Z_01a2-root");
    let artifacts_g = artifacts_c.join("C");
    std::fs::create_dir_all(&artifacts_g).expect("create dirs");
    let root = sessions.join("2026-10-15T10-24-49-741Z_01a2-root.jsonl");
    write_transcript(&root, Some("chief"), 1_700_000_000_000);
    write_transcript(
        &artifacts_c.join("C.jsonl"),
        Some("explore"),
        1_700_000_000_100,
    );
    write_transcript(&artifacts_g.join("G.jsonl"), None, 1_700_000_000_200);
    write_transcript(
        &artifacts_c.join("__advisor.jsonl"),
        None,
        1_700_000_000_300,
    );

    let trace = build_session_trace(root.to_str().unwrap(), None).expect("build trace");
    let track = |id: &str| trace.tracks.iter().find(|t| t.id == id).expect("track");
    assert_eq!(
        track("C").label,
        "explore C",
        "child label must use its own agent"
    );
    assert_eq!(
        track("C/G").label,
        "agent G",
        "grandchild label must not be overwritten"
    );
    assert_eq!(track("__advisor").label, "Advisor");
    let root_spans = &track("main").spans;
    let advisor_span = root_spans
        .iter()
        .find(|s| s.child_track_id.as_deref() == Some("__advisor"))
        .expect("advisor Agents-lane span");
    assert_eq!(
        advisor_span.label, "Advisor",
        "unlinked advisor span falls back to the track label"
    );
}

#[test]
fn non_ascii_session_name_does_not_panic() {
    // M3: byte-slicing on a non-ASCII basename used to panic every list call.
    let home = crate::testutil::TestHome::new();
    let sessions = home.path().join(".omp/agent/sessions/proj");
    std::fs::create_dir_all(&sessions).expect("create sessions dir");
    let weird = sessions.join("2026-09-01T10-24-49-74é_x.jsonl");
    write_transcript(&weird, None, 1_700_000_000_000);
    let rows = list_session_summaries(100, None).expect("list must not panic");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].file, weird.to_string_lossy());
}

#[test]
fn huge_timestamps_do_not_panic() {
    // m1: values past i64 range must saturate, not overflow-panic in debug.
    let home = crate::testutil::TestHome::new();
    let sessions = home.path().join(".omp/agent/sessions/proj");
    std::fs::create_dir_all(&sessions).expect("create sessions dir");
    let file = sessions.join("2026-10-15T10-24-49-741Z_01a2-huge.jsonl");
    std::fs::write(
        &file,
        "{\"type\":\"session\",\"timestamp\":\"2026-10-15T10:24:49.741Z\"}\n{\"type\":\"message\",\"id\":\"a1\",\"timestamp\":\"2026-10-15T10:24:50.000Z\",\"message\":{\"role\":\"assistant\",\"model\":\"m1\",\"timestamp\":9e18,\"duration\":1e18,\"usage\":{\"totalTokens\":10,\"cost\":{\"total\":0.001}}}}\n{\"type\":\"message\",\"id\":\"u1\",\"timestamp\":\"2026-10-15T10:25:00.000Z\",\"message\":{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":-1e19}}\n",
    )
    .expect("write session");
    let trace = build_session_trace(file.to_str().unwrap(), None).expect("build trace");
    assert!(trace.summary.wall_ms >= 0, "wall_ms must not wrap negative");
    let tool_result = "{\"type\":\"message\",\"id\":\"t1\",\"timestamp\":\"2026-10-15T10:26:00.000Z\",\"message\":{\"role\":\"toolResult\",\"toolCallId\":\"x\",\"timestamp\":9e18,\"durationMs\":-1e19,\"details\":{\"results\":[{\"id\":\"nope\"}]}}}\n";
    std::fs::write(&file, std::fs::read_to_string(&file).unwrap() + tool_result).expect("append");
    build_session_trace(file.to_str().unwrap(), None).expect("rebuild with huge tool facts");
}
