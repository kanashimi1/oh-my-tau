//! Multi-root ingest: a TS-root and an omp-rs-root transcript of the same cwd
//! sync into one folder, and `/api/session/trace` serves the omp-rs file.

use std::path::Path;

use hyper::HeaderMap;

use super::handle_api;
use crate::aggregator::{sync_all_sessions, SyncOptions};
use crate::testutil::golden::FIXTURE_HOME;
use crate::testutil::TestHome;

fn first_fixture_session() -> Option<std::path::PathBuf> {
    let sessions = Path::new(FIXTURE_HOME).join(".omp/agent/sessions");
    let project = std::fs::read_dir(sessions)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .min()?;
    std::fs::read_dir(project)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .min()
}

/// Copy `src` to `dest` with its header cwd rewritten to `cwd`.
fn copy_with_cwd(src: &Path, dest: &Path, cwd: &str) {
    let text = std::fs::read_to_string(src).expect("read fixture");
    let rewritten: Vec<String> = text
        .lines()
        .map(
            |line| match serde_json::from_str::<serde_json::Value>(line) {
                Ok(mut v) if v["type"] == "session" => {
                    v["cwd"] = cwd.into();
                    v.to_string()
                }
                _ => line.to_string(),
            },
        )
        .collect();
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    std::fs::write(dest, rewritten.join("\n") + "\n").unwrap();
}

#[test]
fn ts_and_omp_rs_roots_merge_by_project_and_trace_serves_both() {
    let Some(src) = first_fixture_session() else {
        return;
    };
    let home = TestHome::new();
    let cwd = home.path().join("work/proj");
    std::fs::create_dir_all(&cwd).unwrap();
    let cwd_str = cwd.to_string_lossy().into_owned();
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap();
    let tmp = crate::paths::ts_tmpdir(|k| std::env::var(k).ok());
    let ts_name = crate::paths::ts_session_dir_name(
        &canon(&cwd),
        &canon(home.path()),
        &canon(Path::new(&tmp)),
    );

    let ts_file = crate::paths::sessions_dir()
        .join(&ts_name)
        .join("2026-10-07T00-00-00-000Z_ts.jsonl");
    let rs_file = crate::paths::omp_rs_sessions_dir()
        .join(cwd_str.replace('/', "-"))
        .join("2026-10-07T00-00-01-000Z_rs.jsonl");
    copy_with_cwd(&src, &ts_file, &cwd_str);
    copy_with_cwd(&src, &rs_file, &cwd_str);

    let result = sync_all_sessions(&SyncOptions::default()).expect("sync");
    assert_eq!(result.files, 2, "both roots ingested");

    let rs_str = rs_file.to_string_lossy().into_owned();
    assert_eq!(
        crate::parser::extract_folder_from_path(&rs_str),
        crate::parser::extract_folder_from_path(&ts_file.to_string_lossy())
    );
    assert_eq!(crate::paths::session_project_dir(&rs_str), ts_name);
    assert_eq!(
        crate::parser::classify_agent_type(&rs_str),
        crate::types::AgentType::Main
    );

    let folders: Vec<String> = crate::db::with_db(|db| {
        let mut stmt = db.prepare("SELECT DISTINCT folder FROM messages")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    })
    .unwrap();
    assert_eq!(folders.len(), 1, "one merged project folder: {folders:?}");

    let target = format!("/api/session/trace?file={}", rs_str.replace('/', "%2F"));
    let response = handle_api("GET", &target, &HeaderMap::new()).expect("trace response");
    assert_eq!(
        response.status,
        200,
        "{}",
        String::from_utf8_lossy(&response.body)
    );

    // A later sync of one root never drops the other root's rows.
    sync_all_sessions(&SyncOptions {
        files: Some(vec![ts_file.to_string_lossy().into_owned()]),
        ..Default::default()
    })
    .unwrap();
    let rows: i64 = crate::db::with_db(|db| {
        Ok(db.query_row("SELECT COUNT(*) FROM file_offsets", [], |r| r.get(0))?)
    })
    .unwrap();
    assert_eq!(rows, 2);
}
