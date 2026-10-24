//! Live ingest end to end: the watcher picks up an appended transcript line,
//! syncs it, and subscribers see the version bump.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::broadcast::error::TryRecvError;

use super::*;
use crate::testutil::golden::FIXTURE_HOME;
use crate::testutil::TestHome;

const DEADLINE: Duration = Duration::from_secs(30);

fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(start.elapsed() < DEADLINE, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn is_assistant_usage(line: &str) -> bool {
    serde_json::from_str::<Value>(line).is_ok_and(|v| {
        v["type"] == "message"
            && v["message"]["role"] == "assistant"
            && v["message"]["usage"].is_object()
    })
}

/// The smallest fixture transcript with an assistant message, and that message's line.
fn small_fixture_session() -> Option<(PathBuf, String)> {
    let root = Path::new(FIXTURE_HOME).join(".omp/agent/sessions");
    let mut files: Vec<(u64, PathBuf)> = walkdir::WalkDir::new(&root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "jsonl"))
        .filter_map(|e| Some((e.metadata().ok()?.len(), e.into_path())))
        .collect();
    files.sort();
    files.into_iter().find_map(|(_, path)| {
        let text = std::fs::read_to_string(&path).ok()?;
        let line = text.lines().find(|l| is_assistant_usage(l))?.to_owned();
        Some((path, line))
    })
}

#[test]
fn watcher_syncs_appended_line_and_notifies_subscribers() {
    let Some((source, line)) = small_fixture_session() else {
        return;
    };
    let home = TestHome::new();
    let folder = home.path().join(".omp/agent/sessions/-tmp-live");
    std::fs::create_dir_all(&folder).expect("sessions dir");
    let session = folder.join(source.file_name().expect("file name"));
    std::fs::copy(&source, &session).expect("copy session");

    let live = StatsLive::new();
    let mut rx = live.subscribe();
    live.start();
    wait_until("initial full sync", || {
        let status = live.status();
        status.sync.phase == LiveSyncPhase::Idle && status.sync.last_synced_at.is_some()
    });
    let before = crate::aggregator::get_total_message_count().expect("count");
    assert!(before > 0, "initial sync ingested the copied transcript");
    wait_until("initial refresh and version bump", || {
        live.inner.quiescent()
    });
    let version = live.status().version;
    assert!(version > 1, "initial sync bumped the version");
    while let Ok(_) | Err(TryRecvError::Lagged(_)) = rx.try_recv() {}

    // A new assistant turn: same payload, fresh entry id.
    let mut entry: Value = serde_json::from_str(&line).expect("entry json");
    entry["id"] = Value::from("live0001");
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&session)
        .expect("open session");
    writeln!(file, "{entry}").expect("append line");
    drop(file);

    let mut bumped = None;
    wait_until("version bump event after append", || {
        match rx.try_recv() {
            Ok(frame) if frame.status.version > version => bumped = Some(frame),
            Ok(_) | Err(TryRecvError::Lagged(_) | TryRecvError::Empty) => {}
            Err(TryRecvError::Closed) => panic!("live channel closed"),
        }
        bumped.is_some()
    });
    let frame = bumped.expect("bump frame");
    let parsed: LiveStatus = serde_json::from_str(&frame.frame).expect("frame is the status json");
    assert_eq!(parsed.version, frame.status.version);
    assert_eq!(
        crate::aggregator::get_total_message_count().expect("count"),
        before + 1
    );

    live.stop();
    wait_until("live threads to settle", || live.inner.quiescent());
}

#[test]
fn request_sync_coalesces_and_emits_distinct_frames_only() {
    let _home = TestHome::new();
    let live = StatsLive::new();
    let mut rx = live.subscribe();
    // No sessions dir: the sync is empty but still runs and reports idle.
    live.request_sync(None);
    live.request_sync(Some(vec!["/nonexistent/a.jsonl".into()]));
    wait_until("sync settles", || {
        let status = live.status();
        live.inner.quiescent() && status.sync.last_synced_at.is_some()
    });
    let mut frames = Vec::new();
    while let Ok(frame) = rx.try_recv() {
        frames.push(frame.frame);
    }
    assert!(
        !frames.is_empty(),
        "the syncing/idle transitions were emitted"
    );
    assert!(
        frames.windows(2).all(|w| w[0] != w[1]),
        "identical statuses are not re-sent"
    );
    assert_eq!(live.status().sync.phase, LiveSyncPhase::Idle);
}

#[test]
fn watcher_without_sessions_dir_picks_up_transcripts_created_later() {
    let Some((source, _)) = small_fixture_session() else {
        return;
    };
    let home = TestHome::new();
    let live = StatsLive::new();
    live.start();
    // No sessions dir: the watch cannot be set up, the thread polls instead.
    wait_until("empty initial sync", || {
        live.status().sync.last_synced_at.is_some() && live.inner.quiescent()
    });
    assert_eq!(
        crate::aggregator::get_total_message_count().expect("count"),
        0
    );

    let folder = home.path().join(".omp/agent/sessions/-tmp-late");
    std::fs::create_dir_all(&folder).expect("sessions dir");
    std::fs::copy(&source, folder.join(source.file_name().expect("file name")))
        .expect("copy session");
    wait_until("late transcript synced", || {
        crate::aggregator::get_total_message_count().is_ok_and(|n| n > 0)
    });

    live.stop();
    wait_until("live threads to settle", || live.inner.quiescent());
}
