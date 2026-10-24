//! Cross-implementation tests: wyhash vectors from `Bun.hash`, and parsing
//! the real fixture sessions against counts produced by the TS
//! `parseSessionFile` (oracle dump in `/tmp/ts-parse-compare.ts`; totals
//! hardcoded below so the test stands alone).

use std::os::unix::fs::symlink;
use std::path::Path;

use super::wyhash_hex;
use super::{list_all_session_files, parse_session_file};
use crate::testutil::TestHome;

const FIXTURE_SESSIONS: &str =
    "/home/user/omrs-fixtures/stats/fixture/home/.omp/agent/sessions";

/// TS `parseSessionFile` totals over the fixture (26 files), from
/// `bun /tmp/ts-parse-compare.ts` against the vendored parser.ts.
const TS_TOTALS: &str = "stats=900 userStats=41 userLinks=899 toolCalls=1113 toolResults=1112";

/// Point a [`TestHome`] at the fixture sessions via a symlink; returns the
/// sessions dir inside the temp home. `None` when the fixture is absent.
fn fixture_home() -> Option<(TestHome, std::path::PathBuf)> {
    if !Path::new(FIXTURE_SESSIONS).is_dir() {
        return None;
    }
    let home = TestHome::new();
    let agent = home.path().join(".omp/agent");
    std::fs::create_dir_all(&agent).expect("agent dir");
    let link = agent.join("sessions");
    symlink(FIXTURE_SESSIONS, &link).expect("symlink fixture sessions");
    Some((home, link))
}

#[test]
fn wyhash_matches_bun_vectors() {
    // Captured: `bun -e 'console.log(Bun.hash(x).toString(16))'`.
    for (expected, input) in [
        ("409638ee2bde459", ""),
        ("28d2053309d28531", "a"),
        ("bc9ce12eaf0083ec", "ab"),
        ("2a4f1d7cb516c72", "abc"),
        ("e24bbd9f93f532d", "hello"),
        ("668d5e431c3b2573", "hello world"),
        ("618062c90763579f", &"x".repeat(256)),
    ] {
        assert_eq!(wyhash_hex(input.as_bytes()), expected, "input {input:?}");
    }
    // 300-byte input exercises the 48-byte-round loop + 16-byte tail path.
    let buf: Vec<u8> = (0..300u32).map(|i| ((i * 7 + 3) % 256) as u8).collect();
    assert_eq!(wyhash_hex(&buf), "b1a7bdf2798cead1");
}

#[test]
fn fixture_parse_matches_ts_counts() -> anyhow::Result<()> {
    let Some((_home, _sessions)) = fixture_home() else {
        eprintln!("fixture sessions not found; skipping");
        return Ok(());
    };
    let files = list_all_session_files();
    assert_eq!(files.len(), 26, "fixture file count (TS parsed 26)");

    let mut totals = [0i64; 5];
    for file in &files {
        let result = parse_session_file(file, 0, None, false)?;
        totals[0] += result.stats.len() as i64;
        totals[1] += result.user_stats.len() as i64;
        totals[2] += result.user_links.len() as i64;
        totals[3] += result.tool_calls.len() as i64;
        totals[4] += result.tool_results.len() as i64;

        // Checkpoint protocol: the stored hash equals Bun.hash of the last
        // ≤256 bytes before the new offset, and the offset lands on a line
        // boundary at or past every complete entry.
        let state = result.parser_state.expect("parser state");
        let bytes = std::fs::read(file)?;
        assert_eq!(result.new_offset, state.offset);
        let checkpoint_start = (result.new_offset - 256).max(0) as usize;
        assert!(
            result.new_offset as usize <= bytes.len(),
            "{file}: newOffset past EOF"
        );
        assert_eq!(
            state.checkpoint,
            wyhash_hex(&bytes[checkpoint_start..result.new_offset as usize]),
            "{file}: checkpoint mismatch"
        );
    }
    assert_eq!(
        totals,
        [900, 41, 899, 1113, 1112],
        "totals must match TS parseSessionFile ({TS_TOTALS})"
    );
    Ok(())
}

#[test]
fn incremental_tail_parse_yields_only_remaining_rows() -> anyhow::Result<()> {
    let Some((_home, _sessions)) = fixture_home() else {
        eprintln!("fixture sessions not found; skipping");
        return Ok(());
    };
    let files = list_all_session_files();
    let file = files
        .iter()
        .find(|f| f.ends_with(".jsonl"))
        .expect("a session file");
    let bytes = std::fs::read(file)?;
    // Mid-file offset on a line boundary (skip the first three lines).
    let mut mid = 0usize;
    for _ in 0..3 {
        mid = bytes[mid..]
            .iter()
            .position(|b| *b == b'\n')
            .map(|p| mid + p + 1)
            .expect("three lines present");
    }

    let full = parse_session_file(file, 0, None, false)?;
    let tail = parse_session_file(file, mid as i64, None, false)?;
    assert_eq!(
        tail.new_offset, full.new_offset,
        "tail converges to the same offset"
    );
    // The tail rows are exactly the suffix of the full parse (the prefix
    // lines may themselves carry no stats rows, so counts can be equal).
    let tail_len = tail.stats.len();
    let suffix = &full.stats[full.stats.len() - tail_len..];
    assert_eq!(
        tail.stats
            .iter()
            .map(|s| (&s.entry_id, s.timestamp))
            .collect::<Vec<_>>(),
        suffix
            .iter()
            .map(|s| (&s.entry_id, s.timestamp))
            .collect::<Vec<_>>(),
        "tail rows are the full-parse suffix"
    );

    // Resume with the persisted cursor: same file identity, unchanged mtime —
    // nothing new to read, offset stays put.
    let resume = parse_session_file(file, 0, None, false)?;
    let state = resume.parser_state.clone();
    let again = parse_session_file(file, resume.new_offset, state.as_ref(), false)?;
    assert_eq!(again.new_offset, resume.new_offset);
    assert_eq!(again.reset, Some(false));
    assert!(again.stats.is_empty() && again.tool_calls.is_empty());
    Ok(())
}
