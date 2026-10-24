//! Integration tests: persistent JS kernel — state across cells, top-level
//! await, timeout force-kill, omp-style error formatting.

use std::io::Write as _;
use std::process::{Command, Stdio};
use std::time::Duration;

use omp_eval::js::{default_runtime, JsKernel};
use omp_eval::{CellResult, CellStatus};
mod common;

fn kernel(tag: &str) -> JsKernel {
    let dir = std::env::temp_dir().join(format!("omp-eval-js-test-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("cwd");
    JsKernel::start(default_runtime(), &dir).expect("kernel starts; bun must be on PATH")
}

/// Runner script sanity: the NDJSON protocol round-trips a simple cell.
#[test]
fn runner_speaks_protocol() {
    if !common::require_bun("runner_speaks_protocol") {
        return;
    }
    let mut child = Command::new(default_runtime())
        .args(["-e", include_str!("../src/runner.js")])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bun");
    let stdin = child.stdin.as_mut().expect("stdin");
    stdin
        .write_all(b"{\"id\": \"r1\", \"code\": \"print('hi')\\n1+1\"}\n")
        .unwrap();
    stdin.write_all(b"{\"type\": \"exit\"}\n").unwrap();
    let out = child.wait_with_output().expect("wait");
    let text = String::from_utf8_lossy(&out.stdout);
    let started = text.lines().any(|l| l.contains("\"type\":\"started\""));
    let stdout_frame = text
        .lines()
        .any(|l| l.contains("\"type\":\"stdout\"") && l.contains("hi"));
    let result = text
        .lines()
        .any(|l| l.contains("\"type\":\"result\"") && l.contains('2'));
    let done = text
        .lines()
        .any(|l| l.contains("\"type\":\"done\"") && l.contains("\"status\":\"ok\""));
    assert!(started && stdout_frame && result && done, "frames: {text}");
}

#[test]
fn state_persists_across_cells() {
    if !common::require_bun("state_persists_across_cells") {
        return;
    }
    let k = kernel("state");
    let r1 = k
        .execute("let accum = 41\naccum + 1", None)
        .expect("cell 1");
    assert_eq!(r1.status, CellStatus::Ok);
    assert_eq!(r1.output, "42\n");
    assert_eq!(r1.execution_count, 1);

    // let/const demoted to globals, function/class declarations persist
    let r2 = k
        .execute(
            "const double = x => x * 2\nclass Box { constructor(v) { this.v = v } }\nnew Box(double(accum)).v",
            None,
        )
        .expect("cell 2");
    assert_eq!(r2.status, CellStatus::Ok);
    assert_eq!(r2.output, "82\n"); // double(41)

    let r3 = k.execute("accum + double(1)", None).expect("cell 3");
    assert_eq!(r3.output, "43\n");
    k.shutdown();
}

#[test]
fn top_level_await_works_and_print_interleaves() {
    if !common::require_bun("top_level_await_works_and_print_interleaves") {
        return;
    }
    let k = kernel("await");
    let r = k
        .execute(
            "print('before')\nconst r = await new Promise(res => setTimeout(() => res(7), 10))\nprint('after')\nr",
            None,
        )
        .expect("await cell");
    assert_eq!(r.status, CellStatus::Ok);
    assert_eq!(r.output, "before\nafter\n7\n", "output: {:?}", r.output);
    k.shutdown();
}

#[test]
fn timeout_kills_worker_and_kernel_recreates() {
    if !common::require_bun("timeout_kills_worker_and_kernel_recreates") {
        return;
    }
    let k = kernel("timeout");
    let started = std::time::Instant::now();
    let r = k.execute(
        "let total = 0\nwhile (true) { total += 1 }",
        Some(Duration::from_millis(300)),
    );
    let r = r.expect("timed-out cell returns a result");
    let elapsed = started.elapsed();
    assert_eq!(r.status, CellStatus::Interrupted, "output: {:?}", r.output);
    assert!(r.timed_out);
    assert!(
        elapsed < Duration::from_secs(5),
        "kill should be prompt: {elapsed:?}"
    );
    assert!(!k.is_alive(), "worker killed on timeout");

    // omp JS semantics: the next call transparently recreates the kernel.
    let after = k.execute("'fresh'", None).expect("post-timeout cell");
    assert_eq!(after.status, CellStatus::Ok);
    // Bun.inspect renders strings quoted (the JS analog of Python's repr)
    assert_eq!(after.output, "\"fresh\"\n");
    assert!(k.is_alive());
    k.shutdown();
}

#[test]
fn error_formatting_matches_omp_js() {
    if !common::require_bun("error_formatting_matches_omp_js") {
        return;
    }
    let k = kernel("error");
    let r = k
        .execute("function boom() { throw new Error('nope') }\nboom()", None)
        .expect("error cell");
    assert_eq!(r.status, CellStatus::Error);
    let err = r.error.as_ref().expect("structured error");
    assert_eq!(err.name, "Error");
    assert_eq!(err.value, "nope");
    // stack pushed like the TS executor (error.stack), runner frames filtered
    assert!(
        err.traceback[0].starts_with("Error: nope"),
        "{:?}",
        err.traceback
    );
    assert!(
        err.traceback.iter().any(|l| l.contains("boom")),
        "user frame kept: {:?}",
        err.traceback
    );
    assert!(
        !err.traceback
            .iter()
            .any(|l| l.contains("runCell") || l.contains("processTicksAndRejections")),
        "runner frames filtered: {:?}",
        err.traceback
    );
    // TS executor pushes `error.stack` verbatim: message first, user frames kept.
    assert!(r.output.starts_with("Error: nope\n"), "{:?}", r.output);
    assert!(r.output.contains("boom"), "{:?}", r.output);

    // Transactional cells (prelude_txn.js, default on): names the failed cell
    // created are rolled back; the kernel itself survives the error.
    let r2 = k.execute("typeof boom", None).expect("post-error cell");
    assert_eq!(r2.output, "\"undefined\"\n");
    k.shutdown();
}

#[test]
fn js_timeout_annotation_matches_ts() {
    assert_eq!(
        CellResult::js_timeout_annotation(None),
        "Command timed out. The JS worker was force-killed and its VM state was reset; variables from earlier cells are gone."
    );
    assert_eq!(
        CellResult::js_timeout_annotation(Some(Duration::from_millis(1500))),
        "Command timed out after 2 seconds. The JS worker was force-killed and its VM state was reset; variables from earlier cells are gone."
    );
}

#[test]
fn display_and_prelude_helpers() {
    if !common::require_bun("display_and_prelude_helpers") {
        return;
    }
    let k = kernel("helpers");
    let dir = std::env::temp_dir();
    let file = dir.join(format!("omp-eval-js-helper-{}.txt", std::process::id()));
    let r = k
        .execute(
            &format!(
                "await write('{}', 'l1\\nl2\\nl3')\nconst t = await read('{}', {{ offset: 2 }})\nt",
                file.display(),
                file.display()
            ),
            None,
        )
        .expect("read/write cell");
    assert_eq!(r.status, CellStatus::Ok);
    assert_eq!(r.output, "\"l2\\nl3\"\n", "output: {:?}", r.output);

    let r2 = k
        .execute("display({ a: 1 })\n'done'", None)
        .expect("display cell");
    assert_eq!(r2.output, "\"done\"\n"); // Bun.inspect: quoted string
    assert_eq!(r2.displays.len(), 1);
    assert!(r2.displays[0]
        .get("text/plain")
        .and_then(|v| v.as_str())
        .is_some_and(|s| s.contains("a: 1")));
    std::fs::remove_file(&file).ok();
    k.shutdown();
}
