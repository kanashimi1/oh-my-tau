//! Integration tests: persistent state, timeout interrupt, omp error framing.

use std::io::Write as _;
use std::process::{Command, Stdio};
use std::time::Duration;

use omp_eval::{CellResult, CellStatus, PythonKernel, INTERRUPT_ESCALATION};

fn kernel(tag: &str) -> PythonKernel {
    let dir = std::env::temp_dir().join(format!("omp-eval-test-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("cwd");
    PythonKernel::start("python3", &dir).expect("kernel starts")
}

/// Runner script sanity: the NDJSON protocol round-trips a simple cell.
#[test]
fn runner_speaks_protocol() {
    let mut child = Command::new("python3")
        .args(["-u", "-c", include_str!("../src/runner.py")])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let stdin = child.stdin.as_mut().expect("stdin");
    stdin
        .write_all(b"{\"id\": \"r1\", \"code\": \"print('hi')\\n1+1\"}\n")
        .unwrap();
    stdin.write_all(b"{\"type\": \"exit\"}\n").unwrap();
    let out = child.wait_with_output().expect("wait");
    let text = String::from_utf8_lossy(&out.stdout);
    let started = text.lines().any(|l| l.contains("\"type\": \"started\""));
    let stdout_frame = text
        .lines()
        .any(|l| l.contains("\"type\": \"stdout\"") && l.contains("hi"));
    let result = text
        .lines()
        .any(|l| l.contains("\"type\": \"result\"") && l.contains('2'));
    let done = text.lines().any(|l| {
        l.contains("\"type\": \"done\"") && l.contains("\"status\": \"ok\"") && l.contains("r1")
    });
    assert!(started && stdout_frame && result && done, "frames: {text}");
}

#[test]
fn state_persists_across_cells() {
    let k = kernel("state");
    let r1 = k.execute("accum = 41\naccum + 1", None).expect("cell 1");
    assert_eq!(r1.status, CellStatus::Ok);
    assert_eq!(r1.output, "42\n");
    assert_eq!(r1.execution_count, 1);

    let r2 = k
        .execute(
            "accum += 1\ndef double(x):\n    return x * 2\n\ndouble(accum)",
            None,
        )
        .expect("cell 2");
    assert_eq!(r2.status, CellStatus::Ok);
    assert_eq!(r2.output, "84\n");
    assert_eq!(r2.execution_count, 2);

    // top-level await cells share the same namespace
    let r3 = k
        .execute(
            "import asyncio\nawait asyncio.sleep(0.01)\ndouble(accum) + accum",
            None,
        )
        .expect("cell 3");
    assert_eq!(r3.status, CellStatus::Ok);
    assert_eq!(r3.output, "126\n");

    // display() helper captures a display bundle and prints nothing to output
    let r4 = k.execute("display({'a': 1})", None).expect("cell 4");
    assert_eq!(r4.output, "");
    assert_eq!(r4.displays.len(), 1);
    k.shutdown();
}

#[test]
fn timeout_interrupts_and_kernel_survives() {
    let k = kernel("timeout");
    let started = std::time::Instant::now();
    let r = k.execute(
        "total = 0\nfor i in range(10**8):\n    total += i",
        Some(Duration::from_millis(300)),
    );
    let r = r.expect("timed-out cell returns a result");
    let elapsed = started.elapsed();
    assert_eq!(r.status, CellStatus::Interrupted, "output: {:?}", r.output);
    assert!(r.cancelled);
    assert!(r.timed_out);
    assert!(
        r.output
            .contains("KeyboardInterrupt: Execution interrupted"),
        "{:?}",
        r.output
    );
    assert!(
        elapsed >= Duration::from_millis(300),
        "interrupted early: {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_millis(300) + INTERRUPT_ESCALATION,
        "no escalation: {elapsed:?}"
    );
    // Kernel survives; the interrupted cell's new names are rolled back
    // (prelude_txn.py, default on), earlier state is untouched.
    let after = k
        .execute("'total' in globals()", None)
        .expect("post-interrupt cell");
    assert_eq!(after.status, CellStatus::Ok);
    assert_eq!(after.output.trim(), "False");
    assert!(k.is_alive());
    k.shutdown();
}

#[test]
fn error_traceback_matches_omp_format() {
    let k = kernel("error");
    // runtime error: traceback with runner frames filtered, "Traceback (most recent call last):" header
    let r = k
        .execute("def boom():\n    raise ValueError('nope')\n\nboom()", None)
        .expect("error cell");
    assert_eq!(r.status, CellStatus::Error);
    let err = r.error.as_ref().expect("structured error");
    assert_eq!(err.name, "ValueError");
    assert_eq!(err.value, "nope");
    assert!(
        err.traceback[0].starts_with("Traceback (most recent call last):"),
        "{:?}",
        err.traceback
    );
    assert!(
        err.traceback.iter().any(|l| l.contains("boom")),
        "user frame kept: {:?}",
        err.traceback
    );
    assert!(
        !err.traceback.iter().any(|l| l.contains("runner.py")),
        "runner frames filtered: {:?}",
        err.traceback
    );
    // Traceback first, then the rollback note for `boom` (prelude_txn.py).
    assert!(
        r.output
            .contains("ValueError: nope\n[omp-txn] rolled back: boom"),
        "{:?}",
        r.output
    );

    // syntax error: caret-only formatting, like a REPL
    let r2 = k
        .execute("def broken(:\n    pass", None)
        .expect("syntax cell");
    assert_eq!(r2.status, CellStatus::Error);
    let err2 = r2.error.as_ref().expect("syntax error");
    assert_eq!(err2.name, "SyntaxError");
    assert!(
        err2.traceback[0].starts_with("  File \"<cell>\","),
        "{:?}",
        err2.traceback
    );
    assert!(!err2
        .traceback
        .iter()
        .any(|l| l.contains("Traceback (most recent call last):")));

    // The kernel keeps running after errors and the execution count advances;
    // `boom`, defined by the failed cell, was rolled back (prelude_txn.py).
    let r3 = k
        .execute("'boom' in globals()", None)
        .expect("post-error cell");
    assert_eq!(r3.output, "False\n");
    assert_eq!(r3.execution_count, 3);
    k.shutdown();
}

#[test]
fn stdout_stderr_interleave_in_arrival_order() {
    let k = kernel("interleave");
    let r = k
        .execute(
            "import sys\nprint('out1')\nsys.stderr.write('err1\\n')\nprint('out2')",
            None,
        )
        .expect("interleave cell");
    assert_eq!(r.status, CellStatus::Ok);
    assert_eq!(r.output, "out1\nerr1\nout2\n", "arrival-order interleaving");
    k.shutdown();
}

#[test]
fn restart_after_kill_recreates_kernel() {
    let k = kernel("restart");
    k.execute("marker = 'kept'", None).expect("cell");
    // simulate a wedged kernel: hard-exit from the cell (like the watchdog path)
    let r = k
        .execute("import os\nos._exit(137)", None)
        .expect("dead kernel reported");
    assert!(!k.is_alive() || r.timed_out);
    k.restart().expect("restart");
    assert!(k.is_alive());
    // state is gone after a kill + recreate (omp semantics), but cells run again
    let r2 = k
        .execute("'marker' in dir()", None)
        .expect("cell after restart");
    assert_eq!(r2.output, "False\n");
    k.shutdown();
}

#[test]
fn timeout_annotation_texts_match_omp() {
    assert_eq!(CellResult::timeout_annotation(None), "Command timed out");
    assert_eq!(
        CellResult::timeout_annotation(Some(Duration::from_millis(1500))),
        "Command timed out after 2 seconds"
    );
    let killed = CellResult {
        output: String::new(),
        status: CellStatus::KernelKilled,
        cancelled: true,
        execution_count: 0,
        error: None,
        displays: Vec::new(),
        timed_out: true,
    };
    assert_eq!(
        killed.kernel_timeout_annotation().expect("annotation"),
        "eval cell timed out and the kernel was unresponsive to interrupt; the kernel has been killed and will \
         be recreated on the next call."
    );
    let interrupted = CellResult {
        output: String::new(),
        status: CellStatus::Interrupted,
        cancelled: true,
        execution_count: 0,
        error: None,
        displays: Vec::new(),
        timed_out: true,
    };
    let note = interrupted.kernel_timeout_annotation().expect("annotation");
    assert!(note.starts_with("eval cell timed out; kernel interrupted but remains running."));
}
