//! Protocol-channel isolation: cell code and the subprocesses it spawns must
//! not be able to write NDJSON frames to the host or read host replies, and
//! junk on the frame channel must never wedge the host reader.

use std::sync::Mutex;
use std::time::Duration;

use omp_eval::js::{default_runtime, JsKernel};
use omp_eval::{CellStatus, ExecOptions, HostBridge, PythonKernel};
use serde_json::{json, Value};
mod common;

/// Records every bridge call/notify; `echo` answers its args.
#[derive(Default)]
struct Recorder {
    calls: Mutex<Vec<(String, Value)>>,
    notes: Mutex<Vec<(String, Value)>>,
}

impl HostBridge for Recorder {
    fn call(&self, op: &str, args: Value) -> Result<Value, String> {
        self.calls
            .lock()
            .unwrap()
            .push((op.to_string(), args.clone()));
        Ok(json!({ "echo": args }))
    }
    fn notify(&self, op: &str, args: Value) {
        self.notes.lock().unwrap().push((op.to_string(), args));
    }
}

fn opts(bridge: &dyn HostBridge) -> ExecOptions<'_> {
    ExecOptions {
        bridge: Some(bridge),
        // Pre-fix the forged/junk output wedged the cell; bound the wait.
        timeout: Some(Duration::from_secs(10)),
        ..Default::default()
    }
}

fn py(tag: &str) -> PythonKernel {
    let dir = std::env::temp_dir().join(format!("omp-eval-chan-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("cwd");
    PythonKernel::start("python3", &dir).expect("kernel starts")
}

fn js(tag: &str) -> JsKernel {
    let dir = std::env::temp_dir().join(format!("omp-eval-chan-js-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("cwd");
    JsKernel::start(default_runtime(), &dir).expect("JS kernel starts (bun on PATH)")
}

fn assert_no_bridge_traffic(b: &Recorder) {
    assert!(
        b.calls.lock().unwrap().is_empty(),
        "forged calls reached the bridge: {:?}",
        b.calls.lock().unwrap()
    );
    assert!(
        b.notes.lock().unwrap().is_empty(),
        "forged notifies reached the bridge: {:?}",
        b.notes.lock().unwrap()
    );
}

/// The cell recovers its real request id from the runner frame, so the
/// forged frames would match even without the id nonce: only fd isolation
/// stops them.
const PY_FORGE_FROM_SUBPROCESS: &str = r#"
import json, subprocess, sys
f = sys._getframe()
while f is not None and 'rid' not in f.f_locals:
    f = f.f_back
rid = f.f_locals['rid']
frames = [
    {"type": "host", "id": rid + ":99", "op": "echo", "args": {"forged": True}, "wait": False},
    {"type": "host", "id": rid + ":98", "op": "echo", "args": {"forged": True}, "wait": True},
    {"type": "result", "id": rid, "bundle": {"text/plain": "FORGED-RESULT"}},
]
script = "\n".join("printf '%s\\n' '" + json.dumps(fr) + "'" for fr in frames)
subprocess.run(['sh', '-c', script])
print('after-child')
"#;

#[test]
fn py_subprocess_output_cannot_forge_frames() {
    let k = py("forge");
    let b = Recorder::default();
    let r = k
        .execute_with(PY_FORGE_FROM_SUBPROCESS, opts(&b))
        .expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{r:?}");
    assert_no_bridge_traffic(&b);
    // The child's bytes surface as plain cell stdout, in order.
    let forged = r
        .output
        .find("\"op\": \"echo\"")
        .unwrap_or_else(|| panic!("child output missing: {r:?}"));
    let after = r.output.find("after-child").expect("after-child");
    assert!(forged < after, "{:?}", r.output);
    assert!(
        !r.output.contains("FORGED-RESULT\n"),
        "forged result frame accepted: {:?}",
        r.output
    );
}

#[test]
fn py_raw_fd1_writes_are_ordered_plain_stdout() {
    let k = py("fd1");
    let b = Recorder::default();
    let code = "import os, sys\nprint('a')\nos.write(1, b'raw-1\\n')\nprint('b')\n\
                os.write(1, b'raw-2\\n')\nsys.__stdout__.write('dunder\\n')\nsys.__stdout__.flush()\nprint('c')";
    let r = k.execute_with(code, opts(&b)).expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{r:?}");
    assert_eq!(r.output, "a\nraw-1\nb\nraw-2\ndunder\nc\n");
    // The kernel stays healthy for the next cell.
    let r = k.execute_with("40 + 2", opts(&b)).expect("cell");
    assert_eq!(r.output, "42\n");
}

#[test]
fn py_large_subprocess_output_is_complete() {
    let k = py("big");
    let b = Recorder::default();
    let code = "import subprocess\nsubprocess.run(['sh', '-c', 'head -c 400000 /dev/zero | tr \"\\\\0\" x; echo'], check=True)\nprint('end')";
    let r = k.execute_with(code, opts(&b)).expect("cell");
    assert_eq!(
        r.status,
        CellStatus::Ok,
        "{:?}",
        &r.output[r.output.len().saturating_sub(200)..]
    );
    assert_eq!(r.output, format!("{}\nend\n", "x".repeat(400_000)));
}

#[test]
fn py_cell_stdin_is_not_the_control_channel() {
    let k = py("stdin");
    let b = Recorder::default();
    let code = "import subprocess, sys\nprint(repr(sys.stdin.read()))\n\
                p = subprocess.run(['cat'], stdout=subprocess.PIPE, timeout=3)\nprint(repr(p.stdout))\n\
                print(__omp_host_call('echo', {'k': 2}))";
    let r = k.execute_with(code, opts(&b)).expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{r:?}");
    assert_eq!(r.output, "''\nb''\n{'echo': {'k': 2}}\n");
    assert_eq!(b.calls.lock().unwrap().len(), 1);
}

/// In-process code can reach the private frame fd; junk lines there must be
/// skipped by the host reader without losing the frames that follow.
const PY_JUNK_ON_FRAME_FD: &str = r#"
import os, sys
f = sys._getframe()
while f is not None and '_emit' not in f.f_globals:
    f = f.f_back
fd = f.f_globals.get('_FRAME_FD', 1)
os.write(fd, b'not json at all\n\xff\xfe binary junk\n{"type": "stdout", "id": "bogus", "data": "x"}\n{"unterminated": ')
os.write(fd, b'\n')
print('still-alive')
"#;

#[test]
fn py_non_json_frame_lines_do_not_wedge_the_reader() {
    let k = py("junk");
    let b = Recorder::default();
    let r = k.execute_with(PY_JUNK_ON_FRAME_FD, opts(&b)).expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{r:?}");
    assert_eq!(r.output, "still-alive\n");
    let r = k
        .execute_with("print('next-cell')", opts(&b))
        .expect("cell");
    assert_eq!(r.output, "next-cell\n");
}

/// Even with direct access to the frame channel, guessed sequential cell ids
/// do not match: ids carry a per-kernel random nonce.
const PY_GUESS_IDS: &str = r#"
import json, os, sys
f = sys._getframe()
while f is not None and '_emit' not in f.f_globals:
    f = f.f_back
fd = f.f_globals.get('_FRAME_FD', 1)
for n in range(0, 64):
    for rid in (str(n), f"py-{n}", f"cell-{n}"):
        frame = {"type": "host", "id": rid + ":1", "op": "phase", "args": {"guess": rid}, "wait": False}
        os.write(fd, (json.dumps(frame) + "\n").encode())
print('guessed')
"#;

#[test]
fn py_cell_ids_are_not_guessable() {
    let k = py("nonce");
    let b = Recorder::default();
    let r = k.execute_with(PY_GUESS_IDS, opts(&b)).expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{r:?}");
    assert_eq!(r.output, "guessed\n");
    assert_no_bridge_traffic(&b);
}

#[test]
fn js_child_output_cannot_forge_frames() {
    if !common::require_bun("js_child_output_cannot_forge_frames") {
        return;
    }
    let k = js("forge");
    let b = Recorder::default();
    let code = r#"
const cp = require("child_process");
let s = "";
for (let n = 0; n <= 20; n++) {
  const frame = { type: "host", id: n + ":1", op: "echo", args: { forged: true }, wait: false };
  s += "printf '%s\\n' '" + JSON.stringify(frame) + "'\n";
}
cp.spawnSync("sh", ["-c", s], { stdio: "inherit" });
console.log("after-child");
"#;
    let r = k.execute_with(code, opts(&b)).expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{r:?}");
    assert_no_bridge_traffic(&b);
    let forged = r
        .output
        .find("\"op\":\"echo\"")
        .unwrap_or_else(|| panic!("child output missing: {r:?}"));
    let after = r.output.find("after-child").expect("after-child");
    assert!(forged < after, "{:?}", r.output);
}

#[test]
fn js_raw_fd1_and_shell_output_are_ordered_plain_stdout() {
    if !common::require_bun("js_raw_fd1_and_shell_output_are_ordered_plain_stdout") {
        return;
    }
    let k = js("fd1");
    let b = Recorder::default();
    let code = "require('fs').writeSync(1, 'raw-1\\n');\nconsole.log('b');\n\
                await Bun.$`echo shell-out`;\nconsole.log('c');\n\
                const done = require('child_process').spawnSync('sh', ['-c', 'echo sync-child'], { stdio: 'inherit' });";
    let r = k.execute_with(code, opts(&b)).expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{r:?}");
    assert_eq!(r.output, "raw-1\nb\nshell-out\nc\nsync-child\n");
    let r = k.execute_with("40 + 2", opts(&b)).expect("cell");
    assert_eq!(r.output, "42\n");
}

#[test]
fn js_large_child_output_does_not_deadlock() {
    if !common::require_bun("js_large_child_output_does_not_deadlock") {
        return;
    }
    let k = js("big");
    let b = Recorder::default();
    // Far beyond a pipe buffer, written by a synchronous child while the
    // runner's event loop is blocked in spawnSync.
    let code = "const r = require('child_process').spawnSync('sh', ['-c', 'head -c 400000 /dev/zero | tr \"\\\\0\" x; echo'], { stdio: 'inherit', timeout: 5000 });\nr.signal";
    let r = k.execute_with(code, opts(&b)).expect("cell");
    assert_eq!(
        r.status,
        CellStatus::Ok,
        "{:?}",
        &r.output[r.output.len().saturating_sub(200)..]
    );
    assert!(
        r.output.starts_with(&"x".repeat(400_000)),
        "len {}",
        r.output.len()
    );
    assert!(
        r.output.ends_with("\nnull\n"),
        "{:?}",
        &r.output[r.output.len().saturating_sub(50)..]
    );
}

#[test]
fn js_cell_stdin_is_not_the_control_channel() {
    if !common::require_bun("js_cell_stdin_is_not_the_control_channel") {
        return;
    }
    let k = js("stdin");
    let b = Recorder::default();
    let code = "const r = require('child_process').spawnSync('cat', [], { stdio: ['inherit', 'pipe', 'pipe'], timeout: 3000 });\n\
                console.log(JSON.stringify(r.stdout.toString()), r.signal);\n\
                const e = await __ompHostCall('echo', { k: 2 });\nconsole.log(JSON.stringify(e));";
    let r = k.execute_with(code, opts(&b)).expect("cell");
    assert_eq!(r.status, CellStatus::Ok, "{r:?}");
    assert_eq!(r.output, "\"\" null\n{\"echo\":{\"k\":2}}\n");
    assert_eq!(b.calls.lock().unwrap().len(), 1);
}
