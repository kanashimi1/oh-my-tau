// omp-rs JavaScript runner — persistent Bun subprocess driven by the Rust host.
//
// NDJSON protocol over stdin/stdout, same frame shape as the Python runner
// (crates/omp-eval/src/runner.py), which mirrors the omp coding-agent host
// protocol (pi-coding-agent/src/eval/py/runner.py):
//
// Host -> runner:
//   {"id": str, "code": str, "storeHistory": bool?, "timeoutMs": int?,
//    "idleTimeout": bool?}                # idleTimeout: pause the deadline
//                                          window while the cell awaits host
//                                          bridge calls (host still enforces)
//   {"type": "host_reply", "id": "<cellId>:<n>", "ok": bool,
//    "result": any | "error": str}        # reply to a wait:true host frame
//   {"type": "exit"}                     # graceful shutdown
//
// Runner -> host:
//   {"type": "started", "id": ...}
//   {"type": "stdout",  "id": ..., "data": str}
//   {"type": "stderr",  "id": ..., "data": str}
//   {"type": "display", "id": ..., "bundle": {<mime>: <value>}}
//   {"type": "result",  "id": ..., "bundle": {<mime>: <value>}}
//   {"type": "error",   "id": ..., "ename": str, "evalue": str, "traceback": [str]}
//   {"type": "host",     "id": "<cellId>:<n>", "op": str, "args": {...},
//                         "wait": bool}    # cell -> host bridge request
//   {"type": "done",    "id": ..., "status": "ok"|"error",
//                        "executionCount": int, "cancelled": bool}
//
// Semantics ported from omp's JS kernel (pi-coding-agent/src/eval/js): one
// long-lived interpreter per kernel so top-level state persists across cells;
// top-level `await` works; the trailing expression of a cell is captured as
// the cell result. Unlike Python there is no safe in-process interrupt for
// synchronous user code (omp force-kills the worker), so timeouts are
// enforced host-side by killing this process — the runner ignores timeoutMs.
//
// Cell transforms (minimal port of omp `wrapCode`, without babel — a small
// top-level scanner instead of an AST):
//   - `returnFinalExpression`: the final top-level expression statement is
//     rewritten to `__omp_set_final_expr__((expr))`.
//   - `demoteTopLevelLexicals`: top-level `let`/`const`/`class`/function
//     declarations are demoted to sloppy global assignments so bindings
//     survive the cell (omp publishes them onto the VM context global).
//   - `requiresAsyncWrapper`: cells with a top-level `await` run inside an
//     async IIFE (still sloppy, so demoted assignments stay global).
// Self-contained: Bun built-ins only, no npm dependencies.

"use strict";

const fs = require("node:fs");

let currentId = null;
let executionCount = 0;

// --- parent watchdog --------------------------------------------------------
// When the host process dies (SIGKILL included) this runner must exit
// promptly even while a cell is spinning synchronously: the main thread is
// then unreachable, so a Worker thread (real OS thread) polls the parent pid
// and SIGKILLs the process — the signal lands regardless of what the main
// thread is doing.
try {
  const { Worker } = require("node:worker_threads");
  const ppid = process.ppid;
  const w = new Worker(
    "setInterval(() => { if (process.ppid !== " + ppid + ") process.kill(process.pid, 'SIGKILL'); }, 250);",
    { eval: true },
  );
  // Unref so the watchdog never keeps the runner alive on its own.
  w.unref();
} catch {
  // worker_threads unavailable: rely on host-side cleanup and stdin EOF.
}

// --- protocol channel isolation ---------------------------------------------
// Port of the TS py runner's fd isolation (Linux + Bun FFI): frames go to a
// private close-on-exec dup of the original stdout and host messages come from
// a private dup of the original stdin. fd 1 becomes a capture pipe drained by
// a worker thread into plain `stdout` frames, fd 0 becomes /dev/null, so
// children spawned by cells (Bun.$, child_process with inherited stdio) can
// neither write protocol frames nor read host replies. Without FFI the runner
// falls back to the raw fds 1/0.
let frameFd = 1;
let controlFd = 0;
// Capture state (port, shared ctl, sentinel marker/seq, helpers) while fd 1
// is captured; null otherwise.
let capture = null;

// Capture drain: blocking reads (keeps up while the main thread sits in
// spawnSync), text forwarded over a MessagePort. A `<marker><seq>\0` sentinel
// written by the main thread acknowledges via ctl[0] once every byte written
// to fd 1 before it has been posted (pipe FIFO = ordering barrier); ctl[1] is
// non-zero while a read chunk is still being posted.
const CAPTURE_WORKER = `
const { workerData } = require("node:worker_threads");
const fs = require("node:fs");
const { fd, port, ctl, marker } = workerData;
const buf = Buffer.alloc(65536);
const decoder = new TextDecoder("utf-8");
let pending = "";
for (;;) {
  let n;
  try {
    n = fs.readSync(fd, buf, 0, buf.length, null);
  } catch (e) {
    if (e && (e.code === "EAGAIN" || e.code === "EINTR")) continue;
    break;
  }
  if (n <= 0) break;
  Atomics.store(ctl, 1, 1);
  pending += decoder.decode(buf.subarray(0, n), { stream: true });
  let i;
  while ((i = pending.indexOf(marker)) >= 0) {
    const end = pending.indexOf("\\u0000", i + marker.length);
    if (end < 0) break;
    const digits = pending.slice(i + marker.length, end);
    if (!/^[0-9]{1,15}$/.test(digits)) {
      port.postMessage(pending.slice(0, i + 1));
      pending = pending.slice(i + 1);
      continue;
    }
    if (i > 0) port.postMessage(pending.slice(0, i));
    pending = pending.slice(end + 1);
    Atomics.store(ctl, 0, Number(digits));
    Atomics.notify(ctl, 0);
  }
  // Hold back an incomplete sentinel (or a trailing NUL that may start one).
  let hold = pending.indexOf(marker);
  if (hold < 0) {
    const z = pending.lastIndexOf("\\u0000");
    hold = z >= 0 && marker.startsWith(pending.slice(z)) ? z : pending.length;
  }
  if (hold > 0) {
    port.postMessage(pending.slice(0, hold));
    pending = pending.slice(hold);
  }
  Atomics.store(ctl, 1, 0);
}
`;

function isolateProtocolFds() {
  if (typeof Bun === "undefined" || process.platform !== "linux") return;
  let libc = null;
  try {
    const { dlopen, FFIType, ptr } = require("bun:ffi");
    const symbols = {
      fcntl: { args: [FFIType.i32, FFIType.i32, FFIType.i32], returns: FFIType.i32 },
      dup2: { args: [FFIType.i32, FFIType.i32], returns: FFIType.i32 },
      pipe2: { args: [FFIType.ptr, FFIType.i32], returns: FFIType.i32 },
      close: { args: [FFIType.i32], returns: FFIType.i32 },
      ioctl: { args: [FFIType.i32, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
    };
    for (const name of ["libc.so.6", "libc.so"]) {
      try {
        libc = dlopen(name, symbols).symbols;
        break;
      } catch {
        // try the next soname
      }
    }
    if (libc === null) return;
    const F_DUPFD_CLOEXEC = 1030;
    const F_GETFL = 3;
    const F_SETFL = 4;
    const O_NONBLOCK = 0o4000;
    const O_CLOEXEC = 0o2000000;
    // Bun flips its stdout description to O_NONBLOCK when process.stdout is
    // first touched; do that now, on the original fd 1, so the capture pipe
    // installed below stays blocking for children (a non-blocking fd 1 makes
    // `tr`, `cat`… fail with EAGAIN once the pipe fills).
    void process.stdout;
    void process.stderr;
    const makeBlocking = fd => {
      const flags = libc.fcntl(fd, F_GETFL, 0);
      if (flags >= 0 && flags & O_NONBLOCK) libc.fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
    };
    const out = libc.fcntl(1, F_DUPFD_CLOEXEC, 3);
    const ctl = libc.fcntl(0, F_DUPFD_CLOEXEC, 3);
    if (out < 0 || ctl < 0) {
      if (out >= 0) libc.close(out);
      if (ctl >= 0) libc.close(ctl);
      return;
    }
    frameFd = out;
    controlFd = ctl;
    // Only the runner writes this description now: blocking writes, no spin.
    makeBlocking(frameFd);

    const fds = new Int32Array(2);
    if (libc.pipe2(ptr(fds), O_CLOEXEC) === 0) {
      const { Worker, MessageChannel } = require("node:worker_threads");
      const { port1, port2 } = new MessageChannel();
      const shared = new Int32Array(new SharedArrayBuffer(8));
      const marker = "\u0000omp-sync-" + crypto.randomUUID() + ":";
      const worker = new Worker(CAPTURE_WORKER, {
        eval: true,
        workerData: { fd: fds[0], port: port2, ctl: shared, marker },
        transferList: [port2],
      });
      worker.unref();
      if (libc.dup2(fds[1], 1) === 1) {
        const readFd = fds[0];
        const count = new Int32Array(1);
        const FIONREAD = 0x541b;
        // Bytes written to fd 1 that the worker has not read yet.
        const unread = () => (libc.ioctl(readFd, FIONREAD, ptr(count)) === 0 ? count[0] : 1);
        capture = {
          port: port1,
          ctl: shared,
          marker,
          seq: 0,
          broken: false,
          unread,
          // Re-asserted per cell in case user code flipped fd 1 non-blocking.
          makeStdoutBlocking: () => makeBlocking(1),
        };
        port1.on("message", handleCaptured);
        port1.unref();
      }
      libc.close(fds[1]);
    }

    const nullFd = fs.openSync("/dev/null", "r");
    libc.dup2(nullFd, 0);
    fs.closeSync(nullFd);
  } catch {
    // FFI unavailable: keep whatever isolation was set up so far
  }
}

// Bun marks its stdio pipes O_NONBLOCK, so a full host pipe yields EAGAIN or a
// short write: retry until the whole line is out (the host reader always
// drains), give up only when the fd is gone.
const writeBackoff = new Int32Array(new SharedArrayBuffer(4));
function writeFrameLine(frame) {
  const bytes = Buffer.from(JSON.stringify(frame) + "\n", "utf8");
  let off = 0;
  while (off < bytes.length) {
    try {
      off += fs.writeSync(frameFd, bytes, off, bytes.length - off);
    } catch (e) {
      if (e && e.code === "EAGAIN") {
        Atomics.wait(writeBackoff, 0, 0, 1);
        continue;
      }
      return; // fd closed; the host is gone
    }
  }
}

// fd-1 bytes become plain stdout of the running cell; between cells (a child
// outliving its cell) they go to fd 2 like the TS py runner.
function handleCaptured(text) {
  if (typeof text !== "string" || text === "") return;
  if (currentId !== null) writeFrameLine({ type: "stdout", id: currentId, data: text });
  else forwardToFd2(text);
}

function drainCapturePort() {
  if (capture === null) return;
  const { receiveMessageOnPort } = require("node:worker_threads");
  let m;
  while ((m = receiveMessageOnPort(capture.port)) !== undefined) handleCaptured(m.message);
}

// Ordering barrier: everything written to fd 1 so far is forwarded before
// the caller's next frame (used ahead of result/error/done).
function syncCapture() {
  if (capture === null) return;
  if (!capture.broken) {
    const seq = ++capture.seq;
    try {
      fs.writeSync(1, capture.marker + seq + "\u0000");
      const deadline = Date.now() + 2000;
      for (;;) {
        const seen = Atomics.load(capture.ctl, 0);
        if (seen >= seq) break;
        const left = deadline - Date.now();
        if (left <= 0) {
          capture.broken = true; // drain worker gone: stop writing sentinels
          break;
        }
        Atomics.wait(capture.ctl, 0, seen, left);
      }
    } catch {
      capture.broken = true;
    }
  }
  drainCapturePort();
}

isolateProtocolFds();

function emit(frame) {
  // fd-1 output written before this frame goes first: a full barrier only
  // when bytes are still in the pipe or in the worker's hands.
  if (capture !== null && !capture.broken && (Atomics.load(capture.ctl, 1) !== 0 || capture.unread() > 0)) {
    syncCapture();
  } else {
    drainCapturePort();
  }
  writeFrameLine(frame);
}

function writeFrame(type, data) {
  if (currentId === null) return;
  emit({ type, id: currentId, data });
}
// Preludes (prelude_txn.js) report through the cell's stderr frame.
globalThis.__ompWriteStderr = data => writeFrame("stderr", String(data));

function inspect(value) {
  try {
    return typeof Bun !== "undefined" ? Bun.inspect(value) : JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function formatArg(arg) {
  if (typeof arg === "string") return arg;
  return inspect(arg);
}

// --- user-facing output: console + raw stdout/stderr forwarding -------------
// Raw writes are forwarded to fd 2 (inherited by the host) so a cell that
// grabs process.stdout directly cannot corrupt the NDJSON protocol channel.
function forwardToFd2(chunk) {
  try {
    fs.writeSync(2, String(chunk));
  } catch {
    // ignore
  }
  return true;
}
try {
  process.stdout.write = chunk => forwardToFd2(chunk);
} catch {
  // non-fatal
}

const consoleMethods = ["log", "info", "debug", "warn", "error", "dir"];
for (const name of consoleMethods) {
  // eslint-disable-next-line no-console
  console[name] = (...args) => {
    writeFrame("stdout", args.map(formatArg).join(" ") + "\n");
  };
}

// --- prelude helpers --------------------------------------------------------
// display/log/phase/read/write/env/tool/output come from the embedded
// preludes (loaded at startup below) and reach the host through the bridge;
// only `print` stays native because the preludes deliberately do not define
// it.
globalThis.print = (...args) => {
  writeFrame("stdout", args.map(formatArg).join(" ") + "\n");
};

// --- host bridge ------------------------------------------------------------
// Prelude helpers call `globalThis.__ompHostCall(op, args)` (waits for the
// host reply) and `globalThis.__ompHostNotify(op, args)` (fire-and-forget).
// Both emit a `host` frame on the protocol channel; for `wait: true` the
// host answers with a `host_reply` frame on our stdin, routed to the pending
// promise in the stdin loop below. Host frame ids are `<cellId>:<n>` so the
// host can attribute them to the running cell.
let hostCallSeq = 0;
const pendingHostCalls = new Map();

// Per-cell deadline bookkeeping. The host enforces the deadline (it kills
// this process on expiry — JS has no safe in-process interrupt), so the
// runner only tracks it to implement idle-timeout semantics: when the cell
// request carried `idleTimeout: true`, the deadline window is paused while
// the cell awaits a host bridge call and re-armed from the remaining budget
// when the reply arrives.
let cellDeadline = null;

function armDeadline(timeoutMs, idleTimeout) {
  if (typeof timeoutMs !== "number" || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    cellDeadline = null;
    return;
  }
  cellDeadline = { endsAt: Date.now() + timeoutMs, idle: !!idleTimeout, pausedAt: null };
}

function pauseDeadline() {
  if (cellDeadline && cellDeadline.idle && cellDeadline.pausedAt === null) {
    cellDeadline.pausedAt = Date.now();
  }
}

function resumeDeadline() {
  if (cellDeadline && cellDeadline.pausedAt !== null) {
    cellDeadline.endsAt += Date.now() - cellDeadline.pausedAt;
    cellDeadline.pausedAt = null;
  }
}

globalThis.__ompHostCall = (op, args) => {
  if (currentId === null) {
    return Promise.reject(new Error("the host bridge is only available while a cell is running"));
  }
  const id = currentId + ":" + ++hostCallSeq;
  return new Promise((resolve, reject) => {
    pendingHostCalls.set(id, { resolve, reject });
    pauseDeadline();
    emit({ type: "host", id, op, args, wait: true });
  });
};

globalThis.__ompHostNotify = (op, args) => {
  if (currentId === null) return;
  emit({ type: "host", id: currentId + ":" + ++hostCallSeq, op, args, wait: false });
};

// Display bundles are cell output, not host status: they ride the cell's own
// `display` frame (omp worker semantics), so they also work without a bridge.
globalThis.__ompDisplay = (...values) => {
  if (currentId === null) return;
  const bundle = { "text/plain": values.map(formatArg).join(" ") };
  if (values.length === 1 && values[0] !== null && typeof values[0] === "object") {
    try {
      bundle["application/json"] = JSON.parse(JSON.stringify(values[0]));
    } catch {
      // not JSON-safe: text/plain only
    }
  }
  emit({ type: "display", id: currentId, bundle });
};

function settleHostReply(msg) {
  const pending = pendingHostCalls.get(msg.id);
  if (!pending) return false;
  pendingHostCalls.delete(msg.id);
  resumeDeadline();
  if (msg.ok) {
    pending.resolve(msg.result);
  } else {
    pending.reject(new Error(typeof msg.error === "string" ? msg.error : inspect(msg.error)));
  }
  return true;
}

// --- cell transforms --------------------------------------------------------
// Scanner over the cell source tracking brace depth and string/comment state
// so rewrites apply only to top-level constructs.
function scanTopLevel(code) {
  const n = code.length;
  let i = 0;
  let depth = 0;
  // stack of closing contexts: '}' for templates `${...}`
  const templateDepths = [];
  let inLineComment = false;
  let inBlockComment = false;
  let quote = null; // "'", '"', '`'
  let hasTopLevelAwait = false;
  let lastBoundary = 0; // start index of the current top-level statement
  const statementStarts = [0];
  let pendingTopLevelName = null; // identifier awaiting `=>`

  const atWordBoundary = idx =>
    idx <= 0 || !/[A-Za-z0-9_$]/.test(code[idx - 1]);

  while (i < n) {
    const ch = code[i];
    const next = code[i + 1];

    if (inLineComment) {
      if (ch === "\n") inLineComment = false;
      i += 1;
      continue;
    }
    if (inBlockComment) {
      if (ch === "*" && next === "/") {
        inBlockComment = false;
        i += 2;
      } else i += 1;
      continue;
    }
    if (quote !== null) {
      if (ch === "\\") {
        i += 2;
        continue;
      }
      if (quote === "`" && ch === "$" && next === "{") {
        templateDepths.push(depth);
        depth += 1;
        i += 2;
        quote = null;
        continue;
      }
      if (ch === quote) quote = null;
      i += 1;
      continue;
    }
    if (ch === "/" && next === "/") {
      inLineComment = true;
      i += 2;
      continue;
    }
    if (ch === "/" && next === "*") {
      inBlockComment = true;
      i += 2;
      continue;
    }
    if (ch === "'" || ch === '"' || ch === "`") {
      quote = ch;
      i += 1;
      continue;
    }
    if (ch === "{") {
      depth += 1;
      i += 1;
      continue;
    }
    if (ch === "}") {
      if (templateDepths.length > 0 && depth - 1 === templateDepths[templateDepths.length - 1]) {
        templateDepths.pop();
        quote = "`";
        i += 1;
        continue;
      }
      depth -= 1;
      i += 1;
      continue;
    }
    if (ch === "(" || ch === "[") {
      depth += 1;
      i += 1;
      continue;
    }
    if (ch === ")" || ch === "]") {
      depth -= 1;
      i += 1;
      continue;
    }
    // `await` anywhere outside strings/comments (also inside a top-level
    // `try {}`/`if {}` block) needs the async wrapper. An await that only
    // lives in a nested async function is a harmless false positive: the
    // cell body is the same, just run inside an async IIFE.
    if (
      atWordBoundary(i) &&
      code.startsWith("await", i) &&
      !/[A-Za-z0-9_$]/.test(code[i + 5] || " ")
    ) {
      hasTopLevelAwait = true;
    }
    if (depth === 0) {
      if (ch === ";" || ch === "\n") {
        statementStarts.push(i + 1);
        lastBoundary = i + 1;
        pendingTopLevelName = null;
        i += 1;
        continue;
      }
      if (/\s/.test(ch)) {
        i += 1;
        continue;
      }
    }
    i += 1;
  }

  // Locate the last non-empty top-level statement: spans from the final
  // boundary to EOF, trimmed of trailing whitespace/semicolons.
  let stmtStart = code.length;
  while (statementStarts.length > 0) {
    const start = statementStarts[statementStarts.length - 1];
    const tail = code.slice(start).replace(/[\s;]*$/, "");
    if (tail.replace(/\/\/[^\n]*$/s, "").trim() === "") {
      statementStarts.pop();
    } else {
      stmtStart = start;
      break;
    }
  }
  return { hasTopLevelAwait, stmtStart, lastBoundary };
}

// Statement keywords that mean the trailing top-level statement is NOT a
// plain expression (expressions like `new Foo()`, `await x`, `typeof y`,
// `delete a.b` stay capturable, matching omp's returnFinalExpression).
const KEYWORD_STATEMENT =
  /^\s*(?:const|let|var|function|async\s+function|class|if|for|while|do|switch|try|return|throw|import|export|break|continue)\b/;

// Demote top-level declarations to sloppy global assignments and capture the
// trailing expression. Returns { source, finalExpr } where finalExpr is the
// source text of the trailing expression statement (without `;`), or null.
function transformCell(code) {
  const { hasTopLevelAwait, stmtStart } = scanTopLevel(code);
  let body = code;

  const tail = body.slice(stmtStart).replace(/[\s;]*$/, "");
  let finalExpr = null;
  let prefix = body.slice(0, stmtStart);
  if (
    tail !== "" &&
    !KEYWORD_STATEMENT.test(tail) &&
    !/^[!#]/.test(tail) &&
    // no top-level assignment (= not ==, ===, =>, >=, <=, !=)
    !/(^|[^\s=!<>])=(?!=|>)/.test(tail.replace(/\s+/g, " "))
  ) {
    finalExpr = tail;
    body = prefix + "__omp_set_final_expr__((" + tail + "));";
  }

  // Demote top-level let/const/class/function declarations. A declaration is
  // "top-level" when it appears at statement-start at brace depth 0.
  body = body.replace(
    /(^|[;{]\s*|[\n\r])\s*(let|const)\s+/g,
    (m, lead) => lead + " ",
  );
  body = body.replace(
    /(^|[;{]\s*|[\n\r])\s*class\s+([A-Za-z_$][\w$]*)/g,
    (m, lead, name) => lead + " " + name + " = class",
  );
  body = body.replace(
    /(^|[;{]\s*|[\n\r])\s*(async\s+)?function\s+([A-Za-z_$][\w$]*)\s*(\()/g,
    (m, lead, asyncKw, name, paren) =>
      lead + " " + name + " = " + (asyncKw || "") + "function " + name + paren,
  );

  return { source: body, finalExpr, hasTopLevelAwait };
}

// --- cell execution ---------------------------------------------------------
async function runCell(id, code, timeoutMs, idleTimeout) {
  currentId = id;
  if (capture !== null) capture.makeStdoutBlocking();
  executionCount += 1;
  armDeadline(timeoutMs, idleTimeout);
  emit({ type: "started", id });

  let finalExprValue;
  let finalExprSet = false;
  const setFinalExpr = value => {
    finalExprValue = value;
    finalExprSet = true;
  };
  globalThis.__omp_set_final_expr__ = setFinalExpr;

  const { source, hasTopLevelAwait } = transformCell(code);

  // Transactional cell (prelude_txn.js): snapshot own globals before the
  // cell, roll back on failure. Hook absence or errors must never break the
  // cell (the prelude already swallows internally; belt and braces here).
  try {
    if (typeof globalThis.__ompTxnBegin === "function") globalThis.__ompTxnBegin();
  } catch {
    // non-fatal
  }
  let ok = false;
  // The rollback notice must land inside the cell output, i.e. before `done`.
  let txnSettled = false;
  const txnEnd = success => {
    if (txnSettled) return;
    txnSettled = true;
    try {
      if (typeof globalThis.__ompTxnEnd === "function") globalThis.__ompTxnEnd(success);
    } catch {
      // non-fatal
    }
  };

  try {
    // Indirect eval runs in the global scope; the async wrapper keeps cells
    // sloppy so demoted declarations land on globalThis and persist.
    const evaluated = (0, eval)(
      hasTopLevelAwait ? "(async () => {\n" + source + "\n})()" : source,
    );
    if (hasTopLevelAwait) await evaluated;
    ok = true;
    // Children's fd-1 output belongs before the cell's result/done frames.
    syncCapture();
    txnEnd(true);
    if (finalExprSet && finalExprValue !== undefined) {
      emit({ type: "result", id, bundle: { "text/plain": inspect(finalExprValue) } });
    }
    emit({ type: "done", id, status: "ok", executionCount: executionCount, cancelled: false });
  } catch (error) {
    syncCapture();
    const err = error instanceof Error ? error : new Error(typeof error === "string" ? error : inspect(error));
    const stack = (err.stack || String(err.message || err)).split("\n");
    // Filter runner-internal frames (omp filters runner.py frames likewise):
    // keep everything up to the first frame attributed to the runner loop.
    const internal = /at runCell |at processTicksAndRejections|node:events|internal:streams|internalRead|readableAddChunk|at addChunk|native:/;
    const cut = stack.findIndex(line => internal.test(line));
    const traceback = cut >= 0 ? stack.slice(0, cut) : stack;
    emit({
      type: "error",
      id,
      ename: err.name || "Error",
      evalue: err.message || String(err),
      traceback: traceback.length > 0 ? traceback : [String(err.message || err)],
    });
    txnEnd(false); // after the traceback: the error reads first, the rollback note follows
    emit({ type: "done", id, status: "error", executionCount: executionCount, cancelled: false });
  } finally {
    txnEnd(ok); // no-op when already settled above
    currentId = null;
    cellDeadline = null;
    // A cell cannot finish while it still awaits a host reply, but reject
    // any stragglers so their promises never leak into the next cell.
    for (const [hostId, pending] of pendingHostCalls) {
      pendingHostCalls.delete(hostId);
      pending.reject(new Error("the cell finished before the host replied"));
    }
    delete globalThis.__omp_set_final_expr__;
  }
}

// --- main loop --------------------------------------------------------------
let buffer = "";
// Sequential drain chain: every accepted cell appends to `drain` so shutdown
// (exit message or stdin close) waits for all in-flight cells, like the py
// runner's request loop.
let drain = Promise.resolve();
// Host messages arrive on the private control fd (process.stdin when the fd
// isolation above is unavailable).
const controlIn =
  controlFd === 0
    ? process.stdin
    : fs.createReadStream(null, { fd: controlFd, encoding: "utf8", highWaterMark: 65536 });
controlIn.on("data", chunk => {
  buffer += chunk;
  let idx;
  while ((idx = buffer.indexOf("\n")) >= 0) {
    const line = buffer.slice(0, idx);
    buffer = buffer.slice(idx + 1);
    if (line.trim() === "") continue;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      continue; // invalid line: host logs and skips (same as the py runner)
    }
    if (msg && msg.type === "exit") {
      // Drain any in-flight cells before exiting, like the py runner's
      // request-sequential shutdown.
      void drain.then(() => process.exit(0));
    }
    if (msg && msg.type === "host_reply" && typeof msg.id === "string") {
      settleHostReply(msg);
      continue;
    }
    if (msg && typeof msg.id === "string" && typeof msg.code === "string") {
      // Cells are sequential in the real protocol: the Rust host writes one
      // request at a time and waits for `done` before sending the next.
      drain = drain.then(() =>
        runCell(msg.id, msg.code, msg.timeoutMs, msg.idleTimeout).catch(() => {}),
      );

    }
  }
});
const shutdownAfterDrain = () => {
  void drain.then(() => process.exit(0));
};
controlIn.on("end", shutdownAfterDrain);
controlIn.on("error", shutdownAfterDrain);

// --- prelude loading ----------------------------------------------------------
// The host (js.rs) embeds the prelude sources via include_str! and injects
// them as `globalThis.__ompPreludeSources__` (array of { name, source })
// ahead of this script. Loading is silent like the Python runner's startup
// cells; a failing prelude is logged to stderr (fd 2) and never emits
// protocol frames, which would confuse the host's cell loop.
const PRELUDE_SOURCES = globalThis.__ompPreludeSources__ || [];
delete globalThis.__ompPreludeSources__;
for (const prelude of PRELUDE_SOURCES) {
  try {
    (0, eval)(prelude.source);
  } catch (error) {
    forwardToFd2(
      `[omp] failed to load prelude ${prelude.name}: ${error && error.message ? error.message : error}\n`,
    );
  }
}
