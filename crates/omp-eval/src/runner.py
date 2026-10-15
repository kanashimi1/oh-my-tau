"""omp-rs Python runner — subprocess wrapper driven by the Rust host.

NDJSON protocol over stdin/stdout, wire-compatible with the omp coding-agent
host (pi-coding-agent/src/eval/py/runner.py, protocol section):

Host -> wrapper:
  {"id": str, "code": str, "silent": bool?, "storeHistory": bool?,
   "timeoutMs": int?, "idleTimeout": bool?}
  {"type": "exit"}                     # graceful shutdown
  {"type": "host_reply", "id": str, "ok": bool, "result"?/ "error"?}
                                       # only for host frames with wait:true

Wrapper -> host:
  {"type": "started", "id": ...}
  {"type": "stdout",  "id": ..., "data": str}
  {"type": "stderr",  "id": ..., "data": str}
  {"type": "display", "id": ..., "bundle": {<mime>: <value>}}
  {"type": "result",  "id": ..., "bundle": {<mime>: <value>}}
  {"type": "error",   "id": ..., "ename": str, "evalue": str, "traceback": [str]}
  {"type": "done",    "id": ..., "status": "ok"|"error",
                       "executionCount": int, "cancelled": bool}
  {"type": "host",     "id": "<cellId>:<n>", "op": str, "args": {...},
                       "wait": bool}     # cell-side bridge request; wait:true
                                       # blocks the cell on a host_reply line

Cell requests may set "idleTimeout": true to make the timeout count only time
spent outside host bridge calls: the runner pauses its SIGALRM (and the
escalation watchdog) for the duration of each `__omp_host_call` and re-arms a
fresh full window when the call returns (TS idle-timeout.ts).

Frames travel on a private, non-inheritable dup of the original stdout and
host requests/replies arrive on a private dup of the original stdin (TS
py/runner.py fd isolation). fd 1 itself is a capture pipe whose bytes (from
`os.write(1, …)`, `sys.__stdout__` or any subprocess that inherits fd 1) are
forwarded as plain `stdout` frames, and fd 0 is /dev/null: nothing a cell
spawns can write frames or read host replies. User stdout/stderr are
redirected into typed frames by line-buffered proxies; every frame written
from the runner first drains the capture pipe, so interleaving is preserved.

Timeouts are enforced inside the runner (the Rust workspace forbids unsafe,
so no kill(2) from the host): a SIGALRM raises `KeyboardInterrupt` into the
running cell — the same observable behavior as omp's host-side SIGINT — and a
watchdog thread force-exits the process (emitting a final `done` frame with
`cancelled: true` first) if the cell ignores the interrupt for longer than the
omp escalation window. Self-contained: stdlib only, no IPython. Cell magics
are NOT translated (minimal port); use plain Python.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
import signal
import sys
import threading
import time
import traceback

# --- protocol channel isolation (TS py/runner.py) ---------------------------
# Frames go to a private dup of the original stdout (os.dup: non-inheritable,
# PEP 446); fd 1 is repointed at a capture pipe drained into stdout frames, so
# children spawned by cells cannot write raw bytes into the NDJSON channel.
_RAW_STDERR = sys.__stderr__
try:
    _FRAME_FD = os.dup(1)
    try:
        _CAPTURE_READ_FD, _capture_write_fd = os.pipe()
        os.dup2(_capture_write_fd, 1)
        os.close(_capture_write_fd)
        os.set_blocking(_CAPTURE_READ_FD, False)
    except OSError:
        _CAPTURE_READ_FD = None
except OSError:
    _FRAME_FD = 1
    _CAPTURE_READ_FD = None
# Host requests and host_reply lines likewise come from a private dup of the
# original stdin; fd 0 becomes /dev/null so cells and their children read EOF
# instead of blocking on (or stealing replies from) the control channel.
try:
    _CONTROL_IN = os.fdopen(os.dup(0), "r", encoding="utf-8", errors="replace")
    _devnull_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(_devnull_fd, 0)
    os.close(_devnull_fd)
except OSError:
    _CONTROL_IN = sys.stdin
_OUT_LOCK = threading.Lock()
_CAPTURE_LOCK = threading.Lock()
_CAPTURE_DECODER = None
# Set when a frame write was interrupted half-way: the next frame starts with
# a newline so the host skips only the torn line.
_FRAME_TORN = False

# With `python -c SCRIPT` (the host spawn mode) there is no __file__; the
# runner's own frames then report filename "<string>".
_RUNNER_FILE = globals().get("__file__", "<string>")
_USER_NS: dict = {"__name__": "__main__"}
_EXECUTION_COUNT = 0
_CURRENT_RID: str | None = None

# Host bridge state: monotonically numbered per cell so one cell can pipeline
# several notify frames; `wait` calls read their one host_reply from stdin.
_HOST_SEQ = 0
# True while the current cell requested idle-timeout semantics (host request
# field "idleTimeout"): the SIGALRM is paused for the duration of each host
# call and re-armed with a fresh full window when the cell resumes.
_IDLE_TIMEOUT_CELL = False
_HOST_CALL_DEPTH = 0
_CURRENT_TIMEOUT_MS: float | None = None
_WATCHDOG: threading.Timer | None = None

# omp: INTERRUPT_ESCALATION_MS = 5_000
ESCALATION_SECONDS = 5.0

# Process-global (not thread-local): the watchdog thread must observe the
# main thread's completion. The GIL makes plain bool reads/writes safe here.
_CELL_DONE = True


# Parent watchdog: when the host process dies (SIGKILL included), this
# runner must exit promptly even while a cell is busy — an orphaned kernel
# keeps its interpreter, memory and any busy-loop CPU forever otherwise.
_PARENT_PID = os.getppid()


def _watch_parent() -> None:
    # os.getppid() changes to 1 (or a subreaper) once the parent is reaped;
    # time.sleep releases the GIL, so a busy cell cannot delay detection.
    while os.getppid() == _PARENT_PID:
        time.sleep(0.2)
    os._exit(0)


def _start_parent_watchdog() -> None:
    threading.Thread(
        target=_watch_parent, name="omp-parent-watchdog", daemon=True
    ).start()


def _write_frame(frame: dict) -> None:
    """Write one NDJSON frame to the private frame fd (thread-safe)."""
    global _FRAME_TORN
    data = (json.dumps(frame, ensure_ascii=False, default=str) + "\n").encode("utf-8", "replace")
    with _OUT_LOCK:
        if _FRAME_TORN:
            data = b"\n" + data
        view = memoryview(data)
        try:
            while view:
                view = view[os.write(_FRAME_FD, view):]
        except BaseException:
            if len(view) != len(data):
                _FRAME_TORN = True
            raise
        _FRAME_TORN = False


def _drain_capture_locked() -> bool:
    """Forward bytes pending in the fd-1 capture pipe; caller holds _CAPTURE_LOCK.

    Returns False once the pipe reached EOF or broke.
    """
    global _CAPTURE_DECODER
    if _CAPTURE_DECODER is None:
        import codecs

        _CAPTURE_DECODER = codecs.getincrementaldecoder("utf-8")("replace")
    while True:
        try:
            chunk = os.read(_CAPTURE_READ_FD, 65536)
        except BlockingIOError:
            return True
        except OSError:
            return False
        if not chunk:
            return False
        text = _CAPTURE_DECODER.decode(chunk)
        if not text:
            continue
        rid = _CURRENT_RID
        if rid is None:
            # Output from a child outliving its cell: no frame to attach to (TS).
            _RAW_STDERR.write(text)
            _RAW_STDERR.flush()
        else:
            _write_frame({"type": "stdout", "id": rid, "data": text})


def _sync_capture() -> None:
    """Flush captured fd-1 output ahead of the next runner frame (ordering)."""
    if _CAPTURE_READ_FD is None:
        return
    if _CAPTURE_LOCK.acquire(timeout=1.0):
        try:
            _drain_capture_locked()
        finally:
            _CAPTURE_LOCK.release()


def _capture_drain_loop() -> None:
    """Background drain so children writing to fd 1 never block on a full pipe."""
    import select

    poller = select.poll()
    poller.register(_CAPTURE_READ_FD, select.POLLIN)
    while True:
        try:
            poller.poll()
        except OSError:
            return
        with _CAPTURE_LOCK:
            if not _drain_capture_locked():
                return


def _start_capture_drain() -> None:
    if _CAPTURE_READ_FD is None:
        return
    threading.Thread(
        target=_capture_drain_loop, name="omp-fd1-capture", daemon=True
    ).start()


def _emit(frame: dict) -> None:
    """Serialize one frame as a single NDJSON line on the protocol channel,
    after any fd-1 output captured before it."""
    _sync_capture()
    _write_frame(frame)


class _StreamProxy:
    """Replaces sys.stdout/sys.stderr during a cell; emits line frames."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._buf = ""

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit_line(line + "\n")
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._emit_line(self._buf)
            self._buf = ""

    def _emit_line(self, text: str) -> None:
        if _CURRENT_RID is not None:
            _emit({"type": self._kind, "id": _CURRENT_RID, "data": text})


def display(obj: object, **reprs: str) -> None:
    """Prelude helper: rich-ish display of an object (repr + JSON when safe)."""
    bundle: dict = {"text/plain": reprs.pop("text/plain", None) or repr(obj)}
    try:
        json.dumps(obj)
        bundle["application/json"] = obj
    except (TypeError, ValueError):
        pass
    if reprs:
        bundle.update(reprs)
    _emit({"type": "display", "id": _CURRENT_RID, "bundle": bundle})


_USER_NS["display"] = display


def __omp_display(value: object, *, raw: bool = False, kind: str = "display") -> None:
    """Runner-injected display primitive (TS runner `__ompDisplay`).

    `raw=True` takes a mime bundle dict verbatim (status/image bundles are
    emitted as display frames of the given `kind`, never as host notifies).
    Otherwise a bundle is built from repr/json like the builtin display.
    """
    if raw:
        bundle = value
    else:
        bundle = {"text/plain": repr(value)}
        try:
            json.dumps(value)
            bundle["application/json"] = value
        except (TypeError, ValueError):
            pass
    _emit({"type": kind, "id": _CURRENT_RID, "bundle": bundle})


class OmpHostError(RuntimeError):
    """Raised in a cell when a host bridge call fails (TS OmpHostError)."""


def _pause_cell_alarm() -> None:
    """Suspend the cell's SIGALRM and escalation watchdog (TS IdleTimeout.pause)."""
    global _WATCHDOG
    if hasattr(signal, "setitimer"):
        signal.setitimer(signal.ITIMER_REAL, 0)
    if _WATCHDOG is not None:
        _WATCHDOG.cancel()
        _WATCHDOG = None


def _resume_cell_alarm(rid: str) -> None:
    """Re-arm with a fresh full window after a host call (TS IdleTimeout.resume)."""
    global _WATCHDOG
    if _CURRENT_TIMEOUT_MS and hasattr(signal, "setitimer"):
        seconds = _CURRENT_TIMEOUT_MS / 1000.0
        signal.setitimer(signal.ITIMER_REAL, seconds)
        _WATCHDOG = threading.Timer(
            seconds + ESCALATION_SECONDS, _escalate_kill, args=(rid,)
        )
        _WATCHDOG.daemon = True
        _WATCHDOG.start()


def __omp_host_call(op: str, args: dict) -> object:
    """Blocking host bridge request; raises OmpHostError on host failure."""
    global _HOST_SEQ, _HOST_CALL_DEPTH
    rid = _CURRENT_RID
    if rid is None:
        raise OmpHostError("host bridge is only available while a cell is running")
    _HOST_SEQ += 1
    hid = f"{rid}:{_HOST_SEQ}"
    outermost = _HOST_CALL_DEPTH == 0
    _HOST_CALL_DEPTH += 1
    if outermost and _IDLE_TIMEOUT_CELL:
        _pause_cell_alarm()
    try:
        _emit({"type": "host", "id": hid, "op": op, "args": args, "wait": True})
        # The host never sends a new cell while one runs, so the reply line on
        # the private control channel is unambiguously ours.
        line = _CONTROL_IN.readline()
        try:
            reply = json.loads(line)
        except ValueError as exc:
            raise OmpHostError(f"malformed host reply: {line!r}") from exc
        if reply.get("id") != hid or reply.get("type") != "host_reply":
            raise OmpHostError(f"unexpected host reply: {reply!r}")
        if reply.get("ok"):
            return reply.get("result")
        raise OmpHostError(str(reply.get("error", "host call failed")))
    finally:
        _HOST_CALL_DEPTH -= 1
        if outermost and _IDLE_TIMEOUT_CELL:
            _resume_cell_alarm(rid)


def __omp_host_notify(op: str, args: dict) -> None:
    """Fire-and-forget host bridge status (log/phase/progress style)."""
    global _HOST_SEQ
    rid = _CURRENT_RID
    if rid is None:
        return
    _HOST_SEQ += 1
    _emit(
        {
            "type": "host",
            "id": f"{rid}:{_HOST_SEQ}",
            "op": op,
            "args": args,
            "wait": False,
        }
    )


_USER_NS["__omp_display"] = __omp_display
_USER_NS["__omp_host_call"] = __omp_host_call
_USER_NS["__omp_host_notify"] = __omp_host_notify
_USER_NS["OmpHostError"] = OmpHostError


class _CellTimeout(BaseException):
    """Internal interrupt marker (caught by the cell runner, never user code)."""


def _emit_done(rid: str, status: str, cancelled: bool) -> None:
    _emit(
        {
            "type": "done",
            "id": rid,
            "status": status,
            "executionCount": _EXECUTION_COUNT,
            "cancelled": cancelled,
        }
    )


def _arm_timeout(rid: str, timeout_ms: float | None) -> None:
    global _CELL_DONE, _CURRENT_TIMEOUT_MS, _WATCHDOG
    _CELL_DONE = False
    _CURRENT_TIMEOUT_MS = timeout_ms
    has_alarm = hasattr(signal, "setitimer")
    if timeout_ms is not None and timeout_ms > 0 and has_alarm:
        seconds = timeout_ms / 1000.0
        signal.signal(signal.SIGALRM, _sigalrm_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        # Watchdog: if the cell ignored the alarm (C extension holding the
        # GIL), settle the request and hard-exit so the host unblocks, like
        # omp's kernel kill + recreate path. Kept in a global so host-call
        # pause/resume can cancel and re-create it (idle-timeout cells).
        _WATCHDOG = threading.Timer(seconds + ESCALATION_SECONDS, _escalate_kill, args=(rid,))
        _WATCHDOG.daemon = True
        _WATCHDOG.start()
    elif has_alarm:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _disarm_timeout() -> None:
    global _CELL_DONE, _CURRENT_TIMEOUT_MS, _WATCHDOG
    _CELL_DONE = True
    _CURRENT_TIMEOUT_MS = None
    if _WATCHDOG is not None:
        _WATCHDOG.cancel()
        _WATCHDOG = None
    if hasattr(signal, "setitimer"):
        signal.setitimer(signal.ITIMER_REAL, 0)


def _sigalrm_handler(signum: int, frame: object) -> None:  # noqa: ARG001
    if _CELL_DONE:
        return
    raise _CellTimeout()


def _escalate_kill(rid: str) -> None:
    if _CELL_DONE:
        return
    try:
        _emit_done(rid, "error", True)
    except Exception:  # noqa: BLE001 - best effort before a hard exit
        pass
    os._exit(137)


def _emit_error(rid: str, exc: BaseException) -> None:
    """Match omp runner error framing: caret-only for cell syntax errors,
    runner frames filtered out of user tracebacks, lines rstripped."""
    if isinstance(exc, SyntaxError):
        tb_lines = [line.rstrip("\n") for line in traceback.format_exception_only(type(exc), exc)]
    else:
        formatted = traceback.TracebackException.from_exception(exc)
        user_frames = [fr for fr in formatted.stack if fr.filename != _RUNNER_FILE]
        if user_frames:
            formatted.stack = traceback.StackSummary.from_list(user_frames)
        tb_lines = [line.rstrip("\n") for line in formatted.format()]
    _emit(
        {
            "type": "error",
            "id": rid,
            "ename": type(exc).__name__,
            "evalue": str(exc),
            "traceback": tb_lines,
        }
    )


def _run_cell(
    rid: str,
    code: str,
    timeout_ms: float | None,
    filename: str = "<cell>",
    store_history: bool = True,
    idle_timeout: bool = False,
) -> None:
    global _EXECUTION_COUNT, _CURRENT_RID, _IDLE_TIMEOUT_CELL
    if store_history:
        _EXECUTION_COUNT += 1
    _CURRENT_RID = rid
    _IDLE_TIMEOUT_CELL = idle_timeout
    status, cancelled = "ok", False
    _emit({"type": "started", "id": rid})
    # TS resets the prelude's per-cell call accounting before every user cell.
    reset_occ = _USER_NS.get("__omp_reset_call_occurrences__")
    if callable(reset_occ):
        try:
            reset_occ()
        except Exception as exc:  # noqa: BLE001 - hooks never break the cell
            sys.__stderr__.write(f"txn hook error: {exc!r}\n")
    # Transactional cells: snapshot names before, commit/rollback after.
    txn_begin = _USER_NS.get("__omp_txn_begin") if store_history else None
    txn_end = _USER_NS.get("__omp_txn_end") if store_history else None
    if callable(txn_begin):
        try:
            txn_begin()
        except Exception as exc:  # noqa: BLE001 - hooks never break the cell
            sys.__stderr__.write(f"txn hook error: {exc!r}\n")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _StreamProxy("stdout"), _StreamProxy("stderr")
    _arm_timeout(rid, timeout_ms)
    try:
        tree = ast.parse(code, filename=filename, mode="exec")
        tail = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tail = tree.body.pop()
            if not tree.body:
                tree.body = [ast.fix_missing_locations(ast.Pass())]
        # PyCF_ALLOW_TOP_LEVEL_AWAIT: cells may `await` at the top level, like
        # IPython/Jupyter; exec/eval then return a coroutine we drive to completion.
        flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        value: object = None
        if tree.body:
            co = compile(tree, filename, "exec", flags=flags, dont_inherit=True)  # noqa: S102 - by design
            is_async = bool(co.co_flags & (inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR))
            # eval() on an exec-mode code object is the IPython trick that yields
            # the module coroutine for top-level-await cells (exec returns None)
            body_value = eval(co, _USER_NS) if is_async else None  # noqa: S307 - by design
            if not is_async:
                exec(co, _USER_NS)  # noqa: S102 - by design
            elif body_value is not None:
                value = asyncio.run(body_value)
        if tail is not None:
            co_tail = compile(ast.Expression(tail.value), filename, "eval", flags=flags, dont_inherit=True)
            value = eval(co_tail, _USER_NS)  # noqa: S307 - by design
            if inspect.isawaitable(value):
                value = asyncio.run(value)
        if value is not None:
            _emit({"type": "result", "id": rid, "bundle": {"text/plain": repr(value)}})
    except _CellTimeout:
        cancelled, status = True, "error"
        # Like omp: SIGINT into a running cell reports KeyboardInterrupt("Execution interrupted")
        _emit_error(rid, KeyboardInterrupt("Execution interrupted"))
    except KeyboardInterrupt:
        cancelled, status = True, "error"
        _emit_error(rid, KeyboardInterrupt("Execution interrupted"))
    except BaseException as exc:  # noqa: BLE001 - surface every user error, like omp
        status = "error"
        _emit_error(rid, exc)
    finally:
        if callable(txn_end):
            # ok=False when the cell raised or was cancelled/timed out
            try:
                txn_end(status == "ok" and not cancelled)
            except Exception as exc:  # noqa: BLE001 - hooks never break the cell
                sys.__stderr__.write(f"txn hook error: {exc!r}\n")
        _disarm_timeout()
        _IDLE_TIMEOUT_CELL = False
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = old_out, old_err
        # Children of this cell wrote their fd-1 bytes before exiting; attach
        # them to this cell while its id is still current.
        _sync_capture()
        _CURRENT_RID = None
        _emit_done(rid, status, cancelled)


def main() -> None:
    _start_parent_watchdog()
    _start_capture_drain()
    for line in _CONTROL_IN:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if req.get("type") == "exit":
            break
        code = req.get("code")
        if not isinstance(code, str):
            continue
        rid = str(req.get("id", ""))
        timeout_ms = req.get("timeoutMs")
        timeout = float(timeout_ms) if isinstance(timeout_ms, (int, float)) and timeout_ms > 0 else None
        filename = req.get("filename") if isinstance(req.get("filename"), str) else "<cell>"
        store_history = req.get("storeHistory") is not False
        idle_timeout = req.get("idleTimeout") is True
        _run_cell(rid, code, timeout, filename, store_history, idle_timeout)


if __name__ == "__main__":
    main()
