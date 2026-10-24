"""Drive a full-screen terminal app in a real PTY and read its screen.

A pyte virtual terminal (same size as the PTY) consumes everything the app
writes, so `screen()` is what a user would see. Used by the omp vs omp-rs
end-to-end comparison (`run.py`).
"""
from __future__ import annotations

import fcntl
import os
import pty
import re
import select
import signal
import struct
import subprocess
import termios
import copy
import time

import pyte

KEYS = {
    "enter": b"\r",
    "esc": b"\x1b",
    "ctrl-c": b"\x03",
    "ctrl-d": b"\x04",
    "tab": b"\t",
    "up": b"\x1b[A",
    "down": b"\x1b[B",
    "right": b"\x1b[C",
    "ctrl-u": b"\x15",
    "backspace": b"\x7f",
    "ctrl-shift-v": b"\x1b[118;6u",
}


class AltScreen(pyte.HistoryScreen):
    """pyte screen with xterm's DEC 1049 alternate buffer (pyte ignores it):
    `?1049h` saves the normal grid + cursor and clears, `?1049l` restores
    them, so a fullscreen overlay closing reveals the untouched chat. Also
    tracks the mouse-reporting modes (1000/1003/1006) the app has on."""

    MOUSE_MODES = (1000, 1003, 1006)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alt_active = False
        self.mouse_modes: set[int] = set()
        self._saved = None

    def set_mode(self, *modes, **kwargs):
        if kwargs.get("private"):
            if 1049 in modes and not self.alt_active:
                self._saved = (copy.deepcopy(self.buffer), copy.copy(self.cursor))
                self.alt_active = True
                self.erase_in_display(2)
            self.mouse_modes.update(m for m in modes if m in self.MOUSE_MODES)
            modes = tuple(m for m in modes if m != 1049)
        if modes:
            super().set_mode(*modes, **kwargs)

    def reset_mode(self, *modes, **kwargs):
        if kwargs.get("private"):
            if 1049 in modes and self.alt_active:
                buffer, cursor = self._saved
                self.buffer.clear()
                self.buffer.update(buffer)
                self.cursor = cursor
                self.dirty.update(range(self.lines))
                self.alt_active = False
                self._saved = None
            self.mouse_modes.difference_update(modes)
            modes = tuple(m for m in modes if m != 1049)
        if modes:
            super().reset_mode(*modes, **kwargs)


def sgr_click(col: int, row: int) -> bytes:
    """Left press + release at 1-based screen cell (col, row), SGR encoded."""
    return f"\x1b[<0;{col};{row}M\x1b[<0;{col};{row}m".encode()


def sgr_wheel(col: int, row: int, down: bool = True) -> bytes:
    """One vertical wheel notch at 1-based (col, row)."""
    return f"\x1b[<{65 if down else 64};{col};{row}M".encode()


class Term:
    def __init__(self, argv: list[str], *, env: dict[str, str], cwd: str, rows: int = 40, cols: int = 120, kitty: bool = False):
        self.rows, self.cols = rows, cols
        self.kitty = kitty
        self.screen_ = AltScreen(cols, rows, history=5000)
        self.screen_.set_mode(pyte.modes.LNM)
        self.stream = pyte.ByteStream(self.screen_)
        self.raw = bytearray()
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        self.proc = subprocess.Popen(
            argv, stdin=slave, stdout=slave, stderr=slave, cwd=cwd, env=env, start_new_session=True
        )
        os.close(slave)
        self.master = master

    # -- io -------------------------------------------------------------
    def pump(self, seconds: float) -> None:
        end = time.time() + seconds
        while True:
            left = end - time.time()
            if left <= 0:
                return
            r, _, _ = select.select([self.master], [], [], min(left, 0.05))
            if not r:
                if self.proc.poll() is not None:
                    return
                continue
            try:
                data = os.read(self.master, 65536)
            except OSError:
                return
            if not data:
                return
            self.raw += data
            self._answer_queries(data)
            self.stream.feed(data)

    def _answer_queries(self, data: bytes) -> None:
        # Terminal capability probes a real terminal would answer; without a
        # reply some TUIs wait for a timeout at startup.
        if b"\x1b[6n" in data:  # cursor position report
            y, x = self.screen_.cursor.y + 1, self.screen_.cursor.x + 1
            os.write(self.master, f"\x1b[{y};{x}R".encode())
        if b"\x1b]11;?" in data:  # background colour
            os.write(self.master, b"\x1b]11;rgb:1e1e/1e1e/1e1e\x1b\\")
        if b"\x1b[c" in data or b"\x1b[0c" in data:  # primary device attributes
            os.write(self.master, b"\x1b[?62;22c")
        if b"\x1b[?u" in data:  # kitty keyboard protocol query
            if self.kitty:
                os.write(self.master, b"\x1b[?0u")  # speaks kitty, fresh stack

    def send(self, text: str | bytes) -> None:
        os.write(self.master, text.encode() if isinstance(text, str) else text)

    def key(self, name: str) -> None:
        self.send(KEYS[name])

    def type(self, text: str, *, per_char: float = 0.0) -> None:
        if per_char <= 0:
            self.send(text)
            return
        for ch in text:
            self.send(ch)
            self.pump(per_char)

    # -- screen ---------------------------------------------------------
    def screen(self) -> list[str]:
        return [line.rstrip() for line in self.screen_.display]

    def text(self) -> str:
        return "\n".join(self.screen())

    def transcript(self) -> str:
        """Scrollback + visible screen (inline TUIs push history up)."""
        hist = ["".join(ch.data for ch in (row[x] for x in range(self.cols))).rstrip()
                for row in self.screen_.history.top]
        return "\n".join(hist + self.screen())

    def wait_for(self, pattern: str, timeout: float, *, in_transcript: bool = True) -> bool:
        rx = re.compile(pattern)
        end = time.time() + timeout
        while time.time() < end:
            self.pump(0.1)
            if rx.search(self.transcript() if in_transcript else self.text()):
                return True
            if self.proc.poll() is not None:
                return bool(rx.search(self.transcript()))
        return False

    def settle(self, quiet: float = 0.6, timeout: float = 15.0) -> None:
        """Pump until the app has written nothing for `quiet` seconds."""
        end = time.time() + timeout
        last = len(self.raw)
        quiet_since = time.time()
        while time.time() < end:
            self.pump(0.05)
            if len(self.raw) != last:
                last = len(self.raw)
                quiet_since = time.time()
            elif time.time() - quiet_since >= quiet:
                return

    def close(self) -> int | None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        try:
            os.close(self.master)
        except OSError:
            pass
        return self.proc.returncode
