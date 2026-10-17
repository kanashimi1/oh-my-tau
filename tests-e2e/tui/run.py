#!/usr/bin/env python3
"""End-to-end TUI parity: the same scripted scenarios against TS `omp` and
`omp-rs`, each in a real PTY with an isolated HOME and a local mock provider.

Every scenario states what the user must see. A check counts against omp-rs
when it fails there; when it also fails on omp the harness itself is suspect
and the run reports it as `baseline` instead of a parity failure. Screens of
both clients are written side by side for a visual comparison.

usage: run.py [--rs BIN] [--ts BIN] [--only NAME[,NAME]] [--out DIR] [--client rs|ts|both]
exit 0 = omp-rs passes every check omp passes.
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from driver import KEYS, Term, sgr_click, sgr_wheel  # noqa: E402
import mock_provider  # noqa: E402

REPO = HERE.parent.parent
DEFAULT_RS = REPO / "target/shared/release/omp-rs"


# ── isolated client home ────────────────────────────────────────────────────
def make_home(root: Path, port: int, extra_config: str = "") -> dict[str, str]:
    agent = root / ".omp" / "agent"
    (agent / "sessions").mkdir(parents=True)
    (agent / "models.yml").write_text(
        "providers:\n"
        "  mock:\n"
        f"    baseUrl: http://127.0.0.1:{port}/v1\n"
        "    apiKey: e2e-not-a-secret\n"
        "    api: openai-completions\n"
        "    models:\n"
        + "".join(f"      - id: {m}\n        name: e2e {m}\n        contextWindow: 128000\n        maxTokens: 4096\n"
                  + ("        reasoning: true\n" if m == "think" else "")
                  for m in ("text", "tool", "tools", "fail", "think"))
    )
    # No onboarding wizard / update probe / changelog: they would cover the
    # chat screen on a fresh HOME (TS omp; omp-rs ignores unknown keys).
    (agent / "config.yml").write_text(
        "startup:\n  setupWizard: false\n  checkUpdate: false\n  changelogMode: hidden\n" + extra_config
    )
    work = root / "work"
    work.mkdir()
    (work / "e2e-read-me.txt").write_text("E2E-FILE-CONTENT\n")
    keep = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "USER", "SHELL", "TMPDIR")}
    return {
        **keep,
        "HOME": str(root),
        "PI_CODING_AGENT_DIR": str(agent),
        "OMP_AGENT_DIR": str(agent),
        "TERM": "xterm-256color",
        "COLORTERM": "truecolor",
        "PI_NO_DESKTOP_NOTIFY": "1",
        "OMP_NO_UPDATE_CHECK": "1",
        "PI_SKIP_VERSION_CHECK": "1",
        "E2E_WORK": str(work),
    }


# ── scenarios ───────────────────────────────────────────────────────────────
@dataclass
class Result:
    checks: dict[str, bool] = field(default_factory=dict)
    screens: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool) -> bool:
        self.checks[name] = bool(ok)
        return ok


def submit(t: Term, text: str) -> None:
    t.type(text, per_char=0.005)
    t.pump(0.2)
    t.key("enter")


USAGE_RE = r"⏱|tok/s|\d+(\.\d+)?[kKM]? tok"
ERROR_RE = r"(?i)\berror\b|prompt failed"
# Status line both clients draw once the chat is interactive: `π > ⬢ <model> > …`.
READY_RE = r"π > ⬢"


def ready(t: Term, timeout: float = 60) -> bool:
    """Wait until the chat is interactive (TS shows a splash first)."""
    ok = t.wait_for(READY_RE, timeout, in_transcript=False)
    t.settle(1.0, 15)
    return ok


def error_headers(text: str, needle: str) -> int:
    """Lines that present `needle` as an error (✗/✘ glyph or `Error:`),
    ignoring detail lines that merely repeat the provider text."""
    return sum(1 for line in text.splitlines()
               if needle in line and re.search(r"[✗✘]|(?i:error:)", line))


def sc_startup(t: Term, r: Result) -> None:
    r.check("startup: chat ready", ready(t))
    r.screens["startup"] = t.text()
    r.check("startup: model shown", "mock/text" in t.text() or "e2e text" in t.text())
    r.check("startup: no error", re.search(ERROR_RE, t.text()) is None)


def sc_two_prompts(t: Term, r: Result) -> None:
    ready(t)
    submit(t, "first e2e prompt")
    r.check("prompt1: reply shown", t.wait_for(r"E2E-REPLY-1", 90))
    t.wait_for(READY_RE, 30, in_transcript=False)
    t.settle(1.0, 30)
    r.screens["after-prompt-1"] = t.transcript()
    r.check("prompt1: no error", re.search(ERROR_RE, t.transcript()) is None)
    submit(t, "second e2e prompt")
    r.check("prompt2: reply shown", t.wait_for(r"E2E-REPLY-2", 60))
    t.settle(1.0, 30)
    tr = t.transcript()
    r.screens["after-prompt-2"] = tr
    r.check("prompt2: no error", re.search(ERROR_RE, tr) is None)
    r.check("prompts: both user messages kept", "first e2e prompt" in tr and "second e2e prompt" in tr)
    r.notes.append(f"usage rows: {len(re.findall(USAGE_RE, tr))}")
    r.check("prompts: no usage row (showTokenUsage off)", re.search(USAGE_RE, tr) is None)


def sc_tool(t: Term, r: Result) -> None:
    ready(t)
    submit(t, "read the file")
    r.check("tool: final reply", t.wait_for(r"E2E-TOOL-DONE", 60))
    t.settle(1.0, 30)
    tr = t.transcript()
    r.screens["tool-turn"] = tr
    r.check("tool: read card names the file", "e2e-read-me.txt" in tr)
    r.check("tool: no error", re.search(ERROR_RE, tr) is None)


def sc_fail(t: Term, r: Result) -> None:
    ready(t)
    submit(t, "this will fail")
    r.check("fail: provider message shown", t.wait_for(r"E2E-PROVIDER-FAILURE", 60))
    t.settle(1.5, 30)
    tr = t.transcript()
    r.screens["provider-error"] = tr
    r.check("fail: shown once", error_headers(tr, "E2E-PROVIDER-FAILURE") == 1)
    r.check("fail: pinned banner shown", DISMISS_HINT in t.text())
    submit(t, "after failure")
    r.check("fail: next prompt accepted", t.wait_for(r"after failure", 20))
    r.check("fail: next reply after failure", t.wait_for(r"E2E-REPLY-2", 60))
    t.settle(1.5, 30)
    tr = t.transcript()
    r.screens["after-dismiss"] = tr
    r.check("fail: banner dismissed on next submit", DISMISS_HINT not in t.text())
    r.check("fail: error kept in history", "E2E-PROVIDER-FAILURE" in tr.split("after failure")[0])


DISMISS_HINT = "Dismissed when you send your next message."


def sc_usage(t: Term, r: Result) -> None:
    """display.showTokenUsage on: a metric row per billed assistant message;
    the read-only tool turn's row attaches to the read group."""
    ready(t)
    submit(t, "read the file")
    r.check("usage: final reply", t.wait_for(r"E2E-TOOL-DONE", 60))
    t.settle(1.5, 30)
    tr = t.transcript()
    r.screens["usage-turn"] = tr
    lines = [line for line in tr.splitlines() if line.strip()]
    is_row = lambda line: re.search(r"⤵ ?100\s+⤴ ?7\b", line) is not None
    rows = [line for line in lines if is_row(line)]
    r.notes.append(f"usage rows: {len(rows)}")
    r.check("usage: one row per billed message", len(rows) == 2)
    read_at = next((i for i, line in enumerate(lines) if "e2e-read-me.txt" in line), None)
    r.check("usage: read turn row attached under the read card",
            read_at is not None and read_at + 1 < len(lines) and is_row(lines[read_at + 1])
            and lines[read_at + 1].startswith("   "))
    r.check("usage: no error", re.search(ERROR_RE, tr) is None)


def sc_tools(t: Term, r: Result) -> None:
    """bash/write/grep/glob cards: framed bash/write cards, grep/glob title rows + result rows."""
    ready(t)
    submit(t, "use the tools")
    r.check("tools: final reply", t.wait_for(r"E2E-TOOLS-DONE", 90))
    t.settle(1.5, 30)
    tr = t.transcript()
    r.screens["tools-turn"] = tr
    has = lambda pat: re.search(pat, tr, re.M) is not None
    r.check("tools: Bash card command row", has(r"^│ \$ echo E2E-BASH-OUT\s+│$"))
    r.check("tools: Bash card Output section bar", has(r"^├─── Output ─+┤$"))
    r.check("tools: Bash card wall/timeout line", has(r"^│ ⟦Wall: \d+\.\d\ds \| Timeout: 300s⟧\s+│$"))
    r.check("tools: Bash card no model-facing wall notice", not has(r"Wall time: [\d.]+ seconds"))
    r.check("tools: Write card title row", has(r"^╭─── ✎ Write: 🗒 e2e-new\.txt · 3 lines ─+╮$"))
    r.check("tools: Write card content rows", has(r"^│ +1 alpha\s+│$") and has(r"^│ +2 beta\s+│$"))
    r.check("tools: Grep title row", has(r"^ 🔍 Grep: beta 1 match · 1 file · in \.\s*$"))
    r.check("tools: Grep hashline result rows", has(r"^ └─ # e2e-new\.txt#[0-9a-f]+\s*$") and has(r"^\s+\*2│beta\s*$"))
    r.check("tools: Glob title row", has(r"^ 🔍 Glob: \*\.txt 2 files · in \.\s*$"))
    r.check("tools: Glob file rows", has(r"^ ├─ 🗒 e2e-new\.txt\s*$") and has(r"^ └─ 🗒 e2e-read-me\.txt\s*$"))
    r.check("tools: no tool error", re.search(ERROR_RE, tr) is None)


def sc_slash(t: Term, r: Result) -> None:
    ready(t)
    before = t.transcript()
    submit(t, "/session")
    t.settle(1.0, 15)
    tr = t.transcript()
    r.screens["slash-session"] = tr
    added = tr[len(before):] if tr.startswith(before) else tr
    r.check("slash: /session prints session info", re.search(r"(?i)session", added) is not None)
    r.check("slash: no error", re.search(r"(?i)unknown command|not ported|error", added) is None)


def sc_exit(t: Term, r: Result) -> None:
    ready(t)
    t.key("ctrl-c")
    t.pump(0.2)
    t.key("ctrl-c")
    deadline = time.time() + 15
    while time.time() < deadline and t.proc.poll() is None:
        t.pump(0.1)
    r.screens["exit"] = t.text()
    r.check("exit: double Ctrl+C quits", t.proc.poll() is not None)


def fullscreen_overlay(command: str):
    """Fullscreen overlay lifecycle (TS tui.ts alt-screen short-circuit):
    opening `command` enters the alternate screen with SGR mouse tracking
    (1000/1003/1006) and uses the whole terminal; clicks and wheel are routed
    to the overlay (never typed into the composer); Esc leaves the alt screen,
    turns tracking off and reveals the chat exactly as before."""
    tag = command.strip("/")

    def sc(t: Term, r: Result) -> None:
        ready(t)
        submit(t, "first e2e prompt")
        t.wait_for(r"E2E-REPLY-1", 90)
        t.wait_for(READY_RE, 30, in_transcript=False)
        t.settle(1.0, 30)
        before = t.screen()
        submit(t, command)
        t.settle(1.5, 15)
        s = t.screen_
        r.screens[f"{tag}-open"] = t.text()
        r.check(f"{tag}: alt screen on open", s.alt_active)
        r.check(f"{tag}: mouse tracking 1000/1003/1006 on", s.mouse_modes == {1000, 1003, 1006})
        used = sum(1 for line in t.screen() if line.strip())
        r.notes.append(f"{tag} rows used {used}/{t.rows}")
        r.check(f"{tag}: uses the full terminal height", used >= t.rows - 1)
        # Wheel + click inside the overlay body: consumed by the overlay.
        t.send(sgr_wheel(10, 6, down=True))
        t.send(sgr_wheel(10, 6, down=False))
        t.send(sgr_click(10, 6))
        t.settle(0.8, 10)
        r.screens[f"{tag}-after-mouse"] = t.text()
        r.check(f"{tag}: overlay survives mouse input", s.alt_active and t.proc.poll() is None)
        t.key("esc")
        t.settle(1.0, 10)
        # A cleared search / sub-view may take a second Esc (TS model hub).
        if s.alt_active:
            t.key("esc")
            t.settle(1.0, 10)
        after = t.screen()
        r.screens[f"{tag}-closed"] = t.text()
        r.check(f"{tag}: Esc leaves the alt screen", not s.alt_active)
        r.check(f"{tag}: mouse tracking off after close", not s.mouse_modes)
        r.check(f"{tag}: chat restored (transcript + status visible)",
                "E2E-REPLY-1" in t.transcript() and re.search(READY_RE, t.text()) is not None)
        r.check(f"{tag}: no SGR bytes leaked into the composer", "[<" not in t.text())
        same = sum(1 for a, b in zip(before, after) if a == b)
        r.notes.append(f"{tag} rows identical to pre-open screen {same}/{t.rows}")

    return sc



def _find_cell(t: Term, needle: str, pred=None):
    """0-based (row, col) of the first screen cell run matching `needle`."""
    for row, line in enumerate(t.screen()):
        col = line.find(needle)
        if col >= 0 and (pred is None or pred(line)):
            return row, col
    return None


def _open_model_menu(t: Term) -> None:
    submit(t, "/model")
    t.settle(1.0, 15)


def sc_model(t: Term, r: Result) -> None:
    """/model: fullscreen two-pane browser (alt screen + mouse), models.yml
    models listed, selecting a model by keyboard and by mouse switches it."""
    ready(t)
    _open_model_menu(t)
    scr = t.text()
    r.screens["model-menu"] = scr
    r.check("model: alt screen on", t.screen_.alt_active)
    r.check("model: mouse tracking on", {1000, 1003, 1006} <= t.screen_.mouse_modes)
    lines = t.screen()
    r.check("model: Models frame fills the terminal", "Models" in lines[0] and lines[-1].startswith("╰"))
    r.check("model: sidebar roles/all/providers", "Roles" in scr and "All models" in scr and "mock" in scr)
    r.check("model: models.yml models listed", all(f"mock/{m}" in scr for m in ("text", "tool", "fail")))
    r.check("model: footer keys", "type to search" in scr and "close" in scr)
    # Keyboard: focus the list, search, pick mock/tool as the default role.
    t.send(KEYS["enter"])
    t.type("tool")
    t.settle(0.5, 8)
    t.send(KEYS["enter"])
    t.settle(0.5, 8)
    strip = t.screen()[-2]
    r.screens["model-role-strip"] = t.text()
    r.check("model: role strip offers default", "default" in strip and "smol" in strip)
    t.send(KEYS["enter"])
    t.settle(0.8, 10)
    for _ in range(3):  # Esc ladder: cancel strip / clear the query, then close
        if not t.screen_.alt_active:
            break
        t.send(KEYS["esc"])
        t.settle(0.8, 10)
    r.check("model: Esc closes overlay (alt screen + mouse off)",
            not t.screen_.alt_active and not t.screen_.mouse_modes)
    r.check("model: keyboard pick switches model", t.wait_for(r"e2e tool\b", 10, in_transcript=False))
    # Mouse: wheel is harmless, click a row (select), click again (activate),
    # click the default chip.
    _open_model_menu(t)
    t.send(sgr_wheel(60, 10))
    t.pump(0.3)
    pos = _find_cell(t, "mock/fail")
    r.check("model: mock/fail row visible for click", pos is not None)
    if pos:
        row, col = pos
        t.send(sgr_click(col + 3, row + 1))
        t.settle(0.5, 8)
        t.send(sgr_click(col + 3, row + 1))
        t.settle(0.5, 8)
        chip = _find_cell(t, "default", lambda l: "\u2192" in l)
        r.screens["model-role-strip-mouse"] = t.text()
        r.check("model: click opens role strip", chip is not None)
        if chip:
            t.send(sgr_click(chip[1] + 2, chip[0] + 1))
            t.settle(0.8, 10)
    for _ in range(3):
        if not t.screen_.alt_active:
            break
        t.send(KEYS["esc"])
        t.settle(0.8, 10)
    r.check("model: mouse pick switches model", t.wait_for(r"e2e fail\b", 10, in_transcript=False))
    r.check("model: closed after mouse flow", not t.screen_.alt_active and not t.screen_.mouse_modes)


def overlay_settings_edit():
    """Editing settings by keyboard AND by mouse persists to the temp HOME
    config.yml: /settings opens fullscreen with mouse tracking, arrow keys
    move and Enter cycles a bool, SGR clicks select/activate a row, the
    wheel scrolls the list, and Esc leaves the alt screen + tracking off."""
    def sc(t: Term, r: Result) -> None:
        def home_config() -> str:
            env = Path(f"/proc/{t.proc.pid}/environ").read_bytes().split(b"\0")
            home = next(e.split(b"=", 1)[1] for e in env if e.startswith(b"HOME="))
            config = Path(home.decode()) / ".omp" / "agent" / "config.yml"
            return config.read_text() if config.exists() else ""

        def close_overlay() -> None:
            t.key("esc")
            t.settle(0.8, 10)
            if t.screen_.alt_active:
                t.key("esc")
                t.settle(0.8, 10)

        def find_row(label: str) -> tuple[int, int] | None:
            """1-based (col, row) of a rendered row label, for SGR input."""
            for y, line in enumerate(t.screen(), start=1):
                x = line.find(label)
                if x >= 0:
                    return x + 3, y
            return None

        ready(t)
        submit(t, "/settings")
        t.settle(1.5, 15)
        s = t.screen_
        r.check("settings-edit: alt screen on open", s.alt_active)
        r.check("settings-edit: mouse tracking on open", s.mouse_modes == {1000, 1003, 1006})
        r.screens["settings-open"] = t.text()

        # Keyboard edit: ↓ moves to the Color-Blind Mode row, ⏎ cycles the
        # bool to true (Appearance tab order: Dark, Light, Symbol, Blind).
        for _ in range(3):
            t.key("down")
        t.settle(0.3, 5)
        row = find_row("Color-Blind Mode")
        r.check("settings-edit: row visible after arrow keys", row is not None)
        r.check("settings-edit: arrows select the row", _selected_label(t).startswith("Color-Blind"))
        t.key("enter")
        t.settle(0.5, 10)
        r.check("settings-edit: keyboard edit persisted",
                "colorBlindMode: true" in home_config())

        # Mouse edit: move off the row so the first click only selects;
        # the second click activates (cycles back to false).
        t.key("down")
        t.settle(0.3, 5)
        pos = find_row("Color-Blind Mode")
        r.check("settings-edit: row still visible", pos is not None)
        t.send(sgr_click(*pos))
        t.settle(0.4, 8)
        t.send(sgr_click(*pos))
        t.settle(0.5, 10)
        r.check("settings-edit: mouse edit persisted",
                "colorBlindMode: false" in home_config())
        before = _selected_label(t)
        t.send(sgr_wheel(pos[0], pos[1], down=True))
        t.settle(0.4, 8)
        r.check("settings-edit: wheel scrolls the list", _selected_label(t) != before)

        close_overlay()
        r.screens["settings-closed"] = t.text()
        r.check("settings-edit: Esc leaves the alt screen", not s.alt_active)
        r.check("settings-edit: mouse tracking off after close", not s.mouse_modes)
        r.check("settings-edit: edits survived in config",
                "colorBlindMode: false" in home_config())

    return sc


def _model_glyph(t: Term, model: str) -> str:
    """The status-line glyph in front of the model name (`π > ◒ e2e think`):
    the thinking-level icon in compact mode, `⬢` for a model without thinking."""
    m = re.search(r"π > (\S+) e2e " + re.escape(model), t.text())
    return m.group(1) if m else ""


def _composer_look(t: Term) -> dict[str, bool]:
    """What the composer frame looks like on the ready screen."""
    lines = t.screen()
    return {
        "box-top": any(l.startswith("╭") and "π" in l for l in lines),
        "box-bottom": any(l.startswith("╰") and l.rstrip().endswith("╯") for l in lines),
        "band-row": any(l.startswith(" π > ") for l in lines),
        "rule-row": any(set(l.strip()) == {"─"} and len(l.strip()) > 40 for l in lines),
    }


def sc_composer_shape(t: Term, r: Result) -> None:
    """Editing Composer Shape in /settings changes the editor's look live and
    persists: band (status band + `╰─` prompt) → Rounded Box (status embedded
    in a framed top border, `╰─ … ─╯` prompt row)."""
    def home_config() -> str:
        env = Path(f"/proc/{t.proc.pid}/environ").read_bytes().split(b"\0")
        home = next(e.split(b"=", 1)[1] for e in env if e.startswith(b"HOME="))
        config = Path(home.decode()) / ".omp" / "agent" / "config.yml"
        return config.read_text() if config.exists() else ""

    ready(t)
    before = _composer_look(t)
    r.check("composer-shape: default is the band", before["band-row"] and not before["box-top"])
    r.screens["composer-before"] = t.text()
    submit(t, "/settings")
    t.settle(1.5, 15)
    for _ in range(12):
        if _selected_label(t).startswith("Composer Shape"):
            break
        t.key("down")
        t.settle(0.2, 3)
    r.check("composer-shape: row selected", _selected_label(t).startswith("Composer Shape"))
    t.key("enter")
    t.settle(0.8, 8)
    r.check("composer-shape: opens a choice list", "Rounded Box" in t.text())
    r.screens["composer-menu"] = t.text()
    t.key("down")
    t.settle(0.4, 5)
    t.key("enter")
    t.settle(0.8, 8)
    r.check("composer-shape: edit persisted", "shape: box" in home_config())
    t.key("esc")
    t.settle(1.0, 10)
    if t.screen_.alt_active:
        t.key("esc")
        t.settle(1.0, 10)
    after = _composer_look(t)
    r.screens["composer-after"] = t.text()
    r.check("composer-shape: box frame drawn live", after["box-top"] and after["box-bottom"])
    r.check("composer-shape: band gone", not after["band-row"])


def sc_effort(t: Term, r: Result) -> None:
    """Shift+Tab cycles the thinking effort (off → auto → minimal … xhigh) and
    the status line's level glyph follows; the editor hint advertises the
    chord; a bare /effort opens the Thinking Level selector."""
    ready(t)
    r.check("effort: editor hint", "⇧⇥ to change thinking effort" in t.text())
    start = _model_glyph(t, "think")
    glyphs = []
    for _ in range(8):
        t.send(b"\x1b[Z")
        t.settle(0.5, 6)
        glyphs.append(_model_glyph(t, "think"))
    r.notes.append(f"glyphs {start!r} -> {glyphs}")
    r.check("effort: shift+tab changes the status glyph", bool(start) and glyphs[0] not in ("", start))
    # off, auto, then the efforts the model supports, in that order (the
    # ladder's top differs per model, so compare the common prefix).
    want = ["⦸", "⟳", "○", "◔", "◑"]
    it = iter(glyphs)
    r.check("effort: cycle order off/auto/min/low/med", all(any(g == w for g in it) for w in want))
    r.check("effort: cycle wraps", len(set(glyphs)) >= 6 and glyphs[-1] in glyphs[:-1])
    submit(t, "/effort")
    t.settle(1.0, 8)
    r.screens["effort-selector"] = t.text()
    r.check("effort: selector title", "Thinking Level" in t.text())
    r.check("effort: selector rows", "No reasoning" in t.text() and "Very brief reasoning" in t.text())
    t.key("esc")
    t.settle(0.8, 8)
    r.check("effort: selector closes", "Thinking Level" not in t.text())


def _selected_label(t: Term) -> str:
    """The label of the ❯-selected settings row on the current screen."""
    for line in t.screen():
        idx = line.find("❯ ")
        if idx >= 0:
            return line[idx + 2:].strip()
    return ""


SCENARIOS = {
    "startup": ("text", sc_startup),
    "two_prompts": ("text", sc_two_prompts),
    "tool": ("tool", sc_tool),
    "fail": ("fail", sc_fail),
    "tools": ("tools", sc_tools),
    "usage": ("tool", sc_usage, "display:\n  showTokenUsage: true\n"),
    "slash": ("text", sc_slash),
    "exit": ("text", sc_exit),
    "model": ("text", sc_model),
    "overlay_model": ("text", fullscreen_overlay("/model")),
    "overlay_settings": ("text", fullscreen_overlay("/settings")),
    "settings_edit": ("text", overlay_settings_edit()),
    "composer_shape": ("text", sc_composer_shape),
    "effort": ("think", sc_effort),
}


def reap_home_processes(root: Path) -> list[str]:
    """Stop daemons the client detached under this throwaway HOME (TS omp
    spawns e.g. text-predict). Only processes whose environment has exactly
    this HOME are touched."""
    killed = []
    needle = f"HOME={root}".encode() + b"\0"
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit() or int(pid_dir.name) == os.getpid():
            continue
        try:
            environ = (pid_dir / "environ").read_bytes() + b"\0"
        except OSError:
            continue
        if needle in environ:
            try:
                os.kill(int(pid_dir.name), signal.SIGTERM)
                killed.append(pid_dir.name)
            except OSError:
                pass
    return [f"reaped daemons {','.join(killed)}"] if killed else []


def run_client(binary: str, model: str, fn, port: int, keep: Path, extra_config: str = "") -> Result:
    root = Path(tempfile.mkdtemp(prefix="e2e-home-"))
    env = make_home(root, port, extra_config)
    r = Result()
    t = Term([binary, "--model", f"mock/{model}"], env=env, cwd=env["E2E_WORK"])
    try:
        fn(t, r)
    except Exception as e:  # a crashed scenario is a failed check, not a crashed run
        r.check(f"scenario raised {type(e).__name__}", False)
        r.notes.append(repr(e))
    finally:
        code = t.close()
        r.notes.append(f"exit code {code}")
        sessions = root / ".omp" / "agent" / "sessions"
        if sessions.is_dir():
            shutil.copytree(sessions, keep / "sessions", dirs_exist_ok=True)
        r.notes.extend(reap_home_processes(root))
        shutil.rmtree(root, ignore_errors=True)
    return r


def normalize(text: str) -> list[str]:
    """Screen lines with run-specific noise masked: temp HOME names, session
    ids, numbers, the rotating startup tip, trailing blanks. Leading
    indentation is kept (layout is part of the look)."""
    out = []
    skip_tip = False
    for line in text.splitlines():
        if re.match(r"\s*Tip:", line):
            skip_tip = True
            continue
        if skip_tip and line.strip() and line.startswith(" " * 20):
            continue  # wrapped continuation of the tip
        skip_tip = False
        line = re.sub(r"e2e-home-[A-Za-z0-9_]+", "<HOME>", line)
        line = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f-]{14,}", "<ID>", line)
        line = re.sub(r"\d+", "#", line).rstrip()
        if line.strip():
            out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", default=str(DEFAULT_RS))
    ap.add_argument("--ts", default=shutil.which("omp") or "omp")
    ap.add_argument("--only", default="")
    ap.add_argument("--client", default="both", choices=["rs", "ts", "both"])
    ap.add_argument("--out", default=str(REPO.parent / "runs/omp-rust/e2e-tui" / time.strftime("%Y%m%d-%H%M%S")))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mock_provider.LOG = str(out / "mock.jsonl")
    srv = mock_provider.serve(0)
    port = srv.server_address[1]

    names = [n for n in SCENARIOS if not args.only or n in args.only.split(",")]
    clients = {"ts": args.ts, "rs": args.rs}
    if args.client != "both":
        clients = {args.client: clients[args.client]}
    results: dict[str, dict[str, Result]] = {}
    for name in names:
        model, fn, *extra = SCENARIOS[name]
        for tag, binary in clients.items():
            print(f"[{tag}] {name} …", flush=True)
            keep = out / name / tag
            keep.mkdir(parents=True, exist_ok=True)
            res = run_client(binary, model, fn, port, keep, *extra)
            results.setdefault(name, {})[tag] = res
            for screen, text in res.screens.items():
                (keep / f"{screen}.txt").write_text(text)

    parity_fail = 0
    lines = ["# TUI e2e: omp vs omp-rs", ""]
    for name in names:
        lines.append(f"## {name}")
        per = results[name]
        checks = sorted({c for r in per.values() for c in r.checks})
        lines.append("| check | omp | omp-rs | verdict |")
        lines.append("|---|---|---|---|")
        for c in checks:
            ts_ok = per.get("ts", Result()).checks.get(c)
            rs_ok = per.get("rs", Result()).checks.get(c)
            if rs_ok is None:
                verdict = "baseline only" if ts_ok else "baseline (omp fails)"
            elif rs_ok:
                verdict = "ok"
            elif ts_ok is False:
                verdict = "baseline (omp fails too)"
            else:
                verdict = "**FAIL**"
                parity_fail += 1
            fmt = lambda v: "-" if v is None else ("✓" if v else "✗")
            lines.append(f"| {c} | {fmt(ts_ok)} | {fmt(rs_ok)} | {verdict} |")
        for tag, r in per.items():
            if r.notes:
                lines.append(f"- {tag}: " + "; ".join(r.notes))
        if "ts" in per and "rs" in per:
            for screen in per["rs"].screens:
                a = normalize(per["ts"].screens.get(screen, ""))
                b = normalize(per["rs"].screens[screen])
                ratio = difflib.SequenceMatcher(None, a, b).ratio()
                lines.append(f"- screen `{screen}` similarity {ratio:.2f}")
                diff = difflib.unified_diff(a, b, "omp", "omp-rs", lineterm="")
                (out / name / f"{screen}.diff").write_text("\n".join(diff))
        lines.append("")
    lines.append(f"PARITY FAILURES: {parity_fail}")
    report = "\n".join(lines)
    (out / "REPORT.md").write_text(report)
    print(report)
    print(f"\nscreens + diffs: {out}")
    srv.shutdown()
    return 1 if parity_fail else 0


if __name__ == "__main__":
    sys.exit(main())
