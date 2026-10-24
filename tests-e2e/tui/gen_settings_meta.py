#!/usr/bin/env python3
"""Extract setting id -> (label, description, tab, group) from TS register({...}) blocks.

Scans vendor TS for `register({ ... })` object literals, pulls `id` and the
`ui: { label, description, tab, group }` metadata, emits a sorted Rust table.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "vendor-src/oh-my-pi/packages"

STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def find_registers(text: str, base: int = 0):
    """Yield (position, raw body) of each register({...}) call (balanced braces)."""
    for m in re.finditer(r"\bregister\(\s*\{", text):
        depth = 1
        i = m.end()
        start = i
        in_str = None
        while i < len(text) and depth > 0:
            c = text[i]
            if in_str:
                if c == "\\":
                    i += 2
                    continue
                if c == in_str:
                    in_str = None
            elif c in "\"'`":
                in_str = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth == 0:
            yield base + m.start(), text[start : i - 1]


def domain_order(root: Path) -> list[Path]:
    """TS settings-panel domain order: `import * as x from "..."` lines of
    config/all-settings.ts, resolved to file paths."""
    all_settings = root / "coding-agent/src/config/all-settings.ts"
    order: list[Path] = []
    try:
        text = all_settings.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return order
    src_dir = all_settings.parent
    for m in re.finditer(r'import \* as \w+ from "(\.[^"]+)";', text):
        rel = m.group(1)
        p = (src_dir / rel).resolve()
        for candidate in (p.with_suffix(".ts"), p / "index.ts", p.with_suffix(".d.ts")):
            if candidate.exists():
                order.append(candidate)
                break
    return order


def field(body: str, name: str):
    """Extract a top-level scalar string field `name: "..."`."""
    m = re.search(rf"\b{name}\s*:\s*\"((?:[^\"\\]|\\.)*)\"", body)
    return m.group(1) if m else None


def ui_block(body: str):
    m = re.search(r"\bui\s*:\s*\{", body)
    if not m:
        return None
    depth = 1
    i = m.end()
    start = i
    in_str = None
    while i < len(body) and depth > 0:
        c = body[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
        elif c in "\"'`":
            in_str = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return body[start : i - 1] if depth == 0 else None


def unescape(s: str) -> str:
    return (
        s.replace('\\"', '"')
        .replace("\\n", " ")
        .replace("\\t", " ")
        .replace("\\\\", "\\")
    )


def main():
    root = ROOT
    domains = domain_order(root)
    # Entries: id -> (label, description, sort key). First write wins for
    # duplicates (domain files are scanned before the tail).
    out: dict[str, tuple[str, str, tuple]] = {}

    def consider(path: Path, order_hint: tuple):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        for pos, body in find_registers(text):
            sid = field(body, "id")
            if not sid:
                continue
            ui = ui_block(body)
            if not ui:
                continue
            label = field(ui, "label")
            desc = field(ui, "description")
            tab = field(ui, "tab")
            group = field(ui, "group") or ""
            if not label and not desc:
                continue
            if sid in out:
                continue
            out[sid] = (
                unescape(label) if label else "",
                unescape(desc) if desc else "",
                (unescape(tab) if tab else "", unescape(group)),
                (*order_hint, pos),
            )

    for idx, path in enumerate(domains):
        consider(path, (idx, 0, str(path)))
    # Then every other TS file.
    dir_domain: dict[Path, int] = {}
    for idx, path in enumerate(domains):
        dir_domain.setdefault(path.parent, idx)
    for path in sorted(root.rglob("*.ts")):
        if "node_modules" in path.parts or "/vendor/" in str(path):
            continue
        resolved = path.resolve()
        if resolved in domains:
            continue
        # Files living in a domain directory follow that domain (registration
        # happens when the domain module tree is evaluated).
        for parent in resolved.parents:
            if parent in dir_domain:
                consider(path, (dir_domain[parent], 1, str(resolved)))
                break
        else:
            consider(path, (len(domains), 1, str(resolved)))

    ordered = sorted(out.items(), key=lambda kv: kv[1][3])

    def rs(s: str) -> str:
        body = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{body}"'

    print("// Generated from omp TS v18.6.1 register({...}) ui metadata.")
    print(f"// {len(ordered)} entries, in TS settings-panel order. Do not edit by hand;", end=" ")
    print("regenerate with tests-e2e/tui/gen_settings_meta.py.")
    print("pub static SETTING_UI: &[(&str, &str, &str, (&str, &str))] = &[", end="")
    first = True
    for sid, (label, desc, (tab, group), _key) in ordered:
        comma = "," if first else ","
        first = False
        print(f"\n    ({rs(sid)}, {rs(label)}, {rs(desc)}, ({rs(tab)}, {rs(group)})){comma}", end="")
    print("\n];", end="")
    print()


if __name__ == "__main__":
    main()
