#!/usr/bin/env python3
"""Generate crates/omp-tui/src/overlays/settings_defs.rs from the TS settings panel.

Runs dump_settings_defs.ts under bun against the vendored TS source
(createSettingsHost + getAllSettingDefs), so the Rust panel lists exactly the
rows TS builds, with the same kinds (boolean toggle / enum cycle / submenu /
text / multiselect / provider limits), options, defaults and conditions.
"""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODING_AGENT = REPO / "vendor-src/oh-my-pi/packages/coding-agent"
OUT = REPO / "crates/omp-tui/src/overlays/settings_defs.rs"


def rs(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    dump_json = tmp / "defs.json"
    probe = CODING_AGENT / "_dump_defs.ts"
    shutil.copy(Path(__file__).with_name("dump_settings_defs.ts"), probe)
    try:
        subprocess.run(["bun", "run", str(probe)], cwd=CODING_AGENT, check=True,
                       env={**os.environ, "DEFS_OUT": str(dump_json)})
    finally:
        probe.unlink(missing_ok=True)
    defs = json.loads(dump_json.read_text())
    shutil.rmtree(tmp, ignore_errors=True)

    kinds = {"boolean": "Bool", "enum": "Enum", "submenu": "Submenu", "text": "Text",
             "multiselect": "Multi", "providerLimits": "ProviderLimits"}
    lines = [
        "// Generated from the TS settings panel (createSettingsHost + getAllSettingDefs).",
        f"// {len(defs)} rows, in TS panel order. Do not edit by hand; regenerate with tests-e2e/tui/gen_settings_defs.py.",
        "",
        "/// One submenu choice (TS `SubmenuOption`).",
        "#[derive(Debug)]",
        "pub struct DefOption {",
        "    pub value: &'static str,",
        "    pub label: &'static str,",
        "    pub description: &'static str,",
        "}",
        "",
        "/// How TS edits the setting (`SettingDef.type`).",
        "#[derive(Debug, Clone, Copy, PartialEq, Eq)]",
        "pub enum DefKind {",
        "    Bool,",
        "    Enum,",
        "    Submenu,",
        "    Text,",
        "    Multi,",
        "    ProviderLimits,",
        "}",
        "",
        "/// One TS panel row definition.",
        "#[derive(Debug)]",
        "pub struct SettingDefMeta {",
        "    pub id: &'static str,",
        "    pub kind: DefKind,",
        "    /// Registry value type (`boolean|enum|number|string|array|record`).",
        "    pub schema: &'static str,",
        "    pub tab: &'static str,",
        "    pub group: &'static str,",
        "    pub label: &'static str,",
        "    pub description: &'static str,",
        "    /// Declared default as JSON (`null` when none).",
        "    pub default_json: &'static str,",
        "    /// Enum cycle values (kind `Enum`).",
        "    pub values: &'static [&'static str],",
        "    /// Submenu / multiselect choices.",
        "    pub options: &'static [DefOption],",
        "    /// Choices come from the host at runtime (themes, composer shapes).",
        "    pub runtime: bool,",
        "    /// Visibility predicate name (TS `ui.condition`).",
        "    pub condition: &'static str,",
        "    pub ordered: bool,",
        "}",
        "",
        "pub static SETTING_DEFS: &[SettingDefMeta] = &[",
    ]
    for d in defs:
        opts = d["options"] or []
        vals = d["values"] or []
        lines.append("    SettingDefMeta {")
        lines.append(f"        id: {rs(d['path'])}, kind: DefKind::{kinds[d['type']]}, schema: {rs(d['schemaType'])},")
        lines.append(f"        tab: {rs(d['tab'])}, group: {rs(d['group'] or '')},")
        lines.append(f"        label: {rs(d['label'])},")
        lines.append(f"        description: {rs(d['description'])},")
        lines.append(f"        default_json: {rs(json.dumps(d['default']))},")
        lines.append("        values: &[" + ", ".join(rs(v) for v in vals) + "],")
        lines.append("        options: &[" + ", ".join(
            f"DefOption {{ value: {rs(o['value'])}, label: {rs(o['label'])}, description: {rs(o.get('description') or '')} }}"
            for o in opts) + "],")
        lines.append(f"        runtime: {'true' if d['runtime'] else 'false'}, condition: {rs(d['condition'] or '')}, "
                     f"ordered: {'true' if d['ordered'] else 'false'},")
        lines.append("    },")
    lines.append("];")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(defs)} rows)")


if __name__ == "__main__":
    main()
