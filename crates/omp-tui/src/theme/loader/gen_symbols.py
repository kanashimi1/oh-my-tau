import json
import os

src = json.load(open('/tmp/symbols.json'))
presets = src['presets']
spinners = src['spinners']

def rs_str(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'

out = ["//! Generated: symbol presets and spinner frames (TS `theme/symbols.ts`).",
       "//! Regenerate by dumping omp's `SYMBOL_PRESETS` / `SPINNER_FRAMES` and",
       "//! running `gen_symbols.py`.",
       "",
       "/// One symbol preset: logical glyph name → display string.",
       "pub type PresetMap = &'static [(&'static str, &'static str)];",
       "",
       "/// Unicode preset (TS `UNICODE_SYMBOLS`).",
       "pub static UNICODE: PresetMap = &["]
for k, v in presets['unicode'].items():
    out.append(f"    ({rs_str(k)}, {rs_str(v)}),")
out += ["];", "", "/// Nerd-font preset (TS `NERD_SYMBOLS`), escapes kept as literal codepoints.",
        "pub static NERD: PresetMap = &["]
for k, v in presets['nerd'].items():
    out.append(f"    ({rs_str(k)}, {rs_str(v)}),")
out += ["];", "", "/// ASCII preset (TS `ASCII_SYMBOLS`).", "pub static ASCII: PresetMap = &["]
for k, v in presets['ascii'].items():
    out.append(f"    ({rs_str(k)}, {rs_str(v)}),")
out += ["];", "", "/// Spinner frames per preset and type (TS `SPINNER_FRAMES`).",
        "pub static SPINNERS: &[(&str, &str, &[&str])] = &["]
for preset, kinds in spinners.items():
    for kind, frames in kinds.items():
        fs = ", ".join(rs_str(f) for f in frames)
        out.append(f"    ({rs_str(preset)}, {rs_str(kind)}, &[{fs}]),")
out.append("];")

open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'symbols_data.rs'), 'w').write("\n".join(out) + "\n")
print("symbols:", {p: len(m) for p, m in presets.items()}, "spinner sets:", sum(len(k) for k in spinners.values()))
