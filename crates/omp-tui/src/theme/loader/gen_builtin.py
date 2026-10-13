import os

D = os.path.join(os.path.dirname(__file__), 'builtin')
entries = sorted(f[:-5] for f in os.listdir(D) if f.endswith('.json'))
lines = [
    "//! Generated: built-in theme registry (TS `theme/loader.ts` BUILTIN_THEMES).",
    "//! Regenerate by re-copying omp's `pi-tui/src/theme/{dark,light,defaults/*.json}`.",
    "",
    "/// Built-in theme name → embedded JSON (TS `getBuiltinThemes`).",
    "pub static BUILTIN: &[(&str, &str)] = &[",
]
for name in entries:
    lines.append(f'    ("{name}", include_str!("builtin/{name}.json")),')
lines.append("];")
with open(os.path.join(os.path.dirname(__file__), 'builtin_index.rs'), 'w') as fh:
    fh.write("\n".join(lines) + "\n")
print(len(entries), "themes")
