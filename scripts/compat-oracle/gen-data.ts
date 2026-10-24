// Regenerates the data tables the Rust compat resolver embeds:
//   vendor/pa-ai/data/compat/rules.json  (verbatim copy of the compiled TS rule tree)
//   vendor/pa-ai/data/compat/axes.json   (AXES wire/request applicability + API_COMPAT_RECORDS)
// Run: bun scripts/compat-oracle/gen-data.ts
//
// The oh-my-pi sources come from `OMP_TS_SRC` (root of an oh-my-pi checkout) if set, else
// `<repo>/vendor-src/oh-my-pi`.
import { copyFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "../..");
const TS_SRC = process.env.OMP_TS_SRC ?? join(ROOT, "vendor-src/oh-my-pi");
const { AXES, API_COMPAT_RECORDS } = await import(join(TS_SRC, "packages/catalog/src/compat/axes"));
const out = `${ROOT}/vendor/pa-ai/data/compat`;
mkdirSync(out, { recursive: true });
copyFileSync(join(TS_SRC, "packages/catalog/src/compat/rules.json"), `${out}/rules.json`);
const wire: Record<string, string[]> = {};
for (const directive in AXES) {
	const axis = AXES[directive];
	if (axis.set === "wire" && axis.records) wire[axis.key] = [...axis.records];
}
const request = Object.values(AXES)
	.filter(a => a.records?.includes("request"))
	.map(a => a.key);
writeFileSync(`${out}/axes.json`, JSON.stringify({ wire, request, apiRecords: API_COMPAT_RECORDS }));
console.log(Object.keys(wire).length, request.length);
