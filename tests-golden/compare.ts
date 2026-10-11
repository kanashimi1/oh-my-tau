#!/usr/bin/env bun
// Usage: bun compare.ts [scenario...]   compares ts/<s>.jsonl with rs/<s>.jsonl
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";

const here = dirname(new URL(import.meta.url).pathname);
type Row = { dir: string; frame: any };
const load = (p: string): Row[] =>
	readFileSync(p, "utf8").split("\n").filter(Boolean).map(l => JSON.parse(l));

// Shape: replace leaves by their type; arrays collapse to the union of element shapes.
function shape(v: unknown): unknown {
	if (v === null) return "null";
	if (Array.isArray(v)) {
		const seen = new Map<string, unknown>();
		for (const e of v) seen.set(JSON.stringify(shape(e)), shape(e));
		return seen.size === 0 ? [] : [...seen.values()];
	}
	if (typeof v === "object") {
		const o: Record<string, unknown> = {};
		for (const k of Object.keys(v as object).sort()) o[k] = shape((v as any)[k]);
		return o;
	}
	return typeof v; // ids, timestamps, usage numbers, texts all collapse to their type
}

// flatten shape to path -> type-string set for readable diffs
function paths(s: unknown, p = "", out = new Map<string, Set<string>>()) {
	const add = (k: string, t: string) => (out.get(k) ?? out.set(k, new Set()).get(k)!).add(t);
	if (Array.isArray(s)) {
		if (!s.length) add(p, "[]");
		for (const e of s) paths(e, `${p}[]`, out);
	} else if (s && typeof s === "object") {
		if (!Object.keys(s).length) add(p, "{}");
		for (const [k, v] of Object.entries(s)) paths(v, p ? `${p}.${k}` : k, out);
	} else add(p, String(s));
	return out;
}

function diffShape(a: unknown, b: unknown): string[] {
	const pa = paths(shape(a)), pb = paths(shape(b));
	const lines: string[] = [];
	for (const k of new Set([...pa.keys(), ...pb.keys()])) {
		const x = pa.get(k), y = pb.get(k);
		if (!x) lines.push(`    + rs-only  ${k}: ${[...y!].join("|")}`);
		else if (!y) lines.push(`    - ts-only  ${k}: ${[...x].join("|")}`);
		else if ([...x].sort().join() !== [...y].sort().join())
			lines.push(`    ~ type     ${k}: ts=${[...x].join("|")} rs=${[...y].join("|")}`);
	}
	return lines;
}

// group incoming frames per command: responses keyed by id, events kept as ordered type list
function summarize(rows: Row[]) {
	const out = rows.filter(r => r.dir === "out").map(r => r.frame);
	const inn = rows.filter(r => r.dir === "in").map(r => r.frame);
	const resp = new Map<string, any>();
	for (const f of inn) if (f.type === "response") resp.set(f.id, f);
	const nonResp = inn.filter(f => f.type !== "response");
	return { out, resp, events: nonResp.map(f => f.type as string) };
}
const collapse = (xs: string[]) => {
	const r: string[] = [];
	for (const x of xs) {
		const l = r[r.length - 1];
		if (l && l.replace(/ x\d+$/, "") === x) r[r.length - 1] = `${x} x${(Number(/ x(\d+)$/.exec(l)?.[1] ?? 1)) + 1}`;
		else r.push(x);
	}
	return r;
};

const names = process.argv.length > 2
	? process.argv.slice(2)
	: readdirSync(join(here, "ts")).filter(f => f.endsWith(".jsonl")).map(f => f.replace(".jsonl", ""));
let total = 0;
for (const n of names) {
	const tp = join(here, "ts", `${n}.jsonl`), rp = join(here, "rs", `${n}.jsonl`);
	console.log(`\n=== ${n} ===`);
	if (!existsSync(tp) || !existsSync(rp)) { console.log(`  missing ${!existsSync(tp) ? "ts" : "rs"} capture`); total++; continue; }
	const T = summarize(load(tp)), R = summarize(load(rp));
	for (const c of T.out) {
		const t = T.resp.get(c.id), r = R.resp.get(c.id);
		const label = `${c.type}`;
		if (!t || !r) { console.log(`  [${label}] response missing: ts=${!!t} rs=${!!r}`); total++; continue; }
		const d = diffShape(t, r);
		if (d.length) { console.log(`  [${label}] shape diff`); d.forEach(l => console.log(l)); total++; }
		else console.log(`  [${label}] ok`);
	}
	const te = collapse(T.events), re = collapse(R.events);
	if (te.join() === re.join()) console.log(`  events: identical order (${te.join(" > ")})`);
	else { console.log(`  events differ\n    ts: ${te.join(" > ")}\n    rs: ${re.join(" > ")}`); total++; }
	// event frame shapes by type
	const byType = (rows: Row[]) => {
		const m = new Map<string, any>();
		for (const r of rows) if (r.dir === "in" && r.frame.type !== "response" && !m.has(r.frame.type)) m.set(r.frame.type, r.frame);
		return m;
	};
	const bt = byType(load(tp)), br = byType(load(rp));
	for (const [k, v] of bt) {
		const o = br.get(k);
		if (!o) continue;
		const d = diffShape(v, o);
		if (d.length) { console.log(`  [event ${k}] shape diff`); d.forEach(l => console.log(l)); total++; }
	}
}
console.log(`\n${total} differing item(s)`);
