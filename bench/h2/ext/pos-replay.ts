import { reset, step } from "./pos-guard.ts";
import { readFileSync } from "node:fs";
const rows = readFileSync(process.argv[2], "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const out: any[] = [];
for (const e of rows) {
  reset(); let fires = 0; const pats: string[] = []; let first = -1;
  for (const s of e.steps) {
    let input: unknown = s.args; try { input = JSON.parse(s.args); } catch { input = { raw: s.args }; }
    const r = step(s.tool, input, s.out, s.error);
    if (r && r.would) { fires++; pats.push(`${r.pattern}/${r.gap}`); if (first < 0) first = r.idx; if (process.env.SHOW) console.error(JSON.stringify({p: r.pattern, H: r.H, win: r.win.map((w: any) => [w.sig.slice(0, 110), w.err, w.old])})); }
  }
  out.push({ task: e.task, score: e.score, n: e.steps.length, fires, first, pats });
}
console.log(JSON.stringify(out));
