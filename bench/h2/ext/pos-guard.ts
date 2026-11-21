// pos-guard.ts — PoS-style (arXiv 2610.01415) stagnation / loop detector for omt, eval build.
// Health over the last K tool results: S = share of steps without new information (error or a
// (call, result) pair already seen), R = recurrence at lag 1..3; H = 1 - max(S, R) (gap persistence
// taken as 1 while the task runs). H <= THETA -> "trapped": classify Static / Cycle / Drift and the blocked
// gap type (epistemic = mostly inspection, achievement = mostly mutation/execution), then append a
// recovery constraint to the tool result.  POS_GUARD_MODE: pos (factorized text) | generic (same trigger,
// generic text) | shadow (log only).  POS_GUARD_LOG: jsonl path for decisions.
import { appendFileSync } from "node:fs";
import { createHash } from "node:crypto";

const MODE = process.env.POS_GUARD_MODE ?? "pos";
const LOG = process.env.POS_GUARD_LOG ?? "";
const K = Number(process.env.POS_GUARD_K ?? 5);
const THETA = 0.34;
const COOLDOWN = 4;

type Step = { tool: string; sig: string; pair: string; err: boolean; old: boolean; kind: "inspect" | "mutate" };
const steps: Step[] = [];
const seenPairs = new Set<string>();
let lastFire = -100;
export function reset(): void { steps.length = 0; seenPairs.clear(); lastFire = -100; }

const isRec = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;
const h = (s: string) => createHash("sha1").update(s).digest("hex").slice(0, 12);

function canon(input: unknown): string {
  if (!isRec(input)) return "";
  return Object.keys(input).sort().filter((k) => k !== "i")
    .map((k) => `${k}=${typeof input[k] === "string" ? input[k] : JSON.stringify(input[k])}`).join("|");
}
function text(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((b) => (isRec(b) && typeof b.text === "string" ? b.text : "")).join("\n");
}
function kindOf(tool: string, input: unknown): "inspect" | "mutate" {
  if (tool === "write" || tool === "edit") return "mutate";
  if (tool === "bash" && isRec(input) && typeof input.command === "string") {
    const c = input.command;
    if (/(^|[;&|(]\s*)(python3?|node|bun|make|cargo|pytest)\b|>\s*\S|\bmv\b|\bcp\b|\bsed -i\b|\btee\b/.test(c)) return "mutate";
  }
  return "inspect";
}
function log(o: Record<string, unknown>) {
  if (LOG) try { appendFileSync(LOG, JSON.stringify(o) + "\n"); } catch { /* observability only */ }
}

// Window statistics (PoS eq. 5): S = steps without new information, R = recurrence at lag 1..3.
export function decide(win: Step[]) {
  const n = win.length;
  const S = win.filter((s) => s.err || s.old).length / n;
  let R = 0, lag = 0;
  for (let l = 1; l <= 3; l++) {
    let m = 0;
    for (let t = l; t < n; t++) if (win[t].sig === win[t - l].sig) m++;
    const r = m / (n - l);
    if (r > R) { R = r; lag = l; }
  }
  const H = 1 - Math.max(S, R);
  let pattern = "drift";
  if (win[n - 1].sig === win[n - 2].sig) pattern = "static";
  else if (R >= 0.5 && lag > 1) pattern = "cycle";
  const gap = win.filter((s) => s.kind === "mutate").length >= n / 2 ? "achievement" : "epistemic";
  return { S, R, H, lag, pattern, gap };
}

export function recovery(mode: string, d: ReturnType<typeof decide>, win: Step[]): string {
  if (mode === "generic")
    return "[progress monitor] Your recent steps are not making progress. Step back and try a different approach.";
  const show = (s: Step) => `${s.tool}(${s.sig.slice(s.tool.length + 2, s.tool.length + 90)})`;
  const pat = d.pattern === "static"
    ? `STATIC: you repeated ${show(win[win.length - 1])} and it keeps giving the same or a failing result. Do not run it again unchanged.`
    : d.pattern === "cycle"
      ? `CYCLE: the last steps alternate with period ${d.lag} (${win.slice(-d.lag).map(show).join(" -> ")}) without new results. Break the loop: change what you do, not how you phrase it.`
      : `DRIFT: your recent steps change things but do not close the remaining requirement (${win.filter((s) => s.err).length}/${win.length} failed, ${win.filter((s) => s.old).length} repeated an earlier result). Re-anchor: restate the one unmet requirement in one line, then act only on it.`;
  const gap = d.gap === "epistemic"
    ? "Next step must acquire DISCRIMINATING evidence: one check whose outcome decides between your hypotheses (e.g. compute the expected value for one concrete input by hand and compare), not another broad look."
    : "Next step must make ONE targeted state change aimed at the unmet requirement and verify it directly; do not rewrite unrelated parts.";
  return `[progress monitor] ${pat}\n${gap}`;
}

export function observe(tool: string, input: unknown, out: string, isError: boolean): Step {
  const err = isError || /\bTraceback\b|\bError:|exit code [1-9]/.test(out.slice(0, 3000));
  const sig = `${tool}::${canon(input)}`;
  const pair = `${sig}#${h(out.slice(0, 4000))}`;
  const st: Step = { tool, sig, pair, err, old: seenPairs.has(pair), kind: kindOf(tool, input) };
  seenPairs.add(pair); steps.push(st);
  return st;
}

// Polling / bookkeeping tools report other work; alternating them with a real step is not a loop
// (replay over 60 real sessions: every "cycle" fire was wait <-> bash/todo polling).
const REPORTING = new Set(["wait", "todo", "task", "job"]);

export function step(tool: string, input: unknown, out: string, isError: boolean) {
  if (REPORTING.has(tool)) return undefined;
  observe(tool, input, out, isError);
  const idx = steps.length - 1;
  if (steps.length < K) return undefined;
  const win = steps.slice(-K); const d = decide(win);
  const would = d.H <= THETA && idx - lastFire >= COOLDOWN;
  if (would) lastFire = idx;
  return { idx, would, ...d, win };
}

export default function posGuard(api: { on: (ev: string, fn: (e: unknown) => unknown) => void }): void {
  api.on("tool_result", async (event: unknown) => {
    if (!isRec(event)) return undefined;
    const tool = typeof event.toolName === "string" ? event.toolName : "";
    const out = text(event.content);
    const s = step(tool, event.input, out, event.isError === true);
    if (!s) return undefined;
    const { idx, would, win, ...d } = s;
    log({ step: idx, tool, H: +d.H.toFixed(2), S: +d.S.toFixed(2), R: +d.R.toFixed(2), pattern: d.pattern, gap: d.gap, would, fired: would && MODE !== "shadow" });
    if (!would || MODE === "shadow") return undefined;
    const note = recovery(MODE, d, win);
    return { content: Array.isArray(event.content) ? [...event.content, { type: "text", text: `\n\n${note}` }] : `${out}\n\n${note}` };
  });
}
