// rlm-clm — host-side RLM offload + CLM fold for omp / omt. Cuts the tokens every request re-sends.
//
// RLM leg (`tool_result`): a large tool output (bash, grep, find, lsp, fetch, …; not `read`) is stored
//   outside the context under a ref; the transcript keeps a digest (head, tail, error/warning lines
//   with line numbers). The model reads the rest on demand with `rlm_recall {ref, lines|grep}`.
//   The rewrite happens once, when the result enters the transcript, so the prompt-cache prefix
//   never changes because of it.
// CLM leg (`context`): once the LLM-bound view exceeds RLM_CLM_FOLD_CHARS, tool results of earlier
//   prompts (finished tasks) become one-line stubs (ref kept, recall reopens them) and their turns lose
//   reasoning blocks (tool-call arguments stay verbatim). The prompt being worked on stays raw unless it alone
//   passes twice the budget; the last RLM_CLM_KEEP_TURNS assistant turns are never folded. The folded set only grows,
//   only after half a budget of growth since the last fold, and only when the gain is worth one cache
//   break, so between folds the wire grows only at the end.
// No extra model calls anywhere. Fail-open: any error leaves the message untouched.
//
// Env: RLM_CLM=0 (off) · RLM_CLM_OFFLOAD_CHARS (6000) · RLM_CLM_FOLD_CHARS (16000)
//      RLM_CLM_KEEP_TURNS (3) · RLM_CLM_STORE (~/.omp/agent/rlm-store) · RLM_CLM_LOG (jsonl events)
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

function num(name: string, fallback: number): number {
  const v = Number(process.env[name]);
  return Number.isFinite(v) && v > 0 ? v : fallback;
}

export interface Config {
  offloadChars: number;
  foldChars: number;
  keepTurns: number;
  store: string;
}

export function config(): Config {
  return {
    offloadChars: num("RLM_CLM_OFFLOAD_CHARS", 6000),
    foldChars: num("RLM_CLM_FOLD_CHARS", 16000),
    keepTurns: num("RLM_CLM_KEEP_TURNS", 3),
    store: process.env.RLM_CLM_STORE || join(homedir(), ".omp", "agent", "rlm-store"),
  };
}

export const RECALL = "rlm_recall";
/** Tools whose output the model needs verbatim, or that are tiny by construction. */
const KEEP_RAW: Record<string, true> = {
  read: true, edit: true, write: true, [RECALL]: true, context_edit: true, todo: true, todo_write: true, ask: true,
};
const HEAD_CHARS = 1500;
const TAIL_CHARS = 1500;
const SIGNAL_LINES = 25;
const LINE_CAP = 240;
const STUB_MIN = 800;
const RECALL_CAP = 5000;
const SIGNAL =
  /\b(error|errors|warning|panicked|panic|failed|failure|fatal|exception|traceback|assert\w*)\b|^error\[|^\s*-->\s|FAILED|\bE\d{4}\b/i;
const REASONING: Record<string, true> = { thinking: true, redactedThinking: true, redacted_thinking: true, reasoning: true };
const REF_RE = /^\[rlm ref=([0-9a-f]{12})\b/;
/** Output another compressor already shrank and archived (sol-pi ObservationPack / EvidenceReducer /
 *  DelegateReceipt): offloading it again would only stack a second digest on the first. */
const PACKED_RE = /\[(ObservationPack|EvidenceReducer|DelegateReceipt)\]/;

type Block = { type?: string; text?: string; id?: string; name?: string; arguments?: Record<string, unknown>; [k: string]: unknown };
type Msg = { role?: string; content?: unknown; toolCallId?: string; toolName?: string; [k: string]: unknown };

export function textOf(content: unknown): string | undefined {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return undefined;
  let out = "";
  for (const b of content as Block[]) {
    if (!b || b.type !== "text" || typeof b.text !== "string") return undefined; // images etc.: leave alone
    out += (out ? "\n" : "") + b.text;
  }
  return out;
}

function clip(line: string): string {
  return line.length > LINE_CAP ? `${line.slice(0, LINE_CAP)}…[+${line.length - LINE_CAP}]` : line;
}

export function refOf(text: string): string {
  return createHash("sha1").update(text).digest("hex").slice(0, 12);
}

export function save(store: string, text: string): string {
  const ref = refOf(text);
  const file = join(store, `${ref}.txt`);
  if (!existsSync(file)) {
    mkdirSync(store, { recursive: true });
    writeFileSync(file, text);
  }
  return ref;
}

/** Digest kept in the transcript in place of an offloaded output. Deterministic in (text, ref, tool). */
export function digest(text: string, ref: string, tool: string): string {
  const lines = text.split("\n");
  let h = 0;
  for (let used = 0; h < lines.length && used < HEAD_CHARS; h++) used += clip(lines[h]).length + 1;
  let t = lines.length;
  for (let used = 0; t > h && used < TAIL_CHARS; t--) used += clip(lines[t - 1]).length + 1;
  const signal: string[] = [];
  for (let i = h; i < t && signal.length < SIGNAL_LINES; i++) {
    if (SIGNAL.test(lines[i])) signal.push(`L${i + 1}: ${clip(lines[i])}`);
  }
  const out = [
    `[rlm ref=${ref}: ${tool} output, ${lines.length} lines / ${text.length} chars, kept outside the context. ` +
      `Shown: head, error/warning lines, tail. More: ${RECALL} {ref:"${ref}", lines:"A-B"} or {ref:"${ref}", grep:"regex"}]`,
    `--- head (lines 1-${h}) ---`,
    ...lines.slice(0, h).map(clip),
  ];
  if (signal.length) out.push(`--- error/warning lines (${signal.length}) ---`, ...signal);
  if (t < lines.length) out.push(`--- tail (lines ${t + 1}-${lines.length}) ---`, ...lines.slice(t).map(clip));
  return out.join("\n");
}

/** RLM leg: the replacement text for a tool result, or undefined to keep it. */
export function offload(cfg: Config, tool: string, content: unknown): { text: string; ref: string; chars: number } | undefined {
  if (KEEP_RAW[tool]) return undefined;
  const text = textOf(content);
  if (text === undefined || text.length <= cfg.offloadChars || REF_RE.test(text) || PACKED_RE.test(text.slice(0, 400)))
    return undefined;
  const ref = save(cfg.store, text);
  return { text: digest(text, ref, tool), ref, chars: text.length };
}

function msgChars(m: Msg): number {
  const c = m?.content;
  if (typeof c === "string") return c.length;
  if (!Array.isArray(c)) return 0;
  let n = 0;
  for (const b of c as Block[]) {
    if (!b) continue;
    if (typeof b.text === "string") n += b.text.length;
    else if (typeof b.thinking === "string") n += (b.thinking as string).length;
    else if (b.type === "toolCall") n += JSON.stringify(b.arguments ?? {}).length;
  }
  return n;
}

function callSummary(call: Block | undefined): string {
  if (!call) return "";
  const a = call.arguments ?? {};
  const pick = a.path ?? a.command ?? a.pattern ?? a.query ?? a.url ?? a.file;
  if (typeof pick !== "string") return "";
  const one = pick.replace(/\s+/g, " ");
  return ` (${one.length > 80 ? `${one.slice(0, 80)}…` : one})`;
}

/** CLM leg state: the folded tool-call ids (only grows) and the stubs they map to. */
export class Folder {
  readonly stubs = new Map<string, string>();
  /** Assistant turns (from the start) whose results and bodies are folded. */
  turns = 0;
  /** View size right after the last fold: the next fold waits for half a budget of growth. */
  private after = 0;
  constructor(private cfg: Config, private onFold?: (e: { turns: number; ids: number; saved: number }) => void) {}

  private view(messages: Msg[], turns: number, stubs: Map<string, string>): { out: Msg[]; chars: number } {
    let seen = 0;
    let chars = 0;
    const out = messages.map((m) => {
      let r = m;
      if (m?.role === "assistant") {
        if (seen < turns && Array.isArray(m.content)) r = { ...m, content: slimTurn(m.content as Block[]) };
        seen++;
      } else if (m?.role === "toolResult" && typeof m.toolCallId === "string") {
        const stub = stubs.get(m.toolCallId);
        if (stub !== undefined) r = { ...m, content: [{ type: "text", text: stub }] };
      }
      chars += msgChars(r);
      return r;
    });
    return { out, chars };
  }

  /** Returns the LLM-bound list, or undefined when nothing is folded. Never mutates the input. */
  apply(messages: Msg[]): Msg[] | undefined {
    if (!Array.isArray(messages)) return undefined;
    const calls = new Map<string, Block>();
    const turnOf = new Map<string, number>();
    let assistants = 0;
    let beforeUser = 0; // assistant turns before the latest user message = finished prompts
    for (const m of messages) {
      if (m?.role === "user") beforeUser = assistants;
      if (m?.role !== "assistant") continue;
      if (Array.isArray(m.content)) {
        for (const b of m.content as Block[]) {
          if (b?.type === "toolCall" && typeof b.id === "string") {
            calls.set(b.id, b);
            turnOf.set(b.id, assistants);
          }
        }
      }
      assistants++;
    }
    let cur = this.view(messages, this.turns, this.stubs);
    // Fold what earlier prompts produced; the prompt being worked on stays raw unless it alone
    // grows past twice the budget.
    const recent = assistants - this.cfg.keepTurns;
    const target = cur.chars > 2 * this.cfg.foldChars ? recent : Math.min(recent, beforeUser);
    if (cur.chars > Math.max(this.cfg.foldChars, this.after + this.cfg.foldChars / 2) && target > this.turns) {
      const next = new Map(this.stubs);
      for (const m of messages) {
        if (m?.role !== "toolResult" || typeof m.toolCallId !== "string" || next.has(m.toolCallId)) continue;
        const turn = turnOf.get(m.toolCallId);
        if (turn === undefined || turn >= target) continue;
        const text = textOf(m.content);
        if (text === undefined || text.length < STUB_MIN) continue;
        const prior = REF_RE.exec(text)?.[1];
        const ref = prior ?? save(this.cfg.store, text);
        const tool = String(m.toolName ?? calls.get(m.toolCallId)?.name ?? "tool");
        next.set(
          m.toolCallId,
          `[clm: older ${tool} result${callSummary(calls.get(m.toolCallId))} folded, ${text.length} chars; ` +
            `${RECALL} {ref:"${ref}"} reopens it]`,
        );
      }
      const cand = this.view(messages, target, next);
      // One cache break must buy at least a quarter of the budget back.
      if (cur.chars - cand.chars >= this.cfg.foldChars / 4) {
        this.onFold?.({ turns: target, ids: next.size - this.stubs.size, saved: cur.chars - cand.chars });
        for (const [k, v] of next) this.stubs.set(k, v);
        this.turns = target;
        this.after = cand.chars;
        cur = cand;
      }
    }
    return this.turns > 0 || this.stubs.size > 0 ? cur.out : undefined;
  }
}

/** Drops reasoning blocks. Tool-call arguments stay verbatim: an elision marker inside an old edit body
 *  gets copied by the model into new edits (seen in the A/B: corrupted files, extra repair turns). */
function slimTurn(content: Block[]): Block[] {
  const out = content.filter((b) => !(b && REASONING[String(b.type)]));
  if (out.length === content.length) return content;
  return out.length ? out : [{ type: "text", text: "" }];
}

/** `rlm_recall` body: a line range, regex matches, or the head of a stored output. */
export function recall(store: string, ref: string, lines?: string, grep?: string, around = 0): { text: string; ok: boolean } {
  if (!/^[0-9a-f]{12}$/.test(ref)) return { text: `bad ref ${JSON.stringify(ref)} (12 hex chars from an [rlm ref=…] / [clm: …] marker)`, ok: false };
  let body: string;
  try {
    body = readFileSync(join(store, `${ref}.txt`), "utf8");
  } catch {
    return { text: `ref ${ref} not found in ${store}`, ok: false };
  }
  const all = body.split("\n");
  let picked: string[] = [];
  if (grep) {
    let re: RegExp;
    try {
      re = new RegExp(grep, "i");
    } catch (e) {
      return { text: `bad regex: ${(e as Error).message}`, ok: false };
    }
    const keep = new Set<number>();
    for (let i = 0; i < all.length; i++) {
      if (!re.test(all[i])) continue;
      for (let j = Math.max(0, i - around); j <= Math.min(all.length - 1, i + around); j++) keep.add(j);
    }
    picked = [...keep].sort((a, b) => a - b).map((i) => `L${i + 1}: ${all[i]}`);
    if (!picked.length) return { text: `no line of ref ${ref} (${all.length} lines) matches /${grep}/i`, ok: true };
  } else {
    let a = 1;
    let b = Math.min(all.length, 80);
    const m = lines ? /^\s*(\d+)\s*(?:-\s*(\d+))?\s*$/.exec(lines) : null;
    if (lines && !m) return { text: `bad lines ${JSON.stringify(lines)} (expected "A-B")`, ok: false };
    if (m) {
      a = Math.max(1, Number(m[1]));
      b = Math.min(all.length, m[2] ? Number(m[2]) : a);
    }
    picked = all.slice(a - 1, b).map((l, i) => `L${a + i}: ${l}`);
  }
  let out = "";
  let n = 0;
  for (const l of picked) {
    if (out.length + l.length + 1 > RECALL_CAP) break;
    out += `${l}\n`;
    n++;
  }
  if (n < picked.length) out += `[${picked.length - n} more lines; narrow lines/grep]\n`;
  return { text: `ref ${ref}: ${all.length} lines\n${out}`, ok: true };
}

function log(event: Record<string, unknown>): void {
  const file = process.env.RLM_CLM_LOG;
  if (!file) return;
  try {
    appendFileSync(file, `${JSON.stringify({ ts: Date.now(), pid: process.pid, ...event })}\n`);
  } catch {
    /* logging never breaks a turn */
  }
}

/** typebox when the host provides it (omp), the same JSON Schema by hand otherwise. */
function recallSchema(api: ExtensionAPI): unknown {
  let Type: any;
  try {
    Type = (api as unknown as { typebox?: { Type?: unknown } }).typebox?.Type;
  } catch {
    Type = undefined;
  }
  if (Type) {
    return Type.Object({
      ref: Type.String(),
      lines: Type.Optional(Type.String({ description: '"A-B"' })),
      grep: Type.Optional(Type.String({ description: "regex, case-insensitive" })),
      around: Type.Optional(Type.Integer({ minimum: 0, maximum: 20, description: "context lines per grep match" })),
    });
  }
  return {
    type: "object",
    properties: {
      ref: { type: "string" },
      lines: { type: "string", description: '"A-B"' },
      grep: { type: "string", description: "regex, case-insensitive" },
      around: { type: "integer", minimum: 0, maximum: 20, description: "context lines per grep match" },
    },
    required: ["ref"],
  };
}

export default function rlmClm(api: ExtensionAPI): void {
  if (process.env.RLM_CLM === "0") return;
  const cfg = config();
  const folder = new Folder(cfg, (e) => log({ kind: "fold", ...e }));

  api.on("tool_result", async (event) => {
    try {
      const e = event as unknown as { toolName?: string; content?: unknown };
      const r = offload(cfg, String(e.toolName ?? ""), e.content);
      if (!r) return undefined;
      log({ kind: "offload", tool: e.toolName, ref: r.ref, chars: r.chars, kept: r.text.length });
      return { content: [{ type: "text", text: r.text }] } as never;
    } catch {
      return undefined;
    }
  });

  api.on("context", async (event) => {
    try {
      const messages = folder.apply(event.messages as unknown as Msg[]);
      return messages ? ({ messages } as never) : undefined;
    } catch {
      return undefined;
    }
  });

  api.registerTool({
    name: RECALL,
    label: "RLM recall",
    description: "Read a stored tool output by its [rlm ref=…]/[clm: …] ref: a line range or regex matches.",
    parameters: recallSchema(api),
    async execute(_id: string, params: { ref: string; lines?: string; grep?: string; around?: number }) {
      const r = recall(cfg.store, params.ref, params.lines, params.grep, params.around ?? 0);
      log({ kind: "recall", ref: params.ref, ok: r.ok, lines: params.lines, grep: params.grep });
      return { content: [{ type: "text" as const, text: r.text }], isError: !r.ok };
    },
  } as never);
}
