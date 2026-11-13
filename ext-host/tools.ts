// tools.ts — tool/command invocation (mechanical split from host.ts).
import type { Json, Ext } from "./types.ts";
import { exts, events } from "./registry.ts";
import { log, request, notify } from "./transport.ts";
import { runHandler } from "./handlers.ts";
import { ctx } from "./context.ts";

export function summarize(v: unknown): unknown {
  if (v === undefined || v === null) return null;
  const o = v as Json;
  const s: Json = {};
  for (const [k, x] of Object.entries(o)) {
    if (Array.isArray(x)) s[k] = `array(${x.length})`;
    else if (typeof x === "string") s[k] = x.length > 120 ? `${x.slice(0, 120)}…(${x.length})` : x;
    else if (x && typeof x === "object") s[k] = "object";
    else s[k] = x;
  }
  return s;
}

// ---------------------------------------------------------------- tools / commands
// omp tool schemas are arktype-backed callables; the wire form is their JSON Schema.
export function jsonSchema(schema: unknown): unknown {
  const s = schema as { toJsonSchema?: () => unknown } | undefined;
  return typeof s?.toJsonSchema === "function" ? s.toJsonSchema() : schema;
}

export function allTools(): { ext: Ext; tool: Json }[] {
  return exts.flatMap((ext) => [...ext.tools.values()].map((tool) => ({ ext, tool })));
}

// Live tool calls by toolCallId, so a core-side `cancel_tool` can abort them.
export const toolAcs = new Map<string, AbortController>();

export async function callTool(p: Json) {
  if (p.ctx) state = { ...state, ...(p.ctx as Json) };
  const found = allTools().findLast((t) => t.tool.name === p.name);
  if (!found) throw rpcError(-32004, `unknown tool: ${p.name}`);
  const ac = new AbortController();
  const id = String(p.toolCallId ?? "");
  toolAcs.set(id, ac);
  try {
    const execute = found.tool.execute as (...a: unknown[]) => Promise<unknown>;
    return (await execute.call(found.tool, id, p.params ?? {}, ac.signal, (partial: unknown) => notify("tool_update", { toolCallId: p.toolCallId, partial }), ctx())) ?? null;
  } finally {
    toolAcs.delete(id);
  }
}

export async function callCommand(p: Json) {
  for (const e of exts) {
    const c = e.commands.get(String(p.name));
    if (c) return ((await (c.handler as (a: string, c: unknown) => Promise<unknown>)(String(p.args ?? ""), ctx())) ?? null);
  }
  throw rpcError(-32004, `unknown command: ${p.name}`);
}

