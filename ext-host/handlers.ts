// handlers.ts — handler execution with timeout + result classification (mechanical split from host.ts).
import { log } from "./transport.ts";
import { TIMEOUT_MS } from "./env.ts";
import { TIMEOUT } from "./types.ts";
import type { Json, Handler, Ext, HandlerRecord } from "./types.ts";
import { ctx } from "./context.ts";

export async function runHandler(ext: Ext, h: Handler, event: Json, timeoutMs: number, rec: HandlerRecord[]): Promise<{ value?: unknown; failed?: "timeout" | "error"; message?: string }> {
  const t0 = performance.now();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const base = { ext: ext.name, event: String(event.type), tool: typeof event.toolName === "string" ? event.toolName : undefined };
  try {
    const value = await Promise.race([
      Promise.resolve().then(() => h(event, ctx())),
      new Promise<typeof TIMEOUT>((r) => (timer = setTimeout(() => r(TIMEOUT), timeoutMs))),
    ]);
    const ms = Math.round((performance.now() - t0) * 10) / 10;
    if (value === TIMEOUT) {
      const message = `handler timed out after ${timeoutMs}ms`;
      rec.push({ ...base, ms, action: "error", error: message });
      log("warn", `${ext.path} ${base.event}: ${message}`);
      return { failed: "timeout", message };
    }
    rec.push({ ...base, ms, action: classify(value) });
    return { value };
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    rec.push({ ...base, ms: Math.round((performance.now() - t0) * 10) / 10, action: "error", error: message });
    log("warn", `${ext.path} ${base.event} failed: ${message}`, { stack: e instanceof Error ? e.stack : undefined });
    return { failed: "error", message };
  } finally {
    clearTimeout(timer);
  }
}

// Same classification as the user's lib/hook-trace.ts so traces compare across hosts.
export function classify(v: unknown): string {
  const o = v as { block?: unknown; content?: unknown; message?: unknown } | null | undefined;
  if (v === undefined || v === null) return "none";
  if (o?.block === true) return "block";
  if (Array.isArray(o?.content)) return "replace";
  if (o?.message) return "inject";
  return "other";
}
