// transport.ts — newline-delimited JSON-RPC 2.0 transport (mechanical split from host.ts).
import { AsyncLocalStorage } from "node:async_hooks";
import type { Json } from "./types.ts";
import { SHUTDOWN_TIMEOUT_MS } from "./env.ts";

// In stdio mode `send` writes to stdout; in shared-daemon Worker mode it posts
// the line to the daemon, which writes it to the session's unix socket.
let emitLine: (line: string) => void = () => {};
export let closeSession: (() => void) | null = null;
export const out = (msg: Json) => emitLine(`${JSON.stringify({ jsonrpc: "2.0", ...msg })}\n`);
// Extensions (and libs they import) log to stdout; the channel is stdout, so route console.* to stderr.
const toStderr = (...a: unknown[]) => process.stderr.write(`${a.map(String).join(" ")}\n`);
console.log = toStderr;
console.info = toStderr;
console.debug = toStderr;

let nextId = 1;
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
// Fire-and-forget emits (`emit` with `nowait`) run strictly serially, and every
// later awaited `emit`/`call_tool` waits for them (wire order == execution
// order, as when the core awaited each one). `inNowait` marks async contexts
// running such a handler: while one of them is blocked on a core request
// (e.g. a handler calling ctx.compact(), which makes the core emit
// session_before_compact), awaited emits stop waiting so that request cannot
// deadlock on the barrier it is itself holding open.
export const inNowait = new AsyncLocalStorage<true>();
let nowaitBlocked = 0;
// Every awaited emit/call_tool parked on the barrier (parallel tool calls can
// park several at once); a blocking fire-and-forget handler wakes them all.
const nowaitWakers = new Set<() => void>();
export function request(method: string, params: Json): Promise<unknown> {
  const id = nextId++;
  const { promise, resolve, reject } = Promise.withResolvers<unknown>();
  pending.set(id, { resolve, reject });
  out({ id, method, params });
  if (!inNowait.getStore()) return promise;
  nowaitBlocked++;
  for (const wake of nowaitWakers) wake();
  nowaitWakers.clear();
  return promise.finally(() => void nowaitBlocked--);
}
export const notify = (method: string, params: Json) => out({ method, params });
export const log = (level: string, message: string, extra?: Json) => notify("log", { level, message, ...extra });

// --- setters used by the bootstrap (previously module-level assignments) ---
export function setEmitLine(fn: (line: string) => void): void { emitLine = fn; }
export function setCloseSession(fn: (() => void) | null): void { closeSession = fn; }
export function setConsoleToStderr(): void { /* console rerouted above at import time, as in host.ts */ }

// --- accessors for the dispatcher (previously direct module-global reads) ---
export function getPending(): Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }> { return pending; }
export function getNowaitBlocked(): number { return nowaitBlocked; }
export function getNowaitWakers(): Set<() => void> { return nowaitWakers; }
export function getShutdownTimeoutMs(): number { return SHUTDOWN_TIMEOUT_MS; }
