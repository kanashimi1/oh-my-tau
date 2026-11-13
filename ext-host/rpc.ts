// rpc.ts — JSON-RPC 2.0 dispatcher and nowait barrier (mechanical split from host.ts).
import type { Json } from "./types.ts";
import { inNowait, log, out, closeSession, getNowaitBlocked, getNowaitWakers, getPending } from "./transport.ts";
import { exts, flagValues, providers, events, setState } from "./registry.ts";
import { load } from "./loader.ts";
import { emit } from "./emit.ts";
import { allTools, callTool, callCommand, jsonSchema, toolAcs } from "./tools.ts";
import { setTimeout } from "node:timers";

export class RpcError extends Error {
  constructor(public code: number, message: string) {
    super(message);
  }
}
export const rpcError = (code: number, message: string) => new RpcError(code, message);

let loading: Promise<unknown> = Promise.resolve();
export function setLoading(p: Promise<unknown>): void { loading = p.catch(() => undefined); }
export function getLoading(): Promise<unknown> { return loading; }

let nowaitTail: Promise<unknown> = Promise.resolve();
export function setNowaitTail(p: Promise<unknown>): void {
  nowaitTail = p.then(() => undefined, () => undefined);
}
export function getNowaitTail(): Promise<unknown> { return nowaitTail; }

/** Wait for `barrier`, unless a fire-and-forget handler is blocked on a core request (see `inNowait`). */
export async function awaitBarrier(barrier: Promise<unknown>): Promise<void> {
  let done = false;
  void barrier.then(() => void (done = true));
  while (!done && getNowaitBlocked() === 0) {
    const { promise: woken, resolve: wake } = Promise.withResolvers<void>();
    getNowaitWakers().add(wake);
    try {
      await Promise.race([barrier, woken]);
    } finally {
      getNowaitWakers().delete(wake);
    }
  }
}

export async function dispatch(method: string, p: Json): Promise<unknown> {
  if (method === "emit" && p.nowait === true) {
    // Reserved synchronously (before any await) so arrival order == execution order.
    const prev = nowaitTail;
    const run = (async () => {
      await prev;
      await loading;
      return inNowait.run(true, () => emit(p));
    })();
    setNowaitTail(run);
    return run;
  }
  if (method === "load") {
    // Loads are serialized, and every other request waits for them (registration order is deterministic).
    const run = loading.then(() => load(p));
    setLoading(run);
    return run;
  }
  // Awaited emit / call_tool: run after every queued fire-and-forget emit.
  const barrier = method === "emit" || method === "call_tool" ? nowaitTail : undefined;
  await loading;
  if (barrier) await awaitBarrier(barrier);
  switch (method) {
    case "emit":
      return emit(p);
    case "list_tools":
      return allTools().map(({ ext, tool }) => ({ name: tool.name, label: tool.label, description: tool.description, parameters: jsonSchema(tool.parameters), approval: tool.approval, extension: ext.name }));
    case "call_tool":
      return callTool(p);
    case "cancel_tool": {
      // Additive: cores that predate `cancel_tool` never call this.
      const ac = toolAcs.get(String(p.toolCallId ?? ""));
      if (ac) ac.abort();
      return true;
    }
    case "list_commands":
      return exts.flatMap((e) => [...e.commands.values()].map((c) => ({ name: c.name, description: c.description, extension: e.name })));
    case "call_command":
      return callCommand(p);
    case "set_ctx":
      setState(p.ctx as Json);
      return true;
    // Additive introspection/update methods; cores that predate them never call these.
    case "list_flags":
      return exts.flatMap((e) => [...e.flags.values()].map((f) => ({ ...f, extension: e.name, value: flagValues.get(f.name as string) })));
    case "set_flag_value":
      flagValues.set(String(p.name), p.value as boolean | string);
      return true;
    case "list_providers":
      return [...providers.values()].map((pr) => ({ ...pr }));
    case "bus_emit":
      events.emit(String(p.channel), p.data);
      return true;
    case "shutdown":
      // In a shared-daemon Worker, process.exit would kill every session;
      // close the session transport instead (daemon terminates this Worker).
      setTimeout(() => (closeSession ? closeSession() : process.exit(0)), 20);
      return true;
    default:
      throw rpcError(-32601, `method not found: ${method}`);
  }
}

export async function onMessage(line: string) {
  let msg: Json;
  try {
    msg = JSON.parse(line) as Json;
  } catch {
    out({ id: null, error: { code: -32700, message: "parse error" } });
    return;
  }
  if (msg.method === undefined) {
    const w = getPending().get(msg.id as number);
    if (!w) return;
    getPending().delete(msg.id as number);
    if (msg.error) w.reject(new Error(String((msg.error as Json).message)));
    else w.resolve(msg.result);
    return;
  }
  const id = msg.id as number | string | undefined;
  try {
    const result = await dispatch(String(msg.method), (msg.params as Json) ?? {});
    if (id !== undefined) out({ id, result });
  } catch (e) {
    if (id === undefined) return;
    const code = e instanceof RpcError ? e.code : -32000;
    out({ id, error: { code, message: e instanceof Error ? e.message : String(e) } });
  }
}
