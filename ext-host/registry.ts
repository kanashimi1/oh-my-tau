// registry.ts — per-session mutable registries: exts, bus, timers, flags, providers, core state (mechanical split from host.ts).
import type { Json } from "./types.ts";

export const exts: import("./types.ts").Ext[] = [];
export const bus = new Map<string, Set<(d: unknown) => void>>();
export const timers = new Set<ReturnType<typeof setTimeout>>();
// Flag values / providers mirror pi-coding-agent's ExtensionRuntime: registerFlag seeds
// flagValues with the default; the core may update values via the set_flag_value RPC.
export const flagValues = new Map<string, boolean | string>();
export const providers = new Map<string, { name: string; config: Json; extension: string }>();
export let state: Json = { cwd: process.cwd(), hasUI: false, agentKind: "main", branch: [] };
export let ctxCache: unknown;

export function setState(patch: Json): void { state = { ...state, ...patch }; }
export function setStateCwd(cwd: string): void { state.cwd = cwd; }
export function setCtxCache(v: unknown): void { ctxCache = v; }
export function getCtxCache(): unknown { return ctxCache; }
export function getState(): Json { return state; }

export const events = {
  on(ch: string, fn: (d: unknown) => void) {
    const s = bus.get(ch) ?? new Set();
    s.add(fn);
    bus.set(ch, s);
    return () => s.delete(fn);
  },
  emit(ch: string, data: unknown) {
    for (const fn of bus.get(ch) ?? []) {
      try {
        void Promise.resolve(fn(data)).catch((e) => log("warn", `event bus handler ${ch}: ${e}`));
      } catch (e) {
        log("warn", `event bus handler ${ch}: ${e}`);
      }
    }
  },
};

// local import to avoid cycle at module-eval time (events.emit uses log lazily inside try/catch)
import { log } from "./transport.ts";
