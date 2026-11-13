// loader.ts — extension discovery and loading (mechanical split from host.ts).
import { readdirSync, statSync, existsSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { homedir } from "node:os";
import type { Json, Ext } from "./types.ts";
import { exts, setState, setStateCwd } from "./registry.ts";
import { setTimeoutMs, setTrace } from "./env.ts";
import { makeApi } from "./context.ts";

export function discover(dir: string): string[] {
  if (!existsSync(dir)) return [];
  const files: string[] = [];
  for (const name of readdirSync(dir).sort()) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isFile() && /\.(ts|js)$/.test(name)) files.push(p);
    else if (st.isDirectory()) {
      const idx = ["index.ts", "index.js"].find((i) => existsSync(join(p, i)));
      if (idx) files.push(join(p, idx));
    }
  }
  return files;
}

export async function loadOne(path: string): Promise<{ ext?: Ext; error?: string }> {
  const resolved = resolve(path);
  let mod: { default?: unknown };
  try {
    mod = (await import(resolved)) as { default?: unknown };
  } catch (e) {
    return { error: `Failed to load extension: ${e instanceof Error ? e.message : e}` };
  }
  const factory = typeof mod === "function" ? mod : mod.default;
  if (typeof factory !== "function") return { error: `Extension does not export a valid factory function: ${path}` };
  const base = basename(path).replace(/\.(ts|js)$/, "");
  const ext: Ext = { path: resolved, name: base === "index" ? basename(resolve(path, "..")) : base, handlers: new Map(), tools: new Map(), commands: new Map(), flags: new Map() };
  const api = makeApi(ext);
  try {
    await (factory as (a: unknown) => unknown)(api);
  } catch (e) {
    return { error: `Failed to load extension: ${e instanceof Error ? e.message : e}` };
  }
  return { ext };
}

export async function load(p: Json) {
  if (p.ctx) setState(p.ctx as Json);
  if (typeof p.cwd === "string") setStateCwd(p.cwd);
  // Shared-daemon mode: the daemon's env is process-wide, so per-session
  // handler budget and trace file travel with the load request.
  if (typeof p.timeoutMs === "number" && p.timeoutMs > 0) setTimeoutMs(p.timeoutMs);
  if (typeof p.trace === "string") setTrace(p.trace);
  const paths = (p.paths as string[] | undefined) ?? discover(String(p.dir ?? join(homedir(), ".omp/agent/extensions")));
  const loaded: Json[] = [];
  const errors: Json[] = [];
  for (const path of paths) {
    const { ext, error } = await loadOne(path);
    if (error || !ext) {
      errors.push({ path, error });
      continue;
    }
    exts.push(ext);
    loaded.push({
      path,
      name: ext.name,
      handlers: Object.fromEntries([...ext.handlers].map(([k, v]) => [k, v.length])),
      tools: [...ext.tools.keys()],
      commands: [...ext.commands.keys()],
      flags: [...ext.flags.values()].map((f) => ({ ...f })),
    });
  }
  // Capability negotiation: declare what this host supports so the core
  // never calls a method the host doesn't have (additive RPC safety).
  const capabilities = [
    "tools", "commands", "flags", "bus_emit", "set_ctx",
    "cancel_tool", "list_flags", "set_flag_value", "list_providers",
  ];
  return { loaded, errors, capabilities };
}

