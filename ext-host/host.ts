#!/usr/bin/env bun
// omp extension host: loads omp ExtensionAPI extensions (unchanged .ts files) and serves them to the Rust core
// over JSON-RPC 2.0 on stdio (newline-delimited). Handler aggregation mirrors pi-coding-agent's ExtensionRunner.
//
// core -> host requests: load, emit, list_tools, call_tool, cancel_tool, list_commands, call_command, set_ctx, bus_emit, shutdown
// host -> core:          notifications append_entry, send_message, send_user_message, log, ui_notify, set_label,
//                        provider_registered, provider_unregistered;
//                        requests compact, navigate_tree, ui_confirm (core may answer with an error → treated as unsupported)
// Additive RPC:          list_flags, set_flag_value, list_providers (flag/provider state is also queryable host-side).
//
// Env: OMP_EXT_TRACE=<path> appends one JSON line per emit (per-handler action/ms) to <path>;
//      OMP_EXT_TIMEOUT_MS handler budget (default 30000); OMP_PKG_ROOT omp's pi-coding-agent package dir (for typebox/zod).
import { existsSync, appendFileSync, readdirSync, statSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { homedir } from "node:os";
import { spawn } from "node:child_process";

type Json = Record<string, unknown>;
type Handler = (event: Json, ctx: unknown) => unknown;
interface Ext {
  path: string;
  name: string;
  handlers: Map<string, Handler[]>;
  tools: Map<string, Json>;
  commands: Map<string, Json>;
  flags: Map<string, Json>;
  label?: string;
}

const TIMEOUT_MS = Number(process.env.OMP_EXT_TIMEOUT_MS) || 30_000;
const SHUTDOWN_TIMEOUT_MS = 2_000;
const TRACE = process.env.OMP_EXT_TRACE;
const PKG_ROOT =
  process.env.OMP_PKG_ROOT ?? join(homedir(), ".bun/install/global/node_modules/@oh-my-pi/pi-coding-agent");

// ---------------------------------------------------------------- transport
const out = (msg: Json) => process.stdout.write(`${JSON.stringify({ jsonrpc: "2.0", ...msg })}\n`);
// Extensions (and libs they import) log to stdout; the channel is stdout, so route console.* to stderr.
const toStderr = (...a: unknown[]) => process.stderr.write(`${a.map(String).join(" ")}\n`);
console.log = toStderr;
console.info = toStderr;
console.debug = toStderr;

let nextId = 1;
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
function request(method: string, params: Json): Promise<unknown> {
  const id = nextId++;
  const { promise, resolve, reject } = Promise.withResolvers<unknown>();
  pending.set(id, { resolve, reject });
  out({ id, method, params });
  return promise;
}
const notify = (method: string, params: Json) => out({ method, params });
const log = (level: string, message: string, extra?: Json) => notify("log", { level, message, ...extra });

// ---------------------------------------------------------------- state
const exts: Ext[] = [];
const bus = new Map<string, Set<(d: unknown) => void>>();
const timers = new Set<ReturnType<typeof setTimeout>>();
// Flag values / providers mirror pi-coding-agent's ExtensionRuntime: registerFlag seeds
// flagValues with the default; the core may update values via the set_flag_value RPC.
const flagValues = new Map<string, boolean | string>();
const providers = new Map<string, { name: string; config: Json; extension: string }>();
let state: Json = { cwd: process.cwd(), hasUI: false, agentKind: "main", branch: [] };
let ctxCache: unknown;
let typebox: unknown;
let zod: unknown;

const events = {
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

function execCommand(command: string, args: string[], cwd: string, opts: Json = {}) {
  return new Promise<Json>((res) => {
    const child = spawn(command, args, { cwd, stdio: ["ignore", "pipe", "pipe"], signal: opts.signal as AbortSignal | undefined });
    let stdout = "";
    let stderr = "";
    let killed = false;
    const t = typeof opts.timeout === "number" ? setTimeout(() => ((killed = true), child.kill("SIGKILL")), opts.timeout) : undefined;
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    child.on("error", (e) => {
      clearTimeout(t);
      res({ stdout, stderr: stderr || String(e), code: -1, killed: killed || (e as Error).name === "AbortError" });
    });
    child.on("close", (code) => {
      clearTimeout(t);
      res({ stdout, stderr, code: code ?? -1, killed });
    });
  });
}

const sm = () => ({
  getCwd: () => state.cwd,
  getSessionId: () => state.sessionId,
  getSessionFile: () => state.sessionFile,
  getLeafId: () => state.leafId ?? null,
  getBranch: () => (state.branch as Json[]) ?? [],
  getEntries: () => (state.branch as Json[]) ?? [],
  getEntry: (id: string) => ((state.branch as Json[]) ?? []).find((e) => e.id === id),
  getSessionName: () => state.sessionName,
});

function makeCtx() {
  const noop = () => {};
  const unsupported = (what: string) => async () => {
    throw new Error(`${what} is not available in the Rust extension host`);
  };
  return {
    ui: {
      notify: (message: string, type?: string) => notify("ui_notify", { message, type: type ?? "info" }),
      confirm: async (title: string, message: string) => {
        try {
          return (await request("ui_confirm", { title, message })) === true;
        } catch {
          return false;
        }
      },
      setStatus: noop,
      setWidget: noop,
      setTitle: noop,
      // Remaining members mirror pi-coding-agent's noOpUIContext (headless defaults, runner.ts).
      setWorkingMessage: noop,
      setFooter: noop,
      setHeader: noop,
      setEditorText: noop,
      pasteToEditor: noop,
      getEditorText: () => "",
      addAutocompleteProvider: noop,
      setEditorComponent: noop,
      onTerminalInput: () => () => {},
      custom: async () => undefined,
      getToolsExpanded: () => false,
      setToolsExpanded: noop,
      get theme() {
        return undefined;
      },
      getAllThemes: () => Promise.resolve([]),
      getTheme: () => Promise.resolve(undefined),
      setTheme: () => Promise.resolve({ success: false as const, error: "UI not available" }),
      select: async () => undefined,
      input: async () => undefined,
      editor: async () => undefined,
    },
    mode: state.mode ?? "rpc",
    get hasUI() {
      return state.hasUI === true;
    },
    get cwd() {
      return String(state.cwd);
    },
    get model() {
      return state.model;
    },
    get agent() {
      return { kind: state.agentKind ?? "main", ...((state.agent as Json) ?? {}) };
    },
    sessionManager: sm(),
    modelRegistry: { getApiKey: async () => undefined, getAvailable: () => [], getAll: () => [] },
    getContextUsage: () => state.contextUsage,
    getAsyncJobSnapshot: () => (state.asyncJobs as Json) ?? null,
    compact: (o?: unknown) => request("compact", { options: o as Json }).then(() => undefined),
    navigateTree: (target: string, o?: unknown) => request("navigate_tree", { target, options: o as Json }),
    isIdle: () => state.idle !== false,
    abort: () => notify("abort", {}),
    hasPendingMessages: () => state.pending === true,
    shutdown: () => notify("shutdown_requested", {}),
    isProjectTrusted: () => true,
    getSystemPrompt: () => (state.systemPrompt as string[]) ?? [],
    runEphemeralTurn: unsupported("runEphemeralTurn"),
    setInterval: (cb: (...a: unknown[]) => void, ms?: number, ...a: unknown[]) => {
      const t = setInterval(() => safe(cb, a), ms);
      t.unref?.();
      timers.add(t);
      return t;
    },
    setTimeout: (cb: (...a: unknown[]) => void, ms?: number, ...a: unknown[]) => {
      const t = setTimeout(() => safe(cb, a), ms);
      t.unref?.();
      timers.add(t);
      return t;
    },
  };
}
function safe(cb: (...a: unknown[]) => unknown, a: unknown[]) {
  try {
    void Promise.resolve(cb(...a)).catch((e) => log("warn", `timer callback failed: ${e}`));
  } catch (e) {
    log("warn", `timer callback failed: ${e}`);
  }
}
function ctx() {
  return (ctxCache ??= makeCtx());
}

function makeApi(ext: Ext) {
  const api = {
    logger: {
      debug: (m: string, x?: Json) => log("debug", m, x),
      info: (m: string, x?: Json) => log("info", m, x),
      warn: (m: string, x?: Json) => log("warn", m, x),
      error: (m: string, x?: Json) => log("error", m, x),
    },
    get typebox() {
      return typebox;
    },
    get zod() {
      return zod;
    },
    events,
    pi: {},
    on(event: string, handler: Handler) {
      const l = ext.handlers.get(event) ?? [];
      l.push(handler);
      ext.handlers.set(event, l);
    },
    registerTool(tool: Json) {
      ext.tools.set(String(tool.name), tool);
    },
    registerCommand(name: string, options: Json) {
      ext.commands.set(name, { name, ...options });
    },
    registerShortcut: () => {},
    // Semantics mirror pi-coding-agent loader.ts: the default seeds flagValues; getFlag only
    // sees flags this extension registered (omp scopes flags per extension).
    registerFlag(name: string, options: Json) {
      ext.flags.set(name, { name, ...options });
      if (options.default !== undefined) flagValues.set(name, options.default as boolean | string);
    },
    getFlag(name: string) {
      if (!ext.flags.has(name)) return undefined;
      return flagValues.get(name);
    },
    registerMessageRenderer: () => {},
    registerAssistantThinkingRenderer: () => {},
    registerComposerShape: () => {},
    registerFileWriteFallback: () => {},
    registerFileDeleteFallback: () => {},
    // Provider registrations are forwarded to the core as notifications (the Rust core owns
    // the model registry), and kept host-side so list_providers can report them.
    registerProvider(name: string, config: Json) {
      providers.set(name, { name, config, extension: ext.name });
      notify("provider_registered", { name, config, extension: ext.name });
    },
    unregisterProvider(name: string) {
      if (providers.delete(name)) notify("provider_unregistered", { name, extension: ext.name });
    },
    setLabel: (label: string) => {
      ext.label = label;
      notify("set_label", { extension: ext.name, label });
    },
    sendMessage: (message: Json, options?: Json) => notify("send_message", { extension: ext.name, message, options }),
    sendUserMessage: (content: unknown, options?: Json) => notify("send_user_message", { extension: ext.name, content, options }),
    appendEntry: (customType: string, data?: unknown) => notify("append_entry", { extension: ext.name, customType, data }),
    exec: (command: string, args: string[], options?: Json) => execCommand(command, args, String(options?.cwd ?? state.cwd), options),
    getActiveTools: () => (state.activeTools as string[]) ?? [],
    getAllTools: () => (state.allTools as Json[]) ?? [],
    setActiveTools: async (names: string[]) => void (await request("set_active_tools", { names })),
    getCommands: () => exts.flatMap((e) => [...e.commands.values()]),
    setModel: async (model: unknown) => (await request("set_model", { model })) === true,
    getThinkingLevel: () => state.thinkingLevel,
    setThinkingLevel: (level: string, persist?: boolean) => notify("set_thinking_level", { level, persist }),
    getServiceTiers: () => ({}),
    setServiceTier: () => {},
    getSessionName: () => state.sessionName,
    setSessionName: async (name: string) => void (await request("set_session_name", { name })),
  };
  return api;
}

// ---------------------------------------------------------------- loading
function discover(dir: string): string[] {
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

async function loadOne(path: string): Promise<{ ext?: Ext; error?: string }> {
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

async function load(p: Json) {
  if (p.ctx) state = { ...state, ...(p.ctx as Json) };
  if (typeof p.cwd === "string") state.cwd = p.cwd;
  for (const [key, mod] of [["typebox", "legacy-typebox.ts"], ["zod", ""]] as const) {
    try {
      if (key === "typebox") typebox = await import(join(PKG_ROOT, "src/extensibility", mod));
      else zod = await import(join(PKG_ROOT, "node_modules/@oh-my-pi/omptype/zod")).catch(() => import("@oh-my-pi/omptype/zod"));
    } catch (e) {
      if (key === "typebox") log("warn", `typebox unavailable (api.typebox undefined): ${e}`);
    }
  }
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
  return { loaded, errors };
}

// ---------------------------------------------------------------- emit
const TIMEOUT = Symbol("timeout");
interface HandlerRecord { ext: string; event: string; tool?: string; ms: number; action: string; error?: string }

async function runHandler(ext: Ext, h: Handler, event: Json, timeoutMs: number, rec: HandlerRecord[]): Promise<{ value?: unknown; failed?: "timeout" | "error"; message?: string }> {
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
function classify(v: unknown): string {
  const o = v as { block?: unknown; content?: unknown; message?: unknown } | null | undefined;
  if (v === undefined || v === null) return "none";
  if (o?.block === true) return "block";
  if (Array.isArray(o?.content)) return "replace";
  if (o?.message) return "inject";
  return "other";
}

const isNonBlank = (s: unknown): s is string => typeof s === "string" && s.trim().length > 0;
const joinCtx = (xs: string[]) => {
  const u = [...new Set(xs)];
  return u.length ? u.join("\n\n") : undefined;
};

async function emit(p: Json) {
  if (p.ctx) state = { ...state, ...(p.ctx as Json) };
  // No structuredClone: like the TS runner, every handler for this emit sees
  // this same object (handler-side mutations are visible to later handlers,
  // and the core's copy is independent — it came off the wire).
  const event = p.event as Json;
  const type = String(event.type);
  const timeoutMs = Number(p.timeoutMs) || TIMEOUT_MS;
  const rec: HandlerRecord[] = [];
  const handlersFor = (e: Ext) => e.handlers.get(type) ?? [];
  let result: unknown;

  if (type === "session_shutdown") {
    await Promise.all(exts.flatMap((e) => handlersFor(e).map((h) => runHandler(e, h, event, SHUTDOWN_TIMEOUT_MS, rec))));
    for (const t of timers) clearTimeout(t);
    timers.clear();
  } else if (type === "tool_call") {
    const agg = { input: undefined as Json | undefined, ctxs: [] as string[] };
    let control: Json | undefined;
    outer: for (const e of exts)
      for (const h of handlersFor(e)) {
        const r = await runHandler(e, h, event, timeoutMs, rec);
        let hr = r.value as Json | undefined;
        if (r.failed) hr = { block: true, reason: r.failed === "timeout" ? `Extension ${e.path} timed out after ${timeoutMs}ms` : `Extension ${e.path} failed: ${r.message}` };
        if (!hr) continue;
        if (hr.block) {
          result = hr;
          control = undefined;
          agg.input = undefined;
          agg.ctxs = [];
          break outer;
        }
        const { additionalContext, input, ...rest } = hr;
        if (isNonBlank(additionalContext)) agg.ctxs.push(additionalContext);
        if (input !== undefined) agg.input = input as Json;
        control = rest;
      }
    if (result === undefined) {
      const c = joinCtx(agg.ctxs);
      result = c === undefined && agg.input === undefined ? control : { ...control, ...(agg.input ? { input: agg.input } : {}), ...(c ? { additionalContext: c } : {}) };
    }
  } else if (type === "tool_result") {
    const cur = { ...event };
    let modified = false;
    const ctxs: string[] = [];
    for (const e of exts)
      for (const h of handlersFor(e)) {
        const hr = (await runHandler(e, h, cur, timeoutMs, rec)).value as Json | undefined;
        if (!hr) continue;
        for (const k of ["content", "details", "isError"]) if (hr[k] !== undefined) ((cur[k] = hr[k]), (modified = true));
        if (isNonBlank(hr.additionalContext)) ctxs.push(hr.additionalContext);
      }
    const c = joinCtx(ctxs);
    result = !modified ? (c === undefined ? undefined : { additionalContext: c }) : { content: cur.content, details: cur.details, isError: cur.isError, ...(c ? { additionalContext: c } : {}) };
  } else if (type === "context") {
    let messages = event.messages as unknown[];
    for (const e of exts)
      for (const h of handlersFor(e)) {
        const hr = (await runHandler(e, h, { type, messages }, timeoutMs, rec)).value as Json | undefined;
        if (hr && Array.isArray(hr.messages)) messages = hr.messages;
      }
    result = { messages };
  } else if (type === "before_agent_start") {
    const messages: unknown[] = [];
    let sp = event.systemPrompt as string[];
    let spMod = false;
    for (const e of exts)
      for (const h of handlersFor(e)) {
        const hr = (await runHandler(e, h, { ...event, systemPrompt: sp }, timeoutMs, rec)).value as Json | undefined;
        if (!hr) continue;
        if (hr.message) messages.push(hr.message);
        if (hr.systemPrompt !== undefined) ((sp = typeof hr.systemPrompt === "string" ? [hr.systemPrompt] : (hr.systemPrompt as string[])), (spMod = true));
      }
    if (messages.length || spMod) result = { messages: messages.length ? messages : undefined, systemPrompt: spMod ? sp : undefined };
  } else if (type === "input") {
    let text = String(event.text);
    let images = event.images;
    const orig = { text, images };
    for (const e of exts)
      for (const h of handlersFor(e)) {
        const hr = (await runHandler(e, h, { ...event, text, images }, timeoutMs, rec)).value as Json | undefined;
        if (hr?.handled) {
          result = hr;
          break;
        }
        if (hr?.text !== undefined) text = String(hr.text);
        if (hr?.images !== undefined) images = hr.images;
      }
    if (result === undefined) {
      const t: Json = {};
      if (text !== orig.text) t.text = text;
      if (images !== orig.images) t.images = images;
      result = t;
    }
  } else if (type === "assistant_message") {
    // Chained text-only rewrites of the finalized assistant message (ExtensionRunner.emitAssistantMessage).
    const original = ((event.message as Json).content ?? []) as Json[];
    const texts = original.map((b) => (b.type === "text" ? b.text : undefined));
    const rewritten = (i: number) => original[i]?.type === "text" && texts[i] !== original[i].text;
    const current = (): Json[] =>
      original.map((b, i) => {
        if (b.type !== "text" || !rewritten(i)) return b;
        const { textSignature: _stale, ...rest } = b;
        return { ...rest, text: texts[i] };
      });
    const metadata = (b: Json) => {
      const { text: _t, textSignature: _s, ...rest } = b;
      return rest;
    };
    for (const e of exts)
      for (const h of handlersFor(e)) {
        const presented = structuredClone({ ...(event.message as Json), content: current() });
        const hr = (await runHandler(e, h, { type, message: presented }, timeoutMs, rec)).value as Json | undefined;
        if (hr?.content === undefined) continue;
        const expected = structuredClone(current());
        const rep = hr.content as Json[];
        const invalid =
          !Array.isArray(rep) ||
          rep.length !== expected.length ||
          rep.some((b, i) => {
            const prev = expected[i];
            if (!b || typeof b !== "object" || b.type !== prev?.type) return true;
            if (b.type !== "text") return !Bun.deepEquals(b, prev);
            return typeof b.text !== "string" || !Bun.deepEquals(metadata(b), metadata(prev));
          });
        if (invalid) {
          log("warn", `${e.path} assistant_message: content replacement may only change text in existing blocks`);
          continue;
        }
        for (const [i, b] of rep.entries()) if (b.type === "text") texts[i] = b.text;
      }
    if (original.some((_b, i) => rewritten(i))) result = { content: current() };
  } else if (type === "user_bash" || type === "user_python") {
    // First handler returning a result wins.
    outer3: for (const e of exts)
      for (const h of handlersFor(e)) {
        const hr = (await runHandler(e, h, event, timeoutMs, rec)).value as Json | undefined;
        if (hr) {
          result = hr;
          break outer3;
        }
      }
  } else {
    // session_before_* (cancel short-circuits, last result wins), session_stop (block short-circuits), everything else: observe.
    const isBefore = type.startsWith("session_before_");
    outer2: for (const e of exts)
      for (const h of handlersFor(e)) {
        const hr = (await runHandler(e, h, event, timeoutMs, rec)).value as Json | undefined;
        if (!hr) continue;
        if (isBefore) {
          result = hr;
          if (hr.cancel) break outer2;
        } else if (type === "session_stop") {
          if (hr.decision === "block") {
            result = hr;
            break outer2;
          }
          if (hr.continue === true && (isNonBlank(hr.additionalContext) || isNonBlank(hr.reason))) result ??= hr;
        } else if (type === "session.compacting") result = hr;
      }
  }

  if (TRACE) {
    try {
      appendFileSync(TRACE, `${JSON.stringify({ ts: Date.now(), event: type, tool: typeof event.toolName === "string" ? event.toolName : undefined, handlers: rec, result: summarize(result) })}\n`);
    } catch {}
  }
  return { result: result ?? null, handlers: rec, errors: rec.filter((r) => r.action === "error").map((r) => ({ extension: r.ext, event: r.event, error: r.error })) };
}

function summarize(v: unknown): unknown {
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
function jsonSchema(schema: unknown): unknown {
  const s = schema as { toJsonSchema?: () => unknown } | undefined;
  return typeof s?.toJsonSchema === "function" ? s.toJsonSchema() : schema;
}

function allTools(): { ext: Ext; tool: Json }[] {
  return exts.flatMap((ext) => [...ext.tools.values()].map((tool) => ({ ext, tool })));
}

// Live tool calls by toolCallId, so a core-side `cancel_tool` can abort them.
const toolAcs = new Map<string, AbortController>();

async function callTool(p: Json) {
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

async function callCommand(p: Json) {
  for (const e of exts) {
    const c = e.commands.get(String(p.name));
    if (c) return ((await (c.handler as (a: string, c: unknown) => Promise<unknown>)(String(p.args ?? ""), ctx())) ?? null);
  }
  throw rpcError(-32004, `unknown command: ${p.name}`);
}

// ---------------------------------------------------------------- dispatch
class RpcError extends Error {
  constructor(public code: number, message: string) {
    super(message);
  }
}
const rpcError = (code: number, message: string) => new RpcError(code, message);

let loading: Promise<unknown> = Promise.resolve();

async function dispatch(method: string, p: Json): Promise<unknown> {
  if (method === "load") {
    // Loads are serialized, and every other request waits for them (registration order is deterministic).
    const run = loading.then(() => load(p));
    loading = run.catch(() => undefined);
    return run;
  }
  await loading;
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
      state = { ...state, ...(p.ctx as Json) };
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
      setTimeout(() => process.exit(0), 20);
      return true;
    default:
      throw rpcError(-32601, `method not found: ${method}`);
  }
}

async function onMessage(line: string) {
  let msg: Json;
  try {
    msg = JSON.parse(line) as Json;
  } catch {
    out({ id: null, error: { code: -32700, message: "parse error" } });
    return;
  }
  if (msg.method === undefined) {
    const w = pending.get(msg.id as number);
    if (!w) return;
    pending.delete(msg.id as number);
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

process.on("uncaughtException", (e) => log("error", `uncaughtException: ${e instanceof Error ? (e.stack ?? e.message) : e}`));
process.on("unhandledRejection", (e) => log("error", `unhandledRejection: ${e instanceof Error ? (e.stack ?? e.message) : e}`));

let buf = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk: string) => {
  buf += chunk;
  let i: number;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i).trim();
    buf = buf.slice(i + 1);
    if (line) void onMessage(line);
  }
});
process.stdin.on("end", () => process.exit(0));
