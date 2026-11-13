// context.ts — the ExtensionAPI context factory (mechanical split from host.ts).
import { dirname } from "node:path";
import type { Json } from "./types.ts";
import type { Ext } from "./types.ts";
import { request, notify, log } from "./transport.ts";
import { loadSchemaLib } from "./schemas.ts";
import { events } from "./registry.ts";
import { getState, setCtxCache, getCtxCache } from "./registry.ts";

export const sm = () => ({
  getCwd: () => getState().cwd,
  getSessionId: () => getState().sessionId,
  getSessionFile: () => getState().sessionFile,
  getSessionDir: () => getState().sessionFile ? dirname(String(getState().sessionFile)) : undefined,
  getLeafId: () => getState().leafId ?? null,
  getBranch: () => (getState().branch as Json[]) ?? [],
  getEntries: () => (getState().branch as Json[]) ?? [],
  getEntry: (id: string) => ((getState().branch as Json[]) ?? []).find((e) => e.id === id),
  getSessionName: () => getState().sessionName,
});

const orUndefined = (v: unknown) => (typeof v === "string" ? v : undefined);
export function makeCtx() {
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
      select: async (title: string, options: string[]) => orUndefined(await request("ui_select", { title, options }).catch(() => null)),
      input: async (title: string, placeholder?: string) => orUndefined(await request("ui_input", { title, placeholder }).catch(() => null)),
      editor: async (title: string, prefill?: string) => orUndefined(await request("ui_editor", { title, prefill }).catch(() => null)),
    },
    mode: getState().mode ?? "rpc",
    get hasUI() {
      return getState().hasUI === true;
    },
    get cwd() {
      return String(getState().cwd);
    },
    get model() {
      return getState().model;
    },
    get agent() {
      return { kind: getState().agentKind ?? "main", ...((getState().agent as Json) ?? {}) };
    },
    sessionManager: sm(),
    modelRegistry: {
      getApiKey: async () => undefined,
      getAvailable: () => [],
      getAll: () => [],
      complete: async (model: Json, context: Json, options: Json) => request("model_complete", {
        provider: String(model.provider), model: String(model.id), context, options,
      }),
    },
    getContextUsage: () => getState().contextUsage,
    getAsyncJobSnapshot: () => (getState().asyncJobs as Json) ?? null,
    compact: (o?: unknown) => request("compact", { options: o as Json }).then(() => undefined),
    navigateTree: (target: string, o?: unknown) => request("navigate_tree", { target, options: o as Json }),
    isIdle: () => getState().idle !== false,
    abort: () => notify("abort", {}),
    hasPendingMessages: () => getState().pending === true,
    shutdown: () => notify("shutdown_requested", {}),
    isProjectTrusted: () => true,
    getSystemPrompt: () => (getState().systemPrompt as string[]) ?? [],
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
export function safe(cb: (...a: unknown[]) => unknown, a: unknown[]) {
  try {
    void Promise.resolve(cb(...a)).catch((e) => log("warn", `timer callback failed: ${e}`));
  } catch (e) {
    log("warn", `timer callback failed: ${e}`);
  }
}
export function ctx() {
  const cached = getCtxCache();
  if (cached !== undefined && cached !== null) return cached;
  const made = makeCtx();
  setCtxCache(made);
  return made;
}

export function makeApi(ext: Ext) {
  const api = {
    logger: {
      debug: (m: string, x?: Json) => log("debug", m, x),
      info: (m: string, x?: Json) => log("info", m, x),
      warn: (m: string, x?: Json) => log("warn", m, x),
      error: (m: string, x?: Json) => log("error", m, x),
    },
    get typebox() {
      return loadSchemaLib("typebox");
    },
    get zod() {
      return loadSchemaLib("zod");
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
    exec: (command: string, args: string[], options?: Json) => execCommand(command, args, String(options?.cwd ?? getState().cwd), options),
    getActiveTools: () => (getState().activeTools as string[]) ?? [],
    getAllTools: () => (getState().allTools as Json[]) ?? [],
    setActiveTools: async (names: string[]) => void (await request("set_active_tools", { names })),
    getCommands: () => exts.flatMap((e) => [...e.commands.values()]),
    setModel: async (model: unknown) => (await request("set_model", { model })) === true,
    getThinkingLevel: () => getState().thinkingLevel,
    setThinkingLevel: (level: string, persist?: boolean) => notify("set_thinking_level", { level, persist }),
    getServiceTiers: () => ({}),
    setServiceTier: () => {},
    getSessionName: () => getState().sessionName,
    setSessionName: async (name: string) => void (await request("set_session_name", { name })),
  };
  return api;
}

// ---------------------------------------------------------------- loading
