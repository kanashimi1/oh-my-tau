// emit.ts — core->host emit: rebuild deltas, run handlers, fold results (mechanical split from host.ts).
import type { Json, Ext, HandlerRecord } from "./types.ts";
import { exts, timers, setState } from "./registry.ts";
import { log } from "./transport.ts";
import { TIMEOUT_MS, SHUTDOWN_TIMEOUT_MS, TRACE } from "./env.ts";
import { appendFileSync } from "node:fs";
import { runHandler } from "./handlers.ts";
import { rebuildMessageUpdate, clearUpdateChains } from "./delta.ts";
import { isNonBlank, joinCtx } from "./json.ts";
import { summarize } from "./tools.ts";

export async function emit(p: Json) {
  if (p.ctx) setState(p.ctx as Json);
  // No structuredClone: like the TS runner, every handler for this emit sees
  // this same object (handler-side mutations are visible to later handlers,
  // and the core's copy is independent — it came off the wire).
  const event = p.event as Json;
  const type = String(event.type);
  if (type === "message_update" && !rebuildMessageUpdate(event)) {
    // Unappliable delta: no handler can see a correct message. Ask the core
    // for a full frame (the next update) instead of running handlers on a
    // corrupt one.
    return { result: { resync: true }, handlers: [], errors: [] };
  }
  // Message boundaries carry no encoder id, and the core starts every message
  // with a full frame (which replaces that encoder's chain), so only the end
  // of the session drops chains.
  if (type === "session_shutdown") clearUpdateChains();
  const ame = type === "message_update" ? (event.assistantMessageEvent as Json | undefined) : undefined;
  if (ame?.partialIsMessage === true) {
    // The core did not send `partial` a second time: it equals `message` (same object, as in omp).
    delete ame.partialIsMessage;
    ame.partial = event.message;
  }
  const timeoutMs = Number(p.timeoutMs) || TIMEOUT_MS;
  const rec: HandlerRecord[] = [];
  // `skipExtensions`: extensions (by `ext.name`) whose handlers sit out this one emit
  // (e.g. `context-slim` while the core runs its own CLM context strategy).
  const skip = Array.isArray(p.skipExtensions) ? new Set((p.skipExtensions as unknown[]).map(String)) : undefined;
  const handlersFor = (e: Ext) => (skip?.has(e.name) ? [] : (e.handlers.get(type) ?? []));
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
    // A handler may return a replacement list, or (Pi semantics: handlers get a clone of the
    // conversation) edit that clone in place — SoL-Pi ObservationPack sets `message.content`
    // and returns nothing. Both travel back; an untouched list does not (the core keeps its own).
    const shape = (ms: unknown[]) => ms.map((m) => {
      const c = (m as { content?: unknown } | null)?.content;
      return [m, c, Array.isArray(c) ? c.length : -1, Array.isArray(c) ? c.map((b) => b) : null] as const;
    });
    const before = shape(messages);
    let modified = false;
    for (const e of exts)
      for (const h of handlersFor(e)) {
        const hr = (await runHandler(e, h, { type, messages }, timeoutMs, rec)).value as Json | undefined;
        if (hr && Array.isArray(hr.messages)) ((messages = hr.messages), (modified = true));
      }
    if (!modified) {
      modified = messages.length !== before.length || messages.some((m, i) => {
        const [m0, c0, n0, blocks0] = before[i];
        const c = (m as { content?: unknown } | null)?.content;
        if (m !== m0 || c !== c0) return true;
        if (!Array.isArray(c)) return false;
        return c.length !== n0 || c.some((b, j) => b !== blocks0?.[j]);
      });
    }
    result = modified ? { messages } : undefined;
  } else if (type === "before_agent_start") {
    // Handlers run concurrently: every one gates the prompt, and several of them
    // are network round-trips (recall, planning hints), so a serial chain made
    // the prompt wait for their sum. Each handler sees the system prompt as of
    // the emit; results fold in registration order (messages in order, the last
    // `systemPrompt` replacement wins), as in the serial TS runner unless two
    // handlers both rewrite the system prompt.
    const messages: unknown[] = [];
    const sp0 = event.systemPrompt as string[];
    let sp = sp0;
    let spMod = false;
    const runs = exts.flatMap((e) => handlersFor(e).map((h) => runHandler(e, h, { ...event, systemPrompt: sp0 }, timeoutMs, rec)));
    for (const r of await Promise.all(runs)) {
      const hr = r.value as Json | undefined;
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
