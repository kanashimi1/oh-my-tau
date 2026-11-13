// delta.ts — message_update delta chains and rebuild (mechanical split from host.ts).
import type { Json, UpdateChain } from "./types.ts";
import { DeltaError, STREAM_KEY } from "./types.ts";
import { copyJson, setOwn } from "./json.ts";

const updateChains = new Map<string, UpdateChain>();
// Encoders that went away (dropped subscriptions) leave their last chain
// behind; evicting the oldest only costs that encoder a resync.
const MAX_UPDATE_CHAINS = 16;

export function setUpdateChain(stream: string, chain: UpdateChain): void {
  updateChains.delete(stream);
  updateChains.set(stream, chain);
  if (updateChains.size > MAX_UPDATE_CHAINS) updateChains.delete(updateChains.keys().next().value as string);
}

export function applyValuePatch(prev: unknown, p: Json): unknown {
  if ("=" in p) return p["="];
  if ("+" in p) {
    if (typeof prev !== "string" || typeof p["+"] !== "string") throw new DeltaError("string append on a non-string");
    return prev + p["+"];
  }
  if ("o" in p) {
    if (!prev || typeof prev !== "object" || Array.isArray(prev)) throw new DeltaError("object patch on a non-object");
    const out: Json = { ...(prev as Json) };
    for (const [k, sp] of Object.entries(p.o as Json)) setOwn(out, k, applyValuePatch((prev as Json)[k], sp as Json));
    return out;
  }
  if ("a" in p) {
    if (!Array.isArray(prev) || typeof p.n !== "number") throw new DeltaError("array patch on a non-array");
    const out = prev.slice();
    out.length = p.n;
    for (const [i, sp] of p.a as [number, Json][]) out[i] = applyValuePatch(prev[i], sp);
    return out;
  }
  throw new DeltaError("unknown patch");
}

/**
 * Restore `event.message` from a delta frame (and drop the `delta` field).
 * `false` when the frame does not apply to the chain this host holds (gap,
 * unknown id, malformed entry): the chain is dropped and the core resends full.
 */
export function rebuildMessageUpdate(event: Json): boolean {
  const d = event.delta as Json | undefined;
  if (d === undefined) return true;
  delete event.delta;
  const stream = String(d.s);
  if (event.message !== undefined) {
    if (typeof d.id !== "number") {
      updateChains.delete(stream);
      return true;
    }
    setUpdateChain(stream, { id: d.id, seq: 0, message: event.message as Json });
    event.message = copyJson(event.message);
    return true;
  }
  const chain = updateChains.get(stream);
  updateChains.delete(stream);
  if (!chain || chain.id !== d.id || chain.seq + 1 !== d.seq) return false;
  try {
    const ame = event.assistantMessageEvent as Json | undefined;
    const prev = (chain.message.content ?? []) as Json[];
    const content = prev.slice();
    for (const e of (d.b ?? []) as Json[]) {
      const i = e.i as number;
      if (!Number.isInteger(i) || i < 0 || i > content.length) throw new DeltaError(`block index ${i}`);
      const old = prev[i] as Json | undefined;
      const base = (e.blk as Json | undefined) ?? old;
      const key = base ? STREAM_KEY[String(base.type)] : undefined;
      if (!base || !key) throw new DeltaError(`block ${i}: unknown base`);
      const oldValue = old && old.type === base.type ? old[key] : undefined;
      let v: unknown;
      switch (e.v) {
        case "k":
          if (oldValue === undefined) throw new DeltaError(`block ${i}: keep without a previous value`);
          v = oldValue;
          break;
        case "a":
          if (typeof oldValue !== "string" || typeof ame?.delta !== "string") throw new DeltaError(`block ${i}: bad append`);
          v = oldValue + ame.delta;
          break;
        case "+":
          if (typeof oldValue !== "string" || typeof e.x !== "string") throw new DeltaError(`block ${i}: bad append`);
          v = oldValue + e.x;
          break;
        case "=":
          v = e.x;
          break;
        case "p":
          v = applyValuePatch(oldValue, e.x as Json);
          break;
        default:
          throw new DeltaError(`block ${i}: unknown op ${String(e.v)}`);
      }
      // `base` carries the streamed key (skeleton or previous block), so the
      // assignment keeps a full frame's key order.
      const block: Json = { ...base };
      block[key] = v;
      content[i] = block;
    }
    if (content.length !== d.n) throw new DeltaError(`content length ${content.length} != ${String(d.n)}`);
    const message: Json = d.hdr !== undefined ? { ...(d.hdr as Json), content } : { ...chain.message, content };
    // Key order of a full frame / omp's event: type, message, assistantMessageEvent.
    const a = event.assistantMessageEvent;
    delete event.assistantMessageEvent;
    event.message = copyJson(message);
    event.assistantMessageEvent = a;
    setUpdateChain(stream, { id: chain.id, seq: d.seq as number, message });
    return true;
  } catch (e) {
    if (!(e instanceof DeltaError)) throw e;
    log("warn", `message_update delta rejected (${e.message}); core will resend the full message`);
    return false;
  }
}

/** Drop all delta chains (session_shutdown). */
export function clearUpdateChains(): void {
  updateChains.clear();
}
