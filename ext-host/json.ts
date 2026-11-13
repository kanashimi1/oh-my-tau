// json.ts — JSON helpers (mechanical split from host.ts).
import type { Json } from "./types.ts";

/** Set `key` on `o` as an own data property (also for `__proto__`, as JSON.parse does). */
export function setOwn(o: Json, key: string, v: unknown): void {
  if (key === "__proto__") Object.defineProperty(o, key, { value: v, writable: true, enumerable: true, configurable: true });
  else o[key] = v;
}

/** Deep copy of a JSON value: fresh objects and arrays, strings shared. */
export function copyJson(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(copyJson);
  if (v === null || typeof v !== "object") return v;
  const out: Json = {};
  for (const [k, x] of Object.entries(v)) setOwn(out, k, copyJson(x));
  return out;
}

export const isNonBlank = (s: unknown): s is string => typeof s === "string" && s.trim().length > 0;
export const joinCtx = (xs: string[]) => {
  const u = [...new Set(xs)];
  return u.length ? u.join("\n\n") : undefined;
};
