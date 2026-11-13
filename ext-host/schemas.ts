// schemas.ts — lazy schema libs (typebox/zod) resolution (mechanical split from host.ts).
import { createRequire } from "node:module";
import { join } from "node:path";
import { PKG_ROOT } from "./env.ts";
import { log } from "./transport.ts";

const ompRequire = createRequire(join(PKG_ROOT, "src", "extensibility", "noop.js"));
let typebox: unknown;
let zod: unknown;

// `api.typebox` / `api.zod` are imported synchronously on FIRST access (Bun's
// `require` handles TS/ESM). Eagerly importing them at `load` costs ~39 MB RSS
// (omptype + pi-ai's schema barrel) in every session, including sessions whose
// extensions never touch either. typebox is a vendored copy of omp's
// legacy-typebox.ts that pulls two leaf functions instead of pi-ai's whole
// schema index (see legacy-typebox.ts). zod resolution is anchored at the omp
// package so it resolves exactly as omp's own import would.
export function loadSchemaLib(which: "typebox" | "zod"): unknown {
  if (which === "typebox" ? typebox !== undefined : zod !== undefined) return which === "typebox" ? typebox : zod;
  let mod: unknown;
  try {
    mod = which === "typebox" ? require("./legacy-typebox.ts") : ompRequire("@oh-my-pi/omptype/zod");
  } catch (e) {
    log("warn", `${which} unavailable (api.${which} undefined): ${e instanceof Error ? e.message : e}`);
    return undefined;
  }
  if (which === "typebox") typebox = mod;
  else zod = mod;
  return mod;
}
