// env.ts — environment + package root resolution (mechanical split from host.ts).
import { join } from "node:path";
import { homedir } from "node:os";

// `load` may additionally override TIMEOUT_MS/TRACE per session (the daemon's env is shared).
export let TIMEOUT_MS = Number(process.env.OMP_EXT_TIMEOUT_MS) || 30_000;
export const SHUTDOWN_TIMEOUT_MS = 2_000;
export let TRACE = process.env.OMP_EXT_TRACE;
export const PKG_ROOT =
  process.env.OMP_PKG_ROOT ?? join(homedir(), ".bun/install/global/node_modules/@oh-my-pi/pi-coding-agent");

export function setTimeoutMs(ms: number): void { TIMEOUT_MS = ms; }
export function setTrace(path: string): void { TRACE = path; }
