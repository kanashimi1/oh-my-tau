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
//
// NOTE: the implementation lives in the sibling modules (env/types/json/transport/registry/schemas/
// process-util/context/loader/handlers/delta/tools/emit/rpc); this file only wires transport + dispatches lines.
import { onMessage } from "./rpc.ts";
import { log, setEmitLine, setCloseSession } from "./transport.ts";
import { getState } from "./registry.ts";

process.on("uncaughtException", (e) => log("error", `uncaughtException: ${e instanceof Error ? (e.stack ?? e.message) : e}`));
process.on("unhandledRejection", (e) => log("error", `unhandledRejection: ${e instanceof Error ? (e.stack ?? e.message) : e}`));

export interface SessionTransport {
  /** Send one newline-terminated JSON-RPC line to the core. */
  send(line: string): void;
  /** Called with each raw incoming chunk (newline framing happens here). */
  onData(cb: (chunk: string) => void): void;
  /** Input ended / core disconnected: the session is over. */
  onClose?(cb: () => void): void;
  /** Host-initiated session end (shutdown RPC): close this session only. */
  close?(): void;
}

/** The session's cwd as last pushed by the core (shared-daemon Workers route
 *  `process.cwd()` and child-process defaults through it). */
export function sessionCwd(): string {
  return String(getState().cwd);
}

/**
 * Start serving one session on `t`. Each call creates no explicit state -- the
 * module-level state above belongs to the importing isolate, so a fresh Worker
 * per session gets isolation equivalent to a separate process.
 */
export function startSession(t: SessionTransport): void {
  setEmitLine(t.send);
  setCloseSession(t.close ?? null);
  let buf = "";
  t.onData((chunk: string) => {
    buf += chunk;
    let i: number;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) void onMessage(line);
    }
  });
  t.onClose?.(() => process.exit(0));
}

if (import.meta.main) {
  // stdio entry point (one bun process per session; the default).
  process.stdin.setEncoding("utf8");
  startSession({
    send: (line) => process.stdout.write(line),
    onData: (cb) => process.stdin.on("data", cb),
    onClose: (cb) => process.stdin.on("end", cb),
  });
}

