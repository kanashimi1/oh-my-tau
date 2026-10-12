#!/usr/bin/env bun
// Shared extension-host daemon (opt-in via OMP_RS_EXT_SHARED=1): ONE bun process
// per user serves any number of omp-rs sessions. Each accepted unix-socket
// connection gets its own Worker running ext-host/host.ts (session.worker.ts),
// so extension state stays per-session exactly as with separate processes;
// what is shared is the bun runtime baseline (and, lazily, the compiled
// schema-lib code cache).
//
// Lifecycle: started on demand by the Rust core (which retries connecting),
// exits a short grace period after the LAST session disconnects
// (OMP_RS_EXT_SHARED_IDLE_MS, default 2000; the socket file is removed on
// exit). Per-session settings (trace file, handler timeout) travel with the
// `load` RPC, not the environment, because the daemon env is process-wide.
//
// Env: OMP_RS_EXT_SOCKET (required) — unix socket path. The Rust core removes
// a stale socket file before spawning; if another daemon wins the bind race,
// this instance exits and the core connects to the winner.
const SOCKET = process.env.OMP_RS_EXT_SOCKET;
if (!SOCKET) {
  process.stderr.write("shard.ts: OMP_RS_EXT_SOCKET is required\n");
  process.exit(2);
}
const IDLE_MS = Number(process.env.OMP_RS_EXT_SHARED_IDLE_MS ?? 2_000);
const CLOSE = " __omp_close_session__";

import { existsSync, unlinkSync } from "node:fs";

/** Daemon timer handle (bun returns a NodeJS-style Timeout). */
type Timer = ReturnType<typeof setTimeout>;

let sessions = 0;
let idleTimer: Timer | null = null;

function armIdleExit(): void {
  if (sessions > 0 || IDLE_MS < 0) return;
  idleTimer = setTimeout(() => {
    try {
      if (existsSync(SOCKET!)) unlinkSync(SOCKET!);
    } catch {}
    process.stderr.write("shard: last session disconnected, exiting\n");
    process.exit(0);
  }, IDLE_MS);
  idleTimer.unref?.();
}

const server = Bun.listen({
  unix: SOCKET,
  socket: {
    open(socket) {
      sessions += 1;
      if (idleTimer) {
        clearTimeout(idleTimer);
        idleTimer = null;
      }
      const worker = new Worker(new URL("./session.worker.ts", import.meta.url));
      const state = { worker, socket, open: true };
      worker.onmessage = (ev: MessageEvent<string>) => {
        const data = String(ev.data ?? "");
        if (data === CLOSE) {
          // Session host answered the shutdown RPC: end only this session.
          setTimeout(() => {
            state.open = false;
            try {
              socket.end();
            } catch {}
            try {
              worker.terminate();
            } catch {}
            closeSession();
          }, 10);
          return;
        }
        try {
          socket.write(data);
        } catch {
          cleanup();
        }
      };
      worker.onerror = (e) => {
        process.stderr.write(`shard: session worker error: ${e?.message ?? e}\n`);
        cleanup();
      };
      worker.addEventListener("close", () => {
        // Worker gone (crash or close): drop the connection.
        cleanup();
      });
      const cleanup = () => {
        if (!state.open) return;
        state.open = false;
        try {
          socket.end();
        } catch {}
        try {
          worker.terminate();
        } catch {}
        closeSession();
      };
      // socket handlers
      socket.data = state;
    },
    data(socket, data) {
      const st = socket.data as { worker: Worker; open: boolean };
      if (!st.open) return;
      // Forward raw bytes; the Worker re-assembles newline framing.
      st.worker.postMessage(data.toString());
    },
    close(socket) {
      const st = socket.data as { worker: Worker; open: boolean };
      if (!st.open) return;
      st.open = false;
      try {
        st.worker.terminate();
      } catch {}
      closeSession();
    },
    error(socket, error) {
      process.stderr.write(`shard: socket error: ${error}\n`);
      const st = socket.data as { worker: Worker; open: boolean };
      if (st.open) {
        st.open = false;
        try {
          st.worker.terminate();
        } catch {}
        closeSession();
      }
    },
  },
});

function closeSession(): void {
  sessions = Math.max(0, sessions - 1);
  if (sessions === 0) armIdleExit();
}

process.on("uncaughtException", (e) =>
  process.stderr.write(`shard: uncaughtException: ${e instanceof Error ? (e.stack ?? e.message) : e}\n`),
);
process.on("unhandledRejection", (e) =>
  process.stderr.write(`shard: unhandledRejection: ${e instanceof Error ? (e.stack ?? e.message) : e}\n`),
);

armIdleExit();
