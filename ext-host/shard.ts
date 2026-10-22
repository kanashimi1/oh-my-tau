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

interface SessionState {
  worker: Worker | null;
  /** Bytes received before the env handshake line completed. */
  pre: Uint8Array;
  open: boolean;
  /** Bytes the kernel did not accept yet, flushed on `drain` in order. */
  out: Uint8Array[];
}

/**
 * Bun's `socket.write` is non-buffering: under backpressure it writes a prefix
 * and returns its length, dropping the rest unless the caller keeps it. A
 * large frame (a `context` event with a long history, a big tool result) was
 * truncated mid-line and the client waited forever for its newline.
 */
function flush(socket: { write(b: Uint8Array): number }, st: SessionState): void {
  while (st.out.length > 0) {
    const head = st.out[0];
    const n = socket.write(head);
    if (n >= head.byteLength) {
      st.out.shift();
      continue;
    }
    if (n > 0) st.out[0] = head.subarray(n);
    return;
  }
}

/** First line of every connection: `{"env": {...}}`, the client's environment. */
const ENV_HANDSHAKE = "omp-rs-env";

/**
 * Session bytes travel to the Worker as transferred Uint8Arrays (decoded and
 * framed there): bun retained every string posted to a Worker, so a session
 * fed ~0.5 MB `context` frames grew ~90 KB per turn until it closed.
 */
function startWorker(socket: any, state: SessionState, env: Record<string, string>, rest: Uint8Array): void {
  // The Worker gets the CLIENT's environment (HOME, OMP_*, HOOK_TRACE, pane
  // ids, ...): the daemon's own env belongs to whichever session started it.
  const worker = new Worker(new URL("./session.worker.ts", import.meta.url), { env });
  state.worker = worker;
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
  worker.onmessage = (ev: MessageEvent<string>) => {
    const data = String(ev.data ?? "");
    if (data === CLOSE) {
      // Session host answered the shutdown RPC: end only this session.
      setTimeout(cleanup, 10);
      return;
    }
    try {
      state.out.push(Buffer.from(data));
      flush(socket, state);
    } catch {
      cleanup();
    }
  };
  worker.onerror = (e) => {
    process.stderr.write(`shard: session worker error: ${e?.message ?? e}\n`);
    cleanup();
  };
  // Worker gone (crash or close): drop the connection.
  worker.addEventListener("close", cleanup);
  if (rest.byteLength) worker.postMessage(rest, [rest.buffer]);
}

function endSession(st: SessionState): void {
  if (!st.open) return;
  st.open = false;
  try {
    st.worker?.terminate();
  } catch {}
  closeSession();
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
      socket.data = { worker: null, pre: new Uint8Array(0), open: true, out: [] } satisfies SessionState;
    },
    drain(socket) {
      flush(socket, socket.data as SessionState);
    },
    data(socket, data) {
      const st = socket.data as SessionState;
      if (!st.open) return;
      if (st.worker) {
        const bytes = new Uint8Array(data); // own copy; ownership moves to the Worker
        st.worker.postMessage(bytes, [bytes.buffer]);
        return;
      }
      const pre = new Uint8Array(st.pre.byteLength + data.byteLength);
      pre.set(st.pre);
      pre.set(data, st.pre.byteLength);
      const nl = pre.indexOf(0x0a);
      if (nl < 0) {
        st.pre = pre;
        return;
      }
      let env: Record<string, string> = {};
      try {
        const hs = JSON.parse(new TextDecoder().decode(pre.subarray(0, nl)));
        if (hs?.method === ENV_HANDSHAKE && hs.params?.env) env = hs.params.env;
      } catch {}
      st.pre = new Uint8Array(0);
      startWorker(socket, st, env, pre.slice(nl + 1));
    },
    close(socket) {
      endSession(socket.data as SessionState);
    },
    error(socket, error) {
      process.stderr.write(`shard: socket error: ${error}\n`);
      endSession(socket.data as SessionState);
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
