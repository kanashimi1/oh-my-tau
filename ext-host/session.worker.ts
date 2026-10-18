// Shared-daemon session Worker: runs the full extension host (host.ts) for ONE
// session inside the shared bun process. Bun gives each Worker its own isolate
// and its own module-graph instance, so all module-level state in host.ts
// (extensions, bus, timers, flags, providers, ctx cache, schema-lib caches)
// is per-session, exactly as in a separate per-session process.
//
// Environment: the daemon is started with an empty env and each Worker gets the
// owning client's env (`process.env`, via the Worker `env` option). Bun's
// `os.homedir()` reads the native env, so it is redirected to the session's
// HOME before any module that imports it is evaluated.
//
// Transport: each postMessage is one newline-terminated JSON-RPC line; the
// daemon (shard.ts) maps it onto the session's unix socket. The sentinel
// message SHARD_CLOSE asks the daemon to end just this session (shutdown RPC).
const SHARD_CLOSE = " __omp_close_session__";

const nodeOs = require("node:os");
const nativeHomedir: () => string = nodeOs.homedir;
nodeOs.homedir = () => process.env.HOME || nativeHomedir();

let feed: ((chunk: string) => void) | null = null;
const early: string[] = [];
let buf = "";
self.onmessage = (ev: MessageEvent<string>) => {
  // The daemon may batch several lines into one message.
  buf += String(ev.data ?? "");
  let i: number;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i).trim();
    buf = buf.slice(i + 1);
    if (!line) continue;
    if (feed) feed(`${line}\n`);
    else early.push(`${line}\n`);
  }
};

const { startSession } = await import("./host.ts");
startSession({
  send(line: string) {
    postMessage(line);
  },
  onData(cb) {
    feed = cb;
    for (const l of early.splice(0)) cb(l);
  },
  close() {
    postMessage(SHARD_CLOSE);
  },
});

export {};
