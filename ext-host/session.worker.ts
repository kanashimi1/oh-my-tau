// Shared-daemon session Worker: runs the full extension host (host.ts) for ONE
// session inside the shared bun process. Bun gives each Worker its own isolate
// and its own module-graph instance, so all module-level state in host.ts
// (extensions, bus, timers, flags, providers, ctx cache, schema-lib caches)
// is per-session, exactly as in a separate per-session process.
//
// Transport: each postMessage is one newline-terminated JSON-RPC line; the
// daemon (shard.ts) maps it onto the session's unix socket. The sentinel
// message SHARD_CLOSE asks the daemon to end just this session (shutdown RPC).
import { startSession } from "./host.ts";

const SHARD_CLOSE = " __omp_close_session__";

let feed: (chunk: string) => void = () => {};
let buf = "";
self.onmessage = (ev: MessageEvent<string>) => {
  const data = String(ev.data ?? "");
  // The daemon may batch several lines into one message.
  buf += data;
  let i: number;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i).trim();
    buf = buf.slice(i + 1);
    if (line) feed(`${line}\n`);
  }
};

startSession({
  send(line: string) {
    postMessage(line);
  },
  onData(cb) {
    feed = cb;
  },
  close() {
    postMessage(SHARD_CLOSE);
  },
});

export {};
