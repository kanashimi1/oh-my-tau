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

// Child processes: Bun spawns children with the NATIVE (daemon) environment,
// which is empty, and the daemon's cwd, not the Worker's `process.env`.
// Extensions shelling out (`herdr`, `git`, ...) then ran
// without HOME/PATH/HERDR_* and could hang the session. Default every spawn
// to this session's env and cwd unless the caller passes its own; the cwd
// follows host.ts's session state once it is loaded.
type SpawnOpts = Record<string, unknown>;
type SpawnFn = (...args: unknown[]) => unknown;
let hostCwd: (() => string) | null = null;
const nativeCwd = process.cwd.bind(process);
process.cwd = () => hostCwd?.() || process.env.PWD || nativeCwd();
function isOpts(o: unknown): o is SpawnOpts {
  return o !== null && typeof o === "object" && !Array.isArray(o);
}
function withSessionDefaults(o: SpawnOpts | undefined): SpawnOpts {
  const r: SpawnOpts = { ...o };
  if (r.env === undefined) r.env = process.env;
  if (r.cwd === undefined) r.cwd = process.cwd();
  return r;
}
// node:child_process is a mutable CJS object; extensions are imported later
// (dynamic import of host.ts), so their named imports bind to these wrappers.
const cp = require("node:child_process") as Record<string, SpawnFn>;
for (const [name, hasArgs] of [
  ["spawn", true],
  ["spawnSync", true],
  ["execFile", true],
  ["execFileSync", true],
  ["fork", true],
  ["exec", false],
  ["execSync", false],
] as const) {
  const orig = cp[name];
  cp[name] = function (this: unknown, ...a: unknown[]) {
    const i = hasArgs && Array.isArray(a[1]) ? 2 : 1;
    const slot = a[i];
    if (isOpts(slot)) a[i] = withSessionDefaults(slot);
    else if (slot === undefined) a[i] = withSessionDefaults(undefined);
    else a.splice(i, 0, withSessionDefaults(undefined)); // callback sits in the options slot
    return orig.apply(this, a);
  };
}
// Bun.spawn / Bun.spawnSync: `(cmd[], opts?)` or `({ cmd, ... })`.
const bunSpawners = Bun as unknown as Record<"spawn" | "spawnSync", SpawnFn>;
for (const name of ["spawn", "spawnSync"] as const) {
  const orig = bunSpawners[name];
  bunSpawners[name] = (...a: unknown[]) => {
    const [first, second] = a;
    if (Array.isArray(first)) a[1] = withSessionDefaults(isOpts(second) ? second : undefined);
    else if (isOpts(first)) a[0] = withSessionDefaults(first);
    return orig.apply(Bun, a);
  };
}

// Raw session bytes from the daemon (transferred Uint8Arrays); host.ts does
// the newline framing. Chunks that arrive before host.ts finished importing
// are replayed in order.
let feed: ((chunk: string) => void) | null = null;
const early: string[] = [];
const decoder = new TextDecoder();
self.onmessage = (ev: MessageEvent<Uint8Array>) => {
  const chunk = decoder.decode(ev.data, { stream: true });
  if (feed) feed(chunk);
  else early.push(chunk);
};

const { startSession, sessionCwd } = await import("./host.ts");
hostCwd = sessionCwd;
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
