// Shared helpers for the omp-rs vs omp (TS) benchmark harness.
import { spawn, type Subprocess } from "bun";
import { readdirSync, readFileSync } from "node:fs";

export interface RunResult {
  ms: number;
  code: number;
  stdout: string;
  stderr: string;
  peakRssKb: number;
}

export interface BinSpec {
  name: string; // "omp-rs" | "omp"
  path: string;
}

export interface BenchEnv {
  [key: string]: string | undefined;
}

/** Spawn a command, stream stdout/stderr to buffers, sample RSS of the process
 * group until exit. Returns wall time, exit code and peak tree RSS (KB). */
export async function runTimed(
  cmd: string[],
  opts: { cwd: string; env: BenchEnv; timeoutMs?: number; rssSampling?: boolean },
): Promise<RunResult> {
  const proc: Subprocess = spawn(cmd, {
    cwd: opts.cwd,
    env: { ...process.env, ...opts.env } as Record<string, string>,
    stdout: "pipe",
    stderr: "pipe",
    stdin: "ignore",
  });
  const t0 = performance.now();
  let stdout = "";
  let stderr = "";
  let peakRssKb = 0;
  const timeoutMs = opts.timeoutMs ?? 120_000;
  const decoder = new TextDecoder();
  const readLoop = async (stream: ReadableStream<Uint8Array> | null, sink: (s: string) => void) => {
    if (!stream) return;
    const reader = stream.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      sink(decoder.decode(value, { stream: true }));
    }
  };
  const rssLoop = async () => {
    if (!opts.rssSampling || !proc.pid) return;
    const pgid = proc.pid;
    for (;;) {
      if ((proc.exited as Promise<number> | undefined) === undefined) break;
      const done = (proc.exited as Promise<number>).then(
        () => true,
        () => true,
      );
      const tick = Promise.withResolvers<boolean>();
      setTimeout(() => tick.resolve(false), 50);
      if (await Promise.race([done, tick.promise])) break;
      peakRssKb = Math.max(peakRssKb, await treeRssKb(pgid));
    }
  };
  const timer = Promise.withResolvers<null>();
  const timeout = setTimeout(() => {
    try {
      process.kill(-proc.pid!, "SIGKILL");
    } catch {
      try {
        proc.kill("SIGKILL");
      } catch {
        /* already dead */
      }
    }
    timer.resolve(null);
  }, timeoutMs);
  const [code] = await Promise.all([
    Promise.race([proc.exited, timer.promise.then(() => -1)]),
    readLoop(proc.stdout as ReadableStream<Uint8Array> | null, (s) => (stdout += s)),
    readLoop(proc.stderr as ReadableStream<Uint8Array> | null, (s) => (stderr += s)),
    rssLoop(),
  ]);
  clearTimeout(timeout);
  // final RSS sample (process tree may still exist briefly)
  if (opts.rssSampling && proc.pid) {
    peakRssKb = Math.max(peakRssKb, await treeRssKb(proc.pid));
  }
  return { ms: performance.now() - t0, code: code ?? -1, stdout, stderr, peakRssKb };
}

interface ProcStat {
  ppid: number;
  comm: string;
  cmdline: string;
}

const procTableCache: { at: number; table: Map<number, ProcStat> } = { at: 0, table: new Map() };

/** ppid/comm/cmdline for every live pid; cached briefly because sampling loops
 * call this in quick succession. stat only (one read per pid) — no statm here. */
function readProcStats(): Map<number, ProcStat> {
  if (performance.now() - procTableCache.at < 750) return procTableCache.table;
  const table = new Map<number, ProcStat>();
  let entries: string[];
  try {
    entries = readdirSync("/proc");
  } catch {
    return table;
  }
  for (const name of entries) {
    if (!/^\d+$/.test(name)) continue;
    const pid = Number(name);
    try {
      const stat = readFileSync(`/proc/${name}/stat`, "utf8");
      const close = stat.lastIndexOf(")");
      const fields = stat.slice(close + 2).split(" ");
      // fields[0] = state (index 2 in proc(5)); ppid is index 3.
      const ppid = Number(fields[1]);
      if (!Number.isFinite(ppid)) continue;
      let comm = "";
      let cmdline = "";
      try {
        comm = readFileSync(`/proc/${name}/comm`, "utf8").trim();
      } catch {
        /* gone */
      }
      try {
        cmdline = readFileSync(`/proc/${name}/cmdline`, "utf8").replace(/\0/g, " ").trim();
      } catch {
        /* gone */
      }
      table.set(pid, { ppid, comm, cmdline });
    } catch {
      /* process vanished */
    }
  }
  procTableCache.at = performance.now();
  procTableCache.table = table;
  return table;
}

/** PSS (KB) of one pid from smaps_rollup; 0 if unavailable. */
function pssKbOf(pid: number): number {
  try {
    const rollup = readFileSync(`/proc/${pid}/smaps_rollup`, "utf8");
    const m = /(^|\n)Pss:\s+(\d+) kB/.exec(rollup);
    return m ? Number(m[2]) : 0;
  } catch {
    return 0;
  }
}

/** Sum PSS (KB) over a process and all its descendants. PSS splits shared
 * pages fairly, so summing trees that share libraries does not double-count. */
export async function treePssKb(pid: number): Promise<number> {
  const table = procTableWith(pid);
  let total = 0;
  for (const [p] of table) {
    let cur: number | undefined = p;
    let depth = 0;
    while (cur !== undefined && depth < 64) {
      if (cur === pid) {
        total += pssKbOf(p);
        break;
      }
      cur = table.get(cur)?.ppid;
      depth++;
    }
  }
  return total;
}

export interface ProcEntry {
  pid: number;
  comm: string;
  cmdline: string;
  /** "core" | "ext-host" | "eval" | "lsp" | "other" */
  role: string;
  rssKb: number;
  pssKb: number;
}

/** Coarse classification of a child process by comm/cmdline. */
function classifyRole(comm: string, cmdline: string, rootComm: string): string {
  const cl = cmdline.toLowerCase();
  if (comm === rootComm) return "core";
  if (cl.includes("ext-host") || cl.includes("host.ts")) return "ext-host";
  if (cl.includes("eval") || cl.includes("kernel")) return "eval";
  if (cl.includes("lsp") || cl.includes("language-server") || cl.includes("typescript-language-server")) return "lsp";
  return "other";
}

/** Per-process RSS/PSS breakdown of a process tree, grouped by role. */
export async function treeBreakdown(pid: number): Promise<{ procs: ProcEntry[]; totalRssKb: number; totalPssKb: number }> {
  const table = procTableWith(pid);
  const root = table.get(pid);
  const rootComm = root?.comm ?? "";
  const procs: ProcEntry[] = [];
  for (const [p, st] of table) {
    let cur: number | undefined = p;
    let depth = 0;
    while (cur !== undefined && depth < 64) {
      if (cur === pid) {
        procs.push({
          pid: p,
          comm: st.comm,
          cmdline: st.cmdline.slice(0, 200),
          role: p === pid ? "core" : classifyRole(st.comm, st.cmdline, rootComm),
          rssKb: rssKbOf(p),
          pssKb: pssKbOf(p),
        });
        break;
      }
      cur = table.get(cur)?.ppid;
      depth++;
    }
  }
  return {
    procs,
    totalRssKb: procs.reduce((a, b) => a + b.rssKb, 0),
    totalPssKb: procs.reduce((a, b) => a + b.pssKb, 0),
  };
}

function rssKbOf(pid: number): number {
  try {
    const statm = readFileSync(`/proc/${pid}/statm`, "utf8").split(" ");
    return Math.round((Number(statm[1]) * 4096) / 1024);
  } catch {
    return 0;
  }
}

/** The proc table cache can be up to 750ms stale; a freshly spawned process
 * may be missing entirely, which silently zeroes a whole measurement. */
function procTableWith(pid: number): Map<number, ProcStat> {
  let table = readProcStats();
  if (!table.has(pid)) {
    procTableCache.at = 0;
    table = readProcStats();
  }
  return table;
}

/** Sum RSS (KB) over a process and all its descendants, reading /proc directly
 * (no subprocess spawning — the ps-based version leaked zombies and slowed down). */
export async function treeRssKb(pid: number): Promise<number> {
  const table = procTableWith(pid);
  const matched = new Set<number>();
  for (const [p] of table) {
    let cur: number | undefined = p;
    let depth = 0;
    while (cur !== undefined && depth < 64) {
      if (cur === pid) {
        matched.add(p);
        break;
      }
      cur = table.get(cur)?.ppid;
      depth++;
    }
  }
  let total = 0;
  for (const p of matched) total += rssKbOf(p);
  return total;
}

export function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return NaN;
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, idx)];
}

export interface Stats {
  n: number;
  median: number;
  p95: number;
  min: number;
  max: number;
}

export function stats(values: number[]): Stats {
  const s = [...values].sort((a, b) => a - b);
  return {
    n: s.length,
    median: percentile(s, 50),
    p95: percentile(s, 95),
    min: s[0] ?? NaN,
    max: s[s.length - 1] ?? NaN,
  };
}

/** Newline-delimited-JSON RPC client over stdio (omp --mode rpc protocol v1). */
export class RpcClient {
  #proc: Subprocess;
  #reader: ReadableStreamDefaultReader<Uint8Array>;
  #decoder = new TextDecoder();
  #buf = "";
  #nextId = 1;
  pid: number;

  private constructor(proc: Subprocess) {
    this.#proc = proc;
    this.pid = proc.pid!;
    this.#reader = (proc.stdout as ReadableStream<Uint8Array>).getReader();
  }

  static async start(
    cmd: string[],
    opts: { cwd: string; env: BenchEnv; timeoutMs?: number },
  ): Promise<RpcClient> {
    const proc = spawn(cmd, {
      cwd: opts.cwd,
      env: { ...process.env, ...opts.env } as Record<string, string>,
      stdout: "pipe",
      stderr: process.env.BENCH_STDERR_INHERIT ? "inherit" : "pipe",
      stdin: "pipe",
    });
    const client = new RpcClient(proc);
    const timeoutMs = opts.timeoutMs ?? 120_000;
    setTimeout(() => {
      try {
        process.kill(-proc.pid!, "SIGKILL");
      } catch {
        try {
          proc.kill("SIGKILL");
        } catch {
          /* already dead */
        }
      }
    }, timeoutMs).unref();
    // Both binaries emit a first `ready` frame declaring the protocol version.
    for (;;) {
      const frame = await client.#readFrame();
      if (frame["type"] === "ready") break;
    }
    return client;
  }

  async #readFrame(): Promise<Record<string, unknown>> {
    for (;;) {
      const nl = this.#buf.indexOf("\n");
      if (nl >= 0) {
        const line = this.#buf.slice(0, nl);
        this.#buf = this.#buf.slice(nl + 1);
        if (line.trim().length > 0) return JSON.parse(line) as Record<string, unknown>;
        continue;
      }
      const { done, value } = await this.#reader.read();
      if (done) throw new Error("rpc process closed stdout");
      this.#buf += this.#decoder.decode(value, { stream: true });
    }
  }

  /** Send a request; returns the matching response frame (or the request id frame). */
  async call(method: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    const id = this.#nextId++;
    const line = JSON.stringify({ id, method, type: method, ...params }) + "\n";
    const stdin = this.#proc.stdin as unknown as { write: (s: string) => void; flush?: () => void };
    stdin.write(line);
    stdin.flush?.();
    for (;;) {
      const frame = await this.#readFrame();
      if (frame["id"] === id && (frame["type"] === "response" || frame["method"] === undefined)) {
        return frame;
      }
      // event frames (agent_start etc.) are skipped
    }
  }

  /** Send a prompt and wait for the prompt_result frame with the same id. */
  async prompt(message: string, timeoutMs = 120_000): Promise<{ ms: number; status: string }> {
    const id = this.#nextId++;
    const line = JSON.stringify({ id, method: "prompt", type: "prompt", message }) + "\n";
    const t0 = performance.now();
    const stdin = this.#proc.stdin as unknown as { write: (s: string) => void; flush?: () => void };
    stdin.write(line);
    stdin.flush?.();
    // Reject instead of hanging forever: long-lived sessions can stall (seen
    // with omp-rs + extensions after a few dozen turns) and the outer kill
    // only fires at the whole-run timeout.
    let fail: (e: Error) => void;
    const timeout = new Promise<never>((_, reject) => {
      fail = reject;
      setTimeout(() => reject(new Error(`prompt timed out after ${timeoutMs}ms`)), timeoutMs).unref();
    });
    const wait = (async () => {
      for (;;) {
        const frame = await this.#readFrame();
        if (
          frame["id"] === id &&
          (frame["type"] === "prompt_result" || (frame["type"] === "response" && frame["status"] !== undefined))
        ) {
          return { ms: performance.now() - t0, status: String(frame["status"] ?? "completed") };
        }
      }
    })();
    const result = await Promise.race([wait, timeout]);
    void fail!;
    return result;
  }

  async stop(): Promise<void> {
    try {
      this.#proc.kill("SIGKILL");
    } catch {
      /* already dead */
    }
    await this.#proc.exited.catch(() => -1);
  }

  async rssNow(): Promise<number> {
    return treeRssKb(this.pid);
  }
}
