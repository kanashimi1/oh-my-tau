#!/usr/bin/env bun
// Benchmark harness: omp-rs vs omp (TS), PLAN 'eval производительности' metrics.
// All LLM traffic goes to bench/mock-server.ts (no network).
//
// Usage:
//   bun bench/run.ts --quick        # smoke run, low repeats
//   bun bench/run.ts                # full run, >=10 repeats where applicable
//   bun bench/run.ts --scenario 1,3 # run only listed scenarios
//
// Output: bench/results/<timestamp>.json + bench/results/<timestamp>.md

import { spawn, type Subprocess } from "bun";
import { mkdirSync, rmSync, writeFileSync, symlinkSync, copyFileSync, existsSync, readdirSync, statSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import {
  runTimed,
  RpcClient,
  stats,
  treeRssKb,
  treePssKb,
  treeBreakdown,
  type BenchEnv,
  type BinSpec,
  type Stats,
} from "./lib.ts";

const BENCH_DIR = resolve(import.meta.dir);
const REPO_ROOT = resolve(BENCH_DIR, "..");
const TS_OMP = process.env.OMP_TS_BIN ?? "omp";
const RS_OMP = process.env.OMP_RS_BIN ?? join(REPO_ROOT, "target/shared/release/omt");
const WORKDIR = process.env.BENCH_WORKDIR ?? REPO_ROOT;
const REAL_AGENT_DIR = process.env.OMP_REAL_AGENT_DIR ?? `${process.env.HOME}/.omp/agent`;
const EXT_HOST = process.env.OMP_RS_EXT_HOST ?? join(REPO_ROOT, "ext-host/host.ts");
const MOCK_PORT = Number(process.env.MOCK_PORT ?? 48731);
const NOW = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const RESULTS_DIR = join(BENCH_DIR, "results");

const args = process.argv.slice(2);
const QUICK = args.includes("--quick");
const REPEATS = QUICK ? 2 : Number(process.env.BENCH_REPEATS ?? 12);
const scIdx = args.indexOf("--scenario");
let ONLY: Set<string> | null = scIdx >= 0 ? new Set(args[scIdx + 1].split(",").map((s) => s.trim())) : null;
// --only mem: memory scenarios m1 (per-session breakdown), m2 (N-session fleet
// RSS/PSS), m3 (resume RSS+wall). Usage: bun bench/run.ts --quick --only mem
if (args.includes("--only") && args[args.indexOf("--only") + 1] === "mem") {
  ONLY = new Set(["m1", "m2", "m3"]);
}

const BINS: BinSpec[] = [
  { name: "omp-rs", path: RS_OMP },
  { name: "omp", path: TS_OMP },
];

// ---------- temp agent dir ----------

interface AgentHome {
  root: string;
  env: BenchEnv;
  enableExtensions: (on: boolean) => void;
}

function makeAgentHome(tag: string): AgentHome {
  const root = join(BENCH_DIR, "tmp", `${NOW}-${tag}`);
  const agentDir = join(root, ".omp", "agent");
  mkdirSync(join(agentDir, "sessions"), { recursive: true });
  writeFileSync(
    join(agentDir, "models.yml"),
    [
      "providers:",
      "  mock:",
      `    baseUrl: http://127.0.0.1:${MOCK_PORT}/v1`,
      "    apiKey: mock-key-not-a-secret",
      "    api: openai-completions",
      "    models:",
      "      - id: text",
      "        name: mock text",
      "        contextWindow: 128000",
      "        maxTokens: 4096",
      "      - id: toolturn",
      "        name: mock 5-tool turn",
      "        contextWindow: 128000",
      "        maxTokens: 4096",
      ...["read", "grep", "glob", "find", "edit", "bash"].flatMap((t) => [
        `      - id: bench-${t}`,
        `        name: mock bench ${t}`,
        "        contextWindow: 128000",
        "        maxTokens: 4096",
      ]),
      "",
    ].join("\n"),
  );
  writeFileSync(join(agentDir, "config.yml"), "# bench config: intentionally minimal\n");
  // The ext host resolves legacy-typebox via $HOME/.bun/install/global; link it
  // read-only so extension APIs work under the redirected HOME.
  const bunGlobal = join(root, ".bun", "install", "global");
  mkdirSync(join(root, ".bun", "install"), { recursive: true });
  if (!existsSync(bunGlobal)) {
    symlinkSync(`${process.env.HOME}/.bun/install/global`, bunGlobal);
  }
  const extLink = join(agentDir, "extensions");
  return {
    root,
    env: {
      HOME: root,
      PI_CODING_AGENT_DIR: agentDir,
      OMP_AGENT_DIR: agentDir,
      OMP_RS_EXT_HOST: EXT_HOST,
      NO_COLOR: "1",
      TERM: "dumb",
      OMP_BENCH: "1",
    },
    enableExtensions: (on: boolean) => {
      if (on && !existsSync(extLink)) {
        symlinkSync(join(REAL_AGENT_DIR, "extensions"), extLink);
      } else if (!on && existsSync(extLink)) {
        rmSync(extLink);
      }
    },
  };
}

/** The 5 largest real session files, copied read-only into the temp agent dir. */
function setupSessions(agentDir: string): string[] {
  const sessionsRoot = join(REAL_AGENT_DIR, "sessions");
  const files: { path: string; size: number }[] = [];
  for (const dir of readdirSync(sessionsRoot)) {
    const sub = join(sessionsRoot, dir);
    let st;
    try {
      st = statSync(sub);
    } catch {
      continue;
    }
    if (!st.isDirectory()) continue;
    for (const f of readdirSync(sub)) {
      if (!f.endsWith(".jsonl")) continue;
      const p = join(sub, f);
      files.push({ path: p, size: statSync(p).size });
    }
  }
  files.sort((a, b) => b.size - a.size);
  const top = files.slice(0, 5);
  const destDir = join(agentDir, "sessions", "-bench");
  mkdirSync(destDir, { recursive: true });
  const copied: string[] = [];
  for (const f of top) {
    const dest = join(destDir, f.path.split("/").pop()!);
    copyFileSync(f.path, dest);
    copied.push(dest);
  }
  return copied;
}

// ---------- mock provider lifecycle ----------

let mockProc: Subprocess | null = null;
let mockLog = "";

async function startMock(logSuffix: string): Promise<void> {
  await stopMock();
  mockLog = join(BENCH_DIR, "tmp", `mock-${NOW}-${logSuffix}.jsonl`);
  mockProc = Bun.spawn(["bun", join(BENCH_DIR, "mock-server.ts")], {
    env: {
      ...process.env,
      PORT: String(MOCK_PORT),
      MOCK_LATENCY_MS: "0",
      MOCK_LOG: mockLog,
      MOCK_READ_FILE: join(REPO_ROOT, "crates/omp-cli/src/rpc.rs"),
      MOCK_EDIT_FILE: join(BENCH_DIR, "tmp", "edit-target.txt"),
      MOCK_WORKDIR: WORKDIR,
    } as Record<string, string>,
    stdout: "ignore",
    stderr: "inherit",
  });
  writeFileSync(join(BENCH_DIR, "tmp", "edit-target.txt"), "alpha-bravo-charlie\n");
  // wait until OUR provider answers (spawn + bind can take >1s). A foreign process on the port
  // (e.g. a stale mock from an earlier run) answers with a different mock_log: fail loudly
  // instead of silently benchmarking against it.
  for (let i = 0; i < 200; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${MOCK_PORT}/v1/models`);
      if (res.ok) {
        const body = (await res.json()) as { mock_log?: string };
        if (body.mock_log === mockLog) return;
        throw new Error(`port ${MOCK_PORT} is held by a foreign mock (log ${body.mock_log ?? "?"}); kill it and rerun`);
      }
    } catch (e) {
      if (e instanceof Error && e.message.includes("foreign mock")) throw e;
      /* not up yet */
    }
    if (mockProc?.exitCode !== null && mockProc?.exitCode !== undefined) {
      throw new Error(`mock provider exited with code ${mockProc.exitCode} (port ${MOCK_PORT} busy?)`);
    }
    await Bun.sleep(100);
  }
  throw new Error("mock provider did not start");
}

async function stopMock(): Promise<void> {
  if (mockProc) {
    try {
      mockProc.kill("SIGKILL");
    } catch {
      /* dead */
    }
    mockProc = null;
    await Bun.sleep(250); // let the port be released
  }
}

interface MockEvent {
  t: number;
  kind: string;
  model: string;
}

function readMockLog(): MockEvent[] {
  if (!existsSync(mockLog)) return [];
  return readFileSync(mockLog, "utf8")
    .trim()
    .split("\n")
    .filter((l) => l.length > 0)
    .map((l) => JSON.parse(l) as MockEvent);
}

/** Median gap between resp_end and the next req in the same toolturn session. */
function interRequestGaps(): number[] {
  const evs = readMockLog().filter((e) => e.kind === "req" || e.kind === "resp_end");
  const gaps: number[] = [];
  for (let i = 1; i < evs.length; i++) {
    if (evs[i].kind === "req" && evs[i - 1].kind === "resp_end") gaps.push(evs[i].t - evs[i - 1].t);
  }
  return gaps;
}

// ---------- results ----------

interface ScenarioResult {
  scenario: string;
  bins: Record<
    string,
    {
      ok: boolean;
      error?: string;
      runs?: Stats;
      extra?: Record<string, unknown>;
    }
  >;
}

const results: ScenarioResult[] = [];
const mdLines: string[] = [];

function record(scenario: string, bin: string, r: ScenarioResult["bins"][string]): void {
  let entry = results.find((s) => s.scenario === scenario);
  if (!entry) {
    entry = { scenario, bins: {} };
    results.push(entry);
  }
  entry.bins[bin] = r;
}

function fmtMs(s: Stats | undefined): string {
  if (!s) return "-";
  return `${s.median.toFixed(0)}ms (p95 ${s.p95.toFixed(0)})`;
}

// ---------- scenario helpers ----------

function binCmd(bin: BinSpec, model: string, extra: string[] = []): string[] {
  return [bin.path, "run the benchmark", "-p", "--model", `mock/${model}`, "--max-time", "120", ...extra];
}

async function repeatRuns(
  bin: BinSpec,
  env: BenchEnv,
  makeCmd: () => string[],
  n: number,
  opts: { cwd?: string; rss?: boolean } = {},
): Promise<{ runs: Stats; rssPeakKb?: Stats; samples: number[]; rssSamples: number[] }> {
  const samples: number[] = [];
  const rssSamples: number[] = [];
  for (let i = 0; i < n; i++) {
    const r = await runTimed(makeCmd(), { cwd: opts.cwd ?? WORKDIR, env, rssSampling: opts.rss });
    if (r.code !== 0) throw new Error(`exit ${r.code}: ${r.stderr.slice(-400)}`);
    samples.push(r.ms);
    if (opts.rss) rssSamples.push(r.peakRssKb);
  }
  return { runs: stats(samples), rssPeakKb: rssSamples.length ? stats(rssSamples) : undefined, samples, rssSamples };
}

function checkBin(bin: BinSpec): string | null {
  if (bin.name === "omp-rs" && !existsSync(bin.path)) {
    return `binary missing: ${bin.path} (build first: cargo build --release -p omp-cli)`;
  }
  return null;
}

// ---------- scenarios ----------

async function scenario1(): Promise<void> {
  await startMock("s1");
  const title = "1. cold start to ready (-p and rpc), with/without extensions";
  console.log(`\n== ${title}`);
  mdLines.push(`\n## ${title}\n`);
  mdLines.push("| bin | mode | extensions | median ms | p95 ms |");
  mdLines.push("|---|---|---|---|---|");
  for (const bin of BINS) {
    const err = checkBin(bin);
    for (const mode of ["print", "rpc"] as const) {
      for (const withExt of [false, true]) {
        const tag = `s1-${bin.name}-${mode}-${withExt ? "ext" : "noext"}`;
        const home = makeAgentHome(tag);
        home.enableExtensions(withExt);
        if (err) {
          record(`s1-${mode}-${withExt}`, bin.name, { ok: false, error: err });
          mdLines.push(`| ${bin.name} | ${mode} | ${withExt} | ERROR | |`);
          continue;
        }
        try {
          let samples: number[] = [];
          if (mode === "print") {
            ({ samples } = await repeatRuns(bin, home.env, () => binCmd(bin, "text"), REPEATS, {
              cwd: home.root,
            }));
          } else {
            for (let i = 0; i < REPEATS; i++) {
              const t0 = performance.now();
              const client = await RpcClient.start(
                [bin.path, "--mode", "rpc", "--model", "mock/text"],
                { cwd: home.root, env: home.env, timeoutMs: 120_000 },
              );
              await client.call("get_state");
              samples.push(performance.now() - t0);
              await client.stop();
            }
          }
          const s = stats(samples);
          record(`s1-${mode}-${withExt}`, bin.name, { ok: true, runs: s });
          mdLines.push(`| ${bin.name} | ${mode} | ${withExt ? "yes" : "no"} | ${s.median.toFixed(0)} | ${s.p95.toFixed(0)} |`);
        } catch (e) {
          record(`s1-${mode}-${withExt}`, bin.name, { ok: false, error: String(e) });
          mdLines.push(`| ${bin.name} | ${mode} | ${withExt} | ERROR | |`);
        }
        rmSync(home.root, { recursive: true, force: true });
      }
    }
  }
}

async function scenario2(): Promise<void> {
  await startMock("s2");
  const title = "2. RSS peak/steady: 1 session, N=1/4/16 parallel rpc sessions";
  console.log(`\n== ${title}`);
  mdLines.push(`\n## ${title}\n`);
  mdLines.push("| bin | sessions | prompt | RSS peak KB (median) | steady KB after settle |");
  mdLines.push("|---|---|---|---|---|");
  for (const bin of BINS) {
    const err = checkBin(bin);
    for (const n of [1, 4, 16]) {
      const home = makeAgentHome(`s2-${bin.name}-${n}`);
      if (err) {
        record(`s2-n${n}`, bin.name, { ok: false, error: err });
        mdLines.push(`| ${bin.name} | ${n} | toolturn | ERROR | |`);
        continue;
      }
      try {
        const peaks: number[] = [];
        const steadys: number[] = [];
        const reps = QUICK ? 1 : 3;
        for (let rep = 0; rep < reps; rep++) {
          const clients: RpcClient[] = await Promise.all(
            Array.from({ length: n }, () =>
              RpcClient.start([bin.path, "--mode", "rpc", "--model", "mock/toolturn"], {
                cwd: home.root,
                env: home.env,
                timeoutMs: 300_000,
              }),
            ),
          );
          const idle = await Promise.all(clients.map((c) => c.rssNow()));
          for (const c of clients) await c.prompt(`turn ${n}`);
          // sample RSS across the whole turn fleet
          const pids = clients.map((c) => c.pid);
          let peak = 0;
          for (let k = 0; k < 40; k++) {
            const now = await Promise.all(pids.map((p) => treeRssKb(p)));
            peak = Math.max(peak, ...now);
            await Bun.sleep(25);
          }
          await Bun.sleep(1500); // settle
          const settled = (await Promise.all(pids.map((p) => treeRssKb(p)))).reduce((a, b) => a + b, 0);
          const idleSum = idle.reduce((a, b) => a + b, 0);
          peaks.push(Math.max(peak, idleSum ? peak : peak));
          steadys.push(settled);
          for (const c of clients) await c.stop();
        }
        const pk = stats(peaks);
        const st = stats(steadys);
        record(`s2-n${n}`, bin.name, {
          ok: true,
          runs: pk,
          extra: { steadyKb: st, sessions: n },
        });
        mdLines.push(`| ${bin.name} | ${n} | toolturn | ${pk.median.toFixed(0)} | ${st.median.toFixed(0)} |`);
      } catch (e) {
        record(`s2-n${n}`, bin.name, { ok: false, error: String(e) });
        mdLines.push(`| ${bin.name} | ${n} | toolturn | ERROR | |`);
      }
      rmSync(home.root, { recursive: true, force: true });
    }
  }
}

async function scenario3(): Promise<void> {
  const title = "3. per-turn overhead: scripted 5-tool-call turn (mock, latency 0)";
  console.log(`\n== ${title}`);
  mdLines.push(`\n## ${title}\n`);
  mdLines.push("| bin | turn wall median ms | turn wall p95 ms | last-token->next-req median ms |");
  mdLines.push("|---|---|---|---|");
  for (const bin of BINS) {
    const err = checkBin(bin);
    const home = makeAgentHome(`s3-${bin.name}`);
    if (err) {
      record("s3", bin.name, { ok: false, error: err });
      mdLines.push(`| ${bin.name} | ERROR | | |`);
      continue;
    }
    try {
      const turnSamples: number[] = [];
      const gapSamples: number[] = [];
      for (let i = 0; i < REPEATS; i++) {
        await startMock(`s3-${i}`);
        const client = await RpcClient.start([bin.path, "--mode", "rpc", "--model", "mock/toolturn"], {
          cwd: home.root,
          env: home.env,
        });
        const { ms } = await client.prompt("run the turn");
        turnSamples.push(ms);
        await client.stop();
        await stopMock();
        gapSamples.push(...interRequestGaps());
      }
      const t = stats(turnSamples);
      const g = stats(gapSamples);
      record("s3", bin.name, {
        ok: true,
        runs: t,
        extra: { dispatchGapMs: g, note: "gap = mock resp_end -> next LLM req (dispatch + tool exec + hook overhead)" },
      });
      mdLines.push(`| ${bin.name} | ${t.median.toFixed(0)} | ${t.p95.toFixed(0)} | ${g.n ? g.median.toFixed(1) : "-"} |`);
    } catch (e) {
      record("s3", bin.name, { ok: false, error: String(e) });
      mdLines.push(`| ${bin.name} | ERROR | | |`);
    }
    rmSync(home.root, { recursive: true, force: true });
  }
}

async function scenario4(): Promise<void> {
  const title = "4. tool microbenchmarks (read/grep/glob/find/edit/bash) on a large repo";
  console.log(`\n== ${title}`);
  mdLines.push(`\n## ${title}\n`);
  mdLines.push("| bin | tool | median turn ms | p95 ms | tool-phase gap median ms |");
  mdLines.push("|---|---|---|---|---|");
  const tools = ["read", "grep", "glob", "find", "edit", "bash"] as const;
  for (const bin of BINS) {
    const err = checkBin(bin);
    const home = makeAgentHome(`s4-${bin.name}`);
    home.enableExtensions(false);
    for (const tool of tools) {
      if (err) {
        record(`s4-${tool}`, bin.name, { ok: false, error: err });
        mdLines.push(`| ${bin.name} | ${tool} | ERROR | | |`);
        continue;
      }
      try {
        const samples: number[] = [];
        const gaps: number[] = [];
        const reps = QUICK ? 2 : REPEATS;
        for (let i = 0; i < reps; i++) {
          await startMock(`s4-${tool}-${i}`);
          const client = await RpcClient.start([bin.path, "--mode", "rpc", "--model", `mock/bench-${tool}`], {
            cwd: home.root,
            env: home.env,
          });
          const { ms } = await client.prompt(`bench ${tool}`);
          samples.push(ms);
          await client.stop();
          await stopMock();
          gaps.push(...interRequestGaps());
        }
        const s = stats(samples);
        const g = stats(gaps);
        record(`s4-${tool}`, bin.name, { ok: true, runs: s, extra: { toolPhaseGapMs: g } });
        mdLines.push(`| ${bin.name} | ${tool} | ${s.median.toFixed(0)} | ${s.p95.toFixed(0)} | ${g.n ? g.median.toFixed(1) : "-"} |`);
      } catch (e) {
        record(`s4-${tool}`, bin.name, { ok: false, error: String(e) });
        mdLines.push(`| ${bin.name} | ${tool} | ERROR | | |`);
      }
    }
    rmSync(home.root, { recursive: true, force: true });
  }
}

async function scenario5(): Promise<void> {
  await startMock("s5");
  const title = "5. session load/resume of the 5 largest real session files (read-only copies)";
  console.log(`\n== ${title}`);
  mdLines.push(`\n## ${title}\n`);
  mdLines.push("| bin | session file size MB | resume wall median ms | baseline (empty session) ms |");
  mdLines.push("|---|---|---|---|");
  for (const bin of BINS) {
    const err = checkBin(bin);
    const home = makeAgentHome(`s5-${bin.name}`);
    let files: string[] = [];
    try {
      files = setupSessions(join(home.root, ".omp", "agent"));
    } catch (e) {
      record("s5", bin.name, { ok: false, error: `session setup: ${e}` });
      mdLines.push(`| ${bin.name} | - | ERROR | |`);
      continue;
    }
    if (err) {
      record("s5", bin.name, { ok: false, error: err });
      mdLines.push(`| ${bin.name} | - | ERROR | |`);
      rmSync(home.root, { recursive: true, force: true });
      continue;
    }
    try {
      // baseline: same -p with a fresh (empty) session
      const base = await repeatRuns(bin, home.env, () => binCmd(bin, "text"), QUICK ? 2 : REPEATS, { cwd: home.root });
      const sizes: number[] = [];
      const walls: number[] = [];
      const fileErrors: Record<string, string> = {};
      for (const f of files) {
        const sizeMb = statSync(f).size / 1e6;
        try {
          const r = await repeatRuns(
            bin,
            home.env,
            () => binCmd(bin, "text", ["--session-dir", join(home.root, ".omp", "agent", "sessions", "-bench"), "--resume", f]),
            QUICK ? 1 : 3,
            { cwd: home.root },
          );
          sizes.push(sizeMb);
          walls.push(r.runs.median);
          mdLines.push(`| ${bin.name} | ${sizeMb.toFixed(1)} | ${r.runs.median.toFixed(0)} | ${base.runs.median.toFixed(0)} |`);
        } catch (e) {
          fileErrors[`${sizeMb.toFixed(1)}MB`] = String(e).slice(0, 200);
          mdLines.push(`| ${bin.name} | ${sizeMb.toFixed(1)} | ERROR | ${base.runs.median.toFixed(0)} |`);
        }
      }
      record("s5", bin.name, {
        ok: walls.length > 0,
        runs: stats(walls),
        error: walls.length > 0 ? undefined : "all session files failed to resume",
        extra: { baselineMs: base.runs, sizesMb: sizes, fileErrors },
      });
    } catch (e) {
      record("s5", bin.name, { ok: false, error: String(e) });
      mdLines.push(`| ${bin.name} | - | ERROR | |`);
    }
    rmSync(home.root, { recursive: true, force: true });
  }
}

async function scenario6(): Promise<void> {
  await startMock("s6");
  const title = "6. ext-host emit cost per event (all extensions loaded vs --no-extensions)";
  // Cold columns (unchanged method, kept for history): every sample is the
  // FIRST turn of a fresh process, so the delta is dominated by extension
  // loading and process-start noise. The warm table below is the per-event
  // measurement (s6w-* records).
  console.log(`\n== ${title}`);
  mdLines.push(`\n## ${title}\n`);
  mdLines.push("| bin | extensions | turn wall median ms | delta vs no-ext ms | per-event delta ms (est.) |");
  mdLines.push("|---|---|---|---|---|");
  for (const bin of BINS) {
    const err = checkBin(bin);
    const deltas: { withExt: number; noExt: number } = { withExt: NaN, noExt: NaN };
    for (const withExt of [false, true]) {
      const home = makeAgentHome(`s6-${bin.name}-${withExt}`);
      home.enableExtensions(withExt);
      if (err) {
        record(`s6-${withExt}`, bin.name, { ok: false, error: err });
        mdLines.push(`| ${bin.name} | ${withExt} | ERROR | | |`);
        continue;
      }
      try {
        const samples: number[] = [];
        for (let i = 0; i < REPEATS; i++) {
          const client = await RpcClient.start(
            [bin.path, "--mode", "rpc", "--model", "mock/toolturn", ...(withExt ? [] : ["--no-extensions"])] as string[],
            { cwd: home.root, env: home.env, timeoutMs: 180_000 },
          );
          const { ms } = await client.prompt("run the turn");
          samples.push(ms);
          await client.stop();
        }
        const s = stats(samples);
        if (withExt) deltas.withExt = s.median;
        else deltas.noExt = s.median;
        record(`s6-${withExt}`, bin.name, { ok: true, runs: s });
        mdLines.push(`| ${bin.name} | ${withExt ? "yes" : "no"} | ${s.median.toFixed(0)} | | |`);
      } catch (e) {
        record(`s6-${withExt}`, bin.name, { ok: false, error: String(e) });
        mdLines.push(`| ${bin.name} | ${withExt} | ERROR | | |`);
      }
      rmSync(home.root, { recursive: true, force: true });
    }
    if (Number.isFinite(deltas.withExt) && Number.isFinite(deltas.noExt)) {
      const delta = deltas.withExt - deltas.noExt;
      // toolturn turn emits >= 6 agent events (agent_start, tool_call x5, tool_result x5, message_end...)
      const perEvent = delta / 16;
      record("s6-delta", bin.name, {
        ok: true,
        runs: { n: 1, median: delta, p95: delta, min: delta, max: delta },
        extra: { perEventMs: perEvent, assumedEventsPerTurn: 16 },
      });
      mdLines.push(`| ${bin.name} | delta | ${deltas.withExt.toFixed(0)} | ${delta.toFixed(0)} | ${perEvent.toFixed(2)} |`);
    }
  }
  await scenario6Warm();
}

/** Minimal stdio RPC client that also counts the event frames of each request. */
class CountingRpc {
  #proc: Subprocess;
  #reader: ReadableStreamDefaultReader<Uint8Array>;
  #decoder = new TextDecoder();
  #buf = "";
  #nextId = 1;

  private constructor(proc: Subprocess) {
    this.#proc = proc;
    this.#reader = (proc.stdout as ReadableStream<Uint8Array>).getReader();
  }

  static async start(cmd: string[], cwd: string, env: BenchEnv, timeoutMs: number): Promise<CountingRpc> {
    const proc = spawn(cmd, {
      cwd,
      env: { ...process.env, ...env } as Record<string, string>,
      stdout: "pipe",
      // Never "pipe" unread: a full stderr pipe would stall the process mid-turn.
      stderr: process.env.BENCH_STDERR_INHERIT ? "inherit" : "ignore",
      stdin: "pipe",
    });
    setTimeout(() => {
      try {
        proc.kill("SIGKILL");
      } catch {
        /* dead */
      }
    }, timeoutMs).unref();
    const c = new CountingRpc(proc);
    for (;;) if ((await c.#frame())["type"] === "ready") return c;
  }

  async #frame(): Promise<Record<string, unknown>> {
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

  /**
   * Send `type` and wait for its final frame (`prompt_result` for prompts,
   * the response otherwise), like lib.ts RpcClient. Returns the wall time from
   * write to final frame and the event frames seen meanwhile, by type.
   */
  async request(
    type: string,
    params: Record<string, unknown>,
    timeoutMs: number,
  ): Promise<{ ms: number; frame: Record<string, unknown>; events: Record<string, number> }> {
    const id = this.#nextId++;
    const stdin = this.#proc.stdin as unknown as { write: (s: string) => void; flush?: () => void };
    const events: Record<string, number> = {};
    const t0 = performance.now();
    stdin.write(`${JSON.stringify({ id, method: type, type, ...params })}\n`);
    stdin.flush?.();
    const wait = (async () => {
      for (;;) {
        const f = await this.#frame();
        if (f["id"] === id) {
          const final =
            type === "prompt"
              ? f["type"] === "prompt_result" || (f["type"] === "response" && f["status"] !== undefined)
              : f["type"] === "response" || f["method"] === undefined;
          if (final) return { ms: performance.now() - t0, frame: f, events };
          continue;
        }
        const t = String(f["type"] ?? f["method"] ?? "?");
        events[t] = (events[t] ?? 0) + 1;
      }
    })();
    const timeout = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error(`${type} timed out after ${timeoutMs}ms`)), timeoutMs).unref();
    });
    return Promise.race([wait, timeout]);
  }

  /** Fresh empty session in the same warm process (keeps every turn's context identical). */
  async reset(): Promise<boolean> {
    const r = await this.request("new_session", {}, 30_000);
    return r.frame["success"] !== false && r.frame["error"] === undefined;
  }

  async stop(): Promise<void> {
    try {
      this.#proc.kill("SIGKILL");
    } catch {
      /* dead */
    }
    await this.#proc.exited.catch(() => -1);
  }
}

function quantile(values: number[], q: number): number {
  const s = [...values].sort((a, b) => a - b);
  if (s.length === 0) return NaN;
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/** Lines in `path` (0 when absent). */
function lineCount(path: string): number {
  if (!existsSync(path)) return 0;
  const t = readFileSync(path, "utf8");
  return t.length === 0 ? 0 : t.split("\n").filter((l) => l.length > 0).length;
}

/**
 * s6 warm: per-event extension cost from warm processes. One process with all
 * extensions and one with --no-extensions stay alive side by side; after
 * warm-up turns both run the same toolturn N times, interleaved (turn i on
 * both before turn i+1, alternating which goes first), each turn in a fresh
 * empty session (`new_session`, outside the timed window) so context size is
 * identical every turn. Reported: median per-turn wall of each, the median
 * (and IQR) of the PAIRED per-turn deltas, the event frames the core actually
 * emitted per turn (RPC stream), and — for omp-rs — the emits that actually
 * crossed to the ext host per turn (OMP_EXT_TRACE, counted in a separate
 * traced process so the trace writes stay out of the timed runs).
 */
async function scenario6Warm(): Promise<void> {
  const WARM = QUICK ? 2 : 5;
  const TURNS = QUICK ? 8 : Math.max(30, REPEATS * 3);
  mdLines.push("");
  mdLines.push(
    `warm, paired (s6w): ${WARM} warm-up + ${TURNS} measured turns per process, interleaved, fresh session per turn; delta = median of per-turn (ext − no-ext)`,
  );
  mdLines.push("");
  mdLines.push(
    "| bin | no-ext turn median ms | ext turn median ms | paired delta median ms (IQR) | rpc events/turn (message_update) | ext-host emits/turn | per-event delta ms |",
  );
  mdLines.push("|---|---|---|---|---|---|---|");
  for (const bin of BINS) {
    const err = checkBin(bin);
    if (err) {
      record("s6w", bin.name, { ok: false, error: err });
      mdLines.push(`| ${bin.name} | ERROR | | | | | |`);
      continue;
    }
    const home = makeAgentHome(`s6w-${bin.name}`);
    home.enableExtensions(true);
    const cmd = (withExt: boolean): string[] => [
      bin.path,
      "--mode",
      "rpc",
      "--model",
      "mock/toolturn",
      ...(withExt ? [] : ["--no-extensions"]),
    ];
    const clients: CountingRpc[] = [];
    try {
      const [noExt, ext] = await Promise.all([
        CountingRpc.start(cmd(false), home.root, home.env, 900_000),
        CountingRpc.start(cmd(true), home.root, home.env, 900_000),
      ]);
      clients.push(noExt, ext);
      let resetOk = true;
      const turn = async (c: CountingRpc): Promise<{ ms: number; events: Record<string, number> }> => {
        resetOk = (await c.reset()) && resetOk;
        const r = await c.request("prompt", { message: "run the turn" }, 60_000);
        const status = String(r.frame["status"] ?? "completed");
        if (status !== "completed" && status !== "ok") throw new Error(`turn status ${status}`);
        return { ms: r.ms, events: r.events };
      };
      for (let i = 0; i < WARM; i++) {
        await turn(noExt);
        await turn(ext);
      }
      const walls = { noExt: [] as number[], ext: [] as number[] };
      const paired: number[] = [];
      let eventsExt: Record<string, number> = {};
      for (let i = 0; i < TURNS; i++) {
        const order: ["noExt" | "ext", CountingRpc][] =
          i % 2 === 0
            ? [
                ["noExt", noExt],
                ["ext", ext],
              ]
            : [
                ["ext", ext],
                ["noExt", noExt],
              ];
        const got: Record<string, number> = {};
        for (const [k, c] of order) {
          const r = await turn(c);
          got[k] = r.ms;
          walls[k].push(r.ms);
          if (k === "ext") eventsExt = r.events;
        }
        paired.push(got.ext - got.noExt);
      }
      await Promise.all(clients.map((c) => c.stop()));
      clients.length = 0;

      // ext-host emits per turn: a separate traced process (omp-rs; the TS
      // binary ignores OMP_EXT_TRACE, reported as "-").
      let extEmitsPerTurn: number | null = null;
      if (bin.name === "omp-rs") {
        const trace = join(home.root, "ext-trace.jsonl");
        const traced = await CountingRpc.start(cmd(true), home.root, { ...home.env, OMP_EXT_TRACE: trace }, 300_000);
        try {
          await turn(traced);
          const counts: number[] = [];
          for (let i = 0; i < 3; i++) {
            await traced.reset();
            await Bun.sleep(300); // session-switch emits land before the baseline
            const before = lineCount(trace);
            await traced.request("prompt", { message: "run the turn" }, 60_000);
            await Bun.sleep(300); // trailing fire-and-forget emits
            counts.push(lineCount(trace) - before);
          }
          extEmitsPerTurn = quantile(counts, 0.5);
        } finally {
          await traced.stop();
        }
      }

      // Agent lifecycle frames only (UI/extension side-channel frames are not agent events).
      const rpcEvents = Object.entries(eventsExt)
        .filter(([t]) => /^(agent|turn|message|tool_execution)_/.test(t))
        .reduce((a, [, n]) => a + n, 0);
      const msgUpdates = eventsExt["message_update"] ?? 0;
      const deltaMed = quantile(paired, 0.5);
      const denom = extEmitsPerTurn ?? rpcEvents;
      const perEvent = denom > 0 ? deltaMed / denom : NaN;
      const sNo = stats(walls.noExt);
      const sExt = stats(walls.ext);
      record("s6w-false", bin.name, { ok: true, runs: sNo });
      record("s6w-true", bin.name, { ok: true, runs: sExt });
      record("s6w-delta", bin.name, {
        ok: true,
        runs: stats(paired),
        extra: {
          method: "warm-paired",
          warmTurns: WARM,
          turns: TURNS,
          resetOk,
          deltaP25Ms: quantile(paired, 0.25),
          deltaMedianMs: deltaMed,
          deltaP75Ms: quantile(paired, 0.75),
          rpcEventsPerTurn: rpcEvents,
          rpcEventsByType: eventsExt,
          extEmitsPerTurn,
          perEventMs: perEvent,
          perEventDenominator: extEmitsPerTurn !== null ? "extEmitsPerTurn" : "rpcEventsPerTurn",
        },
      });
      mdLines.push(
        `| ${bin.name} | ${sNo.median.toFixed(1)} | ${sExt.median.toFixed(1)} | ${deltaMed.toFixed(1)} (${quantile(paired, 0.25).toFixed(1)}–${quantile(paired, 0.75).toFixed(1)}) | ${rpcEvents} (${msgUpdates}) | ${extEmitsPerTurn ?? "-"} | ${Number.isFinite(perEvent) ? perEvent.toFixed(3) : "-"} |`,
      );
      if (!resetOk) mdLines.push(`| ${bin.name} | note: new_session failed; turns did not start from an empty session | | | | | |`);
    } catch (e) {
      record("s6w-delta", bin.name, { ok: false, error: String(e) });
      mdLines.push(`| ${bin.name} | ERROR | | | | | |`);
    } finally {
      await Promise.all(clients.map((c) => c.stop()));
    }
    rmSync(home.root, { recursive: true, force: true });
  }
}

async function scenario7(): Promise<void> {
  await startMock("s7");
  const title = "7. throughput: N concurrent rpc sessions running the toolturn";
  console.log(`\n== ${title}`);
  mdLines.push(`\n## ${title}\n`);
  mdLines.push("| bin | N | wall median ms | turns/sec |");
  mdLines.push("|---|---|---|---|");
  for (const bin of BINS) {
    const err = checkBin(bin);
    for (const n of [4, 16]) {
      const home = makeAgentHome(`s7-${bin.name}-${n}`);
      if (err) {
        record(`s7-n${n}`, bin.name, { ok: false, error: err });
        mdLines.push(`| ${bin.name} | ${n} | ERROR | |`);
        continue;
      }
      try {
        const walls: number[] = [];
        const reps = QUICK ? 1 : 3;
        for (let rep = 0; rep < reps; rep++) {
          const clients: RpcClient[] = await Promise.all(
            Array.from({ length: n }, () =>
              RpcClient.start([bin.path, "--mode", "rpc", "--model", "mock/toolturn"], {
                cwd: home.root,
                env: home.env,
                timeoutMs: 300_000,
              }),
            ),
          );
          const t0 = performance.now();
          await Promise.all(clients.map((c) => c.prompt(`throughput ${n}`)));
          walls.push(performance.now() - t0);
          for (const c of clients) await c.stop();
        }
        const s = stats(walls);
        record(`s7-n${n}`, bin.name, { ok: true, runs: s, extra: { turnsPerSec: n / (s.median / 1000) } });
        mdLines.push(`| ${bin.name} | ${n} | ${s.median.toFixed(0)} | ${(n / (s.median / 1000)).toFixed(2)} |`);
      } catch (e) {
        record(`s7-n${n}`, bin.name, { ok: false, error: String(e) });
        mdLines.push(`| ${bin.name} | ${n} | ERROR | |`);
      }
      rmSync(home.root, { recursive: true, force: true });
    }
  }
}


// ---------- memory scenarios (m1-m3) ----------

/** Aggregate a role-grouped breakdown: sum rss/pss per role. */
function roleSums(bd: { procs: { role: string; rssKb: number; pssKb: number }[] }): Record<string, { rssKb: number; pssKb: number }> {
  const out: Record<string, { rssKb: number; pssKb: number }> = {};
  for (const pr of bd.procs) {
    const e = (out[pr.role] ??= { rssKb: 0, pssKb: 0 });
    e.rssKb += pr.rssKb;
    e.pssKb += pr.pssKb;
  }
  return out;
}

interface TreeSample {
  rssKb: number;
  pssKb: number;
}

/** Total tree RSS+PSS for a set of root pids (PSS summed over all trees so
 * shared pages across sessions are counted fairly). */
async function fleetSample(pids: number[]): Promise<TreeSample> {
  let rss = 0;
  let pss = 0;
  for (const pid of pids) {
    rss += await treeRssKb(pid);
    pss += await treePssKb(pid);
  }
  return { rssKb: rss, pssKb: pss };
}

async function scenarioM1(): Promise<void> {
  await startMock("m1");
  const title = "m1. one rpc session: per-process RSS/PSS breakdown, growth over 50 turns";
  console.log(`\n== ${title}`);
  mdLines.push(`\n## ${title}\n`);
  mdLines.push("| bin | point | core KB | ext-host KB | other KB | tree RSS KB | tree PSS KB |");
  mdLines.push("|---|---|---|---|---|---|---|");
  const TURN_MARKS = [0, 1, 2, 5, 10, 20, 30, 40, 50];
  for (const bin of BINS) {
    const err = checkBin(bin);
    const home = makeAgentHome(`m1-${bin.name}`);
    home.enableExtensions(true);
    if (err) {
      record("m1", bin.name, { ok: false, error: err });
      mdLines.push(`| ${bin.name} | - | ERROR | | | | | |`);
      rmSync(home.root, { recursive: true, force: true });
      continue;
    }
    try {
      const client = await RpcClient.start([bin.path, "--mode", "rpc", "--model", "mock/toolturn"], {
        cwd: home.root,
        env: home.env,
        timeoutMs: 600_000,
      });
      const points: { turn: number; roles: Record<string, { rssKb: number; pssKb: number }>; rssKb: number; pssKb: number }[] = [];
      const pushPoint = async (turn: number): Promise<void> => {
        await Bun.sleep(200); // let the ext-host/event backlog settle
        const bd = await treeBreakdown(client.pid);
        const roles = roleSums(bd);
        points.push({ turn, roles, rssKb: bd.totalRssKb, pssKb: bd.totalPssKb });
        const fmt = (r: string): string => (roles[r] ? roles[r].rssKb.toFixed(0) : "0");
        mdLines.push(`| ${bin.name} | ${turn === 0 ? "idle" : `turn ${turn}`} | ${fmt("core")} | ${fmt("ext-host")} | ${fmt("other")} | ${bd.totalRssKb} | ${bd.totalPssKb} |`);
      };
      await client.call("get_state");
      await pushPoint(0);
      let failedTurn: string | null = null;
      for (let t = 1; t <= 50 && !failedTurn; t++) {
        try {
          await client.prompt(`turn ${t}`, 30_000);
        } catch (e) {
          failedTurn = `turn ${t}: ${String(e).slice(0, 160)}`;
          break;
        }
        if (TURN_MARKS.includes(t)) await pushPoint(t);
      }
      await client.stop();
      record("m1", bin.name, { ok: !failedTurn, error: failedTurn ?? undefined, extra: { points } });
    } catch (e) {
      record("m1", bin.name, { ok: false, error: String(e) });
      mdLines.push(`| ${bin.name} | - | ERROR | | | | | |`);
    }
    rmSync(home.root, { recursive: true, force: true });
  }
}

async function scenarioM2(): Promise<void> {
  await startMock("m2");
  const title = "m2. fleet RSS/PSS for N=1/8/32 concurrent rpc sessions, with/without extensions";
  console.log(`\n== ${title}`);
  mdLines.push(`\n## ${title}\n`);
  mdLines.push("| bin | sessions | extensions | tree RSS KB (idle) | tree PSS KB (idle) | KB/session PSS |");
  mdLines.push("|---|---|---|---|---|---|");
  for (const bin of BINS) {
    const err = checkBin(bin);
    for (const withExt of [true, false]) {
      for (const n of [1, 8, 32]) {
        const home = makeAgentHome(`m2-${bin.name}-${withExt ? "ext" : "noext"}-${n}`);
        home.enableExtensions(withExt);
        if (err) {
          record(`m2-${withExt ? "ext" : "noext"}-n${n}`, bin.name, { ok: false, error: err });
          mdLines.push(`| ${bin.name} | ${n} | ${withExt} | ERROR | | |`);
          rmSync(home.root, { recursive: true, force: true });
          continue;
        }
        try {
          const extraArgs = withExt ? [] : ["--no-extensions"];
          const clients: RpcClient[] = await Promise.all(
            Array.from({ length: n }, () =>
              RpcClient.start([bin.path, "--mode", "rpc", "--model", "mock/toolturn", ...extraArgs], {
                cwd: home.root,
                env: home.env,
                timeoutMs: 300_000,
              }),
            ),
          );
          // ext-host/extension spawn is async and racy; sample a few settle
          // points and keep the max so fleets are not undercounted.
          const t0 = performance.now();
          let sample: TreeSample = { rssKb: 0, pssKb: 0 };
          for (const waitMs of [800, 1200, 2000, 3200]) {
            await Bun.sleep(waitMs);
            const cur = await fleetSample(clients.map((c) => c.pid));
            sample = { rssKb: Math.max(sample.rssKb, cur.rssKb), pssKb: Math.max(sample.pssKb, cur.pssKb) };
          }
          const wallS = (performance.now() - t0) / 1000;
          await Promise.all(clients.map((c) => c.stop()));
          record(`m2-${withExt ? "ext" : "noext"}-n${n}`, bin.name, {
            ok: true,
            extra: { sessions: n, extensions: withExt, rssKb: sample.rssKb, pssKb: sample.pssKb, pssPerSessionKb: sample.pssKb / n, settleWallS: wallS },
          });
          mdLines.push(`| ${bin.name} | ${n} | ${withExt ? "yes" : "no"} | ${sample.rssKb} | ${sample.pssKb} | ${(sample.pssKb / n).toFixed(0)} |`);
        } catch (e) {
          record(`m2-${withExt ? "ext" : "noext"}-n${n}`, bin.name, { ok: false, error: String(e) });
          mdLines.push(`| ${bin.name} | ${n} | ${withExt} | ERROR | | |`);
        }
        rmSync(home.root, { recursive: true, force: true });
      }
    }
  }
}

async function scenarioM3(): Promise<void> {
  await startMock("m3");
  const title = "m3. RSS + wall time resuming the 5 largest real session files (read-only copies)";
  console.log(`\n== ${title}`);
  mdLines.push(`\n## ${title}\n`);
  mdLines.push("| bin | session file size MB | wall ms | tree RSS peak KB | tree PSS KB |");
  mdLines.push("|---|---|---|---|---|");
  for (const bin of BINS) {
    const err = checkBin(bin);
    const home = makeAgentHome(`m3-${bin.name}`);
    let files: string[] = [];
    try {
      files = setupSessions(join(home.root, ".omp", "agent"));
    } catch (e) {
      record("m3", bin.name, { ok: false, error: `session setup: ${e}` });
      mdLines.push(`| ${bin.name} | - | ERROR | | |`);
      continue;
    }
    if (err) {
      record("m3", bin.name, { ok: false, error: err });
      mdLines.push(`| ${bin.name} | - | ERROR | | |`);
      rmSync(home.root, { recursive: true, force: true });
      continue;
    }
    try {
      const perFile: { sizeMb: number; wallMs: number; rssPeakKb: number; pssKb: number }[] = [];
      for (const f of files) {
        const sizeMb = statSync(f).size / 1e6;
        const r = await runTimed(
          binCmd(bin, "text", ["--session-dir", join(home.root, ".omp", "agent", "sessions", "-bench"), "--resume", f]),
          { cwd: home.root, env: home.env, timeoutMs: 300_000, rssSampling: true },
        );
        if (r.code !== 0) throw new Error(`exit ${r.code}: ${r.stderr.slice(-300)}`);
        // PSS after exit is gone; resample peak-period PSS via a second run would
        // double cost, so report peak RSS + a sampled RSS/PSS pair is skipped.
        perFile.push({ sizeMb, wallMs: r.ms, rssPeakKb: r.peakRssKb, pssKb: 0 });
        mdLines.push(`| ${bin.name} | ${sizeMb.toFixed(1)} | ${r.ms.toFixed(0)} | ${r.peakRssKb} | - |`);
      }
      record("m3", bin.name, { ok: true, extra: { files: perFile } });
    } catch (e) {
      record("m3", bin.name, { ok: false, error: String(e) });
      mdLines.push(`| ${bin.name} | - | ERROR | | |`);
    }
    rmSync(home.root, { recursive: true, force: true });
  }
}

// ---------- main ----------

const SCENARIOS: [string, () => Promise<void>][] = [
  ["1", scenario1],
  ["2", scenario2],
  ["3", scenario3],
  ["4", scenario4],
  ["5", scenario5],
  ["6", scenario6],
  ["7", scenario7],
  ["m1", scenarioM1],
  ["m2", scenarioM2],
  ["m3", scenarioM3],
];

async function main(): Promise<void> {
  mkdirSync(RESULTS_DIR, { recursive: true });
  mkdirSync(join(BENCH_DIR, "tmp"), { recursive: true });
  await startMock("idle");
  await stopMock();

  const jsonPath = join(RESULTS_DIR, `${NOW}.json`);
  const mdPath = join(RESULTS_DIR, `${NOW}.md`);
  const writeResults = (): void => {
    const out = {
      ts: NOW,
      quick: QUICK,
      repeats: REPEATS,
      workdir: WORKDIR,
      bins: BINS.map((b) => b.name),
      mock: { port: MOCK_PORT, latencyMs: 0 },
      results,
    };
    writeFileSync(jsonPath, JSON.stringify(out, null, 2));
    const md = [
      `# omp-rs vs omp (TS) benchmark — ${NOW}`,
      "",
      `mode: ${QUICK ? "quick (smoke)" : "full"}; repeats: ${REPEATS}; workdir: ${WORKDIR}; mock provider latency 0ms`,
      "",
      ...mdLines,
      "",
      `raw data: ${jsonPath}`,
      "",
    ].join("\n");
    writeFileSync(mdPath, md);
  };

  for (const [id, fn] of SCENARIOS) {
    if (ONLY && !ONLY.has(id)) continue;
    await fn();
    writeResults();
    console.log(`scenario ${id} done, results flushed`);
  }

  console.log(`\nwrote ${jsonPath}`);
  console.log(`wrote ${mdPath}`);
}

main()
  .catch((e) => {
    console.error(e);
    process.exitCode = 1;
  })
  .finally(() => stopMock().then(() => process.exit(process.exitCode ?? 0)));
