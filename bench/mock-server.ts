#!/usr/bin/env bun
// Mock OpenAI-compatible provider for benchmarking omp-rs vs omp (TS).
// No network, scripted streams, configurable latency.
//
// Endpoints:
//   POST /v1/chat/completions  (SSE streaming, api: openai-completions)
//   GET  /v1/models
//
// Script selection by model id:
//   mock/text       -> single short text reply
//   mock/toolturn   -> thinking + 5x read tool_calls, then final text after tool results
//   mock/bench-<t>  -> thinking + 1x tool_call(<t>), then final text (t in read|grep|glob|find|edit|bash)
//
// Env:
//   PORT             listen port (default 8787 -> use 0 for ephemeral? no, fixed required by models.yml)
//   MOCK_LATENCY_MS  per-chunk delay ms (default 0)
//   MOCK_LOG         path to JSONL event log (req arrival / response end)
//   MOCK_READ_FILE   file read by read/edit scripts (default: this repo's rpc.rs)
//   MOCK_EDIT_FILE   file edited by edit script (default: MOCK_READ_FILE)
//   MOCK_WORKDIR     workdir passed to grep/glob/find/bash tools

const PORT = Number(process.env.PORT ?? 8787);
const LATENCY = Number(process.env.MOCK_LATENCY_MS ?? 0);
const LOG = process.env.MOCK_LOG ?? "";
const REPO_ROOT = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const READ_FILE = process.env.MOCK_READ_FILE ?? `${REPO_ROOT}/crates/omp-cli/src/rpc.rs`;
const EDIT_FILE = process.env.MOCK_EDIT_FILE ?? READ_FILE;
const WORKDIR = process.env.MOCK_WORKDIR ?? REPO_ROOT;

function logEvent(ev: Record<string, unknown>) {
  if (!LOG) return;
  const line = JSON.stringify({ t: Date.now(), ...ev }) + "\n";
  appendFileSync(LOG, line);
}
import { appendFileSync } from "node:fs";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function toolArgs(name: string): Record<string, unknown> {
  switch (name) {
    case "read":
      return { path: READ_FILE };
    case "grep":
      return { pattern: "fn dispatch", path: WORKDIR };
    case "glob":
      return { pattern: "crates/omp-cli/src/**/*.rs", path: WORKDIR };
    case "find":
      return { path: WORKDIR, pattern: "*.rs" };
    case "edit":
      return { path: EDIT_FILE, oldText: "alpha-bravo-charlie", newText: "alpha-bravo-delta" };
    case "bash":
      return { command: "echo benchmark-bash-spawn" };
    default:
      return { path: READ_FILE };
  }
}

type Chunk = Record<string, unknown>;

interface Msg {
  role?: string;
  content?: unknown;
}

function scriptFor(rawModel: string, messages: Message[]): { chunks: Chunk[]; finish: string } {
  const model = rawModel.replace(/^mock\//, "");
    const hasToolResults = messages.some((m) => m.role === "tool");
  const mk = (i: number, name: string, args: Record<string, unknown>): Chunk => ({
    delta: {
      role: "assistant",
      tool_calls: [
        {
          index: i,
          id: `call_${name}_${i}`,
          type: "function",
          function: { name, arguments: JSON.stringify(args) },
        },
      ],
    },
  });
  if (model === "text") {
    return {
      chunks: [
        { delta: { role: "assistant", content: "Understood. " } },
        { delta: { content: "The benchmark reply is ready." } },
      ],
      finish: "stop",
    };
  }
  if (model === "toolturn") {
    if (!hasToolResults) {
      const chunks: Chunk[] = [{ delta: { role: "assistant", content: "", reasoning_content: "planning reads" } }];
      for (let i = 0; i < 5; i++) chunks.push(mk(i, "read", toolArgs("read")));
      return { chunks, finish: "tool_calls" };
    }
    return { chunks: [{ delta: { role: "assistant", content: "All five reads complete." } }], finish: "stop" };
  }
  const m = /^bench-(read|grep|glob|find|edit|bash)$/.exec(model);
  if (m) {
    const tool = m[1];
    if (!hasToolResults) {
      return {
        chunks: [
          { delta: { role: "assistant", content: "", reasoning_content: `bench ${tool}` } },
          mk(0, tool, toolArgs(tool)),
        ],
        finish: "tool_calls",
      };
    }
    return { chunks: [{ delta: { role: "assistant", content: `${tool} done.` } }], finish: "stop" };
  }
  // default: text
  return { chunks: [{ delta: { role: "assistant", content: "ok" } }], finish: "stop" };
}

Bun.serve({
  port: PORT,
  hostname: "127.0.0.1",
  async fetch(req) {
    const url = new URL(req.url);
    if (url.pathname === "/v1/models") {
      const ids = ["text", "toolturn", "bench-read", "bench-grep", "bench-glob", "bench-find", "bench-edit", "bench-bash"];
      return Response.json({
        object: "list",
        // identity: the harness checks this to make sure it is talking to the mock it just spawned
        mock_log: process.env.MOCK_LOG ?? "",
        data: ids.map((id) => ({ id: `mock/${id}`, object: "model", created: 0, owned_by: "mock" })),
      });
    }
    if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
      const body = await req.json();
      const model: string = body.model ?? "mock/text";
      const { chunks, finish } = scriptFor(model, body.messages ?? []);
      logEvent({ kind: "req", model, n_messages: (body.messages ?? []).length });
      const stream = body.stream !== false;
      const id = `chatcmpl-${Date.now()}`;
      const payload = (c: Chunk) =>
        JSON.stringify({
          id,
          object: "chat.completion.chunk",
          created: 0,
          model,
          choices: [{ index: 0, delta: c.delta, finish_reason: c.finish ?? null }],
        });
      if (!stream) {
        const text = chunks
          .map((c) => {
            const d = c.delta;
            return typeof d === "object" && d !== null && "content" in d ? String((d as { content?: unknown }).content ?? "") : "";
          })
          .join("");
        logEvent({ kind: "resp_end", model });
        return Response.json({
          id,
          object: "chat.completion",
          created: 0,
          model,
          choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: finish }],
        });
      }
      const encoder = new TextEncoder();
      const readable = new ReadableStream<Uint8Array>({
        async start(controller) {
          for (const c of chunks) {
            if (LATENCY > 0) await sleep(LATENCY);
            controller.enqueue(encoder.encode(`data: ${payload(c)}\n\n`));
          }
          if (LATENCY > 0) await sleep(LATENCY);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ id, object: "chat.completion.chunk", created: 0, model, choices: [{ index: 0, delta: {}, finish_reason: finish }] })}\n\n`));
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          logEvent({ kind: "resp_end", model });
          controller.close();
        },
      });
      return new Response(readable, {
        headers: {
          "content-type": "text/event-stream",
          "cache-control": "no-cache",
          connection: "keep-alive",
        },
      });
    }
    return new Response("not found", { status: 404 });
  },
});

console.log(`mock provider on 127.0.0.1:${PORT} latency=${LATENCY}ms log=${LOG || "-"}`);
