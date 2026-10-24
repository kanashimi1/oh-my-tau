#!/usr/bin/env bun
// Slash-command golden capture (SlashCmds): drives `--mode rpc --no-session`
// on the TS and RS binaries and records the frames for
//   1. get_available_commands
//   2. prompt "/model"            (builtin, local)
//   3. prompt "/fx-cmd hello"     (extension command, local)
//   4. prompt "/no-such-cmd hi"   (falls through to the model as a prompt)
// Usage: bun slash-capture.ts <ts|rs> [binary]
// Each line: {"dir":"in"|"out","frame":{...}} -> <target>/slash.jsonl
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

const here = dirname(new URL(import.meta.url).pathname);
const target = process.argv[2] ?? "ts";
const bin = process.argv[3] ?? (target === "ts" ? "omp" : resolve(here, "../target/debug/omt"));
const fixture = resolve(here, "../crates/omp-ext/tests/fixtures/fixture.ts");
const MODEL = "zai/glm-5.3-flash";

const proc = spawn(bin, ["--mode", "rpc", "--no-session", "--extension", fixture], { cwd: here, stdio: ["pipe", "pipe", "inherit"] });
const frames = [];
let waiting = [];
let buf = "";
proc.stdout.on("data", d => {
	buf += d.toString();
	let idx;
	while ((idx = buf.indexOf("\n")) !== -1) {
		const line = buf.slice(0, idx);
		buf = buf.slice(idx + 1);
		if (!line.trim()) continue;
		const frame = JSON.parse(line);
		const wrapped = { dir: "in", frame, _i: frames.length };
		frames.push(wrapped);
		waiting = waiting.filter(w => !w.pred(wrapped) || (w.resolve(), false));
	}
});

function send(cmd) {
	frames.push({ dir: "out", frame: cmd });
	proc.stdin.write(JSON.stringify(cmd) + "\n");
}
function untilImpl(label, pred, timeoutMs = 10_000) {
	const { promise, resolve, reject } = Promise.withResolvers();
	const start = frames.length;
	const match = f => f._i >= start && f.dir === "in" && pred(f.frame);
	const hit = frames.some(match);
	if (hit) return resolve();
	const t = setTimeout(() => reject(new Error(`timeout waiting for ${label}`)), timeoutMs);
	waiting.push({ pred: match, resolve: () => (clearTimeout(t), resolve()) });
	return promise;
}
function until(label, pred, timeoutMs = 10_000) {
	return untilImpl(label, pred, timeoutMs);
}

let id = 0;
const rpc = (type, extra = {}) => ({ id: String(++id), type, ...extra });

await until("ready", f => f.type === "ready", 60_000);

// 1. catalog
send(rpc("get_available_commands"));
await until("catalog", f => f.command === "get_available_commands" && f.success !== undefined);

// 2. builtin /model (no args: show current; with args: set)
send(rpc("set_model", { provider: "zai", modelId: MODEL.replace("zai/", "") }));
await until("set_model", f => f.command === "set_model");
send(rpc("prompt", { message: "/model" }));
await until("prompt-response", f => f.command === "prompt" && f.success !== undefined);
send(rpc("prompt", { message: "/effort" }));
await until("prompt-response", f => f.command === "prompt" && f.success !== undefined);

// 3. extension command
send(rpc("prompt", { message: "/fx-cmd hello" }));
await until("prompt-response", f => f.command === "prompt" && f.success !== undefined);

// 4. unknown slash: forwarded to the model as a prompt
send(rpc("abort", {}));
await until("abort", f => f.command === "abort");
send(rpc("prompt", { message: "/no-such-cmd hi", streamingBehavior: undefined }));
await until("run-start", f => f.type === "prompt_result" || f.type === "agent_start", 60_000);
send(rpc("abort", {}));
await until("agent_end", f => f.type === "agent_end", 120_000);

proc.kill();
const out = frames.filter(f => !/stdio|api_key/.test(JSON.stringify(f.frame)));
mkdirSync(join(here, target), { recursive: true });
writeFileSync(join(here, target, "slash.jsonl"), out.map(f => JSON.stringify(f)).join("\n") + "\n");
console.log(`wrote ${target}/slash.jsonl (${out.length} frames)`);
