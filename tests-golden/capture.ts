#!/usr/bin/env bun
// Usage: bun capture.ts <ts|rs> [binary] [scenario...]
//   ts: drives `omp --mode rpc --no-session` -> ts/<scenario>.jsonl
//   rs: drives `<binary> --mode rpc` (default ../target/debug/omt) -> rs/<scenario>.jsonl
// Each line: {"dir":"in"|"out","frame":{...}}  (out = sent to agent, in = received)
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

const here = dirname(new URL(import.meta.url).pathname);
const target = process.argv[2] ?? "ts";
const binArg = process.argv[3];
const only = process.argv.slice(4);
const MODEL = "zai/glm-5.3-flash";

type Cmd = { type: string; [k: string]: unknown };
type Step = { cmd: Cmd; until?: (f: any) => boolean; timeoutMs?: number };

const scenarios: Record<string, Step[]> = {
	state: [
		{ cmd: { type: "get_state" } },
		{ cmd: { type: "get_available_models" } },
		{ cmd: { type: "set_thinking_level", level: "low" } },
		{ cmd: { type: "get_state" } },
		{ cmd: { type: "get_messages" } },
		{ cmd: { type: "get_session_stats" } },
		{ cmd: { type: "get_available_commands" } },
		{ cmd: { type: "get_available_thinking_levels" } },
		{ cmd: { type: "no_such_command" } },
	],
	prompt: [
		{ cmd: { type: "set_model", provider: "zai", modelId: "glm-5.3-flash" } },
		{ cmd: { type: "set_thinking_level", level: "off" } },
		{
			cmd: { type: "prompt", message: "Answer with the single word ok. Do not call any tools." },
			until: f => f.type === "agent_end",
			timeoutMs: 300_000,
		},
		{ cmd: { type: "get_messages" } },
		{ cmd: { type: "get_session_stats" } },
		{ cmd: { type: "get_last_assistant_text" } },
	],
	abort: [
		{ cmd: { type: "set_model", provider: "zai", modelId: "glm-5.3-flash" } },
		{ cmd: { type: "set_thinking_level", level: "off" } },
		{ cmd: { type: "prompt", message: "count slowly from 1 to 200, one number per line" }, until: f => f.type === "agent_start", timeoutMs: 30_000 },
		{ cmd: { type: "abort" }, until: f => f.type === "agent_end", timeoutMs: 30_000 },
		{ cmd: { type: "get_state" } },
	],
	steer_follow_up: [
		{ cmd: { type: "set_model", provider: "zai", modelId: "glm-5.3-flash" } },
		{ cmd: { type: "set_thinking_level", level: "off" } },
		{ cmd: { type: "prompt", message: "Write a brief haiku about programming. Do not call any tools." }, until: f => f.type === "agent_start", timeoutMs: 30_000 },
		{ cmd: { type: "steer", message: "Please be more detailed." }, timeoutMs: 10_000 },
		{ cmd: { type: "follow_up", message: "And add another line." }, timeoutMs: 10_000 },
		{ cmd: { type: "abort" }, until: f => f.type === "agent_end", timeoutMs: 30_000 },
		{ cmd: { type: "get_state" } },
	],
	set_model_prompt: [
		{ cmd: { type: "set_model", provider: "zai", modelId: "glm-5.3-flash" } },
		{ cmd: { type: "set_thinking_level", level: "off" } },
		{ cmd: { type: "prompt", message: "What is 2+2? Answer with a single number. Do not call any tools." }, until: f => f.type === "agent_end", timeoutMs: 300_000 },
		{ cmd: { type: "get_session_stats" } },
		{ cmd: { type: "get_last_assistant_text" } },
	],
	new_session: [
		{ cmd: { type: "new_session" } },
		{ cmd: { type: "get_state" } },
		{ cmd: { type: "get_session_stats" } },
		{ cmd: { type: "get_messages" } },
	],
	compact: [
		{ cmd: { type: "compact", customInstructions: "Focus on preserving the user\'s intent." } },
		{ cmd: { type: "get_state" } },
	],
	branch_messages: [
		{ cmd: { type: "get_branch_messages" } },
		{ cmd: { type: "branch", entryId: "msg-0000000000000000" } },
		{ cmd: { type: "get_state" } },
	],
	thinking_state: [
		{ cmd: { type: "set_thinking_level", level: "high" } },
		{ cmd: { type: "get_state" } },
		{ cmd: { type: "set_thinking_level", level: "off" } },
		{ cmd: { type: "get_state" } },
		{ cmd: { type: "get_available_thinking_levels" } },
	],
	bash_command: [
		{ cmd: { type: "bash", command: "echo golden_test_output" } },
		{ cmd: { type: "get_session_stats" } },
		{ cmd: { type: "get_state" } },
	],
	abort_mid_tool: [
		{ cmd: { type: "set_model", provider: "zai", modelId: "glm-5.3-flash" } },
		{ cmd: { type: "set_thinking_level", level: "off" } },
		{ cmd: { type: "prompt", message: "Run the bash command: sleep 30. Use the bash tool. Only call one tool and say nothing else." }, until: f => f.type === "tool_execution_start", timeoutMs: 30_000 },
		{ cmd: { type: "abort" }, until: f => f.type === "agent_end", timeoutMs: 30_000 },
		{ cmd: { type: "get_state" } },
	],
};

function run(name: string, steps: Step[]): Promise<number> {
	const [bin, args] =
		target === "ts"
			? ["omp", ["--mode", "rpc", "--no-session", "--model", MODEL]]
			: [resolve(here, binArg ?? "../target/debug/omt"), ["--mode", "rpc", "--no-session", "--model", MODEL]];
	const rows: { dir: string; frame: unknown }[] = [];
	const child = spawn(bin, args, { stdio: ["pipe", "pipe", "ignore"], cwd: here });
	let buf = "";
	const waiters: ((f: any) => void)[] = [];
	child.stdout.on("data", (d: Buffer) => {
		buf += d.toString();
		let i: number;
		while ((i = buf.indexOf("\n")) >= 0) {
			const line = buf.slice(0, i).trim();
			buf = buf.slice(i + 1);
			if (!line) continue;
			let frame: unknown;
			try {
				frame = JSON.parse(line);
			} catch {
				frame = { type: "__unparsed", raw: line };
			}
			rows.push({ dir: "in", frame });
			for (const w of [...waiters]) w(frame);
		}
	});
	const waitFor = (pred: (f: any) => boolean, ms: number) =>
		new Promise<boolean>(res => {
			const t = setTimeout(() => {
				waiters.splice(waiters.indexOf(w), 1);
				res(false);
			}, ms);
			const w = (f: any) => {
				if (pred(f)) {
					clearTimeout(t);
					waiters.splice(waiters.indexOf(w), 1);
					res(true);
				}
			};
			waiters.push(w);
		});
	const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
	return (async () => {
		await sleep(1500); // let ready/startup frames arrive
		let n = 0;
		for (const s of steps) {
			const id = `g${++n}`;
			const cmd = { id, ...s.cmd };
			const resp = waitFor(f => f.type === "response" && f.id === id, s.timeoutMs ?? 15_000);
			const until = s.until ? waitFor(s.until, s.timeoutMs ?? 15_000) : Promise.resolve(true);
			rows.push({ dir: "out", frame: cmd });
			child.stdin.write(`${JSON.stringify(cmd)}\n`);
			await Promise.all([resp, until]);
			await sleep(200);
		}
		child.stdin.end();
		await sleep(500);
		child.kill("SIGTERM");
		const dir = join(here, target);
		mkdirSync(dir, { recursive: true });
		writeFileSync(join(dir, `${name}.jsonl`), rows.map(r => JSON.stringify(r)).join("\n") + "\n");
		return rows.length;
	})();
}

for (const [name, steps] of Object.entries(scenarios)) {
	if (only.length && !only.includes(name)) continue;
	console.log(name, await run(name, steps), "rows");
}
process.exit(0);
