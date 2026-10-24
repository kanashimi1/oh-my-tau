// Records TS `todo` tool results (text, isError, details) for scripted op sequences.
// The Rust port's unit tests (crates/omp-cli/src/tools/todo.rs) replay the same
// sequences and must reproduce every recorded step.
//
// Run: bun tests-tools/todo_golden.ts   (writes tests-tools/todo_golden.json)
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// TS sources live outside this repo: set OMP_TS_SRC to the pi-coding-agent package root.
const TS_SRC = process.env.OMP_TS_SRC ?? join(dirname(fileURLToPath(import.meta.url)), "../vendor-src/oh-my-pi/packages/coding-agent");
const { TodoTool } = await import(join(TS_SRC, "src/tools/todo.ts"));

type Session = ConstructorParameters<typeof TodoTool>[0];

interface Scenario {
	name: string;
	steps: Array<Record<string, unknown>>;
}

const SCENARIOS: Scenario[] = [
	{
		name: "flow_init_start_done_block_drop_rm_append",
		steps: [
			{ op: "init", list: [{ phase: "Setup", items: ["install deps", "write config"] }, { phase: "Build", items: ["compile", "bundle"] }] },
			{ op: "done", task: "install deps" },
			{ op: "start", task: "compile" },
			{ op: "view" },
			{ op: "block", task: "write config", reason: "waiting   on\nuser" },
			{ op: "unblock", task: "write config" },
			{ op: "drop", phase: "Build" },
			{ op: "rm", task: "bundle" },
			{ op: "append", phase: "Build", items: ["ship"] },
			{ op: "done", task: "write config" },
			{ op: "done", task: "ship" },
		],
	},
	{
		name: "flat_init_inference",
		steps: [
			{ items: ["a", "b"] },
			{ op: "done", task: "a" },
			{ items: ["c"], phase: "Extra" },
			{ items: ["d"], phase: "Extra" },
			{ op: "view" },
		],
	},
	{
		name: "errors_are_discarded_atomically",
		steps: [
			{ op: "view" },
			{ op: "init", list: [{ phase: "P", items: ["a"] }, { phase: "P", items: ["a"] }] },
			{ op: "init", list: [{ phase: "P", items: ["a", "b"] }] },
			{ op: "start" },
			{ op: "done", task: "task-1" },
			{ op: "done", task: "zzz" },
			{ op: "append", phase: "Q", items: [] },
			{ op: "append", items: ["x"] },
			{ op: "append", phase: "P", items: ["a", "c"] },
			{ op: "block", reason: "no target" },
			{ op: "unblock" },
			{ op: "rm", phase: "Nope" },
			{ op: "init", list: [] },
			{ op: "rm" },
			{ op: "view" },
		],
	},
	{
		name: "in_progress_normalization",
		steps: [
			{ op: "init", list: [{ phase: "One", items: ["a", "b", "c"] }] },
			{ op: "start", task: "b" },
			{ op: "start", task: "c" },
			{ op: "done", task: "c" },
			{ op: "block", task: "a" },
			{ op: "view" },
			{ op: "done", task: "b" },
			{ op: "view" },
		],
	},
	{
		name: "completion_transitions_and_worked_ahead",
		steps: [
			{ op: "init", list: [{ phase: "P1", items: ["a", "b"] }, { phase: "P2", items: ["c"] }] },
			{ op: "done", task: "c" },
			{ op: "done", task: "c" },
			{ op: "view" },
			{ op: "drop", task: "b" },
			{ op: "done", task: "a" },
		],
	},
	{
		name: "block_phase_and_reason_refine",
		steps: [
			{ op: "init", list: [{ phase: "W", items: ["p", "q"] }] },
			{ op: "done", task: "q" },
			{ op: "block", phase: "W" },
			{ op: "block", task: "p", reason: "needs   review" },
			{ op: "view" },
			{ op: "unblock", phase: "W" },
			{ op: "view" },
		],
	},
	{
		name: "empty_and_clear",
		steps: [
			{ op: "view" },
			{ op: "rm" },
			{ op: "init", list: [{ phase: "E", items: ["only"] }] },
			{ op: "rm", task: "only" },
			{ op: "view" },
		],
	},
];

const outDir = dirname(fileURLToPath(import.meta.url));
const results: unknown[] = [];

for (const scenario of SCENARIOS) {
	let phases: unknown[] = [];
	const session = {
		getTodoPhases: () => structuredClone(phases),
		setTodoPhases: (next: unknown[]) => {
			phases = structuredClone(next);
		},
		getSessionFile: () => "/tmp/todo-golden.jsonl",
		settings: { revision: 0, valueCache: {}, rawValue: () => undefined, warnState: { invalid: new Map(), items: new Map() }, getProvenance: () => undefined, getShellConfig: () => ({ env: {} }) },
		cwd: process.cwd(),
	} as unknown as Session; // unchecked cast: schema-free stub; execute only touches the todo state hooks
	const tool = new TodoTool(session);
	const steps: unknown[] = [];
	for (const params of scenario.steps) {
		const result = await tool.execute("golden", params as never);
		const text = result.content.map(c => (c.type === "text" ? c.text : "")).join("");
		steps.push({ params, isError: result.isError === true, text, details: result.details });
	}
	results.push({ name: scenario.name, steps });
}

const outPath = join(outDir, "todo_golden.json");
writeFileSync(outPath, `${JSON.stringify(results, null, 1)}\n`);
console.log(`wrote ${outPath}: ${results.length} scenarios`);
