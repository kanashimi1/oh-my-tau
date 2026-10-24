// Extract name / first description line / JSON parameter schema for omp's built-in
// tools (read, bash, edit, glob, grep, find, write) from pi-coding-agent and write
// them to tests-tools/ts_tools.json.
//
// Run: bun tests-tools/ts_schemas.ts
//
// The omp TS sources live outside this repo: set OMP_TS_SRC to the
// pi-coding-agent package root (default: vendor-src/oh-my-pi/packages/coding-agent).
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const TS_SRC = process.env.OMP_TS_SRC ?? join(dirname(fileURLToPath(import.meta.url)), "../vendor-src/oh-my-pi/packages/coding-agent");
const load = (rel: string) => import(join(TS_SRC, rel));
const { ReadTool } = await load("src/tools/read.ts");
const { BashTool } = await load("src/tools/bash.ts");
const { EditTool } = await load("src/edit/index.ts");
const { GlobTool } = await load("src/tools/glob.ts");
const { GrepTool } = await load("src/tools/grep.ts");
const { FindTool } = await load("src/tools/jfind/index.ts");
const { WriteTool } = await load("src/tools/write.ts");
const { TaskTool } = await load("src/task/index.ts");
const { EvalTool } = await load("src/tools/eval.ts");
const { AstGrepTool } = await load("src/tools/ast-grep.ts");
const { AstEditTool } = await load("src/tools/ast-edit.ts");
const { TodoTool } = await load("src/tools/todo.ts");
const { AskTool } = await load("src/tools/ask.ts");
const { WebSearchTool } = await load("src/web/search");
const { WaitTool } = await load("src/tools/wait");
const { ThinkTool } = await load("src/tools/think");
const { YieldTool } = await load("src/tools/yield");
const { GoalTool } = await load("src/goals/tools/goal-tool");

// Minimal ToolSession stub: tools only read config/description state here, never
// execute. Declared structurally so no omp types need to be imported.
interface StubSession {
	// Must survive `cfg.get(scope)` in src/config/registry.ts (reads revision/valueCache).
	settings: { revision: number; valueCache: Record<string, unknown> };
	isToolActive: () => boolean;
	getActiveModel: () => undefined;
	rawValue: (key: unknown) => undefined;
	getSessionSpawns: () => string;
	deviceOnlyWrite: boolean;
	pendingFullWriteDescription: boolean;
}

function stubSession(): StubSession {
	return {
		settings: { revision: 0, valueCache: {}, rawValue: () => undefined, warnState: { invalid: new Map(), items: new Map() }, getProvenance: () => undefined, getShellConfig: () => ({ env: {} }) },
		isToolActive: () => true,
		getActiveModel: () => undefined,
		rawValue: () => undefined,
		getSessionSpawns: () => "*",
		deviceOnlyWrite: false,
		pendingFullWriteDescription: false,
	};
}

interface ExtractedTool {
	name: string;
	description_first_line: string;
	description: string;
	parameters: unknown;
}

interface ToolLike {
	description: string;
	parameters: unknown;
}

const NAMES = ["read", "bash", "edit", "glob", "grep", "find", "write", "task", "eval", "todo", "ask", "ast_grep", "ast_edit", "web_search", "wait", "think", "yield", "goal"];

const session = stubSession();
const factories: Array<(s: StubSession) => ToolLike> = [
	s => new ReadTool(s),
	s => new BashTool(s),
	s => new EditTool(s, 'replace'), // pin Replace: omp-rs default mode for glm-style models
	s => new GlobTool(s, { rootPathAlias: true }),
	s => new GrepTool(s),
	s => new FindTool(s),
	s => new WriteTool(s),
	// task/eval need a session with many optional hooks (task discovery,
	// spawn policy, eval backend config); a permissive proxy stub satisfies
	// every optional lookup and yields the protocol-default render.
	s => (TaskTool as any).create(permissiveStub()),
	s => new EvalTool(permissiveStub()),
	s => new TodoTool(s),
	s => new AskTool(s),
	s => new AstGrepTool(session as any),
	s => new AstEditTool(session as any),
	s => new WebSearchTool(s),
	s => new WaitTool(s as any),
	s => new ThinkTool(),
	s => new YieldTool({ ...s, outputSchema: undefined } as any),
	s => new GoalTool(s as any),
];

function permissiveStub(): any {
	return new Proxy({}, {
		get(_t, key) {
			if (key === "settings")
				return { revision: 0, valueCache: {}, rawValue: () => undefined, warnState: { invalid: new Map(), items: new Map() }, getProvenance: () => undefined, getShellConfig: () => ({ env: {} }) };
			if (key === "cwd") return process.cwd();
			if (key === "taskDepth") return 0;
			if (typeof key === "symbol") return undefined;
			return (..._a: unknown[]) => undefined;
		},
	});
}

// omp tool schemas are ArkType Type objects: `toJsonSchema()` yields the canonical
// JSON Schema. Older TypeBox-style schemas (plain objects) are round-tripped as-is.
function plainify(value: unknown, seen: Set<object> = new Set()): unknown {
	if (Array.isArray(value)) return value.map(v => plainify(v, seen));
	if (value !== null && (typeof value === "object" || typeof value === "function")) {
		if (seen.has(value)) return undefined; // break schema self/cyclic references
		seen.add(value);
		const source = value as Record<string, unknown>;
		const out: Record<string, unknown> = {};
		for (const key of Object.keys(source)) out[key] = plainify(source[key], seen);
		return out;
	}
	return value;
}

function toSchema(parameters: unknown): unknown {
	// ArkType Type objects are functions carrying `.json` / `.toJsonSchema()`.
	if (typeof parameters === "function" || parameters !== null && typeof parameters === "object") {
		const candidate = parameters as { toJsonSchema?: unknown; json?: unknown };
		if (typeof candidate.toJsonSchema === "function") {
			return JSON.parse(JSON.stringify(candidate.toJsonSchema())) as unknown;
		}
		if (candidate.json !== undefined) return JSON.parse(JSON.stringify(candidate.json)) as unknown;
	}
	if (parameters !== null && typeof parameters === "object") {
		return JSON.parse(JSON.stringify(parameters)) as unknown;
	}
	throw new Error(`unsupported schema object: ${typeof parameters}`);
}

const tools: ExtractedTool[] = [];
// top-level await: task tool construction is async
void 0;
const failures: string[] = [];

for (const [i, make] of factories.entries()) {
	const name = NAMES[i];
	try {
		const tool = await make(session);
		tools.push({
			name,
			description_first_line: tool.description.split("\n", 1)[0] ?? "",
			description: tool.description,
			parameters: toSchema(tool.parameters),
		});
	} catch (e) {
		failures.push(`${name}: ${e instanceof Error ? (e.stack ?? e.message) : String(e)}`);
	}
}

const outDir = dirname(fileURLToPath(import.meta.url));
const outPath = join(outDir, "ts_tools.json");
writeFileSync(outPath, `${JSON.stringify(tools, null, "\t")}\n`);

if (failures.length > 0) {
	console.error(`FAILED tools (${failures.length}):\n${failures.join("\n\n")}`);
	process.exit(1);
}
console.log(`wrote ${outPath} with ${tools.map(t => t.name).join(", ")}`);
