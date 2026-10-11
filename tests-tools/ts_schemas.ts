// Extract name / first description line / JSON parameter schema for omp's built-in
// tools (read, bash, edit, glob, grep, find, write) from pi-coding-agent and write
// them to tests-tools/ts_tools.json.
//
// Run: bun tests-tools/ts_schemas.ts
//
// Note: static imports of absolute paths are used because the omp TS sources live
// outside this repo; the specifiers are fixed install locations, known at author time.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ReadTool } from "/home/kaneki/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/src/tools/read.ts";
import { BashTool } from "/home/kaneki/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/src/tools/bash.ts";
import { EditTool } from "/home/kaneki/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/src/edit/index.ts";
import { GlobTool } from "/home/kaneki/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/src/tools/glob.ts";
import { GrepTool } from "/home/kaneki/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/src/tools/grep.ts";
import { FindTool } from "/home/kaneki/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/src/tools/jfind/index.ts";
import { WriteTool } from "/home/kaneki/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/src/tools/write.ts";

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
	parameters: unknown;
}

interface ToolLike {
	description: string;
	parameters: unknown;
}

const NAMES = ["read", "bash", "edit", "glob", "grep", "find", "write"];

const session = stubSession();
const factories: Array<(s: StubSession) => ToolLike> = [
	s => new ReadTool(s),
	s => new BashTool(s),
	s => new EditTool(s, 'replace'), // pin Replace: omp-rs default mode for glm-style models
	s => new GlobTool(s, { rootPathAlias: true }),
	s => new GrepTool(s),
	s => new FindTool(s),
	s => new WriteTool(s),
];

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
const failures: string[] = [];

factories.forEach((make, i) => {
	const name = NAMES[i];
	try {
		const tool = make(session);
		tools.push({
			name,
			description_first_line: tool.description.split("\n", 1)[0] ?? "",
			parameters: toSchema(tool.parameters),
		});
	} catch (e) {
		failures.push(`${name}: ${e instanceof Error ? (e.stack ?? e.message) : String(e)}`);
	}
});

const outDir = dirname(fileURLToPath(import.meta.url));
const outPath = join(outDir, "ts_tools.json");
writeFileSync(outPath, `${JSON.stringify(tools, null, "\t")}\n`);

if (failures.length > 0) {
	console.error(`FAILED tools (${failures.length}):\n${failures.join("\n\n")}`);
	process.exit(1);
}
console.log(`wrote ${outPath} with ${tools.map(t => t.name).join(", ")}`);
