// Diff omp-rs built-in tool schemas (parsed from Rust source; the omp-rs CLI has no
// schema-dump flag) against the TS-extracted reference in tests-tools/ts_tools.json.
//
// Run: bun tests-tools/compare.ts
// Produces per tool: missing/extra params, type mismatches, required mismatches.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const RUST_TOOLS = join(ROOT, "crates/omp-cli/src/tools.rs");
const RUST_EDIT = join(ROOT, "crates/omp-cli/src/tools/edit.rs");
const EDIT_REPLACE_PROMPT = join(ROOT, "vendor/pi-edit/prompts/replace.md");
const TS_JSON = join(dirname(fileURLToPath(import.meta.url)), "ts_tools.json");

interface ToolDef {
	name: string;
	description_first_line: string;
	description?: string;
	parameters: Record<string, unknown>;
}

/** Tools compared byte-for-byte (canonical JSON + full description), not just per-param. */
const STRICT_TOOLS = new Set(["todo", "ask", "checkpoint", "rewind", "wait", "think", "yield", "goal"]);

/** Recursively sort object keys so JSON equality ignores key order. */
function canonical(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(canonical);
	if (value !== null && typeof value === "object") {
		const out: Record<string, unknown> = {};
		for (const key of Object.keys(value).sort()) out[key] = canonical((value as Record<string, unknown>)[key]);
		return out;
	}
	return value;
}

/** JSON.parse tolerant of Rust `json!` syntax: drop trailing commas (outside strings). */
function parseJsonLiteral(literal: string): unknown {
	let parsed: unknown;
	try {
		return JSON.parse(literal);
	} catch {
		let out = "";
		let inString = false;
		let escaped = false;
		for (let i = 0; i < literal.length; i++) {
			const ch = literal[i];
			if (inString) {
				out += ch;
				if (escaped) escaped = false;
				else if (ch === "\\") escaped = true;
				else if (ch === '"') inString = false;
				continue;
			}
			if (ch === '"') {
				inString = true;
				out += ch;
				continue;
			}
			if (ch === ",") {
				let j = i + 1;
				while (j < literal.length && /\s/.test(literal[j])) j++;
				if (literal[j] === "}" || literal[j] === "]") continue;
			}
			out += ch;
		}
		parsed = JSON.parse(out);
	}
	return parsed;
}

/** Extract the balanced JSON payload of the i-th `json!(` occurrence in `src`. */
function jsonLiteralAt(src: string, from: number): string {
	const start = src.indexOf("json!(", from);
	if (start < 0) throw new Error("json!( not found");
	let depth = 0;
	for (let i = start + "json!(".length; i < src.length; i++) {
		const ch = src[i];
		if (ch === "{") depth++;
		else if (ch === "}") {
			depth--;
			if (depth === 0) return src.slice(start + "json!(".length, i + 1);
		}
	}
	throw new Error("unbalanced json!(");
}

/** Find the `mk(` block for a tool name, returning its json! payload index. */
function rustInlineTool(src: string, name: string): string {
	const marker = `mk(\n\t\t\t"${name}"`;
	let at = src.indexOf(marker);
	if (at < 0) {
		// tolerant to whitespace variants
		const re = new RegExp(`mk\\(\\s*"${name}"`);
		const m = re.exec(src);
		if (!m) throw new Error(`tool "${name}" not found in tools.rs`);
		at = m.index;
	}
	return jsonLiteralAt(src, at);
}

function parseRustTools(): ToolDef[] {
	const toolsSrc = readFileSync(RUST_TOOLS, "utf8");
	const editSrc = readFileSync(RUST_EDIT, "utf8");
	const replacePrompt = readFileSync(EDIT_REPLACE_PROMPT, "utf8");

	// Inline literal tools: read, bash, glob, grep, write, find, task, eval.
	const names = ["read", "bash", "glob", "grep", "write", "find"];
	const defs: ToolDef[] = names.map(name => ({
		name,
		description_first_line: "(from tools.rs, see description const)",
		parameters: parseJsonLiteral(rustInlineTool(toolsSrc, name)) as Record<string, unknown>,
	}));
	// Descriptions: resolve the *_DESCRIPTION consts for exactness.
	for (const def of defs) {
		const re = new RegExp(`mk\\(\\s*"${def.name}",\\s*([A-Z_]+)_DESCRIPTION\\.to_string\\(\\)`);
		const m = re.exec(toolsSrc);
		if (m) {
			const cre = new RegExp(`const ${m[1]}_DESCRIPTION: &str = "((?:[^"\\\\]|\\\\.)*)";`);
			const cm = cre.exec(toolsSrc);
			if (cm) def.description_first_line = (JSON.parse(`"${cm[1]}"`) as string).split("\n", 1)[0] ?? "";
		}
	}
	// task/eval: schemas live in their own modules as `json!(...)` literals.
	// task: `schema()` delegates to `base_schema()` with a conditional rubric `json!`
	// in between, so anchor at `base_schema()` to compare the actual wire schema.
	for (const [name, file] of [["task", "crates/omp-cli/src/tools/task.rs"], ["eval", "crates/omp-cli/src/tools/eval.rs"], ["web_search", "crates/omp-cli/src/tools/web_search.rs"]] as const) {
		const mod = readFileSync(join(ROOT, file), "utf8");
		const at = name === "task" ? mod.indexOf("fn base_schema()") : mod.indexOf("pub fn schema()");
		defs.push({
			name,
			description_first_line: "(from module const, filled below)",
			parameters: parseJsonLiteral(jsonLiteralAt(mod, at)) as Record<string, unknown>,
		});
	}
	// ast_grep/ast_edit: schemas live in tools/ast.rs as named `schema_*()` fns.
	{
		const astNames = ["ast_grep", "ast_edit"] as const;
	const mod = readFileSync(join(ROOT, "crates/omp-cli/src/tools/ast.rs"), "utf8");
		const CONST: Record<(typeof astNames)[number], string> = {
			ast_grep: "AST_GREP_DESCRIPTION",
			ast_edit: "AST_EDIT_DESCRIPTION",
		};
		for (const name of astNames) {
			const marker = `pub fn schema_${name}()`;
			const at = mod.indexOf(marker);
			if (at < 0) throw new Error(`schema_${name} not found in ast.rs`);
			const cre = new RegExp(`const ${CONST[name]}: &str = "((?:[^"\\\\]|\\\\.)*)";`);
			const cm = cre.exec(mod);
			const first = cm ? (JSON.parse(`"${cm[1]}"`) as string).split("\n", 1)[0] ?? "" : "";
			defs.push({
				name,
				description_first_line: first,
				parameters: parseJsonLiteral(jsonLiteralAt(mod, at)) as Record<string, unknown>,
			});
		}
	}
	// task: dynamic description = static header (r#"..."# const) + agent list.
	const taskSrc = readFileSync(join(ROOT, "crates/omp-cli/src/tools/task.rs"), "utf8");
	{
		const m = /DESCRIPTION_HEADER: &str = r#?"([^"]*)"#?;/.exec(taskSrc);
		defs.find(d => d.name === "task")!.description_first_line = (m?.[1] ?? "").split("\n", 1)[0] ?? "";
	}
	// eval: static r#"..."# description const.
	{
		const evalSrc = readFileSync(join(ROOT, "crates/omp-cli/src/tools/eval.rs"), "utf8");
		const m = /const DESCRIPTION: &str = r#?"([^"]*)"#?;/.exec(evalSrc);
		defs.find(d => d.name === "eval")!.description_first_line = (m?.[1] ?? "").split("\n", 1)[0] ?? "";
	}
	// web_search: schema from `pub fn schema()`; description from DESCRIPTION const.
	{
		const wsSrc = readFileSync(join(ROOT, "crates/omp-cli/src/tools/web_search.rs"), "utf8");
		const wsDesc = /pub const DESCRIPTION: &str = "((?:[^"\\]|\\.)*)"/.exec(wsSrc);
		const first = wsDesc ? (JSON.parse(`"${wsDesc[1]}"`) as string).split("\n", 1)[0] ?? "" : "";
		defs.find(d => d.name === "web_search")!.description_first_line = first;
	}
	// lsp: schema from `pub fn schema()`; description from a raw-string DESCRIPTION const.
	{
		const lspSrc = readFileSync(join(ROOT, "crates/omp-cli/src/tools/lsp.rs"), "utf8");
		const at = lspSrc.indexOf("pub fn schema()");
		const lspDesc = /pub const DESCRIPTION: &str = r#"([\s\S]*?)"#;/.exec(lspSrc);
		const first = lspDesc ? (lspDesc[1] ?? "").split("\n", 1)[0] ?? "" : "";
		defs.push({
			name: "lsp",
			description_first_line: first,
			parameters: parseJsonLiteral(jsonLiteralAt(lspSrc, at)) as Record<string, unknown>,
		});
	}
	// wait/think/yield/goal: schema in the module (`pub fn schema()`); description
	// is an escaped-string `DESCRIPTION` const (yield_tool.rs holds omp's `yield`).
	for (const [name, file] of [["wait", "crates/omp-cli/src/tools/wait.rs"], ["think", "crates/omp-cli/src/tools/think.rs"], ["yield", "crates/omp-cli/src/tools/yield_tool.rs"], ["goal", "crates/omp-cli/src/tools/goal.rs"]] as const) {
		const mod = readFileSync(join(ROOT, file), "utf8");
		const at = mod.indexOf("pub fn schema()");
		const desc = /const DESCRIPTION: &str = "((?:[^"\\]|\\.)*)";/.exec(mod);
		const full = desc ? (JSON.parse(`"${desc[1]}"`) as string) : "";
		defs.push({
			name,
			description_first_line: full.split("\n", 1)[0] ?? "",
			description: full,
			parameters: parseJsonLiteral(jsonLiteralAt(mod, at)) as Record<string, unknown>,
		});
	}
	// edit: `edit::schema(edit_mode)` + `pi_edit::description(edit_mode)`; compare the
	// Replace mode, the omp-rs default for glm-style models.
	// verify we grabbed the Replace arm: it follows `EditMode::Replace => json!(`
	const repIdx = editSrc.indexOf("EditMode::Replace => json!(");
	const replaceJson = jsonLiteralAt(editSrc, repIdx);
	defs.push({
		name: "edit",
		description_first_line: replacePrompt.split("\n", 1)[0] ?? "",
		parameters: parseJsonLiteral(replaceJson) as Record<string, unknown>,
	});
	// todo/ask: schema in the module; the description is a const (todo: escaped string, ask: raw string).
	{
		const todoSrc = readFileSync(join(ROOT, "crates/omp-cli/src/tools/todo.rs"), "utf8");
		const askSrc = readFileSync(join(ROOT, "crates/omp-cli/src/tools/ask.rs"), "utf8");
		const todoDesc = /const DESCRIPTION: &str = "((?:[^"\\]|\\.)*)";/.exec(todoSrc);
		const askDesc = /const DESCRIPTION: &str = r#"([\s\S]*?)"#;/.exec(askSrc);
		const defsFor: Array<[string, string, RegExpExecArray | null]> = [
			["todo", todoSrc, todoDesc],
			["ask", askSrc, askDesc],
		];
		for (const [name, src, descMatch] of defsFor) {
			const at = src.indexOf("pub fn schema()");
			// todo's const is an escaped string literal; ask's is a raw string (already literal text).
			const fullDesc = descMatch ? (name === "todo" ? (JSON.parse(`"${descMatch[1]}"`) as string) : descMatch[1]) : "";
			defs.push({
				name,
				description_first_line: fullDesc.split("\n", 1)[0] ?? "",
				description: fullDesc,
				parameters: parseJsonLiteral(jsonLiteralAt(src, at)) as Record<string, unknown>,
			});
		}
	}
	// checkpoint/rewind: schemas as `pub fn schema_<name>()`; descriptions as raw-string consts.
	{
		const ckptSrc = readFileSync(join(ROOT, "crates/omp-cli/src/tools/checkpoint.rs"), "utf8");
		for (const [name, fn, constName] of [["checkpoint", "schema_checkpoint", "CHECKPOINT_DESCRIPTION"], ["rewind", "schema_rewind", "REWIND_DESCRIPTION"]] as const) {
			const at = ckptSrc.indexOf(`pub fn ${fn}()`);
			if (at < 0) throw new Error(`${fn} not found in checkpoint.rs`);
			const cre = new RegExp(`pub const ${constName}: &str = r#"(([\\s\\S]*?))"#;`);
			const cm = cre.exec(ckptSrc);
			const fullDesc = cm?.[1] ?? "";
			defs.push({
				name,
				description_first_line: fullDesc.split("\n", 1)[0] ?? "",
				description: fullDesc,
				parameters: parseJsonLiteral(jsonLiteralAt(ckptSrc, at)) as Record<string, unknown>,
			});
		}
	}
	return defs;
}

interface ParamDiff {
	tool: string;
	missing_in_rust: string[];
	extra_in_rust: string[];
	type_mismatches: Array<{ param: string; ts: unknown; rust: unknown }>;
	required_mismatches: Array<{ param: string; ts: boolean; rust: boolean }>;
}

function props(schema: Record<string, unknown>): Record<string, Record<string, unknown>> {
	return (schema.properties ?? {}) as Record<string, Record<string, unknown>>;
}

function requiredSet(schema: Record<string, unknown>): Set<string> {
	const req = schema.required;
	return new Set(Array.isArray(req) ? (req as string[]) : []);
}

function compare(ts: ToolDef, rust: ToolDef): ParamDiff {
	const tsProps = props(ts.parameters);
	const rustProps = props(rust.parameters);
	const tsReq = requiredSet(ts.parameters);
	const rustReq = requiredSet(rust.parameters);
	const diff: ParamDiff = {
		tool: ts.name,
		missing_in_rust: [],
		extra_in_rust: [],
		type_mismatches: [],
		required_mismatches: [],
	};
	for (const key of Object.keys(tsProps)) {
		if (!(key in rustProps)) {
			diff.missing_in_rust.push(key);
			continue;
		}
		const tt = tsProps[key].type;
		const rt = rustProps[key].type;
		if (JSON.stringify(tt) !== JSON.stringify(rt)) diff.type_mismatches.push({ param: key, ts: tt, rust: rt });
		if (tsReq.has(key) !== rustReq.has(key)) {
			diff.required_mismatches.push({ param: key, ts: tsReq.has(key), rust: rustReq.has(key) });
		}
	}
	for (const key of Object.keys(rustProps)) {
		if (!(key in tsProps)) diff.extra_in_rust.push(key);
	}
	return diff;
}

const tsTools = JSON.parse(readFileSync(TS_JSON, "utf8")) as ToolDef[];
const rustTools = parseRustTools();

const tsNames = new Set(tsTools.map(t => t.name));
const rustNames = new Set(rustTools.map(t => t.name));
const missingInRust = [...tsNames].filter(n => !rustNames.has(n));
const extraInRust = [...rustNames].filter(n => !tsNames.has(n));

console.log("=== tool name diff ===");
console.log(`missing in omp-rs: ${missingInRust.join(", ") || "(none)"}`);
console.log(`extra in omp-rs:   ${extraInRust.join(", ") || "(none)"}`);
console.log("\n=== per-tool parameter diff ===");
let clean = true;
for (const ts of tsTools) {
	const rust = rustTools.find(t => t.name === ts.name);
	if (!rust) continue;
	const d = compare(ts, rust);
	const descMatch = ts.description_first_line === rust.description_first_line;
	const issues: string[] = [];
	if (STRICT_TOOLS.has(ts.name)) {
		if (JSON.stringify(canonical(ts.parameters)) !== JSON.stringify(canonical(rust.parameters))) issues.push("parameters differ from TS (canonical JSON)");
		if (ts.description !== rust.description) issues.push("full description differs from TS");
	}
	if (d.missing_in_rust.length) issues.push(`missing in rust: ${d.missing_in_rust.join(", ")}`);
	if (d.extra_in_rust.length) issues.push(`extra in rust: ${d.extra_in_rust.join(", ")}`);
	for (const m of d.type_mismatches) issues.push(`type ${m.param}: ts=${JSON.stringify(m.ts)} rust=${JSON.stringify(m.rust)}`);
	for (const m of d.required_mismatches) issues.push(`required ${m.param}: ts=${m.ts} rust=${m.rust}`);
	if (!descMatch) issues.push(`description first line differs:\n  ts:   ${ts.description_first_line}\n  rust: ${rust.description_first_line}`);
	if (issues.length) {
		clean = false;
		console.log(`[${d.tool}]\n  - ${issues.join("\n  - ")}`);
	} else {
		console.log(`[${d.tool}] OK`);
	}
}
console.log(clean ? "\nAll compared tools match." : "\nDifferences found (see above).");
