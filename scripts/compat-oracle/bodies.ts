// TS request-body oracle: builds each (model, context, options) case through the real pi-ai
// `streamOpenAICompletions` request builder (captured via `onPayload`, aborted before any I/O) and
// dumps {spec, context, options, body} JSONL for the Rust parity test.
// Run: bun scripts/compat-oracle/bodies.ts [out.jsonl] [out-transport.jsonl]
//
// The oh-my-pi sources come from `OMP_TS_SRC` (root of an oh-my-pi checkout) if set, else
// `<repo>/vendor-src/oh-my-pi`. Output goes to `OMP_COMPAT_ORACLE_DIR` if set, else
// `<repo>/target/compat-oracle/` (created).
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "../..");
const TS_SRC = process.env.OMP_TS_SRC ?? join(ROOT, "vendor-src/oh-my-pi");
const outDir = process.env.OMP_COMPAT_ORACLE_DIR ?? join(ROOT, "target/compat-oracle");
mkdirSync(outDir, { recursive: true });
const out = process.argv[2] ?? join(outDir, "bodies.jsonl");
const { buildModel } = await import(join(TS_SRC, "packages/catalog/src/build"));
const { streamSimple } = await import(join(TS_SRC, "packages/ai/src/stream"));
const { streamOpenAICompletions } = await import(join(TS_SRC, "packages/ai/src/providers/openai-completions"));
const bundled = JSON.parse(readFileSync(join(TS_SRC, "packages/catalog/src/models.json"), "utf8"));

// (provider, id, api override) — the providers the acceptance matrix names plus dialect neighbours.
const TARGETS: [string, string, string?][] = [
	["openai", "gpt-5.5", "openai-completions"],
	["openai", "codex-mini-latest", "openai-completions"],
	["kimi-code", "kimi-for-coding"],
	["kimi-code", "k3"],
	["moonshot", "kimi-k2.6"],
	["zai", "glm-5.1", "openai-completions"],
	["deepseek", "deepseek-v4-pro"],
	["deepseek", "deepseek-v4-flash"],
	["openrouter", "anthropic/claude-sonnet-5"],
	["openrouter", "deepseek/deepseek-v4-pro"],
	["openrouter", "moonshotai/kimi-k3"],
	["opencode-go", "deepseek-v4-pro"],
	["opencode-go", "kimi-k3"],
	["opencode-go", "glm-5.2"],
	["xiaomi", "mimo-v2.5"],
	["groq", "qwen/qwen3-32b"],
	["together", "deepseek-ai/DeepSeek-V3.1"],
	["cerebras", "gpt-oss-120b"],
	["mistral", "mistral-small-latest"],
	["fireworks", "deepseek-v4-pro"],
	["kilo", "~anthropic/claude-sonnet-latest"],
	["zenmux", "anthropic/claude-3.5-sonnet", "openai-completions"],
	["zenmux", "moonshotai/kimi-k2.6"],
	["alibaba-coding-plan", "glm-5"],
	["github-copilot", "gemini-3-pro-preview", "openai-completions"],
	["huggingface", "deepseek-ai/DeepSeek-R1"],
	["xiaomi", "mimo-v2-pro"],
	["kimi-code", "k3-256k"],
	["deepseek", "deepseek-flash"],
	["cline-pass", "cline-free/mimo-v2.6-flash"],
	["nvidia", "deepseek-ai/deepseek-v3.1"],
	["venice", "qwen3-235b-a22b-thinking-2507"],
	["minimax-code", "MiniMax-M2.5", "openai-completions"],
];

// models.yml-style custom specs (no catalog row): local backends, proxies, sparse compat overrides.
const CUSTOM: Record<string, unknown>[] = [
	{ id: "qwen3-coder-30b", name: "Qwen3 Coder", provider: "llama.cpp", api: "openai-completions", baseUrl: "http://127.0.0.1:8080/v1", reasoning: true },
	{ id: "mimo9b", name: "MiMo 9B", provider: "local-mimo", api: "openai-completions", baseUrl: "http://127.0.0.1:11434/v1", reasoning: false },
	{ id: "Atria-Dawn-Preview", name: "Atria", provider: "atria", api: "openai-completions", baseUrl: "https://api.atria-asi.ai/v1", reasoning: true, compat: { streamFirstEventTimeoutMs: 300000, streamIdleTimeoutMs: 600000 } },
	{ id: "gpt-5.5", name: "GPT", provider: "my-proxy", api: "openai-completions", baseUrl: "https://llm.corp.example/v1", reasoning: true, compat: { supportsDeveloperRole: true, supportsStore: true } },
	{ id: "deepseek-v4-pro", name: "DSv4", provider: "my-proxy", api: "openai-completions", baseUrl: "https://llm.corp.example/v1", reasoning: true, compat: { reasoningEffortMap: { high: "max" }, extraBody: { foo: "bar" } } },
	{ id: "kimi-for-coding", name: "Kimi", provider: "kimi-code", api: "openai-completions", baseUrl: "https://api.kimi.com/coding/v1", reasoning: true },
	{ id: "some-unknown-model", name: "Unknown", provider: "zai", api: "openai-completions", baseUrl: "https://api.z.ai/api/coding/paas/v4", reasoning: false },
	{ id: "vision-model", name: "Vision", provider: "my-proxy", api: "openai-completions", baseUrl: "https://llm.corp.example/v1", reasoning: false, compat: { stripImageInput: true } },
];

const tool = {
	name: "read",
	description: "Read a file",
	parameters: { type: "object", properties: { path: { type: "string", description: "File path" } }, required: ["path"] },
};
const richTool = {
	name: "edit",
	description: "Edit a file",
	parameters: {
		type: "object",
		properties: {
			path: { type: "string", description: "File path" },
			mode: { type: "string", enum: ["replace", "insert"], description: "Edit mode" },
			line: { type: "integer", description: "1-based line", default: 1 },
			ranges: { type: "array", items: { type: "object", properties: { from: { type: "number" }, to: { type: "number" } }, required: ["from"] } },
			note: { anyOf: [{ type: "string" }, { type: "null" }], description: "Optional note" },
		},
		required: ["path", "mode"],
	},
};
const image = { type: "image", data: "AAAA", mimeType: "image/png" };
const ts = 1_700_000_000_000;
const user = (text: string) => ({ role: "user", content: text, timestamp: ts });
const developer = (content: unknown, attribution?: string) => ({ role: "developer", content, timestamp: ts, ...(attribution ? { attribution } : {}) });
const usage = { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
function assistant(m: any) {
	return {
		role: "assistant",
		api: m.api,
		provider: m.provider,
		model: m.id,
		content: [
			{ type: "thinking", thinking: "I should read the file.", thinkingSignature: "reasoning_content" },
			{ type: "text", text: "Let me look." },
			{ type: "toolCall", id: "call_abc123", name: "read", arguments: { path: "a.txt" } },
		],
		usage,
		stopReason: "toolUse",
		timestamp: ts,
	};
}
const toolResult = {
	role: "toolResult",
	toolCallId: "call_abc123",
	toolName: "read",
	content: [{ type: "text", text: "file contents" }],
	isError: false,
	timestamp: ts,
};

function contexts(m: any) {
	return {
		rich: { systemPrompt: "You are a helpful assistant.", messages: [user("edit a.txt")], tools: [richTool, tool] },
		images: {
			systemPrompt: "sys",
			messages: [
				{ role: "user", content: [{ type: "text", text: "look" }, image], timestamp: ts },
				assistant(m),
				{ role: "toolResult", toolCallId: "call_abc123", toolName: "read", content: [{ type: "text", text: "shot" }, image], isError: false, timestamp: ts },
				user("ok"),
			],
			tools: [tool],
		},
		thinkOnly: {
			systemPrompt: "sys",
			messages: [
				user("hi"),
				{ role: "assistant", api: m.api, provider: m.provider, model: m.id, content: [{ type: "thinking", thinking: "hmm", thinkingSignature: "reasoning_content" }, { type: "text", text: "answer" }], usage, stopReason: "stop", timestamp: ts },
				user("again"),
			],
		},
		noSystem: { messages: [user("hello")] },
		injected: {
			systemPrompt: "You are a helpful assistant.",
			messages: [developer("ext context: build is green"), user("hello"), developer("ext note", "agent")],
			tools: [tool],
		},
		injectedHistory: {
			systemPrompt: "You are a helpful assistant.",
			messages: [
				user("read a.txt"),
				developer([{ type: "text", text: "injected parts" }]),
				assistant(m),
				toolResult,
				developer("after tool result"),
				user("thanks, summarize"),
				developer([{ type: "text", text: "look" }, image]),
			],
			tools: [tool],
		},
		injectedTail: {
			systemPrompt: "sys",
			messages: [user("hi"), developer("tail developer")],
		},
		plain: { systemPrompt: "You are a helpful assistant.", messages: [user("hello")] },
		tools: { systemPrompt: "You are a helpful assistant.", messages: [user("read a.txt")], tools: [tool] },
		history: {
			systemPrompt: "You are a helpful assistant.",
			messages: [user("read a.txt"), assistant(m), toolResult, user("thanks, summarize")],
			tools: [tool],
		},
	};
}

const OPTIONS: Record<string, Record<string, unknown>> = {
	default: {},
	"high+max": { reasoning: "high", maxTokens: 4096 },
	"medium+temp": { reasoning: "medium", temperature: 0.2 },
	off: { disableReasoning: true },
	session: { sessionId: "sess-1234", reasoning: "low" },
	forced: { reasoning: "high", toolChoice: { type: "function", function: { name: "read" } } },
};

function specOf(m: any, api?: string) {
	const spec: Record<string, unknown> = {
		id: m.id,
		name: m.name,
		provider: m.provider,
		api: api ?? m.api,
		baseUrl: m.baseUrl,
		reasoning: Boolean(m.reasoning),
		input: m.input,
		cost: m.cost,
		contextWindow: m.contextWindow,
		maxTokens: m.maxTokens,
	};
	if (m.providerType !== undefined) spec.providerType = m.providerType;
	if (m.compatConfig !== undefined) spec.compat = m.compatConfig;
	if (m.requestModelId !== undefined) spec.requestModelId = m.requestModelId;
	return spec;
}

function abortedSignal(): AbortSignal {
	const c = new AbortController();
	c.abort();
	return c.signal;
}

const lines: string[] = [];
let n = 0;
let skipped = 0;
const work: { label: string; spec: Record<string, unknown> }[] = [];
for (const [provider, id, api] of TARGETS) {
	const row = bundled[provider]?.[id];
	if (!row) {
		console.error(`missing bundled model ${provider}/${id}`);
		skipped++;
		continue;
	}
	work.push({ label: `${provider}/${id}`, spec: specOf(row, api) });
}
for (const c of CUSTOM) {
	work.push({ label: `custom:${c.provider}/${c.id}`, spec: { input: ["text", "image"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 16384, ...c } });
}
for (const { label, spec } of process.env.T_ONLY ? [] : work) {
	const model = buildModel(spec as any) as any;
	for (const [cname, context] of Object.entries(contexts(model))) {
		for (const [oname, options] of Object.entries(OPTIONS)) {
			let body: unknown;
			let error: string | undefined;
			try {
				body = await new Promise<unknown>((resolve, reject) => {
					streamOpenAICompletions(model, context as any, {
						apiKey: "test-key",
						signal: abortedSignal(),
						onPayload: (payload: unknown) => resolve(JSON.parse(JSON.stringify(payload))),
						...options,
					} as any);
					setTimeout(() => reject(new Error("no payload")), 2000);
				});
			} catch (e) {
				error = String((e as Error).message ?? e);
			}
			lines.push(JSON.stringify({ case: `${label}/${cname}/${oname}`, spec, model: { reasoning: model.reasoning, thinking: model.thinking }, context, options, body, error }));
			n++;
		}
	}
}
if (!process.env.T_ONLY) writeFileSync(out, `${lines.join("\n")}\n`);
console.log(`cases=${n} skipped=${skipped} -> ${out}`);

// ---- transport bodies: the public `streamSimple` facade (kimi routing, openrouter Responses switch,
// per-api option mapping) captured via `onPayload`. -> bodies-transport.jsonl
// The facade mints a random session id per inference when the caller sets none; the Rust test
// masks UUID-shaped values for those cases.
const outT = process.argv[3] ?? join(outDir, "bodies-transport.jsonl");
// (provider, id, env overrides, apiKey)
const T_TARGETS: [string, string, Record<string, string>?, string?][] = [
	...["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1", "claude-3-5-sonnet-20241022", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-opus-5-5", "claude-sonnet-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-mythos-5", "claude-fable-5"].map((id) => ["anthropic", id] as [string, string]),
	["anthropic", "claude-sonnet-4-5", undefined, "sk-ant-oat01-test"],
	["anthropic", "claude-opus-4-6", undefined, "sk-ant-oat01-test"],
	["anthropic", "claude-opus-5-5", undefined, "sk-ant-oat01-test"],
	["minimax", "MiniMax-M2"],
	["zai", "glm-4.6"],
	["opencode-zen", "claude-sonnet-4-5"],
	["github-copilot", "claude-sonnet-4.5"],
	["vercel-ai-gateway", "anthropic/claude-sonnet-4.5"],
	["cloudflare-ai-gateway", "claude-sonnet-4-5", { CLOUDFLARE_ACCOUNT_ID: "acct", CLOUDFLARE_GATEWAY_ID: "gw" }],
	["google-vertex", "claude-sonnet-4-6@default"],
	["umans", "umans-coder"],
	["kimi-code", "kimi-for-coding"],
	["kimi-code", "k3"],
	["kimi-code", "k3-256k"],
	["kimi-code", "kimi-for-coding-highspeed"],
	["kimi-code", "kimi-k2"],
	["kimi-code", "kimi-k2.5"],
	["openrouter", "anthropic/claude-sonnet-4.5"],
	["openrouter", "anthropic/claude-opus-4.5"],
	["openrouter", "openai/gpt-5"],
	["openrouter", "openai/gpt-5.5"],
	["openrouter", "deepseek/deepseek-v4-pro"],
	["openrouter", "z-ai/glm-5.1"],
	["openrouter", "google/gemini-3-pro-preview"],
	["openrouter", "moonshotai/kimi-k2.6"],
	["openrouter", "x-ai/grok-4"],
	["openrouter", "~anthropic/claude-sonnet-latest"],
	["openrouter", "anthropic/claude-sonnet-4.5", { PI_OPENROUTER_RESPONSES: "0" }],
	["openrouter", "openai/gpt-5", { PI_OPENROUTER_RESPONSES: "0" }],
	["openrouter", "deepseek/deepseek-v4-pro", { PI_OPENROUTER_RESPONSES: "0" }],
	["openrouter", "openai/gpt-5", { PI_OPENROUTER_RESPONSES: "1" }],
	...["gpt-5.5", "gpt-5", "gpt-5-codex", "gpt-5.1-codex-max", "gpt-5.2", "gpt-5.4-mini", "gpt-5.6", "gpt-4.1", "gpt-4o", "gpt-5-pro", "gpt-5-mini", "gpt-5.3-codex-spark"].map((id) => ["openai", id] as [string, string]),
	["xai", "grok-4"],
	["github-copilot", "gpt-5"],
	["opencode-zen", "gpt-5"],
	...["gpt-5.5", "gpt-5", "gpt-4.1", "gpt-5-codex"].map((id) => ["azure", id, { AZURE_OPENAI_BASE_URL: "https://myres.openai.azure.com/openai/v1" }] as [string, string, Record<string, string>]),
	["azure", "gpt-5", { AZURE_OPENAI_RESOURCE_NAME: "myres", AZURE_OPENAI_DEPLOYMENT_NAME_MAP: "gpt-5=my-gpt5-deploy" }],
];
const T_CUSTOM: Record<string, unknown>[] = [
	{ id: "claude-sonnet-4-5", name: "proxy", provider: "my-anthropic-proxy", api: "anthropic-messages", baseUrl: "https://llm.corp.example/anthropic", reasoning: true },
	{ id: "claude-sonnet-4-5", name: "proxy-compat", provider: "my-anthropic-proxy", api: "anthropic-messages", baseUrl: "https://llm.corp.example/anthropic", reasoning: true, compat: { supportsEagerToolInputStreaming: true, supportsLongCacheRetention: true } },
	{ id: "mimo9b", name: "MiMo", provider: "local-mimo", api: "anthropic-messages", baseUrl: "http://127.0.0.1:11434", reasoning: false },
	{ id: "gpt-5.5", name: "GPT proxy", provider: "my-proxy", api: "openai-responses", baseUrl: "https://llm.corp.example/v1", reasoning: true },
	{ id: "gpt-5.5", name: "GPT proxy", provider: "my-proxy", api: "openai-responses", baseUrl: "https://llm.corp.example/v1", reasoning: true, compat: { supportsStore: false } },
	{ id: "qwen3-coder", name: "local resp", provider: "llama.cpp", api: "openai-responses", baseUrl: "http://127.0.0.1:8080/v1", reasoning: true },
	{ id: "my-deploy", name: "az", provider: "azure", api: "azure-openai-responses", baseUrl: "https://myres.openai.azure.com/openai/v1", reasoning: true },
	{ id: "kimi-for-coding", name: "Kimi", provider: "kimi-code", api: "openai-completions", baseUrl: "https://api.kimi.com/coding/v1", reasoning: true },
	{ id: "or-custom", name: "or", provider: "openrouter", api: "openrouter", baseUrl: "https://openrouter.ai/api/v1", reasoning: true },
	{ id: "kimi-openai", name: "Kimi (openai format)", provider: "kimi-code", api: "openai-completions", baseUrl: "https://api.kimi.com/coding/v1", reasoning: true, compat: { kimiApiFormat: "openai" } },
];
const tLines: string[] = [];
let tN = 0;
let tSkipped = 0;
const tWork: { label: string; spec: Record<string, unknown>; env: Record<string, string>; apiKey: string }[] = [];
for (const [provider, id, env, apiKey] of T_TARGETS) {
	const row = bundled[provider]?.[id];
	if (!row) {
		console.error(`transport: missing bundled model ${provider}/${id}`);
		tSkipped++;
		continue;
	}
	const envTag = env ? `[${Object.entries(env).map(([k, v]) => `${k}=${v}`).join(",")}]` : "";
	const keyTag = apiKey ? "[oauth]" : "";
	tWork.push({ label: `${provider}/${id}${envTag}${keyTag}`, spec: specOf(row), env: env ?? {}, apiKey: apiKey ?? "test-key" });
}
for (const c of T_CUSTOM) {
	tWork.push({ label: `custom:${c.provider}/${c.id}/${c.api}${c.compat ? "+compat" : ""}`, spec: { input: ["text", "image"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 16384, ...c }, env: {}, apiKey: "test-key" });
}
for (const { label, spec, env, apiKey } of tWork) {
	const model = buildModel(spec as any) as any;
	for (const [k, v] of Object.entries(env)) process.env[k] = v;
	for (const [cname, context] of Object.entries(contexts(model))) {
		for (const [oname, options] of Object.entries(OPTIONS)) {
			let body: unknown;
			let payloadApi: unknown;
			let error: string | undefined;
			try {
				let streamError: string | undefined;
				// Abort from inside `onPayload` (after capture) so image preparation and header resolution
				// run un-raced; nothing is sent.
				const ctl = new AbortController();
				const events = streamSimple(model, context as any, {
					apiKey,
					signal: ctl.signal,
					onPayload: (payload: unknown, m: any) => {
						payloadApi = m?.api;
						body = JSON.parse(JSON.stringify(payload));
						ctl.abort();
					},
					...options,
				} as any);
				for await (const e of events as any) if (e.type === "error") streamError = e.error?.errorMessage;
				if (body === undefined) error = streamError ?? "no payload";
			} catch (e) {
				error = String((e as Error).message ?? e);
			}
			tLines.push(JSON.stringify({ case: `${label}/${cname}/${oname}`, spec, env, apiKey, payloadApi, model: { reasoning: model.reasoning, thinking: model.thinking }, context, options, body, error }));
			tN++;
		}
	}
	for (const k of Object.keys(env)) delete process.env[k];
}
writeFileSync(outT, `${tLines.join("\n")}\n`);
console.log(`transport cases=${tN} skipped=${tSkipped} -> ${outT}`);
