// TS compat oracle. Runs `resolveModelPolicy` for every (provider, model) in
//   (a) a COPY of ~/.omp/agent/models.db (model_cache rows; the real DB is never written), and
//   (b) the bundled generated catalog (packages/catalog/src/models.json),
// in two spec variants, and dumps the resolved policy as JSONL for the Rust oracle test.
//
// Run: bun scripts/compat-oracle/oracle.ts [out.jsonl]
//   variant "sparse": spec.compat = row.compatConfig, spec.thinking = undefined  (TS `buildModel` of a catalog row)
//   variant "full":   spec.compat = row.compat (complete record as overrides), spec.thinking = row.thinking (explicit-thinking path)
//
// The oh-my-pi sources come from `OMP_TS_SRC` (root of an oh-my-pi checkout) if set, else
// `<repo>/vendor-src/oh-my-pi`. Output goes to `OMP_COMPAT_ORACLE_DIR` if set, else
// `<repo>/target/compat-oracle/` (created). A missing `~/.omp/agent/models.db` is fine: only the
// bundled catalog and the synthetic matrix are dumped then.
import { Database } from "bun:sqlite";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "../..");
const TS_SRC = process.env.OMP_TS_SRC ?? join(ROOT, "vendor-src/oh-my-pi");
const outDir = process.env.OMP_COMPAT_ORACLE_DIR ?? join(ROOT, "target/compat-oracle");
mkdirSync(outDir, { recursive: true });
const out = process.argv[2] ?? join(outDir, "dump.jsonl");
const { resolveModelPolicy } = await import(join(TS_SRC, "packages/catalog/src/compat/resolve"));

// ---- copy of models.db (never touch the real file)
const tmp = mkdtempSync(join(tmpdir(), "compat-oracle-"));
const srcDb = join(homedir(), ".omp/agent/models.db");
for (const suffix of ["", "-wal", "-shm"]) {
	if (existsSync(srcDb + suffix)) copyFileSync(srcDb + suffix, join(tmp, `models.db${suffix}`));
}
const rows: { source: string; model: any }[] = [];
const dbPath = join(tmp, "models.db");
if (existsSync(dbPath)) {
	const db = new Database(dbPath, { readonly: true });
	for (const r of db.query("select provider_id, models from model_cache").all() as { provider_id: string; models: string }[]) {
		for (const m of JSON.parse(r.models)) rows.push({ source: `db:${r.provider_id}`, model: m });
	}
	db.close();
}
const bundled = JSON.parse(readFileSync(join(TS_SRC, "packages/catalog/src/models.json"), "utf8"));
for (const provider of Object.keys(bundled)) {
	for (const id of Object.keys(bundled[provider])) rows.push({ source: `bundled:${provider}`, model: bundled[provider][id] });
}

function specOf(m: any, variant: "sparse" | "full") {
	const spec: Record<string, unknown> = {
		id: m.id,
		provider: m.provider,
		api: m.api,
		baseUrl: m.baseUrl,
		reasoning: Boolean(m.reasoning),
	};
	if (m.providerType !== undefined) spec.providerType = m.providerType;
	if (m.__synthThinking !== undefined) spec.thinking = m.__synthThinking;
	if (variant === "sparse") {
		if (m.compatConfig !== undefined && m.compatConfig !== null) spec.compat = m.compatConfig;
	} else {
		if (m.compat !== undefined && m.compat !== null) spec.compat = m.compat;
		if (m.thinking !== undefined && m.thinking !== null) spec.thinking = m.thinking;
	}
	return spec;
}

// ---- synthetic matrix: custom providers / proxies / hosts the cached catalog does not exercise
{
	let seed = 0x5eed1234;
	const rnd = () => {
		seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
		return seed / 2 ** 32;
	};
	const pick = <T>(xs: readonly T[]): T => xs[Math.floor(rnd() * xs.length)];
	const providers = [
		"openai", "azure", "openrouter", "vercel-ai-gateway", "github-copilot", "anthropic", "deepseek", "cerebras", "zai",
		"zhipu-coding-plan", "kilo", "alibaba-coding-plan", "umans", "xiaomi", "xiaomi-token-plan-ams", "xai", "xai-oauth",
		"mistral", "together", "baseten", "groq", "minimax", "minimax-code", "qwen-portal", "nvidia", "venice", "moonshot",
		"kimi-code", "opencode-go", "opencode-zen", "zenmux", "cline-pass", "fireworks", "llama.cpp", "lm-studio", "vllm",
		"ollama", "litellm", "my-custom", "amazon-bedrock", "google", "google-vertex", "nanogpt", "huggingface", "synthetic",
	];
	const urls = [
		"", "https://api.openai.com/v1", "https://x.openai.azure.com/openai", "https://openrouter.ai/api/v1",
		"https://ai-gateway.vercel.sh/v1", "https://api.githubcopilot.com", "https://api.anthropic.com", "https://api.deepseek.com/v1",
		"https://api.cerebras.ai/v1", "https://api.z.ai/api/coding/paas/v4", "https://open.bigmodel.cn/api/paas/v4", "https://api.kilo.ai/api/gateway",
		"https://dashscope.aliyuncs.com/compatible-mode/v1", "https://api.xiaomimimo.com/v1", "https://api.x.ai/v1", "https://api.mistral.ai/v1",
		"https://api.together.xyz/v1", "https://inference.baseten.co/v1", "https://api.fireworks.ai/inference/v1", "https://api.groq.com/openai/v1",
		"https://api.minimax.io/v1", "https://portal.qwen.ai/v1", "https://integrate.api.nvidia.com/v1", "https://api.venice.ai/api/v1",
		"https://api.moonshot.ai/v1", "https://api.kimi.com/coding/v1", "https://opencode.ai/zen/go/v1", "https://zenmux.ai/api/anthropic",
		"http://localhost:1234/v1", "http://127.0.0.1:8080/v1", "http://192.168.1.20:11434/v1", "http://172.20.0.3/v1", "https://llm.corp.local/v1",
		"https://gateway.ai.cloudflare.com/v1/acct/gw/anthropic", "https://aiplatform.googleapis.com/v1/projects/p/publishers/anthropic/models",
		"https://bedrock-runtime.us-east-1.amazonaws.com/anthropic", "https://bedrock-mantle.us-east-1.api.aws/v1", "https://res.services.ai.azure.com/anthropic",
		"https://generativelanguage.googleapis.com/v1beta/openai", "https://chutes.ai/v1", "https://some-proxy.example.com/openai/v1",
	];
	const ids = [
		"gpt-5.6", "gpt-5.5", "gpt-5", "gpt-4o", "o3", "gpt-oss-120b", "claude-opus-4-8", "claude-opus-4-5", "claude-sonnet-5", "claude-sonnet-4-5",
		"claude-haiku-5-5", "claude-haiku-4-5", "claude-3-5-sonnet-20241022", "anthropic/claude-opus-4.7", "kimi-for-coding", "k3", "kimi-k2.7-code",
		"kimi-k2.6", "kimi-k2-thinking", "glm-5.2", "glm-5.3-flash", "glm-4.5", "glm-4.6v", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner",
		"deepseek/deepseek-v3.1-terminus:exacto", "qwen3-coder", "qwen3.8-plus", "Qwen3-32B", "qwen/qwen3-235b-a22b-thinking-2507", "mimo-v2.5-pro",
		"MiniMax-M2", "MiniMax-M3", "grok-4.5", "grok-4", "grok-3-mini", "grok-code-fast-1", "mistral-medium-2604", "mistral-large-latest",
		"gemini-3-pro", "gemini-2.5-flash", "gemma-4-31b", "llama-3.3-70b", "nova-pro-v1:0", "amazon.nova-2-lite-v1:0", "some-unknown-model",
		"foo-thinking", "bar-high", "baz-max", "gpt-5.6-luna-wm", "meta/llama-4-maverick", "unsloth/Qwen3.6-27B-GGUF",
	];
	const apis = [
		"openai-completions", "openai-completions", "openai-completions", "openrouter", "openai-responses", "azure-openai-responses",
		"openai-codex-responses", "anthropic-messages", "anthropic-messages", "bedrock-converse-stream", "google-generative-ai",
		"google-vertex", "google-gemini-cli", "devin-agent", "cursor-agent",
	];
	const compats: (Record<string, unknown> | undefined)[] = [
		undefined, undefined, undefined, undefined,
		{ supportsDeveloperRole: true },
		{ reasoningEffortMap: { medium: "high", minimal: "low" }, supportsReasoningEffort: true },
		{ extraBody: { foo: 1, thinking: { type: "enabled" } } },
		{ whenThinking: { extraBody: { x: 1 }, supportsReasoningEffort: false } },
		{ reasoningDisableMode: "none-effort", omitReasoningEffort: false },
		{ supportsStore: true, maxTokensField: "max_completion_tokens", streamIdleTimeoutMs: 1234 },
		{ unknownJunkKey: 1, thinkingFormat: "qwen" },
		{ streamFirstEventTimeoutMs: 5000, streamIdleTimeoutMs: 7000, supportsReasoningEffort: false },
	];
	for (let i = 0; i < 9000; i++) {
		const spec: Record<string, unknown> = {
			id: pick(ids),
			provider: pick(providers),
			api: pick(apis),
			baseUrl: pick(urls),
			reasoning: rnd() < 0.6,
		};
		if (rnd() < 0.1) spec.providerType = pick(providers);
		const compat = pick(compats);
		if (compat) spec.compat = compat;
		if (rnd() < 0.1) spec.thinking = { mode: "effort", efforts: ["low", "high"], defaultLevel: "high" };
		rows.push({ source: "synthetic", model: { ...spec, compatConfig: spec.compat, compat: undefined, thinking: undefined, __synthThinking: spec.thinking } });
	}
}

const seen = new Set<string>();
const lines: string[] = [];
let n = 0;
let errors = 0;
for (const { source, model } of rows) {
	for (const variant of ["sparse", "full"] as const) {
		const spec = specOf(model, variant);
		const specJson = JSON.stringify(spec);
		const key = `${variant}:${specJson}`;
		if (seen.has(key)) continue;
		seen.add(key);
		let result: unknown;
		let error: string | undefined;
		try {
			const p = resolveModelPolicy(spec as any);
			result = { identity: p.identity, compat: p.compat, thinking: p.thinking, catalog: p.catalog, request: p.request };
		} catch (e) {
			error = String((e as Error).message ?? e);
			errors++;
		}
		lines.push(JSON.stringify({ source, variant, spec, result, error }));
		n++;
	}
}
writeFileSync(out, `${lines.join("\n")}\n`);
console.log(`rows=${rows.length} cases=${n} errors=${errors} -> ${out}`);
