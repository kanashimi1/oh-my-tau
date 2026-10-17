/**
 * TS-readability check for omp-rs-written models.db rows.
 *
 * Reads a provider's `model_cache` row through the *real* TS reader
 * (`catalog/src/model-cache.ts readModelCache`), which enforces
 * CACHE_SCHEMA_VERSION, the exact `materialization_policy` string (app
 * version + builder version + rules hash) and `isMaterializedModel` on every
 * payload row — the same gates a live TS omp applies on startup.
 *
 * Run (from the repo root): bun scripts/verify-models-db.ts <models.db> [provider]
 */
import { readModelCache } from "../vendor-src/oh-my-pi/packages/catalog/src/model-cache";

const [dbPath, provider = "mockprov"] = process.argv.slice(2);
if (!dbPath) {
	console.error("usage: bun verify-models-db.ts <models.db> [provider]");
	process.exit(2);
}
const ttlMs = 2 * 60 * 60 * 1000;
const entry = readModelCache(provider, ttlMs, () => Date.now(), dbPath);
if (!entry) {
	console.error(`REJECTED: TS readModelCache('${provider}') returned null — policy/schema/payload gate failed`);
	process.exit(1);
}
console.log(
	`TS readModelCache OK: provider=${provider} models=${entry.models.length} fresh=${entry.fresh} authoritative=${entry.authoritative}`,
);
console.log(`  ids: ${entry.models.map(m => m.id).join(", ")}`);
for (const model of entry.models) {
	console.log(`  ${model.id}: api=${model.api} ctx=${model.contextWindow} max=${model.maxTokens} reasoning=${model.reasoning}`);
}
