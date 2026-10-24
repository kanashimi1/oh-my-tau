// `bun crates/omp-providers/examples/stream_compare_ts.ts [model_id] [reasoning]`
// Same prompt/model/options as stream_compare.rs, via installed @oh-my-pi/pi-ai.
// Key is read from agent.db read-only and never printed.
import { Database } from "bun:sqlite";
import { getBundledModel } from "@oh-my-pi/pi-catalog/src/models";
import { streamSimple } from "@oh-my-pi/pi-ai/src/stream";

const id = process.argv[2] ?? "glm-5.3-flash";
const reasoning = (process.argv[3] ?? "max") as any;
const db = new Database(`${process.env.HOME}/.omp/agent/agent.db`, { readonly: true });
const row = db.query("select data from auth_credentials where provider='zai' and credential_type='api_key' and disabled_cause is null limit 1").get() as any;
const apiKey = JSON.parse(row.data).key;
const model: any = { ...getBundledModel("zai" as any, id), baseUrl: "https://api.z.ai/api/coding/paas/v4" };
console.log(`model: ${model.provider}/${model.id} api=${model.api} reasoning=${reasoning}`);
const ctx: any = { messages: [{ role: "user", content: "Reply with the single word: ok", timestamp: Date.now() }] };
const seq: string[] = [];
let last: any;
for await (const ev of streamSimple(model, ctx, { apiKey, reasoning })) {
	const tag = ev.type.endsWith("_delta") ? `${ev.type}+` : ev.type;
	if (seq[seq.length - 1] !== tag || !tag.endsWith("+")) seq.push(tag);
	last = ev;
}
console.log(`events: ${seq.join(" ")}`);
const msg = last?.message ?? last?.error ?? {};
console.log(`blocks: ${(msg.content ?? []).map((b: any) => b.type).join(",")}`);
console.log(`stopReason: ${msg.stopReason ?? "?"}`);
const u = msg.usage ?? {};
console.log(`usage: ${["input", "output", "cacheRead", "cacheWrite", "totalTokens"].map(k => `${k}${(u[k] ?? 0) > 0 ? ">0" : "=0"}`).join(" ")}`);
console.log(`errorMessage: ${msg.errorMessage ? "present" : "none"}`);
