import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Config, Folder, digest, offload, recall } from "./rlm-clm";

const cfg = (over: Partial<Config> = {}): Config => ({
  offloadChars: 6000,
  foldChars: 4000,
  keepTurns: 2,
  store: mkdtempSync(join(tmpdir(), "rlm-clm-")),
  ...over,
});

const cargoLog = (n: number) =>
  Array.from({ length: n }, (_, i) =>
    i === 137 ? "error[E0308]: mismatched types" : i === 138 ? "  --> src/lib.rs:42:9" : `   Compiling crate-${i} v0.1.0`,
  ).join("\n");

describe("offload (RLM leg)", () => {
  test("large bash output becomes a digest with head, signal lines, tail and a recallable ref", () => {
    const c = cfg();
    const body = cargoLog(400);
    const r = offload(c, "bash", [{ type: "text", text: body }])!;
    expect(r.chars).toBe(body.length);
    expect(r.text.length).toBeLessThan(body.length / 2);
    expect(r.text).toContain(`[rlm ref=${r.ref}:`);
    expect(r.text).toContain("L138: error[E0308]: mismatched types");
    expect(r.text).toContain("L139:   --> src/lib.rs:42:9");
    expect(r.text).toContain("crate-399");
    const back = recall(c.store, r.ref, "138-139");
    expect(back.ok).toBe(true);
    expect(back.text).toContain("L138: error[E0308]");
    expect(recall(c.store, r.ref, undefined, "E0308", 1).text).toContain("L139:   --> src/lib.rs:42:9");
  });

  test("small outputs, read results, images and already-offloaded text stay untouched", () => {
    const c = cfg();
    expect(offload(c, "bash", [{ type: "text", text: "ok" }])).toBeUndefined();
    expect(offload(c, "read", [{ type: "text", text: cargoLog(400) }])).toBeUndefined();
    expect(offload(c, "bash", [{ type: "image", data: "x" }, { type: "text", text: cargoLog(400) }])).toBeUndefined();
    const once = offload(c, "bash", cargoLog(400))!;
    expect(offload(c, "bash", once.text + "x".repeat(7000))).toBeUndefined();
  });

  test("digest is deterministic (same bytes on every request)", () => {
    expect(digest(cargoLog(300), "0123456789ab", "bash")).toBe(digest(cargoLog(300), "0123456789ab", "bash"));
  });

  test("recall rejects bad refs and ranges", () => {
    const c = cfg();
    expect(recall(c.store, "../etc/passwd").ok).toBe(false);
    expect(recall(c.store, "0123456789ab").ok).toBe(false);
    const r = offload(c, "grep", cargoLog(400))!;
    expect(recall(c.store, r.ref, "x-y").ok).toBe(false);
    expect(recall(c.store, r.ref, undefined, "(").ok).toBe(false);
  });
});

type M = Record<string, unknown>;
function turn(i: number, size: number): M[] {
  return [
    { role: "assistant", content: [{ type: "thinking", thinking: "t".repeat(200) }, { type: "toolCall", id: `c${i}`, name: "bash", arguments: { command: `step ${i}` } }] },
    { role: "toolResult", toolCallId: `c${i}`, toolName: "bash", content: [{ type: "text", text: `${i}:`.padEnd(size, "z") }] },
  ];
}

describe("Folder (CLM leg)", () => {
  test("under budget nothing changes", () => {
    const f = new Folder(cfg({ foldChars: 1_000_000 }));
    const msgs = [{ role: "user", content: "task" }, ...turn(0, 2000), ...turn(1, 2000)];
    expect(f.apply(msgs)).toBeUndefined();
  });

  test("over budget folds old results to stubs, keeps the last turns, never touches user messages or input", () => {
    const events: unknown[] = [];
    const c = cfg();
    const f = new Folder(c, (e) => events.push(e));
    const msgs: M[] = [{ role: "user", content: "the task" }];
    for (let i = 0; i < 5; i++) msgs.push(...turn(i, 2000));
    const snapshot = JSON.stringify(msgs);
    const out = f.apply(msgs)!;
    expect(JSON.stringify(msgs)).toBe(snapshot);
    expect(out[0]).toEqual({ role: "user", content: "the task" });
    const texts = out.filter((m) => m.role === "toolResult").map((m) => (m.content as { text: string }[])[0].text);
    expect(texts.slice(0, 3).every((t) => t.startsWith("[clm: older bash result (step"))).toBe(true);
    expect(texts[3].startsWith("3:")).toBe(true);
    expect(texts[4].startsWith("4:")).toBe(true);
    const firstAssistant = out[1].content as { type: string }[];
    expect(firstAssistant.some((b) => b.type === "thinking")).toBe(false);
    expect(events.length).toBe(1);
    const ref = /ref:"([0-9a-f]{12})"/.exec(texts[0])![1];
    expect(recall(c.store, ref).text).toContain("0:zzz");
  });

  test("between folds the folded prefix is byte-stable and the wire only grows at the end", () => {
    const f = new Folder(cfg());
    const msgs: M[] = [{ role: "user", content: "task" }];
    for (let i = 0; i < 5; i++) msgs.push(...turn(i, 2000));
    const a = JSON.stringify(f.apply(msgs));
    msgs.push(...turn(5, 100));
    const b = JSON.stringify(f.apply(msgs));
    expect(b.startsWith(a.slice(0, -1))).toBe(true);
  });

  test("finished prompts fold first; the current prompt's work stays raw", () => {
    const f = new Folder(cfg({ foldChars: 6000, keepTurns: 1 }));
    const msgs: M[] = [{ role: "user", content: "ticket 1" }];
    for (let i = 0; i < 3; i++) msgs.push(...turn(i, 2000));
    msgs.push({ role: "user", content: "ticket 2" });
    for (let i = 3; i < 5; i++) msgs.push(...turn(i, 2000));
    const out = f.apply(msgs)!;
    const texts = out.filter((m) => m.role === "toolResult").map((m) => (m.content as { text: string }[])[0].text);
    expect(texts.slice(0, 3).every((t) => t.startsWith("[clm:"))).toBe(true);
    expect(texts.slice(3).every((t) => /^\d:z/.test(t))).toBe(true);
  });

  test("no fold when one cache break would not buy back a quarter of the budget", () => {
    const f = new Folder(cfg({ foldChars: 4000, keepTurns: 1 }));
    const msgs: M[] = [{ role: "user", content: "u".repeat(5000) }, ...turn(0, 500), ...turn(1, 300)];
    expect(f.apply(msgs)).toBeUndefined();
  });
});
