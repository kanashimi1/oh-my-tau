"use strict";
// Minimal DOM-shim test for the agent panel's subagent lifecycle:
// spawn (task call) -> tool_execution_end attaches handles, does NOT settle
// -> later <task-results> envelope (wait tool result / toolResult message)
// settles the child; working children with known ids get a Message action.
// Run: node test/agents.lifecycle.test.js
//
// `el` builds plain objects; the dock host records rendered roots so we can
// walk them and assert on visible rows/status/buttons.

const assert = require("assert");

let toasts = [];
function el(spec, props, ...children) {
  const parts = String(spec).split(".");
  return { tag: parts[0], class: parts.slice(1), props: props || {}, children: children.flat(Infinity).filter(Boolean) };
}
const hasClass = (n, cls) => Array.isArray(n.class) && n.class.includes(cls);
function walk(node, fn) {
  if (Array.isArray(node)) { for (const n of node) walk(n, fn); return; }
  if (!node || typeof node !== "object") return;
  fn(node);
  for (const c of node.children || []) walk(c, fn);
}
function collect(node, pred, out = []) {
  walk(node, (n) => { if (pred(n)) out.push(n); });
  return out;
}

function makeHost() {
  return {
    classList: { toggle() {}, add() {}, remove() {} },
    roots: [],
    replaceChildren(...nodes) { this.roots = nodes; },
  };
}

const listeners = [];
global.window = {
  addEventListener: (type, fn) => { if (type === "omp:frame") listeners.push(fn); },
  dispatchEvent: (e) => { for (const fn of listeners) fn(e); },
};
const OMP = {
  el,
  $: makeHost,
  rafThrottle: (fn) => () => fn(),
  toast: (message, type) => toasts.push({ message, type }),
  mascot: { mount() {} },
  dialogs: {
    promptText: () => ({ promise: Promise.resolve("  change course  ") }),
  },
};
global.localStorage = { getItem: () => null, setItem() {} };
global.OMP = OMP;
global.document = {};
window.OMP = OMP;

require("../agents.js");
const panel = OMP.agentPanel;


const frame = (type, extra) =>
  window.dispatchEvent({ type: "omp:frame", detail: { agentId: "a1", frame: { type, ...extra } } });

function render() {
  const dock = makeHost();
  panel.renderRight(dock);
  return dock;
}
function rows(dock) {
  return collect(dock.roots, (n) => n.tag === "div" && n.props.role === "listitem");
}
function statusOf(row) {
  return collect(row, (n) => n.tag === "span" && hasClass(n, "ap-status"))[0];
}

(async () => {
let rpcCalls = [];
panel.init({
  S: { agentId: "a1", agents: [], state: {} },
  rpc: async (type, params) => { rpcCalls.push([type, params]); return {}; },
});
// Let init's own async refreshState/render settle before driving frames.
await new Promise((r) => setImmediate(r));
  // 1. spawn: working children, no ids yet
  frame("tool_execution_start", {
    toolCallId: "t1", toolName: "task",
    args: { tasks: [{ task: "A" }, { task: "B", agent: "coder" }] },
  });
  let rs = rows(render());
  assert.strictEqual(rs.length, 2);
  assert.ok(rs.every((r) => collect(r, (n) => n.tag === "button").length === 0), "no Message action before ids attach");

  // 2. call returns: real handles attach; children must still be working
  frame("tool_execution_end", {
    toolCallId: "t1", toolName: "task", isError: false,
    result: {
      content: [{ type: "text", text: "Spawned 2 task children…" }],
      details: { handles: [{ kind: "agent", id: "agent-7", agent: "task" }, { kind: "agent", id: "agent-8", agent: "coder" }] },
    },
  });
  rs = rows(render());
  assert.strictEqual(rs.length, 2, "still both children, none dropped");
  assert.strictEqual(rs.filter((r) => collect(r, (n) => n.tag === "button" && /Message/.test(n.props["aria-label"] || "")).length).length, 2,
    "working children with known ids have a Message action");

  // 3. unrelated tool result settles nothing
  frame("tool_execution_end", { toolCallId: "t9", toolName: "bash", isError: false, result: { content: [{ type: "text", text: "ok" }] } });
  rs = rows(render());
  assert.ok(rs.every((r) => statusOf(r).props.title === "working"), "unrelated result leaves children working");

  // 4. wait tool result carries a <task-results> envelope (pretty JSON) for
  //    agent-7 only; agent-8 stays working
  frame("tool_execution_end", {
    toolCallId: "t2", toolName: "wait", isError: false,
    result: { content: [{ type: "text", text: "<task-results>\n{\n  \"agent\": \"task\",\n  \"id\": \"agent-7\",\n  \"status\": \"completed\",\n  \"preview\": \"done\"\n}\n" }] },
  });
  rs = rows(render());
  assert.strictEqual(statusOf(rs[0]).props.title, "done", "agent-7 settled completed");
  assert.strictEqual(statusOf(rs[1]).props.title, "working", "agent-8 still working");
  assert.strictEqual(collect(rs[0], (n) => n.tag === "button").length, 0, "settled child loses the Message action");

  // 5. auto-delivery rides a toolResult message frame: agent-8 failed
  frame("message_end", {
    message: { role: "toolResult", toolCallId: "t3", content: [{ type: "text", text: "<task-results>\n{\"id\": \"agent-8\", \"status\": \"failed (exit 1)\"}" }] },
  });
  rs = rows(render());
  assert.strictEqual(statusOf(rs[1]).props.title, "failed", "agent-8 settled failed via message frame");

  // 6. history reconstruction: handles attach, envelope settles from replay
  panel.ingestMessages([
    { role: "assistant", content: [{ type: "toolCall", id: "h1", name: "task", arguments: JSON.stringify({ task: "C" }) }] },
    { role: "toolResult", toolCallId: "h1", content: [{ type: "text", text: "x" }], details: { handles: [{ id: "agent-9", agent: "task" }] } },
    { role: "toolResult", toolCallId: "h2", content: [{ type: "text", text: "<task-results>\n{\"id\": \"agent-9\", \"status\": \"cancelled\"}" }] },
  ]);
  rs = rows(render());
  assert.strictEqual(rs.length, 1, "history: one child");
  assert.strictEqual(statusOf(rs[0]).props.title, "failed", "history: cancelled => failed, no done-on-end shortcut");

  // 7. steer a working child with a known id via the Message action
  rpcCalls = []; toasts = [];
  frame("tool_execution_start", { toolCallId: "t4", toolName: "task", args: { task: "D" } });
  frame("tool_execution_end", {
    toolCallId: "t4", toolName: "task", isError: false,
    result: { details: { handles: [{ id: "agent-10", agent: "task" }] } },
  });
  const btn = collect(render().roots, (n) => n.tag === "button" && n.props["aria-label"] === "Message universal")[0];
  assert.ok(btn, "Message button present for working child with id");
  await btn.props.onclick();
  assert.deepStrictEqual(rpcCalls, [["steer_subagent", { subagentId: "agent-10", message: "change course" }]],
    "rpc steer_subagent gets the trimmed message and the handle id");
  assert.ok(toasts.some((t) => t.type === "info" && /Message sent/.test(t.message)), "success toast");

  // 8. unknown-id envelope never settles anyone
  frame("tool_execution_end", {
    toolCallId: "t5", toolName: "wait", isError: false,
    result: { content: [{ type: "text", text: "<task-results>\n{\"id\": \"agent-nope\", \"status\": \"completed\"}" }] },
  });
  const stillWorking = rows(render()).filter((r) => statusOf(r).props.title === "working").length;
assert.strictEqual(stillWorking, 1, "unknown id ignored; the id'd child is still working");

  // 9. Failed launches settle as errors in both live frames and history replay.
  frame("tool_execution_start", { toolCallId: "t6", toolName: "task", args: { task: "launch fails" } });
  frame("tool_execution_end", { toolCallId: "t6", toolName: "task", isError: true, result: { content: [{ type: "text", text: "invalid call" }] } });
  rs = rows(render());
  assert.strictEqual(statusOf(rs[rs.length - 1]).props.title, "failed", "live failed launch does not throw or stay working");
  panel.ingestMessages([
    { role: "assistant", content: [{ type: "toolCall", id: "h-error", name: "task", arguments: JSON.stringify({ task: "history failure" }) }] },
    { role: "toolResult", toolCallId: "h-error", toolName: "task", isError: true, content: [{ type: "text", text: "validation failed" }] },
  ]);
  rs = rows(render());
  assert.strictEqual(statusOf(rs[rs.length - 1]).props.title, "failed", "history failed launch replays without finishTask ReferenceError");

  (async () => {
let rpcCalls = [];
panel.init({
  S: { agentId: "a1", agents: [], state: {} },
  rpc: async (type, params) => { rpcCalls.push([type, params]); return {}; },
});
// Let init's own async refreshState/render settle before driving frames.
await new Promise((r) => setImmediate(r));
    await Promise.resolve();
    console.log("agents.lifecycle.test.js: all assertions passed");
  })().catch((e) => { console.error(e); process.exit(1); });
})().catch((e) => { console.error(e); process.exit(1); });
