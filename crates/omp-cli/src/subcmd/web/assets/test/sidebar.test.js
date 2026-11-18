"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");

global.window = { OMP: {} };
require("../sidebar.js");
const { agentBrand, agentStatus, compareAgents } = window.OMP.sidebar;

test("sidebar resolves model brands through gateways and provider aliases", () => {
  for (const [model, key] of [
    ["anthropic/claude-opus-4-6", "claude"],
    ["openai-codex/gpt-5.3-codex", "openai"],
    ["google-antigravity/gemini-3-flash", "gemini"],
    ["kimi-code/k3.1-pro", "kimi"],
    ["zai-coding-plan/glm-5.3-flash", "zai"],
    ["openrouter/anthropic/claude-sonnet", "claude"],
    ["openrouter/openai/gpt-5", "openai"],
    ["openrouter/local-llm", "openrouter"],
    ["custom/local-llm", "unknown"],
    [null, "unknown"],
  ]) assert.equal(agentBrand(model).key, key, model);
  assert.notEqual(agentBrand("claude").paths, agentBrand("gpt-5").paths);
});

test("sidebar prioritizes stop and input, and ignores the stale tools flag", () => {
  assert.equal(agentStatus({ runningTools: true }), "idle");
  assert.equal(agentStatus({ awaitingTasksSinceMs: 0, runningTools: true }), "waiting");
  assert.equal(agentStatus({ awaitingTasksSinceMs: 0, streaming: true }), "working");
  assert.equal(agentStatus({ streaming: true, needsInput: true }), "needs-input");
  assert.equal(agentStatus({ alive: false, streaming: true, needsInput: true }), "stopped");
});

test("roster can group by workspace or prioritize agents that need attention", () => {
  const row = (key, cwd, status, ms) => ({ key, cwd, status, ms });
  const rows = [row("a", "/b", "idle", 9), row("b", "/a", "working", 5), row("c", "/b", "needs-input", 1), row("d", "/a", "idle", 7)];
  const order = (mode) => [...rows].sort((a, b) => compareAgents(a, b, mode)).map((r) => r.key);
  assert.deepEqual(order("grouped"), ["d", "b", "a", "c"]);
  assert.deepEqual(order("priority"), ["c", "b", "a", "d"]);
});
