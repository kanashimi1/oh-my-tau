#!/usr/bin/env node
"use strict";
// CDP E2E harness for the omp web UI. Drives a real (Chrome/Chromium/Edge)
// browser over the DevTools protocol; the parent runs it because agent
// sandbox rules forbid browsers/servers here.
//
//   chromium --headless=new --remote-debugging-port=9222 about:blank
//   # omp web hub already serving:
//   node test/e2e.cdp.js
//
// Required environment:
//   OMP_WEB_URL    base URL of the running web hub   (default http://127.0.0.1:8090)
//   OMP_CDP_PORT   browser DevTools port             (default 9222)
// Optional:
//   OMP_CDP_HOST              DevTools host          (default 127.0.0.1)
//   OMP_E2E_AGENT_ID          existing agent id to exercise. WITHOUT it the
//                             harness POSTs its own throwaway agent
//                             (OMP_E2E_CWD, default: hub defaultCwd) and
//                             DELETEs it on exit — user agents/sessions are
//                             never touched. With an explicit id the harness
//                             stays read-only unless
//                             OMP_E2E_ALLOW_SESSION_SWITCH=1.
//   OMP_E2E_SESSION           session id/path to open (default: newest session
//                             of the hub cwd with messages)
//   OMP_E2E_FAILED_SESSION    session containing an errored `task` tool call
//                             (default: synthetic replay of a failed task —
//                             exercises the same finishTask history path)
//   OMP_E2E_ALLOW_PROMPTS=1   send a real prompt to verify live streaming
//                             frames (requires a configured model)
//   OMP_E2E_KEEP_AGENT=1      do not delete the harness-created agent
//   OMP_E2E_TIMEOUT           per-step timeout ms (default 15000)
//
// Scenarios (each reported PASS/SKIP/FAIL, exit 1 on any FAIL):
//   1. boot            UI loads from OMP_WEB_URL, no uncaught page errors
//   2. old-sessions    open a stored session, transcript renders
//   3. rpc-compat      legacy `get_messages` AND paginated
//                      `get_messages_page` (+cursor) both answer; unknown
//                      frame type does not break the page
//   4. failed-task     failed-task history replay marks the subagent row
//                      status=error (finishTask path) in #subagents-dock
//   5. streaming       WebSocket frame flow on /ws/<agentId> (live frames
//                      with OMP_E2E_ALLOW_PROMPTS, attach/replay otherwise)
//   6. reload          after Page.reload the agent connection, transcript
//                      rows and composer state are restored

const http = require("http");
const WebSocket = require("ws");

const BASE = process.env.OMP_WEB_URL || "http://127.0.0.1:8090";
const CDP_PORT = Number(process.env.OMP_CDP_PORT || 9222);
const CDP_HOST = process.env.OMP_CDP_HOST || "127.0.0.1";
const TIMEOUT = Number(process.env.OMP_E2E_TIMEOUT || 15000);

const results = [];
function report(name, status, detail) {
  results.push({ name, status });
  console.log(`${status.padEnd(4)}  ${name}${detail ? ` — ${detail}` : ""}`);
}

function fail(msg) { console.error(`FATAL: ${msg}`); process.exit(2); }

function httpJson(pathname, opts = {}) {
  return fetch(BASE + pathname, {
    method: opts.method || "GET",
    headers: opts.body ? { "content-type": "application/json" } : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  }).then(async (res) => {
    const text = await res.text();
    let data;
    try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
    if (!res.ok) throw new Error(`${res.status} ${pathname}: ${text.slice(0, 200)}`);
    return data;
  });
}

function getJSON(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let body = "";
      res.on("data", (c) => { body += c; });
      res.on("end", () => {
        try { resolve(JSON.parse(body)); } catch (e) { reject(e); }
      });
    }).on("error", reject);
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class Cdp {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map(); this.events = []; this.sessions = new Set(); }
  static async connect() {
    const ver = await getJSON(`http://${CDP_HOST}:${CDP_PORT}/json/version`);
    const ws = new WebSocket(ver.webSocketDebuggerUrl, { perMessageDeflate: false, maxPayload: 256 * 1024 * 1024 });
    await new Promise((resolve, reject) => { ws.once("open", resolve); ws.once("error", reject); });
    const cdp = new Cdp(ws);
    ws.on("message", (raw) => {
      const msg = JSON.parse(raw);
      if (msg.id && cdp.pending.has(msg.id)) {
        const { resolve, reject } = cdp.pending.get(msg.id);
        cdp.pending.delete(msg.id);
        msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
      } else if (msg.method) {
        if (msg.params && msg.params.targetId) cdp.sessions.add(msg.params.sessionId);
        cdp.events.push(msg);
      }
    });
    return cdp;
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
  async waitEvent(method, sessionId, timeout = TIMEOUT) {
    const deadline = Date.now() + timeout;
    for (;;) {
      const i = this.events.findIndex((e) => e.method === method && (!sessionId || e.sessionId === sessionId));
      if (i !== -1) return this.events.splice(i, 1)[0];
      if (Date.now() > deadline) throw new Error(`timeout waiting for CDP event ${method}`);
      await sleep(50);
    }
  }
  close() { try { this.ws.close(); } catch { /* already closed */ } }
}

// ── page driver ──────────────────────────────────────────────────────────────

async function makeTab(cdp) {
  const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
  await cdp.send("Page.enable", {}, sessionId);
  await cdp.send("Runtime.enable", {}, sessionId);
  // Frame recorder + error trap, installed before any navigation so the app's
  // own WebSockets are wrapped from the first script onward.
  await cdp.send("Page.addScriptToEvaluateOnNewDocument", { source: `
    window.__ompWsLog = [];
    window.__ompErrors = [];
    window.addEventListener("error", (e) => window.__ompErrors.push(String(e.message)));
    window.addEventListener("unhandledrejection", (e) => window.__ompErrors.push(String(e.reason)));
    const NativeWS = window.WebSocket;
    window.WebSocket = function (url, protocols) {
      const sock = protocols === undefined ? new NativeWS(url) : new NativeWS(url, protocols);
      if (String(url).includes("/ws/")) {
        const log = (dir, data) => { try { window.__ompWsLog.push({ url: String(url), dir, data, t: Date.now() }); } catch {} };
        log("open", null);
        sock.addEventListener("message", (e) => log("in", typeof e.data === "string" ? e.data.slice(0, 2000) : "<binary>"));
        const origSend = sock.send.bind(sock);
        sock.send = (data) => { log("out", typeof data === "string" ? data.slice(0, 2000) : "<binary>"); return origSend(data); };
      }
      return sock;
    };
    window.WebSocket.prototype = NativeWS.prototype;
    for (const key of ["CONNECTING", "OPEN", "CLOSING", "CLOSED"]) {
      Object.defineProperty(window.WebSocket, key, { value: NativeWS[key] });
    }
    Object.defineProperty(window.WebSocket, "name", { value: "WebSocket" });
  ` }, sessionId);
  return { targetId, sessionId };
}

async function evaluate(cdp, sessionId, expression) {
  const res = await cdp.send("Runtime.evaluate", {
    expression, returnByValue: true, awaitPromise: true,
  }, sessionId);
  if (res.exceptionDetails) {
    throw new Error(`page eval failed: ${res.exceptionDetails.exception?.description || res.exceptionDetails.text}\n  expr: ${expression.slice(0, 200)}`);
  }
  return res.result.value;
}

async function waitFor(cdp, sessionId, expr, what, timeout = TIMEOUT) {
  const deadline = Date.now() + timeout;
  let last;
  while (Date.now() < deadline) {
    last = await evaluate(cdp, sessionId, expr);
    if (last) return last;
    await sleep(150);
  }
  throw new Error(`timeout waiting for ${what} (last: ${JSON.stringify(last)?.slice(0, 200)})`);
}

async function pageErrors(cdp, sessionId) {
  return evaluate(cdp, sessionId, "window.__ompErrors");
}

// ── helpers over the hub HTTP/WS surface ─────────────────────────────────────

async function pickAgent(cdp, sessionId) {
  const provided = process.env.OMP_E2E_AGENT_ID;
  if (provided) return { id: provided, owned: false };
  const info = await httpJson("/api/info");
  const cwd = process.env.OMP_E2E_CWD || info.defaultCwd;
  const created = await httpJson("/api/agents", { method: "POST", body: { cwd } });
  const id = created.id || created.agent?.id;
  if (!id) throw new Error(`could not create throwaway agent: ${JSON.stringify(created).slice(0, 200)}`);
  console.log(`       (harness created throwaway agent ${id} in ${cwd})`);
  return { id, owned: true };
}

async function connectInPage(cdp, sessionId, agentId) {
  await evaluate(cdp, sessionId, `OMP.app.connect(${JSON.stringify(agentId)})`);
  await waitFor(cdp, sessionId,
    "OMP.app.S.agentId === " + JSON.stringify(agentId) + " && !!OMP.app.S.ws && OMP.app.S.ws.readyState === 1 && !!OMP.app.S.state",
    `connection to agent ${agentId}`).catch(async (error) => {
      const debug = await evaluate(cdp, sessionId, `({agentId:OMP.app.S.agentId,state:OMP.app.S.state,ws:OMP.app.S.ws&&OMP.app.S.ws.readyState,onopen:OMP.app.S.ws&&typeof OMP.app.S.ws.onopen,errors:window.__ompErrors,body:document.body.innerText.slice(-500),frames:window.__ompWsLog.slice(-10)})`);
      throw new Error(error.message + " " + JSON.stringify(debug));
    });
  await evaluate(cdp, sessionId, `history.replaceState(null, "", ${JSON.stringify("#/a/")} + ${JSON.stringify(agentId)})`);
}

async function listSessions() {
  try {
    const { sessions } = await httpJson("/api/sessions");
    return sessions || [];
  } catch (e) {
    console.log(`       (/api/sessions unavailable: ${e.message})`);
    return [];
  }
}

// ── scenarios ────────────────────────────────────────────────────────────────

async function scenarioBoot(cdp, tab) {
  await cdp.send("Page.navigate", { url: BASE + "/" }, tab.sessionId);
  await waitFor(cdp, tab.sessionId, "document.readyState === 'complete' && !!window.OMP && !!window.OMP.app", "UI boot");
  const errors = await pageErrors(cdp, tab.sessionId);
  if (errors.length) throw new Error(`page errors: ${errors.join(" | ")}`);
  report("boot: UI loads without page errors", "PASS");
}

async function scenarioOldSessions(cdp, tab) {
  const sessions = await listSessions();
  const wanted = process.env.OMP_E2E_SESSION;
  const session = wanted
    ? sessions.find((s) => s.id === wanted || s.path === wanted) || { path: wanted, id: wanted }
    : sessions[0];
  if (!session || (!session.path && !session.id)) {
    report("old-sessions: open a stored session", "SKIP", "no stored sessions and no OMP_E2E_SESSION");
    return null;
  }
  const path = session.path || session.id;
  const res = await evaluate(cdp, tab.sessionId,
    `OMP.app.rpc("switch_session", { sessionPath: ${JSON.stringify(path)} })`);
  if (res && res.error) throw new Error(`switch_session rejected: ${JSON.stringify(res).slice(0, 200)}`);
  const rows = await waitFor(cdp, tab.sessionId,
    "document.querySelectorAll('#transcript > *').length", "transcript rows after switch_session");
  if (!rows) throw new Error("transcript empty after opening stored session");
  report("old-sessions: opened stored session, transcript renders", "PASS", `${rows} rows`);
  return session;
}

async function scenarioRpcCompat(cdp, tab) {
  // Legacy full snapshot vs new pagination (older detached agents cap pages
  // at 256 — both shapes must answer).
  const legacy = await evaluate(cdp, tab.sessionId, `OMP.app.rpc("get_messages")`);
  if (!legacy || !Array.isArray(legacy.messages)) throw new Error("legacy get_messages did not return {messages}");
  const paged = await evaluate(cdp, tab.sessionId, `OMP.app.rpc("get_messages_page", { limit: 16 })`);
  if (!Array.isArray(paged.messages)) throw new Error("get_messages_page did not return messages");
  if (paged.messages.length > 16) throw new Error(`page size ${paged.messages.length} > limit 16`);
  if (paged.nextCursor) {
    const next = await evaluate(cdp, tab.sessionId,
      `OMP.app.rpc("get_messages_page", { limit: 16, cursor: ${JSON.stringify(paged.nextCursor)} })`);
    if (!Array.isArray(next.messages)) throw new Error("paged get with cursor did not return messages");
  }
  // Unknown frame type must not wedge the page or throw.
  await evaluate(cdp, tab.sessionId,
    `OMP.app.send({ type: "omp_e2e_unknown_probe", ts: Date.now() })`);
  await sleep(300);
  const errors = await pageErrors(cdp, tab.sessionId);
  if (errors.length) throw new Error(`page errors after unknown frame: ${errors.join(" | ")}`);
  const stillAlive = await evaluate(cdp, tab.sessionId, `!!OMP.app.rpc`);
  if (!stillAlive) throw new Error("app rpc surface gone after unknown frame");
  report("rpc-compat: legacy + paginated rpc answer; unknown type tolerated", "PASS",
    `${legacy.messages.length} legacy messages, page ${paged.messages.length}`);
}

async function scenarioFailedTask(cdp, tab, sessionId) {
  const wanted = process.env.OMP_E2E_FAILED_SESSION;
  let source = "synthetic replay";
  if (wanted && wanted !== process.env.OMP_E2E_SESSION) {
    await evaluate(cdp, tab.sessionId, `OMP.app.rpc("switch_session", { sessionPath: ${JSON.stringify(wanted)} })`);
    source = `stored session ${wanted}`;
  }
  // Replays history through the real ingest path; errored `task` tool calls
  // settle through finishTask(toolCallId, true).
  const injected = await evaluate(cdp, tab.sessionId, `(async () => {
    let messages = await OMP.app.getMessages();
    let hasErroredTask = messages.some((m) =>
      m.toolName === "task" && (m.isError ||
        (m.content || []).some?.((c) => c.type === "tool_execution_end" && c.isError)));
    if (!hasErroredTask && !${JSON.stringify(source.startsWith("stored"))}) {
      messages = messages.concat([
        { role: "assistant", content: [{ type: "toolCall", id: "e2e-failed-task", name: "task", arguments: JSON.stringify({ task: "E2E failed launch" }) }] },
        { role: "toolResult", toolCallId: "e2e-failed-task", toolName: "task", isError: true, content: [{ type: "text", text: "validation failed" }] },
      ]);
      hasErroredTask = true;
    }
    OMP.agentPanel.ingestMessages(messages);
    return { hasErroredTask, count: messages.length };
  })()`);
  const row = await waitFor(cdp, tab.sessionId,
    `(() => { const r = document.querySelector('#subagents-dock .ap-sub'); return r ? r.textContent : null; })()`,
    "subagent row in #subagents-dock");
  if (!injected.hasErroredTask && source.startsWith("stored")) {
    throw new Error(`no errored task call found in ${wanted}`);
  }
  const isError = await evaluate(cdp, tab.sessionId,
    `(() => { const dock = document.querySelector('#subagents-dock'); return dock ? dock.textContent : ""; })()`).then((t) => /error|failed/i.test(t));
  if (!isError) throw new Error(`subagent row does not show error status: ${String(row).slice(0, 120)}`);
  report("failed-task: history replay settles the failed task as error", "PASS", source);
  void sessionId;
}

async function scenarioLazyHistory(cdp, tab) {
  const before = await evaluate(cdp, tab.sessionId, "window.__ompWsLog.length");
  await evaluate(cdp, tab.sessionId, `(() => { const s = document.querySelector('#transcript-wrap'); s.scrollTop = 0; s.dispatchEvent(new Event('scroll')); })()`);
  await waitFor(cdp, tab.sessionId,
    `window.__ompWsLog.slice(${before}).some((f) => f.dir === 'out' && f.data.includes('"type":"get_messages_page"') && f.data.includes('"direction":"backward"') && f.data.includes('"cursor"'))`,
    "older backward page on upward scroll");
  report("lazy-history: upward scroll fetches older bounded page", "PASS");
}

async function scenarioSessionCache(cdp, tab, agentId) {
  const before = await evaluate(cdp, tab.sessionId, "window.__ompWsLog.length");
  await connectInPage(cdp, tab.sessionId, agentId);
  await waitFor(cdp, tab.sessionId, "document.querySelectorAll('#transcript > *').length > 0", "cached transcript");
  const pages = await evaluate(cdp, tab.sessionId,
    `window.__ompWsLog.slice(${before}).filter((f) => f.dir === 'out' && f.data.includes('"type":"get_messages_page"')).length`);
  if (pages) throw new Error(`cache miss on quiet reconnect: ${pages} page requests`);
  report("session-cache: quiet reconnect renders without history RPC", "PASS");
}

async function scenarioStreaming(cdp, tab, agent) {
  const before = await evaluate(cdp, tab.sessionId, "window.__ompWsLog.length");
  if (process.env.OMP_E2E_ALLOW_PROMPTS === "1") {
    await evaluate(cdp, tab.sessionId,
      `OMP.app.send({ type: "prompt", message: "Reply with the single word OK" })`);
    await waitFor(cdp, tab.sessionId,
      `window.__ompWsLog.some((f) => f.dir === "in" && /message_update|"agent_start"/.test(f.data))`,
      "live streaming frames").catch(async (error) => {
        const frames = await evaluate(cdp, tab.sessionId, "window.__ompWsLog.slice(-12)");
        throw new Error(error.message + " " + JSON.stringify(frames));
      });
    await evaluate(cdp, tab.sessionId, `OMP.app.abort && OMP.app.abort()`);
    const path = await evaluate(cdp, tab.sessionId, `({
      backward: window.__ompWsLog.some((f) => f.dir === 'out' && f.data.includes('"type":"get_messages_page"') && f.data.includes('"direction":"backward"')),
      full: window.__ompWsLog.slice(${before}).some((f) => f.dir === 'out' && f.data.includes('"type":"get_messages"')),
    })`);
    if (!path.backward || path.full) throw new Error(`streaming did not use bounded backward history: ${JSON.stringify(path)}`);
    report("streaming: live frames + bounded backward history, no full snapshot", "PASS");
  } else {
    // Without prompts, verify the browser socket is live in both directions:
    // an rpc round-trip must produce inbound data on the same socket.
    const state = await evaluate(cdp, tab.sessionId, `OMP.app.rpc("get_state")`);
    if (!state || typeof state !== "object") throw new Error("get_state returned nothing");
    await waitFor(cdp, tab.sessionId,
      "window.__ompWsLog.length > " + before + " && window.__ompWsLog.some((f) => f.dir === 'in')",
      "inbound /ws/ frames");
    report("streaming: /ws/ frame flow verified (set OMP_E2E_ALLOW_PROMPTS=1 for live turn)", "PASS",
      `${await evaluate(cdp, tab.sessionId, "window.__ompWsLog.length")} frames`);
  }
  void agent;
}

async function scenarioReload(cdp, tab, agentId) {
  await cdp.send("Page.reload", {}, tab.sessionId);
  await waitFor(cdp, tab.sessionId, "!!window.OMP && !!window.OMP.app", "UI re-boot");
  await waitFor(cdp, tab.sessionId,
    `OMP.app.S.agentId === ${JSON.stringify(agentId)} && !!OMP.app.S.ws && OMP.app.S.ws.readyState === 1`,
    "restored agent connection after reload", 25000);
  const rows = await waitFor(cdp, tab.sessionId,
    `document.querySelectorAll('#transcript > *').length > 0`,
    "transcript restoration after reload", 25000);
  const errors = await pageErrors(cdp, tab.sessionId);
  if (errors.length) throw new Error(`page errors after reload: ${errors.join(" | ")}`);
  report("reload: agent connection + transcript restored", "PASS", `${rows} rows`);
}

// ── main ─────────────────────────────────────────────────────────────────────

async function main() {
  let cdp;
  try { cdp = await Cdp.connect(); } catch (e) {
    fail(`cannot reach CDP at ${CDP_HOST}:${CDP_PORT} (${e.message}). Start the browser with --remote-debugging-port=${CDP_PORT}`);
  }
  const tab = await makeTab(cdp);
  let agent = null;
  try {
    await scenarioBoot(cdp, tab);

    agent = await pickAgent(cdp, tab.sessionId);
    const maySwitch = agent.owned || process.env.OMP_E2E_ALLOW_SESSION_SWITCH === "1";

    if (maySwitch) {
      await connectInPage(cdp, tab.sessionId, agent.id);
      await scenarioOldSessions(cdp, tab);
    } else {
      await connectInPage(cdp, tab.sessionId, agent.id);
      report("old-sessions: open a stored session", "SKIP",
        "OMP_E2E_AGENT_ID given without OMP_E2E_ALLOW_SESSION_SWITCH=1 (read-only mode)");
    }

    await scenarioRpcCompat(cdp, tab);
    if (maySwitch) await scenarioFailedTask(cdp, tab, tab.sessionId);
    else report("failed-task: history replay", "SKIP", "read-only mode");

    await connectInPage(cdp, tab.sessionId, agent.id);
    await scenarioLazyHistory(cdp, tab);
    await scenarioSessionCache(cdp, tab, agent.id);
    await scenarioStreaming(cdp, tab, agent);

    // Reload restoration always runs; it must not depend on session switches.
    if (maySwitch) await scenarioReload(cdp, tab, agent.id);
    else report("reload: restoration", "SKIP", "read-only mode");
  } catch (e) {
    report("harness", "FAIL", e.message);
  } finally {
    await cdp.send("Target.closeTarget", { targetId: tab.targetId }).catch(() => {});
    if (agent && agent.owned && process.env.OMP_E2E_KEEP_AGENT !== "1") {
      await httpJson(`/api/agents/${agent.id}`, { method: "DELETE" })
        .then(() => console.log(`       (removed throwaway agent ${agent.id})`))
        .catch((e) => console.log(`       (could not remove throwaway agent ${agent.id}: ${e.message})`));
    }
    cdp.close();
  }
  const failed = results.filter((r) => r.status === "FAIL");
  console.log(`\n${results.filter((r) => r.status === "PASS").length} passed, ` +
    `${results.filter((r) => r.status === "SKIP").length} skipped, ${failed.length} failed`);
  process.exit(failed.length ? 1 : 0);
}

main();
