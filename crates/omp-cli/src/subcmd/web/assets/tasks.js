"use strict";
// Background task center: one panel listing every agent with live status,
// elapsed time, token/cost stats and open/stop actions. Wired via the public
// `omp:agents` / `omp:frame` events; no polling of its own beyond those.
window.OMP = window.OMP || {};
(function (OMP) {
  const { el, $, api } = OMP;
  const STAT_TTL = 20000;      // cache window for get_session_stats results
  const STAT_TIMEOUT = 4000;   // per-agent stats websocket deadline
  const STAT_CONCURRENCY = 2;

  let app = null;
  let modal = null;            // open OMP.dialogs.modal handle
  let listEl = null;
  let filterEl = null;
  let stateEl = null;
  let btn = null;
  let badge = null;
  let tick = null;

  /** Roster snapshot from /api/agents (kept fresh by the shell), plus
   *  locally derived live state for the currently connected agent. */
  let agents = [];             // last known roster
  const live = new Map();      // id -> {streaming,runningTools,needsInput,tool,dead}
  const stats = new Map();     // id -> {status:"pending"|"ok"|"empty"|"fail", data?, at}
  let statQueue = [];          // ids waiting for stats
  let statActive = 0;
  let notified = new Map();    // id -> {busy, needsInput, alive} for transition detection

  const basename = (p) => String(p || "").replace(/\/+$/, "").split("/").pop() || p;

  function view(a) {
    const l = live.get(a.id) || {};
    return {
      ...a,
      streaming: l.streaming ?? a.streaming,
      runningTools: l.runningTools ?? a.runningTools,
      needsInput: l.needsInput ?? a.needsInput,
      alive: l.dead ? false : a.alive !== false,
      tool: l.tool || null,
    };
  }

  function statusOf(a) {
    if (a.alive === false) return { cls: "dead", label: "stopped" };
    if (a.needsInput) return { cls: "wait", label: "needs input" };
    if (a.runningTools > 0) return { cls: "busy", label: `running tool${a.runningTools > 1 ? "s" : ""} (${a.runningTools})` };
    if (a.streaming) return { cls: "busy", label: "streaming" };
    if (a.awaitingTasksSinceMs) return { cls: "wait", label: "waiting on subagents" };
    return { cls: "done", label: "waiting" };
  }

  const isRunning = (a) => !!(a.streaming || a.runningTools > 0);

  // ── stats over a short-lived agent websocket ───────────────────────────────
  function fetchStats(id) {
    const hit = stats.get(id);
    if (hit && (hit.status === "pending" || (hit.status === "ok" && Date.now() - hit.at < STAT_TTL))) return;
    stats.set(id, { status: "pending", at: Date.now() });
    statQueue.push(id);
    pumpStats();
  }

  function pumpStats() {
    while (statActive < STAT_CONCURRENCY && statQueue.length) {
      const id = statQueue.shift();
      statActive++;
      statsOnce(id).catch(() => { }).finally(() => {
        statActive--;
        if (isOpen()) renderRows();
        pumpStats();
      });
    }
  }

  function statsOnce(id) {
    return new Promise((resolve) => {
      let ws;
      const done = (result) => {
        clearTimeout(timer);
        try { if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) ws.close(); } catch { /* noop */ }
        resolve(result);
      };
      const timer = setTimeout(() => {
        stats.set(id, { status: "fail", at: Date.now() });
        done(false);
      }, STAT_TIMEOUT);
      try {
        ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/${id}`);
      } catch {
        clearTimeout(timer);
        stats.set(id, { status: "fail", at: Date.now() });
        return done(false);
      }
      ws.onopen = () => { try { ws.send(JSON.stringify({ id: "tc-stats", type: "get_session_stats" })); } catch { done(false); } };
      ws.onmessage = (e) => {
        let f;
        try { f = JSON.parse(e.data); } catch { return; }
        if (f.type === "response" && f.id === "tc-stats") {
          const data = f.success === false ? null : (f.data || null);
          stats.set(id, data && (data.tokens || data.cost != null)
            ? { status: "ok", data, at: Date.now() }
            : { status: "empty", at: Date.now() });
          done(true);
        }
      };
      ws.onerror = () => { stats.set(id, { status: "fail", at: Date.now() }); done(false); };
      ws.onclose = () => done(false);
    });
  }

  function statsText(id) {
    const hit = stats.get(id);
    if (!hit || hit.status === "fail") return "";
    if (hit.status === "pending") return "…";
    if (hit.status !== "ok" || !hit.data) return "";
    const parts = [];
    const t = hit.data.tokens;
    if (t && (t.input || t.output)) {
      parts.push(`↑${OMP.fmtTokens(t.input + (t.cacheRead || 0) + (t.cacheWrite || 0))} ↓${OMP.fmtTokens(t.output)}`);
    }
    if (hit.data.cost) parts.push(OMP.fmtCost(hit.data.cost));
    return parts.join(" · ");
  }

  // ── notifications (only when the desktop bridge is not handling them) ──────
  const NOTIFY_PREF = "omp-tasks-notify";
  const notifyEnabled = () => localStorage.getItem(NOTIFY_PREF) !== "off";
  function setNotifyPref(on) {
    localStorage.setItem(NOTIFY_PREF, on ? "on" : "off");
    OMP.toast(on ? "Task notifications on" : "Task notifications off", "success");
  }

  function trackTransitions() {
    for (const raw of agents) {
      const a = view(raw);
      const prev = notified.get(a.id);
      notified.set(a.id, { busy: isRunning(a), needsInput: !!a.needsInput, alive: a.alive !== false });
      if (!prev) continue; // prime on first sight, no announcement
      if (prev.busy && a.needsInput) announce(a, "wait");
      else if (prev.busy && !isRunning(a) && a.alive !== false && !a.needsInput) announce(a, "done");
    }
  }

  function announce(a, kind) {
    // The desktop bridge already notifies (native or hidden-tab web) for the
    // same transitions; only cover setups where it is missing.
    if (OMP.desktop || !notifyEnabled()) return;
    if (!("Notification" in window) || document.hasFocus()) return;
    const name = a.sessionName || basename(a.cwd);
    const go = () => {
      try {
        new Notification(kind === "wait" ? "Task needs your input" : "Task finished",
          { body: name, tag: `omp-tasks:${a.id}:${kind}`, icon: "/favicon.svg" });
      } catch { /* noop */ }
    };
    if (Notification.permission === "granted") go();
    else if (Notification.permission !== "denied") Notification.requestPermission().then((p) => { if (p === "granted") go(); });
  }

  // ── rendering ───────────────────────────────────────────────────────────────
  function updateBadge() {
    if (!badge) return;
    const n = agents.map(view).filter((a) => a.alive !== false && isRunning(a)).length;
    badge.textContent = String(n);
    badge.hidden = n === 0;
    btn.title = n ? `Background tasks (${n} running)` : "Background tasks";
  }

  function rows() {
    const q = (filterEl && filterEl.value || "").trim().toLowerCase();
    let list = agents.map(view);
    if (q) list = list.filter((a) =>
      `${a.sessionName || ""} ${a.cwd || ""} ${a.model || ""} ${a.id}`.toLowerCase().includes(q));
    const weight = (a) => (a.alive === false ? 2 : isRunning(a) || a.needsInput ? 0 : 1);
    return list.sort((x, y) => weight(x) - weight(y) || (y.startedMs || 0) - (x.startedMs || 0));
  }

  function elapsed(ms) {
    return ms ? OMP.fmtDuration(Date.now() - ms) : "";
  }

  function renderRows() {
    if (!listEl) return;
    const list = rows();
    if (!list.length) {
      listEl.replaceChildren();
      stateEl.hidden = false;
      stateEl.replaceChildren(el("div.tasks-state", {
        text: agents.length ? "No agents match this filter." : "No background agents. Start one with “+ New session”.",
      }));
      return;
    }
    stateEl.hidden = true;
    const current = app.S.agentId;
    listEl.replaceChildren(...list.map((a) => {
      const st = statusOf(a);
      const title = a.sessionName || basename(a.cwd) || a.id;
      const sub = [OMP.shortPath(a.cwd, app.S.info.home), a.sessionFile ? basename(a.sessionFile) : "", a.model]
        .filter(Boolean).join("  ·  ");
      const statsTxt = statsText(a.id);
      return el("li.task-row" + (a.id === current ? ".active" : ""), null,
        el("span.task-dot." + st.cls, { title: st.label }),
        el("div.task-main", null,
          el("div.task-title-line", null,
            el("span.task-title", { text: title, title }),
            el("span.task-status." + st.cls, { text: st.label })),
          a.tool ? el("div.task-tool", { text: "⚙ " + a.tool, title: "Active tool: " + a.tool }) : null,
          el("div.task-sub", { text: sub, title: sub }),
          el("div.task-stats", { text: statsTxt, title: statsTxt || "token usage and cost (when reported by the agent)" })),
        el("div.task-actions", null,
          el("span.task-elapsed", {
            dataset: { elapsed: a.startedMs ? String(a.startedMs) : "" },
            text: elapsed(a.startedMs),
            title: a.startedMs ? "Started " + new Date(a.startedMs).toLocaleString() : "",
          }),
          el("button.icon-btn", {
            text: "Open", title: "Open this agent",
            onclick: () => { close(); location.hash = "#/a/" + a.id; },
          }),
          a.alive !== false
            ? el("button.icon-btn", { text: "Stop", title: "Stop this agent", onclick: () => stopAgent(a.id) })
            : null));
    }));
  }

  function renderFoot(done) {
    return [
      el("label", { style: "display:flex;gap:6px;align-items:center;color:var(--fg-muted);font-size:12.5px" },
        el("input", { type: "checkbox", checked: notifyEnabled(), onchange: (e) => setNotifyPref(e.target.checked) }),
        "Notify when a task needs input or finishes"),
      el("span", { style: "flex:1" }),
      el("button.secondary", { text: "Refresh", onclick: () => refresh() }),
      el("button.secondary", { text: "Close", onclick: () => done(null) }),
    ];
  }

  function isOpen() { return !!modal; }

  function open() {
    if (isOpen()) return;
    listEl = el("ul.tasks-list", { role: "list" });
    stateEl = el("div", { hidden: false }, el("div.tasks-state", { text: "Loading agents…" }));
    filterEl = el("input.filter", {
      type: "search", placeholder: "Filter agents…", "aria-label": "Filter agents",
      oninput: () => renderRows(),
    });
    const handle = OMP.dialogs.modal({
      title: "Background tasks",
      cls: "tasks-modal",
      body: [
        el("div.tasks-toolbar", null, filterEl),
        listEl, stateEl,
      ],
      foot: renderFoot,
    });
    modal = { done: handle.done, box: handle.box };
    // Keep refresh/stop working after the modal is dismissed.
    handle.promise.then(() => { modal = null; listEl = stateEl = filterEl = null; clearInterval(tick); tick = null; });
    if (btn) btn.setAttribute("aria-pressed", "true");
    refresh().then(() => { renderRows(); scheduleStats(); });
    clearInterval(tick);
    tick = setInterval(() => {
      if (!listEl) return;
      for (const n of listEl.querySelectorAll("[data-elapsed]")) {
        const ms = Number(n.dataset.elapsed);
        if (ms) n.textContent = elapsed(ms);
      }
    }, 1000);
  }

  function close() {
    if (modal) modal.done(null);
    modal = null; listEl = stateEl = filterEl = null;
    clearInterval(tick); tick = null;
    if (btn) btn.setAttribute("aria-pressed", "false");
  }

  function scheduleStats() {
    for (const a of agents.map(view)) {
      if (a.alive !== false) fetchStats(a.id);
    }
  }

  // ── actions ────────────────────────────────────────────────────────────────
  async function stopAgent(id) {
    const a = agents.find((x) => x.id === id);
    const name = a ? (a.sessionName || basename(a.cwd) || id) : id;
    const ok = await OMP.dialogs.confirmBox("Stop agent",
      `Stop “${name}”? The session stays on disk and can be resumed.`).promise;
    if (!ok) return;
    try {
      await api("/api/agents/" + encodeURIComponent(id), { method: "DELETE" });
      OMP.toast(`Stopped ${name}`, "success");
      live.set(id, { ...(live.get(id) || {}), dead: true, streaming: false, runningTools: 0 });
      await refresh();
    } catch (e) {
      OMP.toast("Could not stop agent: " + e.message, "error");
    }
  }

  /** Re-read the roster; also called by the topbar button and after stops. */
  async function refresh() {
    let res;
    try {
      res = await api("/api/agents");
    } catch (e) {
      if (isOpen()) {
        stateEl.hidden = false;
        stateEl.replaceChildren(el("div.tasks-state", { text: "Failed to load agents: " + e.message }));
      }
      return;
    }
    agents = res.agents || [];
    // Keep only live-state entries that still exist; stale agent -> drop.
    const ids = new Set(agents.map((a) => a.id));
    for (const id of [...live.keys()]) if (!ids.has(id)) live.delete(id);
    // Agents that disappeared came from the shell's own list too: sync it so
    // the sidebar reflects a stop immediately.
    app.S.agents = agents;
    window.dispatchEvent(new CustomEvent("omp:agents", { detail: { agents } }));
    trackTransitions();
    updateBadge();
    if (isOpen()) { renderRows(); scheduleStats(); }
  }

  // ── event wiring ───────────────────────────────────────────────────────────
  function onAgents(e) {
    agents = (e.detail && e.detail.agents) || [];
    trackTransitions();
    updateBadge();
    if (isOpen()) { renderRows(); scheduleStats(); }
  }

  function onFrame(e) {
    const d = e.detail || {};
    if (!d.agentId || !d.frame) return;
    const id = d.agentId;
    const f = d.frame;
    const l = live.get(id) || {};
    const next = { ...l };
    switch (f.type) {
      case "agent_start": next.streaming = true; next.dead = false; next.tool = null; break;
      case "agent_end": case "session_settled": case "agent_settled":
        next.streaming = false; next.runningTools = 0; next.tool = null;
        // Fresh totals right after a run finishes.
        stats.delete(id);
        break;
      case "tool_execution_start":
        next.runningTools = (next.runningTools || 0) + 1;
        next.tool = f.toolName || next.tool;
        break;
      case "tool_execution_end":
        next.runningTools = Math.max(0, (next.runningTools || 1) - 1);
        if (next.runningTools === 0) next.tool = null;
        break;
      case "web_agent_exit": next.dead = true; next.streaming = false; next.runningTools = 0; break;
      default: return; // unknown frame: nothing to do, avoid rerenders
    }
    live.set(id, next);
    updateBadge();
    trackTransitions();
    if (isOpen()) renderRows();
  }

  // ── init ───────────────────────────────────────────────────────────────────
  function init(a) {
    app = a;
    const host = $("#topbar .top-actions");
    btn = el("button.icon-btn.tasks-btn", {
      id: "tasks-btn",
      title: "Background tasks",
      "aria-label": "Background tasks",
      "aria-pressed": "false",
      onclick: () => (isOpen() ? close() : open()),
      html: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 6h16M4 12h10M4 18h7"/><circle cx="18" cy="16" r="3.5"/><path d="M18 14.5v1.5l1 1"/></svg>',
    });
    badge = el("span.tasks-badge", { text: "0", hidden: true });
    btn.append(badge);
    host.prepend(btn);
    window.addEventListener("omp:agents", onAgents);
    window.addEventListener("omp:frame", onFrame);
    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "k" && !$("#modal-root").childElementCount) {
        e.preventDefault();
        open();
      }
    });
    // Prime the roster/badge; the shell's own refresh keeps it fresh later.
    refresh();
  }

  OMP.tasks = { init, open, refresh };
})(window.OMP);
