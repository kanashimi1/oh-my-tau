"use strict";
// Shell: sidebar lists, header, status bar, pickers and session actions.
// `init(app)` (called by app.js) wires the DOM and boots routing.
(function (OMP) {
  const { el, $, api } = OMP;
  let app = null;
  const S = () => app.S;

  const shortPath = (p) => OMP.shortPath(p, S().info.home);
  const basename = (p) => String(p || "").replace(/\/+$/, "").split("/").pop() || p;

  function navigate(id) {
    const target = id ? "#/a/" + id : "#/";
    if (location.hash === target) route();
    else location.hash = target;
  }
  function route() {
    const m = location.hash.match(/^#\/a\/([\w-]+)/);
    const id = m ? m[1] : null;
    if (id === S().agentId && S().ws) return;
    app.connect(id);
  }

  // ── rendering ──────────────────────────────────────────────────────────────
  function renderHeader(s) {
    const st = s.state || {};
    const info = s.agents.find((a) => a.id === s.agentId) || s.agentInfo || {};
    const title = st.sessionName || info.sessionName || (s.agentId ? "New session" : "omt");
    $("#session-title").textContent = title;
    document.title = s.agentId ? `${title} — omt` : "omt";
    $("#session-cwd").textContent = info.cwd ? shortPath(info.cwd) : "";
    const model = st.model ? `${st.model.provider}/${st.model.id}` : "model";
    $("#model-btn").textContent = model;
    $("#model-btn").disabled = !s.agentId;
    const tb = $("#thinking-btn");
    if (st.ultra) {
      tb.replaceChildren(el("span", { text: "τ " }), el("span.ultra-text", { text: st.ultra, dataset: { mode: st.ultra } }));
    } else {
      tb.textContent = "◇ " + (st.thinkingLevel || "thinking");
    }
    tb.classList.toggle("ultra", !!st.ultra);
    $("#thinking-btn").disabled = !s.agentId;
    $("#thinking-btn").hidden = !!(st.model && !st.model.reasoning && !st.ultra);
  }

  function renderStatus(s) {
    const bar = $("#statusbar");
    if (!s.agentId) { bar.replaceChildren(); return; }
    const parts = [];
    const st = s.state || {};
    const stats = s.stats || {};
    if (s.dead) parts.push(el("span.warn", { text: "● stopped" }));
    else if (s.streaming) parts.push(el("span.busy", { text: `● working ${OMP.fmtDuration(Date.now() - s.runStart)}` }));
    else parts.push(el("span", { text: "● ready" }));
    const cu = stats.contextUsage;
    if (cu && cu.contextWindow) {
      const pct = Math.min(100, cu.percent != null ? cu.percent : (100 * cu.tokens) / cu.contextWindow);
      const color = pct > 85 ? "var(--err)" : pct > 60 ? "var(--warn)" : "var(--ok)";
      parts.push(el("span", { title: `${cu.tokens} / ${cu.contextWindow} tokens` },
        el("span.ctx-bar", null, el("i", { style: `width:${pct}%;background:${color}` })),
        `${OMP.fmtTokens(cu.tokens)}/${OMP.fmtTokens(cu.contextWindow)} (${pct.toFixed(pct < 10 ? 1 : 0)}%)`));
    }
    const t = stats.tokens;
    if (t && t.total) parts.push(el("span", { text: `↑${OMP.fmtTokens(t.input + (t.cacheRead || 0) + (t.cacheWrite || 0))} ↓${OMP.fmtTokens(t.output)}`, title: `input ${t.input} · output ${t.output} · cache read ${t.cacheRead} · cache write ${t.cacheWrite}` }));
    if (stats.cost) parts.push(el("span", { text: OMP.fmtCost(stats.cost) }));
    const q = (st.queuedMessageCount || 0);
    if (q) parts.push(el("span.warn", { text: `${q} queued` }));
    for (const [k, v] of s.status) parts.push(el("span", { text: v, title: k }));
    if (st.sessionId) parts.push(el("span.faint", { text: st.sessionId.slice(0, 8), title: st.sessionFile || st.sessionId }));
    bar.replaceChildren(...parts.flatMap((p, i) => (i ? [el("span.sep", { text: "│" }), p] : [p])));
  }

  function renderQueue(s) {
    const band = $("#queue-band");
    const items = [
      ...(s.queue.steering || []).map((m) => ["steer", m]),
      ...(s.queue.followUp || []).map((m) => ["follow-up", m]),
    ];
    band.hidden = !items.length;
    band.replaceChildren(...items.map(([k, m]) => el("div.q", { text: `${k}: ${typeof m === "string" ? m : OMP.contentText(m.content) || JSON.stringify(m)}` })));
  }

  const TODO_ICON = { completed: "✓", in_progress: "▸", inProgress: "▸", pending: "○", blocked: "⊘", abandoned: "✗" };
  function todoWidget(s) {
    const phases = (s.state && Array.isArray(s.state.todoPhases)) ? s.state.todoPhases : [];
    const tasks = phases.flatMap((p) => (p.tasks || []).map((t) => ({ ...t, phase: p.name })));
    if (!tasks.length) return null;
    const done = tasks.filter((t) => t.status === "completed" || t.status === "abandoned").length;
    if (done === tasks.length) return null;
    const lines = [];
    for (const p of phases) {
      if (!(p.tasks || []).length) continue;
      if (phases.length > 1) lines.push(p.name);
      for (const t of p.tasks) lines.push(`${phases.length > 1 ? "  " : ""}${TODO_ICON[t.status] || "○"} ${t.content}${t.blocker ? "  (blocked: " + t.blocker + ")" : ""}`);
    }
    return el("details.todo-hud", { open: true }, el("summary", { text: `Todos ${done}/${tasks.length}` }), el("div", { text: lines.join("\n") }));
  }

  function renderWidgets(s) {
    const above = [], below = [];
    const todo = todoWidget(s);
    if (todo) above.push(todo);
    for (const w of s.widgets.values()) (w.below ? below : above).push(el("div", { text: w.lines.join("\n") }));
    $("#widgets-above").replaceChildren(...above);
    $("#widgets-below").replaceChildren(...below);
  }

  function renderSidebar(s) {
    const agents = $("#agent-list");
    agents.replaceChildren(...(s.agents.length ? s.agents.map((a) => {
      const title = a.sessionName || basename(a.cwd);
      const dot = el("span.dot" + (a.streaming ? ".busy" : ".live"));
      return el("button.item" + (a.id === s.agentId ? ".active" : ""), { onclick: () => navigate(a.id), title: a.cwd },
        el("div.row", null, el("span.name", null, dot, " ", title),
          el("span.x", { text: "✕", title: "Stop agent", onclick: (e) => { e.stopPropagation(); stopAgent(s, a.id); } })),
        el("div.sub", { text: `${shortPath(a.cwd)}${a.model ? " · " + a.model : ""}` }));
    }) : [el("div.muted.small", { text: "No running agents" })]));

    $("#project-path").textContent = shortPath(s.project);
    $("#project-path").title = s.project;
    const q = $("#session-filter").value.trim().toLowerCase();
    const current = s.agents.find((a) => a.id === s.agentId);
    const openFiles = new Set(s.agents.map((a) => a.sessionFile).filter(Boolean));
    const rows = s.sessions.filter((r) => !q || `${r.title || ""} ${r.preview || ""} ${r.id}`.toLowerCase().includes(q));
    $("#session-list").replaceChildren(...(rows.length ? rows.slice(0, 300).map((r) => {
      const active = current && current.sessionFile === r.path;
      return el("button.item" + (active ? ".active" : ""), { onclick: () => openSession(s, r), title: r.path },
        el("div.row", null, el("span.name", { text: r.title || r.preview || r.id }),
          openFiles.has(r.path) ? el("span.dot.live", { title: "open" }) : r.store === "omp" ? el("span.badge", { text: "ts" }) : null),
        el("div.sub", { text: `${OMP.timeAgo(r.modifiedMs)}${r.title && r.preview ? " · " + r.preview : ""}` }));
    }) : [el("div.muted.small", { text: s.project ? "No sessions in this folder" : "" })]));
  }

  // ── data ─────────────────────────────────────────────────────────────────
  async function refreshAgents(s) {
    try {
      s.agents = (await api("/api/agents")).agents || [];
    } catch { return; }
    renderSidebar(s);
    renderHeader(s);
  }

  async function loadSessions(s) {
    if (!s.project) return;
    try {
      s.sessions = (await api("/api/sessions?cwd=" + encodeURIComponent(s.project))).sessions || [];
    } catch (e) {
      s.sessions = [];
      OMP.toast("Sessions: " + e.message, "error");
    }
    renderSidebar(s);
  }

  function setProject(s, cwd) {
    s.project = cwd;
    localStorage.setItem("omp-web-project", cwd);
    loadSessions(s);
  }

  async function spawn(body) {
    try {
      const a = await api("/api/agents", { method: "POST", body });
      await refreshAgents(S());
      navigate(a.id);
      return a;
    } catch (e) {
      OMP.toast("Could not start agent: " + e.message, "error", 8000);
      return null;
    }
  }

  function openSession(s, row) {
    const open = s.agents.find((a) => a.sessionFile === row.path);
    if (open) return navigate(open.id);
    return spawn({ cwd: row.cwd || s.project, resume: row.path });
  }

  async function newSession(s, cwd) {
    let projects = [];
    try { projects = (await api("/api/projects")).projects || []; } catch { /* none */ }
    const dirs = [cwd || s.project, s.project, s.info.defaultCwd, ...projects.filter((p) => p.exists).map((p) => p.cwd)]
      .filter((d, i, all) => d && all.indexOf(d) === i);
    const items = [...dirs.map((d) => ({ label: shortPath(d), description: d })), { label: "Browse…", description: "Pick another folder" }];
    const i = await OMP.dialogs.pick("New session — working directory", items, { searchable: true }).promise;
    if (i == null) return;
    let dir = dirs[i];
    if (i === dirs.length) dir = await OMP.dialogs.pickFolder(s.project || s.info.home).promise;
    if (!dir) return;
    setProject(s, dir);
    await spawn({ cwd: dir });
  }

  async function resumePicker(s, cwd) {
    let rows = [];
    try { rows = (await api("/api/sessions?cwd=" + encodeURIComponent(cwd))).sessions || []; } catch (e) { return OMP.toast(e.message, "error"); }
    if (!rows.length) return OMP.toast("No stored sessions for " + shortPath(cwd), "warning");
    const items = rows.map((r) => ({ label: r.title || r.preview || r.id, description: `${OMP.timeAgo(r.modifiedMs)} · ${r.id}${r.store === "omp" ? " · ts" : ""}` }));
    const i = await OMP.dialogs.pick("Resume session", items, { searchable: true, wide: true }).promise;
    if (i == null) return;
    const row = rows[i];
    if (s.agentId && !s.streaming && !s.dead) {
      try {
        await app.rpc("switch_session", { sessionPath: row.path });
        const msgs = await app.rpc("get_messages");
        app.transcript.loadMessages(msgs.messages);
        await refreshAgents(s);
        app.S.state = await app.rpc("get_state");
        renderHeader(s);
        return;
      } catch (e) {
        OMP.toast("switch_session failed (" + e.message + "); opening a new agent", "warning");
      }
    }
    openSession(s, { ...row, cwd });
  }

  async function stopAgent(s, id) {
    if (!id) return;
    const ok = await OMP.dialogs.confirmBox("Stop agent", "Stop this agent process? The session stays on disk and can be resumed.").promise;
    if (!ok) return;
    try { await api("/api/agents/" + id, { method: "DELETE" }); } catch (e) { OMP.toast(e.message, "error"); }
    if (id === s.agentId) { s.dead = { code: null }; navigate(null); }
    setTimeout(() => refreshAgents(s), 400);
    loadSessions(s);
  }

  // ── pickers / actions ──────────────────────────────────────────────────────
  async function modelPicker(s) {
    const picked = await OMP.models.browse(app, s.state && s.state.model);
    if (!picked) return;
    try {
      await app.rpc("set_model", { provider: picked.provider, modelId: picked.id });
      OMP.toast(`Model: ${picked.provider}/${picked.id}`, "success");
    } catch (e) { OMP.toast(e.message, "error"); }
    app.S.state = await app.rpc("get_state").catch(() => app.S.state);
    renderHeader(s);
    refreshAgents(s);
  }

  // omp-rs effort levels past `max`: workflow by default and its RLM/CLM hybrids.
  const ULTRA_MODES = [
    ["ultracode", "max thinking + workflow (eval workpool) on every substantial task"],
    ["ultradeep", "ultracode + RLM (llm_query, λ-RLM, SERA rubrics) for huge inputs"],
    ["ultralong", "ultracode + CLM context strategy for long sessions"],
    ["hypercode", "ultracode + RLM + CLM — everything at once"],
  ];

  async function thinkingPicker(s) {
    let levels;
    try { levels = (await app.rpc("get_available_thinking_levels")).levels || []; } catch (e) { return OMP.toast(e.message, "error"); }
    const st = s.state || {};
    const cur = st.ultra || st.thinkingLevel;
    const values = [...levels, ...ULTRA_MODES.map(([n]) => n)];
    const items = [
      ...levels.map((l) => ({ label: l, tag: l === cur ? "current" : undefined })),
      ...ULTRA_MODES.map(([n, d]) => ({ label: n, description: d, tag: n === cur ? "current" : "ultra", cls: "ultra-opt", mode: n })),
    ];
    if (st.ultra) { values.push("__ultra_off"); items.push({ label: "ultra off", description: "leave the ultra mode, keep the current effort" }); }
    const i = await OMP.dialogs.pick("Thinking level", items, { initial: Math.max(0, values.indexOf(cur)) }).promise;
    if (i == null) return;
    try {
      if (values[i] === "__ultra_off") await app.rpc("prompt", { message: "/ultra off" });
      else await app.rpc("set_thinking_level", { level: values[i] });
    } catch (e) { return OMP.toast(e.message, "error"); }
    app.S.state = await app.rpc("get_state").catch(() => app.S.state);
    renderHeader(s);
  }

  function exportHtml(s) {
    const file = s.state && s.state.sessionFile;
    if (!file) return OMP.toast("This session has no file yet", "warning");
    window.open("/api/export?session=" + encodeURIComponent(file), "_blank", "noopener");
  }

  async function rename(s) {
    if (!s.agentId) return;
    const cur = (s.state && s.state.sessionName) || "";
    const name = await OMP.dialogs.promptText("Rename session", { value: cur }).promise;
    if (!name || !name.trim()) return;
    try { await app.rpc("set_session_name", { name: name.trim() }); } catch (e) { return OMP.toast(e.message, "error"); }
    app.S.state = await app.rpc("get_state").catch(() => app.S.state);
    renderHeader(s);
    refreshAgents(s);
    loadSessions(s);
  }

  async function sessionInfo(s) {
    const st = await app.rpc("get_session_stats").catch((e) => ({ error: e.message }));
    const state = s.state || {};
    const lines = [
      ["Session", state.sessionId], ["File", state.sessionFile], ["Folder", (s.agentInfo || {}).cwd],
      ["Model", state.model ? `${state.model.provider}/${state.model.id}` : ""], ["Thinking", state.thinkingLevel],
      ["Messages", `${st.userMessages ?? "?"} user · ${st.assistantMessages ?? "?"} assistant · ${st.toolCalls ?? "?"} tool calls`],
      ["Tokens", st.tokens ? `in ${st.tokens.input} · out ${st.tokens.output} · cache r/w ${st.tokens.cacheRead}/${st.tokens.cacheWrite}` : ""],
      ["Cost", st.cost != null ? OMP.fmtCost(st.cost) : ""],
      ["Context", st.contextUsage ? `${st.contextUsage.tokens} / ${st.contextUsage.contextWindow}` : ""],
    ].filter(([, v]) => v);
    OMP.dialogs.modal({
      title: "Session info",
      wide: true,
      body: [el("div.system", { text: lines.map(([k, v]) => `${k.padEnd(9)} ${v}`).join("\n") })],
      foot: (done) => [el("button.secondary", { text: "Close", onclick: () => done(null) })],
    });
  }

  function showHotkeys() {
    const rows = [
      ["Enter", "send (steer while the agent works)"], ["Alt+Enter", "queue a follow-up while the agent works"],
      ["Shift+Enter", "new line"], ["Esc", "interrupt the running turn"], ["↑ / ↓", "prompt history"],
      ["/", "slash commands (Tab completes)"], ["!cmd", "run a shell command (!!cmd: keep it out of the context)"],
      ["Ctrl+Shift+M", "model picker"], ["Ctrl+Shift+T", "thinking level"], ["Ctrl+Shift+N", "new session"], ["Ctrl+Shift+R", "resume a session"],
      ["paste / drop", "attach images"],
    ];
    OMP.dialogs.modal({
      title: "Keyboard shortcuts",
      body: [el("div.system", { text: rows.map(([k, v]) => `${k.padEnd(14)} ${v}`).join("\n") })],
      foot: (done) => [el("button.secondary", { text: "Close", onclick: () => done(null) })],
    });
  }

  function sessionMenu(anchor, s) {
    const has = !!s.agentId && !s.dead;
    const items = [
      { label: "New session", run: () => newSession(s, app.currentCwd()) },
      { label: "Resume session…", run: () => resumePicker(s, app.currentCwd()) },
    ];
    if (has) {
      items.push("-",
        { label: "Rename…", run: () => rename(s) },
        { label: "Fork session", run: async () => {
          try { await app.rpc("fork"); OMP.toast("Forked into a new session file", "success"); } catch (e) { return OMP.toast(e.message, "error"); }
          app.S.state = await app.rpc("get_state").catch(() => app.S.state); renderHeader(s); refreshAgents(s); loadSessions(s);
        } },
        { label: "Compact context", run: async () => {
          app.transcript.addDivider("compacting…");
          try { await app.rpc("compact"); app.transcript.addDivider("context compacted"); } catch (e) { app.transcript.addSystem("Compaction failed: " + e.message, "error"); }
          app.S.stats = await app.rpc("get_session_stats").catch(() => app.S.stats); renderStatus(s);
        } },
        { label: "Copy last response", run: async () => {
          try { const r = await app.rpc("get_last_assistant_text"); const ok = await OMP.copyText(r.text || ""); OMP.toast(ok ? "Copied" : "Clipboard unavailable", ok ? "success" : "error"); } catch (e) { OMP.toast(e.message, "error"); }
        } },
        { label: "Export HTML", run: () => exportHtml(s) },
        { label: "Session info", run: () => sessionInfo(s) },
        "-",
        { label: "Stop agent", run: () => stopAgent(s, s.agentId) });
    }
    items.push("-", { label: "Keyboard shortcuts", run: showHotkeys });
    OMP.popupMenu(anchor, items);
  }

  async function projectMenu(anchor, s) {
    let projects = [];
    try { projects = (await api("/api/projects")).projects || []; } catch { /* none */ }
    const items = projects.slice(0, 14).map((p) => ({ label: `${shortPath(p.cwd)}  (${p.sessions})`, run: () => setProject(s, p.cwd) }));
    items.push("-", { label: "Browse…", run: async () => { const d = await OMP.dialogs.pickFolder(s.project).promise; if (d) setProject(s, d); } });
    OMP.popupMenu(anchor, items);
  }

  // ── boot ────────────────────────────────────────────────────────────────────
  async function init(a) {
    app = a;
    const s = S();
    $("#appearance-btn").onclick = () => OMP.appearance.open();
    $("#sidebar-toggle").onclick = () => {
      const appEl = $("#app");
      if (window.innerWidth <= 760) appEl.classList.toggle("show-sidebar"); else appEl.classList.toggle("no-sidebar");
    };
    $("#new-session").onclick = () => newSession(s, s.project);
    $("#project-pick").onclick = (e) => projectMenu(e.currentTarget, s);
    $("#session-filter").oninput = () => renderSidebar(s);
    $("#model-btn").onclick = () => modelPicker(s);
    $("#thinking-btn").onclick = () => thinkingPicker(s);
    $("#menu-btn").onclick = (e) => sessionMenu(e.currentTarget, s);
    $("#session-title").onclick = () => rename(s);
    const wrap = $("#transcript-wrap");
    const jump = $("#jump-bottom");
    wrap.addEventListener("scroll", () => {
      const near = app.transcript.isNearBottom();
      app.transcript.stick = near;
      jump.hidden = near;
    });
    jump.onclick = () => app.transcript.scrollToBottom(true);
    document.addEventListener("keydown", (e) => {
      if ($("#modal-root").childElementCount) return;
      const k = e.key.toLowerCase();
      if (e.ctrlKey && e.shiftKey && k === "m") { e.preventDefault(); if (s.agentId) modelPicker(s); }
      else if (e.ctrlKey && e.shiftKey && k === "t") { e.preventDefault(); if (s.agentId) thinkingPicker(s); }
      else if (e.ctrlKey && e.shiftKey && k === "n") { e.preventDefault(); newSession(s, app.currentCwd()); }
      else if (e.ctrlKey && e.shiftKey && k === "r") { e.preventDefault(); resumePicker(s, app.currentCwd()); }
      else if (e.ctrlKey && e.shiftKey && k === "a") { e.preventDefault(); OMP.appearance.open(); }
      else if (e.key === "Escape" && (s.streaming || s.bashRunning) && document.activeElement !== $("#composer")) { e.preventDefault(); app.abort(); }
    });
    setInterval(() => { if (s.streaming) renderStatus(s); }, 1000);
    setInterval(() => refreshAgents(s), 5000);
    window.addEventListener("hashchange", route);
    try { s.info = await api("/api/info"); } catch (e) { OMP.toast("Server unreachable: " + e.message, "error"); }
    if (!s.project) s.project = s.info.defaultCwd;
    await refreshAgents(s);
    loadSessions(s);
    const m = location.hash.match(/^#\/a\/([\w-]+)/);
    if (m && !s.agents.some((x) => x.id === m[1])) navigate(null);
    else route();
  }

  OMP.shellImpl = {
    init, renderHeader, renderStatus, renderSidebar, renderQueue, renderWidgets, refreshAgents, loadSessions,
    openSession, newSession, resumePicker, stopAgent, showHotkeys, modelPicker, thinkingPicker, exportHtml,
  };
})(window.OMP);
