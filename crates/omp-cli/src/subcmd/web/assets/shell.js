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
    const modelBtn = $("#composer-model");
    modelBtn.textContent = st.model ? `${st.model.provider}/${st.model.id}` : "model";
    modelBtn.disabled = !s.agentId;
    const tBtn = $("#composer-thinking");
    if (st.ultra) tBtn.textContent = "τ " + st.ultra;
    else tBtn.textContent = "◇ " + (st.thinkingLevel || "thinking");
    tBtn.classList.toggle("ultra", !!st.ultra);
    tBtn.disabled = !s.agentId;
    tBtn.hidden = !!(st.model && !st.model.reasoning && !st.ultra);
  }

  // Tau in the status bar: one persistent sprite, re-framed only when the
  // agent's state changes (re-inserting it would restart its animation).
  const liveHost = el("span.tau-live");
  let live = null;

  function renderStatus(s) {
    const bar = $("#statusbar");
    if (!s.agentId) { bar.replaceChildren(); return; }
    if (!live) live = OMP.mascot.mount(liveHost, { size: 22 });
    const mood = OMP.mascot.stateOf(s);
    live.set(mood);
    const parts = [];
    const st = s.state || {};
    const stats = s.stats || {};
    if (s.dead) parts.push(el("span.warn", { text: "stopped" }));
    else if (mood === "wave") parts.push(el("span.busy", { text: "waiting for you" }));
    else if (s.streaming) parts.push(el("span.busy", { text: `${mood === "work" ? "working" : "thinking"} ${OMP.fmtDuration(Date.now() - s.runStart)}` }));
    else parts.push(el("span", { text: "ready" }));
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
    while (bar.lastChild && bar.lastChild !== liveHost) bar.lastChild.remove();
    if (bar.firstChild !== liveHost) bar.prepend(liveHost);
    bar.append(...parts.flatMap((p, i) => (i ? [el("span.sep", { text: "│" }), p] : [p])));
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
    $("#project-path").textContent = shortPath(s.project);
    $("#project-path").title = s.project;

    const pinned = OMP.sidebar.readPinned();
    const q = $("#session-filter").value.trim().toLowerCase();
    const current = s.agents.find((a) => a.id === s.agentId);
    const openByAgent = new Set(s.agents.map((a) => a.sessionFile).filter(Boolean));

    const rows = [];

    // Running agents inline (live dot, needs-input/running chips).
    for (const a of s.agents) {
      const key = "a:" + a.id;
      const title = a.sessionName || basename(a.cwd);
      const sub = `${shortPath(a.cwd)}${a.model ? " · " + a.model : ""}`;
      if (q && !`${title} ${sub} ${a.id}`.toLowerCase().includes(q)) continue;
      rows.push({
        key, kind: "agent", id: a.id, path: a.sessionFile || null, cwd: a.cwd,
        title, sub, ms: a.startedMs || Date.now(),
        active: a.id === s.agentId, pinned: pinned.has(key),
        dot: a.streaming || a.runningTools ? "busy" : "live",
        status: a.needsInput ? "needs input" : a.runningTools ? "running" : a.streaming ? "working" : null,
      });
    }

    // Stored sessions (skipped when an agent already has the file open).
    for (const r of s.sessions) {
      if (openByAgent.has(r.path)) continue;
      const key = "s:" + r.path;
      const title = r.title || r.preview || r.id;
      const preview = r.preview ? (r.title ? " · " + r.preview : r.preview) : "";
      const sub = `${OMP.timeAgo(r.modifiedMs)}${preview}`;
      if (q && !`${title} ${r.preview || ""} ${r.id}`.toLowerCase().includes(q)) continue;
      rows.push({
        key, kind: "session", id: null, path: r.path, cwd: r.cwd || s.project,
        title, sub, ms: r.modifiedMs || 0,
        active: current && current.sessionFile === r.path, pinned: pinned.has(key),
        dot: null, status: r.store === "omp" ? "ts" : null,
      });
    }

    rows.sort((a, b) => b.ms - a.ms);

    const groups = new Map();
    const pinnedRows = [];
    for (const row of rows) {
      if (row.pinned) { pinnedRows.push(row); continue; }
      const g = OMP.sidebar.dateGroup(row.ms);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(row);
    }

    const frag = [];
    if (pinnedRows.length) {
      frag.push(groupHead("Pinned", pinnedRows.length));
      frag.push(...pinnedRows.slice(0, 300).map((r) => rowNode(s, r)));
    }
    let shown = pinnedRows.length;
    for (const g of OMP.sidebar.GROUP_ORDER) {
      const list = groups.get(g);
      if (!list || !list.length) continue;
      frag.push(groupHead(OMP.sidebar.GROUP_LABELS[g], list.length));
      const take = list.slice(0, Math.max(0, 300 - shown));
      frag.push(...take.map((r) => rowNode(s, r)));
      shown += take.length;
    }
    if (!frag.length) {
      frag.push(el("div.sess-empty.muted.small", { text: s.project ? "No matching sessions" : "No sessions yet" }));
    }
    $("#session-list").replaceChildren(...frag);
  }

  const PIN_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3h8l-1 8 3 3H6l3-3z"/><line x1="12" y1="18" x2="12" y2="21"/></svg>';
  const NEWWIN_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>';

  const groupHead = (label, count) => el("div.group-head", null,
    el("span", { text: label }), el("span.count", { text: String(count) }));

  const rowNode = (s, row) => {
    const dot = row.dot ? el("span.sess-dot." + row.dot, { title: row.dot === "busy" ? "working" : "running" })
      : el("span.sess-dot", { title: "stored" });
    const chip = row.status === "needs input"
      ? el("span.needs-input", { text: row.status })
      : row.status ? el("span.muted-run", { text: row.status }) : null;
    const actions = el("span.sess-actions", null,
      el("button.row-action" + (row.pinned ? ".pinned" : ""), { title: row.pinned ? "Unpin" : "Pin", html: PIN_SVG, onclick: (e) => { e.stopPropagation(); togglePin(s, row); } }),
      el("button.row-action", { title: "Open in new window", html: NEWWIN_SVG, onclick: (e) => { e.stopPropagation(); openInNewWindow(s, row); } }),
      el("button.row-action", { title: "More", text: "⋯", onclick: (e) => { e.stopPropagation(); rowMenu(e.currentTarget, s, row); } }));
    const main = el("div.main", null, dot,
      el("span.sess-title", { text: row.title }), chip, actions);
    return el("div.session-row" + (row.active ? ".active" : ""), {
      title: row.path || row.sub,
      onclick: () => { if (row.kind === "agent") navigate(row.id); else openSession(s, row); },
      oncontextmenu: (e) => { e.preventDefault(); rowMenu(e.target, s, row); },
    }, main, el("div.sess-sub", { text: row.sub }));
  };

  function togglePin(s, row) {
    const now = OMP.sidebar.togglePinned(row.key);
    OMP.toast(now ? "Pinned" : "Unpinned", "success");
    renderSidebar(S());
  }

  async function renameRow(s, row) {
    const name = await OMP.dialogs.promptText("Rename session", { value: row.title || "" }).promise;
    if (!name || !name.trim()) return;
    const n = name.trim();
    try {
      if (row.kind === "agent" && row.id === S().agentId && S().ws) {
        await app.rpc("set_session_name", { name: n });
      } else if (row.path) {
        await api("/api/sessions/rename", { method: "POST", body: { path: row.path, name: n } });
      } else {
        return OMP.toast("No session file to rename", "warning");
      }
    } catch (e) { return OMP.toast(e.message, "error"); }
    await refreshAgents(S());
    loadSessions(S());
    if (row.id === S().agentId) {
      S().state = await app.rpc("get_state").catch(() => S().state);
      renderHeader(S());
    }
  }

  // Resume a stored session into an agent without hijacking this tab.
  async function spawnQuiet(body) {
    try {
      const a = await api("/api/agents", { method: "POST", body });
      await refreshAgents(S());
      return a;
    } catch (e) { OMP.toast("Could not start agent: " + e.message, "error", 8000); return null; }
  }

  function openInNewWindow(s, row) {
    const ow = (id) => OMP.desktop && typeof OMP.desktop.openWindow === "function"
      && OMP.desktop.openWindow("#/a/" + id);
    if (row.kind === "agent") { if (row.id) ow(row.id); return; }
    spawnQuiet({ cwd: row.cwd || s.project, resume: row.path }).then((a) => { if (a) ow(a.id); });
  }

  // Continue a session in a terminal (`omt attach <id>` in a new Windows
  // Terminal tab); stored sessions are opened as an agent first.
  async function openInTerminal(s, row) {
    const id = row.kind === "agent" ? row.id : (await spawnQuiet({ cwd: row.cwd || s.project, resume: row.path }) || {}).id;
    if (!id) return;
    try {
      await api("/api/agents/" + encodeURIComponent(id) + "/terminal", { method: "POST", body: {} });
      OMP.toast("Opened in a terminal (omt attach)", "success");
    } catch (e) { OMP.toast("Terminal: " + e.message, "error", 8000); }
  }

  function rowMenu(anchor, s, row) {
    const items = [
      { label: "Rename…", run: () => renameRow(s, row) },
      { label: row.pinned ? "Unpin" : "Pin", run: () => togglePin(s, row) },
      "-",
      { label: "Open in new window", run: () => openInNewWindow(s, row) },
      { label: "Open in terminal", run: () => openInTerminal(s, row) },
    ];
    if (row.kind === "agent" && row.id === S().agentId) {
      items.push("-", { label: "Stop agent", run: () => stopAgent(S(), row.id) });
    }
    OMP.popupMenu(anchor, items);
  }

  // ── data ─────────────────────────────────────────────────────────────────
  async function refreshAgents(s) {
    let res;
    try {
      res = await api("/api/agents");
      s.agents = res.agents || [];
    } catch { return; }
    // `/desktop` from a terminal asks open pages to show its agent. On the
    // first poll only a fresh request counts (omrs was just started for it).
    const f = res.focus;
    if (f && f.ms > (s.focusSeen || 0)) {
      const fresh = s.focusSeen !== undefined || Date.now() - f.ms < 60000;
      s.focusSeen = f.ms;
      if (fresh && f.id !== s.agentId && s.agents.some((a) => a.id === f.id)) navigate(f.id);
    } else if (s.focusSeen === undefined) {
      s.focusSeen = 0;
    }
    renderSidebar(s);
    renderHeader(s);
  }

  async function loadSessions(s) {
    if (!s.project) return;
    try {
      s.sessions = (await api("/api/sessions?cwd=" + encodeURIComponent(s.project))).sessions || [];
    } catch (e) {
      s.sessions = s.sessions || [];
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
      ["Ctrl+Shift+M / Ctrl+Shift+T", "model picker / thinking level"],
      ["Ctrl+N / Ctrl+Shift+N", "new session"], ["Ctrl+Shift+R", "resume a session"],
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
        { label: "Open in terminal", run: () => openInTerminal(s, { kind: "agent", id: s.agentId }) },
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
    // The stored list is a snapshot: re-scan when the user goes looking.
    $("#session-filter").onfocus = () => loadSessions(s);
    window.addEventListener("focus", () => { loadSessions(s); refreshAgents(s); });
    $("#composer-model").onclick = () => modelPicker(s);
    $("#composer-thinking").onclick = () => thinkingPicker(s);
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
      else if (e.ctrlKey && !e.shiftKey && k === "n") { e.preventDefault(); newSession(s, app.currentCwd()); }
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
