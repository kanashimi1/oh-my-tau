"use strict";
// Shell: sidebar lists, header, status bar, pickers and session actions.
// `init(app)` (called by app.js) wires the DOM and boots routing.
(function (OMP) {
  const { el, $, api } = OMP;
  let app = null;
  const S = () => app.S;

  const shortPath = (p) => OMP.shortPath(p, S().info.home);
  const fmtMS = (ms) => { const s = Math.max(0, Math.round(ms / 1000)); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };
  const basename = (p) => String(p || "").replace(/\/+$/, "").split("/").pop() || p;

  function navigate(id) {
    const target = id ? "#/a/" + id : "#/";
    if (location.hash === target) route();
    else location.hash = target;
  }
  function route() {
    if (/^#maps(\/|$)/.test(location.hash)) { app.setView("maps"); return; }
    if (S().view === "maps") app.setView("chat");
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
    const fullModel = st.model ? `${st.model.provider}/${st.model.id}` : "model";
    const modelBtn = $("#composer-model");
    modelBtn.textContent = st.model ? st.model.id : "model";
    modelBtn.title = st.model ? `${fullModel} (/model)` : "Model (/model)";
    modelBtn.disabled = !s.agentId;
    const tBtn = $("#composer-thinking");
    const effort = st.ultra || st.thinkingLevel || "thinking";
    const previousEffort = tBtn.dataset.effort;
    const rank = st.ultra || /^(high|xhigh|max)$/i.test(effort) ? 3 : /^(medium|mid|auto)$/i.test(effort) ? 2 : /^(off|none)$/i.test(effort) ? 0 : 1;
    tBtn.querySelector(".effort-label").textContent = effort;
    tBtn.querySelectorAll(".effort-segment").forEach((segment, i) => segment.classList.toggle("active", i < rank));
    tBtn.dataset.effort = effort;
    tBtn.dataset.level = String(rank);
    tBtn.classList.toggle("ultra", !!st.ultra);
    if (previousEffort && previousEffort !== effort) {
      tBtn.classList.remove("effort-changing");
      void tBtn.offsetWidth;
      tBtn.classList.add("effort-changing");
    }
    tBtn.disabled = !s.agentId;
    tBtn.hidden = !!(st.model && !st.model.reasoning && !st.ultra);
  }

  // Tau in the status bar: one persistent sprite, re-framed only when the
  // agent's state changes (re-inserting it would restart its animation).
  const liveHost = el("span.tau-live");
  let runOut0 = null; // stats.tokens.output snapshot at run start
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
    if (s.streaming && runOut0 == null) runOut0 = stats.tokens?.output ?? 0;
    if (!s.streaming && runOut0 != null) {
      const durMs = s.runStart ? Date.now() - s.runStart : null;
      const out = Math.max(0, (stats.tokens?.output || 0) - runOut0);
      if (durMs > 0) OMP.perf.last = { ...(OMP.perf.last || {}), durMs, tps: out / (durMs / 1000), endedAt: Date.now() };
      runOut0 = null;
    }
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
    if (t && (t.cacheRead || t.cacheWrite)) parts.push(el("span", { text: "cache " + OMP.fmtTokens((t.cacheRead || 0) + (t.cacheWrite || 0)), title: "cache read " + (t.cacheRead || 0) + " · cache write " + (t.cacheWrite || 0) }));
    if (s.streaming && s.runStart) {
      const out = Math.max(0, (t?.output || 0) - (runOut0 || 0));
      const sec = (Date.now() - s.runStart) / 1000;
      if (sec >= 2 && out > 0) parts.push(el("span.busy", { text: (out / sec).toFixed(1) + " t/s", title: "output tokens per second (live)" }));
    }
    const L = !s.streaming ? OMP.perf.last : null;
    if (L) {
      const segs = [];
      if (L.durMs != null) segs.push(fmtMS(L.durMs));
      if (L.tps) segs.push(L.tps.toFixed(1) + " t/s");
      if (L.endedAt) segs.push(new Date(L.endedAt).toTimeString().slice(0, 5));
      if (segs.length) parts.push(el("span", { text: segs.join(" · "), title: "last response: duration · speed · finished at" }));
    }
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

  function renderWidgets(s) {
    const above = [], below = [];
    for (const w of s.widgets.values()) (w.below ? below : above).push(el("div", { text: w.lines.join("\n") }));
    $("#widgets-above").replaceChildren(...above);
    $("#widgets-below").replaceChildren(...below);
    if (OMP.agentPanel) { OMP.agentPanel.renderLeft($("#agent-dock")); OMP.agentPanel.renderRight($("#subagents-dock")); }
  }

  function renderSidebar(s) {
    $("#project-path").textContent = shortPath(s.project);
    $("#project-path").title = s.project;

    const pinned = OMP.sidebar.readPinned();
    const archived = OMP.sidebar.readArchived();
    const q = $("#session-filter").value.trim().toLowerCase();
    const current = s.agents.find((a) => a.id === s.agentId);
    const anyOpen = new Set(s.agents.map((a) => a.sessionFile).filter(Boolean));
    // A dead agent gets no row by default, so its file must fall back to the
    // stored list — otherwise a finished session vanishes with its agent.
    const aliveOpen = new Set(s.agents.filter((a) => a.alive !== false).map((a) => a.sessionFile).filter(Boolean));
    // Stored titles by file: resumed agents often have no sessionName, and
    // their cwd fallback (e.g. tmp from desktop spawns) makes fleet rows
    // read as identical duplicates. Prefer the stored session title.
    const storedTitle = new Map(s.sessions.map((r) => [r.path, r.title || r.preview || ""]).filter(([, t]) => t));

    // Explorer buckets: the live fleet on top, then pinned, then the stored
    // history (date or project groups), then soft-hidden archived rows.
    const liveRows = [], histRows = [], archivedRows = [];

    // Running now: live agents (pulsing dot, needs-input/running chips).
    // Dead agents fall back to their stored row unless a search names them.
    for (const a of s.agents) {
      const key = "a:" + a.id;
      if (archived.has(key)) continue;
      if (a.alive === false && !q) continue;
      const title = a.sessionName || storedTitle.get(a.sessionFile) || basename(a.cwd);
      // Worktree agents run in a fresh git worktree; the slug after the
      // last `-wt-` names the `omt/<slug>` branch to merge back.
      const wtSlug = a.worktree ? String(a.worktree).split("/").pop().replace(/^.*-wt-/, "") : null;
      const sub = wtSlug
        ? `${shortPath(a.worktree)} · merge: /wt merge ${wtSlug}${a.model ? " · " + a.model : ""}`
        : `${shortPath(a.cwd)}${a.model ? " · " + a.model : ""}`;
      if (q && !`${title} ${sub} ${a.id}`.toLowerCase().includes(q)) continue;
      const row = {
        key, kind: "agent", id: a.id, path: a.sessionFile || null, cwd: a.cwd,
        title, sub, ms: a.startedMs || Date.now(),
        active: a.id === s.agentId, pinned: pinned.has(key),
        dot: a.streaming || a.runningTools ? "busy" : "live", pulse: a.alive !== false,
        status: a.needsInput ? "needs input" : a.runningTools ? "running" : a.streaming ? "working" : null,
        waitingMs: !a.streaming && a.awaitingTasksSinceMs ? Date.now() - a.awaitingTasksSinceMs : null,
      };
      (a.alive !== false ? liveRows : histRows).push(row);
    }

    // Default view: all live agents + five recent stored sessions, with a
    // "show all" expander for the rest. Search deliberately scans the
    // complete stored history; pinned/current sessions stay reachable even
    // when they are older than the recent slice.
    const openSet = q ? anyOpen : aliveOpen;
    const stored = s.sessions.filter((r) => !openSet.has(r.path))
      .sort((a, b) => (b.modifiedMs || 0) - (a.modifiedMs || 0));
    const showAll = !q && !!s.sessionsShowAll;
    const visibleStored = q ? stored : stored.filter((r, i) => showAll || i < 5 || pinned.has("s:" + r.path) || (current && current.sessionFile === r.path));
    const hiddenStored = q ? 0 : stored.length - visibleStored.length;
    // With several projects loaded, flat rows carry their project so the
    // date-grouped list still tells sessions apart.
    const multiCwd = new Set(s.sessions.map((r) => r.cwd).filter(Boolean)).size > 1;
    const byProject = s.groupByProject && multiCwd;
    // Stored sessions (skipped when a rendered agent already has the file open).
    for (const r of visibleStored) {
      if (openSet.has(r.path)) continue;
      const key = "s:" + r.path;
      const title = r.title || r.preview || r.id;
      const preview = r.preview ? (r.title ? " · " + r.preview : r.preview) : "";
      const proj = !byProject && multiCwd && r.cwd && r.cwd !== s.project ? shortPath(r.cwd) + " · " : "";
      const sub = `${proj}${OMP.timeAgo(r.modifiedMs)}${preview}`;
      if (q && !`${title} ${r.preview || ""} ${r.id} ${r.cwd || ""}`.toLowerCase().includes(q)) continue;
      const row = {
        key, kind: "session", id: null, path: r.path, cwd: r.cwd || s.project,
        title, sub, ms: r.modifiedMs || 0,
        active: current && current.sessionFile === r.path, pinned: pinned.has(key),
        dot: null, status: r.store === "omp" ? "ts" : null,
      };
      (archived.has(key) ? archivedRows : histRows).push(row);
    }
    for (const list of [liveRows, histRows, archivedRows]) list.sort((a, b) => b.ms - a.ms);

    // History grouping: per project when the explorer spans projects and the
    // user opted in, Kimi date buckets otherwise.
    const pinnedRows = histRows.filter((r) => r.pinned);
    const unpinned = histRows.filter((r) => !r.pinned);
    const groups = new Map();
    for (const row of unpinned) {
      const g = byProject ? (row.cwd || s.project) : OMP.sidebar.dateGroup(row.ms);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(row);
    }
    const order = byProject ? [...groups.keys()] : OMP.sidebar.GROUP_ORDER.filter((g) => groups.has(g));

    const frag = [];
    if (liveRows.length) {
      frag.push(groupHead("Running now", liveRows.length));
      frag.push(...liveRows.map((r) => rowNode(s, r)));
    }
    if (pinnedRows.length) {
      frag.push(groupHead("Pinned", pinnedRows.length));
      frag.push(...pinnedRows.slice(0, 300).map((r) => rowNode(s, r)));
    }
    let shown = liveRows.length + pinnedRows.length;
    for (const g of order) {
      const list = groups.get(g);
      if (!list || !list.length) continue;
      frag.push(groupHead(byProject ? shortPath(g) : OMP.sidebar.GROUP_LABELS[g], list.length));
      const take = list.slice(0, Math.max(0, 300 - shown));
      frag.push(...take.map((r) => rowNode(s, r)));
      shown += take.length;
    }
    if (!q && (hiddenStored > 0 || (showAll && stored.length > 5))) {
      frag.push(el("button.sess-more", {
        text: showAll ? "Show recent only" : `Show all ${stored.length} sessions`,
        title: showAll ? null : `${hiddenStored} older sessions are hidden`,
        onclick: () => { s.sessionsShowAll = !showAll; renderSidebar(s); },
      }));
    }
    // Archived: collapsible group at the bottom; search always expands it so
    // a soft-hidden session stays findable.
    if (archivedRows.length) {
      const open = s.archivedOpen || !!q;
      frag.push(el("button.group-head.arch-head", {
        title: open ? "Hide archived sessions" : "Show archived sessions",
        onclick: () => {
          s.archivedOpen = !open;
          try { localStorage.setItem("omp-archived-open", String(s.archivedOpen)); } catch { /* storage unavailable */ }
          renderSidebar(s);
        },
      }, el("span", { text: "Archived" }), el("span.count", { text: `${open ? "−" : "+"} ${archivedRows.length}` })));
      if (open) frag.push(...archivedRows.map((r) => rowNode(s, r)));
    }
    if (!frag.length) {
      frag.push(el("div.sess-empty.muted.small", { text: s.project ? "No matching sessions" : "No sessions yet" }));
    }
    const groupBtn = $("#group-toggle");
    groupBtn.hidden = !multiCwd;
    groupBtn.textContent = byProject ? "by project" : "by date";
    groupBtn.title = byProject ? "Grouped by project — switch to date groups" : "Grouped by date — switch to project groups";
    $("#session-list").replaceChildren(...frag);
    if (OMP.agentPanel) { OMP.agentPanel.renderLeft($("#agent-dock")); OMP.agentPanel.renderRight($("#subagents-dock")); }
  }

  const PIN_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3h8l-1 8 3 3H6l3-3z"/><line x1="12" y1="18" x2="12" y2="21"/></svg>';
  const NEWWIN_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>';
  const FORK_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>';
  const ARCHIVE_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="21 8 21 21 3 21 3 8"/><rect x="1" y="3" width="22" height="5"/><line x1="10" y1="12" x2="14" y2="12"/></svg>';

  const groupHead = (label, count) => el("div.group-head", null,
    el("span", { text: label }), el("span.count", { text: String(count) }));

  const rowNode = (s, row) => {
    const dot = row.dot ? el("span.sess-dot." + row.dot + (row.pulse ? ".pulse" : ""), { title: row.dot === "busy" ? "working" : "running" })
      : el("span.sess-dot", { title: "stored" });
    // Parked on background subagents: harmless for a few minutes, a likely
    // stuck child after that (the session never settles on its own).
    const waitChip = row.waitingMs != null && !row.status
      ? el("span." + (row.waitingMs > STUCK_MS ? "needs-input.stuck" : "muted-run"), {
          text: "subagent " + waitText(row.waitingMs),
          title: "Waiting for its background subagents since " + new Date(Date.now() - row.waitingMs).toLocaleString() + ". Click to cancel them.",
          onclick: (e) => { e.stopPropagation(); cancelSubagents(S(), row.id); },
        })
      : null;
    const chip = row.status === "needs input"
      ? el("span.needs-input", { text: row.status })
      : row.status ? el("span.muted-run", { text: row.status }) : waitChip;
    const actions = el("span.sess-actions", null,
      el("button.row-action" + (row.pinned ? ".pinned" : ""), { title: row.pinned ? "Unpin" : "Pin", html: PIN_SVG, onclick: (e) => { e.stopPropagation(); togglePin(s, row); } }),
      el("button.row-action", { title: "New session from this one", html: FORK_SVG, onclick: (e) => { e.stopPropagation(); forkFrom(s, row); } }),
      el("button.row-action", { title: row.archived ? "Unarchive" : "Archive", html: ARCHIVE_SVG, onclick: (e) => { e.stopPropagation(); toggleArchive(s, row); } }),
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

  // "New from": continue a session as a fresh agent without hijacking this
  // tab — the stored resume flow for rows with a file, the attached agent's
  // own fork rpc for the current session.
  async function forkFrom(s, row) {
    if (row.kind === "agent" && row.id === S().agentId && S().ws) {
      try { await app.rpc("fork"); OMP.toast("Forked into a new session file", "success"); } catch (e) { return OMP.toast(e.message, "error"); }
      s.state = await app.rpc("get_state").catch(() => s.state);
      renderHeader(s);
      refreshAgents(s);
      loadSessions(s);
      return;
    }
    if (!row.path) return OMP.toast("No session file to fork", "warning");
    const a = await spawnQuiet({ cwd: row.cwd || s.project, resume: row.path });
    if (a) OMP.toast("New session started from this one", "success");
  }

  // Soft-hide a row behind the collapsible "Archived" group (persisted in
  // localStorage "omp-archived-sessions" via the sidebar helpers).
  function toggleArchive(s, row) {
    const now = OMP.sidebar.toggleArchived(row.key);
    OMP.toast(now ? "Archived" : "Unarchived", "success");
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
      { label: "New session from this one", run: () => forkFrom(s, row) },
      { label: row.archived ? "Unarchive" : "Archive", run: () => toggleArchive(s, row) },
      { label: "Open in new window", run: () => openInNewWindow(s, row) },
      { label: "Open in terminal", run: () => openInTerminal(s, row) },
    ];
    if (row.waitingMs != null) {
      items.push("-", { label: "Cancel waiting subagents", run: () => cancelSubagents(S(), row.id) });
    }
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
    window.dispatchEvent(new CustomEvent("omp:agents", { detail: { agents: s.agents } }));
    // `/desktop` from a terminal asks open pages to show its agent. On the
    // first poll only a fresh request counts (Tau was just started for it).
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
      // Empty `cwd` = the explorer's cross-project history across both
      // stores (see sessions.rs); rows carry their own `cwd`.
      s.sessions = (await api("/api/sessions?cwd=")).sessions || [];
    } catch (e) {
      s.sessions = s.sessions || [];
      OMP.toast("Sessions: " + e.message, "error");
    }
    renderSidebar(s);
  }

  function setProject(s, cwd) {
    s.project = cwd;
    s.sessionsShowAll = false;
    localStorage.setItem("omp-web-project", cwd);
    window.dispatchEvent(new CustomEvent("omp:project", { detail: { cwd } }));
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
    await spawn({ cwd: dir, ...(OMP.agentPanel ? OMP.agentPanel.creationOptions() : {}) });
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
        app.transcript.loadMessages(await app.getMessages());
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

  const STUCK_MS = 10 * 60 * 1000;
  const waitText = (ms) => ms < 3600000 ? Math.max(1, Math.round(ms / 60000)) + "m" : (ms / 3600000).toFixed(ms < 36000000 ? 1 : 0).replace(/\.0$/, "") + "h";

  /** Abort a session parked on background subagents (kills the children). */
  async function cancelSubagents(s, id) {
    const ok = await OMP.dialogs.confirmBox("Cancel subagents",
      "This session is waiting for its background subagents. Cancel them and end the wait? Their unfinished results are lost; the session stays and you can continue it.").promise;
    if (!ok) return;
    try { await api("/api/agents/" + id + "/abort", { method: "POST" }); OMP.toast("Subagents cancelled", "success"); }
    catch (e) { OMP.toast(e.message, "error"); }
    setTimeout(() => refreshAgents(s), 600);
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
      ["Ctrl+B", "hide or show the sidebar"],
      ["Ctrl+Shift+M / Ctrl+Shift+T", "model picker / thinking level"],
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
    // Ticker: keeps the streaming "working m:ss" chip and live tok/s fresh;
    // otherwise a safe no-op (app.js renders status only on events).
    setInterval(() => { if (app && S().agentId && (S().streaming || OMP.perf.last)) renderStatus(S()); }, 1000);
    const s = S();
    // Explorer prefs survive restarts: history grouping, archived group state.
    try { s.groupByProject = localStorage.getItem("omp-web-group-by") === "project"; } catch { /* storage unavailable */ }
    try { s.archivedOpen = localStorage.getItem("omp-archived-open") === "true"; } catch { /* storage unavailable */ }
    $("#appearance-btn").onclick = () => OMP.appearance.open();
    $("#glyph-appearance-btn").onclick = () => OMP.appearance.open();
    const appEl = $("#app");
    const sidebarButton = $("#sidebar-toggle");
    const hideButton = $("#sidebar-hide");
    const activityCollapse = $("#activity-collapse");
    const activityExpand = $("#activity-expand");
    const setActivityCollapsed = (collapsed, persist = true) => {
      appEl.classList.toggle("activity-collapsed", collapsed);
      activityCollapse.setAttribute("aria-expanded", String(!collapsed));
      activityExpand.hidden = !collapsed;
      if (persist) try { localStorage.setItem("omp-activity-collapsed", String(collapsed)); } catch { /* storage unavailable */ }
    };
    let activityCollapsed = false;
    try { activityCollapsed = localStorage.getItem("omp-activity-collapsed") === "true"; } catch { /* storage unavailable */ }
    setActivityCollapsed(activityCollapsed, false);
    activityCollapse.onclick = () => setActivityCollapsed(true);
    activityExpand.onclick = () => setActivityCollapsed(false);
    const setSidebarHidden = (hidden, persist = true) => {
      appEl.classList.toggle("no-sidebar", hidden);
      sidebarButton.setAttribute("aria-expanded", String(!hidden));
      sidebarButton.title = `${hidden ? "Show" : "Hide"} sidebar (Ctrl+B)`;
      if (hideButton) {
        hideButton.setAttribute("aria-pressed", String(hidden));
        hideButton.title = `${hidden ? "Show" : "Hide"} sidebar (Ctrl+B)`;
      }
      if (persist) try { localStorage.setItem("omp-web-sidebar-hidden", String(hidden)); } catch { /* storage unavailable */ }
    };
    let sidebarHidden = false;
    try { sidebarHidden = localStorage.getItem("omp-web-sidebar-hidden") === "true"; } catch { /* storage unavailable */ }
    setSidebarHidden(sidebarHidden, false);
    sidebarButton.onclick = () => {
      if (window.innerWidth <= 760) appEl.classList.toggle("show-sidebar");
      else setSidebarHidden(!appEl.classList.contains("no-sidebar"));
    };
    // Optional in-sidebar hide button (parent adds it): same path as the topbar
    // toggle, so a hidden menu stays reopenable via #sidebar-toggle / Ctrl+B.
    if (hideButton) {
      setSidebarHidden(sidebarHidden, false); // sync its aria/title state
      hideButton.onclick = () => setSidebarHidden(!appEl.classList.contains("no-sidebar"));
    }
    $("#new-session").onclick = () => newSession(s, s.project);
    $("#maps-btn").onclick = () => { if (OMP.maps) OMP.maps.toggle(); };
    $("#project-pick").onclick = (e) => projectMenu(e.currentTarget, s);
    // Debounced: the explorer holds the whole history, so re-rendering on
    // every keystroke would be wasteful.
    $("#session-filter").oninput = OMP.sidebar.debounce(() => renderSidebar(s), 150);
    $("#group-toggle").onclick = () => {
      s.groupByProject = !s.groupByProject;
      try { localStorage.setItem("omp-web-group-by", s.groupByProject ? "project" : "date"); } catch { /* storage unavailable */ }
      renderSidebar(s);
    };
    // The stored list is a snapshot: re-scan when the user goes looking.
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
      if (e.ctrlKey && !e.shiftKey && (k === "b" || e.code === "KeyB")) {
        e.preventDefault();
        if (window.innerWidth <= 760) appEl.classList.toggle("show-sidebar");
        else setSidebarHidden(!appEl.classList.contains("no-sidebar"));
      }
      else if (e.ctrlKey && e.shiftKey && k === "m") { e.preventDefault(); if (s.agentId) modelPicker(s); }
      else if (e.ctrlKey && e.shiftKey && k === "t") { e.preventDefault(); if (s.agentId) thinkingPicker(s); }
      else if (e.ctrlKey && e.shiftKey && k === "n") { e.preventDefault(); newSession(s, app.currentCwd()); }
      else if (e.ctrlKey && !e.shiftKey && k === "n") { e.preventDefault(); newSession(s, app.currentCwd()); }
      else if (e.ctrlKey && e.shiftKey && k === "r") { e.preventDefault(); resumePicker(s, app.currentCwd()); }
      else if (e.ctrlKey && e.shiftKey && k === "a") { e.preventDefault(); OMP.appearance.open(); }
      else if (e.key === "Escape" && OMP.maps && OMP.maps.isOpen()) { e.preventDefault(); if (!OMP.maps.closeDetail()) OMP.maps.close(); }
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
    openSession, newSession, resumePicker, stopAgent, showHotkeys, modelPicker, thinkingPicker, exportHtml, setProject,
  };
})(window.OMP);
