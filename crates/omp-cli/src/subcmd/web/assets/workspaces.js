"use strict";
// Project workspaces (feature: workspaces).
// Per-project persisted state: active + recently open sessions, selected/open
// preview file, artifacts-panel size/visibility, sidebar visibility, terminal
// tab definitions (via the optional OMP.terminal API) and model/thinking
// preferences. Everything lives in one versioned, bounded localStorage key.
// No secrets and no message transcripts are ever stored.
//
// Integration contract (parent wires these):
//   - `init(app)` after the shell booted (OMP.app exists, topbar rendered).
//   - Events consumed: `omp:agents` {agents}, `omp:open-session` {id, cwd},
//     `omp:project` {cwd}, `omp:frame` {agentId, frame}.
//   - Project switching: setting `omp-web-project` + app.S.project, then a
//     `omp:project` {cwd} event that the shell listens to for its UI refresh.
//   - Session restore uses `OMP.shellImpl.openSession(S, row)` when present,
//     otherwise the `omp:open-session` event.
(function (OMP) {
  const { el, $ } = OMP;

  const LS_KEY = "omt-workspaces-v1";
  const VERSION = 1;
  const MAX_WORKSPACES = 24;      // bounded store: LRU eviction beyond this
  const MAX_RECENT = 10;
  const SAVE_DEBOUNCE_MS = 800;

  const app = { ref: null };
  let S = null;                   // app.S
  let store = null;               // {version, workspaces:{}, names:{}, order:[]}
  let saveTimer = 0;
  let currentKey = "";            // project cwd the snapshot is for
  let restoredFor = "";           // guard: restore once per project
  let lastPreviewFile = "";       // captured by wrapping OMP.artifacts.open
  let pendingSwitch = null;       // workspace to restore after agents refresh
  let applying = false;           // suppress capture while we restore
  let quotaWarned = false;

  const basename = (p) => String(p || "").replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p;
  const normCwd = (p) => String(p || "").replace(/[\\/]+$/, "");
  const project = () => normCwd((S && S.project) || "");
  const home = () => (S && S.info && S.info.home) || "";
  const short = (p) => OMP.shortPath(p, home());

  // ── storage ───────────────────────────────────────────────────────────────
  function load() {
    let raw = null;
    try { raw = localStorage.getItem(LS_KEY); } catch { /* private mode */ }
    if (!raw) return fresh();
    try {
      const parsed = JSON.parse(raw);
      if (!parsed || parsed.version !== VERSION || typeof parsed.workspaces !== "object") return fresh();
      parsed.order = Array.isArray(parsed.order) ? parsed.order : Object.keys(parsed.workspaces);
      parsed.names = parsed.names && typeof parsed.names === "object" ? parsed.names : {};
      return parsed;
    } catch {
      // Corrupted payload: keep it aside once, start clean, never crash.
      try { localStorage.setItem(LS_KEY + ".corrupt", String(raw).slice(0, 65536)); } catch { /* ignore */ }
      return fresh();
    }
  }
  function fresh() { return { version: VERSION, workspaces: {}, names: {}, order: [] }; }

  function persist() {
    clearTimeout(saveTimer);
    saveTimer = 0;
    try { localStorage.setItem(LS_KEY, JSON.stringify(store)); }
    catch (e) {
      // Quota or serialization failure: drop the oldest half, retry once.
      if (!quotaWarned) { OMP.toast("Workspace storage is full — oldest workspaces were forgotten", "warning"); quotaWarned = true; }
      store.order = store.order.slice(-Math.ceil(MAX_WORKSPACES / 2));
      store.workspaces = Object.fromEntries(store.order.filter((k) => store.workspaces[k]).map((k) => [k, store.workspaces[k]]));
      try { localStorage.setItem(LS_KEY, JSON.stringify(store)); } catch { /* give up silently */ }
    }
  }
  // Expensive writes are debounced; navigation never blocks on storage.
  function scheduleSave() {
    if (saveTimer) return;
    saveTimer = setTimeout(persist, SAVE_DEBOUNCE_MS);
  }

  function wsFor(cwd) {
    cwd = normCwd(cwd);
    if (!cwd) return null;
    if (!store.workspaces[cwd]) {
      store.workspaces[cwd] = { activeSessionId: "", recentSessions: [], preview: null, sidebar: null, terminal: null, model: null, thinking: "", ts: 0 };
      store.order.push(cwd);
    }
    // LRU touch + bound.
    const i = store.order.indexOf(cwd);
    if (i !== -1) store.order.splice(i, 1);
    store.order.push(cwd);
    while (store.order.length > MAX_WORKSPACES) {
      const drop = store.order.shift();
      delete store.workspaces[drop];
      delete store.names[drop];
    }
    return store.workspaces[cwd];
  }

  const wsName = (cwd) => store.names[normCwd(cwd)] || basename(cwd);

  // ── snapshot / restore ────────────────────────────────────────────────────
  function currentSnapshot() {
    const cwd = project();
    if (!cwd) return null;
    const ws = wsFor(cwd);
    if (!ws) return null;

    // Active + recent sessions come from live UI state, never from transcripts.
    const agents = (S.agents || []).filter((a) => normCwd(a.cwd) === cwd);
    const active = agents.find((a) => a.id === S.agentId);
    ws.activeSessionId = active ? active.id : "";
    ws.activeSessionPath = active ? (active.sessionFile || "") : "";
    const seen = new Set();
    ws.recentSessions = agents
      .filter((a) => a.id !== S.agentId && a.sessionFile && !seen.has(a.sessionFile) && seen.add(a.sessionFile))
      .slice(0, MAX_RECENT)
      .map((a) => ({ id: a.id, name: a.sessionName || basename(a.sessionFile), path: a.sessionFile || "" }));

    // Preview panel: open state, width and last file (from the open-wrapper).
    const panel = $("#artifacts-panel");
    ws.preview = {
      open: !!(OMP.artifacts && OMP.artifacts.isOpen()),
      file: lastPreviewFile || "",
      width: panel ? parseInt(panel.style.width, 10) || 0 : 0,
    };

    // Sidebar visibility (shell toggles `.no-sidebar` on #app).
    const appEl = $("#app");
    ws.sidebar = appEl ? appEl.classList.contains("no-sidebar") : null;

    // Terminal tabs: optional sibling feature API; only a small definition
    // snapshot (titles/cwd), never scrollback content.
    ws.terminal = OMP.terminal && typeof OMP.terminal.serialize === "function"
      ? (function () { try { return OMP.terminal.serialize() || null; } catch { return null; } })()
      : null;

    // Model / thinking preferences as of now.
    const st = S.state || {};
    ws.model = st.model ? { provider: st.model.provider, id: st.model.id } : null;
    ws.thinking = String(st.ultra || st.thinkingLevel || "");
    ws.ts = Date.now();
    return ws;
  }

  function save() {
    if (applying) return;
    try { currentSnapshot(); } catch { /* never let capture break the UI */ }
    scheduleSave();
  }

  function snapshot() {
    // Read-only view for other features.
    try { return JSON.parse(JSON.stringify(store.workspaces[project()] || null)); } catch { return null; }
  }

  async function restore() {
    const cwd = project();
    if (!cwd || restoredFor === cwd) return;
    restoredFor = cwd;
    const ws = store.workspaces[cwd];
    if (!ws) return;
    applying = true;
    try {
      // 1. Sessions: prefer the still-running agent, else resume the stored file.
      const list = S.agents || [];
      let opened = false;
      if (ws.activeSessionId && list.some((a) => a.id === ws.activeSessionId)) {
        location.hash = "#/a/" + ws.activeSessionId;
        opened = true;
      } else if (ws.activeSessionPath && OMP.shellImpl && OMP.shellImpl.openSession) {
        OMP.shellImpl.openSession(S, { kind: "session", path: ws.activeSessionPath, cwd });
        opened = true;
      } else if (ws.recentSessions && ws.recentSessions.length) {
        for (const r of ws.recentSessions) {
          if (!r.path) continue;
          const running = list.find((a) => a.sessionFile === r.path);
          if (running) { location.hash = "#/a/" + running.id; opened = true; break; }
          if (OMP.shellImpl && OMP.shellImpl.openSession) {
            OMP.shellImpl.openSession(S, { kind: "session", path: r.path, cwd });
            opened = true;
            break;
          }
        }
      }
      if (!opened) window.dispatchEvent(new CustomEvent("omp:open-session", { detail: { id: "", cwd } }));

      // 2. Preview panel: width first (it is applied on next open), then content.
      if (ws.preview) {
        if (ws.preview.width) {
          const panel = $("#artifacts-panel");
          if (panel) panel.style.width = ws.preview.width + "px";
          try { localStorage.setItem("omt-art-width", String(ws.preview.width)); } catch { /* ignore */ }
        }
        if (ws.preview.open && ws.preview.file) {
          if (OMP.artifacts && OMP.artifacts.open) {
            OMP.artifacts.open({ kind: "file", path: ws.preview.file, title: basename(ws.preview.file) });
          }
        }
      }

      // 3. Sidebar visibility.
      const appEl = $("#app");
      if (appEl && ws.sidebar != null && window.innerWidth > 760) appEl.classList.toggle("no-sidebar", !!ws.sidebar);

      // 4. Terminal definitions, only through the optional terminal API.
      if (ws.terminal && OMP.terminal && typeof OMP.terminal.restore === "function") {
        try { OMP.terminal.restore(ws.terminal); } catch (e) { console.warn("terminal restore failed", e); }
      }

      // 5. Model / thinking: applied once the agent connection has state.
      applyPrefsSoon(ws);
    } finally {
      applying = false;
      save(); // record anything the restore changed, debounced
    }
  }

  let prefsTimer = 0;
  function applyPrefsSoon(ws) {
    clearTimeout(prefsTimer);
    let tries = 0;
    const tick = async () => {
      const st = S.state;
      if ((!st || !S.agentId || !S.ws) && ++tries < 40) { prefsTimer = setTimeout(tick, 500); return; }
      await applyPrefs(ws);
    };
    prefsTimer = setTimeout(tick, 500);
  }

  async function applyPrefs(ws) {
    const st = S.state;
    if (!st || !S.agentId || !S.ws || S.ws.readyState !== WebSocket.OPEN) return;
    const wantModel = ws.model;
    const haveModel = st.model ? `${st.model.provider}/${st.model.id}` : "";
    try {
      if (wantModel && haveModel && haveModel !== `${wantModel.provider}/${wantModel.id}`) {
        await app.ref.rpc("set_model", { provider: wantModel.provider, modelId: wantModel.id });
        OMP.toast(`Model restored: ${wantModel.provider}/${wantModel.id}`, "info");
      }
      const wantThink = String(ws.thinking || "");
      const haveThink = String(st.ultra || st.thinkingLevel || "");
      if (wantThink && haveThink && wantThink !== haveThink) {
        await app.ref.rpc("set_thinking_level", { level: wantThink });
      }
    } catch { /* older agent or refused preference: stay on the current one */ }
  }

  // ── project switch ────────────────────────────────────────────────────────
  function saveNow() {
    clearTimeout(saveTimer);
    saveTimer = 0;
    save();
    persist();
  }

  function onProjectChange(cwd) {
    cwd = normCwd(cwd || project());
    if (!cwd) return;
    if (currentKey && currentKey !== cwd) saveNow(); // persist the old project
    currentKey = cwd;
    restoredFor = ""; // allow restore for the new project
    clearTimeout(prefsTimer);
    restore();
  }

  async function switchTo(cwd) {
    cwd = normCwd(cwd);
    if (!cwd) return;
    if (cwd === project()) return restore();
    saveNow(); // persist the project we are leaving
    if (OMP.shellImpl && OMP.shellImpl.setProject) OMP.shellImpl.setProject(S, cwd);
    else {
      try { localStorage.setItem("omp-web-project", cwd); } catch { /* ignore */ }
      S.project = cwd;
      window.dispatchEvent(new CustomEvent("omp:project", { detail: { cwd } }));
    }
    currentKey = cwd;
    restoredFor = "";
    const side = $("#project-path");
    if (side) side.textContent = OMP.shortPath(cwd, home());
    // Wait for the agent roster to include this project's sessions, then restore.
    pendingSwitch = cwd;
    await refreshAgents();
    restore().then(() => { pendingSwitch = null; });
  }

  async function refreshAgents() {
    try {
      if (OMP.shellImpl && OMP.shellImpl.refreshAgents) await OMP.shellImpl.refreshAgents(S);
      else {
        const data = await OMP.api("/api/agents");
        const list = data.agents || data || [];
        if (S) S.agents = list;
        window.dispatchEvent(new CustomEvent("omp:agents", { detail: { agents: list, list, active: (S && S.agentId) || "" } }));
      }
    } catch { /* server hiccup: restore continues with what we have */ }
  }

  // ── UI: topbar button + switcher dialog ───────────────────────────────────
  const BTN_SVG = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="9" height="7" rx="1.5"/><rect x="13" y="4" width="9" height="7" rx="1.5"/><rect x="2" y="13" width="9" height="7" rx="1.5"/><rect x="13" y="13" width="9" height="7" rx="1.5"/></svg>';

  function addTopbarButton() {
    const top = $("#topbar .top-actions");
    if (!top || $("#ws-btn")) return;
    const b = el("button.icon-btn", {
      id: "ws-btn", title: "Workspaces (Ctrl+Shift+K)", "aria-label": "Workspaces", html: BTN_SVG,
      onclick: () => open(),
    });
    top.insertBefore(b, top.firstChild);
    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey && e.shiftKey && !e.altKey && e.code === "KeyK" && !$("#modal-root").childElementCount) {
        e.preventDefault();
        open();
      }
    });
  }

  function markSaved() {
    const b = $("#ws-btn");
    if (b) { b.classList.add("saved"); setTimeout(() => b.classList.remove("saved"), 1600); }
  }

  function open() {
    if (!S) S = app.ref.S;
    const cur = project();
    const known = store.order.filter((k) => store.workspaces[k]);
    if (cur && !known.includes(cur)) known.push(cur);
    const curName = wsName(cur);

    const rows = known.map((cwd) => {
      const ws = store.workspaces[cwd] || {};
      const meta = [
        ws.model ? `${ws.model.provider}/${ws.model.id}` : "",
        (ws.recentSessions && ws.recentSessions.length) ? `${ws.recentSessions.length + (ws.activeSessionId ? 1 : 0)} sessions` : "",
        ws.ts ? OMP.timeAgo(ws.ts) : "never opened",
      ].filter(Boolean).join(" · ");
      const row = el("button.ws-row" + (cwd === cur ? ".current" : ""), {
        title: cwd,
        onclick: () => { m.done({ action: "open", cwd }); },
      },
        el("span.ws-main", null,
          el("span.ws-name", null,
            el("span.ellipsis", { text: wsName(cwd) }),
            cwd === cur ? el("span.ws-badge", { text: "current" }) : null,
            cwd === cur && curName !== basename(cwd) ? el("span.ws-badge", { text: "renamed" }) : null),
          el("span.ws-sub", { text: short(cwd) })),
        el("span.ws-meta", { text: meta }),
        el("span.ws-actions", null,
          el("button.row-action", { title: "Rename", text: "✎", onclick: (e) => { e.stopPropagation(); m.done({ action: "rename", cwd }); } }),
          el("button.row-action", { title: "Forget this workspace", text: "✕", onclick: (e) => { e.stopPropagation(); m.done({ action: "forget", cwd }); } })));
      return row;
    });

    const list = el("div.ws-list", null, rows.length ? rows : el("div.muted.small", { text: "No workspaces yet — they are created as you use projects." }));
    const body = el("div", null, list);
    const m = OMP.dialogs.modal({
      title: "Project workspaces",
      body,
      wide: true,
      cls: "ws-dialog",
      foot: (done) => [
        el("span.ws-hint", { text: "Sessions, preview, layout, terminals, model" }),
        el("span.ws-buttons", null,
          el("button.secondary", { text: "Save now", onclick: () => { saveNow(); markSaved(); done(null); } }),
          el("button.primary", { text: "Close", onclick: () => done(null) })),
      ],
      onKey: (e) => { if (e.key === "Enter" && !e.ctrlKey && !e.shiftKey) { e.preventDefault(); if (cur) m.done({ action: "open", cwd: cur }); } },
    });

    m.promise.then(async (pick) => {
      if (!pick) return;
      if (pick.action === "open") return switchTo(pick.cwd);
      if (pick.action === "rename") {
        const name = await OMP.dialogs.promptText("Workspace name", { value: wsName(pick.cwd) }).promise;
        if (name && name.trim()) {
          if (name.trim() === basename(pick.cwd)) delete store.names[pick.cwd];
          else store.names[pick.cwd] = name.trim();
          persist();
        }
        return open(); // reopen the list
      }
      if (pick.action === "forget") {
        const ok = await OMP.dialogs.confirmBox("Forget workspace", `Forget the saved state of “${wsName(pick.cwd)}”?\n${short(pick.cwd)}`).promise;
        if (ok) {
          delete store.workspaces[pick.cwd];
          delete store.names[pick.cwd];
          store.order = store.order.filter((k) => k !== pick.cwd);
          persist();
          if (pick.cwd === cur) restoredFor = "";
        }
        return open();
      }
    });
  }

  // ── capture hooks ─────────────────────────────────────────────────────────
  function wrapArtifacts() {
    if (!OMP.artifacts || OMP.artifacts.__wsWrapped) return;
    const origOpen = OMP.artifacts.open.bind(OMP.artifacts);
    OMP.artifacts.open = (opts) => {
      if (!applying && opts && typeof opts === "object" && opts.kind === "file" && opts.path) lastPreviewFile = opts.path;
      return origOpen(opts);
    };
    OMP.artifacts.__wsWrapped = true;
    // Panel open/close is a class on the root: observe it to save promptly.
    const panelObs = new MutationObserver(() => save());
    const watch = () => {
      const p = $("#artifacts-panel");
      if (p) panelObs.observe(p, { attributes: true, attributeFilter: ["class", "style"] });
      else setTimeout(watch, 500);
    };
    watch();
  }

  function init(a) {
    app.ref = a;
    S = a.S;
    store = load();
    currentKey = project();
    addTopbarButton();
    wrapArtifacts();

    // Project switches: explicit event first, then detect the stored flag.
    window.addEventListener("omp:project", (e) => onProjectChange(e.detail && e.detail.cwd));
    window.addEventListener("hashchange", () => scheduleSave());
    window.addEventListener("omp:open-session", () => scheduleSave());

    window.addEventListener("omp:agents", (e) => {
      const list = (e.detail && (e.detail.agents || e.detail.list)) || [];
      // Startup / project switch: restore exactly once the roster is known.
      if (pendingSwitch === project() || (!restoredFor && project())) restore();
      scheduleSave();
      void list;
    });

    // Agent frames: model/thinking may have been changed via the UI; persist
    // the new values (debounced). Restore-side application is one-shot.
    window.addEventListener("omp:frame", (e) => {
      const f = e.detail && e.detail.frame;
      if (!f) return;
      if (f.type === "model_changed" || f.type === "thinking_level_changed" || f.type === "session_info") scheduleSave();
      if (f.type === "session_loaded") { restoredFor = project(); } // user navigation wins
      if (f.type === "agent_end") scheduleSave();
    });

    window.addEventListener("beforeunload", () => { if (currentKey) saveNow(); });
    document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden" && currentKey) saveNow(); });
    window.addEventListener("resize", OMP.rafThrottle(() => scheduleSave()));

    // First restore: give the shell a beat to fetch its initial roster.
    if (project()) setTimeout(() => refreshAgents().then(() => { if (!restoredFor) restore(); }), 400);
  }

  OMP.workspaces = { init, open, save: saveNow, restore, snapshot };
})(window.OMP);
