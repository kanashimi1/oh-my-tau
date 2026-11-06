"use strict";
// Tau — session panes (OMP.panes): tmux-style multiplexing of agent sessions
// in ONE window. Each pane is an <iframe src="/?agent=<id>&pane=1"> — the app
// inside keeps its own WebSocket (/ws/<id>), so panes are fully independent
// session views with zero refactoring of the app singleton.
//
//   OMP.panes.init(app)   — mount the topbar button, restore saved layout
//   OMP.panes.enter()     — open the pane workspace
//   OMP.panes.exit()      — back to the normal single-session view
//
// Keys (tmux-style prefix, capture phase so iframes never swallow them):
//   Ctrl+B c  new pane (agent/session picker)
//   Ctrl+B %  tile columns   Ctrl+B quote  tile rows
//   Ctrl+B n / p  next / previous pane     Ctrl+B 1-9  pane by index
//   Ctrl+B x  close active pane            Ctrl+B d  exit panes
// Bare Ctrl+B (no follow-up key in 800 ms) still toggles the sidebar.
(function (OMP) {
  const { el, $ } = OMP;
  const STORE_KEY = "omp-panes";
  const PREFIX_MS = 800;

  let app = null;
  let root = null, strip = null, grid = null, exitBtn = null, newBtn = null, flowBtn = null;
  let panes = [];          // [{ agentId, frame, tab }]
  let active = -1;
  let flow = "column";     // "column" | "row" — how panes tile
  let prefixTimer = 0, armed = false;
  let btn = null;

  // ── persistence ──────────────────────────────────────────────────────
  function save() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify({ ids: panes.map((p) => p.agentId), flow })); } catch { /* storage unavailable */ }
  }
  function load() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch { return {}; }
  }

  // ── api helpers ──────────────────────────────────────────────────────
  async function jget(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error("GET " + url + " → " + r.status);
    return r.json();
  }
  async function jpost(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    if (!r.ok) throw new Error("POST " + url + " → " + r.status);
    return r.json();
  }

  function isActive() { return !!root && document.body.classList.contains("panes-on"); }

  // ── layout ───────────────────────────────────────────────────────────
  function reflow() {
    if (!grid) return;
    const n = Math.max(panes.length, 1);
    if (flow === "column") {
      grid.style.gridTemplateColumns = "repeat(" + n + ", 1fr)";
      grid.style.gridTemplateRows = "1fr";
    } else {
      grid.style.gridTemplateColumns = "1fr";
      grid.style.gridTemplateRows = "repeat(" + n + ", 1fr)";
    }
    panes.forEach((p, i) => { p.tab.classList.toggle("active", i === active); p.frame.classList.toggle("active", i === active); });
    if (active < 0 && panes.length) setActive(0, false);
  }

  function setActive(i, focus) {
    active = i;
    panes.forEach((p, k) => { p.tab.classList.toggle("active", k === i); p.frame.classList.toggle("active", k === i); });
    if (focus && panes[i]) { try { panes[i].frame.focus({ preventScroll: true }); } catch { /* detached frame */ } }
  }

  function label(agentId) {
    const a = ((app && app.S && app.S.agents) || []).find((x) => x.id === agentId);
    if (a) {
      const base = (a.cwd || "").split("/").filter(Boolean).pop() || "";
      return a.role || base || agentId.slice(0, 8);
    }
    return agentId.slice(0, 8);
  }

  function renderStrip() {
    strip.replaceChildren(newBtn, flowBtn, ...panes.map((p) => p.tab), exitBtn);
  }

  function addPane(agentId) {
    if (!agentId) return;
    const have = panes.findIndex((p) => p.agentId === agentId);
    if (have !== -1) { setActive(have, true); return; }
    const frame = el("iframe.pane-frame", {
      src: "/?agent=" + encodeURIComponent(agentId) + "&pane=1",
      title: "Session pane " + agentId,
    });
    frame.addEventListener("pointerenter", () => setActive(panes.findIndex((p) => p.frame === frame), false));
    const tab = el("button.pane-tab", {
      onclick: () => setActive(panes.findIndex((p) => p.tab === tab), true),
    }, el("span.pane-tab-label", { text: label(agentId) }), el("span.pane-tab-x", {
      text: "×", title: "Close pane",
      onclick: (e) => { e.stopPropagation(); closePane(panes.findIndex((p) => p.tab === tab)); },
    }));
    panes.push({ agentId, frame, tab });
    renderStrip();
    grid.appendChild(frame);
    setActive(panes.length - 1, false);
    reflow(); save();
  }

  function closePane(i) {
    const p = panes[i];
    if (!p) return;
    p.frame.remove(); p.tab.remove();
    panes.splice(i, 1);
    renderStrip();
    if (!panes.length) { exit(); return; }
    setActive(Math.min(i, panes.length - 1), false);
    reflow(); save();
  }

  // ── picker ───────────────────────────────────────────────────────────
  async function picker() {
    let agents = [], sessions = [];
    try { agents = (await jget("/api/agents")).agents || []; } catch { /* service hiccup: picker still offers new session */ }
    try { sessions = (await jget("/api/sessions")).sessions || []; } catch { /* optional */ }
    const live = agents.filter((a) => !panes.some((p) => p.agentId === a.id));
    const stored = sessions.slice(0, 30);
    const body = el("div.pane-picker",
      el("div.pane-sec", null, el("div.pane-sec-title", { text: "Live agents" }),
        live.length ? live.map((a) => el("button.pane-row", {
          onclick: () => { modal.done(null); addPane(a.id); },
        }, el("span.pane-dot"), el("span.pane-row-main", null,
          el("span.pane-row-title", { text: a.role || a.id.slice(0, 12) }),
          el("span.pane-row-sub", { text: [a.model, (a.cwd || "").split("/").filter(Boolean).pop()].filter(Boolean).join(" · ") }))))
          : el("div.pane-empty", { text: "No other live agents." })),
      el("div.pane-sec", null, el("div.pane-sec-title", { text: "Recent sessions" }),
        stored.length ? stored.map((s) => el("button.pane-row", {
          onclick: async () => {
            modal.done(null);
            try {
              const a = await jpost("/api/agents", { cwd: s.cwd || undefined, resume: s.file || s.id });
              addPane(a.id || (a.agent && a.agent.id));
            } catch (e) { OMP.toast("Could not resume session: " + (e.message || e), "error"); }
          },
        }, el("span.pane-row-main", null,
          el("span.pane-row-title", { text: s.title || s.file || s.id || "session" }),
          el("span.pane-row-sub", { text: s.when || s.cwd || "" }))))
          : el("div.pane-empty", { text: "No stored sessions." })),
      el("div.pane-sec", null,
        el("button.primary.wide", { text: "+ New session pane", onclick: async () => {
          modal.done(null);
          try {
            const a = await jpost("/api/agents", {});
            addPane(a.id || (a.agent && a.agent.id));
          } catch (e) { OMP.toast("Could not start a session: " + (e.message || e), "error"); }
        } })));
    const modal = OMP.dialogs.modal({ title: "Open pane", cls: "pane-modal", body });
  }

  // ── enter / exit ─────────────────────────────────────────────────────
  function build() {
    exitBtn = el("button.pane-exit", { text: "✕ panes", title: "Exit panes (Ctrl+B d)", onclick: exit });
    newBtn = el("button.pane-new", { text: "+ pane", title: "New pane (Ctrl+B c)", onclick: () => picker() });
    flowBtn = el("button.pane-flow", {
      text: "⇉",
      title: "Tile direction: columns ↔ rows (Ctrl+B % or Ctrl+B quote)",
      onclick: () => { flow = flow === "column" ? "row" : "column"; reflow(); save(); },
    });
    strip = el("div.pane-strip");
    grid = el("div.pane-grid");
    root = el("div#panes-root", null, strip, grid);
    document.body.appendChild(root);
    renderStrip();
  }

  function enter() {
    if (isActive()) return;
    if (!root) build();
    document.body.classList.add("panes-on");
    if (btn) btn.setAttribute("aria-pressed", "true");
    renderStrip();
    reflow();
  }
  function exit() {
    document.body.classList.remove("panes-on");
    if (btn) btn.setAttribute("aria-pressed", "false");
    save();
  }
  function toggle() { if (isActive()) exit(); else enter(); }

  // ── keyboard (capture phase, above the iframes) ──────────────────────
  function onKey(e) {
    if (e.key.toLowerCase() !== "b" || !e.ctrlKey || e.shiftKey || e.altKey || e.metaKey) {
      if (armed && (e.ctrlKey || e.key === "Escape")) handleFollow(e);
      return;
    }
    // Never steal Ctrl+B from text fields (cursor-back in inputs/composer).
    const t = e.target;
    if (t && (/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) || t.isContentEditable)) return;
    // Own the prefix completely: shell.js must not toggle the sidebar on the
    // same press — the 800 ms fallback below replays that click when the
    // user pressed a bare Ctrl+B.
    e.preventDefault();
    e.stopPropagation();
    if (armed) return; // key auto-repeat while armed
    armed = true;
    clearTimeout(prefixTimer);
    prefixTimer = setTimeout(() => {
      armed = false;
      // Bare Ctrl+B keeps its old meaning: sidebar toggle (panes closed).
      if (!isActive()) {  b = $("#sidebar-hide"); if (b) b.click(); }
    }, PREFIX_MS);
  }
  function handleFollow(e) {
    const k = e.key.toLowerCase();
    let used = true;
    if (!isActive()) {
      if (k === "c") enter(); else used = false;
    } else if (k === "c") picker();
    else if (k === "%") { flow = "column"; reflow(); save(); }
    else if (e.code === "Quote") { flow = "row"; reflow(); save(); }
    else if (k === "n" && panes.length) setActive((active + 1) % panes.length, true);
    else if (k === "p" && panes.length) setActive((active - 1 + panes.length) % panes.length, true);
    else if (k === "x") closePane(active);
    else if (k === "d") exit();
    else if (/^[1-9]$/.test(k) && panes[Number(k) - 1]) setActive(Number(k) - 1, true);
    else used = false;
    if (used) { e.preventDefault(); clearTimeout(prefixTimer); armed = false; }
  }

  // ── init ─────────────────────────────────────────────────────────────
  function init(a) {
    app = a;
    const host = $("#topbar .top-actions");
    if (host) {
      btn = el("button.icon-btn.panes-btn", {
        title: "Session panes — tmux-style multi-session view (Ctrl+B c)",
        "aria-label": "Session panes",
        "aria-pressed": "false",
        onclick: toggle,
      }, el("span", { text: "▤" }));
      host.appendChild(btn);
    }
    window.addEventListener("keydown", onKey, true);
    // Restore the last layout (tmux attach).
    const saved = load();
    if (saved.flow === "row" || saved.flow === "column") flow = saved.flow;
    if (Array.isArray(saved.ids) && saved.ids.length) {
      saved.ids.forEach((id) => { if (typeof id === "string" && id) addPane(id); });
      enter();
    }
  }

  OMP.panes = { init, enter, exit, toggle, addPane };
})(window.OMP);
