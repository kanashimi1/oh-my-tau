"use strict";
// Command palette (Ctrl/Cmd+K): commands, agents, sessions, message search
// and file-name search. Owns OMP.palette; integration adds the script tag,
// the CSS link and one OMP.palette.init(app) call after OMP.shellImpl.init.
(function (OMP) {
  let app = null;
  let agents = [];          // live agents from `omp:agents`
  let backdrop = null;
  let input = null;
  let listEl = null;
  let items = [];           // current flat item list
  let selected = 0;
  let searchSeq = 0;
  let debounceTimer = 0;

  // ---------------------------------------------------------------- commands

  function commands() {
    const s = app.S;
    const cmds = [
      {
        icon: "＋", label: "New session", hint: "current folder",
        run: () => OMP.shellImpl?.newSession?.(s, s.project),
      },
      {
        icon: "↺", label: "Resume session…", hint: "stored sessions here",
        run: () => OMP.shellImpl?.resumePicker?.(s, s.project),
      },
      {
        icon: "■", label: "Stop current agent", hint: s.agentId || "no agent",
        enabled: () => !!s.agentId,
        run: () => OMP.shellImpl?.stopAgent?.(s, s.agentId),
      },
      {
        icon: "⧉", label: "Open folder in terminal", hint: s.project,
        enabled: () => !!OMP.desktop?.openTerminal,
        run: () => OMP.desktop.openTerminal(s.project),
      },
      {
        icon: "▣", label: "Open file viewer", hint: "preview current folder",
        enabled: () => !!OMP.desktop?.openWindow,
        run: () => OMP.desktop.openWindow("#/files"),
      },
      {
        icon: "✎", label: "Rename session", hint: "current session",
        enabled: () => !!s.agentId,
        run: () => OMP.shellImpl?.renameSession?.(s),
      },
      {
        icon: "⇪", label: "Export session as HTML", hint: "",
        enabled: () => !!s.agentId,
        run: () => OMP.shellImpl?.exportHtml?.(s),
      },
      {
        icon: "?", label: "Show keyboard shortcuts", hint: "",
        run: () => OMP.shellImpl?.showHotkeys?.(),
      },
      {
        icon: "◐", label: "Switch model", hint: "",
        enabled: () => !!s.agentId,
        run: () => OMP.shellImpl?.modelPicker?.(s),
      },
      {
        icon: "⋯", label: "Thinking level", hint: "",
        enabled: () => !!s.agentId,
        run: () => OMP.shellImpl?.thinkingPicker?.(s),
      },
    ];
    // Task / change hooks from other features, when present.
    if (OMP.tasks?.open) cmds.push({ icon: "☑", label: "Tasks", hint: "", run: () => OMP.tasks.open() });
    if (OMP.changes?.open) cmds.push({ icon: "⑂", label: "Changes", hint: "", run: () => OMP.changes.open() });
    if (OMP.workspaces?.open) cmds.push({ icon: "🗔", label: "Workspaces", hint: "", run: () => OMP.workspaces.open() });
    return cmds.filter((c) => c.enabled?.() !== false);
  }

  // ------------------------------------------------------------- data source

  function baseItems() {
    const out = [];
    for (const c of commands()) {
      out.push({ kind: "cmd", icon: c.icon, label: c.label, hint: c.hint, run: c.run });
    }
    for (const a of agents) {
      out.push({
        kind: "agent", icon: "◆", label: a.sessionName || a.sessionId || a.id,
        hint: [a.model, a.streaming ? "streaming" : a.alive ? "alive" : "stopped", short(a.cwd)].filter(Boolean).join(" · "),
        run: () => app.connect(a.id),
      });
    }
    return out;
  }

  function short(p) { return p ? String(p).replace(/^\/home\/[^/]+/, "~") : p; }

  function score(item, q) {
    if (!q) return 0;
    const l = item.label.toLowerCase(), h = (item.hint || "").toLowerCase();
    if (l.startsWith(q)) return 0;
    if (l.includes(q)) return 1;
    if (h.includes(q)) return 2;
    return -1;
  }

  async function searchBackend(q, cwd) {
    const seq = ++searchSeq;
    let res;
    try {
      res = await OMP.api("/api/search?q=" + encodeURIComponent(q) + "&cwd=" + encodeURIComponent(cwd || ""));
    } catch { return; }
    if (seq !== searchSeq || !backdrop) return; // stale or closed
    const found = [];
    for (const s of res.sessions || []) {
      found.push({
        kind: "session", icon: "▤", label: s.title || s.id,
        hint: "session · " + short(s.cwd || ""),
        run: () => openStoredSession(s),
      });
    }
    for (const m of res.messages || []) {
      found.push({
        kind: "msg", icon: "💬", label: m.snippet,
        hint: (m.role || "msg") + " · " + short(m.cwd || ""),
        run: () => openStoredSession({ id: m.sessionId, cwd: m.cwd, sessionFile: m.sessionFile, agentId: m.agentId }),
      });
    }
    for (const f of res.files || []) {
      found.push({
        kind: "file", icon: "🗎", label: f.name,
        hint: short(f.dir || ""),
        run: () => openFile(f.path),
      });
    }
    mergeRemote(found);
  }

  function mergeRemote(found) {
    const q = (input.value || "").trim().toLowerCase();
    const keep = found
      .map((f) => ({ f, sc: f.label.toLowerCase().includes(q) ? 1 : 2 }))
      .filter((x) => x.sc > 0)
      .sort((a, b) => a.sc - b.sc)
      .map((x) => x.f)
      .slice(0, 18);
    // Commands and live agents first, remote hits after.
    items = baseItems().filter((it) => q ? score(it, q) >= 0 : true).concat(keep);
    selected = 0;
    render();
  }

  function openStoredSession(s) {
    const row = { id: s.id, path: s.sessionFile, title: s.title || s.sessionName, cwd: s.cwd };
    const live = agents.find((a) => a.sessionFile && a.sessionFile === row.path);
    if (live) { app.connect(live.id); return; }
    if (row.path && OMP.shellImpl?.openSession) {
      OMP.shellImpl.openSession(app.S, row);
      return;
    }
    OMP.toast("Session could not be opened: missing session file", "warning");
  }

  function openFile(path) {
    if (OMP.artifacts?.openFile) { OMP.artifacts.openFile(path); return; }
    window.open("/preview/" + encodeURIComponent(path), "_blank", "noopener");
  }

  // ------------------------------------------------------------------ render

  const KIND_LABEL = { cmd: "Commands", agent: "Agents", session: "Sessions", msg: "Messages", file: "Files" };

  function render() {
    listEl.textContent = "";
    if (!items.length) {
      listEl.append(OMP.el("div.palette-empty", { text: "No matches" }));
      return;
    }
    let lastKind = null;
    items.forEach((item, i) => {
      if (item.kind !== lastKind) {
        listEl.append(OMP.el("div.palette-kind", { text: KIND_LABEL[item.kind] || item.kind }));
        lastKind = item.kind;
      }
      const row = OMP.el("div.palette-item" + (i === selected ? ".active" : ""), {
        role: "option", "aria-selected": String(i === selected), onclick: () => pick(i),
        onmousemove: () => { if (selected !== i) { selected = i; paintSelection(); } },
      },
        OMP.el("span.palette-icon", { text: item.icon || "•" }),
        OMP.el("span.palette-label", { text: item.label }),
        item.hint ? OMP.el("span.palette-hint", { text: item.hint }) : null,
      );
      row.dataset.index = String(i);
      listEl.append(row);
    });
    paintSelection();
  }

  function paintSelection() {
    listEl.querySelectorAll(".palette-item").forEach((n) => {
      const on = Number(n.dataset.index) === selected;
      n.classList.toggle("active", on);
      n.setAttribute("aria-selected", String(on));
      if (on) n.scrollIntoView({ block: "nearest" });
    });
  }

  function pick(i) {
    const item = items[i];
    close();
    if (item) {
      try { item.run(); } catch (e) { OMP.toast?.("Palette: " + e.message, "error"); }
    }
  }

  function refresh() {
    const q = (input.value || "").trim();
    const cwd = app?.currentCwd() || "";
    if (!q) {
      searchSeq++;
      items = baseItems();
      selected = 0;
      render();
      return;
    }
    items = baseItems().filter((it) => score(it, q.toLowerCase()) >= 0);
    selected = 0;
    render();
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => searchBackend(q, cwd), 180);
  }

  function onKey(e) {
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
    if (e.key === "ArrowDown") { e.preventDefault(); selected = Math.min(selected + 1, items.length - 1); paintSelection(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); selected = Math.max(selected - 1, 0); paintSelection(); }
    else if (e.key === "Home" && items.length) { e.preventDefault(); selected = 0; paintSelection(); }
    else if (e.key === "End" && items.length) { e.preventDefault(); selected = items.length - 1; paintSelection(); }
    else if (e.key === "Enter") { e.preventDefault(); pick(selected); }
  }

  // ------------------------------------------------------------ open / close

  function open(initialQuery) {
    if (backdrop) { input?.focus(); return; }
    backdrop = OMP.el("div.palette-backdrop", {
      onclick: (e) => { if (e.target === backdrop) close(); },
    },
      OMP.el("div.palette", { role: "dialog", "aria-modal": "true", "aria-label": "Command palette" },
        OMP.el("div.palette-input-row", {},
          OMP.el("span.palette-glyph", { text: "⌕" }),
          input = OMP.el("input.palette-input", {
            type: "text", placeholder: "Search commands, sessions, messages, files…",
            spellcheck: "false", oninput: () => refresh(),
          }),
          OMP.el("kbd.palette-kbd", { text: "esc" }),
        ),
        listEl = OMP.el("div.palette-list", { role: "listbox", id: "palette-listbox" }),
      ),
    );
    input.setAttribute("aria-controls", "palette-listbox");
    input.setAttribute("aria-autocomplete", "list");
    document.body.append(backdrop);
    input.value = initialQuery || "";
    refresh();
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
    input.addEventListener("keydown", onKey);
    searchBackendDebounced(initialQuery);
  }

  // If opened with a query, hit the backend right away too.
  function searchBackendDebounced(q) {
    if (!q) return;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => searchBackend(q, app?.currentCwd() || ""), 120);
  }

  function close() {
    clearTimeout(debounceTimer);
    searchSeq++;
    backdrop?.remove();
    backdrop = input = listEl = null;
    items = [];
  }

  // -------------------------------------------------------------------- init

  OMP.palette = {
    init(a) {
      app = a;
      window.addEventListener("omp:agents", (e) => {
        const d = e.detail || {};
        agents = d.list || d.agents || [];
      });
      // Seed once so the palette works before the first `omp:agents`.
      OMP.api("/api/agents").then((r) => { agents = r.agents || r.list || []; }).catch(() => {});
      window.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === "k") {
          e.preventDefault();
          if (backdrop) close(); else open();
        }
      });
      const topbar = document.getElementById("topbar");
      if (topbar) {
        const btn = OMP.el("button.icon-btn.palette-btn", {
          title: "Command palette (Ctrl+K)", "aria-label": "Command palette",
          onclick: () => open(),
        }, OMP.el("span", { text: "⌕" }), OMP.el("kbd", { text: "Ctrl K" }));
        const actions = topbar.querySelector(".top-actions");
        if (actions) actions.prepend(btn); else topbar.append(btn);
      }
    },
    open,
    close,
  };
})(window.OMP);
