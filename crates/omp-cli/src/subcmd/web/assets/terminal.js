"use strict";
// Integrated terminal: a resizable bottom panel with real shell tabs (bash /
// WSL / PowerShell) driven by /api/terminal/*. Output is polled only while the
// panel is visible, incrementally by cursor, and trimmed client-side; only
// tab *definitions* (shell/cwd/title) are persisted, never output.
window.OMP = window.OMP || {};
(function (OMP) {
  const { el, $, api } = OMP;
  const POLL_MS = 350;
  const MAX_TEXT = 200000;   // client-side output cap per tab (chars)
  const DEFS_KEY = "omp-terminal-defs";

  let app = null;
  let btn = null;
  let panel = null;
  let tabsEl = null;
  let bodyEl = null;
  let sysEl = null;
  let statusEl = null;
  let pollTimer = null;
  let activeId = null;
  const tabs = new Map();    // id -> {id, shell, cwd, title, out, cursor, running, exitCode, done}
  const history = [];        // per-tab-less input history
  let histPos = -1;

  const S = () => (typeof app.S === "function" ? app.S() : app.S);

  // ── definitions (serialize/restore; never output) ─────────────────────────
  function serialize() {
    const defs = [...tabs.values()].map((t) => ({ shell: t.shell, cwd: t.cwd, title: t.title }));
    try { localStorage.setItem(DEFS_KEY, JSON.stringify(defs)); } catch { /* ignore */ }
    return defs;
  }

  function restore(defs) {
    if (!Array.isArray(defs)) {
      try { defs = JSON.parse(localStorage.getItem(DEFS_KEY) || "[]"); } catch { defs = []; }
    }
    if (!defs || !defs.length) return Promise.resolve();
    return Promise.all(
      defs.slice(0, 8).map((d) =>
        d && d.cwd && d.shell ? createTab(d.shell, d.cwd, d.title).catch(() => {}) : null)
    );
  }

  // ── tabs ──────────────────────────────────────────────────────────────────
  function defaultCwd() {
    const s = S();
    return s.project || s.info?.defaultCwd || s.info?.home || "";
  }

  async function createTab(shell, cwd, title) {
    cwd = cwd || defaultCwd();
    const t = await api("/api/terminal/create", { method: "POST", body: { shell, cwd, title } });
    const view = {
      id: t.id, shell: t.shell, cwd: t.cwd, title: t.title || t.shell,
      out: el("pre.term-out", { tabindex: "0", "aria-label": "terminal output" }),
      cursor: 0, running: true, exitCode: null, done: false,
    };
    tabs.set(t.id, view);
    renderTabs();
    activate(t.id);
    serialize();
    return view;
  }

  function closeTab(id, silent) {
    const t = tabs.get(id);
    if (!t) return;
    if (!silent) api("/api/terminal/kill", { method: "POST", body: { id } }).catch(() => {});
    tabs.delete(id);
    if (activeId === id) {
      activeId = null;
      const next = [...tabs.keys()][0];
      if (next) activate(next); else renderTabs();
    } else renderTabs();
    serialize();
  }

  function activate(id) {
    activeId = id;
    const t = tabs.get(id);
    if (!t) return;
    for (const v of tabs.values()) if (v.out.parentNode) v.out.remove();
    bodyEl.insertBefore(t.out, bodyEl.firstChild);
    renderTabs();
    pollOnce(t).catch(() => {});
    t.out.scrollIntoView({ block: "end" });
    const input = $(".term-input", panel);
    if (input) input.focus();
  }

  function sendInput(t, data) {
    if (!t || t.done) return;
    api("/api/terminal/input", { method: "POST", body: { id: t.id, data } })
      .catch((e) => OMP.toast("Terminal input failed: " + e.message, "error"));
  }

  async function pollOnce(t) {
    const res = await api(`/api/terminal/output?id=${encodeURIComponent(t.id)}&cursor=${t.cursor}`);
    t.running = !!res.running;
    t.exitCode = res.exitCode ?? null;
    if (res.truncated) {
      // Server dropped the requested range: show the retained tail only.
      t.out.textContent = "…[earlier output dropped]\n" + res.data;
    } else {
      t.out.textContent += res.data;
      const txt = t.out.textContent;
      if (txt.length > MAX_TEXT) t.out.textContent = txt.slice(txt.length - MAX_TEXT);
      const nearBottom = t.out.scrollHeight - t.out.scrollTop - t.out.clientHeight < 60;
      if (nearBottom || !t.done) t.out.scrollTop = t.out.scrollHeight;
    }
    t.cursor = res.nextCursor;
    if (!res.running && !t.done) {
      t.done = true;
      t.out.textContent += `\n[process exited${res.exitCode != null ? ` code ${res.exitCode}` : ""}]\n`;
    }
    renderTabs();
  }

  function pollLoop() {
    if (!panel || panel.hidden) return;
    const t = tabs.get(activeId);
    if (t && document.visibilityState === "visible") pollOnce(t).catch(() => {});
  }

  // ── rendering ─────────────────────────────────────────────────────────────
  function renderTabs() {
    if (!tabsEl) return;
    const home = S().info?.home || "";
    tabsEl.replaceChildren(
      ...[...tabs.values()].map((t) =>
        el("button.term-tab" + (t.id === activeId ? ".active" : ""), {
          title: `${t.title} — ${OMP.shortPath(t.cwd, home)}`,
          onclick: () => activate(t.id),
        },
          el("span.term-tab-dot." + (t.done ? "dead" : "live"), { "aria-hidden": "true" }),
          el("span", { text: t.title }),
          el("span.term-tab-x", {
            text: "×", role: "button", "aria-label": `Close ${t.title}`,
            onclick: (e) => { e.stopPropagation(); closeTab(t.id); },
          }))),
      el("button.term-tab.term-new", {
        text: "+ ▾", title: "New terminal (bash, WSL, PowerShell)",
        "aria-label": "New terminal",
        onclick: (e) => OMP.popupMenu(e.currentTarget, [
          { label: "bash / zsh", run: () => createTab("bash").catch((err) => OMP.toast(err.message, "error")) },
          { label: "WSL", run: () => createTab("wsl").catch((err) => OMP.toast(err.message, "error")) },
          { label: "PowerShell", run: () => createTab("powershell").catch((err) => OMP.toast(err.message, "error")) },
          "-",
          { label: "Restore saved tabs", run: () => restore() },
        ]),
      }),
    );
    const t = tabs.get(activeId);
    if (statusEl) {
      const st = t
        ? (t.running ? "running" : `exited${t.exitCode != null ? ` (${t.exitCode})` : ""}`) + " · " + OMP.shortPath(t.cwd, home)
        : "no terminal";
      statusEl.textContent = st;
    }
    updateCwd();
  }

  function errorLines(text) {
    return text.split("\n").filter((l) =>
      /(^|\s)(error|ERROR|Error|panic|FAILED|failed|Traceback|error:|warning:)/.test(l)).join("\n");
  }

  async function showSystem() {
    if (!sysEl) return;
    sysEl.hidden = !sysEl.hidden;
    if (sysEl.hidden) return;
    sysEl.replaceChildren(el("div.term-sys-state", { text: "Loading processes and ports…" }));
    try {
      const res = await api("/api/terminal/system");
      const procs = (res.processes || []).filter((p) => p.ours);
      const ports = res.ports || [];
      sysEl.replaceChildren(
        el("h3", { text: `Terminal child processes (${procs.length})` }),
        procs.length
          ? el("ul", null, procs.map((p) =>
            el("li", { text: `${p.pid}  ${p.comm}` })))
          : el("p.term-sys-state", { text: "No terminal child processes." }),
        el("h3", { text: `Listening TCP ports (${ports.length})` }),
        ports.length
          ? el("ul", null, ports.slice(0, 200).map((p) =>
            el("li", { text: `${p.port}${p.v6 ? " (v6)" : ""}` })))
          : el("p.term-sys-state", { text: "No listening TCP ports." }),
      );
    } catch (e) {
      sysEl.replaceChildren(el("p.term-sys-state", { text: "Failed: " + e.message }));
    }
  }

  function buildPanel() {
    if (panel) return;
    panel = el("section.term-panel", { id: "terminal-panel", hidden: true, role: "region", "aria-label": "Integrated terminal" },
      el("div.term-resize", {
        title: "Drag to resize", onmousedown: startResize,
        ondblclick: () => { panel.style.height = ""; },
      }),
      el("div.term-head", null,
        el("div.term-tabs", { role: "tablist" }),
        el("span", { style: "flex:1" }),
        el("button.icon-btn", {
          text: "⧉ Out", title: "Copy terminal output into the composer",
          onclick: () => copyToComposer(false),
        }),
        el("button.icon-btn", {
          text: "⧉ Err", title: "Copy error lines into the composer",
          onclick: () => copyToComposer(true),
        }),
        el("button.icon-btn", { text: "⚙", title: "Running terminal processes & listening ports", onclick: showSystem }),
        el("button.icon-btn", {
          text: "↻", title: "Restart this terminal (same shell/cwd)",
          onclick: () => restart(),
        }),
        el("button.icon-btn", {
          text: "■", title: "Stop this terminal's shell",
          onclick: () => { const t = tabs.get(activeId); if (t) closeTab(t.id); },
        }),
        el("button.icon-btn", { text: "✕", title: "Hide terminal panel", onclick: close }),
      ),
      el("div.term-body", null),
      el("form.term-input-row", {
        onsubmit: (e) => {
          e.preventDefault();
          const t = tabs.get(activeId);
          const input = $(".term-input", panel);
          const line = input.value;
          if (!t || !line) return;
          sendInput(t, line + "\n");
          history.push(line);
          if (history.length > 100) history.shift();
          histPos = -1;
          input.value = "";
        },
      },
        el("span.term-cwd", { title: "Working directory (click to change)", onclick: changeCwd, text: "" }),
        el("input.term-input", {
          type: "text", placeholder: "Type a command…", "aria-label": "Terminal input",
          autocomplete: "off", spellcheck: "false",
          onkeydown: (e) => {
            if (e.key === "ArrowUp" && !history.length) return;
            if (e.key === "ArrowUp") {
              e.preventDefault();
              histPos = histPos < 0 ? history.length - 1 : Math.max(0, histPos - 1);
              e.target.value = history[histPos] || "";
            } else if (e.key === "ArrowDown" && histPos >= 0) {
              e.preventDefault();
              histPos++;
              if (histPos >= history.length) { histPos = -1; e.target.value = ""; }
              else e.target.value = history[histPos];
            }
          },
        })),
      el("div.term-status", null),
      el("aside.term-sys", { hidden: true, "aria-label": "System processes and ports" }),
    );
    document.body.append(panel);
    tabsEl = $(".term-tabs", panel);
    bodyEl = $(".term-body", panel);
    statusEl = $(".term-status", panel);
    sysEl = $(".term-sys", panel);
    updateCwd();
  }

  function updateCwd() {
    const cwdEl = panel && $(".term-cwd", panel);
    if (!cwdEl) return;
    const t = tabs.get(activeId);
    cwdEl.textContent = t ? OMP.shortPath(t.cwd, S().info?.home || "") : "—";
  }

  function startResize(e) {
    e.preventDefault();
    const startY = e.clientY;
    const startH = panel.offsetHeight;
    const move = (ev) => {
      const h = Math.min(window.innerHeight - 120, Math.max(120, startH + (startY - ev.clientY)));
      panel.style.height = h + "px";
    };
    const up = () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
    };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  }

  function activeText() {
    const t = tabs.get(activeId);
    return t ? t.out.textContent : "";
  }

  function copyToComposer(errorsOnly) {
    const full = activeText();
    const text = errorsOnly ? errorLines(full) : full.slice(-8000);
    if (!text.trim()) { OMP.toast("Nothing to copy", "info"); return; }
    const body = (errorsOnly ? "Terminal errors:\n```\n" : "Terminal output:\n```\n")
      + text + "\n```\n";
    app.composer.setText(body);
    OMP.toast(errorsOnly ? "Error lines added to composer" : "Output added to composer", "success");
  }

  async function restart() {
    const t = tabs.get(activeId);
    if (!t) return;
    const def = { shell: t.shell, cwd: t.cwd, title: t.title };
    closeTab(t.id);
    try { await createTab(def.shell, def.cwd, def.title); }
    catch (e) { OMP.toast("Restart failed: " + e.message, "error"); }
  }

  function changeCwd() {
    const t = tabs.get(activeId);
    if (!t) return;
    OMP.dialogs.promptText("Working directory", { value: t.cwd }).promise.then((v) => {
      if (!v || v === t.cwd) return;
      // Restart the shell in the new directory (spawn-time cwd is authoritative).
      closeTab(t.id);
      createTab(t.shell, v, t.title).catch((e) => OMP.toast(e.message, "error"));
    });
  }

  // ── open/close ────────────────────────────────────────────────────────────
  function open() {
    buildPanel();
    panel.hidden = false;
    if (btn) btn.setAttribute("aria-pressed", "true");
    if (!tabs.size) createTab("bash").catch((e) => OMP.toast(e.message, "error"));
    clearInterval(pollTimer);
    pollTimer = setInterval(pollLoop, POLL_MS);
    const input = $(".term-input", panel);
    if (input) input.focus();
  }

  function close() {
    if (!panel) return;
    panel.hidden = true;
    if (btn) btn.setAttribute("aria-pressed", "false");
    clearInterval(pollTimer);
    pollTimer = null;
    if (sysEl) sysEl.hidden = true;
  }

  // ── init ──────────────────────────────────────────────────────────────────
  function init(a) {
    app = a;
    const host = $("#topbar .top-actions");
    btn = el("button.icon-btn.term-btn", {
      id: "terminal-btn",
      title: "Integrated terminal",
      "aria-label": "Integrated terminal",
      "aria-pressed": "false",
      onclick: () => (panel && !panel.hidden ? close() : open()),
      html: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2.5" y="4" width="19" height="16" rx="2"/><path d="M6.5 9.5l3 2.5-3 2.5M12.5 15h5"/></svg>',
    });
    host.prepend(btn);
    window.addEventListener("beforeunload", serialize);
    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey && e.key === "`" && !OMP.dialogs.isOpen()) {
        e.preventDefault();
        panel && !panel.hidden ? close() : open();
      }
    });
  }

  OMP.terminal = { init, open, close, serialize, restore };
})(window.OMP);
