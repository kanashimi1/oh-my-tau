"use strict";
// Change center: changed files of the current git project with selectable
// unified diffs, open-in-preview actions, and per-file checkpoint / revert
// (the backend always snapshots before reverting). Also shows this session's
// command / test executions observed in `omp:frame` tool events.
window.OMP = window.OMP || {};
(function (OMP) {
  const { el, $, api } = OMP;

  let app = null;
  let modal = null;          // open OMP.dialogs.modal handle
  let btn = null;
  let listEl = null;         // file list container
  let diffEl = null;         // diff pane
  let actEl = null;          // command/test activity container
  let stateEl = null;
  let selected = null;       // selected file path (repo-relative)
  let files = [];            // last /api/changes payload.files
  const activity = new Map(); // tool call id -> {cmd, kind, status, output, error, at}
  let activitySeq = 0;

  const cwd = () => app?.currentCwd() || "";
  const isOpen = () => !!modal;

  // ── tool / test activity from frames ───────────────────────────────────────
  const TEST_RE = /(^|[\s"'`(&|;])(cargo\s+(test|nextest)|npm(\s+\S+)?\s+(run\s+)?test|pnpm\s+(run\s+)?test|yarn\s+(run\s+)?test|vitest|jest|pytest|python\s+-m\s+pytest|go\s+test|make\s+test|dotnet\s+test|mvn\s+test|gradle(\s+\S+)?\s+test|rspec|mix\s+test|swift\s+test)\b/i;

  function isTestCommand(cmd) {
    return TEST_RE.test(String(cmd || ""));
  }

  /** Extract tool_execution_start / tool_execution_end info from a frame.
   *  Tolerant of several frame shapes; returns [{id, name, command, status, output, error}]. */
  function extractTools(frame) {
    const out = [];
    if (!frame || typeof frame !== "object") return out;
    const visit = (n, depth) => {
      if (!n || typeof n !== "object" || depth > 6) return;
      if (Array.isArray(n)) { n.forEach((x) => visit(x, depth + 1)); return; }
      const type = n.type || n.event || n.kind || "";
      if (typeof type === "string" && type.startsWith("tool_execution")) {
        const start = type === "tool_execution_start" || type.endsWith("_start");
        const end = type === "tool_execution_end" || type.endsWith("_end");
        const id = n.id || n.call_id || n.callId || n.tool_call_id || `evt-${++activitySeq}`;
        const tool = n.tool || n.name || n.tool_name || (n.tool_call && n.tool_call.name) || "tool";
        const input = n.input ?? n.arguments ?? n.args ?? (n.tool_call && n.tool_call.input) ?? null;
        const command = typeof input === "string" ? input
          : input && typeof input === "object"
            ? (input.command ?? input.cmd ?? input.argv ?? (Array.isArray(input.args) ? input.args.join(" ") : ""))
            : "";
        const entry = activity.get(id) || {
          id, cmd: "", kind: "command", status: "running", output: "", error: "", at: Date.now(),
        };
        if (start) {
          entry.cmd = String(command || tool);
          entry.kind = isTestCommand(entry.cmd) ? "test" : "command";
          entry.status = "running";
          entry.at = Date.now();
        }
        if (end) {
          entry.status = (n.error || n.is_error || n.success === false) ? "error" : "ok";
          const o = n.output ?? n.result ?? "";
          entry.output = typeof o === "string" ? o : JSON.stringify(o);
          entry.error = typeof n.error === "string" ? n.error : "";
        }
        activity.set(id, entry);
        out.push(entry);
        return;
      }
      for (const v of Object.values(n)) visit(v, depth + 1);
    };
    visit(frame, 0);
    return out;
  }

  /** Called for every omp:frame event; records command/test activity. */
  function recordFrame(evt) {
    const frame = evt && evt.frame ? evt.frame : evt && evt.detail && evt.detail.frame;
    if (!extractTools(frame || evt).length) return;
    if (isOpen() && actEl) renderActivity();
  }

  // ── api ────────────────────────────────────────────────────────────────────
  async function fetchChanges(path) {
    const q = new URLSearchParams({ cwd: cwd() });
    if (path) q.set("path", path);
    return api("/api/changes?" + q);
  }

  async function refresh() {
    if (!cwd()) return;
    try {
      const data = await fetchChanges();
      files = data.files || [];
      if (isOpen()) render();
    } catch (e) {
      files = [];
      if (isOpen()) render();
      OMP.toast("Could not list changes: " + (e.message || e), "error");
    }
  }

  async function showDiff(path) {
    selected = path;
    if (!diffEl) { if (isOpen()) render(); return; }
    diffEl.replaceChildren(el("p.changes-state", { text: "Loading diff…" }));
    renderList();
    try {
      const d = await fetchChanges(path);
      const stat = `${d.additions || 0} additions, ${d.deletions || 0} deletions` +
        (d.truncated ? " (truncated)" : "");
      diffEl.replaceChildren(
        el("div.diff-head", null,
          el("span.diff-path", { text: path }),
          el("span.diff-stat", { text: stat })),
        el("pre.diff-body", null, d.diff || "(no textual diff)"),
      );
      markSelected();
    } catch (e) {
      diffEl.replaceChildren(el("p.changes-state.error", { text: "Diff failed: " + (e.message || e) }));
    }
  }

  function openFile(path) {
    const f = files.find((x) => x.path === path);
    const abs = f && f.abs ? f.abs : path;
    if (OMP.artifacts && typeof OMP.artifacts.openFile === "function") {
      OMP.artifacts.openFile(abs);
    } else {
      window.dispatchEvent(new CustomEvent("omp:open-file", { detail: { path: abs } }));
    }
  }

  async function checkpointFile(path) {
    try {
      const r = await api("/api/changes/checkpoint", { method: "POST", body: { cwd: cwd(), path } });
      OMP.toast(`Checkpoint ${r.checkpoint} created for ${path}`, "success");
    } catch (e) {
      OMP.toast("Checkpoint failed: " + (e.message || e), "error");
    }
  }

  async function revertFile(path) {
    const f = files.find((x) => x.path === path);
    const kind = f && f.untracked ? "untracked (a backup is kept in the checkpoint)" : "tracked";
    const ok = await OMP.dialogs.confirmBox("Revert file",
      `Revert “${path}” (${kind})? A checkpoint of its current content is created first, so this can be undone.`).promise;
    if (!ok) return;
    let undo = null;
    try {
      const c = await api("/api/changes/checkpoint", { method: "POST", body: { cwd: cwd(), path } });
      undo = c.checkpoint || null;
    } catch (e) {
      OMP.toast("Checkpoint failed: " + (e.message || e), "error");
    }
    try {
      const r = await api("/api/changes/revert", { method: "POST", body: { cwd: cwd(), paths: [path] } });
      const refused = (r.refused || []).includes(path);
      const done = (r.reverted || []).includes(path);
      if (refused || !done) {
        OMP.toast("Revert failed: " + (r.error || path + " refused (untracked or unchanged)"), "error");
        return;
      }
      OMP.toast(`Reverted ${path}${undo ? ` (undo checkpoint ${undo})` : ""}`, "success");
      if (selected === path) selected = null;
      await refresh();
      if (diffEl) diffEl.replaceChildren(el("p.changes-state", { text: "Select a file to see its diff." }));
    } catch (e) {
      OMP.toast("Revert failed: " + (e.message || e), "error");
    }
  }

  // ── review decisions ──────────────────────────────────────────────────────
  const DECISIONS_KEY = "omp-review-decisions";
  function decisions() {
    try { return JSON.parse(localStorage.getItem(DECISIONS_KEY)) || {}; } catch { return {}; }
  }
  function decisionKey(path) { return (cwd() || "") + "::" + path; }
  function isApproved(path) { return decisions()[decisionKey(path)] === true; }
  function setApproved(path, on) {
    const d = decisions();
    if (on) d[decisionKey(path)] = true; else delete d[decisionKey(path)];
    try { localStorage.setItem(DECISIONS_KEY, JSON.stringify(d)); } catch { /* storage unavailable */ }
  }
  /** Pending-review counter on the topbar Changes button. */
  function updateBadge() {
    if (!btn) return;
    const pending = files.filter((f) => !isApproved(f.path)).length;
    if (pending > 0) {
      btn.setAttribute("data-badge", String(pending));
      btn.title = `Changes (git diff, checkpoints) — ${pending} to review`;
    } else {
      btn.removeAttribute("data-badge");
      btn.title = "Changes (git diff, checkpoints)";
    }
  }
// ── rendering ──────────────────────────────────────────────────────────────
  function markSelected() {
    if (!listEl) return;
    listEl.querySelectorAll(".chg-row").forEach((r) =>
      r.classList.toggle("active", r.dataset.path === selected));
  }

  function renderList() {
    if (!listEl) return;
    if (!files.length) {
      listEl.replaceChildren(el("p.changes-state", { text: cwd() ? "No changes." : "No project folder." }));
      return;
    }
    listEl.replaceChildren(...files.map((f) => {
      const approved = isApproved(f.path);
      return el("button.chg-row" + (approved ? " approved" : ""), {
        dataset: { path: f.path },
        onclick: () => showDiff(f.path),
      },
        el("span.chg-code." + statusClass(f), { text: f.code || "M", title: f.status }),
        el("span.chg-path", { text: f.path, title: f.abs || f.path }),
        el("span.chg-actions", { onclick: (e) => e.stopPropagation() },
          el("button.chip", { text: approved ? "Unapprove" : "Approve", title: "Approve = reviewed, keep this change", onclick: () => { setApproved(f.path, !approved); renderList(); } }),
          el("button.chip", { text: "Open", title: "Open file", onclick: () => openFile(f.path) }),
          el("button.chip", { text: "Checkpoint", title: "Back up this file now", onclick: () => checkpointFile(f.path) }),
          el("button.chip.danger", { text: "Revert", title: "Revert to HEAD (with checkpoint)", onclick: () => revertFile(f.path) }),
        ),
      );
    }));
    markSelected();
    updateBadge();
  }

  function statusClass(f) {
    if (f.untracked) return "untracked";
    if (f.code === "D" || String(f.code).includes("D")) return "deleted";
    if (f.code === "A" || String(f.code).includes("A")) return "added";
    return "modified";
  }

  function renderActivity() {
    if (!actEl) return;
    const items = [...activity.values()].sort((a, b) => b.at - a.at).slice(0, 50);
    if (!items.length) {
      actEl.replaceChildren(el("p.changes-state", { text: "No commands run in this session yet." }));
      return;
    }
    actEl.replaceChildren(...items.map((a) =>
      el("div.act-row." + a.status,
        el("span.act-dot", { title: a.status }),
        el("div.act-main", null,
          el("div.act-line", null,
            a.kind === "test" ? el("span.act-kind", { text: "test" }) : null,
            el("code.act-cmd", { text: a.cmd, title: a.cmd }),
            el("span.act-status", { text: a.status })),
          a.error ? el("div.act-err", { text: a.error }) : null,
          a.output ? el("pre.act-out", { text: a.output.slice(0, 600) }) : null),
      )));
  }

  function render() {
    renderList();
    renderActivity();
  }

  // ── modal / topbar ─────────────────────────────────────────────────────────
  function open() {
    if (isOpen()) return;
    if (btn) btn.setAttribute("aria-pressed", "true");
    listEl = el("div.chg-list");
    diffEl = el("div.chg-diff", null,
      el("p.changes-state", { text: "Select a file to see its diff." }));
    actEl = el("div.chg-activity");
    const refreshBtn = el("button.secondary", { text: "Refresh", onclick: () => refresh() });
    const body = el("div.changes-layout", null,
      el("div.chg-col", null,
        el("div.chg-col-head", null, el("span", { text: "Changed files" }), refreshBtn),
        listEl,
        el("div.chg-col-head", null, el("span", { text: "Session commands & tests" })),
        actEl),
      diffEl);
    modal = OMP.dialogs.modal({
      title: "Changes",
      cls: "changes-modal",
      wide: true,
      body,
      onKey: (e) => {
        if (e.key === "r" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); refresh(); }
      },
    });
    modal.promise.then(() => {
      modal = null; listEl = diffEl = actEl = stateEl = null;
      if (btn) btn.setAttribute("aria-pressed", "false");
    });
    refresh();
  }

  function close() {
    if (modal) modal.done(null);
    modal = null;
  }

  function init(a) {
    app = a;
    const host = $("#topbar .top-actions");
    if (host) {
      btn = el("button.icon-btn.changes-btn", {
        id: "changes-btn",
        title: "Changes (git diff, checkpoints)",
        "aria-label": "Changes",
        "aria-pressed": "false",
        onclick: () => (isOpen() ? close() : open()),
        html: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="7" cy="7" r="3"/><circle cx="17" cy="17" r="3"/><path d="M10 7h4a3 3 0 0 1 3 3v4M7 10v4a3 3 0 0 0 3 3h4"/></svg>',
      });
      host.prepend(btn);
    }
    window.addEventListener("omp:frame", (e) => {
      recordFrame(e.detail || e);
    });
    window.addEventListener("omp:project", () => {
      activity.clear();
      selected = null;
      refresh();
    });
  }

  OMP.changes = { init, open, refresh, recordFrame };
})(window.OMP);
