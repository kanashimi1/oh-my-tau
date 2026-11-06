"use strict";
// Agent panel (`OMP.agentPanel`): split into two surfaces.
//
// - Left sidebar (#agent-dock): the role picker applied to new sessions.
// - Right sidebar (#subagents-dock, inside the parent's #subagents-sidebar):
//   the current session's Todo plan and subagents (task-tool children) with
//   their live status and individual Tau logos.
//
// Data sources (all real, no synthesized state):
// - Todos: `get_state` → `todoPhases` (phase → tasks with per-task status),
//   refreshed on todo/turn frames.
// - Subagents: the parent session's `task` tool calls, observed live on the
//   `omp:frame` bus and reconstructed from history on every session attach —
//   one child per `tasks[]` item, with its role and task line.
//   A child's status is "working" from the call start; the call result
//   attaches the real handle ids (`details.handles`) but does NOT settle
//   outcomes. Actual completion arrives later inside `<task-results>`
//   envelopes (wait tool results, auto-delivered toolResult messages):
//   completed => "done", failed/cancelled => "error". Only exact handle-id
//   matches settle; unknown ids and unknown statuses never invent state.
// - Blockers: the roster entry of the current agent (`needsInput`,
//   `awaitingTasksSinceMs`) — the session-level wait states.
//
// The role picker persists the selected role (localStorage) and
// `creationOptions()` returns it for POST /api/agents; the backend resolves
// it to a registered agent definition (see web/agent.rs). The worktree
// checkbox persists `omp-worktree-default`; creationOptions adds
// `worktree:true` so the backend runs the child in a fresh git worktree.
(function (OMP) {
  const { el, $ } = OMP;
  const ROLE_KEY = "omp-agent-role";
  const WORKTREE_KEY = "omp-worktree-default";
  const TODO_COLLAPSE_KEY = "omp-todo-collapsed";
  let todoCollapsed = false;
  try { todoCollapsed = localStorage.getItem(TODO_COLLAPSE_KEY) === "true"; } catch { /* storage unavailable */ }

  const ROLES = [
    { id: "universal", label: "Universal", title: "Default session, full capabilities" },
    { id: "coder", label: "Coder", title: "A registered coder agent definition" },
    { id: "blockbench", label: "Blockbench", title: "Registered blockbench modeling agent" },
    { id: "blender", label: "Blender", title: "Registered Blender modeling agent" },
    { id: "reviewer", label: "Reviewer", title: "Registered review agent" },
  ];

  // Tau mark: the Greek letter, one stroked path (inline SVG, no assets).
  const TAU = (cls) =>
    `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" aria-hidden="true"><path d="M4 6.5h16"/><path d="M12 6.5V20"/></svg>`;

  const ICONS = {
    working: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M21 12a9 9 0 1 1-6.2-8.56"/></svg>',
    done: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>',
    waiting: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>',
    error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="m9 9 6 6M15 9l-6 6"/></svg>',
    todo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="3"/><path d="M8 12.5l3 3 5-6"/></svg>',
  };

  let app = null; // OMP.app (set by init)
  let lastAgentId = null;
  let subs = new Map(); // toolCallId -> { children: [{name, role, action, status}], ended }
  const firstLine = (s, max = 120) => {
    const line = String(s ?? "").trim().split("\n")[0];
    return line.length > max ? line.slice(0, max) + "…" : line;
  };

  // ── live data ─────────────────────────────────────────────────────────────
  async function refreshState() {
    if (!app || !app.S.agentId) return;
    try { app.S.state = await app.rpc("get_state"); } catch { /* disconnected */ }
  }

  function startTask(toolCallId, args) {
    const fallbackRole = (args && (args.agent || args.role)) || "universal";
    const tasks = Array.isArray(args.tasks) ? args.tasks : [{ task: args.task || args.description, agent: fallbackRole }];
    subs.set(toolCallId, {
      ended: false,
      children: tasks.map((t) => ({
        name: t.name || null,
        role: t.agent || fallbackRole,
        action: firstLine(t.task || t.description || ""),
        id: null, // real handle id, attached when the call returns
        status: "working",
      })),
    });
  }

  // The task tool returns immediately: the call result carries the real
  // child handles (`details.handles`, one per tasks[] item, in order).
  // Attach the ids but do NOT settle: actual outcomes arrive later as
  // `<task-results>` envelopes inside wait results / auto-deliveries.
  function attachHandles(result) {
    const batch = subs.get(result && result.toolCallId);
    const handles = result && result.details && Array.isArray(result.details.handles) ? result.details.handles : null;
    if (batch) {
      batch.ended = true;
      if (handles) for (let i = 0; i < Math.min(handles.length, batch.children.length); i++) {
        const h = handles[i];
        if (h && typeof h.id === "string") batch.children[i].id = h.id;
      }
    }
    return handles;
  }

  // A launch/validation failure is already the final outcome: no handle or
  // later <task-results> envelope will arrive for these children.
  function finishTask(toolCallId, failed) {
    const batch = subs.get(toolCallId);
    if (!batch) return false;
    batch.ended = true;
    if (failed) for (const child of batch.children) child.status = "error";
    return true;
  }

  // Find the child with a real handle id across all batches.
  function childById(id) {
    if (typeof id !== "string" || !id) return null;
    for (const batch of subs.values())
      for (const c of batch.children) if (c.id === id) return c;
    return null;
  }

  // Extract concatenated pretty-JSON objects from `text` (inside
  // `<task-results>` blocks): brace-scan each top-level object honoring
  // strings/escapes, JSON.parse it; unparsable slices are skipped.
  function jsonObjects(text) {
    const out = [];
    for (let start = text.indexOf("{"); start !== -1; start = text.indexOf("{", start + 1)) {
      let depth = 0, inStr = false, esc = false, end = -1;
      for (let i = start; i < text.length; i++) {
        const ch = text[i];
        if (inStr) {
          if (esc) esc = false;
          else if (ch === "\\") esc = true;
          else if (ch === '"') inStr = false;
        } else if (ch === '"') inStr = true;
        else if (ch === "{") depth++;
        else if (ch === "}") {
          depth--;
          if (depth === 0) { end = i; break; }
        }
      }
      if (end === -1) break;
      try { out.push(JSON.parse(text.slice(start, end + 1))); } catch { /* not JSON */ }
    }
    return out;
  }

  // A settled envelope: `{id, status: "completed"|"failed…"|"cancelled"}`.
  // Only exact-id matches settle; unknown ids are ignored (never invented).
  function settleEnvelope(env) {
    if (!env || typeof env.id !== "string") return false;
    const c = childById(env.id);
    if (!c) return false;
    const status = typeof env.status === "string" ? env.status : "";
    if (status === "completed") c.status = "done";
    else if (status.startsWith("failed") || status === "cancelled") c.status = "error";
    // other/unknown status: leave the child as-is rather than guessing
    return true;
  }

  // Settle children from any payload that may carry `<task-results>`
  // envelopes: tool-execution results and auto-delivered toolResult/user
  // messages all funnel through their text content.
  function settleFromContent(content, details) {
    let touched = false;
    if (details && Array.isArray(details.results)) {
      for (const env of details.results) touched = settleEnvelope(env) || touched;
    }
    const texts = Array.isArray(content)
      ? content.filter((p) => p && p.type === "text" && typeof p.text === "string").map((p) => p.text)
      : typeof content === "string" ? [content] : [];
    for (const text of texts) {
      if (!text.includes("<task-results>")) continue;
      for (const env of jsonObjects(text)) touched = settleEnvelope(env) || touched;
    }
    return touched;
  }

  function onFrame(e) {
    if (!app || e.detail.agentId !== app.S.agentId) return;
    const f = e.detail.frame;
    if (f.type === "tool_execution_start" && f.toolName === "task") {
      startTask(f.toolCallId, f.args);
      schedule();
    } else if (f.type === "tool_execution_end" && f.toolName === "task" && subs.has(f.toolCallId)) {
      attachHandles({ toolCallId: f.toolCallId, details: f.result && f.result.details });
      if (f.isError) finishTask(f.toolCallId, true);
      schedule();
    } else if (f.type === "tool_execution_end" && f.toolName === "todo") {
      // The todo tool persists new phases into settings; pick them up now
      // instead of waiting for the next turn boundary.
      refreshState().then(schedule);
    } else if (f.type === "tool_execution_end") {
      // Real completion of task children rides later results (wait/other
      // tools) as `<task-results>` envelopes; parse any tool result.
      if (settleFromContent(f.result && f.result.content, f.result && f.result.details)) schedule();
    } else if ((f.type === "message_start" || f.type === "message_end")
      && f.message && f.message.role === "toolResult") {
      // Auto-deliveries land as toolResult messages, too.
      if (settleFromContent(f.message.content, f.message.details)) schedule();
    } else if (f.type === "turn_end" || f.type === "todo_reminder" || f.type === "todo_auto_clear"
      || f.type === "session_loaded" || f.type === "session_reset") {
      refreshState().then(schedule);
      if (f.type === "session_loaded" || f.type === "session_reset") resetSubs();
    }
  }

  function resetSubs() {
    subs = new Map();
    lastAgentId = app && app.S.agentId;
  }

  function ingestMessages(messages) {
    resetSubs();
    for (const m of messages || []) {
      if (m.role === "assistant" && Array.isArray(m.content)) {
        for (const b of m.content) {
          if (b.type === "toolCall" && b.name === "task") {
            let args = b.arguments;
            if (typeof args === "string") try { args = JSON.parse(args); } catch { args = {}; }
            startTask(b.id, args);
          }
        }
      } else if (m.role === "toolResult") {
        // History replay: the task call result carries the handles; any
        // later result may carry the settling `<task-results>` envelopes.
        attachHandles({ toolCallId: m.toolCallId, details: m.details });
        if (m.toolName === "task" && m.isError) finishTask(m.toolCallId, true);
        settleFromContent(m.content, m.details);
      } else if (Array.isArray(m.content)) {
        // Auto-delivered results may also replay as user/system messages.
        settleFromContent(m.content, m.details);
      }
    }
    schedule();
  }

  // Redraw both surfaces from current state. No state is reset here; the
  // right dock may not exist yet (parent controls #subagents-sidebar), so
  // each surface renders only when its host is present.
  const schedule = OMP.rafThrottle(() => {
    renderLeft($("#agent-dock"));
    renderRight($("#subagents-dock"));
  });

  // ── rendering ─────────────────────────────────────────────────────────────
  function currentRoster() {
    return (app && app.S.agents.find((a) => a.id === app.S.agentId)) || null;
  }

  function statusLine(status) {
    const map = { working: ["working", "ap-working"], waiting: ["waiting", "ap-waiting"], done: ["done", "ap-done"], error: ["failed", "ap-error"] };
    const [label, cls] = map[status] || map.working;
    return el("span.ap-status." + cls, { title: label }, el("span.ap-ic", { html: ICONS[status] }), el("span", { text: label }));
  }

  function renderTodoSection() {
    const st = app && app.S.state;
    const phases = Array.isArray(st && st.todoPhases) ? st.todoPhases : [];
    const head = el("button.ap-head.ap-collapse-toggle", {
      type: "button",
      "aria-expanded": String(!todoCollapsed),
      "aria-controls": "ap-todo-body",
      title: todoCollapsed ? "Expand Todos" : "Collapse Todos",
      onclick: () => {
        todoCollapsed = !todoCollapsed;
        try { localStorage.setItem(TODO_COLLAPSE_KEY, String(todoCollapsed)); } catch { /* storage unavailable */ }
        schedule();
      },
    }, el("span.ap-ic", { html: ICONS.todo }), el("span", { text: "Todos" }),
    el("span.ap-collapse-chevron", { html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>' }));
    if (!phases.length) {
      return el("section.ap-section.ap-todo-section", null, head,
        el("div.ap-todo-body", { id: "ap-todo-body", hidden: todoCollapsed }, el("div.ap-empty", { text: "No todo list yet" })));
    }
    const done = (p) => p.tasks.filter((t) => t.status === "completed").length;
    const activeIdx = phases.findIndex((p) => p.tasks.some((t) => t.status === "in_progress" || t.status === "pending"));
    const TASK_MARK = {
      in_progress: ["working", ICONS.working],
      pending: ["pending", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="8"/></svg>'],
      completed: ["done", ICONS.done],
      abandoned: ["dropped", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>'],
      blocked: ["blocked", ICONS.waiting],
    };
    const phaseNodes = phases.map((p, i) => {
      const total = p.tasks.length;
      const count = el("span.ap-progress", { text: `${done(p)}/${total}`, title: `${done(p)} of ${total} tasks done` });
      const tasks = p.tasks.map((t) => {
        const [label, mark] = TASK_MARK[t.status] || TASK_MARK.pending;
        const cls = t.status === "completed" ? ".is-done" : t.status === "in_progress" ? ".is-active" : t.status === "blocked" ? ".is-blocked" : "";
        return el("li.ap-task" + cls, { title: t.blocker ? `Blocked: ${t.blocker}` : null },
          el("span.ap-task-mark", { html: mark, "aria-hidden": "true" }),
          el("span.ap-task-text", { text: t.content || "(untitled)" }),
          el("span.ap-task-status", { text: label }));
      });
      return el("div.ap-phase" + (i === activeIdx ? ".active" : ""), { role: "group" },
        el("div.ap-phase-head", null,
          el("span.ap-phase-name", { text: p.name || "Phase", title: p.name }),
          count),
        el("ul.ap-tasks", { role: "list" }, ...tasks));
    });
    return el("section.ap-section.ap-todo-section", null, head,
      el("div.ap-todo-body", { id: "ap-todo-body", hidden: todoCollapsed }, ...phaseNodes));
  }

  function blockerLines() {
    const a = currentRoster();
    const lines = [];
    if (a && a.needsInput) {
      lines.push(el("div.ap-blocker.ap-waiting", { title: "The session is parked on a dialog in the transcript." },
        el("span.ap-ic", { html: ICONS.waiting }),
        el("span", { text: "Waiting for your input — answer the dialog in the transcript" })));
    } else if (a && a.awaitingTasksSinceMs) {
      const ms = Date.now() - a.awaitingTasksSinceMs;
      lines.push(el("div.ap-blocker.ap-waiting", { title: "The run ended but background subagents are still settling." },
        el("span.ap-ic", { html: ICONS.waiting }),
        el("span", { text: "Waiting on subagents " + OMP.fmtDuration(ms) })));
    }
    return lines;
  }

  // Steer a working child: ask for the message, then send it via the
  // backend rpc (routed to tools::eval_agents::send_to_agent by the parent).
  async function steerChild(c) {
    if (!c.id) return;
    try {
      const message = await OMP.dialogs.promptText(`Message ${c.name || c.role}`, {
        placeholder: "What should this subagent do differently?",
        message: "Sent to the running subagent; it sees this on its next step.",
      }).promise;
      if (!message || !message.trim()) return;
      await app.rpc("steer_subagent", { subagentId: c.id, message: message.trim() });
      OMP.toast(`Message sent to ${c.name || c.role}`, "info", 2500);
    } catch (e) {
      OMP.toast(`Steer failed: ${String((e && e.message) || e)}`, "error");
    }
  }

  function renderSubSection() {
    const head = el("div.ap-head", null, el("span.ap-head-tau", { html: TAU("ap-tau") }), el("span", { text: "Subagents" }));
    const children = [];
    for (const batch of subs.values()) children.push(...batch.children);
    if (!children.length) {
      const empty = blockerLines().length
        ? "No task children this session yet."
        : "No subagents this session — spawned `task` children appear here.";
      return el("section.ap-section.ap-sub-section", null, head, ...blockerLines(), el("div.ap-empty", { text: empty }));
    }
    const rows = children.map((c) => {
      const name = c.name || c.role;
      const logo = el("span.ap-sub-tau." + ("role-" + c.role), { "aria-hidden": "true" });
      OMP.mascot.mount(logo, { size: 16, state: c.status === "working" ? "work" : c.status === "error" ? "error" : "idle" });
      // Steer only children that are actually still working with a known
      // handle id; finished/failed/unknown children get no active action.
      const steerable = c.status === "working" && typeof c.id === "string";
      const steer = steerable ? el("button.ap-sub-msg", {
        type: "button",
        "aria-label": `Message ${name}`,
        title: `Message ${name} (agent://${c.id})`,
        onclick: () => steerChild(c),
      }, "Message") : null;
      return el("div.ap-sub" + (c.status === "working" ? ".is-working" : ""), {
        role: "listitem",
        title: `${name} (${c.role}) — ${c.action || "no task text"}`,
      }, logo,
        el("span.ap-sub-main", null,
          el("span.ap-sub-name", { text: name }),
          el("span.ap-sub-role", { text: c.role }),
          c.action ? el("span.ap-sub-action", { text: c.action, title: c.action }) : null),
        steer,
        statusLine(c.status));
    });
    return el("section.ap-section.ap-sub-section", null, head, ...blockerLines(),
      el("div.ap-subs", { role: "list", "aria-label": "Subagents of the current session" }, ...rows));
  }

  function renderRolePicker() {
    const saved = localStorage.getItem(ROLE_KEY) || "universal";
    const select = el("select.ap-role-select", {
      id: "agent-role",
      "aria-label": "Role for new sessions",
      onchange: () => {
        localStorage.setItem(ROLE_KEY, select.value);
        OMP.toast(`New sessions will run as “${select.value}”`, "info", 2500);
      },
    }, ...ROLES.map((r) => el("option", { value: r.id, selected: r.id === saved, title: r.title, text: r.label })));
    let worktree = false;
    try { worktree = localStorage.getItem(WORKTREE_KEY) === "true"; } catch { /* storage unavailable */ }
    let box = el("input", {
      type: "checkbox",
      checked: worktree,
      onchange: () => {
        try { localStorage.setItem(WORKTREE_KEY, String(box.checked)); } catch { /* storage unavailable */ }
        OMP.toast(box.checked
          ? "New sessions will run in an isolated git worktree"
          : "New sessions will run in the project directory", "info", 2500);
      },
    });
    return el("div", null,
      el("div.ap-role", null,
        el("label.ap-role-label", { for: "agent-role", text: "New session role" }),
        select),
      el("div.ap-role", null,
        box,
        el("label", {
          for: "agent-worktree",
          title: "Run new sessions in a fresh git worktree (branch omt/<role>); falls back to the plain directory when git fails",
        }, el("span.ap-role-label", { text: "isolated git worktree" }))));
  }

  // ── public interface ──────────────────────────────────────────────────────
  function renderLeft(host) {
    if (!host) return;
    if (app && app.S.agentId !== lastAgentId) resetSubs();
    host.replaceChildren(renderRolePicker());
  }

  function renderRight(host) {
    if (!host) return;
    if (app && app.S.agentId !== lastAgentId) resetSubs();
    host.replaceChildren(renderTodoSection(), renderSubSection());
    $("#app").classList.toggle("has-subagents", hasActivity());
  }

  /** True when the right activity panel has real Todo or subagent content. */
  function hasActivity() {
    if (subs.size > 0) return true;
    const phases = app && app.S.state && app.S.state.todoPhases;
    if (Array.isArray(phases) && phases.some((p) => (p.tasks || []).length)) return true;
    const a = currentRoster();
    return !!(a && a.awaitingTasksSinceMs);
  }

  OMP.agentPanel = {
    /** Called once after shell init. */
    init(a) {
      app = a;
      window.addEventListener("omp:frame", onFrame);
      refreshState().then(() => {
        renderLeft($("#agent-dock"));
        renderRight($("#subagents-dock"));
      });
    },
    /** Render the left sidebar surface (new-session role picker). */
    renderLeft,
    /** Render the right activity surface (Todo + subagents). */
    renderRight,
    /** True when Todo or task-tool children need the right panel. */
    hasActivity,
    ingestMessages,
    /** Extra JSON fields for POST /api/agents (backend resolves `role`). */
    creationOptions() {
      let role = localStorage.getItem(ROLE_KEY);
      let opts = role && role !== "universal" ? { role } : {};
      let worktree = false;
      try { worktree = localStorage.getItem(WORKTREE_KEY) === "true"; } catch { /* storage unavailable */ }
      if (worktree) opts.worktree = true;
      return opts;
    },
  };
})(window.OMP);
