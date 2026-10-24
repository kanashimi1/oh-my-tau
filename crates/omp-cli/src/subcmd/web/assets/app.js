"use strict";
// App shell: sidebar (running agents, stored sessions), the agent connection
// (rpc over WebSocket), header/status bar and session actions.
(function (OMP) {
  const { el, $, api } = OMP;
  const DIALOG_METHODS = new Set(["select", "confirm", "input", "editor", "ask"]);
  const LOCAL_COMMANDS = [
    { name: "new", description: "Start a new session in this folder" },
    { name: "resume", description: "Resume a stored session of this folder" },
    { name: "exit", description: "Stop this agent", aliases: ["quit"] },
    { name: "hotkeys", description: "Keyboard shortcuts", aliases: ["help"] },
    { name: "export", description: "Open the HTML export of this session" },
    { name: "tree", description: "Rewind to an earlier message of this session (alias of /branch)" },
  ];

  const S = {
    info: { home: "", defaultCwd: "" },
    project: localStorage.getItem("omp-web-project") || "",
    agents: [],
    sessions: [],
    agentId: null,
    ws: null,
    seq: 0,
    pending: new Map(),
    state: null,
    stats: null,
    commands: [],
    streaming: false,
    runStart: 0,
    status: new Map(),
    widgets: new Map(),
    queue: { steering: [], followUp: [] },
    dead: null,
    bashRunning: 0,
  };

  const transcript = new OMP.Transcript($("#transcript"), $("#transcript-wrap"));
  const composer = new OMP.Composer({ onSubmit: submit, onAbort: abort });
  OMP.dialogs.afterClose = () => composer.focus();

  // ── rpc ──────────────────────────────────────────────────────────────────
  function send(frame) {
    if (!S.ws || S.ws.readyState !== WebSocket.OPEN) {
      OMP.toast("Not connected to an agent", "error");
      return false;
    }
    S.ws.send(JSON.stringify(frame));
    return true;
  }
  function rpc(type, params = {}) {
    const id = "w" + ++S.seq;
    return new Promise((resolve, reject) => {
      if (!send({ id, type, ...params })) return reject(new Error("not connected"));
      S.pending.set(id, { resolve, reject, type });
    });
  }
  OMP.rpc = rpc;

  function connect(agentId) {
    if (S.ws) { S.ws.onclose = null; S.ws.close(); }
    for (const p of S.pending.values()) p.reject(new Error("disconnected"));
    S.pending.clear();
    OMP.dialogs.clear();
    transcript.reset();
    Object.assign(S, { agentId, agentInfo: null, state: null, stats: null, streaming: false, dead: null, queue: { steering: [], followUp: [] } });
    S.status.clear();
    S.widgets.clear();
    renderWidgets();
    renderQueue();
    composer.setStreaming(false);
    composer.setEnabled(false);
    $("#empty-state").hidden = !!agentId;
    renderSidebar();
    if (!agentId) { renderHeader(); renderStatus(); return; }
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/${agentId}`);
    S.ws = ws;
    ws.onopen = () => onOpen();
    ws.onmessage = (e) => {
      let frame;
      try { frame = JSON.parse(e.data); } catch { return; }
      onFrame(frame);
    };
    ws.onclose = () => {
      if (S.ws !== ws) return;
      S.ws = null;
      composer.setEnabled(false);
      if (!S.dead) {
        transcript.addSystem("Connection lost. Reconnecting…", "warn");
        setTimeout(async () => {
          await refreshAgents();
          if (S.agentId === agentId && S.agents.some((a) => a.id === agentId)) connect(agentId);
          else if (S.agentId === agentId) showDead(null);
        }, 1500);
      }
    };
  }

  async function onOpen() {
    composer.setEnabled(true);
    send({ type: "set_ask_dialog", enabled: true });
    try {
      const [state, msgs, cmds] = await Promise.all([rpc("get_state"), rpc("get_messages"), rpc("get_available_commands")]);
      S.state = state;
      renderWidgets();
      setCommands(cmds.commands);
      transcript.loadMessages(msgs.messages);
      setStreaming(!!state.isStreaming);
      renderHeader();
      refreshStats();
    } catch (e) {
      transcript.addSystem("Failed to load session: " + e.message, "error");
    }
  }

  async function refreshState() {
    try { S.state = await rpc("get_state"); } catch { return; }
    renderHeader();
    renderStatus();
    renderWidgets();
  }
  async function refreshStats() {
    try { S.stats = await rpc("get_session_stats"); } catch { /* older agent */ }
    renderStatus();
  }

  function setCommands(list) {
    S.commands = list || [];
    const names = new Set(S.commands.map((c) => c.name));
    composer.setCommands([...S.commands, ...LOCAL_COMMANDS.filter((c) => !names.has(c.name))]);
  }

  function setStreaming(on) {
    if (on && !S.streaming) S.runStart = Date.now();
    S.streaming = on;
    composer.setStreaming(on);
    renderStatus();
  }

  function answerDialog(id, value) {
    send({ type: "extension_ui_response", id, ...value });
  }

  function onFrame(f) {
    if (f.type === "response" && f.id && S.pending.has(f.id)) {
      const p = S.pending.get(f.id);
      S.pending.delete(f.id);
      if (f.success === false) p.reject(new Error(f.error || "command failed"));
      else p.resolve(f.data || {});
      return;
    }
    switch (f.type) {
      case "web_attached":
        S.agentInfo = f.agent;
        renderHeader();
        return;
      case "web_agent_exit": return showDead(f);
      case "web_lagged": return resync();
      case "web_error": return OMP.toast(f.message, "error");
      case "response":
        if (f.success === false && f.error) transcript.addSystem(`${f.command}: ${f.error}`, "error");
        return;
      case "extension_ui_request": return uiRequest(f);
      case "clipboard_copy":
        OMP.copyText(f.text || "").then((ok) => OMP.toast(ok ? "Copied to clipboard" : "Clipboard unavailable", ok ? "success" : "error"));
        return;
      case "available_commands_update": return setCommands(f.commands);
      case "agent_start": setStreaming(true); return;
      case "agent_end": setStreaming(false); refreshStats(); refreshAgents(); loadSessions(); return;
      case "session_settled": case "agent_settled": setStreaming(false); return;
      case "turn_end": refreshStats(); refreshState(); return;
      case "queue_update":
        S.queue = { steering: f.steering || [], followUp: f.followUp || [] };
        return renderQueue();
      case "session_loaded":
        transcript.loadMessages(f.messages);
        refreshState(); refreshStats(); refreshAgents();
        return;
      case "session_reset":
        transcript.reset();
        refreshState(); refreshStats(); refreshAgents();
        return;
      case "thinking_level_changed": case "model_changed": case "session_info": case "models_changed":
        refreshState(); refreshAgents();
        if (f.type === "session_info") loadSessions();
        return;
      case "prompt_result":
        if (f.status === "failed" && f.error) transcript.addSystem(f.error, "error");
        return;
      default:
        transcript.onEvent(f);
    }
  }

  function uiRequest(f) {
    const m = f.method;
    if (DIALOG_METHODS.has(m)) return OMP.dialogs.request(f, answerDialog);
    if (m === "cancelDialog") return OMP.dialogs.cancel((f.params && f.params.id) || f.targetId || f.id);
    if (m === "notify" || m === "ui_notify") return OMP.toast(f.message || f.text || "", f.notifyType || f.level || "info");
    if (m === "setStatus" || m === "set_status") {
      const key = f.statusKey || f.key || "ext";
      if (f.statusText || f.text) S.status.set(key, f.statusText || f.text); else S.status.delete(key);
      return renderStatus();
    }
    if (m === "setWidget" || m === "set_widget") {
      const key = f.widgetKey || f.key || "ext";
      const lines = f.widgetLines || f.lines || (f.content ? [].concat(f.content) : null);
      if (lines && lines.length) S.widgets.set(key, { lines, below: f.widgetPlacement === "belowEditor" || f.placement === "belowEditor" });
      else S.widgets.delete(key);
      return renderWidgets();
    }
    if (m === "setTitle" || m === "set_title") { document.title = f.title || "omt"; return; }
    if (m === "set_editor_text" || m === "setEditorText") return composer.setText(f.text || "");
    if (m === "setWorkingMessage") { S.status.set("working", f.message || ""); return renderStatus(); }
  }

  async function resync() {
    try {
      const msgs = await rpc("get_messages");
      transcript.loadMessages(msgs.messages);
    } catch { /* ignore */ }
  }

  function showDead(f) {
    S.dead = f || {};
    setStreaming(false);
    composer.setEnabled(false);
    const tail = f && f.stderr && f.stderr.length ? "\n\n" + f.stderr.slice(-12).join("\n") : "";
    const box = transcript.addSystem(`Agent stopped${f && f.code != null ? ` (exit ${f.code})` : ""}.${tail}`, "error");
    const file = S.state && S.state.sessionFile;
    if (file) {
      box.append(el("div", null, el("button.secondary", { text: "Restart this session", onclick: () => openSession({ path: file, cwd: currentCwd() }) })));
    }
    refreshAgents();
  }

  // ── submit / abort ───────────────────────────────────────────────────────
  function submit(text, images, mode) {
    if (!S.agentId || S.dead) {
      OMP.toast("Start or open a session first", "warning");
      return false;
    }
    const trimmed = text.trim();
    if (trimmed.startsWith("/") && !images.length) {
      const name = trimmed.slice(1).split(/\s+/)[0];
      const args = trimmed.slice(1 + name.length).trim();
      if (!name.includes("/")) {
        if (localCommand(name, args)) return true;
        const known = S.commands.some((c) => c.name === name || (c.aliases || []).includes(name));
        if (!known) {
          transcript.addSystem(`Unknown command: /${name}`, "warn");
          return false;
        }
        transcript.addDivider(trimmed.length > 80 ? trimmed.slice(0, 80) + "…" : trimmed);
      }
    }
    if (trimmed.startsWith("!") && !images.length) {
      runBash(trimmed.replace(/^!!?/, ""), trimmed.startsWith("!!"));
      return true;
    }
    const frame = { type: "prompt", message: text };
    if (images.length) frame.images = images;
    if (S.streaming) frame.streamingBehavior = mode === "followUp" ? "followUp" : "steer";
    rpc(frame.type, Object.fromEntries(Object.entries(frame).filter(([k]) => k !== "type")))
      .catch((e) => transcript.addSystem(e.message, "error"));
    return true;
  }

  async function runBash(command, excluded) {
    const card = transcript.card("bash-" + Date.now(), "bash");
    card.setArgs({ command });
    card.start();
    transcript.append(card.node);
    S.bashRunning++;
    $("#stop-btn").hidden = false;
    try {
      const r = await rpc("bash", { command, excludeFromContext: excluded || undefined });
      let out = r.output || "";
      if (r.exitCode) out += `\n[exit ${r.exitCode}]`;
      if (r.cancelled) out += "\n[cancelled]";
      card.finish({ content: [{ type: "text", text: out }] }, !!r.exitCode);
    } catch (e) {
      card.finish({ content: [{ type: "text", text: e.message }] }, true);
    } finally {
      S.bashRunning--;
      $("#stop-btn").hidden = !S.streaming && !S.bashRunning;
    }
  }

  function abort() {
    if (S.bashRunning) rpc("abort_bash").catch(() => {});
    if (S.streaming) rpc("abort").catch(() => {});
  }

  function localCommand(name, args) {
    switch (name) {
      case "new": newSessionHere(); return true;
      case "resume": resumePicker(); return true;
      case "exit": case "quit": stopAgent(S.agentId); return true;
      case "hotkeys": case "help": showHotkeys(); return true;
      case "model": case "switch": if (!args) { modelPicker(); return true; } return false;
      case "effort": if (!args) { thinkingPicker(); return true; } return false;
      case "export": if (!args) { exportHtml(); return true; } return false;
      case "tree":
        transcript.addDivider("/branch");
        rpc("prompt", { message: "/branch" + (args ? " " + args : "") }).catch((e) => transcript.addSystem(e.message, "error"));
        return true;
      default: return false;
    }
  }

  function currentCwd() {
    return (S.agentInfo && S.agentInfo.cwd) || (S.agents.find((a) => a.id === S.agentId) || {}).cwd || S.project || S.info.defaultCwd;
  }

  // Shell rendering and actions live in shell.js (loaded first).
  function renderHeader() { OMP.shellImpl.renderHeader(S); }
  function renderStatus() { OMP.shellImpl.renderStatus(S); }
  function renderSidebar() { OMP.shellImpl.renderSidebar(S); }
  function renderQueue() { OMP.shellImpl.renderQueue(S); }
  function renderWidgets() { OMP.shellImpl.renderWidgets(S); }
  function refreshAgents() { return OMP.shellImpl.refreshAgents(S); }
  function loadSessions() { return OMP.shellImpl.loadSessions(S); }
  function openSession(row) { return OMP.shellImpl.openSession(S, row); }
  function newSessionHere() { return OMP.shellImpl.newSession(S, currentCwd()); }
  function resumePicker() { return OMP.shellImpl.resumePicker(S, currentCwd()); }
  function stopAgent(id) { return OMP.shellImpl.stopAgent(S, id); }
  function showHotkeys() { return OMP.shellImpl.showHotkeys(); }
  function modelPicker() { return OMP.shellImpl.modelPicker(S); }
  function thinkingPicker() { return OMP.shellImpl.thinkingPicker(S); }
  function exportHtml() { return OMP.shellImpl.exportHtml(S); }

  OMP.app = { S, rpc, send, connect, abort, transcript, composer, currentCwd, setCommands, localCommand };
  OMP.shellImpl.init(OMP.app);
})(window.OMP);
