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
    herdrPane: null,
    herdrModel: null,
    herdrThinkingLevel: null,
    herdrUltra: null,
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
    view: "chat",
  };

  const transcript = new OMP.Transcript($("#transcript"), $("#transcript-wrap"));
  const composer = new OMP.Composer({ onSubmit: submit, onAbort: abort });
  OMP.dialogs.afterClose = () => { composer.focus(); renderStatus(); };

  // Bounded cache of loaded session histories (see OMP.SessionHistoryCache).
  // Any live frame that can mutate history marks every entry dirty, so a
  // cached snapshot is only ever reused when the session is provably quiet.
  const sessionCache = new OMP.SessionHistoryCache();

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

  function setView(view) {
    S.view = view;
    const maps = S.view === "maps";
    $("#maps-view").hidden = !maps;
    const fleet = S.view === "fleet";
    $("#fleet-view").hidden = !fleet;
    $("#fleet-btn").setAttribute("aria-pressed", String(fleet));
    $("#main").classList.toggle("fleet-active", fleet);
    document.body.classList.toggle("maps-active", maps);
    $("#maps-btn").setAttribute("aria-pressed", String(maps));
    if (OMP.maps) { if (maps) OMP.maps.show(S); else OMP.maps.hide(); }
    renderHerdrView();
  }

  // Ask the agent for its protocol capabilities; older detached agents reject
  // the method, and the conservative legacy profile (same shape, protocol
  // versions [1], getMessages.maxPageSize 256) is assumed instead — the
  // session load proceeds either way.
  async function getCapabilities() {
    try {
      return OMP.normalizeCapabilities(await rpc("get_capabilities"));
    } catch {
      return OMP.normalizeCapabilities(null);
    }
  }

  function historyPageLimit() {
    return Math.max(1, Math.min(S.caps?.limits?.getMessages?.maxPageSize || 256, 512));
  }

  // Public full-history semantics (export panel, session switching, e2e):
  // every message. Backward-capable agents page from the newest offset down
  // (bounded responses, and paging is allowed while streaming); legacy agents
  // page forward on quiet sessions and take one get_messages snapshot while
  // streaming.
  async function getMessages() {
    if (S.caps?.features?.messagesBackwardPaging) {
      const buffer = new OMP.HistoryBuffer(S.caps, { backward: true });
      const pages = [];
      while (!buffer.done) pages.push(await buffer.fetchPage((params) => rpc("get_messages_page", params)));
      return [].concat(...pages.reverse());
    }
    // Full history can contain many MiB of inline images. Keep such responses
    // out of the hub's broadcast queue whenever the snapshot is stable.
    if (S.state?.isStreaming) {
      const response = await rpc("get_messages");
      return response.messages;
    }
    const messages = [];
    let cursor = null;
    const limit = historyPageLimit();
    do {
      // The capabilities fallback keeps this at 256 for older detached agents.
      const page = await rpc("get_messages_page", { limit, ...(cursor ? { cursor } : {}) });
      messages.push(...page.messages);
      cursor = page.nextCursor || null;
    } while (cursor);
    return messages;
  }
  function connect(agentId, herdrPane = null) {
    clearTimeout(S.herdrPoll);
    if (S.herdrPane !== herdrPane) composer.setText("");
    const prev = S.ws;
    S.ws = null;
    if (prev) { prev.onclose = null; prev.close(); }
    for (const p of S.pending.values()) p.reject(new Error("disconnected"));
    S.pending.clear();
    OMP.dialogs.clear();
    transcript.reset();
    $("#transcript").removeAttribute("aria-busy");
    Object.assign(S, { agentId, herdrPane, caps: null, bashRunning: 0, agentInfo: null, state: null, stats: null, streaming: false, dead: null, toolsRunning: 0, errorUntil: 0, queue: { steering: [], followUp: [] } });
    S.status.clear();
    S.widgets.clear();
    renderWidgets();
    renderQueue();
    composer.setStreaming(false);
    // A real target disables the composer until its socket opens; the
    // no-agent home screen keeps it live for the agent-less commands.
    composer.setEnabled(!agentId);
    composer.setTextOnly(!!herdrPane);
    $("#attach-btn").hidden = false;
    $("#composer").placeholder = herdrPane ? "Message this Herdr agent…" : "Ask anything… (/ commands · ! shell · Enter send · Shift+Enter newline)";
    Object.assign(S, { herdrActivity: false, herdrSending: null, herdrOmitted: false, herdrTokens: null, herdrModel: null, herdrThinkingLevel: null, herdrUltra: null, liveReply: null });
    $("#herdr-output").textContent = "";
    $("#empty-state").hidden = !!(agentId || herdrPane);
    $("#menu-btn").hidden = !!herdrPane;
    $("#preview-btn").hidden = !!herdrPane;
    renderHerdrView();
    if (herdrPane) updateHerdrStatus();
    if (OMP.agentPanel && herdrPane) OMP.agentPanel.ingestMessages([]);
    renderSidebar();
    updateHome();
    if (!agentId) { renderHeader(); renderStatus(); return; }
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/${agentId}`);
    S.ws = ws;
    ws.onopen = () => { if (S.ws === ws) onOpen(ws); };
    ws.onmessage = (e) => {
      if (S.ws !== ws) return;
      S.lastFrameAt = Date.now();
      let frame;
      try { frame = JSON.parse(e.data); } catch { return; }
      onFrame(frame);
    };
    ws.onclose = (ev) => {
      if (S.ws !== ws) return;
      S.ws = null;
      composer.setEnabled(false);
      sessionCache.markAllDirty(); // frames may have been missed while detached
      if (!S.dead) {
        const quiet = S.lastFrameAt ? Math.round((Date.now() - S.lastFrameAt) / 1000) : null;
        const why = "code " + ev.code + (ev.reason ? " " + ev.reason : "") + (ev.wasClean ? "" : ", unclean")
          + (quiet != null ? ", " + quiet + "s since last data" : "");
        console.warn("agent websocket closed:", why, { agentId, at: new Date().toISOString() });
        transcript.addSystem("Connection lost (" + why + "). Reconnecting…", "warn");
        setTimeout(async () => {
          await refreshAgents();
          if (S.agentId === agentId && S.agents.some((a) => a.id === agentId)) connect(agentId);
          else if (S.agentId === agentId) showDead(null);
        }, 1500);
      }
    };
  }

  function renderHerdrView() {
    const away = S.view !== "chat", active = !!S.herdrPane, activity = active && S.herdrActivity;
    $("#composer-wrap").hidden = away;
    $("#transcript-wrap").hidden = away || activity;
    $("#herdr-banner").hidden = away || !active;
    $("#herdr-activity").hidden = away || !activity;
    const btn = $("#herdr-activity-btn");
    btn.textContent = activity ? "Session history" : "Live activity";
    btn.setAttribute("aria-pressed", String(!!activity));
  }

  function updateHerdrStatus() {
    if (!S.herdrPane) return;
    const peer = S.herdr?.agents.find((a) => a.paneId === S.herdrPane);
    const state = peer?.status === "blocked" ? "Needs input in Herdr" :
      peer?.status === "working" ? "Working" : peer?.status === "done" ? "Done" : peer?.status || "Connecting";
    $("#herdr-state").textContent = `Herdr · ${state}${S.herdrOmitted ? " · Some large messages omitted" : ""}`;
    renderHeader();
    renderStatus();
  }

  async function sendHerdr(text, images) {
    if (images.length) { OMP.toast("Herdr messages support text only", "warn"); return false; }
    if (!text.trim() || S.herdrSending) return false;
    const pane = S.herdrPane, ticket = {};
    S.herdrSending = ticket;
    try {
      await api("/api/herdr/prompt?pane=" + encodeURIComponent(pane), {method: "POST", body: {message: text}});
      if (S.herdrPane !== pane || S.herdrSending !== ticket) return false;
      OMP.toast("Sent to Herdr", "success");
      return true;
    } catch (e) {
      if (S.herdrSending === ticket) OMP.toast(e.message, "error", 6500);
      return false;
    } finally { if (S.herdrSending === ticket) S.herdrSending = null; }
  }

  $("#herdr-activity-btn").onclick = () => {
    if (!S.herdrPane) return;
    S.herdrActivity = !S.herdrActivity;
    renderHerdrView();
    if (S.herdrActivity) S.refreshHerdrLive?.();
  };

  async function openHerdr(pane) {
    if (S.herdrPane === pane) return;
    connect(null, pane);
    const load = {};
    S.herdrLoad = load;
    const current = () => S.herdrPane === pane && S.herdrLoad === load;
    const url = "/api/herdr/session?pane=" + encodeURIComponent(pane);
    let before = null, lastOffset = null, havePage = false, busy = false;
    async function refreshLive() {
      if (!current() || busy) return;
      busy = true;
      clearTimeout(S.herdrPoll);
      try {
        if (S.view !== "chat" || document.hidden) return;
        if (S.herdrActivity) {
          try {
            const {text} = await api("/api/herdr/activity?pane=" + encodeURIComponent(pane));
            if (!current() || !S.herdrActivity) return;
            const box = $("#herdr-activity"), out = $("#herdr-output");
            if (out.textContent !== text) {
              const stick = box.scrollHeight - box.clientHeight - box.scrollTop < 44;
              out.textContent = text;
              if (stick) box.scrollTop = box.scrollHeight;
            }
          } catch (e) { if (current() && S.herdrActivity) $("#herdr-output").textContent = e.message; }
        }
        const page = await api(url);
        if (!current()) return;
        if (!havePage) {
          transcript.loadMessages(page.messages);
          before = page.before;
          havePage = true;
        } else for (let i = 0; i < page.offsets.length; ++i) {
          if (lastOffset == null || page.offsets[i] > lastOffset) transcript.renderMessage(page.messages[i]);
        }
        lastOffset = page.offsets.at(-1) ?? lastOffset;
        S.herdrOmitted ||= page.omitted;
        S.herdrTokens = page.tokens;
        S.herdrModel = page.model || S.herdrModel;
        S.herdrThinkingLevel = page.thinkingLevel || S.herdrThinkingLevel;
        S.herdrUltra = page.ultra || null;
        updateHerdrStatus();
      } catch { /* A journal can appear after the first prompt; keep the composer usable. */ }
      finally {
        busy = false;
        if (current()) S.herdrPoll = setTimeout(refreshLive, S.herdrActivity ? 1500 : 2500);
      }
    }
    S.refreshHerdrLive = refreshLive;
    async function readPage(older = false) {
      for (;;) {
        const page = await api(url + (older ? "&before=" + encodeURIComponent(before) : ""));
        if (!current()) return null;
        before = page.before;
        const messages = page.messages || [];
        S.herdrOmitted ||= page.omitted;
        if (!older) {
          havePage = true;
          lastOffset = page.offsets.at(-1) ?? lastOffset;
          S.herdrTokens = page.tokens;
          S.herdrModel = page.model || S.herdrModel;
          S.herdrThinkingLevel = page.thinkingLevel || S.herdrThinkingLevel;
          S.herdrUltra = page.ultra || null;
        }
        updateHerdrStatus();
        if (messages.length || before == null) return messages;
        older = true;
      }
    }
    $("#transcript").setAttribute("aria-busy", "true");
    try {
      const messages = await readPage();
      if (!current()) return;
      transcript.loadMessages(messages);
      if (!messages.length) transcript.addSystem("No messages in this Herdr session yet.");
      transcript.setOlderSource(async () => {
        if (before == null || !current()) return null;
        try { const page = await readPage(true); return page?.length ? page : null; }
        catch (e) { if (current()) transcript.addSystem("Could not load older messages: " + e.message, "warn"); return null; }
      });
    } catch (e) {
      if (current()) transcript.addSystem(e.message, "warn");
    } finally {
      if (current()) {
        $("#transcript").removeAttribute("aria-busy");
        S.herdrPoll = setTimeout(refreshLive, 1600);
      }
    }
  }

  async function onOpen(ws) {
    composer.setEnabled(true);
    send({ type: "set_ask_dialog", enabled: true });
    try {
      const [caps, state, cmds] = await Promise.all([
        getCapabilities(),
        rpc("get_state"),
        rpc("get_available_commands"),
      ]);
      if (S.ws !== ws) return;
      S.caps = caps;
      S.state = state;
      const streaming = !!state.isStreaming;
      // Latest-first load for capable agents: one bounded backward page (works
      // during streaming too) renders immediately, older pages fetch lazily.
      // Legacy agents page forward (quiet) or take one full get_messages
      // snapshot (streaming). Cached backward windows render instantly and are
      // completed from a fresh cursor only when older history is requested.
      let messages, backward = false, fetchOlder = null, cachedBackward = false;
      const cacheKey = streaming ? null : sessionCache.key(S.agentId, state.sessionFile);
      const cached = cacheKey ? sessionCache.get(cacheKey) : null;
      if (cached) {
        messages = cached;
        backward = !!(caps.features && caps.features.messagesBackwardPaging);
        cachedBackward = backward;
      } else {
        const loaded = await OMP.loadSessionHistory(rpc, caps);
        if (S.ws !== ws) return;
        messages = loaded.messages;
        backward = loaded.backward;
        fetchOlder = loaded.fetchOlder;
        if (loaded.paged && !loaded.streaming) sessionCache.put(cacheKey, messages);
      }
      if (OMP.agentPanel) OMP.agentPanel.ingestMessages(messages);
      // Inline text completion also learns from this session's own prompts.
      composer.setSessionPrompts(messages.map(userPromptText).filter((t) => t && t.trim()));
      renderWidgets();
      updateHome();
      setCommands(cmds.commands);
      if (backward) {
        // The latest bounded page is the whole fetch so far; older pages come
        // from the agent only on upward scroll.
        transcript.loadMessages(messages);
        let loadedPrefix = messages;
        const initialCachedCount = cachedBackward ? messages.length : 0;
        let refreshed = null;
        transcript.setOlderSource(async () => {
          try {
            if (S.ws !== ws) return null;
            let older;
            if (cachedBackward) {
              // A cached window has no reusable RPC closure. Refresh the latest
              // cursor only on demand, skip pages already in the cache, then
              // return the first genuinely older page (never duplicate it).
              if (!refreshed) refreshed = await OMP.loadSessionHistory(rpc, caps);
              if (S.ws !== ws) return null;
              let skipped = refreshed.messages.length;
              do {
                older = refreshed.fetchOlder ? await refreshed.fetchOlder() : null;
                if (S.ws !== ws) return null;
                if (older) skipped += older.length;
              } while (older && skipped <= initialCachedCount);
            } else {
              older = fetchOlder ? await fetchOlder() : null;
            }
            if (S.ws !== ws) return null;
            if (older && older.length) {
              loadedPrefix = older.concat(loadedPrefix); // full prefix only
              if (OMP.agentPanel) OMP.agentPanel.ingestMessages(loadedPrefix);
              sessionCache.put(cacheKey, loadedPrefix);
            }
            return older;
          } catch (e) {
            if (S.ws === ws) transcript.addSystem("Failed to load older messages: " + e.message, "warn");
            return null;
          }
        });
      } else {
        transcript.loadLatest(messages);
      }
      setStreaming(streaming);
      renderHeader();
      refreshStats();
    } catch (e) {
      if (S.ws === ws) transcript.addSystem("Failed to load session: " + e.message, "error");
    }
  }

  async function refreshState() {
    try { S.state = await rpc("get_state"); } catch { return; }
    renderHeader();
    renderStatus();
    renderWidgets();
    updateHome();
  }
  async function refreshStats() {
    const ws = S.ws;
    try { const stats = await rpc("get_session_stats"); if (S.ws === ws) S.stats = stats; } catch { /* older agent */ }
    renderStatus();
  }

  function setCommands(list) {
    S.commands = list || [];
    const names = new Set(S.commands.map((c) => c.name));
    composer.setCommands([...S.commands, ...LOCAL_COMMANDS.filter((c) => !names.has(c.name))]);
  }

  function setStreaming(on) {
    if (on && !S.streaming) S.runStart = Date.now();
    if (!on) S.toolsRunning = 0;
    S.streaming = on;
    composer.setStreaming(on);
    renderStatus();
  }

  function answerDialog(id, value) {
    send({ type: "extension_ui_response", id, ...value });
  }

  function onFrame(f) {
    // Public frame bus: feature modules (artifacts, desktop bridge, …) observe
    // agent frames here instead of patching this switch.
    window.dispatchEvent(new CustomEvent("omp:frame", { detail: { agentId: S.agentId, frame: f } }));
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
      case "agent_start": sessionCache.markAllDirty(); setStreaming(true); refreshAgents(); return;
      case "agent_end": setStreaming(false); refreshStats(); refreshAgents(); loadSessions(); updateHome(); return;
      case "session_settled": setStreaming(false); refreshAgents(); return;
      case "agent_settled": setStreaming(false); return;
      case "turn_end": refreshStats(); refreshState(); return;
      case "queue_update":
        S.queue = { steering: f.steering || [], followUp: f.followUp || [] };
        return renderQueue();
      case "session_loaded":
        sessionCache.markAllDirty();
        transcript.loadLatest(f.messages);
        refreshState(); refreshStats(); refreshAgents();
        return;
      case "session_reset":
        sessionCache.markAllDirty();
        transcript.reset();
        refreshState(); refreshStats(); refreshAgents();
        return;
      case "thinking_level_changed": case "model_changed": case "session_info": case "models_changed":
        refreshState(); refreshAgents();
        if (f.type === "session_info") loadSessions();
        return;
      case "prompt_result":
        if (f.status === "failed" && f.error) transcript.addSystem(f.error, "error");
        if (f.status === "failed" || f.status === "error") flagError();
        return;
      case "tool_execution_start": case "tool_execution_end":
        S.toolsRunning = Math.max(0, (S.toolsRunning || 0) + (f.type === "tool_execution_start" ? 1 : -1));
        renderStatus();
        return transcript.onEvent(f);
      case "message_update": {
        const d = f.assistantMessageEvent;
        if (!S.liveReply || S.liveReply.id !== f.messageId) S.liveReply = {id: f.messageId, bytes: 0, actual: 0, actualBytes: 0};
        if (/^(text|thinking|toolcall)_delta$/.test(d?.type)) S.liveReply.bytes += (d.delta || "").length;
        const actual = f.liveUsage?.output ?? f.message?.usage?.output;
        if (Number.isFinite(actual) && actual > 0) {
          S.liveReply.actual = actual;
          S.liveReply.actualBytes = S.liveReply.bytes;
        }
        OMP.repaintUsage();
        return transcript.onEvent(f);
      }
      case "message_start": case "message_end":
        if (f.type === "message_end" && f.message?.role === "assistant") {
          S.liveReply = null;
          refreshStats();
        }
        sessionCache.markAllDirty();
        return transcript.onEvent(f);
      case "auto_compaction_end":
        sessionCache.markAllDirty();
        return transcript.onEvent(f);
      default:
        transcript.onEvent(f);
    }
  }

  // Tau shows the error face for a few seconds after a failed prompt.
  function flagError() {
    S.errorUntil = Date.now() + 5000;
    renderStatus();
    setTimeout(renderStatus, 5100);
  }

  function uiRequest(f) {
    const m = f.method;
    if (DIALOG_METHODS.has(m)) {
      const pending = OMP.dialogs.request(f, answerDialog);
      refreshAgents(); // roster "needs input" chip now, not on the next 5s tick
      renderStatus();
      return pending;
    }
    if (m === "cancelDialog") {
      OMP.dialogs.cancel((f.params && f.params.id) || f.targetId || f.id);
      refreshAgents();
      renderStatus();
      return;
    }
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
      const messages = await getMessages(); // full refetch, never the cache
      if (OMP.agentPanel) OMP.agentPanel.ingestMessages(messages);
      // Inline text completion also learns from this session's own prompts.
      composer.setSessionPrompts(messages.map(userPromptText).filter((t) => t && t.trim()));
      transcript.loadLatest(messages);
    } catch { /* ignore */ }
  }

  function showDead(f) {
    const info = S.agents.find((a) => a.id === S.agentId);
    const state = S.state || {};
    S.agentInfo = {
      ...S.agentInfo, ...info,
      sessionFile: state.sessionFile || info?.sessionFile || S.agentInfo?.sessionFile,
      sessionName: state.sessionName || info?.sessionName || S.agentInfo?.sessionName,
      model: state.model?.id ? `${state.model.provider}/${state.model.id}` : info?.model || S.agentInfo?.model,
    };
    S.dead = f || {};
    setStreaming(false);
    renderSidebar();
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
  // Commands that need no agent, offered on the home screen before any
  // session is open (the terminal surface lists them there too).
  const HOME_COMMANDS = new Set(["new", "resume", "hotkeys", "help"]);

  /** Plain text of a user message (string content or text blocks). */
  function userPromptText(m) {
    if (!m || m.role !== "user") return "";
    const c = m.content;
    if (typeof c === "string") return c;
    if (Array.isArray(c)) return c.filter((b) => b && b.type === "text").map((b) => b.text).join(" ");
    return "";
  }

  function submit(text, images, mode) {
    if (S.herdrPane) return sendHerdr(text, images);
    const trimmed = text.trim();
    if (!S.agentId || S.dead) {
      if (trimmed.startsWith("/") && !images.length) {
        const name = trimmed.slice(1).split(/\s+/)[0];
        if (!name.includes("/") && HOME_COMMANDS.has(name)
          && localCommand(name, trimmed.slice(1 + name.length).trim())) return true;
      }
      OMP.toast("Start or open a session first", "warning");
      return false;
    }
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
  OMP.repaintUsage = OMP.rafThrottle(renderStatus);
  function renderSidebar() { OMP.shellImpl.renderSidebar(S); }
  function renderQueue() { OMP.shellImpl.renderQueue(S); }
  function renderWidgets() { OMP.shellImpl.renderWidgets(S); }
  // Kimi-style home/empty state: center the composer until a session has
  // messages, then dock it at the bottom (see #app.home rules in composer.css).
  function updateHome() {
    const transcriptEl = $("#transcript");
    const hasAny = !!S.herdrPane || (!!S.agentId && ((transcriptEl && transcriptEl.childElementCount > 0) || (S.state && (S.state.userMessages || 0) > 0)));
    $("#app").classList.toggle("home", !hasAny);
  }
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

  OMP.app = { S, rpc, send, sendHerdr, connect, openHerdr, updateHerdrStatus, abort, transcript, composer, currentCwd, setCommands, localCommand, getMessages, getCapabilities, sessionCache, setView };
  if (OMP.agentPanel && typeof OMP.agentPanel.init === "function") OMP.agentPanel.init(OMP.app);
  OMP.shellImpl.init(OMP.app);
  // Pane boot: "/?agent=<id>&pane=1" is how the tmux-style pane manager
  // (OMP.panes) embeds sessions — auto-attach to that agent and hide the
  // chrome the pane host already provides.
  {
    const q = new URLSearchParams(location.search);
    const wantAgent = q.get("agent");
    if (q.get("pane") === "1") document.body.classList.add("pane-embed");
    if (wantAgent) {
      S.agentId = wantAgent;
      const attach = () => (S.agents.some((a) => a.id === wantAgent) ? (connect(wantAgent), true) : false);
      if (!attach()) {
        const t0 = Date.now();
        const iv = setInterval(() => { if (attach() || Date.now() - t0 > 30000) clearInterval(iv); }, 1000);
      }
    }
  }
  for (const feature of [OMP.palette, OMP.tasks, OMP.changes, OMP.terminal, OMP.workspaces, OMP.quick, OMP.panes]) {
    if (feature && typeof feature.init === "function") feature.init(OMP.app);
  }
  // Home screen: the composer is live before any agent connects and offers
  // the agent-less commands in autocomplete - the "/" hint shown there must
  // not be a dead promise (terminal parity).
  if (!S.herdrPane && !S.agentId) composer.setEnabled(true);
  setCommands([]);
  updateHome(); // no agent yet: start on the centered home composer
  // Dock/undock whenever the transcript gains or loses content (history replay,
  // first live message, /new) instead of chasing every frame type.
  new MutationObserver(updateHome).observe($("#transcript"), { childList: true });
})(window.OMP);
