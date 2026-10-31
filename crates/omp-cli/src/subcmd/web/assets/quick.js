"use strict";
// Quick screen query (feature 5): a focused "ask about this screenshot" window.
//
// Exports OMP.quick = { init(app), open() }.
//   * init(app): when the page is loaded with ?quick=1 the normal app shell is
//     hidden and a minimal quick-ask UI is shown against a freshly created
//     agent (POST /api/agents in the current/default cwd, driven over its own
//     /ws/<id> socket). Otherwise quick.js only listens for the native
//     `omp:quick-continue` event and continues that agent in the main session.
//   * open(): toggle the native quick window (Tauri `quick_toggle`) or, in a
//     plain browser, open a new tab in quick mode.
//
// Payload shapes match app.js/models.js: rpc frames {id,type,...params},
// models from get_available_models ({provider,id,name}), set_model
// {provider,modelId}, prompt {message, images:[{type:"image",data,mimeType}]}
// with base64 `data`, streamed assistant text via message_start/update/end.
(function (OMP) {
  const { el, $ } = OMP;
  const TAURI = window.__TAURI__ || null;
  const invoke = TAURI && TAURI.core && typeof TAURI.core.invoke === "function"
    ? (cmd, args) => TAURI.core.invoke(cmd, args)
    : null;

  const S = {
    agentId: null,
    cwd: "",
    currentModel: "",
    ws: null,
    seq: 0,
    pending: new Map(),
    streaming: false,
    shot: null, // {base64, url}
    models: [],
    built: false,
    decoder: new OMP.RpcDeltaDecoder(),
    pendingMessages: new Map(),
    flushScheduled: false,
  };

  function isQuick() {
    return new URLSearchParams(location.search).get("quick") === "1";
  }

  // ── init (both modes) ─────────────────────────────────────────────────────
  function init(app) {
    appRef = app;
    if (!isQuick() && TAURI?.event?.listen) {
      TAURI.event.listen("omp:quick-continue", (event) => {
        window.dispatchEvent(new CustomEvent("omp:quick-continue", { detail: event.payload }));
      }).catch((err) => console.error("quick continuation listener", err));
    }
    // Main mode: a "Continue in main session" handoff from the quick window.
    window.addEventListener("omp:quick-continue", (e) => {
      const { id } = e.detail || {};
      if (!id) return;
      try {
        history.replaceState(null, "", "#/a/" + encodeURIComponent(id));
        app.connect(id);
        OMP.toast("Continued quick session", "success");
      } catch (err) { OMP.toast(err.message, "error"); }
    });
    // Browser fallback handoff: the quick tab stashed the agent and navigated
    // here; pick it up once the app is ready to connect.
    try {
      const pending = sessionStorage.getItem("omp-quick-continue");
      if (pending && !isQuick()) {
        sessionStorage.removeItem("omp-quick-continue");
        const { id } = JSON.parse(pending) || {};
        if (id) setTimeout(() => {
          try {
            history.replaceState(null, "", "#/a/" + encodeURIComponent(id));
            app.connect(id);
            OMP.toast("Continued quick session", "success");
          } catch (e) { OMP.toast(e.message, "error"); }
        }, 300);
      }
    } catch { /* sessionStorage unavailable */ }

    if (isQuick()) enterQuick();
  }

  function open() {
    if (invoke) { invoke("quick_toggle").catch((e) => OMP.toast(String(e), "error")); return; }
    window.open(location.pathname + "?quick=1", "_blank", "width=560,height=720");
  }

  // ── quick window mode ─────────────────────────────────────────────────────
  function enterQuick() {
    document.body.classList.add("quick-mode");
    if (!S.built) buildUi();
    createAgent().catch((e) => OMP.toast("Failed to start agent: " + e.message, "error"));
  }

  function cwdGuess() {
    try {
      const s = appS();
      if (s && s.project) return s.project;
    } catch { /* app may not be wired */ }
    return ""; // server falls back to its default cwd
  }
  let appRef = null;
  function appS() { return appRef && appRef.S; }

  async function createAgent() {
    setNote("Starting agent…");
    const body = cwdGuess() ? { cwd: cwdGuess() } : {};
    const info = await OMP.api("/api/agents", { method: "POST", body });
    S.agentId = info.id;
    S.cwd = info.cwd || cwdGuess();
    S.currentModel = info.model || "";
    setNote("");
    connect();
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    S.decoder.reset();
    S.pendingMessages.clear();
    const ws = new WebSocket(`${proto}://${location.host}/ws/${S.agentId}`);
    S.ws = ws;
    ws.onopen = () => { loadModels(); };
    ws.onmessage = (e) => { let f; try { f = JSON.parse(e.data); } catch { return; } onFrame(f); };
    ws.onclose = () => { if (S.ws === ws) { S.ws = null; setNote("Agent connection lost."); } };
  }

  function send(frame) {
    if (!S.ws || S.ws.readyState !== WebSocket.OPEN) return false;
    S.ws.send(JSON.stringify(frame));
    return true;
  }
  function rpc(type, params = {}) {
    const id = "q" + ++S.seq;
    return new Promise((resolve, reject) => {
      if (!send({ id, type, ...params })) return reject(new Error("not connected"));
      S.pending.set(id, { resolve, reject });
    });
  }

  function onFrame(f) {
    f = S.decoder.expand(f);
    if (!f) return;
    if (f.type === "response" && f.id && S.pending.has(f.id)) {
      const p = S.pending.get(f.id);
      S.pending.delete(f.id);
      if (f.success === false) p.reject(new Error(f.error || "command failed"));
      else p.resolve(f.data || {});
      return;
    }
    switch (f.type) {
      case "web_attached":
        if (f.agent) {
          S.cwd = f.agent.cwd || S.cwd;
          S.currentModel = f.agent.model || S.currentModel;
          if (els) renderModelPicker();
        }
        return;
      case "agent_start":
        S.streaming = true;
        setButtons();
        return;
      case "agent_end": case "session_settled": case "agent_settled":
        S.streaming = false;
        setButtons();
        return;
      case "message_start": case "message_update": {
        if (!f.message || f.message.role !== "assistant") return;
        S.pendingMessages.set(f.messageId || "live", f.message);
        scheduleMessageFlush();
        return;
      }
      case "message_end": {
        if (!f.message || f.message.role !== "assistant") return;
        S.pendingMessages.delete(f.messageId || "live");
        if (f.message.errorMessage) OMP.toast(f.message.errorMessage, "error");
        answerEl(f.messageId).replaceChildren(renderText(OMP.contentText(f.message.content)));
        return;
      }
      case "prompt_result":
        if ((f.status === "failed" || f.status === "error") && f.error) OMP.toast(f.error, "error");
        return;
      case "web_agent_exit":
        S.ws = null;
        setNote("Agent stopped.");
        return;
    }
  }


  function scheduleMessageFlush() {
    if (S.flushScheduled) return;
    S.flushScheduled = true;
    requestAnimationFrame(() => {
      S.flushScheduled = false;
      const pending = [...S.pendingMessages];
      S.pendingMessages.clear();
      for (const [id, message] of pending) {
        answerEl(id === "live" ? null : id).replaceChildren(renderText(OMP.contentText(message.content)));
      }
    });
  }
  function renderText(text) {
    if (OMP.Markdown && OMP.Markdown.render) {
      const div = el("div.qa-md");
      OMP.Markdown.render(div, text);
      return div;
    }
    return el("div.qa-md", { text: text || "" });
  }

  async function loadModels() {
    try {
      const { models } = await rpc("get_available_models");
      S.models = models || [];
      const state = await rpc("get_state").catch(() => null);
      if (state?.model) S.currentModel = `${state.model.provider}/${state.model.id}`;
    } catch { S.models = []; }
    renderModelPicker();
  }

  // ── UI ────────────────────────────────────────────────────────────────────
  let els = null;
  function buildUi() {
    S.built = true;
    const root = el("div", { id: "quick-root", role: "dialog", "aria-label": "Quick screen query" });
    const ta = el("textarea.qa-input", {
      placeholder: "Ask about the screenshot… (Enter to send, Shift+Enter for a new line)",
      "aria-label": "Question", rows: 3, spellcheck: false,
      onkeydown: (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } },
    });
    const answer = el("div.qa-answer", { "aria-live": "polite" });
    const shotRow = el("div.qa-shot", { hidden: true });
    const modelSel = el("select.qa-model", { "aria-label": "Model", onchange: () => pickModel(modelSel.value) });
    const note = el("div.qa-note", { hidden: true });
    els = { ta, answer, shotRow, modelSel, note };
    root.append(
      el("header.qa-head", null,
        el("span.qa-title", { text: "Quick ask" }),
        el("span.qa-cwd", { text: cwdGuess() || "default folder" }),
        modelSel,
        el("button.qa-icon", { text: "✕", title: "Close", "aria-label": "Close", onclick: () => invoke ? invoke("quick_toggle").catch((e) => OMP.toast(String(e), "error")) : window.close() })),
      answer,
      note,
      shotRow,
      ta,
      el("footer.qa-foot", null,
        el("button.qa-icon", { text: "📸 Screenshot", title: "Capture the screen and attach it", onclick: attachShot }),
        el("span.f", null),
        el("button.qa-stop", { text: "Stop", hidden: true, onclick: stop }),
        el("button.qa-continue", { text: "Continue in main session", hidden: true, onclick: continueMain, title: "Keep this agent and go on in the main window" }),
        el("button.qa-ask", { text: "Ask", onclick: () => ask() })),
    );
    document.body.append(root);
  }

  function answerEl(messageId) {
    const key = messageId || "live";
    let node = els.answer.querySelector(`[data-mid="${key}"]`);
    if (!node) {
      node = el("div.qa-msg", { dataset: { mid: key } });
      els.answer.append(node);
    }
    els.answer.scrollTop = els.answer.scrollHeight;
    return node;
  }

  function addEcho(text) {
    els.answer.append(el("div.qa-echo", { text }));
    els.answer.scrollTop = els.answer.scrollHeight;
  }

  function renderModelPicker() {
    const sel = els.modelSel;
    sel.replaceChildren();
    if (!S.models.length) { sel.hidden = true; return; }
    sel.hidden = false;
    const providers = [...new Set(S.models.map((m) => m.provider))].sort();
    for (const p of providers) {
      const group = el("optgroup", { label: p });
      for (const m of S.models.filter((x) => x.provider === p)) {
        group.append(el("option", { value: `${m.provider}/${m.id}`, text: m.name || m.id }));
      }
      sel.append(group);
    }
    // Prefer the model the agent already runs, else keep the first entry.
    const current = S.currentModel;
    const match = S.models.find((m) => current && (m.id === current || `${m.provider}/${m.id}` === current));
    sel.value = match ? `${match.provider}/${match.id}` : sel.options[0] && sel.options[0].value;
  }

  async function pickModel(value) {
    if (!value) return;
    const [provider, modelId] = value.split("/");
    try { await rpc("set_model", { provider, modelId }); S.currentModel = `${provider}/${modelId}`; } catch (e) { OMP.toast("Model switch failed: " + e.message, "error"); }
  }

  async function attachShot() {
    if (!invoke) return OMP.toast("Screenshots need the desktop app", "warning");
    try {
      const url = await invoke("quick_capture");
      if (!url || typeof url !== "string") throw new Error("capture failed");
      const base64 = url.includes(",") ? url.split(",")[1] : url;
      S.shot = { base64, url };
      els.shotRow.replaceChildren(
        el("img.qa-shot-img", { src: url, alt: "Screenshot attachment" }),
        el("button.qa-icon", { text: "Remove", onclick: removeShot }),
      );
      els.shotRow.hidden = false;
    } catch (e) { OMP.toast("Screenshot failed: " + e, "error"); }
  }

  function removeShot() {
    S.shot = null;
    els.shotRow.hidden = true;
    els.shotRow.replaceChildren();
  }

  async function ask() {
    const text = els.ta.value.trim();
    if (!text && !S.shot) return;
    if (S.streaming) return OMP.toast("Still answering — Stop first", "warning");
    els.ta.value = "";
    addEcho(S.shot ? `${text || "(screenshot)"}  [screenshot attached]` : text);
    const images = S.shot ? [{ type: "image", data: S.shot.base64, mimeType: "image/png" }] : undefined;
    removeShot();
    try {
      const params = { message: text };
      if (images) params.images = images;
      await rpc("prompt", params);
    } catch (e) {
      OMP.toast("Prompt failed: " + e.message, "error");
    }
    setButtons();
  }

  async function stop() {
    try { await rpc("abort"); } catch { /* already idle */ }
  }

  function setButtons() {
    if (!els) return;
    $(".qa-stop", document).hidden = !S.streaming;
    // "Continue" appears once there is a session to hand off.
    $(".qa-continue", document).hidden = !S.agentId;
  }

  function setNote(msg) {
    if (!els) return;
    els.note.textContent = msg || "";
    els.note.hidden = !msg;
  }

  function continueMain() {
    if (!S.agentId) return;
    if (invoke) {
      invoke("continue_main", { id: S.agentId, cwd: S.cwd }).catch((e) => OMP.toast(String(e), "error"));
      return;
    }
    // Browser fallback: keep the agent (do not DELETE it) and navigate the
    // main tab; quick.js there picks the handoff up from sessionStorage.
    try { sessionStorage.setItem("omp-quick-continue", JSON.stringify({ id: S.agentId, cwd: S.cwd })); } catch { /* ignore */ }
    location.href = location.pathname;
  }

  OMP.quick = { init, open };
})(window.OMP);
