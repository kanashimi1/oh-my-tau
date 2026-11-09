"use strict";
// The conversation view: renders stored messages and live rpc events.
(function (OMP) {
  const { el } = OMP;

  // Display name of a model: strip any "provider/" prefix ("kimi-code/k3.1-pro"
  // -> "k3.1-pro"); the full id stays available via the element title.
  const modelName = (m) => {
    const s = String(m || "");
    const i = s.lastIndexOf("/");
    return i > 0 ? s.slice(i + 1) : s;
  };
  const COPY_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';


  const utf8 = new TextEncoder();
  const utf8Decode = new TextDecoder("utf-8", { fatal: true });

  // RPC deltas carry UTF-8 byte offsets. Keep the ASCII path cheap, but never
  // apply those offsets as JavaScript UTF-16 string indexes.
  function applyTextDelta(base, d) {
    if (!d || !Number.isSafeInteger(d.n) || !Number.isSafeInteger(d.p) || !Number.isSafeInteger(d.q) || typeof d.s !== "string") return null;
    if (base.length === d.n && d.p >= 0 && d.q >= 0 && d.p + d.q <= base.length) {
      return base.slice(0, d.p) + d.s + base.slice(base.length - d.q);
    }
    try {
      const prev = utf8.encode(base), insert = utf8.encode(d.s);
      if (prev.length !== d.n || d.p < 0 || d.q < 0 || d.p + d.q > prev.length) return null;
      const out = new Uint8Array(d.p + insert.length + d.q);
      out.set(prev.subarray(0, d.p), 0);
      out.set(insert, d.p);
      out.set(prev.subarray(prev.length - d.q), d.p + insert.length);
      return utf8Decode.decode(out);
    } catch { return null; }
  }

  class RpcDeltaDecoder {
    constructor() { this.reset(); }
    reset() { this.messageJson = null; }
    expand(frame) {
      if (["message_start", "message_end", "agent_start", "agent_end"].includes(frame.type)) this.reset();
      if (frame.type !== "message_update") return frame;
      if (frame.message) {
        this.messageJson = JSON.stringify(frame.message);
        return frame;
      }
      if (!frame.messageDelta || this.messageJson == null) { this.reset(); return null; }
      const json = applyTextDelta(this.messageJson, frame.messageDelta);
      if (json == null) { this.reset(); return null; }
      try {
        const message = JSON.parse(json);
        this.messageJson = json;
        return { ...frame, message };
      } catch { this.reset(); return null; }
    }
    seed(message) { this.messageJson = JSON.stringify(message); }
  }
  OMP.RpcDeltaDecoder = RpcDeltaDecoder;

  // Only blank lines outside fences are safe streaming checkpoints. The final
  // render still reparses the complete source, so Markdown spanning a checkpoint
  // cannot change the settled output.
  function stableCutoff(text) {
    let offset = 0, cut = 0, fence = null;
    for (;;) {
      const nl = text.indexOf("\n", offset);
      const end = nl < 0 ? text.length : nl + 1;
      const line = text.slice(offset, nl < 0 ? text.length : nl);
      const match = line.match(/^\s*(`{3,}|~{3,})/);
      if (match) {
        if (!fence) fence = match[1];
        else if (match[1][0] === fence[0] && match[1].length >= fence.length) fence = null;
      } else if (!fence && !line.trim()) cut = end;
      offset = end;
      if (nl < 0) return cut;
    }
  }

  class StreamingMarkdown {
    constructor(node) {
      this.node = node;
      this.scope = OMP.mdScope();
      this.streaming = false;
      this.finalSource = null;
      this.stableSource = "";
      this.chunk = 0;
    }
    link(node, opts) { if (OMP.linkifyPaths) OMP.linkifyPaths(node, opts); }
    begin() {
      if (this.streaming) return;
      this.streaming = true;
      this.finalSource = null;
      this.stableSource = "";
      this.chunk = 0;
      this.stable = el("div.stream-stable");
      this.tail = el("div.stream-tail");
      this.node.replaceChildren(this.stable, this.tail);
    }
    set(text, done) {
      if (done) {
        if (this.finalSource === text) return;
        this.streaming = false;
        this.finalSource = text;
        this.node.innerHTML = OMP.md(text, `${this.scope}-final`);
        this.link(this.node);
        return;
      }
      this.begin();
      const cut = stableCutoff(text), stable = text.slice(0, cut), tail = text.slice(cut);
      if (!stable.startsWith(this.stableSource)) {
        this.stableSource = "";
        this.stable.replaceChildren();
      }
      if (stable.length > this.stableSource.length) {
        const addition = stable.slice(this.stableSource.length);
        const template = document.createElement("template");
        template.innerHTML = OMP.md(addition, `${this.scope}-stable-${this.chunk++}`);
        const fragment = template.content;
        this.link(fragment);
        this.stable.append(fragment);
        this.stableSource = stable;
      }
      // A huge still-open paragraph/fence is rendered as text until a stable
      // checkpoint or message_end; this bounds work even for one giant block.
      if (tail.length > 32768) {
        this.tail.textContent = tail;
        this.tail.style.whiteSpace = "pre-wrap";
      } else {
        this.tail.style.whiteSpace = "";
        this.tail.innerHTML = OMP.md(tail, `${this.scope}-tail`);
        this.link(this.tail, { media: false });
      }
    }
  }

  // Collapsible thinking block: live mm:ss timer while streaming ("Thinking… 0:12",
  // expanded showing the tail), "Thought for 12s" collapsed once finished. History
  // replay without timing data shows plain "Thought". Redacted gets a neutral label.
  class ThinkingBlock {
    constructor(redacted, live) {
      this.redacted = redacted;
      this.start = live && !redacted ? Date.now() : null;
      this.done = false;
      this.last = null;
      this.open = !!this.start;
      this.head = el("div.thinking-head", { onclick: () => this.toggle() });
      this.body = el("div.md");
      this.inner = el("div.thinking-inner", null, this.body);
      this.node = el("div.thinking", null, this.head, el("div.thinking-clip", null, this.inner));
      this.timer = this.start ? setInterval(() => this.render(), 1000) : null;
      this.render();
    }
    dispose() { if (this.timer) { clearInterval(this.timer); this.timer = null; } }
    toggle(force) { this.open = force ?? !this.open; this.render(); }
    seconds() { return this.start ? Math.floor(((this.end || Date.now()) - this.start) / 1000) : null; }
    finish() { if (this.done) return; this.done = true; if (this.start) this.end = Date.now(); this.dispose(); this.open = false; this.render(); }
    label() {
      if (this.redacted) return "Thinking (redacted)";
      if (this.done) { const s = this.seconds(); return s == null || s < 1 ? "Thought" : `Thought for ${s}s`; }
      const s = this.seconds() || 0;
      return `Thinking… ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }
    setText(text, live) {
      if (!this.redacted) {
        this.markdown ??= new StreamingMarkdown(this.body);
        this.markdown.set(text, !live);
        this.inner.scrollTop = this.inner.scrollHeight;
      }
      if (live && !this.done) this.open = true;
      else this.finish();
      this.render();
    }
    render() {
      this.head.textContent = this.label();
      this.node.classList.toggle("open", this.open);
      this.node.classList.toggle("redacted", this.redacted);
      this.node.classList.toggle("live", !this.done && !this.redacted);
    }
  }

  // ── history paging (latest-first session load) ───────────────────────────
  // `get_messages_page` walks a stable snapshot forward (oldest → newest) with
  // an opaque cursor and refuses to page while the agent is streaming. These
  // helpers keep every response bounded (no full `get_messages` on a quiet
  // session), let the UI render only the newest window first, and feed older
  // pages to the transcript on upward scroll.

  // Shape shared with the agent's `legacy_capabilities_payload()`: what the
  // frontend must assume when `get_capabilities` is rejected by an older
  // detached agent. The legacy page limit is deliberately 256.
  const LEGACY_CAPABILITIES = Object.freeze({
    protocolVersion: 1,
    supportedProtocolVersions: Object.freeze([1]),
    maxFrameBytes: 1 << 20,
    maxReassembledFrameBytes: 16 << 20,
    limits: Object.freeze({
      getMessages: Object.freeze({
        defaultPageSize: 100,
        maxPageSize: 256,
        maxPageBytes: 4 << 20,
        maxCursorChars: 2048,
      }),
    }),
  });

  OMP.legacyCapabilities = () => JSON.parse(JSON.stringify(LEGACY_CAPABILITIES));

  // Normalize a `get_capabilities` response (or its absence) into the same
  // shape, filling gaps from the conservative legacy profile. Never throws.
  OMP.normalizeCapabilities = (caps) => {
    const top = caps && typeof caps === "object" ? caps : {};
    const gl = top.limits && typeof top.limits === "object" ? top.limits.getMessages || {} : {};
    const pick = (v, fallback) => (typeof v === "number" && v > 0 ? v : fallback);
    return {
      protocolVersion: pick(top.protocolVersion, LEGACY_CAPABILITIES.protocolVersion),
      supportedProtocolVersions: Array.isArray(top.supportedProtocolVersions) && top.supportedProtocolVersions.length
        ? [...top.supportedProtocolVersions]
        : [...LEGACY_CAPABILITIES.supportedProtocolVersions],
      maxFrameBytes: pick(top.maxFrameBytes, LEGACY_CAPABILITIES.maxFrameBytes),
      maxReassembledFrameBytes: pick(top.maxReassembledFrameBytes, LEGACY_CAPABILITIES.maxReassembledFrameBytes),
      limits: {
        getMessages: {
          defaultPageSize: pick(gl.defaultPageSize, LEGACY_CAPABILITIES.limits.getMessages.defaultPageSize),
          maxPageSize: pick(gl.maxPageSize, LEGACY_CAPABILITIES.limits.getMessages.maxPageSize),
          maxPageBytes: pick(gl.maxPageBytes, LEGACY_CAPABILITIES.limits.getMessages.maxPageBytes),
          maxCursorChars: pick(gl.maxCursorChars, LEGACY_CAPABILITIES.limits.getMessages.maxCursorChars),
        },
      },
      features: {
        // Backend opt-in; the legacy profile never claims it.
        messagesBackwardPaging: !!(top.features && top.features.messagesBackwardPaging),
      },
    };
  };

  /** Bounded paging over the message history.
   *  - forward (legacy agents): walks oldest → newest to the end of the
   *    snapshot; the full list is then windowed client-side.
   *  - backward (capabilities.features.messagesBackwardPaging): each response
   *    is chronological and its nextCursor points at an older offset, so the
   *    newest bounded page can render immediately and older pages are fetched
   *    lazily on upward scroll. */
  class HistoryBuffer {
    constructor(caps, { backward = false } = {}) {
      const n = OMP.normalizeCapabilities(caps);
      this.backward = backward;
      const requested = backward ? n.limits.getMessages.defaultPageSize : n.limits.getMessages.maxPageSize;
      this.pageLimit = Math.max(1, Math.min(requested, n.limits.getMessages.maxPageSize, 512));
      this.pages = []; // forward mode: chronological chunks, one per response
      this.total = null;
      this.cursor = null;
      this.done = false;
    }
    /** One paged request; keeps the working set out of huge single frames. */
    async fetchPage(page) {
      const params = this.backward
        ? { direction: "backward", limit: this.pageLimit, ...(this.cursor ? { cursor: this.cursor } : {}) }
        : { limit: this.pageLimit, ...(this.cursor ? { cursor: this.cursor } : {}) };
      const p = await page(params);
      const messages = Array.isArray(p && p.messages) ? p.messages : [];
      if (!this.backward) this.pages.push(messages);
      this.total = (p && p.totalMessages) ?? (this.total == null ? null : this.total + messages.length);
      this.cursor = (p && p.nextCursor) || null;
      this.done = !this.cursor;
      return messages;
    }
    /** Fetch the whole snapshot one bounded page at a time (forward mode). */
    async loadAll(page) {
      while (!this.done) await this.fetchPage(page);
      return this;
    }
    /** Full snapshot in conversation order (forward mode). */
    messages() { return [].concat(...this.pages); }
  }
  OMP.HistoryBuffer = HistoryBuffer;

  /**
   * Load session history for rendering.
   * - Backward-capable agents: resolve with only the newest bounded page plus
   *   a `fetchOlder()` continuation — no full snapshot is ever taken.
   * - Legacy agents: page forward through the whole snapshot (bounded
   *   responses instead of one giant `get_messages`).
   * - Either way, if paging fails because the session started streaming
   *   mid-load (`session_busy`), fall back to a single `get_messages` snapshot.
   */
  OMP.loadSessionHistory = async function (rpc, caps) {
    const backward = !!(caps && caps.features && caps.features.messagesBackwardPaging);
    try {
      const buffer = new HistoryBuffer(caps, { backward });
      const latest = await buffer.fetchPage((params) => rpc("get_messages_page", params));
      if (!backward) {
        await buffer.loadAll((params) => rpc("get_messages_page", params));
        return { messages: buffer.messages(), paged: true, backward: false, streaming: false, hasOlder: false, fetchOlder: null, pageLimit: buffer.pageLimit };
      }
      return {
        messages: latest,
        paged: true, backward: true, streaming: false,
        get hasOlder() { return !buffer.done; },
        pageLimit: buffer.pageLimit,
        // Resolves with the next older chunk (chronological) or null at the
        // start of history; safe to call repeatedly until null.
        fetchOlder: async () => (buffer.done ? null : buffer.fetchPage((params) => rpc("get_messages_page", params))),
      };
    } catch (error) {
      const state = await rpc("get_state").catch(() => null);
      if (!state || !state.isStreaming) throw error;
      const full = await rpc("get_messages");
      return { messages: full.messages || [], paged: false, backward: false, streaming: true, hasOlder: false, fetchOlder: null, pageLimit: OMP.normalizeCapabilities(caps).limits.getMessages.maxPageSize };
    }
  };

  /**
   * Bounded LRU cache of loaded session histories, so revisiting a session can
   * skip the paging round-trips. Entries are never reused once any live frame
   * may have changed history (`markAllDirty`), when stale beyond `ttlMs`, or
   * when oversized; `maxEntries` bounds total memory.
   */
  class SessionHistoryCache {
    constructor({ maxEntries = 4, maxMessages = 4000, ttlMs = 60000, now = () => Date.now() } = {}) {
      this.maxEntries = Math.max(1, maxEntries);
      this.maxMessages = Math.max(1, maxMessages);
      this.ttlMs = ttlMs;
      this.now = now;
      this.map = new Map(); // key -> {messages, at, dirty}
    }
    key(agentId, sessionFile) { return `${agentId || ""}\u0000${sessionFile || ""}`; }
    get(k) {
      const e = this.map.get(k);
      if (!e || e.dirty) return null;
      if (this.now() - e.at > this.ttlMs) { this.map.delete(k); return null; }
      this.map.delete(k); this.map.set(k, e); // LRU touch
      return e.messages;
    }
    put(k, messages) {
      if (!Array.isArray(messages) || messages.length > this.maxMessages) return;
      this.map.delete(k);
      this.map.set(k, { messages, at: this.now(), dirty: false });
      while (this.map.size > this.maxEntries) this.map.delete(this.map.keys().next().value);
    }
    /** Any frame that may mutate history invalidates every cached session. */
    markAllDirty() { for (const e of this.map.values()) e.dirty = true; }
    clear() { this.map.clear(); }
  }
  OMP.SessionHistoryCache = SessionHistoryCache;

  class Transcript {
    constructor(root, scroller) {
      this.root = root;
      this.scroller = scroller;
      this.decoder = new RpcDeltaDecoder();
      this.pendingOlder = []; // not-yet-rendered older history, chronological
      this.loadingOlder = false;
      this.batchTarget = null; // set while rendering into a detached fragment
      this.reset();
      this.scheduleScroll = OMP.rafThrottle(() => this.scrollToBottom());
      this.scheduleFlush = OMP.rafThrottle(() => this.flushUpdates());
      scroller.addEventListener?.("scroll", () => this.maybeLoadOlder());
    }

    flushUpdates() {
      const updates = [...this.pendingUpdates];
      this.pendingUpdates.clear();
      for (const [id, msg] of updates) this.updateAssistant(this.assistantEntry(id), msg, true);
    }

    reset() {
      for (const e of (this.live ? this.live.values() : [])) if (e.timer) { clearInterval(e.timer); e.timer = null; }
      this.root.replaceChildren();
      this.live = new Map(); // messageId -> {node, parts, msg}
      this.cards = new Map(); // toolCallId -> ToolCard
      this.pendingUpdates = new Map(); // messageId -> newest streaming message
      this.turnStartedAt = null; // timestamp of the latest user prompt (total response time)
      this.pendingOlder = [];
      this.loadingOlder = false;
      this.batchTarget = null;
      this.olderSource = null;
      this.windowStart = null; // first node of the latest window (insert marker)
      this.decoder.reset();
      this.stick = true;
    }

    isNearBottom() {
      const s = this.scroller;
      return s.scrollHeight - s.scrollTop - s.clientHeight < 120;
    }
    scrollToBottom(force) {
      if (force || this.stick) this.scroller.scrollTop = this.scroller.scrollHeight;
    }
    append(node) {
      if (this.batchTarget) { this.batchTarget.append(node); return node; }
      this.stick = this.isNearBottom() || this.stick;
      this.root.append(node);
      this.scheduleScroll();
      return node;
    }

    loadMessages(messages) {
      this.reset();
      for (const m of messages || []) this.renderMessage(m);
      // get_messages is the keyframe for a browser attaching mid-stream.
      const current = (messages || []).findLast((m) => m && m.role === "assistant");
      if (current) this.decoder.seed(current);
      this.windowStart = this.root.firstChild;
      this.stick = true;
      this.scrollToBottom(true);
    }

    /**
     * Latest-first load: render only the newest `windowSize` messages now and
     * hold the older ones for prepending on upward scroll. Conversation order
     * is preserved; live frames keep appending after the loaded window.
     */
    loadLatest(messages, windowSize = 256) {
      this.reset();
      const list = Array.isArray(messages) ? messages : [];
      const cut = Math.max(0, list.length - Math.max(1, windowSize));
      this.pendingOlder = list.slice(0, cut);
      for (const m of list.slice(cut)) this.renderMessage(m);
      const current = list.findLast((m) => m && m.role === "assistant");
      if (current) this.decoder.seed(current);
      this.windowStart = this.root.firstChild;
      this.stick = true;
      this.scrollToBottom(true);
      this.fillViewport();
    }

    /** Very short sessions may not fill the viewport yet; keep going. */
    fillViewport() {
      let guard = 64;
      while (this.pendingOlder.length && guard-- > 0) {
        const s = this.scroller;
        if (s.scrollHeight > s.clientHeight + 1) break;
        this.prependOlder();
      }
    }

    /** Render the next older chunk (chronologically before what is shown).
     *  atStart=false (default, forward pendingOlder): chunks come off the head
     *  oldest-first, each newer than the previous one, so insertion goes just
     *  before the original latest window. atStart=true (backward source):
     *  chunks arrive nearest-the-window first and each one is older still, so
     *  insertion goes at the very start of the rendered history. */
    prependOlder(chunkSize = 256, list = null, atStart = false) {
      if (this.loadingOlder) return null;
      const chunk = list || (this.pendingOlder.length ? this.pendingOlder.splice(0, Math.max(1, chunkSize)) : null);
      if (!chunk || !chunk.length) return null;
      this.loadingOlder = true;
      try {
        const s = this.scroller;
        const prevHeight = s.scrollHeight, prevTop = s.scrollTop;
        const frag = document.createDocumentFragment();
        this.batchTarget = frag;
        try { for (const m of chunk) this.renderMessage(m); } finally { this.batchTarget = null; }
        const at = atStart ? this.root.firstChild : (this.windowStart || this.root.firstChild);
        this.root.insertBefore(frag, at);
        // Keep the viewport anchored on the same content despite the growth.
        s.scrollTop = prevTop + (s.scrollHeight - prevHeight);
        this.stick = false;
        return chunk;
      } finally { this.loadingOlder = false; }
    }

    /** Register a lazy source of older pages (backward paging): resolves with
     *  a chronological chunk or null at the start of history. */
    setOlderSource(fn) { this.olderSource = fn; }

    /** Upward scroll pulls in older pages: first from the already-fetched
     *  snapshot, then (backward paging) from the agent via the older source. */
    maybeLoadOlder() {
      if (this.loadingOlder || this.scroller.scrollTop > 200) return;
      if (this.pendingOlder.length) { this.prependOlder(128); return; }
      const src = this.olderSource;
      if (!src) return;
      this.loadingOlder = true;
      return Promise.resolve()
        .then(src)
        .then((older) => {
          if (older && older.length) {
            this.loadingOlder = false; // prependOlder guards on the same flag
            this.prependOlder(older.length, older, true);
          } else if (!older) this.olderSource = null; // history start reached
        })
        .catch(() => { this.olderSource = null; })
        .finally(() => { this.loadingOlder = false; });
    }

    addSystem(text, cls = "") {
      const long = String(text).split("\n").length > 18;
      return this.append(el("div.system" + (cls ? "." + cls : "") + (long ? ".collapsible" : ""), { text }));
    }
    addDivider(text) {
      return this.append(el("div.divider", { text }));
    }
    addError(text) {
      return this.append(el("div.error-box", { text }));
    }

    card(id, name) {
      let c = this.cards.get(id);
      if (!c) {
        c = new OMP.ToolCard(id, name || "tool");
        this.cards.set(id, c);
      }
      return c;
    }

    /** Compact vertical step list for consecutive tool calls of one assistant turn. */
    toolGroup(entry, beforeNode) {
      if (entry.group) return entry.group;
      const g = { node: el("div.tool-group"), steps: el("div.tool-steps"), header: null, collapsed: false };
      g.node.append(g.steps);
      g.refresh = () => {
        const n = g.steps.children.length;
        if (n > 3 && !g.header) {
          g.collapsed = true; // many steps: fold into an "N steps" group
          g.header = el("div.group-head", {
            onclick: () => { g.collapsed = !g.collapsed; g.refresh(); },
          });
          g.node.prepend(g.header);
        }
        if (g.header) {
          g.header.replaceChildren(el("span.group-count", { text: `${n} step${n === 1 ? "" : "s"}` }));
          g.node.classList.toggle("collapsed", g.collapsed);
          g.node.classList.toggle("expanded", !g.collapsed);
        }
      };
      entry.node.insertBefore(g.node, beforeNode || null);
      entry.group = g;
      return g;
    }

    /** Render one complete message (history or message_end without a live node). */
    renderMessage(msg, messageId) {
      if (!msg || typeof msg !== "object") return;
      switch (msg.role) {
        case "user":
          if (typeof msg.timestamp === "number") this.turnStartedAt = msg.timestamp;
          return this.renderUser(msg);
        case "assistant": {
          const entry = this.assistantEntry(messageId);
          this.updateAssistant(entry, msg, false);
          return entry.node;
        }
        case "toolResult": return this.toolResult(msg);
        case "bashExecution": {
          const c = this.card("bash-" + (msg.timestamp || Math.random()), "bash");
          c.setArgs({ command: msg.command });
          const out = (msg.output || "") + (msg.exitCode ? `\n[exit ${msg.exitCode}]` : "") + (msg.cancelled ? "\n[cancelled]" : "");
          c.finish({ content: [{ type: "text", text: out }] }, !!msg.exitCode);
          return this.append(c.node);
        }
        case "compactionSummary": case "branchSummary": {
          const label = msg.role === "compactionSummary"
            ? `Context compacted${msg.tokensBefore ? ` (${OMP.fmtTokens(msg.tokensBefore)} tokens before)` : ""}`
            : "Branch summary";
          const box = el("details.summary-box", null, el("summary", { text: label }), el("div.md", { html: OMP.md(msg.summary || "") }));
          return this.append(box);
        }
        case "custom": case "hookMessage": {
          if (msg.display === false) return;
          const text = typeof msg.content === "string" ? msg.content : OMP.contentText(msg.content);
          if (!text) return;
          return this.append(el("div.system", { text: (msg.customType ? `[${msg.customType}] ` : "") + text }));
        }
        default: return;
      }
    }

    renderUser(msg) {
      const text = OMP.contentText(msg.content);
      const node = el("div.msg.user");
      if (text) node.append(el("div", { text }));
      const imgs = OMP.contentImages(msg.content);
      if (imgs.length) node.append(el("div.imgs", null, imgs.map((i) => el("img", { src: `data:${i.mimeType};base64,${i.data}` }))));
      if (!text && !imgs.length) return;
      return this.append(node);
    }

    toolResult(msg) {
      const c = this.cards.get(msg.toolCallId);
      if (c) {
        c.finish(msg, msg.isError);
        return;
      }
      const card = this.card(msg.toolCallId || "r" + Math.random(), msg.toolName);
      card.finish(msg, msg.isError);
      this.append(card.node);
    }

    assistantEntry(messageId) {
      if (messageId && this.live.has(messageId)) return this.live.get(messageId);
      const entry = { node: el("div.msg.assistant"), parts: [], tail: null, timer: null, startedAt: null, badge: null, done: false };
      if (messageId) this.live.set(messageId, entry);
      this.append(entry.node);
      return entry;
    }

    fmtElapsed(ms) {
      const s = Math.max(0, Math.floor(ms / 1000));
      return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }

    // Message-level elapsed timer: starts with the first live content, ticks
    // once per second, freezes at message_end. History never gets one.
    startTimer(entry) {
      if (entry.startedAt && !entry.done) return;
      if (entry.timer) { clearInterval(entry.timer); entry.timer = null; }
      entry.done = false;
      entry.startedAt = Date.now();
      if (!entry.badge) entry.badge = el("span.msg-timer");
      entry.badge.classList.add("live");
      entry.badge.textContent = this.fmtElapsed(0);
      if (!entry.badge.isConnected) entry.node.append(entry.badge);
      entry.timer = setInterval(() => {
        entry.badge.textContent = this.fmtElapsed(Date.now() - entry.startedAt);
      }, 1000);
    }

    stopTimer(entry) {
      if (entry.timer) { clearInterval(entry.timer); entry.timer = null; }
      if (!entry.startedAt) return;
      entry.done = true;
      entry.badge.classList.remove("live");
      entry.badge.textContent = this.fmtElapsed(Date.now() - entry.startedAt);
    }

    /** One copy action per assistant message: created once, updated in place
     *  so streaming rerenders never duplicate it. Hidden while streaming. */
    refreshCopy(entry, text, streaming) {
      if (!entry.copy) {
        entry.copy = el("button.msg-copy", { type: "button", title: "Copy response", "aria-label": "Copy response" },
          el("span.msg-copy-icon", { html: COPY_ICON, "aria-hidden": "true" }), el("span.copy-label", { text: "Copy" }));
        entry.copy.addEventListener("click", async () => {
          const label = entry.copy.querySelector(".copy-label");
          const copied = await OMP.copyText(entry.copyText || "");
          if (copied) {
            entry.copy.classList.add("copied");
            if (label) label.textContent = "Copied";
            clearTimeout(entry.copyTimer);
            entry.copyTimer = setTimeout(() => {
              entry.copy.classList.remove("copied");
              if (label) label.textContent = "Copy";
            }, 1600);
          } else {
            if (label) label.textContent = "Copy";
            OMP.toast("Copy failed — clipboard is unavailable", "error");
          }
        });
      }
      entry.copyText = text;
      entry.copy.hidden = streaming || !text;
      if (!entry.copy.hidden && !entry.copy.isConnected) entry.node.append(entry.copy);
    }

    updateAssistant(entry, msg, streaming) {
      const stick = this.isNearBottom();
      const content = Array.isArray(msg.content) ? msg.content : [{ type: "text", text: String(msg.content || "") }];
      if (streaming && content.length) this.startTimer(entry);
      else if (!streaming && entry.badge) this.stopTimer(entry);
      content.forEach((block, i) => {
        let part = entry.parts[i];
        const kind = block.type;
        if (!part || part.kind !== kind) {
          if (part) { if (part.thinking) part.thinking.dispose(); part.node.remove(); }
          part = { kind, node: null, last: null };
          if (kind === "text") { part.node = el("div.md"); part.stream = new StreamingMarkdown(part.node); }
          else if (kind === "thinking" || kind === "redactedThinking") {
            part.thinking = new ThinkingBlock(kind === "redactedThinking", streaming);
            part.node = part.thinking.node;
          } else if (kind === "toolCall") {
            part.card = this.card(block.id, block.name);
            part.node = part.card.node;
          } else part.node = el("div");
          const before = entry.parts.slice(i + 1).find((p) => p && p.node.isConnected);
          if (kind === "toolCall") {
            // Consecutive calls collect into one compact step group.
            const g = this.toolGroup(entry, before ? before.node : entry.tail);
            g.steps.append(part.node);
            g.refresh();
          } else {
            entry.group = null; // a non-tool part closes the current step group
            entry.node.insertBefore(part.node, before ? before.node : entry.tail);
          }
          entry.parts[i] = part;
        }
        if (kind === "text") {
          if (part.last !== block.text || part.streaming !== streaming) {
            part.stream.set(block.text, !streaming);
            part.last = block.text;
            part.streaming = streaming;
          }
        } else if (kind === "thinking") {
          part.thinking.setText(block.thinking || "", streaming && i === content.length - 1);
        } else if (kind === "redactedThinking") {
          part.thinking.setText("", false);
        } else if (kind === "toolCall") {
          part.card.name = block.name || part.card.name;
          part.card.setArgs(block.arguments);
        }
      });
      for (let i = content.length; i < entry.parts.length; i++) if (entry.parts[i]) entry.parts[i].node.remove();
      entry.parts.length = content.length;
      // Exact response text (Markdown source) for the copy action.
      const textSource = content.filter((b) => b.type === "text").map((b) => b.text || "").join("\n\n").trim();
      this.refreshCopy(entry, textSource, streaming);
      if (!streaming) {
        if (entry.tail) entry.tail.remove();
        entry.tail = null;
        if (msg.stopReason === "error" && msg.errorMessage) entry.tail = el("div.error-box", { text: msg.errorMessage });
        else if (msg.stopReason === "aborted") entry.tail = el("div.divider", { text: "interrupted" });
        else if (msg.stopReason === "length") entry.tail = el("div.system.warn", { text: "Output truncated: max tokens reached." });
        if (entry.tail) entry.node.append(entry.tail);
        const u = msg.usage;
        if (u && (u.output || u.input) && msg.stopReason !== "toolUse") {
          // This response's own footer: duration/output/completedAt are stamped
          // on every assistant turn, so live turns and history replay agree.
          const durMs = typeof msg.duration === "number" && msg.duration > 0
            ? msg.duration
            : (entry.startedAt ? Date.now() - entry.startedAt : null);
          const perf = durMs != null
            ? { durMs, tps: durMs >= 1000 && u.output ? u.output / (durMs / 1000) : null, endedAt: msg.completedAt || Date.now() }
            : null;
          // Live turns only (startedAt set); history replay must not clobber a
          // live value. Replay is chronological, so the newest duration wins.
          if (entry.startedAt || perf || !OMP.perf.last) {
            OMP.perf.last = { output: u.output || 0, cacheRead: u.cacheRead || 0, input: u.input || 0,
                              durMs: perf ? perf.durMs : null, tps: perf ? perf.tps : null, endedAt: perf ? perf.endedAt : Date.now() };
          }
          const perfSpan = perf
            ? el("span", { text: `${this.fmtElapsed(perf.durMs)}${perf.tps ? ` · ${perf.tps.toFixed(1)} t/s` : ""}`,
                title: `this response: ${this.fmtElapsed(perf.durMs)}${perf.tps ? ` · ${perf.tps.toFixed(1)} t/s` : ""} · finished ${new Date(perf.endedAt).toTimeString().slice(0, 5)}` })
            : null;
          // Total response time: from the user's prompt to this final answer,
          // including every tool round-trip (the per-call figure below is
          // only the last model call). Computed once per entry so re-renders
          // of the same message keep it.
          if (entry.totalMs == null && this.turnStartedAt != null) {
            const endedAt = msg.completedAt || Date.now();
            if (this.turnStartedAt <= endedAt) entry.totalMs = endedAt - this.turnStartedAt;
            this.turnStartedAt = null;
          }
          const totalSpan = entry.totalMs > 0
            ? el("span.msg-total", { text: `⏱ ${this.fmtElapsed(entry.totalMs)}`, title: `total response time: ${this.fmtElapsed(entry.totalMs)}` })
            : null;
          // Wall-clock time the answer was finished (local, HH:MM). History
          // replay uses the stamped completedAt; a live turn uses now.
 doneAt = new Date(msg.completedAt || (perf && perf.endedAt) || Date.now());
           clockSpan = el("span.msg-clock", {
            text: doneAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false }),
            title: `answered ${doneAt.toLocaleString()}` });
           meta = el("div.meta", null,
            clockSpan,
            el("span", { text: modelName(msg.model), title: msg.model || "" }),
            totalSpan,
            perfSpan,
            el("span", { text: `↑${OMP.fmtTokens((u.input || 0) + (u.cacheRead || 0) + (u.cacheWrite || 0))} ↓${OMP.fmtTokens(u.output)}` }),
            u.cost && u.cost.total ? el("span", { text: OMP.fmtCost(u.cost.total) }) : null);
          if (entry.meta) entry.meta.replaceWith(meta); else entry.node.append(meta);
          entry.meta = meta;
          // The meta footer carries the final timing; drop the live timer badge.
          if (perfSpan && entry.badge) { entry.badge.remove(); entry.badge = null; }
        }
      }
      // Keep the timer badge last (parts are inserted before it as they stream).
      if (entry.badge && entry.badge.isConnected) entry.node.append(entry.badge);
      if (!entry.node.childNodes.length) entry.node.hidden = true; else entry.node.hidden = false;
      if (stick) this.scheduleScroll();
    }

    /** Apply one rpc event frame. Returns true when handled. */
    onEvent(f) {
      const expanded = this.decoder.expand(f);
      if (!expanded) return true;
      f = expanded;
      switch (f.type) {
        case "message_start":
          if (f.message && f.message.role === "assistant") this.updateAssistant(this.assistantEntry(f.messageId), f.message, true);
          else this.renderMessage(f.message, f.messageId);
          return true;
        case "message_update":
          if (f.message && f.message.role === "assistant") {
            // Coalesce deltas: each update carries the full message, so only
            // the newest one per frame needs rendering.
            this.pendingUpdates.set(f.messageId, f.message);
            this.scheduleFlush();
          }
          return true;
        case "message_end":
          this.pendingUpdates.delete(f.messageId);
          if (f.message && f.message.role === "assistant") {
            this.updateAssistant(this.assistantEntry(f.messageId), f.message, false);
            if (f.messageId) this.live.delete(f.messageId);
          } else if (f.message && f.message.role === "toolResult") this.toolResult(f.message);
          return true;
        case "tool_execution_start":
          this.card(f.toolCallId, f.toolName).start(f.args);
          return true;
        case "tool_execution_update":
          this.card(f.toolCallId, f.toolName).update(f.partialResult);
          return true;
        case "tool_execution_end":
          this.card(f.toolCallId, f.toolName).finish(f.result, f.isError);
          return true;
        case "command_output":
          this.addSystem(f.text || "");
          return true;
        case "notice":
          this.addSystem(f.message || f.text || "", f.level === "error" ? "error" : f.level === "warning" ? "warn" : "");
          return true;
        case "auto_compaction_start": this.addDivider("compacting context…"); return true;
        case "auto_compaction_end":
          if (f.errorMessage) this.addSystem("Compaction failed: " + f.errorMessage, "error");
          else this.addDivider(f.aborted ? "compaction aborted" : "context compacted");
          return true;
        case "auto_retry_start":
          this.addSystem(`Retrying (attempt ${f.attempt || "?"}${f.maxAttempts ? "/" + f.maxAttempts : ""})${f.errorMessage ? ": " + f.errorMessage : ""}`, "warn");
          return true;
        case "error":
          this.addError(f.message || f.error || JSON.stringify(f));
          return true;
        default:
          return false;
      }
    }
  }

  OMP.Transcript = Transcript;
})(window.OMP);
