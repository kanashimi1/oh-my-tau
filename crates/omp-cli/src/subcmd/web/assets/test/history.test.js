"use strict";
// Unit tests for latest-first session history in the web UI:
// capabilities fallback/normalization, bounded forward paging, latest-window
// rendering, older-page prepending on upward scroll, streaming fallback and
// bounded + invalidating session cache.
// Run: node test/history.test.js

const assert = require("assert");

// ── minimal DOM shim (mirrors test/agents.lifecycle.test.js) ────────────────
function el(spec, props, ...children) {
  const parts = String(spec).split(".");
  const node = {
    tag: parts[0], class: parts.slice(1), props: props || {},
    children: [], _text: "",
    append(...ns) { for (const n of ns.flat(Infinity)) if (n) node.children.push(n); },
    replaceChildren(...ns) { node.children = ns.filter(Boolean); },
    insertBefore(n, ref) {
      const i = ref ? node.children.indexOf(ref) : node.children.length;
      const flat = n.tag === "div" && n.children.length && !n._text && !n.class.length ? n.children : [n];
      node.children.splice(i < 0 ? node.children.length : i, 0, ...flat);
    },
    get firstChild() { return node.children[0] || null; },
    get childElementCount() { return node.children.length; },
    textContent: "",
    classList: { toggle() {}, add() {}, remove() {} },
    isConnected: false,
    hidden: false,
  };
  if (props && props.text !== undefined) node._text = props.text;
  for (const c of children.flat(Infinity)) if (c) node.children.push(c);
  return node;
}
global.window = { addEventListener() {} };
global.document = {
  createElement: () => el("div"),
  createDocumentFragment: () => el("div"),
};
global.CustomEvent = class {};
global.WebSocket = class {};
global.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
global.location = { protocol: "http:", host: "x" };
global.MutationObserver = class { observe() {} };
global.$ = () => el("div");

const OMP = global.OMP = {
  el,
  rafThrottle: (fn) => fn,
  contentText: (c) => (typeof c === "string" ? c : (c || []).map((p) => p.text || "").join("")),
  contentImages: () => [],
};
window.OMP = OMP;
require("../transcript.js");

const user = (i) => ({ role: "user", content: `m${i}` });
const textOf = (root) => {
  const out = [];
  const walk = (n) => { if (n._text) out.push(n._text); for (const c of n.children) walk(c); };
  for (const c of root.children) walk(c);
  return out;
};

// ── capabilities: legacy fallback and normalization ─────────────────────────
(function () {
  const legacy = OMP.normalizeCapabilities(null);
  assert.deepStrictEqual(Object.keys(legacy).sort(),
    ["features", "limits", "maxFrameBytes", "maxReassembledFrameBytes", "protocolVersion", "supportedProtocolVersions"]);
  assert.deepStrictEqual(legacy.supportedProtocolVersions, [1]);
  assert.strictEqual(legacy.limits.getMessages.maxPageSize, 256);
  assert.strictEqual(legacy.limits.getMessages.defaultPageSize, 100);
  assert.strictEqual(legacy.limits.getMessages.maxPageBytes, 4 << 20);
  assert.strictEqual(legacy.limits.getMessages.maxCursorChars, 2048);
  // Same shape as the live payload, no throwing on junk.
  assert.strictEqual(OMP.normalizeCapabilities(undefined).limits.getMessages.maxPageSize, 256);
  assert.strictEqual(OMP.normalizeCapabilities({ limits: {} }).supportedProtocolVersions[0], 1);

  const caps = OMP.normalizeCapabilities({
    protocolVersion: 1,
    supportedProtocolVersions: [1, 2],
    maxFrameBytes: 42,
    maxReassembledFrameBytes: 84,
    limits: { getMessages: { defaultPageSize: 100, maxPageSize: 512, maxPageBytes: 1, maxCursorChars: 2 } },
  });
  assert.deepStrictEqual(caps.supportedProtocolVersions, [1, 2]);
  assert.strictEqual(caps.limits.getMessages.maxPageSize, 512);
  console.log("ok capabilities fallback/normalize");
})();

// ── HistoryBuffer: bounded forward paging, chronological order ──────────────
(async function () {
  const messages = Array.from({ length: 5 }, (_, i) => user(i));
  const requested = [];
  let cursor = null;
  const page = async (params) => {
    requested.push(params.limit);
    const offset = cursor ? 3 : 0; // 3 + 2 pages
    const chunk = messages.slice(offset, offset + (cursor ? 2 : 3));
    const out = { messages: chunk, totalMessages: 5 };
    cursor = offset + chunk.length < 5 ? `c${offset + chunk.length}` : null;
    if (cursor) out.nextCursor = cursor; else cursor = null;
    return out;
  };
  const buffer = new OMP.HistoryBuffer({ limits: { getMessages: { maxPageSize: 3 } } });
  assert.strictEqual(buffer.pageLimit, 3);
  await buffer.loadAll(page);
  assert.deepStrictEqual(buffer.messages().map((m) => m.content), ["m0", "m1", "m2", "m3", "m4"]);
  assert.deepStrictEqual(requested, [3, 3]); // capped by capabilities
  console.log("ok HistoryBuffer paging order/limits");
})();

// ── loadSessionHistory: paged happy path + streaming fallback ───────────────
(async function () {
  const full = Array.from({ length: 7 }, (_, i) => user(i));
  // Quiet session: pages of 2 until exhausted, limit never exceeds caps.
  {
    const limitsSeen = [];
    let pageCalls = 0, fullCalls = 0;
    const rpc = async (type, params) => {
      if (type === "get_messages_page") {
        pageCalls++;
        limitsSeen.push(params.limit);
        const offset = (pageCalls - 1) * 2;
        const out = { messages: full.slice(offset, offset + 2), totalMessages: full.length };
        if (offset + 2 < full.length) out.nextCursor = `c${offset + 2}`;
        return out;
      }
      fullCalls++;
      return { messages: full };
    };
    const r = await OMP.loadSessionHistory(rpc, { limits: { getMessages: { maxPageSize: 2 } } });
    assert.strictEqual(r.paged, true);
    assert.strictEqual(r.streaming, false);
    assert.strictEqual(fullCalls, 0); // no full get_messages on a quiet session
    assert.deepStrictEqual(r.messages.map((m) => m.content), full.map((m) => m.content));
    assert.deepStrictEqual(limitsSeen, [2, 2, 2, 2]);
  }
  // Streaming session: paging refused mid-load -> single get_messages snapshot.
  {
    let pageCalls = 0, fullCalls = 0, stateCalls = 0;
    const rpc = async (type) => {
      if (type === "get_messages_page") { pageCalls++; throw new Error("session busy"); }
      if (type === "get_state") { stateCalls++; return { isStreaming: true }; }
      fullCalls++;
      return { messages: full };
    };
    const r = await OMP.loadSessionHistory(rpc, null);
    assert.strictEqual(r.streaming, true);
    assert.strictEqual(r.paged, false);
    assert.strictEqual(pageCalls, 1);
    assert.strictEqual(stateCalls, 1);
    assert.strictEqual(fullCalls, 1);
    assert.strictEqual(r.messages.length, 7);
  }
  // A real page failure on a quiet session must propagate, not be masked.
  {
    const rpc = async (type) => {
      if (type === "get_messages_page") throw new Error("stale cursor");
      if (type === "get_state") return { isStreaming: false };
      throw new Error("get_messages should not be called");
    };
    await assert.rejects(() => OMP.loadSessionHistory(rpc, null), /stale cursor/);
  }
  console.log("ok loadSessionHistory paging/streaming fallback");
})();

// ── transcript: latest-first render, older-page order, scroll prepending ────
(function () {
  const makeTranscript = (clientHeight = 100, startTop = 0) => {
    const root = el("div");
    let top = startTop;
    const scroller = {
      clientHeight, scrollHeight: 0,
      get scrollTop() { return top; }, set scrollTop(v) { top = v; },
      addEventListener() {},
    };
    const t = new OMP.Transcript(root, scroller);
    // scrollHeight tracks rendered messages so fillViewport/anchor math runs.
    Object.defineProperty(scroller, "scrollHeight", { get: () => root.children.length * 10 });
    return { t, root, scroller };
  };

  // Latest window only, conversation order preserved inside the window.
  {
    const { t, root } = makeTranscript(0);
    const messages = Array.from({ length: 10 }, (_, i) => user(i));
    t.loadLatest(messages, 3);
    assert.deepStrictEqual(textOf(root), ["m7", "m8", "m9"]);
    assert.strictEqual(t.pendingOlder.length, 7);
  }
  // Older pages prepend in chronological order, no duplicates, anchor kept.
  {
    const { t, root, scroller } = makeTranscript(0, 90);
    const messages = Array.from({ length: 10 }, (_, i) => user(i));
    t.loadLatest(messages, 4);
    assert.deepStrictEqual(textOf(root), ["m6", "m7", "m8", "m9"]);
    // Explicit older-page load of 4: the oldest four (m0..m3) prepend.
    t.prependOlder(4);
    assert.deepStrictEqual(textOf(root), ["m0", "m1", "m2", "m3", "m6", "m7", "m8", "m9"]);
    assert.deepStrictEqual(t.pendingOlder.map((m) => m.content), ["m4", "m5"]);
    // Viewport keeps the same content in view: shifted by the prepended height.
    assert.strictEqual(scroller.scrollTop, 80);
    // Scroll-up drains the rest, still ordered, nothing duplicated.
    scroller.scrollTop = 0;
    t.maybeLoadOlder();
    assert.deepStrictEqual(textOf(root), ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"]);
    assert.strictEqual(t.pendingOlder.length, 0);
    const seen = textOf(root);
    assert.strictEqual(new Set(seen).size, seen.length);
  }
  // Short sessions auto-fill the viewport (older pages still reachable).
  {
    const { t, root } = makeTranscript(1000, 0);
    t.loadLatest(Array.from({ length: 6 }, (_, i) => user(i)), 2);
    assert.strictEqual(root.children.length, 6);
    assert.strictEqual(t.pendingOlder.length, 0);
  }
  // loadMessages keeps full-history semantics.
  {
    const { t, root } = makeTranscript(0);
    t.loadMessages([user(0), user(1)]);
    assert.deepStrictEqual(textOf(root), ["m0", "m1"]);
  }
  console.log("ok transcript latest-first + older pages");
})();

// ── session cache: bounded, TTL'd, invalidated by live frames ───────────────
(function () {
  let now = 1000;
  const cache = new OMP.SessionHistoryCache({ maxEntries: 2, maxMessages: 4, ttlMs: 50, now: () => now });
  const k = cache.key("a1", "/s.jsonl");
  assert.strictEqual(cache.get(k), null);
  cache.put(k, [user(0), user(1)]);
  assert.deepStrictEqual(cache.get(k).map((m) => m.content), ["m0", "m1"]);
  // TTL expiry
  now += 51;
  assert.strictEqual(cache.get(k), null);
  // Live frames invalidate
  now = 2000;
  cache.put(k, [user(0)]);
  cache.markAllDirty();
  assert.strictEqual(cache.get(k), null);
  // Oversized snapshots are not cached
  cache.put(k, Array.from({ length: 5 }, (_, i) => user(i)));
  assert.strictEqual(cache.get(k), null);
  // LRU bound: oldest entry evicted
  now = 3000;
  const a = cache.key("a", null), b = cache.key("b", null), c = cache.key("c", null);
  cache.put(a, [user(1)]);
  cache.put(b, [user(2)]);
  cache.put(c, [user(3)]); // evicts a
  assert.strictEqual(cache.get(a), null);
  assert.notStrictEqual(cache.get(b), null);
  assert.notStrictEqual(cache.get(c), null);
  assert.strictEqual(cache.map.size <= 2, true);
  console.log("ok session cache bounds/invalidation");
})();

// ── backward paging: latest page first, older pages lazily ──────────────────
(async function () {
  const full = Array.from({ length: 7 }, (_, i) => user(i));
  const caps = { limits: { getMessages: { maxPageSize: 3 } }, features: { messagesBackwardPaging: true } };
  assert.strictEqual(OMP.normalizeCapabilities(caps).features.messagesBackwardPaging, true);
  assert.strictEqual(OMP.normalizeCapabilities(null).features.messagesBackwardPaging, false);

  const requests = [];
  let offset = 7; // exclusive end of the not-yet-fetched suffix
  let fullCalls = 0;
  const rpc = async (type, params = {}) => {
    if (type === "get_messages_page") {
      requests.push({ ...params });
      const start = Math.max(0, offset - params.limit);
      const out = { messages: full.slice(start, offset), totalMessages: full.length };
      offset = start;
      if (start > 0) out.nextCursor = `older-${start}`;
      return out;
    }
    fullCalls++;
    return { messages: full };
  };
  const r = await OMP.loadSessionHistory(rpc, caps);
  // First render payload is only the newest bounded page; nothing else fetched.
  assert.deepStrictEqual(r.messages.map((m) => m.content), ["m4", "m5", "m6"]);
  assert.strictEqual(r.backward, true);
  assert.strictEqual(r.hasOlder, true);
  assert.strictEqual(fullCalls, 0);
  assert.deepStrictEqual(requests, [{ direction: "backward", limit: 3 }]);
  // Older pages arrive chronological and oldest-nextCursor until exhausted.
  const p1 = await r.fetchOlder();
  assert.deepStrictEqual(p1.map((m) => m.content), ["m1", "m2", "m3"]);
  const p2 = await r.fetchOlder();
  assert.deepStrictEqual(p2.map((m) => m.content), ["m0"]);
  assert.strictEqual(r.hasOlder, false);
  assert.strictEqual(await r.fetchOlder(), null);
  assert.deepStrictEqual(requests.map((q) => !!q.cursor), [false, true, true]);
  console.log("ok loadSessionHistory backward latest-first");
})();

// ── transcript: lazy older-source (backward paging) keeps order ─────────────
(async function () {
  const makeTranscript = (clientHeight = 0) => {
    const root = el("div");
    let top = 0;
    const scroller = { clientHeight, scrollHeight: 0, get scrollTop() { return top; }, set scrollTop(v) { top = v; }, addEventListener() {} };
    const t = new OMP.Transcript(root, scroller);
    Object.defineProperty(scroller, "scrollHeight", { get: () => root.children.length * 10 });
    return { t, root, scroller };
  };
  const { t, root, scroller } = makeTranscript();
  // Agent with backward paging: latest page rendered via loadMessages…
  t.loadMessages([user(4), user(5), user(6)]);
  assert.deepStrictEqual(textOf(root), ["m4", "m5", "m6"]);
  // …older pages fetched lazily per scroll-up, in agent order (newest first).
  const pages = [[user(2), user(3)], [user(0), user(1)], null];
  let fetches = 0;
  t.setOlderSource(async () => pages[fetches++]);
  scroller.scrollTop = 0;
  await t.maybeLoadOlder();
  assert.deepStrictEqual(textOf(root), ["m2", "m3", "m4", "m5", "m6"]);
  scroller.scrollTop = 0;
  await t.maybeLoadOlder();
  assert.deepStrictEqual(textOf(root), ["m0", "m1", "m2", "m3", "m4", "m5", "m6"]);
  // History start: source returned null -> source dropped, no further fetches.
  scroller.scrollTop = 0;
  await t.maybeLoadOlder();
  scroller.scrollTop = 0;
  await t.maybeLoadOlder();
  assert.strictEqual(fetches, 3);
  // Failing source detaches instead of throwing from the scroll handler.
  t.setOlderSource(async () => { throw new Error("stale cursor"); });
  scroller.scrollTop = 0;
  await t.maybeLoadOlder();
  assert.deepStrictEqual(textOf(root), ["m0", "m1", "m2", "m3", "m4", "m5", "m6"]);
  console.log("ok transcript lazy older-source ordering");
})();

// Backward paging is allowed during streaming: capable agents never take a
// full get_messages snapshot, even mid-stream.
(async function () {
  const full = Array.from({ length: 4 }, (_, i) => user(i));
  let offset = 4, fullCalls = 0;
  const rpc = async (type, params = {}) => {
    if (type === "get_messages_page") {
      const start = Math.max(0, offset - params.limit);
      const out = { messages: full.slice(start, offset) };
      offset = start;
      if (start > 0) out.nextCursor = `older-${start}`;
      return out;
    }
    fullCalls++; // get_state / get_messages must not be reached
    return type === "get_state" ? { isStreaming: true } : { messages: full };
  };
  const r = await OMP.loadSessionHistory(rpc, { limits: { getMessages: { maxPageSize: 4 } }, features: { messagesBackwardPaging: true } });
  assert.strictEqual(r.backward, true);
  assert.strictEqual(fullCalls, 0);
  assert.strictEqual(r.hasOlder, false); // single page already covers history
  assert.strictEqual(await r.fetchOlder(), null);
  console.log("ok backward paging works while streaming");
})();

console.log("history.test.js: all checks passed");
