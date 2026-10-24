"use strict";
// Full model browser: every chat model of every provider with credentials,
// grouped by provider, searchable, filterable (reasoning / vision), sortable,
// with recent picks and a details line for the highlighted model.
(function (OMP) {
  const { el } = OMP;
  const RECENT_KEY = "omt-recent-models";
  const RECENT_MAX = 8;
  const RENDER_CAP = 400;

  const key = (m) => `${m.provider}/${m.id}`;
  const hasVision = (m) => Array.isArray(m.input) && m.input.includes("image");
  const price = (m) => (m.cost && Number(m.cost.input)) || 0;
  const money = (v) => (v == null || Number.isNaN(Number(v)) ? "" : Number(v) === 0 ? "free" : "$" + (+Number(v).toFixed(3)));

  function loadRecent() {
    try { return JSON.parse(localStorage.getItem(RECENT_KEY) || "[]"); } catch { return []; }
  }
  function pushRecent(m) {
    const list = [key(m), ...loadRecent().filter((k) => k !== key(m))].slice(0, RECENT_MAX);
    localStorage.setItem(RECENT_KEY, JSON.stringify(list));
  }

  /** Subsequence fuzzy score (higher = better), -1 = no match. */
  function score(hay, q) {
    if (!q) return 0;
    const h = hay.toLowerCase();
    const at = h.indexOf(q);
    if (at >= 0) return 1000 - at - (h.length - q.length) * 0.01;
    let pos = 0;
    let gaps = 0;
    for (const ch of q) {
      const n = h.indexOf(ch, pos);
      if (n < 0) return -1;
      gaps += n - pos;
      pos = n + 1;
    }
    return 500 - gaps;
  }

  /** Open the browser; resolves the picked model object or null. */
  async function browse(app, current) {
    let models;
    try { models = (await app.rpc("get_available_models")).models || []; } catch (e) { OMP.toast(e.message, "error"); return null; }
    if (!models.length) { OMP.toast("No models available — use /login or configure an API key", "warning"); return null; }

    const providers = [...new Set(models.map((m) => m.provider))].sort();
    const counts = Object.fromEntries(providers.map((p) => [p, models.filter((m) => m.provider === p).length]));
    const byKey = new Map(models.map((m) => [key(m), m]));
    const recent = loadRecent().filter((k) => byKey.has(k));
    const curKey = current ? key(current) : "";

    const st = { query: "", group: "all", reasoning: false, vision: false, sort: "default", cursor: 0, rows: [] };

    const search = el("input.mb-search", { type: "text", placeholder: `Search ${models.length} models…  (provider, id, name)`, autocomplete: "off", spellcheck: false });
    const groupList = el("div.mb-groups");
    const list = el("div.mb-list");
    const detail = el("div.mb-detail");
    const count = el("span.mb-count");
    const chip = (label, prop) => {
      const b = el("button.mb-chip", { text: label, onclick: () => { st[prop] = !st[prop]; b.classList.toggle("on", st[prop]); st.cursor = 0; render(); search.focus(); } });
      return b;
    };
    const sortSel = el("select.mb-sort", { onchange: () => { st.sort = sortSel.value; render(); search.focus(); } },
      ...[["default", "Sort: catalog"], ["name", "Sort: name"], ["context", "Sort: context ↓"], ["price", "Sort: price ↑"]]
        .map(([v, t]) => el("option", { value: v, text: t })));

    const groups = () => [["all", "All", models.length], ...(recent.length ? [["recent", "Recent", recent.length]] : []), ...providers.map((p) => [p, p, counts[p]])];

    function renderGroups() {
      groupList.replaceChildren(...groups().map(([id, label, n]) => el("button.mb-group" + (st.group === id ? ".sel" : ""), {
        onclick: () => { st.group = id; st.cursor = 0; render(); search.focus(); },
      }, el("span.ellipsis", { text: label }), el("span.mb-n", { text: String(n) }))));
      const sel = groupList.querySelector(".sel");
      if (sel) sel.scrollIntoView({ block: "nearest" });
    }

    function filtered() {
      const q = st.query.trim().toLowerCase();
      let rows = st.group === "recent" ? recent.map((k) => byKey.get(k))
        : st.group === "all" ? models : models.filter((m) => m.provider === st.group);
      if (st.reasoning) rows = rows.filter((m) => m.reasoning);
      if (st.vision) rows = rows.filter(hasVision);
      if (q) {
        rows = rows.map((m) => [Math.max(score(`${m.provider}/${m.id}`, q), score(m.name || "", q)), m])
          .filter(([sc]) => sc >= 0).sort((a, b) => b[0] - a[0]).map(([, m]) => m);
      } else if (st.sort === "name") {
        rows = [...rows].sort((a, b) => (a.name || a.id).localeCompare(b.name || b.id));
      }
      if (st.sort === "context") rows = [...rows].sort((a, b) => (b.contextWindow || 0) - (a.contextWindow || 0));
      if (st.sort === "price") rows = [...rows].sort((a, b) => price(a) - price(b));
      return rows;
    }

    function renderDetail() {
      const m = st.rows[st.cursor];
      if (!m) { detail.replaceChildren(el("span.faint", { text: "No matching models" })); return; }
      const c = m.cost || {};
      const bits = [
        m.contextWindow ? `${OMP.fmtTokens(m.contextWindow)} context` : "",
        m.maxTokens ? `${OMP.fmtTokens(m.maxTokens)} max out` : "",
        c.input != null ? `in ${money(c.input)} · out ${money(c.output)}${c.cacheRead ? ` · cache ${money(c.cacheRead)}` : ""} /M` : "",
        m.reasoning ? "reasoning" : "",
        hasVision(m) ? "vision" : "",
        m.api || "",
      ].filter(Boolean);
      detail.replaceChildren(el("b", { text: m.name || m.id }), el("code", { text: key(m) }), el("span.muted", { text: bits.join(" · ") }));
    }

    function render() {
      renderGroups();
      st.rows = filtered();
      st.cursor = Math.max(0, Math.min(st.cursor, st.rows.length - 1));
      const shown = st.rows.slice(0, RENDER_CAP);
      const multi = st.group === "all" || st.group === "recent";
      list.replaceChildren(...shown.map((m, k) => el("button.mb-row" + (k === st.cursor ? ".sel" : "") + (key(m) === curKey ? ".current" : ""), {
        onclick: () => { st.cursor = k; choose(); },
        onmousemove: () => { if (st.cursor !== k) { st.cursor = k; paintCursor(); } },
      },
      el("span.mb-main", null,
        el("span.mb-name", { text: m.name || m.id }),
        el("span.mb-id", { text: multi ? key(m) : m.id })),
      el("span.mb-badges", null,
        key(m) === curKey ? el("span.mb-badge.cur", { text: "current" }) : null,
        m.reasoning ? el("span.mb-badge.think", { text: "think", title: "reasoning" }) : null,
        hasVision(m) ? el("span.mb-badge.vision", { text: "vision" }) : null),
      el("span.mb-ctx", { text: m.contextWindow ? OMP.fmtTokens(m.contextWindow) : "–" }),
      el("span.mb-price", { text: m.cost && (Number(m.cost.input) || Number(m.cost.output)) ? `${money(m.cost.input)} / ${money(m.cost.output)}` : "—" }))));
      if (st.rows.length > RENDER_CAP) list.append(el("div.mb-more.faint", { text: `… ${st.rows.length - RENDER_CAP} more — refine the search` }));
      count.textContent = `${st.rows.length} shown`;
      paintCursor();
    }

    function paintCursor() {
      [...list.children].forEach((n, k) => n.classList.toggle("sel", k === st.cursor));
      const sel = list.children[st.cursor];
      if (sel) sel.scrollIntoView({ block: "nearest" });
      renderDetail();
    }

    let m;
    function choose() {
      const row = st.rows[st.cursor];
      if (!row) return;
      pushRecent(row);
      m.done(row);
    }
    function cycleGroup(dir) {
      const ids = groups().map(([id]) => id);
      st.group = ids[(ids.indexOf(st.group) + dir + ids.length) % ids.length];
      st.cursor = 0;
      render();
    }

    search.addEventListener("input", () => { st.query = search.value; st.cursor = 0; render(); });
    const body = el("div.mb", null,
      el("div.mb-top", null, search, chip("reasoning", "reasoning"), chip("vision", "vision"), sortSel),
      el("div.mb-cols", null, groupList, list),
      el("div.mb-foot", null, detail, el("span.mb-hint.faint", null, count, " · ↑↓ move · Tab provider · Enter select · Esc cancel")));
    m = OMP.dialogs.modal({
      title: "Select model",
      body,
      wide: true,
      cls: "model-browser",
      onKey: (e) => {
        const page = 10;
        const move = (d) => { e.preventDefault(); st.cursor = Math.max(0, Math.min(st.rows.length - 1, st.cursor + d)); paintCursor(); };
        if (e.key === "ArrowDown") move(1);
        else if (e.key === "ArrowUp") move(-1);
        else if (e.key === "PageDown") move(page);
        else if (e.key === "PageUp") move(-page);
        else if (e.key === "Tab") { e.preventDefault(); cycleGroup(e.shiftKey ? -1 : 1); }
        else if (e.key === "Enter") { e.preventDefault(); choose(); }
      },
    });
    render();
    const at = st.rows.findIndex((r) => key(r) === curKey);
    if (at >= 0 && at < RENDER_CAP) { st.cursor = at; paintCursor(); }
    setTimeout(() => search.focus(), 0);
    return m.promise;
  }

  OMP.models = { browse };
})(window.OMP);
