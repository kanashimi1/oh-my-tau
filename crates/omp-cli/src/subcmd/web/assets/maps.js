"use strict";
// Вкладка «Карты»: граф кода (code), граф знаний (kg), диаграмма (diagram).
// data-terminal поверх переменных app.css; vanilla JS, SVG руками, детерминизм
// через mulberry32(0x2F6E2B1). Стили — assets/maps.css (W4), каркас DOM — index.html (W2).
window.OMP = window.OMP || {};
(function (OMP) {
  const { el, $ } = OMP;
  const SEED = 0x2F6E2B1;
  const MODES = ["code", "kg", "diagram"];
  const KG_TYPES = ["project", "asset", "structure", "tool", "doc", "mod_src"];

  // ── PRNG ─────────────────────────────────────────────────────────────────
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const reducedMotion = () =>
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function svgEl(tag, attrs) {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  }

  const hueClass = (i) => "maps-hue-" + (((i % 5) + 5) % 5);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  // ── Класс вида ───────────────────────────────────────────────────────────
  class Maps {
    constructor(opts) {
      this.opt = opts || {};
      this.data = null;            // {code, kg}
      this.loading = false;
      this.open = false;           // вид активен
      this.mode = "code";          // code | kg | diagram
      this.layout = "force";       // force | clusters
      this.search = "";
      this.typeFilter = "";
      this.off = new Set();        // выключенные кластеры/типы
      this.sel = null;             // id выбранного узла
      this.pin = null;             // id закреплённого (shift+click) узла
      this.nodes = [];             // текущий набор узлов (копии с x/y)
      this.nodeById = new Map();
      this.edges = [];
      this.adj = new Map();        // id -> [{edge, other}]
      this.nodeEls = new Map();    // id -> SVGElement
      this.edgeEls = [];
      this.sim = null;             // {alpha, running, ...}
      this.tf = { k: 1, x: 0, y: 0 };
      this.dragNode = null;
      this._raf = 0;
      this._bindDom();
    }

    // ── DOM ────────────────────────────────────────────────────────────────
    _bindDom() {
      const view = $("#maps-view");
      if (!view) return; // каркас ещё не смонтирован (W2)
      this.view = view;
      this.canvas = $(".maps-canvas", view);
      this.svg = $("#maps-svg", view);
      this.g = $("#maps-g", view);
      this.detail = $("#maps-detail", view);
      this.searchInput = $("#maps-search", view);
      this.chips = $("#maps-cluster-chips", view);
      this.typeSel = $("#maps-type-filter", view);
      this.countsEl = $("#maps-counts", view);
      this.stLoading = $("#maps-loading", view);
      this.stEmpty = $("#maps-empty", view);
      this.stError = $("#maps-error", view);
      this.diagramBox = null; // HTML-слой диаграммы, создаётся лениво

      view.addEventListener("click", (e) => {
        const tab = e.target.closest("[data-mode]");
        if (tab) { this.setMode(tab.dataset.mode); return; }
        const seg = e.target.closest("[data-layout]");
        if (seg) { this.setLayout(seg.dataset.layout); return; }
        const chip = e.target.closest(".chip");
        if (chip && chip.dataset.key) {
          if (this.off.has(chip.dataset.key)) this.off.delete(chip.dataset.key);
          else this.off.add(chip.dataset.key);
          chip.classList.toggle("on", !this.off.has(chip.dataset.key));
          this.applyFilters();
        }
        const act = e.target.closest("[data-action]");
        if (act) {
          if (act.dataset.action === "reload") this.load(true);
          if (act.dataset.action === "close-detail") this.closeDetail();
          if (act.dataset.action === "unpin") { this.pin = null; this.clearPath(); }
        }
      });

      if (this.searchInput)
        this.searchInput.addEventListener("input", () => {
          this.search = this.searchInput.value.trim().toLowerCase();
          this.applyFilters();
        });
      if (this.typeSel)
        this.typeSel.addEventListener("change", () => {
          this.typeFilter = this.typeSel.value;
          this.applyFilters();
        });

      if (this.svg) {
        this.svg.addEventListener("wheel", (e) => this._onWheel(e), { passive: false });
        this.svg.addEventListener("pointerdown", (e) => this._onPointerDown(e));
        this.svg.addEventListener("pointermove", (e) => this._onPointerMove(e));
        this.svg.addEventListener("pointerup", (e) => this._onPointerUp(e));
        this.svg.addEventListener("pointerleave", (e) => this._onPointerUp(e));
      }
      window.addEventListener("resize", () => {
        if (this.open && this.mode === "diagram") this._layoutDiagramEdges();
      });
    }

    _setState(which) {
      for (const [key, node] of [["loading", this.stLoading], ["empty", this.stEmpty], ["error", this.stError]]) {
        if (!node) continue;
        node.hidden = key !== which;
      }
    }

    // ── Публичный контракт ─────────────────────────────────────────────────
    show(S) {
      this.open = true;
      document.body.classList.add("maps-active");
      const ids = ["#transcript-wrap", "#composer-wrap", "#widgets-above", "#widgets-below", "#queue-band"];
      for (const id of ids) { const n = $(id); if (n) n.hidden = true; }
      if (this.view) this.view.hidden = false;
      const btn = $("#maps-btn"); if (btn) btn.classList.add("active");
      // deep-link: #maps[/code|/kg|/diagram]
      const m = /^#maps\/(code|kg|diagram)/.exec(location.hash);
      if (m && m[1] !== this.mode) this.setMode(m[1]);
      if (!this.data && !this.loading) this.load();
      else if (this.data) this.render();
    }

    hide() { this.close(); }

    close() {
      this.open = false;
      document.body.classList.remove("maps-active");
      this._stopSim();
      if (this.view) this.view.hidden = true;
      for (const id of ["#transcript-wrap", "#composer-wrap", "#widgets-above", "#widgets-below", "#queue-band"]) {
        const n = $(id); if (n) n.hidden = false;
      }
      const btn = $("#maps-btn"); if (btn) btn.classList.remove("active");
      this.closeDetail();
    }

    toggle() { if (this.open) this.close(); else this.show(); }
    isOpen() { return !!this.open; }

    closeDetail() {
      if (this.detail && !this.detail.hidden) {
        this.detail.hidden = true;
        this.sel = null;
        this._applyHighlight();
        return true;
      }
      return false;
    }

    setMode(mode) {
      if (!MODES.includes(mode)) return;
      this.mode = mode;
      if (!this.view) return;
      for (const t of this.view.querySelectorAll("[data-mode]"))
        t.setAttribute("aria-selected", String(t.dataset.mode === mode));
      this.typeSel.hidden = mode !== "kg";
      this.closeDetail();
      this.pin = null;
      if (this.data) this.render();
      else if (!this.loading) this.load();
    }

    setLayout(layout) {
      if (layout !== "force" && layout !== "clusters") return;
      this.layout = layout;
      if (!this.view) return;
      for (const s of this.view.querySelectorAll("[data-layout]"))
        s.setAttribute("aria-pressed", String(s.dataset.layout === layout));
      if (this.data) this.render();
    }

    applyFilters() { if (this.data) { this._computeVisibility(); this.render(); } }

    selectNode(id) {
      const n = this.nodeById.get(id);
      if (!n) return;
      this.sel = id;
      this.renderDetail(n);
      this._applyHighlight();
    }

    pinNode(id) { this.pin = id; }

    runPath(a, b) {
      if (!a || !b || a === b) return;
      const prev = new Map([[a, null]]);
      const q = [a];
      while (q.length) {
        const cur = q.shift();
        if (cur === b) break;
        for (const { other } of this.adj.get(cur) || [])
          if (!prev.has(other)) { prev.set(other, cur); q.push(other); }
      }
      if (!prev.has(b)) { OMP.toast && OMP.toast("Путь не найден", "warning"); return; }
      const path = [];
      for (let cur = b; cur != null; cur = prev.get(cur)) path.unshift(cur);
      this.clearPath();
      const pts = path.map((id) => this.nodeById.get(id)).filter(Boolean);
      if (pts.length > 1 && this.g) {
        const pl = svgEl("polyline", {
          class: "maps-path",
          points: pts.map((n) => `${n.x},${n.y}`).join(" "),
        });
        this.g.append(pl);
        this.pathEl = pl;
      }
      this.pathNodes = new Set(path);
      this._applyHighlight();
    }

    clearPath() {
      if (this.pathEl) { this.pathEl.remove(); this.pathEl = null; }
      this.pathNodes = null;
      this._applyHighlight();
    }

    // ── Данные ─────────────────────────────────────────────────────────────
    async load(force) {
      if (this.loading) return;
      this.loading = true;
      this._setState("loading");
      this._renderLoadingSkeleton();
      try {
        const data = await OMP.api("/api/maps");
        this.data = data && (data.code || data.kg) ? data : null;
        this.loading = false;
        if (!this.data ||
            ((!this.data.code || !this.data.code.nodes.length) &&
             (!this.data.kg || !this.data.kg.nodes.length))) {
          this.data = this.data || {};
          this._setState("empty");
          this._renderEmpty();
          return;
        }
        this._setState(null);
        this.render();
      } catch (err) {
        this.loading = false;
        this._setState("error");
        this._renderError(String(err && err.message || err));
      }
    }

    _renderLoadingSkeleton() {
      if (!this.stLoading) return;
      this.stLoading.replaceChildren(
        el("div.maps-skel.w60"), el("div.maps-skel.w40"), el("div.maps-skel.w75"),
        el("div.maps-state-text", { text: "Загрузка карт…" }),
      );
    }

    _renderEmpty() {
      if (!this.stEmpty) return;
      this.stEmpty.replaceChildren(
        el("div.maps-state-text", {
          text: "Данные карт не найдены. Постройте maps.json (постобработка индекса): запусти hypertrail/compose_maps.py",
        }),
        el("button.maps-tool", { text: "Обновить", dataset: { action: "reload" } }),
      );
    }

    _renderError(message) {
      if (!this.stError) return;
      this.stError.replaceChildren(
        el("div.maps-state-text", { text: "Ошибка загрузки: " + message }),
        el("button.maps-tool", { text: "Повторить", dataset: { action: "reload" } }),
      );
    }

    // ── Подготовка набора данных ───────────────────────────────────────────
    _dataset() {
      if (this.mode === "kg") {
        const d = (this.data && this.data.kg) || { nodes: [], edges: [] };
        return { nodes: d.nodes, edges: d.edges, clusters: null, kind: "kg" };
      }
      const d = (this.data && this.data.code) || { nodes: [], edges: [], clusters: [] };
      return { nodes: d.nodes, edges: d.edges, clusters: d.clusters || [], kind: "code" };
    }

    _render() {
      if (!this.view || !this.data) return;
      const ds = this._dataset();
      this._buildGraph(ds);
      this._renderChips(ds);
      this._renderTypeFilter(ds);
      this._computeVisibility();
      if (this.mode === "diagram") this._renderDiagram(ds);
      else this._renderGraph(ds);
      this._updateCounts();
    }

    render() { this._render(); }

    _buildGraph(ds) {
      // Копии узлов, чтобы симуляция не мутировала кэш данных.
      this.nodes = ds.nodes.map((n) => ({
        ...n, x: 0, y: 0, vx: 0, vy: 0, fx: null, fy: null,
        deg: 0, degIn: 0, degOut: 0,
      }));
      this.nodeById = new Map(this.nodes.map((n) => [n.id, n]));
      // В данных встречаются рёбра с отсутствующими концами (kg): синтезируем
      // узлы-заглушки, чтобы граф не терял все рёбра.
      let nph = 0;
      for (const e of ds.edges) {
        for (const end of [e.s, e.t]) {
          if (this.nodeById.has(end)) continue;
          const ph = { id: end, label: end, cluster: "—", type: "—", domain: "прочее", lang: "", out: 0, in: 0, x: 0, y: 0, vx: 0, vy: 0, fx: null, fy: null, deg: 0, degIn: 0, degOut: 0 };
          this.nodes.push(ph);
          this.nodeById.set(end, ph);
          nph++;
        }
      }
      const byId = this.nodeById;
      this.edges = ds.edges
        .filter((e) => byId.has(e.s) && byId.has(e.t))
        .map((e) => ({ ...e, source: byId.get(e.s), target: byId.get(e.t) }));
      this.adj = new Map(this.nodes.map((n) => [n.id, []]));
      let wmax = 1;
      for (const e of this.edges) {
        wmax = Math.max(wmax, e.w || 1);
        e.source.deg++; e.source.degOut++;
        e.target.deg++; e.target.degIn++;
        this.adj.get(e.s).push({ edge: e, other: e.t });
        this.adj.get(e.t).push({ edge: e, other: e.s });
      }
      this.wmax = wmax;
      // Группа (кластер/тип/домен) для раскраски.
      this.groups = ds.kind === "code"
        ? [...new Set(this.nodes.map((n) => n.cluster || "—"))]
        : [...new Set(this.nodes.map((n) => n.type || "—"))];
      for (const n of this.nodes)
        n.hue = hueClass(this.groups.indexOf(this._groupKey(n)));
    }

    _groupKey(n) {
      if (this.mode === "kg" || (this.mode === "diagram" && this._dsKind() === "kg"))
        return n.type || "—";
      return n.cluster || "—";
    }

    _dsKind() { return this.mode === "kg" ? "kg" : this.mode === "diagram" ? this._diagramKind() : "code"; }
    _diagramKind() { return this._diagramMode === "kg" ? "kg" : "code"; }

    _renderChips(ds) {
      if (!this.chips) return;
      const kind = this.mode === "kg" ? "kg" : "code";
      let keys;
      if (kind === "code") keys = (ds.clusters.length ? ds.clusters.map((c) => c.id) : [...new Set(ds.nodes.map((n) => n.cluster || "—"))]);
      else keys = [...new Set(ds.nodes.map((n) => n.type || "—"))];
      this.chips.replaceChildren(...keys.map((k, i) => {
        const off = this.off.has(k);
        const gi = this.groups.indexOf(k);
        const size = kind === "code"
          ? (ds.clusters.find((c) => c.id === k) || { size: ds.nodes.filter((n) => (n.cluster || "—") === k).length }).size
          : ds.nodes.filter((n) => (n.type || "—") === k).length;
        return el("button.chip" + (off ? "" : ".on"), {
          dataset: { key: k }, title: `${k} · ${size}`,
        }, el("span.chip-dot." + hueClass(gi >= 0 ? gi : i)), k);
      }));
    }

    _renderTypeFilter(ds) {
      if (!this.typeSel) return;
      if (this.mode !== "kg") { this.typeSel.hidden = true; return; }
      this.typeSel.hidden = false;
      const cur = this.typeFilter;
      this.typeSel.replaceChildren(
        el("option", { value: "" }, "все типы"),
        ...KG_TYPES.filter((t) => ds.nodes.some((n) => n.type === t))
          .map((t) => el("option", { value: t }, t)),
      );
      this.typeSel.value = KG_TYPES.includes(cur) ? cur : "";
      this.typeFilter = this.typeSel.value;
    }

    _computeVisibility() {
      const q = this.search;
      const tf = this.typeFilter;
      this.visible = new Set();
      for (const n of this.nodes) {
        if (this.off.has(this._groupKey(n))) continue;
        if (this.mode === "kg" && tf && n.type !== tf) continue;
        if (q && !(`${n.label}`.toLowerCase().includes(q) || `${n.id}`.toLowerCase().includes(q))) continue;
        this.visible.add(n.id);
      }
    }

    _updateCounts() {
      if (!this.countsEl) return;
      const n = this.visible ? this.visible.size : this.nodes.length;
      const m = this.edges.filter((e) => this._edgeVisible(e)).length;
      this.countsEl.textContent = `${n} узлов · ${m} рёбер`;
    }

    _edgeVisible(e) {
      return this.visible.has(e.s) && this.visible.has(e.t);
    }

    // ── Лейауты ────────────────────────────────────────────────────────────
    _canvasSize() {
      const r = this.canvas ? this.canvas.getBoundingClientRect() : { width: 1200, height: 800 };
      return { W: Math.max(320, r.width), H: Math.max(240, r.height) };
    }

    _layoutClusters(ds, visibleNodes, W, H) {
      // Колонки по кластерам (code) / доменам (kg); внутри по in+out убыв., затем label.
      const isCode = ds.kind === "code";
      let cols;
      if (isCode) {
        const order = ds.clusters.map((c) => c.id);
        cols = new Map(order.map((id) => [id, []]));
        cols.set("—", cols.get("—") || []);
      } else {
        const domains = [...new Set(visibleNodes.map((n) => n.domain || "прочее"))].sort();
        cols = new Map(domains.map((d) => [d, []]));
      }
      for (const n of visibleNodes) {
        const key = isCode ? (cols.has(n.cluster) ? n.cluster : "—") : (n.domain || "прочее");
        cols.get(key).push(n);
      }
      const colWidth = 190;
      const total = Math.max(1, [...cols.values()].filter((a) => a.length).length);
      const spanX = Math.max(total * colWidth, W * 0.6);
      const x0 = W / 2 - spanX / 2 + colWidth / 2;
      let ci = 0;
      for (const [key, arr] of cols) {
        if (!arr.length) continue;
        arr.sort((a, b) => (b.deg - a.deg) || String(a.label).localeCompare(String(b.label)));
        const rowH = Math.min(34, Math.max(18, (H * 0.85) / Math.max(1, arr.length)));
        const spanY = rowH * arr.length;
        const y0 = H / 2 - spanY / 2;
        arr.forEach((n, ri) => {
          n.x = x0 + ci * colWidth;
          n.y = y0 + ri * rowH + rowH / 2;
          n.fx = n.x; n.fy = n.y;
        });
        ci++;
      }
    }

    // ── Силовая симуляция (детерминированная) ──────────────────────────────
    _startSim(visibleNodes, visibleEdges) {
      this._stopSim();
      const { W, H } = this._canvasSize();
      const rnd = mulberry32(SEED);
      const area = Math.sqrt(W * H);
      // 1) seed-инициализация позиций
      for (const n of visibleNodes) {
        const a = rnd() * Math.PI * 2;
        const r = (0.2 + 0.6 * rnd()) * area / 2;
        n.x = W / 2 + Math.cos(a) * r;
        n.y = H / 2 + Math.sin(a) * r;
        n.vx = 0; n.vy = 0; n.fx = null; n.fy = null;
      }
      // 2) clusters-пре-пасс
      if (this.layout === "clusters") this._layoutClusters(this._dataset(), visibleNodes, W, H);
      const alphaMin = 0.001;
      const sim = {
        alpha: 1,
        alphaMin,
        alphaDecay: 1 - Math.pow(alphaMin, 1 / 320),
        velocityDecay: 0.35,
        charge: -1400 * (area * area) / (1200 * 800),
        nodes: visibleNodes,
        edges: visibleEdges,
        W, H, ticks: 0,
        last: 0,
      };
      this.sim = sim;
      if (reducedMotion()) {
        while (sim.alpha > sim.alphaMin && sim.ticks < 320) this._simTick(sim);
        this._renderPositions();
        return;
      }
      const step = (ts) => {
        if (this.sim !== sim) return; // заменено/остановлено
        if (ts - sim.last >= 33) { // ~30 fps тиков
          sim.last = ts;
          this._simTick(sim);
          this._renderPositions();
        }
        if (sim.alpha > sim.alphaMin && sim.ticks < 320) this._raf = requestAnimationFrame(step);
        else this.sim = null;
      };
      this._raf = requestAnimationFrame(step);
    }

    _stopSim() {
      this.sim = null;
      if (this._raf) { cancelAnimationFrame(this._raf); this._raf = 0; }
    }

    _simTick(sim) {
      const { nodes, edges, W, H } = sim;
      const alpha = sim.alpha;
      // отталкивание (заряд) — попарно, O(n²), n ≤ 150
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          let dx = b.x - a.x, dy = b.y - a.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 1) { dx = (j % 2 ? 1 : -1) * 0.1; dy = (i % 2 ? 1 : -1) * 0.1; d2 = 0.02; }
          const d = Math.sqrt(d2);
          // кап импульса: без него близкие пары дают 1/d²-скачок и симуляция расходится
          const rep = Math.min(48, (sim.charge * alpha) / d2);
          const fx = (dx / d) * rep, fy = (dy / d) * rep;
          a.vx -= fx; a.vy -= fy;
          b.vx += fx; b.vy += fy;
        }
      }
      // расталкивание кластеров: -600 × размер соседнего кластера
      const sizes = new Map();
      for (const n of nodes) {
        const k = this._groupKey(n);
        sizes.set(k, (sizes.get(k) || 0) + 1);
      }
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i], ka = this._groupKey(a);
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const kb = this._groupKey(b);
          if (ka === kb) continue;
          let dx = b.x - a.x, dy = b.y - a.y;
          const d2 = Math.max(1, dx * dx + dy * dy);
          const d = Math.sqrt(d2);
          const rep = Math.min(36, (600 * alpha * ((sizes.get(kb) || 1) + (sizes.get(ka) || 1)) / 2) / d2);
          const fx = (dx / d) * rep, fy = (dy / d) * rep;
          a.vx -= fx; a.vy -= fy;
          b.vx += fx; b.vy += fy;
        }
      }
      // рёбра: пружины
      for (const e of edges) {
        const { source: a, target: b } = e;
        const w = e.w || 1;
        const dist = clamp(26, 78 - 10 * Math.log2(w + 1), 90);
        const strength = 0.5 + 0.5 * (w / this.wmax);
        let dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.max(0.5, Math.hypot(dx, dy));
        const k = ((d - dist) / d) * strength * alpha;
        const fx = dx * k, fy = dy * k;
        a.vx += fx; a.vy += fy;
        b.vx -= fx; b.vy -= fy;
      }
      // центр (слабый) + интегрирование
      const cx = W / 2, cy = H / 2;
      for (const n of nodes) {
        n.vx += (cx - n.x) * 0.05 * alpha;
        n.vy += (cy - n.y) * 0.05 * alpha;
        if (n.fx != null) { n.x = n.fx; n.y = n.fy; n.vx = 0; n.vy = 0; }
        else {
          n.vx *= 1 - sim.velocityDecay;
          n.vy *= 1 - sim.velocityDecay;
          // кламп скорости: ≤ 24 px/тик — гарантия сходимости при 320 тиках
          if (n.vx > 24) n.vx = 24; else if (n.vx < -24) n.vx = -24;
          if (n.vy > 24) n.vy = 24; else if (n.vy < -24) n.vy = -24;
          n.x += n.vx; n.y += n.vy;
        }
      }
      sim.alpha -= sim.alpha * sim.alphaDecay;
      sim.ticks++;
    }

    _renderPositions() {
      const sim = this.sim;
      const nodes = sim ? sim.nodes : this.nodes;
      for (const n of nodes) {
        const eln = this.nodeEls.get(n.id);
        if (eln) {
          if (Number.isFinite(n.x) && Number.isFinite(n.y)) eln.setAttribute("transform", `translate(${n.x},${n.y})`);
          const lbl = eln._lbl;
          if (lbl) {
            lbl.setAttribute("x", n.x); lbl.setAttribute("y", n.y);
          }
        }
      }
      for (const { eln, e } of this.edgeEls) {
        if (!sim || (this.visible.has(e.s) && this.visible.has(e.t))) {
          eln.setAttribute("x1", e.source.x); eln.setAttribute("y1", e.source.y);
          eln.setAttribute("x2", e.target.x); eln.setAttribute("y2", e.target.y);
        }
      }
      if (this.pathEl && this.pathNodes) {
        const pts = [...this.pathNodes].map((id) => this.nodeById.get(id)).filter(Boolean);
        this.pathEl.setAttribute("points", pts.map((n) => `${n.x},${n.y}`).join(" "));
      }
    }

    // ── Рендер графа ───────────────────────────────────────────────────────
    _renderGraph(ds) {
      if (!this.g || !this.canvas) return;
      this._hideDiagram();
      this.g.replaceChildren();
      this.nodeEls = new Map();
      this.edgeEls = [];
      const edgeG = svgEl("g", { class: "maps-edges" });
      const nodeG = svgEl("g", { class: "maps-nodes" });
      this.g.append(edgeG, nodeG);
      const visEdges = this.edges.filter((e) => this._edgeVisible(e));
      // рёбра
      for (const e of visEdges) {
        const ln = svgEl("line", {
          class: "maps-edge" + (e.kind === "import" ? " maps-edge-import" : ""),
          "stroke-width": e.kind === "import" ? 1 : clamp(Math.round(Math.log2((e.w || 1) + 1)), 1, 3),
        });
        ln._edge = e;
        edgeG.append(ln);
        this.edgeEls.push({ eln: ln, e });
      }
      // узлы + подписи (только крупные/видимые подписи: label при k≥0.7 или deg>4)
      const visNodes = this.nodes.filter((n) => this.visible.has(n.id));
      for (const n of visNodes) {
        const grp = svgEl("g", {
          class: "maps-node " + n.hue,
          "data-id": n.id,
          tabindex: "0",
          role: "button",
          "aria-label": n.label,
        });
        const r = clamp(3 + Math.log2(1 + n.deg), 3, 9);
        grp.append(svgEl("circle", { r: r.toFixed(1) }));
        const lbl = svgEl("text", {
          class: "maps-label",
          x: n.x, y: n.y, dx: (r + 4).toFixed(1), dy: "0.32em",
        });
        lbl.textContent = n.label || n.id;
        grp.append(lbl);
        grp._lbl = lbl;
        grp.addEventListener("pointerenter", () => this._hover(n.id, true));
        grp.addEventListener("pointerleave", () => this._hover(n.id, false));
        nodeG.append(grp);
        this.nodeEls.set(n.id, grp);
      }
      // размер канвы и старт симуляции
      this._resetTransform();
      this._startSim(visNodes, visEdges);
      this._updateCounts();
    }

    // ── Взаимодействие: pan / zoom / drag / hover / click ──────────────────
    _applyTransform() {
      if (!this.g) return;
      const { k, x, y } = this.tf;
      this.g.setAttribute("transform", `translate(${x},${y}) scale(${k})`);
      // подписи узлов прячем при сильном отдалении
      this.g.classList.toggle("labels-off", k < 0.7);
    }

    _resetTransform() { this.tf = { k: 1, x: 0, y: 0 }; this._applyTransform(); }

    _onWheel(e) {
      if (this.mode === "diagram") return; // диаграмма скроллится, не зумится
      e.preventDefault();
      const rect = this.svg.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      const k = clamp(this.tf.k * factor, 0.15, 4);
      const real = k / this.tf.k;
      this.tf.x = mx - (mx - this.tf.x) * real;
      this.tf.y = my - (my - this.tf.y) * real;
      this.tf.k = k;
      this._applyTransform();
    }

    _svgPoint(e) {
      const rect = this.svg.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    _onPointerDown(e) {
      const gnode = e.target.closest && e.target.closest(".maps-node");
      this._panning = null;
      if (gnode && this.mode !== "diagram") {
        const id = gnode.dataset.id;
        const n = this.nodeById.get(id);
        this.dragNode = n;
        this._dragMoved = false;
        if (this.svg.setPointerCapture) try { this.svg.setPointerCapture(e.pointerId); } catch {}
        return;
      }
      if (this.mode !== "diagram") {
        this._panning = { ...this._svgPoint(e), tx: this.tf.x, ty: this.tf.y };
        if (this.svg.setPointerCapture) try { this.svg.setPointerCapture(e.pointerId); } catch {}
      }
    }

    _onPointerMove(e) {
      if (this.dragNode) {
        const p = this._svgPoint(e);
        const x = (p.x - this.tf.x) / this.tf.k;
        const y = (p.y - this.tf.y) / this.tf.k;
        if (!this._dragMoved && Math.hypot(x - this.dragNode.x, y - this.dragNode.y) > 2)
          this._dragMoved = true;
        this.dragNode.fx = x; this.dragNode.fy = y;
        this.dragNode.x = x; this.dragNode.y = y;
        this._renderPositions();
        return;
      }
      if (this._panning) {
        const p = this._svgPoint(e);
        this.tf.x = this._panning.tx + (p.x - this._panning.x);
        this.tf.y = this._panning.ty + (p.y - this._panning.y);
        this._applyTransform();
      }
    }

    _onPointerUp(e) {
      if (this.dragNode) {
        const n = this.dragNode;
        if (!this._dragMoved) this._clickNode(n, e);
        n.fx = null; n.fy = null;
        this.dragNode = null;
        return;
      }
      this._panning = null;
    }

    _hover(id, on) {
      if (!on) { this._applyHighlight(); return; }
      const neigh = new Set([id]);
      for (const { other, edge } of this.adj.get(id) || [])
        if (this._edgeVisible(edge)) neigh.add(other);
      this._applyHighlight(neigh);
    }

    _applyHighlight(keep) {
      const focusSel = this.sel || null;
      for (const [id, eln] of this.nodeEls) {
        let vis = true;
        if (keep) vis = keep.has(id);
        else if (this.search) vis = this.visible.has(id);
        else if (focusSel) {
          vis = id === focusSel ||
            (this.adj.get(focusSel) || []).some(({ other, edge }) => other === id && this._edgeVisible(edge));
        }
        else if (this.pathNodes) vis = this.pathNodes.has(id);
        eln.classList.toggle("dim", !vis);
      }
      for (const { eln, e } of this.edgeEls) {
        let vis = true;
        if (keep) vis = keep.has(e.s) && keep.has(e.t);
        else if (this.search) vis = this.visible.has(e.s) && this.visible.has(e.t);
        else if (focusSel) vis = e.s === focusSel || e.t === focusSel;
        else if (this.pathNodes) {
          const p = [...this.pathNodes];
          vis = false;
          for (let i = 0; i < p.length - 1; i++)
            if ((e.s === p[i] && e.t === p[i + 1]) || (e.t === p[i] && e.s === p[i + 1])) vis = true;
        }
        eln.classList.toggle("dim", !vis);
      }
    }

    _clickNode(n, e) {
      if (e && e.shiftKey) {
        if (!this.pin) { this.pin = n.id; this.clearPath(); return; }
        if (this.pin !== n.id) { this.runPath(this.pin, n.id); return; }
        this.pin = null; this.clearPath(); return;
      }
      this.selectNode(n.id);
    }

    // ── Панель деталей ─────────────────────────────────────────────────────
    renderDetail(n) {
      if (!this.detail) return;
      const ds = this._dataset();
      const isCode = ds.kind === "code";
      const inc = this.edges.filter((e) => e.t === n.id).sort((a, b) => (b.w || 1) - (a.w || 1)).slice(0, 8);
      const out = this.edges.filter((e) => e.s === n.id).sort((a, b) => (b.w || 1) - (a.w || 1)).slice(0, 8);
      const row = (e, dir) =>
        el("button.maps-row", {
          dataset: { action: "select" },
          onclick: () => this.selectNode(dir === "in" ? e.s : e.t),
        },
          el("span.maps-cell.grow", {
            text: (dir === "in" ? e.source : e.target).label || (dir === "in" ? e.s : e.t),
            title: dir === "in" ? e.s : e.t,
          }),
          el("span.maps-cell.maps-num", { text: isCode ? `w${e.w || 1}` : (e.label || "") }),
        );
      this.detail.replaceChildren(
        el("div.maps-detail-head", {},
          el("div.maps-detail-title", { text: n.label || n.id }),
          el("button.maps-tool.maps-detail-close", { text: "×", title: "Закрыть (Esc)", dataset: { action: "close-detail" } }),
        ),
        el("div.maps-detail-path.mono", { text: n.id, title: n.id }),
        el("div.maps-detail-meta", {},
          isCode
            ? [el("span.maps-tag", { text: n.cluster || "—" }), el("span.maps-tag", { text: n.lang || "" })]
            : [el("span.maps-tag", { text: n.type || "—" }), el("span.maps-tag", { text: n.domain || "прочее" })],
        ),
        el("div.maps-table", {},
          el("div.maps-num.maps-table-caption", {
            text: `входящих ${n.degIn} · исходящих ${n.degOut}`,
          }),
        ),
        el("div.maps-table", { }, el("div.maps-table-caption", { text: "Топ входящих" }),
          ...inc.map((e) => row(e, "in"))),
        el("div.maps-table", { }, el("div.maps-table-caption", { text: "Топ исходящих" }),
          ...out.map((e) => row(e, "out"))),
        this.pin ? el("button.maps-tool", { text: "Снять закрепление", dataset: { action: "unpin" } }) : null,
      );
      this.detail.hidden = false;
    }

    // ── Диаграмма (§5) ─────────────────────────────────────────────────────
    _hideDiagram() {
      if (this.diagramBox) this.diagramBox.hidden = true;
    }

    _renderDiagram(ds) {
      if (!this.canvas) return;
      this.g.replaceChildren();
      this.nodeEls = new Map();
      this.edgeEls = [];
      this._stopSim();
      this._resetTransform();
      if (!this.diagramBox) {
        this.diagramBox = el("div.maps-diagram");
        this.canvas.append(this.diagramBox);
      }
      this.diagramBox.hidden = false;
      const kind = this.mode === "kg" ? "kg" : "code";
      this._diagramMode = kind;
      const box = this.diagramBox;
      box.replaceChildren(svgEl("svg", { class: "maps-dedges" }), el("div.maps-dgrid"));
      const grid = $(".maps-dgrid", box);
      const edgeSvg = $("svg.maps-dedges", box);

      if (kind === "code") this._diagramCode(ds, grid);
      else this._diagramKg(ds, grid);

      // стрелки между блоками — SVG поверх, координаты после layout
      this._arrowSvg = edgeSvg;
      requestAnimationFrame(() => this._layoutDiagramEdges(kind));
    }

    _block(title, size, items, cls) {
      return el("div.maps-dblock" + (cls ? "." + cls : {}), {},
        el("div.maps-dhead", {},
          el("span.maps-dlabel", { text: title }),
          el("span.maps-dsize.maps-num", { text: String(size) })),
        ...items.map((it) => el("div.maps-dfile", { text: it.label, title: it.id || it.label })),
      );
    }

    _diagramCode(ds, grid) {
      // агрегированные рёбра между кластерами
      const agg = new Map(); // "a\u0000b" -> {call, imp}
      for (const e of this.edges) {
        const a = e.source.cluster || "—", b = e.target.cluster || "—";
        if (a === b) continue;
        const k = `${a}\u0000${b}`;
        const rec = agg.get(k) || { call: 0, imp: 0 };
        if (e.kind === "import") rec.imp++; else rec.call += e.w || 1;
        agg.set(k, rec);
      }
      this._diagAgg = agg;
      const clusters = (ds.clusters.length
        ? ds.clusters
        : [...new Set(this.nodes.map((n) => n.cluster || "—"))].map((id) => ({ id, label: id, size: 0 })));
      grid.replaceChildren(...clusters.map((c, i) => {
        const members = this.nodes.filter((n) => (n.cluster || "—") === c.id)
          .sort((a, b) => (b.deg - a.deg) || String(a.label).localeCompare(String(b.label)));
        return this._block(c.label, c.size || members.length,
          members.slice(0, 5), hueClass(i));
      }));
    }

    _diagramKg(ds, grid) {
      // колонки-домены, внутри группы по типам
      const domains = new Map();
      for (const n of this.nodes) {
        const d = n.domain || "прочее";
        if (!domains.has(d)) domains.set(d, new Map());
        const tmap = domains.get(d);
        if (!tmap.has(n.type || "—")) tmap.set(n.type || "—", []);
        tmap.get(n.type || "—").push(n);
      }
      this._diagAgg = new Map();
      for (const e of this.edges) {
        const a = e.source.domain || "прочее", b = e.target.domain || "прочее";
        if (a === b) continue;
        const k = `${a}\u0000${b}`;
        this._diagAgg.set(k, (this._diagAgg.get(k) || 0) + 1);
      }
      const colEls = [];
      for (const [d, tmap] of [...domains].sort((a, b) => a[0].localeCompare(b[0]))) {
        let total = 0; for (const arr of tmap.values()) total += arr.length;
        const col = el("div.maps-dcol", {}, el("div.maps-dcol-title", { text: d }));
        for (const [t, arr] of [...tmap].sort((a, b) => a[0].localeCompare(b[0]))) {
          arr.sort((a, b) => String(a.label).localeCompare(String(b.label)));
          col.append(this._block(t, arr.length, arr.slice(0, 5),
            hueClass(Math.max(0, KG_TYPES.indexOf(t)))));
        }
        grid.append(col);
        colEls.push(col);
      }
    }

    _layoutDiagramEdges(kind) {
      const svg = this._arrowSvg;
      if (!svg || this.diagramBox.hidden) return;
      const boxR = this.diagramBox.getBoundingClientRect();
      svg.setAttribute("width", boxR.width);
      svg.setAttribute("height", boxR.height);
      svg.replaceChildren();
      const blocks = [...this.diagramBox.querySelectorAll(".maps-dblock")];
      const center = (elq) => {
        const r = elq.getBoundingClientRect();
        return {
          x: r.left + r.width / 2 - boxR.left, y: r.top + r.height / 2 - boxR.top,
          l: r.left - boxR.left, r: r.right - boxR.left, t: r.top - boxR.top, b: r.bottom - boxR.top,
        };
      };
      // блок → кластер/тип: по заголовку
      const titleOf = (elq) => {
        const t = $(".maps-dlabel", elq);
        return t ? t.textContent : "";
      };
      for (const [k, rec] of this._diagAgg || []) {
        const [a, b] = k.split("\u0000");
        const ea = blocks.find((elq) => titleOf(elq) === a);
        const eb = blocks.find((elq) => titleOf(elq) === b);
        if (!ea || !eb) continue;
        const ca = center(ea), cb = center(eb);
        const count = kind === "code" ? (rec.call + rec.imp) : rec;
        if (!count) continue;
        const x1 = ca.x <= cb.x ? ca.r : ca.l, x2 = ca.x <= cb.x ? cb.l : cb.r;
        const mx = (x1 + x2) / 2;
        const path = svgEl("path", {
          class: "maps-darrow",
          d: `M${x1},${ca.y} C${mx},${ca.y} ${mx},${cb.y} ${x2},${cb.y}`,
          "stroke-width": clamp(Math.round(Math.log2(count + 1)), 1, 3),
        });
        svg.append(path);
        const label = kind === "code"
          ? `${count} ${rec.imp && rec.call ? "рёбер" : rec.imp ? "импорт" : rec.call === 1 ? "вызов" : "вызовов"}`
          : `${count} рёбер`;
        const txt = svgEl("text", { class: "maps-darrow-label", x: mx, y: (ca.y + cb.y) / 2 - 4, "text-anchor": "middle" });
        txt.textContent = label;
        svg.append(txt);
      }
    }
  }

  OMP.Maps = Maps;
  // Синглтон: если app.js (W2/W3-интеграция) создаёт свой экземпляр — он перезапишет.
  if (!OMP.maps) OMP.maps = new Maps({ el: "#maps-view" });
})(window.OMP);
