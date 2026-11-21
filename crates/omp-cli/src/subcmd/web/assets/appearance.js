"use strict";
// Themes and wallpapers: the palette (`data-theme` on <html>), a fixed
// wallpaper layer behind glass panels, dim/blur/glass sliders, and user
// wallpapers uploaded to the server (`/api/wallpapers`). Also the activity
// glyph (`data-glyph` on <body>) shown by the transcript. Persisted in
// localStorage; applied before first paint by `OMP.appearance.apply()`.
(function (OMP) {
  const { el } = OMP;
  const KEY = "omt-appearance";

  // Transcript activity glyph: collapse markers and the running spinner
  // take their shape from `data-glyph` on <body> (see stream.css).
  const GLYPH_KEY = "omp-activity-glyph";
  const GLYPHS = ["pulse", "triangle", "minimal"];

  const BUILTIN_WALLPAPERS = [
    { url: "/wallpapers/saturn.webp", name: "Saturn", credit: "NASA/JPL/Space Science Institute", source: "https://science.nasa.gov/photojournal/in-saturn-shadow/" },
    { url: "/wallpapers/clementine.webp", name: "Moon", credit: "NASA/JPL/USGS", source: "https://images.nasa.gov/details/PIA00302" },
  ];
  const THEMES = [
    { id: "calm", name: "Graphite", sw: ["#181818", "#1f1f1f", "#1a88ff", "#d6d6d6"] },
    { id: "ink", name: "Ink", sw: ["#101519", "#171e23", "#a0c8d9", "#e0e8ed"] },
    { id: "night", name: "Night", sw: ["#0b1020", "#131a2e", "#8fb5ff", "#dfe6f5"] },
    { id: "vaporwave", name: "Vaporwave", sw: ["#1a1030", "#241740", "#ff71ce", "#f2e9ff"] },
    { id: "paper", name: "Paper", sw: ["#f5f3ed", "#fcfaf5", "#35594a", "#252821"] },
  ];
  const DEFAULTS = { theme: "calm", wallpaper: "", dim: 55, blur: 0, glass: 92 };
  const RETIRED_WALL = /^\/wallpapers\/(?:anime-(?:sakura|city|sky)|vaporwave-(?:sunset|grid)|midnight)\.svg$/;

  function load() {
    let saved = {};
    let legacy;
    try {
      saved = JSON.parse(localStorage.getItem(KEY) || "{}") || {};
      legacy = localStorage.getItem("omp-web-theme");
    } catch { /* reset */ }
    const out = { ...DEFAULTS, ...saved };
    if (saved.theme === "light" || saved.theme === "summer-sky" || (!saved.theme && legacy === "light")) out.theme = "paper";
    if (saved.theme === "midnight" || saved.theme === "drafting") out.theme = "ink";
    if (!THEMES.some((t) => t.id === out.theme)) out.theme = DEFAULTS.theme;
    out.wallpaper = safeUrl(out.wallpaper);
    if (RETIRED_WALL.test(out.wallpaper)) out.wallpaper = "";
    for (const [key, min, max] of [["dim", 0, 85], ["blur", 0, 24], ["glass", 75, 100]]) {
      out[key] = Number.isFinite(out[key]) ? Math.max(min, Math.min(max, out[key])) : DEFAULTS[key];
    }
    return out;
  }
  let cfg = load();
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(cfg)); } catch { /* temporary preview */ } };

  function loadGlyph() {
    try {
      const saved = localStorage.getItem(GLYPH_KEY);
      if (GLYPHS.includes(saved)) return saved;
    } catch { /* storage can be unavailable in embedded previews */ }
    return "pulse";
  }
  let glyph = loadGlyph();

  /** Only same-origin wallpaper paths and http(s) URLs reach CSS. */
  function safeUrl(u) {
    if (typeof u !== "string" || !u) return "";
    if (u.startsWith("/wallpapers/")) return u;
    try {
      const p = new URL(u);
      return p.protocol === "https:" || p.protocol === "http:" ? p.href : "";
    } catch { return ""; }
  }
  const cssUrl = (u) => `url("${u.replace(/["\\\n\r]/g, (c) => encodeURIComponent(c))}")`;

  function apply() {
    const root = document.documentElement;
    root.dataset.theme = cfg.theme;
    document.body.dataset.glyph = glyph;
    const url = safeUrl(cfg.wallpaper);
    const layer = document.getElementById("wallpaper");
    if (layer) layer.style.backgroundImage = url ? cssUrl(url) : "";
    root.classList.toggle("has-wall", !!url);
    root.style.setProperty("--wall-dim", String(cfg.dim / 100));
    root.style.setProperty("--wall-blur", `${cfg.blur}px`);
    root.style.setProperty("--glass", `${cfg.glass}%`);
  }

  function set(patch) {
    Object.assign(cfg, patch);
    save();
    apply();
  }

  function setGlyph(id) {
    if (!GLYPHS.includes(id)) return;
    glyph = id;
    try { localStorage.setItem(GLYPH_KEY, id); } catch { /* non-persistent preview */ }
    apply();
  }

  async function customWallpapers() {
    try { return (await OMP.api("/api/wallpapers")).custom || []; } catch { return []; }
  }

  async function upload(files) {
    let last = null;
    for (const f of files) {
      if (!f.type.startsWith("image/")) { OMP.toast(`${f.name}: not an image`, "warning"); continue; }
      if (f.type === "image/svg+xml") { OMP.toast(`${f.name}: SVG uploads are not allowed (use PNG/JPEG/WebP)`, "warning"); continue; }
      try {
        const res = await fetch("/api/wallpapers?name=" + encodeURIComponent(f.name), { method: "POST", body: f });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);
        last = data;
      } catch (e) { OMP.toast(`${f.name}: ${e.message}`, "error"); }
    }
    return last;
  }

  async function open(sectionId) {
    if (document.querySelector(".modal.appearance")) return;
    const themeGrid = el("div.ap-themes");
    const wallGrid = el("div.ap-walls");
    const glyphGrid = el("div.ap-themes");
    const motionPreview = el("div.ap-motion-preview", { "aria-label": "Circuit: monochrome thinking and tool animations" });
    const fileInput = el("input", { type: "file", accept: "image/png,image/jpeg,image/gif,image/webp,image/avif", multiple: true, hidden: true });
    let custom = [];

    function choose(theme, wallpaper) {
      const focused = document.activeElement?.dataset.apChoice;
      set(theme ? { theme } : { wallpaper });
      renderAll();
      if (focused) [...body.querySelectorAll("[data-ap-choice]")].find((node) => node.dataset.apChoice === focused)?.focus({ preventScroll: true });
    }
    function renderThemes() {
      themeGrid.replaceChildren(...THEMES.map((t) => el("button.ap-theme", {
        type: "button", "aria-pressed": String(cfg.theme === t.id), dataset: { apChoice: t.id },
        onclick: () => choose(t.id),
      }, el("span.ap-swatches", { "aria-hidden": "true" }, t.sw.map((color) => el("span", { style: `background:${color}` }))),
      el("span", { text: t.name }))));
    }

    function wallTile(w, removable) {
      const active = cfg.wallpaper === w.url;
      const button = el("button.ap-wall", {
        type: "button", "aria-pressed": String(active), dataset: { apChoice: w.url || "none" },
        onclick: () => choose(null, w.url),
      }, el("span.ap-wall-preview", { style: w.url ? `background-image:${cssUrl(w.url)}` : null, "aria-hidden": "true" },
        !w.url ? el("span.ap-no-wall", { text: "—" }) : null), el("span.ap-wall-name", { text: w.name }));
      const item = el("div.ap-wall-item", null, button);
      if (removable) item.append(el("button.ap-del", {
        type: "button", text: "×", title: "Delete wallpaper", "aria-label": `Delete ${w.name}`,
        onclick: async () => {
          try { await OMP.api("/api/wallpapers/" + encodeURIComponent(w.name), { method: "DELETE" }); } catch (err) { return OMP.toast(err.message, "error"); }
          if (cfg.wallpaper === w.url) set({ wallpaper: "" });
          custom = await customWallpapers();
          renderAll();
          wallGrid.querySelector("button")?.focus();
        },
      }));
      return item;
    }
    const credit = el("div.ap-credit");
    function renderWalls() {
      const customUrl = cfg.wallpaper && !cfg.wallpaper.startsWith("/wallpapers/");
      wallGrid.replaceChildren(wallTile({ name: "None", url: "" }, false),
        ...BUILTIN_WALLPAPERS.map((w) => wallTile(w, false)), ...custom.filter((w) => safeUrl(w.url)).map((w) => wallTile(w, true)),
        ...(customUrl ? [wallTile({ url: cfg.wallpaper, name: "Custom URL" }, false)] : []));
      const source = BUILTIN_WALLPAPERS.find((w) => w.url === cfg.wallpaper);
      credit.replaceChildren(...(source ? [el("a", { text: `Image: ${source.credit}`, href: source.source, target: "_blank", rel: "noopener noreferrer" })] : []));
    }
    const wallActions = el("div.ap-actions", null,
      el("button.secondary", { type: "button", text: "Upload image…", onclick: () => fileInput.click() }),
      el("button.secondary", { type: "button", text: "From URL…", onclick: async () => {
        const input = await OMP.dialogs.promptText("Wallpaper URL", { placeholder: "https://example.com/wallpaper.jpg" }).promise;
        if (input == null) return;
        const safe = safeUrl(input.trim());
        if (!safe || !/^https?:/.test(safe)) return OMP.toast("Only http(s) image URLs are supported", "warning");
        choose(null, safe);
      } }));

    const slider = (label, prop, min, max, unit) => {
      const out = el("span.ap-val", { text: `${cfg[prop]}${unit}` });
      const input = el("input", { type: "range", min: String(min), max: String(max), value: String(cfg[prop]) });
      input.addEventListener("input", () => { set({ [prop]: Number(input.value) }); out.textContent = `${cfg[prop]}${unit}`; });
      return el("label.ap-slider", null, el("span", { text: label }), input, out);
    };
    const sliders = el("div.ap-sliders");
    function renderSliders() {
      sliders.replaceChildren(
        slider("Dim", "dim", 0, 85, "%"),
        slider("Blur", "blur", 0, 24, "px"),
        slider("Panel opacity", "glass", 75, 100, "%"),
      );
      sliders.hidden = !cfg.wallpaper;
    }
    function renderGlyphs() {
      const labels = { pulse: "Pulse", triangle: "Arrow", minimal: "Dot" };
      const symbols = { pulse: "span.activity-pulse", triangle: "span.ap-arrow", minimal: "span.ap-dot" };
      glyphGrid.replaceChildren(...GLYPHS.map((id) => el("button.ap-theme", {
        type: "button", "aria-pressed": String(glyph === id), dataset: { apChoice: id },
        onclick: () => {
          const focused = document.activeElement?.dataset.apChoice;
          setGlyph(id);
          renderGlyphs();
          if (focused === id) glyphGrid.querySelector(`[data-ap-choice="${id}"]`)?.focus();
        },
      }, el(symbols[id], { "aria-hidden": "true", text: id === "triangle" ? "▸" : null }), el("span", { text: labels[id] }))));
    }
    const brainPreview = el("span"), toolPreview = el("span");
    motionPreview.append(brainPreview, toolPreview);
    OMP.mascot.mount(brainPreview, { size: 48, state: "think" });
    OMP.mascot.mount(toolPreview, { size: 48, state: "work" });
    function renderAll() { renderThemes(); renderWalls(); renderSliders(); renderGlyphs(); }

    fileInput.addEventListener("change", async () => {
      const added = await upload([...fileInput.files]);
      fileInput.value = "";
      custom = await customWallpapers();
      if (added) set({ wallpaper: added.url });
      renderAll();
    });

    const section = (id, name, ...nodes) => el("section.ap-section", { id }, el("h3.ap-title", { text: name }), ...nodes);
    const body = el("div.ap", null,
      section("ap-theme", "Theme", themeGrid),
      section("ap-wallpaper", "Wallpaper", wallGrid, credit, wallActions, sliders, fileInput),
      section("ap-motion", "Thinking & tools · Circuit", motionPreview),
      section("ap-activity", "Activity glyph", glyphGrid));
    if (OMP.desktop && OMP.desktop.available) body.append(desktopSection());
    renderAll();
    const m = OMP.dialogs.modal({
      title: "Settings",
      body,
      wide: true,
      cls: "appearance",
      foot: (done) => [el("span.hint", { text: "Applies immediately" }), el("button.primary", { text: "Done", onclick: () => done(true) })],
    });
    if (sectionId === "activity") m.box.querySelector("#ap-activity button[aria-pressed='true']")?.focus();
    customWallpapers().then((walls) => { if (m.box.isConnected) { custom = walls; renderWalls(); } });
    m.box.addEventListener("dragover", (e) => { e.preventDefault(); m.box.classList.add("drop"); });
    m.box.addEventListener("dragleave", () => m.box.classList.remove("drop"));
    m.box.addEventListener("drop", async (e) => {
      e.preventDefault();
      m.box.classList.remove("drop");
      const added = await upload([...(e.dataTransfer?.files || [])]);
      custom = await customWallpapers();
      if (added) set({ wallpaper: added.url });
      renderAll();
    });
    return m.promise;
  }

  // ── Tau desktop settings: capture shortcut, DevTools port ──────────────
  /** "Ctrl+Alt+S" from a keydown, or null while only modifiers are held. */
  function comboOf(e) {
    if (["Control", "Shift", "Alt", "Meta"].includes(e.key)) return null;
    const key = /^Key[A-Z]$/.test(e.code) ? e.code.slice(3) : /^Digit\d$/.test(e.code) ? e.code.slice(5) : e.code;
    const mods = [e.ctrlKey && "Ctrl", e.altKey && "Alt", e.shiftKey && "Shift", e.metaKey && "Super"].filter(Boolean);
    return mods.length ? [...mods, key].join("+") : null;
  }

  function desktopSection() {
    const status = el("span.ap-desk-status.small");
    const keyBtn = el("button.chip.ap-key", { text: "…", title: "Click, then press the new shortcut (Esc cancels)" });
    const portIn = el("input.ap-port", { type: "number", min: "1024", max: "65535", placeholder: "off" });
    const portNote = el("span.faint.small");
    let current = "";

    async function load() {
      const s = await OMP.desktop.settings();
      current = s.captureShortcut;
      keyBtn.textContent = current;
      status.textContent = s.captureShortcutError ? "⚠ " + s.captureShortcutError : "active";
      status.classList.toggle("err", !!s.captureShortcutError);
      portIn.value = s.devtoolsPort ?? "";
      portNote.textContent = s.devtoolsActive ? `running on 127.0.0.1:${s.devtoolsActive}` : "off in this run";
    }

    keyBtn.addEventListener("click", () => {
      keyBtn.textContent = "press keys…";
      keyBtn.classList.add("rec");
      const onKey = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.key === "Escape") { done(); keyBtn.textContent = current; return; }
        const combo = comboOf(e);
        if (!combo) return;
        done();
        try {
          await OMP.desktop.setCaptureShortcut(combo);
          OMP.toast(`Capture shortcut: ${combo}`, "success");
        } catch (err) {
          OMP.toast(String(err), "error", 7000);
        }
        load();
      };
      const done = () => { keyBtn.classList.remove("rec"); window.removeEventListener("keydown", onKey, true); };
      window.addEventListener("keydown", onKey, true);
    });

    portIn.addEventListener("change", async () => {
      const v = portIn.value.trim();
      try {
        await OMP.desktop.setDevtoolsPort(v ? Number(v) : null);
        OMP.toast(v ? `DevTools port ${v}: restart Tau (tray → Quit) to apply` : "DevTools port off after restart", "info", 6000);
      } catch (err) {
        OMP.toast(String(err), "error");
      }
      load();
    });

    load().catch((e) => { status.textContent = "⚠ " + e; });
    return el("div.ap-desk", null,
      el("div.ap-title", { text: "Desktop (Tau)" }),
      el("label.ap-desk-row", null, el("span", { text: "Screen capture shortcut" }), keyBtn, status),
      el("label.ap-desk-row", null, el("span", { text: "DevTools port (automation, 127.0.0.1)" }), portIn, portNote));
  }

  apply();
  OMP.appearance = { apply, open, set, setGlyph, get: () => ({ ...cfg, glyph }), THEMES, BUILTIN_WALLPAPERS };
})(window.OMP);
