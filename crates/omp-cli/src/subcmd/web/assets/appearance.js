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
  const GLYPHS = ["brain", "triangle", "minimal"];

  const W = (file, name) => ({ url: `/wallpapers/${file}`, name });
  const BUILTIN_WALLPAPERS = [
    W("anime-sakura.svg", "Sakura night"),
    W("anime-city.svg", "City dusk"),
    W("anime-sky.svg", "Summer sky"),
    W("vaporwave-sunset.svg", "Vaporwave sunset"),
    W("vaporwave-grid.svg", "Synthwave grid"),
    W("midnight.svg", "Midnight aurora"),
  ];

  const THEMES = [
    { id: "calm", name: "Graphite", sw: ["#181818", "#1f1f1f", "#1a88ff", "#d6d6d6"], wall: "" },
    { id: "omt", name: "OMT", sw: ["#171024", "#1f1730", "#ff5cc8", "#ece8f4"], wall: "" },
    { id: "midnight", name: "Midnight", sw: ["#0b0f16", "#121826", "#7fd4ff", "#e6edf6"], wall: "/wallpapers/midnight.svg" },
    { id: "light", name: "Light", sw: ["#eef3f6", "#ffffff", "#119c95", "#19303f"], wall: "" },
    { id: "sakura", name: "Sakura", sw: ["#1a1230", "#221a3c", "#ff9ec7", "#f5edf6"], wall: "/wallpapers/anime-sakura.svg" },
    { id: "tokyo-dusk", name: "Tokyo Dusk", sw: ["#211a17", "#2a211d", "#ff8c42", "#f2e9e0"], wall: "/wallpapers/anime-city.svg" },
    { id: "summer-sky", name: "Summer Sky", sw: ["#e9f2fb", "#ffffff", "#ffb020", "#16334f"], wall: "/wallpapers/anime-sky.svg" },
    { id: "vaporwave", name: "Vaporwave", sw: ["#221739", "#2c2050", "#ff9ad5", "#f0e8fa"], wall: "/wallpapers/vaporwave-sunset.svg" },
    { id: "synthwave", name: "Synthwave", sw: ["#0d0a24", "#161134", "#00e0ff", "#eef0ff"], wall: "/wallpapers/vaporwave-grid.svg" },
    { id: "phosphor", name: "Phosphor", sw: ["#0a0f0c", "#111811", "#3df59a", "#ccf2d8"], wall: "" },
    { id: "paper", name: "Paper", sw: ["#f6f3ec", "#ffffff", "#e23b23", "#191612"], wall: "" },
    { id: "drafting", name: "Drafting", sw: ["#0e1c30", "#142542", "#5bd0ff", "#d9eaf5"], wall: "" },
  ];
  const DEFAULTS = { theme: "calm", wallpaper: "", dim: 35, blur: 0, glass: 78 };

  function load() {
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { /* reset */ }
    // Legacy single toggle (dark | light).
    const legacy = localStorage.getItem("omp-web-theme");
    if (!saved.theme && legacy) saved.theme = legacy === "light" ? "light" : "omt";
    const out = { ...DEFAULTS, ...saved };
    if (!THEMES.some((t) => t.id === out.theme)) out.theme = DEFAULTS.theme;
    return out;
  }
  let cfg = load();
  const save = () => localStorage.setItem(KEY, JSON.stringify(cfg));

  function loadGlyph() {
    try {
      const saved = localStorage.getItem(GLYPH_KEY);
      if (GLYPHS.includes(saved)) return saved;
    } catch { /* storage can be unavailable in embedded previews */ }
    return "brain";
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

  async function open() {
    const themeGrid = el("div.ap-themes");
    const wallGrid = el("div.ap-walls");
    const glyphGrid = el("div.ap-themes");
    const fileInput = el("input", { type: "file", accept: "image/png,image/jpeg,image/gif,image/webp,image/avif", multiple: true, hidden: true });
    let custom = await customWallpapers();

    function renderThemes() {
      themeGrid.replaceChildren(...THEMES.map((t) => el("button.ap-theme" + (cfg.theme === t.id ? ".sel" : ""), {
        title: t.wall ? `${t.name} (with its wallpaper)` : t.name,
        onclick: () => { set({ theme: t.id, ...(t.wall !== undefined ? { wallpaper: t.wall } : {}) }); renderAll(); },
      },
      el("span.ap-preview", { style: `background:${t.sw[0]}${t.wall ? `;background-image:${cssUrl(t.wall)}` : ""}` },
        el("span.ap-pv-panel", { style: `background:${t.sw[1]}` }),
        el("span.ap-pv-bubble", { style: `background:${t.sw[2]}` }),
        el("span.ap-pv-line", { style: `background:${t.sw[3]}` }),
        el("span.ap-pv-line.short", { style: `background:${t.sw[3]}` })),
      el("span.ap-name", { text: t.name }))));
    }

    function wallTile(w, removable) {
      const tile = el("div.ap-wall" + (cfg.wallpaper === w.url ? ".sel" : ""), {
        title: w.name,
        role: "button",
        tabindex: "0",
        style: `background-image:${cssUrl(w.url)}`,
        onclick: () => { set({ wallpaper: w.url }); renderAll(); },
      }, el("span.ap-wall-name", { text: w.name }));
      if (removable) {
        tile.append(el("button.ap-del", {
          html: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>',
          title: "Delete wallpaper",
          onclick: async (e) => {
            e.stopPropagation();
            try { await OMP.api("/api/wallpapers/" + encodeURIComponent(w.name), { method: "DELETE" }); } catch (err) { return OMP.toast(err.message, "error"); }
            if (cfg.wallpaper === w.url) set({ wallpaper: "" });
            custom = await customWallpapers();
            renderAll();
          },
        }));
      }
      return tile;
    }

    function renderWalls() {
      const isCustomUrl = cfg.wallpaper && !cfg.wallpaper.startsWith("/wallpapers/");
      wallGrid.replaceChildren(
        el("div.ap-wall.none" + (!cfg.wallpaper ? ".sel" : ""), { role: "button", tabindex: "0", onclick: () => { set({ wallpaper: "" }); renderAll(); } }, el("span.ap-wall-name", { text: "None" })),
        ...BUILTIN_WALLPAPERS.map((w) => wallTile(w, false)),
        ...custom.map((w) => wallTile(w, true)),
        ...(isCustomUrl ? [wallTile({ url: cfg.wallpaper, name: "From URL" }, false)] : []),
        el("div.ap-wall.add", { role: "button", tabindex: "0", title: "Upload images (or drop them here)", onclick: () => fileInput.click() }, el("span.ap-plus", { text: "+" }), el("span.ap-wall-name", { text: "Add wallpaper" })),
        el("div.ap-wall.add", {
          role: "button", tabindex: "0", title: "Use an image URL",
          onclick: async () => {
            const u = await OMP.dialogs.promptText("Wallpaper URL", { placeholder: "https://example.com/wallpaper.jpg" }).promise;
            if (u == null) return;
            const safe = safeUrl(u.trim());
            if (!safe) return OMP.toast("Only http(s) image URLs are supported", "warning");
            set({ wallpaper: safe });
            renderAll();
          },
        }, el("span.ap-plus", { html: '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>' }), el("span.ap-wall-name", { text: "From URL" })),
      );
    }

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
        slider("Panels", "glass", 30, 100, "%"),
      );
      sliders.hidden = !cfg.wallpaper;
    }
    function renderGlyphs() {
      const labels = { brain: "Brain", triangle: "Triangle", minimal: "Minimal" };
      const hints = { brain: "Brains on the markers (default)", triangle: "Classic arrows", minimal: "Plain dots" };
      const swatch = {
        brain: () => el("span", { style: "width:30px;height:24px;background:currentColor;-webkit-mask:var(--brain-glyph) center / contain no-repeat;mask:var(--brain-glyph) center / contain no-repeat" }),
        triangle: () => el("span", { text: "\u25B8", style: "font-size:17px" }),
        minimal: () => el("span", { style: "width:10px;height:10px;border-radius:50%;background:currentColor" }),
      };
      glyphGrid.replaceChildren(...GLYPHS.map((id) => el("button.ap-theme" + (glyph === id ? ".sel" : ""), {
        title: hints[id],
        onclick: () => { setGlyph(id); renderAll(); },
      },
      el("span.ap-preview", { style: "display:flex;align-items:center;justify-content:center;background:var(--bg-inset);color:var(--fg-muted)" }, swatch[id]()),
      el("span.ap-name", { text: labels[id] }))));
    }
    function renderAll() { renderThemes(); renderWalls(); renderSliders(); renderGlyphs(); }

    fileInput.addEventListener("change", async () => {
      const added = await upload([...fileInput.files]);
      fileInput.value = "";
      custom = await customWallpapers();
      if (added) set({ wallpaper: added.url });
      renderAll();
    });

    const body = el("div.ap", null,
      el("div.ap-title", { text: "Theme" }), themeGrid,
      el("div.ap-title", { text: "Activity glyph" }), glyphGrid,
      el("div.ap-title", null, el("span", { text: "Wallpaper" }), el("span.faint.small", { text: "drop images here to add them" })), wallGrid,
      sliders, fileInput);
    if (OMP.desktop && OMP.desktop.available) body.append(desktopSection());
    renderAll();
    const m = OMP.dialogs.modal({
      title: "Appearance",
      body,
      wide: true,
      cls: "appearance",
      foot: (done) => [el("span.hint", { text: "Saved in this browser · custom wallpapers live in ~/.omp-rs/web/wallpapers" }), el("button.primary", { text: "Done", onclick: () => done(true) })],
    });
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
