"use strict";
// Themes and wallpapers: the palette (`data-theme` on <html>), a fixed
// wallpaper layer behind glass panels, dim/blur/glass sliders, and user
// wallpapers uploaded to the server (`/api/wallpapers`). Persisted in
// localStorage; applied before first paint by `OMP.appearance.apply()`.
(function (OMP) {
  const { el } = OMP;
  const KEY = "omt-appearance";

  const W = (file, name) => ({ url: `/wallpapers/${file}`, name });
  const BUILTIN_WALLPAPERS = [
    W("anime-sakura.svg", "Sakura night"),
    W("anime-city.svg", "City dusk"),
    W("anime-sky.svg", "Summer sky"),
    W("vaporwave-sunset.svg", "Vaporwave sunset"),
    W("vaporwave-grid.svg", "Synthwave grid"),
    W("midnight.svg", "Midnight aurora"),
  ];

  /** id, name, preview swatches [bg, panel, accent, fg], suggested wallpaper. */
  const THEMES = [
    { id: "omt", name: "OMT", sw: ["#0f0b14", "#16111c", "#ed4abf", "#e6e3ea"], wall: "" },
    { id: "midnight", name: "Midnight", sw: ["#050506", "#0c0c0f", "#ff5ca8", "#e8e8ec"], wall: "/wallpapers/midnight.svg" },
    { id: "light", name: "Light", sw: ["#fbf9fc", "#ffffff", "#c42a95", "#2b2233"], wall: "" },
    { id: "sakura", name: "Sakura", sw: ["#120f24", "#1a1630", "#ff8fc7", "#f3eaf6"], wall: "/wallpapers/anime-sakura.svg" },
    { id: "tokyo-dusk", name: "Tokyo Dusk", sw: ["#141024", "#1d1730", "#ffa65c", "#f2e9f0"], wall: "/wallpapers/anime-city.svg" },
    { id: "summer-sky", name: "Summer Sky", sw: ["#eef6ff", "#ffffff", "#2f7de1", "#1c2a3d"], wall: "/wallpapers/anime-sky.svg" },
    { id: "vaporwave", name: "Vaporwave", sw: ["#1a0b2e", "#241040", "#ff71ce", "#f4e8ff"], wall: "/wallpapers/vaporwave-sunset.svg" },
    { id: "synthwave", name: "Synthwave", sw: ["#0b0820", "#120e2e", "#01cdfe", "#e6f7ff"], wall: "/wallpapers/vaporwave-grid.svg" },
  ];
  const DEFAULTS = { theme: "omt", wallpaper: "", dim: 35, blur: 0, glass: 78 };

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
          text: "✕",
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
    function renderAll() { renderThemes(); renderWalls(); renderSliders(); }

    fileInput.addEventListener("change", async () => {
      const added = await upload([...fileInput.files]);
      fileInput.value = "";
      custom = await customWallpapers();
      if (added) set({ wallpaper: added.url });
      renderAll();
    });

    const body = el("div.ap", null,
      el("div.ap-title", { text: "Theme" }), themeGrid,
      el("div.ap-title", null, el("span", { text: "Wallpaper" }), el("span.faint.small", { text: "drop images here to add them" })), wallGrid,
      sliders, fileInput);
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

  apply();
  OMP.appearance = { apply, open, set, get: () => ({ ...cfg }), THEMES, BUILTIN_WALLPAPERS };
})(window.OMP);
