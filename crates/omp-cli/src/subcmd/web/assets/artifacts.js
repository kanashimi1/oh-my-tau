"use strict";
// Right-hand artifact panel (code, diff, HTML, markdown, images). Owner: artifacts slice.
// API: OMP.artifacts.open({kind,title,content,lang,src}) / close() / isOpen().
(function (OMP) {
  const { el, esc } = OMP;
  const LS_WIDTH = "omp-artifacts-width";
  const OPEN_KINDS = new Set(["code", "diff", "html", "markdown", "image", "file", "files", "gallery"]);

  // Files on the agent host previewed via /preview/<path> (preview.rs).
  const EXT = {
    html: ["html", "htm", "svg"],
    image: ["png", "jpg", "jpeg", "gif", "webp", "avif", "bmp", "ico"],
    video: ["mp4", "m4v", "webm", "mov", "mkv", "ogv"],
    audio: ["mp3", "wav", "ogg", "oga", "opus", "m4a", "flac"],
    pdf: ["pdf"],
    office: ["ppt", "pptx", "pps", "ppsx", "odp", "doc", "docx", "odt", "rtf", "xls", "xlsx", "ods"],
    markdown: ["md", "markdown"],
    model: ["bbmodel", "glb", "gltf"],
    text: ["txt", "log", "csv", "tsv", "json", "py", "rs", "js", "mjs", "ts", "tsx", "jsx", "css", "toml", "yaml", "yml", "sh", "xml", "ini", "cfg", "c", "h", "cpp", "go", "java", "lua", "sql"],
  };
  const extOf = (p) => (String(p).match(/\.([A-Za-z0-9]+)$/) || [])[1]?.toLowerCase() || "";
  const fileKind = (p) => Object.keys(EXT).find((k) => EXT[k].includes(extOf(p))) || null;
  const previewUrl = (p) => "/preview" + String(p).split("/").map(encodeURIComponent).join("/");
  const baseName = (p) => String(p).split(/[\\/]/).pop();

  let root = null;          // aside#artifacts-panel
  let appEl = null;
  let current = null;       // active artifact object
  const history = [];       // [{kind,title,content,lang,src}]
  let width = clampWidth(Number(localStorage.getItem(LS_WIDTH)) || 460);
  let htmlMode = "preview"; // per-kind toggle state for html
  let imageFit = "fit";

  function clampWidth(w) { return Math.min(Math.max(w, 280), Math.round(window.innerWidth * 0.85)); }

  function setWidth(w) {
    width = clampWidth(w);
    if (appEl) appEl.style.setProperty("--artifacts-w", width + "px");
    if (root) root.style.setProperty("--artifacts-w", width + "px");
  }

  function ensureDom() {
    if (root) return;
    appEl = document.getElementById("app");
    root = el("aside.artifacts-panel", { id: "artifacts-panel", tabindex: "-1" });
    setWidth(width);
    root.innerHTML =
      '<header class="art-head">' +
      '<div class="art-title-row"><span class="art-title" id="art-title"></span>' +
      '<span class="art-actions" id="art-actions"></span>' +
      '<button class="icon-btn art-close" title="Close (Esc)" aria-label="Close">' +
      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg></button></div>' +
      '<div class="art-tabs" id="art-tabs"></div></header>' +
      '<div class="art-body" id="art-body"></div>' +
      '<div class="art-resize" title="Drag to resize"></div>';
    // insert next to <main id="main">
    const main = document.getElementById("main");
    main.parentNode.insertBefore(root, main.nextSibling);
    applyOverlayMode();

    root.querySelector(".art-close").addEventListener("click", () => OMP.artifacts.close());

    // resize drag
    const grip = root.querySelector(".art-resize");
    grip.addEventListener("pointerdown", (e) => {
      if (overlayMode()) return;
      e.preventDefault();
      grip.setPointerCapture(e.pointerId);
      const startX = e.clientX, startW = root.getBoundingClientRect().width;
      const move = (ev) => setWidth(Math.round(startW + (startX - ev.clientX)));
      const up = () => {
        grip.removeEventListener("pointermove", move);
        grip.removeEventListener("pointerup", up);
        localStorage.setItem(LS_WIDTH, String(width));
      };
      grip.addEventListener("pointermove", move);
      grip.addEventListener("pointerup", up);
    });

    // Esc closes when focus is inside the panel; media keys for gallery/video/pdf views
    root.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        if (compareOpen) { compareOpen = false; show(); return; }
        OMP.artifacts.close();
        return;
      }
      const typing = e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.isContentEditable;
      if (typing) return;
      if (current === GALLERY && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
        if (galleryMove(e.key === "ArrowRight" ? 1 : -1)) e.preventDefault();
      } else if (live && live.kind === "video" && !e.ctrlKey && !e.altKey && !e.metaKey) {
        if (e.key === ",") { e.preventDefault(); live.stepFrame(-1); }
        else if (e.key === ".") { e.preventDefault(); live.stepFrame(1); }
      }
    });

    window.addEventListener("resize", () => { applyOverlayMode(); setWidth(width); });
  }

  function overlayMode() { return window.innerWidth < 900; }
  function applyOverlayMode() { root.classList.toggle("overlay", overlayMode()); }

  // ── renderers ──────────────────────────────────────────────────────────
  function renderBody(a) {
    stopLive();
    root.classList.remove("wide");
    const body = root.querySelector("#art-body");
    body.replaceChildren();
    const actions = root.querySelector("#art-actions");
    actions.replaceChildren();
    htmlMode = "preview";
    imageFit = "fit";

    if (a.kind === "code" || a.kind === "diff") renderCode(body, actions, a);
    else if (a.kind === "html") renderHtml(body, actions, a);
    else if (a.kind === "markdown") renderMarkdown(body, actions, a);
    else if (a.kind === "image") renderImage(body, actions, a);
    else if (a.kind === "gallery") renderGallery(body, actions);
    else if (a.kind === "file") renderFile(body, actions, a);
    else if (a.kind === "files") renderFiles(body);
    else body.append(el("p.muted", { text: "Unsupported artifact." }));
  }

  function renderCode(body, actions, a) {
    const pre = el("pre.art-code");
    if (a.kind === "diff") {
      pre.innerHTML = OMP.highlightDiff(a.content || "");
    } else {
      const lines = String(a.content ?? "").replace(/\r\n?/g, "\n").split("\n");
      const gutter = lines.map((_, i) => `<span class="ln">${i + 1}</span>`).join("");
      const hl = lines.map((l) => `<span class="lc">${OMP.highlight(l, a.lang)}</span>`).join("\n");
      pre.innerHTML = `<code class="art-code-inner"><span class="art-gutter">${gutter}</span><span class="art-code-lines">${hl}\n</span></code>`;
    }
    actions.append(
      el("button.icon-btn", { text: "copy", title: "Copy content", onclick: async (e) => {
        if (await OMP.copyText(a.content ?? "")) { e.target.textContent = "copied"; setTimeout(() => (e.target.textContent = "copy"), 1200); }
      } }));
    if (a.kind === "diff") pre.classList.add("art-diff");
    body.append(pre);
  }

  function renderHtml(body, actions, a) {
    const toggle = el("button.icon-btn", { text: "Source", title: "Toggle preview/source", onclick: () => {
      htmlMode = htmlMode === "preview" ? "source" : "preview";
      toggle.textContent = htmlMode === "preview" ? "Source" : "Preview";
      draw();
    } });
    actions.append(toggle, openTabBtn(a));
    const draw = () => {
      body.replaceChildren();
      if (htmlMode === "preview") {
        const frame = el("iframe.art-frame", { sandbox: "allow-scripts", title: a.title || "HTML preview" });
        frame.setAttribute("srcdoc", String(a.content ?? ""));
        body.append(frame);
      } else {
        const pre = el("pre.art-code");
        pre.innerHTML = OMP.highlight(a.content ?? "", "html");
        body.append(pre);
      }
    };
    draw();
  }

  // ── image gallery of this session ──────────────────────────────────────
  // Only images actually seen (opened, embedded as artifacts, or mentioned
  // and confirmed to exist) — never a blind directory listing.
  const GALLERY = { kind: "gallery", title: "Image gallery" };
  const gallery = []; // {path?, src?, title}, newest first, capped
  function galleryAdd(g) {
    const key = g.path || g.src;
    if (!key) return;
    const i = gallery.findIndex((x) => (x.path || x.src) === key);
    if (i !== -1) gallery.splice(i, 1);
    gallery.unshift({ path: g.path, src: g.src, title: g.title });
    if (gallery.length > 60) gallery.pop();
  }
  const gallerySrc = (g) => (g.path ? previewUrl(g.path) : g.src);
  const galleryLabel = (g) => g.title || (g.path ? baseName(g.path) : "image");

  let compareOpen = false;
  let compareSel = new Set(); // gallery entries selected for comparison

  function galleryMove(dir) {
    const thumbs = [...root.querySelectorAll("#art-body .art-thumb")];
    if (!thumbs.length) return false;
    const i = thumbs.indexOf(document.activeElement);
    const n = i === -1 ? (dir > 0 ? 0 : Math.max(0, thumbs.length - 1)) : Math.min(thumbs.length - 1, Math.max(0, i + dir));
    thumbs[n].focus();
    return true;
  }

  function renderGallery(body, actions) {
    compareOpen = false;
    compareSel.clear();
    const cmpBtn = el("button.icon-btn", { text: "Compare", title: "Compare the two selected images with a slider", disabled: true, onclick: () => renderCompare([...compareSel]) });
    const refreshCmp = () => { cmpBtn.disabled = compareSel.size !== 2; };
    const strip = el("div.art-gallery", { role: "listbox", "aria-label": "Images seen in this session" });
    for (const g of gallery) {
      const cell = el("button.art-thumb", { title: galleryLabel(g), "aria-pressed": "false" },
        el("img", { src: gallerySrc(g), alt: galleryLabel(g), loading: "lazy" }));
      cell.addEventListener("click", () => {
        if (compareSel.has(g)) compareSel.delete(g);
        else { if (compareSel.size >= 2) compareSel.delete([...compareSel][0]); compareSel.add(g); }
        cell.classList.toggle("sel", compareSel.has(g));
        cell.setAttribute("aria-pressed", String(compareSel.has(g)));
        refreshCmp();
      });
      cell.addEventListener("dblclick", () => openGalleryItem(g));
      strip.append(cell);
    }
    if (!gallery.length) strip.append(el("p.muted.art-files-empty", { text: "No images yet. Images you preview or that appear as artifacts are collected here." }));
    body.append(el("div.art-gallery-wrap", null,
      el("p.muted.art-gallery-hint", { text: "Click to select · double-click to open · select two thumbnails, then Compare · ←/→ move" }),
      strip));
    actions.append(cmpBtn);
  }

  function openGalleryItem(g) {
    if (g.path) OMP.preview.open(g.path);
    else OMP.artifacts.open({ kind: "image", src: g.src, title: g.title || "image" });
  }

  // Before/after slider for two gallery images. The range input keeps the
  // whole control keyboard-accessible (native arrow-key handling).
  function renderCompare(items) {
    if (items.length !== 2) return;
    const [a, b] = items;
    compareOpen = true;
    const body = root.querySelector("#art-body");
    const actions = root.querySelector("#art-actions");
    actions.replaceChildren(el("button.icon-btn", { text: "← Gallery", title: "Back to the gallery (Esc)", onclick: backToGallery }));
    const handle = el("div.art-cmp-handle");
    const topImg = el("img.art-cmp-img", { src: gallerySrc(b), alt: galleryLabel(b) });
    const slider = el("input.art-cmp-range", { type: "range", min: "0", max: "100", value: "50",
      "aria-label": `Compare: left ${galleryLabel(a)}, right ${galleryLabel(b)}` });
    const setPos = () => {
      const v = Number(slider.value);
      topImg.style.clipPath = `inset(0 0 0 ${v}%)`;
      handle.style.left = `calc(${v}% - 1px)`;
      slider.setAttribute("aria-valuetext", v + "%");
    };
    slider.addEventListener("input", setPos);
    body.replaceChildren(el("div.art-compare", null,
      el("img.art-cmp-img", { src: gallerySrc(a), alt: galleryLabel(a) }),
      topImg, handle,
      el("span.art-cmp-tag.a", { text: galleryLabel(a) }),
      el("span.art-cmp-tag.b", { text: galleryLabel(b) }),
      slider));
    setPos();
  }

  function backToGallery() {
    compareOpen = false;
    if (current === GALLERY) show();
  }

  // Shared frame/time indicator text: m:ss.mmm · ~frame n/total (30fps est.)
  const fmtT = (t) => {
    t = Math.max(0, t || 0);
    return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}.${String(Math.floor((t % 1) * 1000)).padStart(3, "0")}`;
  };

  // Frame-by-frame controls for <video>. Stepping uses
  // requestVideoFrameCallback to confirm the frame is actually shown when
  // available, with a `seeked` listener + timeout as the safe fallback.
  function videoControls(media, L) {
    const STEP = 1 / 30;
    const ind = el("span.art-vid-ind", { text: "–:––.––·" });
    const update = () => {
      if (!media.duration) { ind.textContent = "–:––.––·"; return; }
      const f = Math.floor(media.currentTime / STEP) + 1;
      ind.textContent = `${fmtT(media.currentTime)} · ~frame ${f}/${Math.max(1, Math.floor(media.duration / STEP))}`;
    };
    const onTime = () => update();
    media.addEventListener("timeupdate", onTime);
    media.addEventListener("loadedmetadata", onTime);
    L.cleanups.push(() => media.removeEventListener("timeupdate", onTime));
    const step = (d) => {
      if (!media.duration || media.seeking) return;
      media.pause();
      let settled = false;
      const done = () => { if (settled) return; settled = true; update(); };
      if (typeof media.requestVideoFrameCallback === "function") {
        try { media.requestVideoFrameCallback(done); } catch { /* fall back below */ }
      }
      media.addEventListener("seeked", done, { once: true });
      setTimeout(done, 300); // rVFC may never fire while paused/hidden
      media.currentTime = Math.min(media.duration, Math.max(0, media.currentTime + d * STEP));
    };
    L.stepFrame = step;
    L.videoEl = media;
    return el("div.art-vid-bar", { role: "group", "aria-label": "Video frame controls" },
      el("button.icon-btn", { text: "⏮", title: "Previous frame (,)", "aria-label": "Previous frame", onclick: () => step(-1) }),
      el("button.icon-btn", { text: "⏭", title: "Next frame (.)", "aria-label": "Next frame", onclick: () => step(1) }),
      ind,
      el("button.icon-btn", { text: "⏯", title: "Play / pause", "aria-label": "Play or pause", onclick: () => (media.paused ? media.play().catch(() => {}) : media.pause()) }));
  }

  function renderMarkdown(body, actions, a) {
    actions.append(copyBtn(a.content ?? ""), openTabBtnForHtml(a));
    const art = el("div.art-md", { html: OMP.md(a.content ?? "") });
    body.append(art);
  }

  // Resolve an image artifact's src to {data, mimeType} so it can be handed
  // to the composer: inline data URLs as-is, same-origin preview URLs via
  // fetch. Anything else is unsupported and reported, not ignored.
  async function imageSrcToData(src) {
    const m = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(String(src || ""));
    if (m) {
      if (!m[2]) throw new Error("Non-base64 (encoded) data URLs are not supported.");
      return { data: m[3], mimeType: m[1] || "image/png" };
    }
    let url;
    try { url = new URL(src, location.href); } catch { throw new Error("Unusable image address."); }
    if (url.origin !== location.origin || !/^https?:$/.test(url.protocol)) {
      throw new Error("Only inline images and same-origin previews can be annotated.");
    }
    const r = await fetch(url.href);
    if (!r.ok) throw new Error(`Preview fetch failed (${r.status}).`);
    const blob = await r.blob();
    const dataUrl = await new Promise((ok, bad) => {
      const reader = new FileReader();
      reader.onload = () => ok(reader.result);
      reader.onerror = () => bad(reader.error || new Error("Read failed."));
      reader.readAsDataURL(blob);
    });
    return { data: dataUrl.slice(dataUrl.indexOf(",") + 1), mimeType: blob.type || r.headers.get("content-type") || "image/png" };
  }

  // "Annotate" action for image artifacts: bring the image into the composer
  // and open the existing numbered pin/comment editor on it.
  function annotateBtn(a) {
    const btn = el("button.icon-btn", { text: "Annotate", title: "Pin numbered comments on this image and attach it to your message" });
    btn.addEventListener("click", async () => {
      const composer = OMP.app && OMP.app.composer;
      if (!composer) return OMP.toast("Open a session to annotate", "error");
      btn.disabled = true;
      try {
        const { data, mimeType } = await imageSrcToData(a.src);
        composer.addImageData({ data, mimeType, annotate: true, caption: a.title || "" });
      } catch (e) {
        OMP.toast("Annotate: " + (e.message || e), "error", 8000);
      } finally {
        btn.disabled = false;
      }
    });
    return btn;
  }

  function renderImage(body, actions, a) {
    const toggle = el("button.icon-btn", { text: "1:1", title: "Toggle fit / actual size", onclick: () => {
      imageFit = imageFit === "fit" ? "actual" : "fit";
      toggle.textContent = imageFit === "fit" ? "1:1" : "Fit";
      draw();
    } });
    actions.append(toggle, annotateBtn(a), openTabBtn(a),
      el("button.icon-btn", { text: "🖼", title: "Image gallery (images seen this session)", onclick: () => { current = GALLERY; show(); } }));
    galleryAdd({ src: a.src, title: a.title });
    const draw = () => {
      body.replaceChildren();
      const img = el("img.art-img", { src: a.src, alt: a.title || "artifact image" });
      if (imageFit === "actual") img.classList.add("actual");
      body.append(img);
    };
    draw();
  }

  // ── files on the agent host ────────────────────────────────────────────
  // `live` is the open file view: polled for changes (live preview, keeps
  // scroll / playback time / 3D camera) and asked for a still for pins.
  let live = null;
  function stopLive() {
    if (!live) return;
    clearInterval(live.timer);
    window.removeEventListener("message", live.onMessage);
    for (const fn of live.cleanups || []) { try { fn(); } catch { /* listener already gone */ } }
    if (live.viewer) live.viewer.dispose();
    live = null;
  }

  // ── files of this session ─────────────────────────────────────────────
  // Every previewable path the session mentioned (reply links, tool args),
  // newest first; the panel's "Files" view lists them with a path box.
  const seen = new Map(); // path → last-seen order
  const exists = new Map(); // path → Promise<bool>
  function fileExists(path) {
    if (!exists.has(path)) {
      exists.set(path, OMP.api("/api/preview/info?path=" + encodeURIComponent(path)).then(() => true, () => false));
    }
    return exists.get(path);
  }
  let seenSeq = 0;
  const FILES = { kind: "files", title: "Files in this session" };
  function note(path) {
    if (!path || !fileKind(path)) return;
    seen.delete(path);
    seen.set(path, ++seenSeq);
    if (fileKind(path) === "image") galleryAdd({ path, title: baseName(path) });
    if (current === FILES && root && root.classList.contains("open")) renderBody(FILES);
  }
  function noteArgs(args) {
    if (!args || typeof args !== "object") return;
    for (const v of Object.values(args)) {
      if (typeof v !== "string") continue;
      const text = v.length > 4000 ? v.slice(0, 4000) : v;
      PATH_RE.lastIndex = 0;
      let m;
      while ((m = PATH_RE.exec(text))) {
        const prev = text[m.index - 1], p = m[1];
        // Skip format templates (`bust_{n}.png`, `${dir}/x.html`) and
        // tails of joined paths (`f"{B}/HANDOFF.md"`).
        if (prev && /[\w:./\-{}$)]/.test(prev)) continue;
        if (/[{}$%]/.test(p) || /^\/[^/]+$/.test(p)) continue;
        note(p);
      }
    }
  }

  function renderFiles(body) {
    const input = el("input.art-files-input", { placeholder: "Path to a file: /home/…, ~/…, C:\\…", spellcheck: false });
    const go = () => { const p = input.value.trim().replace(/^["']|["']$/g, ""); if (p) OMP.preview.open(p); };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    const list = el("div.art-files-list");
    const paths = [...seen.keys()].reverse();
    for (const p of paths) {
      const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
      const row = el("button.art-file-row", { title: p, onclick: () => OMP.preview.open(p) },
        el("span.art-file-kind", { text: fileKind(p) }),
        el("span.art-file-name", { text: p.slice(i + 1) }),
        el("span.art-file-dir", null, el("bdi", { text: p.slice(0, i + 1) })));
      list.append(row);
      // Temp files get deleted; list only what is still there.
      fileExists(p).then((ok) => { if (!ok) row.remove(); });
    }
    if (!paths.length) list.append(el("p.muted.art-files-empty", { text: "No files mentioned in this session yet. Paste a path above, or click a file path in a reply." }));
    body.append(el("div.art-files", null, el("div.art-files-open", null, input, el("button.chip", { text: "Open", onclick: go })), list));
    setTimeout(() => input.focus(), 0);
  }

  function renderFile(body, actions, a) {
    stopLive();
    const kind = fileKind(a.path);
    const L = (live = { a, kind, mtime: 0, viewer: null, scroll: null, timer: 0, busy: false, on: true, cleanups: [] });
    const url = (p, extra = "") => previewUrl(p) + "?v=" + Date.now() + extra;
    const info = () => OMP.api("/api/preview/info?path=" + encodeURIComponent(a.path));
    // PDF/office page tracking: #page=N keeps the browser PDF viewer on page.
    let page = 1;
    const pageHash = () => (page > 1 ? "#page=" + page : "");
    const pdfBar = () => {
      const ind = el("span.art-vid-ind", { text: `page ${page}` });
      const go = (n) => {
        page = Math.max(1, n);
        ind.textContent = `page ${page}`;
        const f = body.querySelector("iframe");
        if (f && f.src) { try { f.src = f.src.split("#")[0] + "#page=" + page; } catch { /* cross-origin: ignore */ } }
      };
      return el("div.art-vid-bar", { role: "group", "aria-label": "Document page controls" },
        el("button.icon-btn", { text: "◀ page", title: "Previous page", "aria-label": "Previous page", onclick: () => go(page - 1) }),
        ind,
        el("button.icon-btn", { text: "page ▶", title: "Next page", "aria-label": "Next page", onclick: () => go(page + 1) }));
    };
    // Embedded HTML reports its scroll position; restore it after a reload.
    L.onMessage = (e) => {
      const frame = body.querySelector("iframe");
      if (!frame || e.source !== frame.contentWindow || !e.data) return;
      if (e.data.tauScroll) L.scroll = e.data.tauScroll;
      if (e.data.tauLoaded && L.scroll) frame.contentWindow.postMessage({ tauScrollTo: L.scroll }, "*");
    };
    window.addEventListener("message", L.onMessage);

    async function draw(reload) {
      if (live !== L) return;
      let i;
      try { i = await info(); } catch (e) { body.replaceChildren(el("p.art-error", { text: e.message })); return; }
      a.hostPath = i.path;
      L.mtime = i.mtimeMs;
      const cur = body.firstElementChild;
      if (kind === "model") {
        if (reload && L.viewer) return L.viewer.reload(url(i.path)).catch((e) => OMP.toast("3D: " + e.message, "error"));
        body.replaceChildren(el("p.muted.art-loading", { text: "Loading 3D…" }));
        try {
          const V = await import("/assets/viewer3d.js");
          const host = el("div.art-3d");
          body.replaceChildren(host);
          const viewer = await V.mount(host, { url: url(i.path), format: extOf(i.path) === "bbmodel" ? "bbmodel" : "gltf" });
          if (live === L) L.viewer = viewer; else viewer.dispose();
        } catch (e) {
          if (live === L) body.replaceChildren(el("p.art-error", { text: "3D preview: " + e.message }));
        }
        return;
      }
      if (kind === "html" || kind === "pdf") {
        const src = url(i.path, kind === "html" ? "&embed=1" : "") + (kind === "pdf" ? pageHash() : "");
        if (reload && cur && cur.tagName === "IFRAME") { cur.src = src; return; }
        const frame = el("iframe.art-frame", { src, title: a.title || "preview" });
        if (kind === "html") frame.setAttribute("sandbox", "allow-scripts allow-forms allow-popups allow-modals allow-downloads");
        body.replaceChildren(frame);
        if (kind === "pdf") body.append(pdfBar());
      } else if (kind === "image") {
        galleryAdd({ path: i.path, title: a.title || baseName(i.path) });
        if (reload && cur && cur.tagName === "IMG") { cur.src = url(i.path); return; }
        body.replaceChildren(el("img.art-img", { src: url(i.path), alt: a.title || "" }));
      } else if (kind === "video" || kind === "audio") {
        const old = body.querySelector("video, audio");
        const keep = old ? { t: old.currentTime, paused: old.paused, rate: old.playbackRate } : null;
        const media = kind === "video"
          ? el("video.art-media", { src: url(i.path), controls: true, preload: "metadata" })
          : el("audio", { src: url(i.path), controls: true, preload: "metadata" });
        if (keep) media.addEventListener("loadedmetadata", () => {
          media.playbackRate = keep.rate || 1;
          media.currentTime = Math.min(keep.t, media.duration || keep.t);
          if (!keep.paused) media.play().catch(() => {});
        }, { once: true });
        // Targeted refresh: only the media element is replaced; position and
        // playback state survive, and the panel (tabs, gallery) is untouched.
        if (kind === "video") body.replaceChildren(media, videoControls(media, L));
        else body.replaceChildren(el("div.art-audio", null, el("div.art-audio-name", { text: baseName(i.path) }), media));
      } else if (kind === "office") {
        if (!reload) body.replaceChildren(el("p.muted.art-loading", { text: "Converting with Microsoft Office… (the first open of a document takes a few seconds)" }));
        try {
          const r = await OMP.api("/api/preview/office?path=" + encodeURIComponent(i.path), { method: "POST" });
          if (live !== L) return;
          const src = url(r.pdf) + pageHash();
          const frame = body.querySelector("iframe");
          if (frame) frame.src = src;
          else body.replaceChildren(el("iframe.art-frame", { src, title: a.title || "document" }), pdfBar());
        } catch (e) {
          body.replaceChildren(el("p.art-error", { text: e.message }));
        }
      } else {
        const top = body.scrollTop;
        const text = i.size > 4 << 20 ? null : await fetch(url(i.path)).then((r) => r.text()).catch(() => null);
        if (live !== L) return;
        if (text == null) { body.replaceChildren(el("p.muted", { text: "Too large to show here — use Open." })); return; }
        let node;
        if (kind === "markdown") node = el("div.art-md", { html: OMP.md(text) });
        else { node = el("pre.art-code"); node.innerHTML = OMP.highlight(text, extOf(i.path)); }
        body.replaceChildren(node);
        body.scrollTop = top;
      }
    }

    // Live preview: redraw when the file's mtime changes.
    const liveChip = el("button.icon-btn.live-chip.on", { text: "● live", title: "Live preview: reloads when the file changes (click to pause)" });
    liveChip.onclick = () => {
      L.on = !L.on;
      liveChip.classList.toggle("on", L.on);
      liveChip.textContent = L.on ? "● live" : "○ paused";
    };
    L.check = async () => {
      if (live !== L || !L.on || L.busy || document.hidden) return;
      L.busy = true;
      try {
        const i = await info();
        if (i.mtimeMs && L.mtime && i.mtimeMs !== L.mtime) {
          L.mtime = i.mtimeMs;
          await draw(true);
          liveChip.classList.remove("flash"); void liveChip.offsetWidth; liveChip.classList.add("flash");
        }
      } catch { /* file being rewritten: next tick */ } finally { L.busy = false; }
    };
    L.timer = setInterval(L.check, 1000);

    const btn = (text, title, onclick) => el("button.icon-btn", { text, title, onclick });
    const winOpen = (reveal) => OMP.api(`/api/open?path=${encodeURIComponent(a.hostPath || a.path)}${reveal ? "&reveal=1" : ""}`, { method: "POST" })
      .catch((e) => OMP.toast(e.message, "error"));
    actions.append(
      liveChip,
      ...(kind === "image" ? [el("button.icon-btn", { text: "🖼", title: "Image gallery (images seen this session)", onclick: () => { current = GALLERY; show(); } })] : []),
      btn("📍", "Pin: freeze what you see here, add numbered comments and attach it to your message", () => pinStill(body, L)),
      btn("↻", "Reload", () => draw(true)),
      btn("⤢", "Wide panel / back", () => root.classList.toggle("wide")),
      btn("↗", "Open in a new window", () => window.open(previewUrl(a.hostPath || a.path), "_blank")),
      btn("Open", "Open with the Windows default app", () => winOpen(false)),
      btn("📁", "Show in Explorer", () => winOpen(true)),
    );
    draw(false);
  }

  // Still image of the current view → pin editor → composer attachment.
  async function pinStill(body, L) {
    const path = L.a.hostPath || L.a.path;
    const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
    let dataUrl = null, note = "";
    try {
      const media = body.querySelector("video");
      const img = body.querySelector("img.art-img");
      if (L.kind === "model" && L.viewer) {
        dataUrl = L.viewer.snapshot();
        if (L.viewer.clip) note = `${L.viewer.clip.replace(/^animation\.[^.]+\./, "")} at ${L.viewer.time.toFixed(2)}s`;
      } else if (media && media.videoWidth) {
        media.pause();
        const c = document.createElement("canvas");
        c.width = media.videoWidth; c.height = media.videoHeight;
        c.getContext("2d").drawImage(media, 0, 0);
        dataUrl = c.toDataURL("image/png");
        note = `frame at ${fmt(media.currentTime)}`;
      } else if (img && img.naturalWidth) {
        const c = document.createElement("canvas");
        c.width = img.naturalWidth; c.height = img.naturalHeight;
        c.getContext("2d").drawImage(img, 0, 0);
        dataUrl = c.toDataURL("image/png");
      } else {
        // Pages, PDFs, documents: the pixels on screen.
        dataUrl = await OMP.desktop.captureRect(body.getBoundingClientRect());
        if (L.kind === "html" && L.scroll) note = `scrolled to y=${Math.round(L.scroll[1])}px`;
      }
    } catch (e) {
      OMP.toast("Pin: " + (e.message || e), "error", 8000);
      return;
    }
    if (!dataUrl) return;
    const composer = OMP.app && OMP.app.composer;
    if (!composer) return OMP.toast("Open a session to send pins", "error");
    composer.addImageData({
      data: dataUrl.slice(dataUrl.indexOf(",") + 1),
      mimeType: "image/png",
      annotate: true,
      caption: `preview of ${path}${note ? ` (${note})` : ""}`,
    });
  }

  function copyBtn(text) {
    return el("button.icon-btn", { text: "copy", title: "Copy content", onclick: async (e) => {
      if (await OMP.copyText(text)) { e.target.textContent = "copied"; setTimeout(() => (e.target.textContent = "copy"), 1200); }
    } });
  }

  function openTabBtn(a) {
    return el("button.icon-btn", { text: "↗", title: "Open in new tab", onclick: () => {
      const w = window.open("", "_blank");
      if (!w) return;
      const title = esc(a.title || "artifact");
      if (a.kind === "image") {
        w.document.write(`<!doctype html><title>${title}</title><body style="margin:0;display:grid;place-items:center;min-height:100vh"><img src="${esc(a.src)}" alt=""></body>`);
      } else {
        const doc = String(a.content ?? "");
        w.document.write(doc.trim().startsWith("<!DOCTYPE") || doc.includes("<html")
          ? doc : `<!doctype html><title>${title}</title>` + doc);
      }
      w.document.close();
    } });
  }

  function openTabBtnForHtml(a) {
    // markdown/html artifacts whose content looks like a full document get an open-in-tab too
    const looksHtml = /<\s*(!doctype|html|svg)/i.test(a.content || "");
    return looksHtml ? openTabBtn({ ...a, kind: "html" }) : null;
  }

  // ── tabs (history) ─────────────────────────────────────────────────────
  function renderTabs() {
    const tabs = root.querySelector("#art-tabs");
    tabs.replaceChildren();
    tabs.append(el("button.art-tab.art-tab-gallery" + (current === GALLERY ? ".active" : ""), {
      title: "Images seen in this session",
      onclick: () => { current = GALLERY; show(); },
    }, el("span.art-tab-kind", { text: "gal" }), el("span.art-tab-title", { text: "gallery" })));
    tabs.append(el("button.art-tab.art-tab-files" + (current === FILES ? ".active" : ""), {
      title: "All previewable files this session mentioned",
      onclick: () => { current = FILES; show(); },
    }, el("span.art-tab-kind", { text: "files" }), el("span.art-tab-title", { text: "all" })));
    history.forEach((h, i) => {
      const t = el("button.art-tab" + (h === current ? ".active" : ""), {
        title: `${h.kind}: ${h.title || "(untitled)"}`,
        onclick: () => { current = h; show(); },
      }, el("span.art-tab-kind", { text: h.kind }), el("span.art-tab-title", { text: h.title || "(untitled)" }));
      tabs.append(t);
      void i;
    });
  }

  function show() {
    root.classList.add("open");
    appEl.classList.add("has-artifacts");
    root.querySelector("#art-title").textContent = current.title || "(untitled)";
    renderBody(current);
    renderTabs();
  }

  // Inline chat previews use the same authenticated host-file endpoint as the
  // side panel, but do not open or move the panel. Media remains click-to-play;
  // images and document canvases are visible immediately.
  function inlinePreview(path) {
    const kind = fileKind(path);
    if (!["image", "video", "audio", "pdf", "office", "html"].includes(kind)) return null;
    const src = previewUrl(path) + (kind === "html" ? "?embed=1" : "");
    const box = el("div.chat-file-preview", { "data-kind": kind, "data-path": path });
    box.append(el("div.chat-file-preview-name", { text: baseName(path), title: path }));
    if (kind === "image") {
      const img = el("img", { src, alt: baseName(path), loading: "lazy" });
      img.addEventListener("load", () => galleryAdd({ path, src, title: baseName(path) }), { once: true });
      box.append(img);
    } else if (kind === "video") {
      box.append(el("video", { src, controls: true, preload: "metadata" }));
    } else if (kind === "audio") {
      box.append(el("audio", { src, controls: true, preload: "metadata" }));
    } else if (kind === "office") {
      const loading = el("p.muted", { text: "Converting with Microsoft Office…" });
      box.append(loading);
      OMP.api("/api/preview/office?path=" + encodeURIComponent(path), { method: "POST" }).then((r) => {
        const frame = el("iframe", { src: previewUrl(r.pdf), title: baseName(path), loading: "lazy" });
        loading.replaceWith(frame);
      }).catch((e) => { loading.textContent = `Preview unavailable: ${e.message || e}`; });
    } else {
      const frame = el("iframe", { src, title: baseName(path), loading: "lazy" });
      if (kind === "html") frame.setAttribute("sandbox", "allow-scripts allow-forms allow-popups allow-modals allow-downloads");
      box.append(frame);
    }
    return box;
  }

  // ── public API ─────────────────────────────────────────────────────────
  /** Preview a file on the agent host (absolute, ~/ or C:\ path). */
  OMP.preview = {
    kind: fileKind,
    note,
    noteArgs,
    inline: inlinePreview,
    /** Topbar button: close, reopen the last preview, or show the file list. */
    toggle() {
      if (OMP.artifacts.isOpen()) return OMP.artifacts.close();
      return OMP.artifacts.open(current || FILES);
    },
    open(path) {
      const same = history.find((h) => h.kind === "file" && h.path === path);
      return OMP.artifacts.open(same || { kind: "file", path, title: baseName(path) });
    },
  };

  OMP.artifacts = {
    open(opts) {
      const a = opts && OPEN_KINDS.has(opts.kind) ? opts : null;
      if (!a) return false;
      ensureDom();
      if (a !== FILES) {
        const idx = history.indexOf(a);
        if (idx !== -1) history.splice(idx, 1);
        history.unshift(a);
        if (history.length > 20) history.pop();
      }
      current = a;
      show();
      return true;
    },
    close() {
      stopLive();
      if (!root) return;
      root.classList.remove("open");
      appEl.classList.remove("has-artifacts");
    },
    isOpen() { return !!(root && root.classList.contains("open")); },
    clearHistory() {
      stopLive();
      history.length = 0;
      seen.clear();
      exists.clear();
      gallery.length = 0;
      compareSel.clear();
      compareOpen = false;
      current = null;
      if (root) { root.classList.remove("open"); appEl.classList.remove("has-artifacts"); renderTabs(); }
    },
    /** Stable entry point for workspaces/other features: preview a host file. */
    openFile(path) { return OMP.preview.open(path); },
    /** Serializable panel state for workspaces. */
    getState() {
      const state = { open: false, width, gallery: gallery.map((g) => ({ ...g })) };
      if (!root || !current) return state;
      state.open = OMP.artifacts.isOpen();
      if (current === FILES) state.current = { kind: "files" };
      else if (current === GALLERY) state.current = { kind: "gallery" };
      else if (current.kind === "file") state.current = { kind: "file", path: current.path, title: current.title };
      else state.current = { kind: current.kind, title: current.title, content: current.content, lang: current.lang, src: current.src };
      return state;
    },
    /** Restore a state from getState(). Returns true when something was opened. */
    restoreState(s) {
      if (!s || typeof s !== "object") return false;
      for (const g of Array.isArray(s.gallery) ? s.gallery : []) if (g && (g.path || g.src)) galleryAdd(g);
      if (typeof s.width === "number") setWidth(s.width);
      if (!s.open || !s.current) return false;
      const c = s.current;
      if (c.kind === "files") { ensureDom(); current = FILES; show(); return true; }
      if (c.kind === "gallery") { ensureDom(); current = GALLERY; show(); return true; }
      if (c.kind === "file" && c.path) { OMP.preview.open(c.path); return true; }
      if (OPEN_KINDS.has(c.kind)) return OMP.artifacts.open(c);
      return false;
    },
    /** Gallery API: images seen this session (never a directory listing). */
    gallery: {
      list: () => gallery.map((g) => ({ ...g })),
      add: galleryAdd,
      open() { ensureDom(); current = GALLERY; show(); return true; },
    },
  };

  // File paths in rendered messages become preview links (absolute, ~/, or
  // Windows drive paths with a previewable extension).
  const exts = Object.values(EXT).flat().join("|");
  const PATH_RE = new RegExp(`((?:[A-Za-z]:[\\\\/]|~/|/)[^\\s"'<>|*?\`]*?\\.(?:${exts}))(?=$|[\\s"'<>)\\],;:!?.]|\\.\\s)`, "gi");
  // A directory mentioned earlier in the same reply ("Files in `~/x/`:")
  // anchors bare file names in inline code that follow it (`report.html`).
  const DIR_RE = /(?:^|[\s"'(`])((?:[A-Za-z]:[\\/]|~\/|\/)[^\s"'<>|*?`]*[\\/])(?=$|[\s"'<>)\],;:!?`])/g;
  const REL_RE = new RegExp(`^(?![A-Za-z]:|~|/)[\\w.-]+(?:/[\\w.-]+)*\\.(?:${Object.values(EXT).flat().join("|")})$`, "i");
  function linkRelative(node, base, root, shown) {
    const name = node.nodeValue.trim();
    const path = base + name;
    const a = el("a.file-link", { href: "#", title: `Preview ${path}`, "data-path": path, text: node.nodeValue });
    node.replaceWith(a);
    // Only keep the link when the file is really there (cached: streaming
    // re-renders the same reply many times).
    fileExists(path).then((ok) => {
      if (ok) { note(path); if (shown) inlineMedia(a, root, shown); }
      else if (a.isConnected) a.replaceWith(document.createTextNode(a.textContent));
    });
  }
  // Inline media: images / video / audio mentioned by path (or ![](path)) render
  // right under the block that mentions them, like tool-card previews. One
  // preview per path per message, capped so a file list cannot flood the chat.
  const INLINE_KINDS = new Set(["image", "video", "audio"]);
  const MAX_INLINE = 8;
  function topBlock(node, root) {
    let n = node;
    while (n && n.parentNode && n.parentNode !== root) n = n.parentNode;
    return n && n.parentNode === root ? n : null;
  }
  function inlineMedia(link, root, shown) {
    const path = link.dataset.path;
    if (!path || shown.has(path) || shown.size >= MAX_INLINE || !INLINE_KINDS.has(fileKind(path))) return;
    const block = topBlock(link, root);
    if (!block) return;
    shown.add(path);
    const box = inlinePreview(path);
    if (!box) return;
    let strip = block.nextElementSibling;
    if (!strip || !strip.classList || !strip.classList.contains("chat-media-strip")) {
      strip = el("div.chat-media-strip");
      block.after(strip);
    }
    strip.append(box);
  }
  function linkify(root, opts) {
    // opts.media === false: the live streaming tail is re-rendered per token, so no embeds there.
    const embed = !(opts && opts.media === false);
    const shown = new Set(Array.from(root.querySelectorAll ? root.querySelectorAll(".chat-media-strip .chat-file-preview") : []).map((b) => b.dataset.path));
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (n.parentElement.closest("pre, a, button, .file-link") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    let base = null;
    for (const node of nodes) {
      const text = node.nodeValue;
      DIR_RE.lastIndex = 0;
      for (let d; (d = DIR_RE.exec(text));) base = d[1];
      if (base && node.parentElement.tagName === "CODE" && REL_RE.test(text.trim())) {
        linkRelative(node, base, root, embed ? shown : null);
        continue;
      }
      PATH_RE.lastIndex = 0;
      if (!PATH_RE.test(text)) continue;
      PATH_RE.lastIndex = 0;
      const frag = document.createDocumentFragment();
      let last = 0, m;
      while ((m = PATH_RE.exec(text))) {
        const path = m[1];
        // Inside a URL or a word ("http://x/a.html", "foo/x.md") it is not a path.
        const prev = text[m.index - 1];
        if ((prev && /[\w:./-]/.test(prev)) || /^[A-Za-z]:\/\//.test(path)) continue;
        frag.append(text.slice(last, m.index));
        frag.append(el("a.file-link", { href: "#", title: `Preview ${path}`, "data-path": path, text: path }));
        note(path);
        last = m.index + path.length;
      }
      if (!last) continue;
      frag.append(text.slice(last));
      node.replaceWith(frag);
    }
    // Markdown-made links (![alt](/abs/x.png)) and the links created above.
    if (embed) for (const link of root.querySelectorAll("a.file-link")) inlineMedia(link, root, shown);
  }
  OMP.linkifyPaths = linkify;
  document.addEventListener("click", (e) => {
    const shot = e.target.closest && e.target.closest(".chat-media-strip .chat-file-preview img");
    if (shot) { OMP.preview.open(shot.closest(".chat-file-preview").dataset.path); return; }
    const link = e.target.closest && e.target.closest("a.file-link");
    if (!link) return;
    e.preventDefault();
    OMP.preview.open(link.dataset.path);
  });

  // History is per-session: clear when the agent/session changes. A finished
  // tool call may have rewritten the previewed file: check right away.
  window.addEventListener("omp:frame", (e) => {
    const f = e.detail && e.detail.frame;
    if (f && f.type === "tool_execution_end" && live && live.check) setTimeout(live.check, 120);
    if (f && (f.type === "session_loaded" || f.type === "session_reset")) OMP.artifacts.clearHistory();
  });
  window.addEventListener("hashchange", () => OMP.artifacts.clearHistory());
  const btn = document.getElementById("preview-btn");
  if (btn) btn.addEventListener("click", () => OMP.preview.toggle());
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.shiftKey && !e.altKey && e.code === "KeyP") { e.preventDefault(); OMP.preview.toggle(); }
  });
})(window.OMP);
