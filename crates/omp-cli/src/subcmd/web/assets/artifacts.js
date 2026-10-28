"use strict";
// Right-hand artifact panel (code, diff, HTML, markdown, images). Owner: artifacts slice.
// API: OMP.artifacts.open({kind,title,content,lang,src}) / close() / isOpen().
(function (OMP) {
  const { el, esc } = OMP;
  const LS_WIDTH = "omp-artifacts-width";
  const OPEN_KINDS = new Set(["code", "diff", "html", "markdown", "image", "file"]);

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

    // Esc closes when focus is inside the panel
    root.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.stopPropagation(); OMP.artifacts.close(); }
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
    else if (a.kind === "file") renderFile(body, actions, a);
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

  function renderMarkdown(body, actions, a) {
    actions.append(copyBtn(a.content ?? ""), openTabBtnForHtml(a));
    const art = el("div.art-md", { html: OMP.md(a.content ?? "") });
    body.append(art);
  }

  function renderImage(body, actions, a) {
    const toggle = el("button.icon-btn", { text: "1:1", title: "Toggle fit / actual size", onclick: () => {
      imageFit = imageFit === "fit" ? "actual" : "fit";
      toggle.textContent = imageFit === "fit" ? "1:1" : "Fit";
      draw();
    } });
    actions.append(toggle, openTabBtn(a));
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
    if (live.viewer) live.viewer.dispose();
    live = null;
  }

  function renderFile(body, actions, a) {
    stopLive();
    const kind = fileKind(a.path);
    const L = (live = { a, kind, mtime: 0, viewer: null, scroll: null, timer: 0, busy: false, on: true });
    const url = (p, extra = "") => previewUrl(p) + "?v=" + Date.now() + extra;
    const info = () => OMP.api("/api/preview/info?path=" + encodeURIComponent(a.path));
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
        if (reload && cur && cur.tagName === "IFRAME") { cur.src = url(i.path, kind === "html" ? "&embed=1" : ""); return; }
        const frame = el("iframe.art-frame", { src: url(i.path, kind === "html" ? "&embed=1" : ""), title: a.title || "preview" });
        if (kind === "html") frame.setAttribute("sandbox", "allow-scripts allow-forms allow-popups allow-modals allow-downloads");
        body.replaceChildren(frame);
      } else if (kind === "image") {
        if (reload && cur && cur.tagName === "IMG") { cur.src = url(i.path); return; }
        body.replaceChildren(el("img.art-img", { src: url(i.path), alt: a.title || "" }));
      } else if (kind === "video" || kind === "audio") {
        const old = body.querySelector("video, audio");
        const keep = old ? { t: old.currentTime, paused: old.paused } : null;
        const media = kind === "video"
          ? el("video.art-media", { src: url(i.path), controls: true, preload: "metadata" })
          : el("audio", { src: url(i.path), controls: true, preload: "metadata" });
        if (keep) media.addEventListener("loadedmetadata", () => { media.currentTime = Math.min(keep.t, media.duration || keep.t); if (!keep.paused) media.play().catch(() => {}); }, { once: true });
        body.replaceChildren(kind === "video" ? media : el("div.art-audio", null, el("div.art-audio-name", { text: baseName(i.path) }), media));
      } else if (kind === "office") {
        if (!reload) body.replaceChildren(el("p.muted.art-loading", { text: "Converting with Microsoft Office… (the first open of a document takes a few seconds)" }));
        try {
          const r = await OMP.api("/api/preview/office?path=" + encodeURIComponent(i.path), { method: "POST" });
          if (live !== L) return;
          const frame = body.querySelector("iframe");
          if (frame) frame.src = url(r.pdf);
          else body.replaceChildren(el("iframe.art-frame", { src: url(r.pdf), title: a.title || "document" }));
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

  // ── public API ─────────────────────────────────────────────────────────
  /** Preview a file on the agent host (absolute, ~/ or C:\ path). */
  OMP.preview = {
    kind: fileKind,
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
      const idx = history.indexOf(a);
      if (idx !== -1) history.splice(idx, 1);
      history.unshift(a);
      if (history.length > 20) history.pop();
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
      current = null;
      if (root) { root.classList.remove("open"); appEl.classList.remove("has-artifacts"); renderTabs(); }
    },
  };

  // File paths in rendered messages become preview links (absolute, ~/, or
  // Windows drive paths with a previewable extension).
  const exts = Object.values(EXT).flat().join("|");
  const PATH_RE = new RegExp(`((?:[A-Za-z]:[\\\\/]|~/|/)[^\\s"'<>|*?\`]*?\\.(?:${exts}))(?=$|[\\s"'<>)\\],;:!?.]|\\.\\s)`, "gi");
  function linkify(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (n.parentElement.closest("pre, a, button, .file-link") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const text = node.nodeValue;
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
        last = m.index + path.length;
      }
      if (!last) continue;
      frag.append(text.slice(last));
      node.replaceWith(frag);
    }
  }
  OMP.linkifyPaths = linkify;
  document.addEventListener("click", (e) => {
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
})(window.OMP);
