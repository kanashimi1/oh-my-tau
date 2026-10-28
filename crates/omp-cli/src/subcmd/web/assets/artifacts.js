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
  function renderFile(body, actions, a) {
    const kind = fileKind(a.path);
    let mount = null;
    const draw = async () => {
      body.replaceChildren(el("p.muted.art-loading", { text: "Loading…" }));
      let info;
      try {
        info = await OMP.api("/api/preview/info?path=" + encodeURIComponent(a.path));
      } catch (e) {
        body.replaceChildren(el("p.art-error", { text: e.message }));
        return;
      }
      a.hostPath = info.path;
      const url = previewUrl(info.path) + "?v=" + Date.now();
      if (kind === "html" || kind === "pdf") {
        mount = el("iframe.art-frame", { src: url, title: a.title || "preview" });
        if (kind === "html") mount.setAttribute("sandbox", "allow-scripts allow-forms allow-popups allow-modals allow-downloads");
      } else if (kind === "image") {
        mount = el("img.art-img", { src: url, alt: a.title || "" });
      } else if (kind === "video") {
        mount = el("video.art-media", { src: url, controls: true, autoplay: false, preload: "metadata" });
      } else if (kind === "audio") {
        mount = el("div.art-audio", null, el("div.art-audio-name", { text: baseName(info.path) }), el("audio", { src: url, controls: true, preload: "metadata" }));
      } else if (kind === "office") {
        body.replaceChildren(el("p.muted.art-loading", { text: "Converting with Microsoft Office… (first open of a document takes a few seconds)" }));
        try {
          const r = await OMP.api("/api/preview/office?path=" + encodeURIComponent(info.path), { method: "POST" });
          mount = el("iframe.art-frame", { src: previewUrl(r.pdf), title: a.title || "document" });
        } catch (e) {
          body.replaceChildren(el("p.art-error", { text: e.message }));
          return;
        }
      } else {
        const text = info.size > 4 << 20 ? null : await fetch(url).then((r) => r.text()).catch(() => null);
        if (text == null) { body.replaceChildren(el("p.muted", { text: "Too large to show here — use Open." })); return; }
        if (kind === "markdown") mount = el("div.art-md", { html: OMP.md(text) });
        else {
          const pre = el("pre.art-code");
          pre.innerHTML = OMP.highlight(text, extOf(info.path));
          mount = pre;
        }
      }
      body.replaceChildren(mount);
    };
    const btn = (text, title, onclick) => el("button.icon-btn", { text, title, onclick });
    const winOpen = (reveal) => OMP.api(`/api/open?path=${encodeURIComponent(a.hostPath || a.path)}${reveal ? "&reveal=1" : ""}`, { method: "POST" })
      .catch((e) => OMP.toast(e.message, "error"));
    actions.append(
      btn("↻", "Reload (after the file changed)", () => draw()),
      btn("⤢", "Wide panel / back", () => root.classList.toggle("wide")),
      btn("↗", "Open in a new window", () => window.open(previewUrl(a.hostPath || a.path), "_blank")),
      btn("Open", "Open with the Windows default app", () => winOpen(false)),
      btn("📁", "Show in Explorer", () => winOpen(true)),
    );
    draw();
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
      if (!root) return;
      root.classList.remove("open");
      appEl.classList.remove("has-artifacts");
    },
    isOpen() { return !!(root && root.classList.contains("open")); },
    clearHistory() {
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

  // History is per-session: clear when the agent/session changes.
  window.addEventListener("omp:frame", (e) => {
    const f = e.detail && e.detail.frame;
    if (f && (f.type === "session_loaded" || f.type === "session_reset")) OMP.artifacts.clearHistory();
  });
  window.addEventListener("hashchange", () => OMP.artifacts.clearHistory());
})(window.OMP);
