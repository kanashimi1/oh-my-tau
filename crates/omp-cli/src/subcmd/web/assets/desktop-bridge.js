"use strict";
// Desktop (Tauri) / browser bridge: OMP.desktop + frame-bus notifications and
// screenshot attachment. Owner: desktop slice.
//
// Contracts (BRIEF.md):
//   OMP.desktop = { available, openWindow(hash), notify(title, body) }
//   OMP.app.composer.addImageData({ data, mimeType }) — guarded, may not exist yet
(function (OMP) {
  const TAURI = window.__TAURI__ || null;
  const DIALOG_METHODS = new Set(["select", "confirm", "input", "editor", "ask"]);

  function inTauri() {
    return !!(TAURI && TAURI.core && typeof TAURI.core.invoke === "function");
  }

  const desktop = { available: inTauri() };

  // ── openWindow ──────────────────────────────────────────────────────────
  desktop.openWindow = function (hash) {
    const h = String(hash || "").replace(/^\//, "");
    if (inTauri()) {
      try { return TAURI.core.invoke("open_window", { hash: h }); }
      catch { /* surface later */ }
      return;
    }
    let url = location.origin + "/" + h;
    window.open(url, "_blank");
  };

  // ── notify ──────────────────────────────────────────────────────────────
  function webNotify(title, body) {
    if (document.hidden && "Notification" in window) {
      const show = () => { try { new Notification(title, { body }); } catch { /* noop */ } };
      if (Notification.permission === "granted") show();
      else if (Notification.permission !== "denied") {
        Notification.requestPermission().then((p) => { if (p === "granted") show(); });
      }
    }
  }

  desktop.notify = function (title, body) {
    if (inTauri()) {
      try { return TAURI.core.invoke("notify", { title, body }); }
      catch { return; }
    }
    webNotify(title, body);
  };

  OMP.desktop = desktop;

  // ── focus detection ──────────────────────────────────────────────────────
  function currentFocused() {
    if (inTauri() && TAURI.window && TAURI.window.getCurrentWindow) {
      try {
        const w = TAURI.window.getCurrentWindow();
        if (w && typeof w.isFocused === "function") return Promise.resolve(w.isFocused());
      } catch { /* fall through */ }
    }
    return Promise.resolve(!document.hidden);
  }

  function maybeNotify(title, body) {
    Promise.resolve(currentFocused()).then((focused) => {
      if (!focused) desktop.notify(title, body);
    }).catch(() => desktop.notify(title, body));
  }

  // ── frame bus: notify when an agent finishes / waits for input ───────────
  window.addEventListener("omp:frame", (e) => {
    const d = e.detail || {};
    const f = d.frame || {};
    if (f.type === "agent_end") {
      maybeNotify("Agent finished", d.agentId ? `Agent ${d.agentId} finished.` : "An agent finished.");
    } else if (f.type === "extension_ui_request" && DIALOG_METHODS.has(f.method)) {
      const msg = f.message || f.title || f.text || "An agent is waiting for your input.";
      maybeNotify("Needs your input", msg);
    }
  });

  // ── screenshot hotkey (Ctrl+Shift+S): the Rust side captured the monitor
  // and made this window fullscreen; drag a region, Enter/release to attach,
  // Esc to cancel. ─────────────────────────────────────────────────────────
  function captureDone() {
    try { TAURI.core.invoke("capture_done"); } catch { /* noop */ }
  }

  function regionSelect(b64) {
    const img = new Image();
    img.onload = () => {
      const ov = document.createElement("div");
      ov.className = "cap-overlay";
      const hint = document.createElement("div");
      hint.className = "cap-hint";
      hint.textContent = "Drag to select a region · Esc to cancel · Enter = whole screen";
      const sel = document.createElement("div");
      sel.className = "cap-sel";
      sel.hidden = true;
      ov.append(img, sel, hint);
      img.className = "cap-img";
      document.body.append(ov);
      let start = null, rect = null;
      // object-fit: contain letterboxes the image inside the element box
      const shown = () => {
        const e = img.getBoundingClientRect();
        const k = Math.max(img.naturalWidth / e.width, img.naturalHeight / e.height);
        const w = img.naturalWidth / k, h = img.naturalHeight / k;
        return { k, left: e.left + (e.width - w) / 2, top: e.top + (e.height - h) / 2 };
      };
      const scale = () => shown().k;
      const finish = (r) => {
        const { k, left, top } = shown(), box = { left, top }; // measure before the overlay leaves the DOM
        cleanup();
        const sx = r ? Math.max(0, (r.x - box.left) * k) : 0, sy = r ? Math.max(0, (r.y - box.top) * k) : 0;
        const sw = r ? Math.min(img.naturalWidth - sx, r.w * k) : img.naturalWidth;
        const sh = r ? Math.min(img.naturalHeight - sy, r.h * k) : img.naturalHeight;
        if (sw < 4 || sh < 4) return;
        const c = document.createElement("canvas");
        c.width = Math.round(sw); c.height = Math.round(sh);
        c.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, c.width, c.height);
        const data = c.toDataURL("image/png").split(",")[1];
        OMP.app?.composer?.addImageData({ data, mimeType: "image/png" });
        OMP.app?.composer?.focus?.();
      };
      const onKey = (e) => {
        if (e.key === "Escape") { e.preventDefault(); cleanup(); }
        else if (e.key === "Enter") { e.preventDefault(); finish(rect); }
      };
      const cleanup = () => { ov.remove(); window.removeEventListener("keydown", onKey, true); captureDone(); };
      ov.addEventListener("pointerdown", (e) => {
        start = { x: e.clientX, y: e.clientY };
        ov.setPointerCapture(e.pointerId);
      });
      ov.addEventListener("pointermove", (e) => {
        if (!start) return;
        rect = { x: Math.min(start.x, e.clientX), y: Math.min(start.y, e.clientY), w: Math.abs(e.clientX - start.x), h: Math.abs(e.clientY - start.y) };
        Object.assign(sel.style, { left: rect.x + "px", top: rect.y + "px", width: rect.w + "px", height: rect.h + "px" });
        sel.hidden = false;
      });
      ov.addEventListener("pointerup", () => { if (start && rect && rect.w > 3 && rect.h > 3) finish(rect); start = null; });
      window.addEventListener("keydown", onKey, true);
    };
    img.onerror = captureDone;
    img.src = "data:image/png;base64," + b64;
  }

  if (inTauri() && TAURI.event && typeof TAURI.event.listen === "function") {
    try {
      TAURI.event.listen("omp:screen-capture", (event) => {
        if (typeof event?.payload === "string") regionSelect(event.payload);
      }).catch(() => { /* noop */ });
    } catch { /* noop */ }
  }
  OMP.desktop && (OMP.desktop._regionSelect = regionSelect); // test hook
})(window.OMP);
