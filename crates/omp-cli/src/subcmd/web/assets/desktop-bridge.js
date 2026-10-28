"use strict";
// Desktop (Tauri) / browser bridge: OMP.desktop + frame-bus notifications and
// screenshot attachment. Owner: desktop slice.
//
// Contracts (BRIEF.md):
//   OMP.desktop = { available, openWindow(hash), notify(title, body, agent?, key?),
//                   captureRect(viewportRect) → PNG data URL }
//   OMP.app.composer.addImageData({ data, mimeType }) — guarded, may not exist yet
(function (OMP) {
  const TAURI = window.__TAURI__ || null;

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
  const goTo = (agent) => { if (agent) location.hash = "#/a/" + agent; };

  function webNotify(title, body, agent, key) {
    if (!("Notification" in window) || (!document.hidden && document.hasFocus())) return false;
    const show = () => {
      try {
        const n = new Notification(title, { body, tag: key || undefined, icon: "/favicon.svg" });
        n.onclick = () => { window.focus(); goTo(agent); n.close(); };
      } catch { /* noop */ }
    };
    if (Notification.permission === "granted") show();
    else if (Notification.permission !== "denied") Notification.requestPermission().then((p) => { if (p === "granted") show(); });
    return true;
  }

  /** System notification; `agent` = session to open on click. Resolves to
   *  whether one was shown (not shown while the app window has focus). */
  desktop.notify = async function (title, body, agent, key) {
    if (inTauri()) {
      try { return !!(await TAURI.core.invoke("notify", { title, body, agent: agent || null, key: key || null })); }
      catch { return false; }
    }
    return webNotify(title, body, agent, key);
  };

  // ── desktop settings (Tau only) ───────────────────────────────────────
  /** {captureShortcut, captureShortcutError, devtoolsPort, devtoolsActive} or null outside Tau. */
  desktop.settings = async function () {
    return inTauri() ? TAURI.core.invoke("tau_settings") : null;
  };
  /** Re-register the capture shortcut ("Ctrl+Alt+S"); rejects when taken. */
  desktop.setCaptureShortcut = function (shortcut) {
    return TAURI.core.invoke("set_capture_shortcut", { shortcut });
  };
  /** DevTools port (null = off); takes effect after restarting Tau. */
  desktop.setDevtoolsPort = function (port) {
    return TAURI.core.invoke("set_devtools_port", { port });
  };

  // ── capture (preview pins) ─────────────────────────────────────────────
  /** Screen pixels of a viewport rect as a PNG data URL: native capture in
   *  the desktop app, tab capture (asks once per pin) in a browser. */
  desktop.captureRect = async function (rect) {
    const r = { x: Math.max(0, rect.left), y: Math.max(0, rect.top), w: rect.width, h: rect.height };
    if (inTauri()) return TAURI.core.invoke("capture_rect", r);
    if (!navigator.mediaDevices?.getDisplayMedia) throw new Error("screen capture is not available in this browser");
    const stream = await navigator.mediaDevices.getDisplayMedia({ video: { displaySurface: "browser" }, preferCurrentTab: true, audio: false });
    try {
      const video = document.createElement("video");
      video.srcObject = stream;
      video.muted = true;
      await video.play();
      await new Promise((ok) => requestAnimationFrame(() => requestAnimationFrame(ok)));
      const kx = video.videoWidth / window.innerWidth, ky = video.videoHeight / window.innerHeight;
      const c = document.createElement("canvas");
      c.width = Math.round(r.w * kx); c.height = Math.round(r.h * ky);
      c.getContext("2d").drawImage(video, r.x * kx, r.y * ky, r.w * kx, r.h * ky, 0, 0, c.width, c.height);
      return c.toDataURL("image/png");
    } finally {
      stream.getTracks().forEach((t) => t.stop());
    }
  };

  OMP.desktop = desktop;

  // ── agent notifications ────────────────────────────────────────────────
  // From roster transitions (`omp:agents`, every session, not just the open
  // one): busy → idle = finished, needsInput rising = waiting for an answer.
  // While the window has focus a background session gets an in-app toast.
  const roster = new Map();
  let primed = false;
  const nameOf = (a) => a.sessionName || String(a.cwd || "").split("/").filter(Boolean).pop() || a.id;
  function announce(a, kind) {
    const current = OMP.app && OMP.app.S && OMP.app.S.agentId;
    const title = kind === "wait" ? "Tau: waiting for your answer" : "Tau: agent finished";
    const body = `${nameOf(a)}${a.model ? " · " + a.model.split("/").pop() : ""}`;
    const key = `${a.id}:${kind}:${Math.floor(Date.now() / 30000)}`;
    desktop.notify(title, body, a.id, key).then((shown) => {
      if (!shown && a.id !== current && document.hasFocus()) {
        OMP.toast(`${kind === "wait" ? "Waiting for you" : "Finished"}: ${nameOf(a)}`, "info", 5000);
      }
    });
  }
  window.addEventListener("omp:agents", (e) => {
    const agents = (e.detail && e.detail.agents) || [];
    for (const a of agents) {
      const busy = !!(a.streaming || a.runningTools);
      const prev = roster.get(a.id);
      if (primed && prev) {
        if (a.needsInput && !prev.needsInput) announce(a, "wait");
        else if (prev.busy && !busy && !a.needsInput && a.alive) announce(a, "done");
      }
      roster.set(a.id, { busy, needsInput: !!a.needsInput });
    }
    primed = true;
  });
  // The open session flips to busy immediately (runs shorter than a roster poll).
  window.addEventListener("omp:frame", (e) => {
    const d = e.detail || {};
    if (d.frame && d.frame.type === "agent_start" && d.agentId && roster.has(d.agentId)) roster.get(d.agentId).busy = true;
  });
  if (inTauri() && TAURI.event && typeof TAURI.event.listen === "function") {
    try { TAURI.event.listen("omp:navigate", (ev) => goTo(ev && ev.payload)).catch(() => {}); } catch { /* noop */ }
  }

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
        OMP.app?.composer?.addImageData({ data, mimeType: "image/png", annotate: true });
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
  // The desktop shell has no browser reload keys; F5 / Ctrl+R reload the
  // page so a restarted omt service's new UI shows up without restarting Tau.
  if (inTauri()) {
    window.addEventListener("keydown", (e) => {
      const ctrlR = (e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.code === "KeyR";
      if (document.querySelector(".ap-key.rec")) return; // recording a shortcut
      if (e.key === "F5" || ctrlR) { e.preventDefault(); location.reload(); }
    }, true);
  }
  OMP.desktop && (OMP.desktop._regionSelect = regionSelect); // test hook
})(window.OMP);
