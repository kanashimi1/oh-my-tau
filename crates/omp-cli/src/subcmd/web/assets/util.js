"use strict";
// Shared helpers. Every script hangs its exports off `window.OMP`.
window.OMP = window.OMP || {};
(function (OMP) {
  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  OMP.esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);

  /** el("div.a.b", {attrs/props/on*}, children...) */
  OMP.el = function el(spec, props, ...children) {
    const [tag, ...classes] = spec.split(".");
    const node = document.createElement(tag || "div");
    if (classes.length) node.className = classes.join(" ");
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v == null || v === false) continue;
        if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
        else if (k === "html") node.innerHTML = v;
        else if (k === "text") node.textContent = v;
        else if (k === "dataset") Object.assign(node.dataset, v);
        else if (k in node && typeof v !== "string") node[k] = v;
        else node.setAttribute(k, v === true ? "" : v);
      }
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return node;
  };
  OMP.$ = (sel, root) => (root || document).querySelector(sel);

  OMP.api = async function api(path, opts = {}) {
    const init = { method: opts.method || "GET", headers: {} };
    if (opts.body !== undefined) {
      init.body = JSON.stringify(opts.body);
      init.headers["content-type"] = "application/json";
    }
    const res = await fetch(path, init);
    const text = await res.text();
    let data;
    try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text }; }
    if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
    return data;
  };

  OMP.fmtTokens = (n) => {
    n = Number(n) || 0;
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k";
    return String(n);
  };
  OMP.fmtCost = (c) => "$" + (Number(c) || 0).toFixed(c >= 10 ? 2 : 3);
  OMP.fmtDuration = (ms) => {
    if (ms == null) return "";
    if (ms < 1000) return ms + "ms";
    const s = ms / 1000;
    if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + "s";
    return Math.floor(s / 60) + "m" + String(Math.round(s % 60)).padStart(2, "0") + "s";
  };
  OMP.timeAgo = (ms) => {
    if (!ms) return "";
    const d = (Date.now() - ms) / 1000;
    if (d < 60) return "just now";
    if (d < 3600) return Math.floor(d / 60) + "m ago";
    if (d < 86400) return Math.floor(d / 3600) + "h ago";
    if (d < 86400 * 7) return Math.floor(d / 86400) + "d ago";
    return new Date(ms).toLocaleDateString();
  };
  OMP.shortPath = (p, home) => {
    if (!p) return "";
    if (home && (p === home || p.startsWith(home + "/"))) return "~" + p.slice(home.length);
    return p;
  };

  OMP.toast = function toast(message, type = "info", ms = 4500) {
    const node = OMP.el("div.toast." + type, { text: message });
    OMP.$("#toasts").append(node);
    setTimeout(() => node.remove(), ms);
    node.addEventListener("click", () => node.remove());
  };

  OMP.copyText = async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      const ta = OMP.el("textarea", { value: text, style: "position:fixed;opacity:0" });
      document.body.append(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    }
  };

  /** Text of a message `content` (string or blocks). */
  OMP.contentText = (content) => {
    if (typeof content === "string") return content;
    if (!Array.isArray(content)) return "";
    return content.filter((b) => b && b.type === "text").map((b) => b.text || "").join("\n");
  };
  OMP.contentImages = (content) =>
    Array.isArray(content) ? content.filter((b) => b && b.type === "image" && b.data) : [];

  /** Run `fn` at most once per animation frame. */
  OMP.rafThrottle = (fn) => {
    let queued = false;
    return (...args) => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; fn(...args); });
    };
  };

  /** Small popup menu at the given anchor element. */
  OMP.popupMenu = function popupMenu(anchor, items) {
    document.querySelectorAll(".menu").forEach((m) => m.remove());
    const menu = OMP.el("div.menu");
    for (const it of items) {
      if (it === "-") { menu.append(OMP.el("hr")); continue; }
      menu.append(OMP.el("button", { text: it.label, onclick: () => { menu.remove(); it.run(); } }));
    }
    document.body.append(menu);
    const r = anchor.getBoundingClientRect();
    const w = menu.offsetWidth;
    menu.style.top = r.bottom + 4 + "px";
    menu.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, r.right - w)) + "px";
    const close = (e) => {
      if (!menu.contains(e.target)) { menu.remove(); document.removeEventListener("mousedown", close, true); }
    };
    setTimeout(() => document.addEventListener("mousedown", close, true));
    return menu;
  };
})(window.OMP);
