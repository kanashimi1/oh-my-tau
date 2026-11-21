"use strict";
// Tau brand mark plus distinct animated thinking/tool-call glyphs.
// Public API is shared by the status bar, roster and transcript.
(function (OMP) {
  const FRAMES = Object.fromEntries(["idle", "think", "work", "wave", "error"].map((s) => [s, s]));
  const W = 64, H = 64;
  const PATH = "M22 6H54Q56 6 56 8V15Q56 17 54 17H38L35 42C34.2 49 36 52 40.5 52C44 52 46.3 50.2 48.2 46.8L53.5 49.3C50.8 56 45.9 59.5 38.8 59.5C29.4 59.5 24.1 54.3 25.6 44.1L28.8 17H22.5C18.7 17 16.3 19.6 15.3 24.7L8.7 23.4C10 13.5 13.8 6 22 6Z";
  const BRAIN = "M32 14C27 9 20 12 19 18C13 17 8 23 11 29C6 32 7 39 12 42C10 48 15 53 20 51C22 57 28 58 32 54C36 59 43 57 45 51C51 53 56 48 53 42C59 39 59 32 54 29C57 23 52 17 46 18C44 11 37 9 32 14Z";
  const PALETTES = Object.freeze({
    volt: { name: "Volt", stops: ["#f1ff9b", "#8aefa7", "#16b68e"], glow: "#77e6a2" },
    ember: { name: "Ember", stops: ["#fff1ae", "#ffb13b", "#ff5c49"], glow: "#ff9a51" },
    tide: { name: "Tide", stops: ["#dcfff2", "#62e6c8", "#168fa0"], glow: "#5bdcc5" },
    titanium: { name: "Titanium", stops: ["#ffffff", "#b8c4cc", "#627381"], glow: "#b5c6d2" },
    copper: { name: "Copper", stops: ["#ffe2ae", "#d99055", "#8f4934"], glow: "#d89561" },
    infrared: { name: "Infrared", stops: ["#ffd36e", "#ff7048", "#d92545"], glow: "#ff604f" },
  });
  const ERROR = { name: "Error", stops: ["#ffe0b8", "#ff8066", "#d92f4d"], glow: "#ff5b62" };
  let sequence = 0;
  let current = "volt";
  try {
    const saved = localStorage.getItem("omp-tau-palette");
    if (PALETTES[saved]) current = saved;
  } catch { /* storage can be unavailable in embedded previews */ }

  function defs(id, palette, shape = PATH) {
    const [a, b, c] = palette.stops;
    return `<defs>
      <linearGradient id="${id}-paint" x1="12" y1="7" x2="52" y2="58" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="${a}"/><stop offset=".48" stop-color="${b}"/><stop offset="1" stop-color="${c}"/>
      </linearGradient>
      <linearGradient id="${id}-sheen" x1="0" y1="0" x2="1" y2="0">
        <stop stop-color="#fff" stop-opacity="0"/><stop offset=".48" stop-color="#fff" stop-opacity=".7"/><stop offset="1" stop-color="#fff" stop-opacity="0"/>
      </linearGradient>
      <clipPath id="${id}-clip"><path d="${shape}"/></clipPath>
      <filter id="${id}-glow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="2.2"/></filter>
    </defs>`;
  }

  function tauMarkup(state, id) {
    const palette = state === "error" ? ERROR : PALETTES[current];
    const [a] = palette.stops;
    return defs(id, palette) + `
      <path class="tau-aura" d="${PATH}" fill="${palette.glow}" opacity=".34" filter="url(#${id}-glow)"/>
      <g class="tau-body">
        <path d="${PATH}" transform="translate(0 1.1)" fill="#05090c" opacity=".38"/>
        <path class="tau-mark" d="${PATH}" fill="url(#${id}-paint)"/>
        <g clip-path="url(#${id}-clip)">
          <g transform="rotate(17 32 32)"><rect class="tau-sheen" x="-35" y="-15" width="19" height="96" fill="url(#${id}-sheen)"/></g>
          <path class="tau-trace" d="${PATH}" fill="none" stroke="#fff" stroke-opacity=".52" stroke-width=".65" stroke-dasharray="7 190"/>
        </g>
        <path d="${PATH}" fill="none" stroke="${a}" stroke-opacity=".32" stroke-width=".35" stroke-linejoin="round"/>
      </g>`;
  }

  function brainMarkup(id) {
    return `<defs><clipPath id="${id}-clip"><path d="${BRAIN}"/></clipPath></defs>
      <g class="brain-motion">
        <path class="brain-outline" d="${BRAIN}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
        <g class="brain-circuit" fill="none" stroke="currentColor" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round">
          <path d="M32 19V47M19 33H26L32 27M45 33H38L32 39"/>
        </g>
        <g fill="currentColor">
          <circle class="brain-node" style="--node:0" cx="32" cy="19" r="1.4"/>
          <circle class="brain-node" style="--node:1" cx="19" cy="33" r="1.4"/>
          <circle class="brain-node" style="--node:2" cx="45" cy="33" r="1.4"/>
          <circle class="brain-node" style="--node:3" cx="32" cy="47" r="1.4"/>
        </g>
        <g clip-path="url(#${id}-clip)"><path class="brain-scan" d="M13 31H51" fill="none" stroke="currentColor" stroke-width="1"/></g>
      </g>`;
  }

  function toolMarkup() {
    return `<g class="tool-terminal" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path class="terminal-chevron" d="M15 24L26 33L15 42"/>
      <path class="terminal-command" d="M32 34H47" pathLength="1"/>
      <path class="terminal-caret" d="M31 43H47"/>
    </g>`;
  }

  function markup(state, id) {
    if (state === "think") return brainMarkup(id);
    if (state === "work") return toolMarkup();
    return tauMarkup(state, id);
  }

  function palette(id) { return PALETTES[id] || PALETTES[current]; }

  function svg(state = "idle", { size = 32, title = "Tau" } = {}) {
    state = FRAMES[state] ? state : "idle";
    const el = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    el.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el.setAttribute("width", String(size)); el.setAttribute("height", String(size));
    el.setAttribute("role", "img"); el.setAttribute("aria-label", title);
    el.classList.add("tau-sprite", "tau-" + state);
    if (state === "think") el.classList.add("activity-brain");
    if (state === "work") el.classList.add("activity-tool");
    el.dataset.palette = state === "error" ? "error" : current;
    el.style.setProperty("--tau-glow", palette(state === "error" ? "error" : current).glow);
    el.innerHTML = `<title>${title}</title>` + markup(state, `tau-mark-${++sequence}`);
    return el;
  }

  const TITLES = { idle: "Tau: ready", think: "Tau: thinking", work: "Tau: running tools", wave: "Tau: needs your input", error: "Tau: something went wrong" };

  function mount(host, { size = 32, state = "idle" } = {}) {
    let cur = null;
    const api = {
      set(next) {
        next = FRAMES[next] ? next : "idle";
        if (next === cur) return;
        cur = next;
        host.style.setProperty("--tau-glow", palette(next === "error" ? "error" : current).glow);
        host.replaceChildren(svg(next, { size, title: TITLES[next] || "Tau" }));
        host.dataset.state = next;
      },
      get state() { return cur; },
    };
    api.set(state);
    return api;
  }

  function stateOf(S) {
    if (!S || !S.agentId) return "idle";
    const info = (S.agents || []).find((a) => a.id === S.agentId) || {};
    if (S.dead || (S.errorUntil && Date.now() < S.errorUntil)) return "error";
    if (info.needsInput || (OMP.dialogs && OMP.dialogs.isOpen && OMP.dialogs.isOpen())) return "wave";
    if (S.streaming) return S.toolsRunning > 0 || S.bashRunning > 0 ? "work" : "think";
    return "idle";
  }

  function hydrate(root = document) {
    for (const host of root.querySelectorAll("[data-tau]")) {
      if (host.firstChild) continue;
      mount(host, { size: Number(host.dataset.size) || 32, state: host.dataset.tau || "idle" });
    }
  }

  function setPalette(id, { persist = true } = {}) {
    if (!PALETTES[id]) return false;
    current = id;
    if (persist) { try { localStorage.setItem("omp-tau-palette", id); } catch { /* non-persistent preview */ } }
    document.documentElement.dataset.tauPalette = id;
    for (const host of document.querySelectorAll("[data-tau]")) {
      const sprite = host.querySelector("svg.tau-sprite");
      const size = Number(host.dataset.size) || Number(sprite?.getAttribute("width")) || 32;
      const state = host.dataset.state || host.dataset.tau || "idle";
      host.replaceChildren(svg(state, { size, title: TITLES[state] || "Tau" }));
      host.dataset.state = state;
    }
    return true;
  }

  const svgText = (state = "idle") => {
    state = FRAMES[state] ? state : "idle";
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}">${markup(state, "tau-icon")}</svg>`;
  };

  OMP.mascot = { svg, svgText, mount, stateOf, hydrate, setPalette, palette, PALETTES, PATH, BRAIN, FRAMES, W, H };
  document.documentElement.dataset.tauPalette = current;
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => hydrate());
  else hydrate();
})(window.OMP = window.OMP || {});
