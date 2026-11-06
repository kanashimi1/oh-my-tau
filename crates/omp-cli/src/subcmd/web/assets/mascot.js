"use strict";
// Tau — the rounded τ from the brand mark, with light moving through the glyph.
// The public API and agent states are shared with the rest of the web UI.
//
//   OMP.mascot.svg(state, { size, title })  → <svg> element
//   OMP.mascot.mount(host)                   → live sprite; .set(state)
//   OMP.mascot.stateOf(S)                    → state for an app state
//   OMP.mascot.setPalette(id)                → recolor every live Tau
(function (OMP) {
  const FRAMES = Object.fromEntries(["idle", "brain", "think", "work", "wave", "error"].map((s) => [s, s]));
  const W = 64, H = 64;
  const PATH = "M22 6H54Q56 6 56 8V15Q56 17 54 17H38L35 42C34.2 49 36 52 40.5 52C44 52 46.3 50.2 48.2 46.8L53.5 49.3C50.8 56 45.9 59.5 38.8 59.5C29.4 59.5 24.1 54.3 25.6 44.1L28.8 17H22.5C18.7 17 16.3 19.6 15.3 24.7L8.7 23.4C10 13.5 13.8 6 22 6Z";
  // Brain state: the mark redrawn as a rounded brain while a turn is thinking.
  const BRAIN = "M32 14.8C33.7 15,35.2 15.1,36.8 15.3C38.4 15.5,40.1 15.6,41.8 16C43.5 16.4,45.4 16.9,47 17.6C48.6 18.4,50.3 19.3,51.6 20.5C52.9 21.7,54.1 23.2,54.9 24.8C55.7 26.4,56.1 28.3,56.2 30C56.3 31.7,56.1 33.5,55.7 35.1C55.4 36.7,54.8 38.3,54.1 39.8C53.4 41.3,52.6 42.8,51.5 44.1C50.4 45.4,49.1 46.6,47.6 47.5C46.2 48.5,44.5 49.2,42.8 49.8C41.1 50.4,39.2 50.8,37.4 51.1C35.6 51.4,33.9 51.6,32 51.7C30.2 51.8,28.1 51.8,26.3 51.5C24.5 51.2,22.6 50.8,21 50.1C19.4 49.4,17.8 48.4,16.5 47.3C15.2 46.2,14.2 44.9,13.2 43.6C12.2 42.3,11.4 41,10.6 39.6C9.8 38.2,9.1 36.7,8.6 35.1C8.1 33.5,7.7 31.7,7.8 30C7.9 28.3,8.3 26.4,9 24.8C9.7 23.2,10.8 21.7,12 20.4C13.2 19.1,14.7 17.8,16.3 16.9C17.9 16,19.6 15.2,21.4 14.8C23.2 14.4,25.1 14.3,26.9 14.3C28.7 14.3,30.4 14.6,32 14.8Z";
  const BRAIN_FOLDS = "M32 19.3L32 47.4M30.6 27.2L29.7 26.3L28.7 25.6L27.8 24.9L26.8 24.4L25.8 24L24.9 23.6L23.9 23.4L22.9 23.3L22 23.3L21 23.4L20 23.6L19 23.9M30.6 36.3L29.6 36.7L28.6 36.9L27.7 37.1L26.7 37.2L25.8 37.2L24.9 37.2L24 37.1L23 36.9L22.2 36.7L21.3 36.4L20.4 36L19.5 35.5M44.4 35.5L43.5 36L42.6 36.4L41.8 36.7L40.9 36.9L40 37.1L39 37.2L38.1 37.2L37.2 37.2L36.2 37.1L35.3 36.9L34.3 36.7L33.3 36.3M44.9 23.9L43.9 23.6L42.9 23.4L42 23.3L41 23.3L40 23.4L39 23.6L38.1 24L37.1 24.4L36.2 24.9L35.2 25.6L34.3 26.3L33.3 27.2";
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

  function markup(state, id) {
    const palette = state === "error" ? ERROR : PALETTES[current];
    const [a, b, c] = palette.stops;
    const brain = state === "brain";
    const shape = brain ? BRAIN : PATH;
    return `
      <defs>
        <linearGradient id="${id}-paint" x1="12" y1="7" x2="52" y2="58" gradientUnits="userSpaceOnUse">
          <stop offset="0" stop-color="${a}"/>
          <stop offset=".48" stop-color="${b}"/>
          <stop offset="1" stop-color="${c}"/>
        </linearGradient>
        <linearGradient id="${id}-sheen" x1="0" y1="0" x2="1" y2="0">
          <stop stop-color="#fff" stop-opacity="0"/>
          <stop offset=".48" stop-color="#fff" stop-opacity=".7"/>
          <stop offset="1" stop-color="#fff" stop-opacity="0"/>
        </linearGradient>
        <clipPath id="${id}-clip"><path d="${shape}"/></clipPath>
        <filter id="${id}-glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="2.4"/>
        </filter>
      </defs>
      <path class="tau-aura" d="${shape}" fill="${palette.glow}" opacity=".34" filter="url(#${id}-glow)"/>
      <g class="tau-body">
        <path d="${shape}" transform="translate(0 1.1)" fill="#05090c" opacity=".38"/>
        <path class="tau-mark" d="${shape}" fill="url(#${id}-paint)"/>
        <g clip-path="url(#${id}-clip)">
          <g transform="rotate(17 32 32)"><rect class="tau-sheen" x="-35" y="-15" width="19" height="96" fill="url(#${id}-sheen)"/></g>
          ${brain
          ? `<path d="${BRAIN_FOLDS}" fill="none" stroke="#05090c" stroke-opacity=".45" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/>`
          : `<path class="tau-trace" d="${shape}" fill="none" stroke="#fff" stroke-opacity=".52" stroke-width=".65" stroke-dasharray="7 190"/>`}
        </g>
        <path d="${shape}" fill="none" stroke="${a}" stroke-opacity=".32" stroke-width=".35" stroke-linejoin="round"/>
      </g>`;
  }

  function palette(id) { return PALETTES[id] || PALETTES[current]; }

  function svg(state = "idle", { size = 32, title = "Tau" } = {}) {
    state = FRAMES[state] ? state : "idle";
    const el = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    el.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el.setAttribute("width", String(size));
    el.setAttribute("height", String(size));
    el.setAttribute("role", "img");
    el.setAttribute("aria-label", title);
    el.classList.add("tau-sprite", "tau-" + state);
    el.dataset.palette = state === "error" ? "error" : current;
    el.style.setProperty("--tau-glow", palette(state === "error" ? "error" : current).glow);
    el.innerHTML = `<title>${title}</title>` + markup(state, `tau-mark-${++sequence}`);
    return el;
  }

  const TITLES = {
    idle: "Tau: ready",
    brain: "Tau: planning",
    think: "Tau: thinking",
    work: "Tau: running tools",
    wave: "Tau: needs your input",
    error: "Tau: something went wrong",
  };

  // Swap only on a state change so a roster refresh does not restart CSS animation.
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

  // Thinking mood follows the activity glyph (appearance.js): the brain by
  // default, the classic tau frame when arrows or dots are picked instead.
  function glyphPref() {
    let saved = null;
    try { saved = localStorage.getItem("omp-activity-glyph"); } catch { /* non-persistent preview */ }
    return saved === "triangle" || saved === "minimal" ? "think" : "brain";
  }

  function stateOf(S) {
    if (!S || !S.agentId) return "idle";
    const info = (S.agents || []).find((a) => a.id === S.agentId) || {};
    if (S.dead || (S.errorUntil && Date.now() < S.errorUntil)) return "error";
    if (info.needsInput || (OMP.dialogs && OMP.dialogs.isOpen && OMP.dialogs.isOpen())) return "wave";
    if (S.streaming) return S.toolsRunning > 0 || S.bashRunning > 0 ? "work" : glyphPref();
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
    if (persist) {
      try { localStorage.setItem("omp-tau-palette", id); } catch { /* non-persistent preview */ }
    }
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

  OMP.mascot = { svg, svgText, mount, stateOf, hydrate, setPalette, palette, PALETTES, PATH, FRAMES, W, H };
  document.documentElement.dataset.tauPalette = current;
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => hydrate());
  else hydrate();
})(window.OMP = window.OMP || {});
