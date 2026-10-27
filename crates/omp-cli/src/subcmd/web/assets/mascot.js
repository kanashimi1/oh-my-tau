"use strict";
// Tau — the omt mascot: a little τ-shaped spirit (hood = crossbar, body =
// stem, wispy tail = the hook). Pixel frames rendered as crisp SVG.
//
//   OMP.mascot.svg(state, { size, title })  → <svg> element
//   OMP.mascot.mount(host)                   → live sprite; .set(state)
//   OMP.mascot.stateOf(S)                    → state for an app state
//
// States: idle (blinks), think, work, wave (needs your input), error.
// Glyphs: # body · h highlight · d shade · D visor · E eyes · R error eyes ·
// s sparkle (animated separately) · . empty. A dark outline is added
// around the silhouette so the sprite reads on light and dark themes.
(function (OMP) {
  const FRAMES = {
    idle: [
      ".......................",
      "........hhhhhhh........",
      ".......#########.......",
      "......##DDDDDDD##......",
      "......#DDEDDDEDD#......",
      "......#DDEDDDEDD#......",
      "......##DDDDDDD##......",
      "..h#################h..",
      ".####################d.",
      ".##d.....#####.....##d.",
      ".#d......#####......#d.",
      ".........#####.........",
      ".........#####.........",
      ".........#####.........",
      "..........#####........",
      "..........######..##...",
      "...........#######d....",
      ".............dddd......",
      ".......................",
    ],
    blink: [
      ".......................",
      "........hhhhhhh........",
      ".......#########.......",
      "......##DDDDDDD##......",
      "......#DDDDDDDDD#......",
      "......#DEEEDEEED#......",
      "......##DDDDDDD##......",
      "..h#################h..",
      ".####################d.",
      ".##d.....#####.....##d.",
      ".#d......#####......#d.",
      ".........#####.........",
      ".........#####.........",
      ".........#####.........",
      "..........#####........",
      "..........######..##...",
      "...........#######d....",
      ".............dddd......",
      ".......................",
    ],
    think: [
      ".......................",
      "........hhhhhhh...s....",
      ".......#########.sss...",
      "....s.##DDEDDDE##.s....",
      "......#DDDEDDDED#......",
      "......#DDDDDDDDD#...s..",
      "......##DDDDDDD##......",
      "..h#################h..",
      ".####################d.",
      ".##d.....#####.....##d.",
      ".#d......#####......#d.",
      ".........#####.........",
      ".........#####.........",
      ".........#####.........",
      "..........#####........",
      "..........######..##...",
      "...........#######d....",
      ".............dddd......",
      ".......................",
    ],
    wave: [
      ".......................",
      "........hhhhhhh........",
      ".......#########....hh.",
      "......##DDDDDDD##...##.",
      "......#DDEDDDEDD#..###.",
      "......#DEDEDEDED#.###..",
      "......##DDDDDDD#####...",
      "..h#################...",
      ".###################d..",
      ".##d.....#####.........",
      ".#d......#####.........",
      ".........#####.........",
      ".........#####.........",
      ".........#####.........",
      "..........#####........",
      "..........######..##...",
      "...........#######d....",
      ".............dddd......",
      ".......................",
    ],
    work: [
      ".......................",
      "........hhhhhhh........",
      ".......#########.......",
      "......##DDDDDDD##......",
      "......#DDDDDDDDD#......",
      "......#DEEDDDEED#......",
      "......##DDDDDDD##......",
      "..h#################h..",
      ".####################d.",
      ".##d.....#####.....##d.",
      ".#d......#####.....s#d.",
      ".........#####.......s.",
      ".........#####......s.s",
      ".........#####.......s.",
      "..........#####........",
      "..........######..##...",
      "...........#######d....",
      ".............dddd......",
      ".......................",
    ],
    error: [
      ".......................",
      "........hhhhhhh........",
      ".......#########.......",
      "......##RDRDRDR##......",
      "......#DDRDDDRDD#......",
      "......#DRDRDRDRD#......",
      "......##DDDDDDD##......",
      "..h#################h..",
      ".####################d.",
      ".##d.....#####.....##d.",
      ".#d......#####......#d.",
      ".........#####.........",
      ".........#####.........",
      ".........#####.........",
      "..........#####........",
      "..........######..##...",
      "...........#######d....",
      ".............dddd......",
      ".......................",
    ],
  };
  const W = 23, H = 19;
  const PINK = [255, 92, 168], VIOLET = [170, 90, 255], CYAN = [80, 220, 255];
  const OUTLINE = [18, 8, 30];
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * Math.min(1, Math.max(0, t))));
  const body = (x, y) => {
    const t = (x / (W - 1)) * 0.55 + (y / (H - 1)) * 0.45;
    return t < 0.5 ? mix(PINK, VIOLET, t * 2) : mix(VIOLET, CYAN, (t - 0.5) * 2);
  };
  const COLOR = {
    "#": body,
    h: (x, y) => mix(body(x, y), [255, 255, 255], 0.5),
    d: (x, y) => mix(body(x, y), [40, 10, 70], 0.35),
    D: () => [30, 14, 48],
    E: () => [150, 252, 255],
    R: () => [255, 84, 104],
    s: () => [255, 236, 140],
  };
  const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
  const at = (f, x, y) => (y >= 0 && y < H && x >= 0 && x < W ? f[y][x] : ".");

  // Horizontal runs of one colour per row → few <rect>s.
  function rects(cells) {
    const out = [];
    for (let y = 0; y < H; y++) {
      let run = null;
      for (let x = 0; x <= W; x++) {
        const c = x < W ? cells[y][x] : null;
        if (run && c === run.c) { run.w++; continue; }
        if (run) out.push(`<rect x="${run.x}" y="${y}" width="${run.w}" height="1" fill="${run.c}"/>`);
        run = c ? { x, w: 1, c } : null;
      }
    }
    return out.join("");
  }

  const cache = new Map();
  // SVG markup of one frame: outline, body, sparkles in their own group.
  function markup(state) {
    if (cache.has(state)) return cache.get(state);
    const f = FRAMES[state] || FRAMES.idle;
    const blank = () => Array.from({ length: H }, () => Array(W).fill(null));
    const main = blank(), spark = blank(), blink = blank();
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const g = f[y][x];
        if (g === "s") { spark[y][x] = rgb(COLOR.s()); continue; }
        if (g !== ".") { main[y][x] = rgb(COLOR[g](x, y)); continue; }
        const solid = (gx) => gx !== "." && gx !== "s";
        if (solid(at(f, x + 1, y)) || solid(at(f, x - 1, y)) || solid(at(f, x, y + 1)) || solid(at(f, x, y - 1))) {
          main[y][x] = rgb(OUTLINE);
        }
      }
    }
    // Idle blink: the pixels where the blink frame differs, shown briefly.
    if (state === "idle") {
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const b = FRAMES.blink[y][x];
        if (b !== f[y][x]) blink[y][x] = rgb(COLOR[b](x, y));
      }
    }
    const s = `<g class="tau-body">${rects(main)}</g>` +
      (state === "idle" ? `<g class="tau-blink">${rects(blink)}</g>` : "") +
      `<g class="tau-spark">${rects(spark)}</g>`;
    cache.set(state, s);
    return s;
  }

  function svg(state = "idle", { size = 32, title = "Tau" } = {}) {
    const ns = "http://www.w3.org/2000/svg";
    const el = document.createElementNS(ns, "svg");
    el.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el.setAttribute("width", String(size));
    el.setAttribute("height", String(Math.round((size * H) / W)));
    el.setAttribute("shape-rendering", "crispEdges");
    el.setAttribute("role", "img");
    el.setAttribute("aria-label", title);
    el.classList.add("tau-sprite", "tau-" + (FRAMES[state] ? state : "idle"));
    el.innerHTML = `<title>${title}</title>` + markup(state);
    return el;
  }

  const TITLES = {
    idle: "Tau: ready",
    think: "Tau: thinking",
    work: "Tau: running tools",
    wave: "Tau: needs your input",
    error: "Tau: something went wrong",
  };

  // A sprite that swaps frames only when the state changes (re-rendering
  // the same state would restart its animation).
  function mount(host, { size = 32, state = "idle" } = {}) {
    let cur = null;
    const api = {
      set(next) {
        next = FRAMES[next] ? next : "idle";
        if (next === cur) return;
        cur = next;
        host.replaceChildren(svg(next, { size, title: TITLES[next] || "Tau" }));
        host.dataset.state = next;
      },
      get state() { return cur; },
    };
    api.set(state);
    return api;
  }

  // Mascot state for the web app state `S` (app.js).
  function stateOf(S) {
    if (!S || !S.agentId) return "idle";
    const info = (S.agents || []).find((a) => a.id === S.agentId) || {};
    if (S.dead || (S.errorUntil && Date.now() < S.errorUntil)) return "error";
    if (info.needsInput || (OMP.dialogs && OMP.dialogs.isOpen && OMP.dialogs.isOpen())) return "wave";
    if (S.streaming) return S.toolsRunning > 0 || S.bashRunning > 0 ? "work" : "think";
    return "idle";
  }

  // Static placeholders: <span data-tau="idle" data-size="96">.
  function hydrate(root = document) {
    for (const host of root.querySelectorAll("[data-tau]")) {
      if (host.firstChild) continue;
      mount(host, { size: Number(host.dataset.size) || 32, state: host.dataset.tau || "idle" });
    }
  }

  // Standalone SVG source of a frame (favicon / icon generation).
  const svgText = (state = "idle") =>
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" shape-rendering="crispEdges">${markup(state)}</svg>`;

  OMP.mascot = { svg, svgText, mount, stateOf, hydrate, FRAMES, W, H };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => hydrate());
  else hydrate();
})(window.OMP = window.OMP || {});
