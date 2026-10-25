"use strict";
// Sidebar helpers: Kimi-style date grouping + pinned sessions. Owner: sidebar slice.
// Shell rendering lives in shell.js; this file only provides the pure helpers.
(function (OMP) {
  const PIN_KEY = "omp-web-pinned";

  const GROUP_LABELS = {
    pinned: "Pinned",
    today: "Today",
    yesterday: "Yesterday",
    seven: "Previous 7 days",
    thirty: "Previous 30 days",
    older: "Older",
  };
  // Render order for the dated groups (pinned is emitted first, separately).
  const GROUP_ORDER = ["today", "yesterday", "seven", "thirty", "older"];

  const startOfDay = (d) => {
    const x = new Date(d);
    x.setHours(0, 0, 0, 0);
    return x.getTime();
  };

  // Bucket a timestamp into a date group label key (Kimi-style).
  function dateGroup(ms) {
    if (!ms) return "older";
    const now = startOfDay(Date.now());
    const day = startOfDay(ms);
    const diff = Math.floor((now - day) / 86400000);
    if (diff <= 0) return "today";
    if (diff === 1) return "yesterday";
    if (diff < 7) return "seven";
    if (diff < 30) return "thirty";
    return "older";
  }

  function readPinned() {
    try {
      const j = JSON.parse(localStorage.getItem(PIN_KEY) || "[]");
      return Array.isArray(j) ? new Set(j) : new Set();
    } catch { return new Set(); }
  }

  function savePinned(set) {
    try { localStorage.setItem(PIN_KEY, JSON.stringify([...set])); } catch { /* quota */ }
  }

  // Toggle a row key in the pinned set; returns the new pinned state.
  function togglePinned(key) {
    const set = readPinned();
    if (set.has(key)) set.delete(key); else set.add(key);
    savePinned(set);
    return set.has(key);
  }

  OMP.sidebar = { dateGroup, GROUP_LABELS, GROUP_ORDER, readPinned, togglePinned };
})(window.OMP);
