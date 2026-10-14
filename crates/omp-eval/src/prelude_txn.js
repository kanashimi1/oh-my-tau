// Transactional cell hooks for the JS runner (loaded last, after the other
// preludes, via include_str! into the interpreter global).
//
// `__ompTxnBegin()` takes a shallow identity snapshot of own globalThis
// bindings the preludes did NOT create (name -> value reference, never cloned).
// `__ompTxnEnd(ok)` with ok=false rolls those bindings back: names the failed
// cell rebound or deleted are restored, names it newly created are deleted.
// Best effort by construction: `let`/`const`/`class` lexicals live in the
// cell's declarative environment, not on globalThis, and in-place mutation of
// a shared object cannot be undone — the rollback line says so when any name
// was rolled back. Both hooks must never throw, run in O(#globals) and are
// skipped when globalThis exceeds 50 000 own keys. Keys present when this
// prelude loads (runner + prelude exports) and `__omp*` keys are excluded.

"use strict";

(() => {
  const LIMIT = 50000;
  const base = new Set(
    Object.getOwnPropertyNames(globalThis).filter(n => !n.startsWith("__omp")),
  );
  const state = { snap: null, huge: false };

  function ownKeys() {
    return Object.getOwnPropertyNames(globalThis).filter(
      n => !n.startsWith("__omp") && !base.has(n),
    );
  }

  globalThis.__ompTxnBegin = () => {
    // Default-on; OMP_EVAL_TXN=0 opts out to old TS semantics (failed cell keeps its bindings).
    if (process.env.OMP_EVAL_TXN === "0") return;
    try {
      const keys = ownKeys();
      if (keys.length > LIMIT) {
        state.snap = null;
        state.huge = true;
        return;
      }
      const snap = new Map();
      for (const k of keys) snap.set(k, globalThis[k]);
      state.snap = snap;
      state.huge = false;
    } catch {
      state.snap = null;
      state.huge = false;
    }
  };

  globalThis.__ompTxnEnd = ok => {
    try {
      const snap = state.snap;
      const huge = state.huge;
      state.snap = null;
      state.huge = false;
      if (ok || (snap === null && !huge)) return;
      if (huge) {
        (globalThis.__ompWriteStderr ?? (d => process.stderr.write(d)))(
          `[omp-txn] rollback skipped: globalThis exceeded ${LIMIT} keys at cell start\n`,
        );
        return;
      }
      const rolled = [];
      // Names the failed cell created.
      for (const k of ownKeys()) {
        if (!snap.has(k)) {
          delete globalThis[k];
          rolled.push(k);
        }
      }
      // Names it rebound (identity change) or deleted.
      for (const [k, v] of snap) {
        if (!(k in globalThis) || globalThis[k] !== v) {
          globalThis[k] = v;
          rolled.push(k);
        }
      }
      if (rolled.length > 0) {
        (globalThis.__ompWriteStderr ?? (d => process.stderr.write(d)))(
          "[omp-txn] rolled back: " + rolled.sort().join(", ") +
            " (in-place mutations of shared objects were not undone)\n",
        );
      }
    } catch {
      state.snap = null;
      state.huge = false;
    }
  };
})();
