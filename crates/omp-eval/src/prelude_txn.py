"""Transactional cell hooks for the Python runner (loaded last, after the
other preludes, via include_str! into the user namespace).

`__omp_txn_begin()` takes a shallow identity snapshot of the user-namespace
bindings the preludes did NOT create (name -> object *reference*, never a deep
copy). `__omp_txn_end(ok)` with ok=False rolls the namespace back to that
snapshot: names the failed cell rebound or deleted are restored to their
previous objects, names it newly created are deleted. In-place mutation of a
shared object (list.append, dict update, ...) cannot be undone — the rollback
line says so when any name was rolled back.

Design constraints from the host contract: both hooks must never raise, run in
O(namespace size) and are skipped entirely when the namespace exceeds
50 000 names. Names present when this prelude loads (runner keys, other
prelude exports) and `__omp*`-prefixed names are excluded from the snapshot,
so a cell rebinding `display` or `read` is not rolled back.
"""

import sys

_TXN_BASE = frozenset(n for n in globals() if not n.startswith("__omp"))
_TXN_LIMIT = 50_000
_TXN = {"snap": None, "huge": False}


def _txn_names():
    """User-created binding names only (prelude/runner exports excluded)."""
    return [n for n in globals() if not n.startswith("__omp") and n not in _TXN_BASE]


def __omp_txn_begin() -> None:
    # Default-on; OMP_EVAL_TXN=0 opts out to the old TS semantics (a failed
    # cell keeps every binding it made).
    if __import__("os").environ.get("OMP_EVAL_TXN") == "0":
        return
    try:
        names = _txn_names()
        if len(names) > _TXN_LIMIT:
            # Snapshot would dominate cell cost; the failed cell keeps its effects.
            _TXN["snap"] = None
            _TXN["huge"] = True
            return
        ns = globals()
        _TXN["snap"] = {n: ns[n] for n in names}
        _TXN["huge"] = False
    except Exception:  # noqa: BLE001 - hooks must never raise
        _TXN["snap"] = None
        _TXN["huge"] = False


def __omp_txn_end(ok: bool) -> None:  # noqa: FBT001 - host contract uses a bool flag
    try:
        snap = _TXN["snap"]
        huge = _TXN["huge"]
        _TXN["snap"] = None
        _TXN["huge"] = False
        if ok or (snap is None and not huge):
            return
        if huge:
            print(
                "[omp-txn] rollback skipped: namespace exceeded "
                f"{_TXN_LIMIT} names at cell start",
                file=sys.stderr,
            )
            return
        ns = globals()
        rolled = []
        # Names the failed cell created.
        for n in [n for n in ns if not n.startswith("__omp") and n not in _TXN_BASE]:
            if n not in snap:
                del ns[n]
                rolled.append(n)
        # Names it rebound (identity change) or deleted.
        for n, obj in snap.items():
            if n not in ns or ns[n] is not obj:
                ns[n] = obj
                rolled.append(n)
        if rolled:
            rolled.sort()
            print(
                "[omp-txn] rolled back: " + ", ".join(rolled)
                + " (in-place mutations of shared objects were not undone)",
                file=sys.stderr,
            )
    except Exception:  # noqa: BLE001 - hooks must never raise
        _TXN["snap"] = None
        _TXN["huge"] = False
