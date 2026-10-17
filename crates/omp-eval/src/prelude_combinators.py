"""λ-RLM typed combinators + SERA rubrics for eval cells (Python).

Loaded after prelude_agents.py (it wraps that file's `llm_query`) and before
the transactional-cell prelude. Everything here is opt-in: with no `rubric=`
argument `llm_query` is the untouched agents-prelude function, and the
combinators are new names.

λ-RLM (arXiv 2603.20105, Table 1) — the paper's eight combinators, types
checked at every call (`LamTypeError`), the neural oracle only at leaves:

    lam_split(str, int) -> [str]          lam_filter(a -> bool, [a]) -> [a]
    lam_peek(str, int, int) -> str        lam_reduce((b, b) -> b, [b]) -> b
    lam_map(a -> b, [a]) -> [b]           lam_concat([str]) -> str
    lam_cross([a], [b]) -> [(a, b)]       lam_M(str[, context]) -> str   (= llm_query)

`lam_rlm(P, query, task=...)` builds the fixed-point executor Φ from them. The
plan (fan-out k*, leaf threshold τ*, depth d) comes from the host planner
(`__lam_plan__`), which refuses before any model call when the worst case does
not fit the shared `OMP_RLM_MAX_*` budget; Φ asserts strict shrinkage and the
planned depth, so it always terminates. The model decides leaves only.

SERA (arXiv 2610.04902): `llm_query(prompt, rubric=[{"criterion", "weight"}],
verify=None, n=1)` freezes the rubric host-side BEFORE the child runs, scores
the answer against it (success gate x weighted criterion scores), logs a row to
`OMP_RUBRIC_LOG` and returns a `str` with `.score`/`.record`/`.candidates`.
`n>1` keeps the best-by-rubric candidate (the paper's inference-time selector);
`verify=fn(text)->bool|float` records an external verified outcome next to the
score, and `rubric_rank(records)` evaluates the paper's ranking loss on them.
"""

if "__omp_comb_prelude_loaded__" not in globals():
    __omp_comb_prelude_loaded__ = True

    import os as _comb_os
    import re as _comb_re
    from types import SimpleNamespace as _SimpleNamespace

    _comb_llm_query = llm_query  # noqa: F821  (agents prelude)

    # ------------------------------------------------------------------ errors

    class LamTypeError(TypeError):
        """A combinator was called with arguments outside its type signature."""

    class LamRangeError(ValueError):
        """Well-typed but out-of-range (peek bounds, empty reduce, product too large)."""

    class LamBudgetError(RuntimeError):
        """The run's planned call bound would be exceeded."""

    class LamTerminationError(RuntimeError):
        """The executor left the planned depth bound (cannot happen with a valid plan)."""

    # ------------------------------------------------------------------- types

    def _lam_ty(x):
        if isinstance(x, bool):
            return "bool"
        if isinstance(x, int):
            return "int"
        if isinstance(x, float):
            return "float"
        if isinstance(x, str):
            return "str"
        if isinstance(x, list):
            kinds = {_lam_ty(e) for e in x}
            return "[]" if not kinds else (f"[{kinds.pop()}]" if len(kinds) == 1 else "[mixed]")
        if isinstance(x, tuple):
            return "(" + ", ".join(_lam_ty(e) for e in x) + ")"
        return type(x).__name__

    def _lam_fail(name, sig, msg):
        raise LamTypeError(f"{name}: {msg}; signature {sig}")

    def _lam_int(name, sig, label, v, lo=None):
        if isinstance(v, bool) or not isinstance(v, int):
            _lam_fail(name, sig, f"{label} must be int, got {_lam_ty(v)}")
        if lo is not None and v < lo:
            raise LamRangeError(f"{name}: {label} must be >= {lo}, got {v}")
        return v

    def _lam_list(name, sig, label, v):
        if not isinstance(v, list):
            _lam_fail(name, sig, f"{label} must be a list, got {_lam_ty(v)}")
        return v

    def _lam_fn(name, sig, label, v):
        if not callable(v):
            _lam_fail(name, sig, f"{label} must be callable, got {_lam_ty(v)}")
        return v

    def _lam_homogeneous(name, sig, what, xs):
        kinds = {_lam_ty(e) for e in xs}
        if len(kinds) > 1:
            _lam_fail(name, sig, f"{what} must be homogeneous, got {sorted(kinds)}")

    _LAM_RUN = {"calls": 0, "max_calls": None}
    _LAM_MAX_CROSS = 1_000_000

    # ------------------------------------------------------------- combinators

    def lam_split(s, k):
        """Split : str x int -> [str]. k contiguous chunks (min(k, len)); ''.join(result) == s."""
        sig = "str x int -> [str]"
        if not isinstance(s, str):
            _lam_fail("split", sig, f"s must be str, got {_lam_ty(s)}")
        _lam_int("split", sig, "k", k, 1)
        if not s:
            return [s]
        parts = min(k, len(s))
        base, extra = divmod(len(s), parts)
        out, pos = [], 0
        for i in range(parts):
            end = pos + base + (1 if i < extra else 0)
            out.append(s[pos:end])
            pos = end
        return out

    def lam_peek(s, start, end):
        """Peek : str x int x int -> str. s[start:end], bounds checked (0 <= start <= end <= len)."""
        sig = "str x int x int -> str"
        if not isinstance(s, str):
            _lam_fail("peek", sig, f"s must be str, got {_lam_ty(s)}")
        _lam_int("peek", sig, "start", start)
        _lam_int("peek", sig, "end", end)
        if not 0 <= start <= end <= len(s):
            raise LamRangeError(f"peek: need 0 <= start <= end <= {len(s)}, got ({start}, {end})")
        return s[start:end]

    def lam_map(f, xs):
        """Map : (a -> b) x [a] -> [b]. Results must be homogeneous."""
        sig = "(a -> b) x [a] -> [b]"
        _lam_fn("map", sig, "f", f)
        _lam_list("map", sig, "xs", xs)
        out = [f(x) for x in xs]
        _lam_homogeneous("map", sig, "f's results", out)
        return out

    def lam_filter(pred, xs):
        """Filter : (a -> bool) x [a] -> [a]. The predicate must return a real bool."""
        sig = "(a -> bool) x [a] -> [a]"
        _lam_fn("filter", sig, "pred", pred)
        _lam_list("filter", sig, "xs", xs)
        out = []
        for x in xs:
            keep = pred(x)
            if not isinstance(keep, bool):
                _lam_fail("filter", sig, f"pred must return bool, got {_lam_ty(keep)}")
            if keep:
                out.append(x)
        return out

    _LAM_NO_INIT = object()

    def lam_reduce(op, xs, init=_LAM_NO_INIT):
        """Reduce : (b x b -> b) x [b] -> b. op must keep the element type; empty list needs `init`."""
        sig = "(b x b -> b) x [b] -> b"
        _lam_fn("reduce", sig, "op", op)
        _lam_list("reduce", sig, "xs", xs)
        _lam_homogeneous("reduce", sig, "xs", xs)
        if init is _LAM_NO_INIT:
            if not xs:
                raise LamRangeError("reduce: empty list without init")
            acc, rest = xs[0], xs[1:]
        else:
            acc, rest = init, xs
        want = _lam_ty(acc)
        for x in rest:
            acc = op(acc, x)
            if _lam_ty(acc) != want:
                _lam_fail("reduce", sig, f"op changed the type {want} -> {_lam_ty(acc)}")
        return acc

    def lam_concat(xs, sep=""):
        """Concat : [str] -> str."""
        sig = "[str] -> str"
        _lam_list("concat", sig, "xs", xs)
        for x in xs:
            if not isinstance(x, str):
                _lam_fail("concat", sig, f"elements must be str, got {_lam_ty(x)}")
        if not isinstance(sep, str):
            _lam_fail("concat", sig, f"sep must be str, got {_lam_ty(sep)}")
        return sep.join(xs)

    def lam_cross(xs, ys):
        """Cross : [a] x [b] -> [(a, b)]. Product size is bounded."""
        sig = "[a] x [b] -> [(a, b)]"
        _lam_list("cross", sig, "xs", xs)
        _lam_list("cross", sig, "ys", ys)
        if len(xs) * len(ys) > _LAM_MAX_CROSS:
            raise LamRangeError(f"cross: {len(xs)} x {len(ys)} exceeds {_LAM_MAX_CROSS} pairs")
        return [(x, y) for x in xs for y in ys]

    def lam_M(prompt, context=None, model=None):
        """M : str -> str, the neural oracle (llm_query). Counts against the run's planned bound
        and, host-side, against the shared llm_query budget."""
        sig = "str -> str"
        if not isinstance(prompt, str) or not prompt:
            _lam_fail("M", sig, f"prompt must be a non-empty str, got {_lam_ty(prompt)}")
        if context is not None and not isinstance(context, str):
            _lam_fail("M", sig, f"context must be str, got {_lam_ty(context)}")
        cap = _LAM_RUN["max_calls"]
        if cap is not None and _LAM_RUN["calls"] >= cap:
            raise LamBudgetError(f"M: planned bound of {cap} model calls reached")
        _LAM_RUN["calls"] += 1
        return str(_comb_llm_query(prompt, context=context, model=model))

    lam = _SimpleNamespace(
        split=lam_split, peek=lam_peek, map=lam_map, filter=lam_filter,
        reduce=lam_reduce, concat=lam_concat, cross=lam_cross, M=lam_M,
    )

    # ------------------------------------------------------------ fixed point

    LAM_TASKS = ("search", "classify", "aggregate", "summarise", "multi_hop")

    class LamResult(str):
        """The answer text plus `.plan`, `.task`, `.calls` (model calls used by this run)."""

    def _lam_keywords(query):
        return {w for w in _comb_re.findall(r"[A-Za-z0-9_]{4,}", query.lower())}

    def _lam_detect(query, preview, n, model):
        menu = ", ".join(LAM_TASKS)
        ans = lam_M(
            f"Select the task type that best fits the request. Options: {menu}.\n"
            f"Reply with exactly one option word.\n\n<request>{query}</request>\n"
            f"<input chars={n}>{preview}</input>",
            model=model,
        ).lower().replace("-", "_")
        for t in LAM_TASKS:
            if t in ans:
                return t
        return "summarise"

    def lam_rlm(P, query, task="auto", *, window=None, tau=None, fanout=None, model=None):
        """Φ = fix(λf.λP. |P|<=τ ? M(P) : Reduce(⊕, Map(f, Split(P, k)))) over the typed combinators.

        task: auto | search | classify | aggregate | summarise | multi_hop (paper Table 1). `window`
        is the leaf window K in characters (env OMP_LAM_WINDOW, default 24000). Returns `LamResult`."""
        if not isinstance(P, str):
            _lam_fail("lam_rlm", "str x str -> str", f"P must be str, got {_lam_ty(P)}")
        if not isinstance(query, str) or not query.strip():
            _lam_fail("lam_rlm", "str x str -> str", "query must be a non-empty str")
        if task != "auto" and task not in LAM_TASKS:
            raise LamTypeError(f"lam_rlm: task must be 'auto' or one of {LAM_TASKS}, got {task!r}")
        if window is None:
            window = int(_comb_os.environ.get("OMP_LAM_WINDOW", "24000"))
        auto = task == "auto" and len(P) > window
        if task == "auto" and not auto:
            task = "summarise"  # fits one leaf: a single direct call, the task type is irrelevant
        # worst case beyond the leaves: detection + one synthesis call (summarise/multi_hop)
        extra = 0 if len(P) <= window else (1 if auto else 0) + (1 if task in ("summarise", "multi_hop", "auto") else 0)
        args = {"n": len(P), "window": window, "extra_calls": extra}
        if tau is not None:
            args["tau"] = tau
        if fanout is not None:
            args["fanout"] = fanout
        plan = _bridge_call("__lam_plan__", args)  # raises before any model call if it cannot fit
        start = _LAM_RUN["calls"]
        saved = dict(_LAM_RUN)
        _LAM_RUN["max_calls"] = start + plan["max_calls"]
        try:
            if plan["direct"]:
                out = lam_M(f"{query}\n\n<input>\n{P}\n</input>", model=model)
                return _lam_result(out, plan, "direct", start)
            if auto:
                task = _lam_detect(query, lam_peek(P, 0, min(500, len(P))), len(P), model)
            out = _lam_phi(P, query, task, plan, model)
            return _lam_result(out, plan, task, start)
        finally:
            _LAM_RUN["max_calls"] = saved["max_calls"]

    def _lam_result(out, plan, task, start):
        r = LamResult(out)
        r.plan, r.task, r.calls = plan, task, _LAM_RUN["calls"] - start
        return r

    def _lam_leaf_prompt(task, query, chunk):
        if task in ("search", "multi_hop"):
            ask = "Answer using ONLY the input. If it does not contain the answer, reply exactly NONE."
        elif task == "classify":
            ask = "Classify the input as the request asks; reply with the label(s) only."
        elif task == "aggregate":
            ask = "Extract the items the request asks for, one per line, no commentary."
        else:
            ask = "Summarise the parts of the input relevant to the request."
        return f"{query}\n{ask}\n\n<input>\n{chunk}\n</input>"

    def _lam_merge(parts, task):
        if task in ("search", "multi_hop", "summarise"):
            kept = lam_filter(lambda r: r.strip().upper() != "NONE", parts)
            if task == "search":
                return "NONE" if not kept else lam_reduce(lambda a, b: a if len(a) >= len(b) else b, kept)
            return lam_concat(kept, "\n") if kept else "NONE"
        if task == "aggregate":  # Merge: ordered union of lines
            seen, merged = set(), []
            for line in lam_concat(parts, "\n").splitlines():
                if line.strip() and line.strip() not in seen:
                    seen.add(line.strip())
                    merged.append(line)
            return lam_concat(merged, "\n")
        return lam_concat(parts, "\n")  # classify

    def _lam_phi(P, query, task, plan, model):
        keys = _lam_keywords(query)

        def phi(p, rank, top):
            if len(p) <= plan["tau"]:
                return lam_M(_lam_leaf_prompt(task, query, p), model=model)
            if rank >= plan["depth"]:
                raise LamTerminationError(f"lam: depth bound {plan['depth']} exceeded")
            parts = lam_split(p, plan["k"])
            if max(len(c) for c in parts) >= len(p):
                raise LamTerminationError("lam: split did not shrink the input")
            if task in ("search", "multi_hop") and keys:  # symbolic prune; keep all if nothing matches
                hits = lam_filter(lambda c: any(w in c.lower() for w in keys), parts)
                parts = hits or parts
            merged = _lam_merge(lam_map(lambda c: phi(c, rank + 1, False), parts), task)
            if top and task in ("summarise", "multi_hop") and merged != "NONE":
                return lam_M(f"{query}\nSynthesize one final answer from these partial results.\n\n<partials>\n{merged}\n</partials>", model=model)
            return merged

        return phi(P, 0, True)

    # ------------------------------------------------------------------- SERA

    class Scored(str):
        """llm_query answer carrying its rubric score: `.score` in [0,1], `.record`, `.candidates`."""

    def _rubric_one(prompt, text, fr_id, verify, idx, rubric_model):
        verified = None
        if verify is not None:
            v = verify(text)
            verified = float(v)
            if not 0.0 <= verified <= 1.0:
                raise ValueError(f"verify() must return a bool or a number in [0,1], got {v!r}")
        args = {"id": fr_id, "prompt": prompt, "output": text, "run_id": f"llm_query#{idx}", "source": "llm_query"}
        if verified is not None:
            args["verified"] = verified
        if rubric_model is not None:
            args["model"] = rubric_model
        return _bridge_call("__rubric_score__", args)

    def llm_query(prompt, context=None, model=None, *args, rubric=None, verify=None, n=1, rubric_model=None, **kw):  # noqa: F811
        """llm_query (see prelude_agents) + SERA rubrics.

        rubric=[{"criterion": str, "weight": number}] is frozen before the call, the answer is scored
        against it (gate x weighted criteria, one budgeted scorer call per candidate) and the record is
        appended to OMP_RUBRIC_LOG. n>1 draws n independent candidates and returns the best by rubric
        score. verify=fn(text)->bool|[0,1] attaches an external verified outcome to each logged row."""
        if rubric is None:
            if verify is not None or n != 1 or rubric_model is not None:
                raise ValueError("llm_query: verify/n/rubric_model require rubric=")
            return _comb_llm_query(prompt, context, model, *args, **kw)
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 16:
            raise ValueError("llm_query: n must be an int in 1..16")
        frozen = _bridge_call("__rubric_freeze__", {"rubric": rubric})  # committed before any child call
        cands = []
        for i in range(n):
            # memo is keyed by prompt: later candidates get a distinct suffix so they are real samples
            p = prompt if i == 0 else f"{prompt}\n\n[independent attempt {i + 1} of {n}]"
            text = str(_comb_llm_query(p, context, model, *args, **kw))
            cands.append((text, _rubric_one(prompt, text, frozen["id"], verify, i + 1, rubric_model)))
        best = max(range(len(cands)), key=lambda i: (cands[i][1]["score"], -i))
        out = Scored(cands[best][0])
        out.score = cands[best][1]["score"]
        out.record = cands[best][1]
        out.candidates = [{"text": t, "score": r["score"], "verified": r.get("verified")} for t, r in cands]
        return out

    def rubric_rank(records, margin=0.1):
        """SERA ranking loss (eq. 3) over scored candidates that carry a verified outcome:
        mean max(0, margin - s_i + s_j) for pairs with y_i > y_j. {'loss': None} = no usable pair."""
        rows = [r for r in records if r.get("verified") is not None]
        return _bridge_call("__rubric_rank__", {
            "scores": [r["score"] for r in rows], "verified": [r["verified"] for r in rows], "margin": margin,
        })
