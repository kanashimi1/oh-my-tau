// λ-RLM typed combinators + SERA rubrics for the JS eval runner (port of
// prelude_combinators.py; loaded after prelude_agents.js, before prelude_txn.js).
//
// λ-RLM (arXiv 2603.20105, Table 1): lamSplit, lamPeek, lamMap, lamFilter,
// lamReduce, lamConcat, lamCross, lamM — types checked at every call
// (LamTypeError), the model only at leaves — plus lamRlm(P, query, opts), the
// fixed-point executor planned by the host (`__lam_plan__`). Map/filter/reduce
// callbacks may be async. SERA (arXiv 2610.04902): llm_query(prompt, { rubric,
// verify, n, rubric_model, ... }) freezes the rubric host-side before the
// child runs, scores the answer, logs to OMP_RUBRIC_LOG and returns a String
// object with .score/.record/.candidates; without `rubric` the original
// llm_query is called unchanged.
(() => {
  if (globalThis.__omp_comb_prelude_loaded__) return;
  globalThis.__omp_comb_prelude_loaded__ = true;

  const hostCall = (op, args) => globalThis.__ompHostCall(op, args);
  const origQuery = globalThis.llm_query;

  class LamTypeError extends TypeError { constructor(m) { super(m); this.name = "LamTypeError"; } }
  class LamRangeError extends RangeError { constructor(m) { super(m); this.name = "LamRangeError"; } }
  class LamBudgetError extends Error { constructor(m) { super(m); this.name = "LamBudgetError"; } }
  class LamTerminationError extends Error { constructor(m) { super(m); this.name = "LamTerminationError"; } }

  const ty = x => {
    if (typeof x === "boolean") return "bool";
    if (typeof x === "number") return Number.isInteger(x) ? "int" : "float";
    if (typeof x === "string") return "str";
    if (Array.isArray(x)) {
      const kinds = new Set(x.map(ty));
      return kinds.size === 0 ? "[]" : kinds.size === 1 ? `[${[...kinds][0]}]` : "[mixed]";
    }
    return x === null ? "null" : typeof x;
  };
  const fail = (name, sig, msg) => { throw new LamTypeError(`${name}: ${msg}; signature ${sig}`); };
  const isInt = v => typeof v === "number" && Number.isInteger(v);
  const needInt = (name, sig, label, v, lo) => {
    if (!isInt(v)) fail(name, sig, `${label} must be int, got ${ty(v)}`);
    if (lo !== undefined && v < lo) throw new LamRangeError(`${name}: ${label} must be >= ${lo}, got ${v}`);
    return v;
  };
  const needList = (name, sig, label, v) => { if (!Array.isArray(v)) fail(name, sig, `${label} must be a list, got ${ty(v)}`); return v; };
  const needFn = (name, sig, label, v) => { if (typeof v !== "function") fail(name, sig, `${label} must be callable, got ${ty(v)}`); return v; };
  const homogeneous = (name, sig, what, xs) => {
    const kinds = new Set(xs.map(ty));
    if (kinds.size > 1) fail(name, sig, `${what} must be homogeneous, got ${JSON.stringify([...kinds].sort())}`);
  };

  const run = { calls: 0, maxCalls: null };
  const MAX_CROSS = 1_000_000;

  const lamSplit = (s, k) => {
    const sig = "str x int -> [str]";
    if (typeof s !== "string") fail("split", sig, `s must be str, got ${ty(s)}`);
    needInt("split", sig, "k", k, 1);
    if (!s) return [s];
    const parts = Math.min(k, s.length);
    const base = Math.floor(s.length / parts), extra = s.length % parts;
    const out = [];
    let pos = 0;
    for (let i = 0; i < parts; i++) {
      const end = pos + base + (i < extra ? 1 : 0);
      out.push(s.slice(pos, end));
      pos = end;
    }
    return out;
  };
  const lamPeek = (s, start, end) => {
    const sig = "str x int x int -> str";
    if (typeof s !== "string") fail("peek", sig, `s must be str, got ${ty(s)}`);
    needInt("peek", sig, "start", start);
    needInt("peek", sig, "end", end);
    if (!(start >= 0 && start <= end && end <= s.length)) {
      throw new LamRangeError(`peek: need 0 <= start <= end <= ${s.length}, got (${start}, ${end})`);
    }
    return s.slice(start, end);
  };
  const lamMap = async (f, xs) => {
    const sig = "(a -> b) x [a] -> [b]";
    needFn("map", sig, "f", f);
    needList("map", sig, "xs", xs);
    const out = [];
    for (const x of xs) out.push(await f(x));
    homogeneous("map", sig, "f's results", out);
    return out;
  };
  const lamFilter = async (pred, xs) => {
    const sig = "(a -> bool) x [a] -> [a]";
    needFn("filter", sig, "pred", pred);
    needList("filter", sig, "xs", xs);
    const out = [];
    for (const x of xs) {
      const keep = await pred(x);
      if (typeof keep !== "boolean") fail("filter", sig, `pred must return bool, got ${ty(keep)}`);
      if (keep) out.push(x);
    }
    return out;
  };
  const NO_INIT = Symbol("no-init");
  const lamReduce = async (op, xs, init = NO_INIT) => {
    const sig = "(b x b -> b) x [b] -> b";
    needFn("reduce", sig, "op", op);
    needList("reduce", sig, "xs", xs);
    homogeneous("reduce", sig, "xs", xs);
    let acc, rest;
    if (init === NO_INIT) {
      if (!xs.length) throw new LamRangeError("reduce: empty list without init");
      [acc, ...rest] = xs;
    } else { acc = init; rest = xs; }
    const want = ty(acc);
    for (const x of rest) {
      acc = await op(acc, x);
      if (ty(acc) !== want) fail("reduce", sig, `op changed the type ${want} -> ${ty(acc)}`);
    }
    return acc;
  };
  const lamConcat = (xs, sep = "") => {
    const sig = "[str] -> str";
    needList("concat", sig, "xs", xs);
    for (const x of xs) if (typeof x !== "string") fail("concat", sig, `elements must be str, got ${ty(x)}`);
    if (typeof sep !== "string") fail("concat", sig, `sep must be str, got ${ty(sep)}`);
    return xs.join(sep);
  };
  const lamCross = (xs, ys) => {
    const sig = "[a] x [b] -> [(a, b)]";
    needList("cross", sig, "xs", xs);
    needList("cross", sig, "ys", ys);
    if (xs.length * ys.length > MAX_CROSS) throw new LamRangeError(`cross: ${xs.length} x ${ys.length} exceeds ${MAX_CROSS} pairs`);
    return xs.flatMap(x => ys.map(y => [x, y]));
  };
  const lamM = async (prompt, opts = {}) => {
    const sig = "str -> str";
    if (typeof prompt !== "string" || !prompt) fail("M", sig, `prompt must be a non-empty str, got ${ty(prompt)}`);
    if (opts.context != null && typeof opts.context !== "string") fail("M", sig, `context must be str, got ${ty(opts.context)}`);
    if (run.maxCalls !== null && run.calls >= run.maxCalls) throw new LamBudgetError(`M: planned bound of ${run.maxCalls} model calls reached`);
    run.calls += 1;
    return String(await origQuery(prompt, opts));
  };
  const lam = { split: lamSplit, peek: lamPeek, map: lamMap, filter: lamFilter, reduce: lamReduce, concat: lamConcat, cross: lamCross, M: lamM };

  const LAM_TASKS = ["search", "classify", "aggregate", "summarise", "multi_hop"];

  const leafPrompt = (task, query, chunk) => {
    let ask;
    if (task === "search" || task === "multi_hop") ask = "Answer using ONLY the input. If it does not contain the answer, reply exactly NONE.";
    else if (task === "classify") ask = "Classify the input as the request asks; reply with the label(s) only.";
    else if (task === "aggregate") ask = "Extract the items the request asks for, one per line, no commentary.";
    else ask = "Summarise the parts of the input relevant to the request.";
    return `${query}\n${ask}\n\n<input>\n${chunk}\n</input>`;
  };
  const merge = async (parts, task) => {
    if (task === "search" || task === "multi_hop" || task === "summarise") {
      const kept = await lamFilter(r => r.trim().toUpperCase() !== "NONE", parts);
      if (task === "search") return kept.length ? lamReduce((a, b) => (a.length >= b.length ? a : b), kept) : "NONE";
      return kept.length ? lamConcat(kept, "\n") : "NONE";
    }
    if (task === "aggregate") {
      const seen = new Set(), merged = [];
      for (const line of lamConcat(parts, "\n").split("\n")) {
        const t = line.trim();
        if (t && !seen.has(t)) { seen.add(t); merged.push(line); }
      }
      return lamConcat(merged, "\n");
    }
    return lamConcat(parts, "\n");
  };

  const lamRlm = async (P, query, opts = {}) => {
    const sig = "str x str -> str";
    if (typeof P !== "string") fail("lam_rlm", sig, `P must be str, got ${ty(P)}`);
    if (typeof query !== "string" || !query.trim()) fail("lam_rlm", sig, "query must be a non-empty str");
    let { task = "auto", window, tau, fanout, model } = opts;
    if (task !== "auto" && !LAM_TASKS.includes(task)) throw new LamTypeError(`lam_rlm: task must be 'auto' or one of ${JSON.stringify(LAM_TASKS)}, got ${JSON.stringify(task)}`);
    window ??= Number(globalThis.process?.env?.OMP_LAM_WINDOW ?? 24000);
    const direct = P.length <= window;
    const auto = task === "auto" && !direct;
    if (task === "auto" && direct) task = "summarise";
    const extra = direct ? 0 : (auto ? 1 : 0) + (["summarise", "multi_hop", "auto"].includes(task) ? 1 : 0);
    const args = { n: P.length, window, extra_calls: extra };
    if (tau != null) args.tau = tau;
    if (fanout != null) args.fanout = fanout;
    const plan = await hostCall("__lam_plan__", args);
    const start = run.calls, saved = run.maxCalls;
    run.maxCalls = start + plan.max_calls;
    const mOpts = model ? { model } : {};
    const result = (text, t) => Object.assign(new String(text), { plan, task: t, calls: run.calls - start });
    try {
      if (plan.direct) return result(await lamM(`${query}\n\n<input>\n${P}\n</input>`, mOpts), "direct");
      if (auto) {
        const menu = LAM_TASKS.join(", ");
        const ans = (await lamM(`Select the task type that best fits the request. Options: ${menu}.\nReply with exactly one option word.\n\n<request>${query}</request>\n<input chars=${P.length}>${lamPeek(P, 0, Math.min(500, P.length))}</input>`, mOpts)).toLowerCase().replace(/-/g, "_");
        task = LAM_TASKS.find(t => ans.includes(t)) ?? "summarise";
      }
      const keys = new Set((query.toLowerCase().match(/[a-z0-9_]{4,}/g)) ?? []);
      const phi = async (p, rank, top) => {
        if (p.length <= plan.tau) return lamM(leafPrompt(task, query, p), mOpts);
        if (rank >= plan.depth) throw new LamTerminationError(`lam: depth bound ${plan.depth} exceeded`);
        let parts = lamSplit(p, plan.k);
        if (Math.max(...parts.map(c => c.length)) >= p.length) throw new LamTerminationError("lam: split did not shrink the input");
        if ((task === "search" || task === "multi_hop") && keys.size) {
          const hits = await lamFilter(c => [...keys].some(w => c.toLowerCase().includes(w)), parts);
          if (hits.length) parts = hits;
        }
        const merged = await merge(await lamMap(c => phi(c, rank + 1, false), parts), task);
        if (top && (task === "summarise" || task === "multi_hop") && merged !== "NONE") {
          return lamM(`${query}\nSynthesize one final answer from these partial results.\n\n<partials>\n${merged}\n</partials>`, mOpts);
        }
        return merged;
      };
      return result(await phi(P, 0, true), task);
    } finally {
      run.maxCalls = saved;
    }
  };

  const RUBRIC_KEYS = ["rubric", "verify", "n", "rubric_model"];
  const llm_query = async (prompt, opts, ...rest) => {
    if (opts == null || typeof opts !== "object" || Array.isArray(opts) || opts.rubric == null) {
      if (opts && typeof opts === "object" && !Array.isArray(opts) && RUBRIC_KEYS.slice(1).some(k => opts[k] != null)) {
        throw new Error("llm_query: verify/n/rubric_model require rubric");
      }
      return origQuery(prompt, opts, ...rest);
    }
    const { rubric, verify, n = 1, rubric_model, ...passthrough } = opts;
    if (!Number.isInteger(n) || n < 1 || n > 16) throw new Error("llm_query: n must be an int in 1..16");
    const frozen = await hostCall("__rubric_freeze__", { rubric }); // committed before any child call
    const cands = [];
    for (let i = 0; i < n; i++) {
      const p = i === 0 ? prompt : `${prompt}\n\n[independent attempt ${i + 1} of ${n}]`;
      const text = String(await origQuery(p, passthrough, ...rest));
      let verified;
      if (verify) {
        const v = await verify(text);
        verified = Number(v);
        if (!(verified >= 0 && verified <= 1)) throw new Error(`verify() must return a bool or a number in [0,1], got ${v}`);
      }
      const args = { id: frozen.id, prompt, output: text, run_id: `llm_query#${i + 1}`, source: "llm_query" };
      if (verified !== undefined) args.verified = verified;
      if (rubric_model) args.model = rubric_model;
      cands.push([text, await hostCall("__rubric_score__", args)]);
    }
    let best = 0;
    cands.forEach(([, r], i) => { if (r.score > cands[best][1].score) best = i; });
    return Object.assign(new String(cands[best][0]), {
      score: cands[best][1].score,
      record: cands[best][1],
      candidates: cands.map(([t, r]) => ({ text: t, score: r.score, verified: r.verified ?? null })),
    });
  };
  const rubric_rank = async (records, margin = 0.1) => {
    const rows = records.filter(r => r.verified != null);
    return hostCall("__rubric_rank__", { scores: rows.map(r => r.score), verified: rows.map(r => r.verified), margin });
  };

  Object.assign(globalThis, {
    llm_query, rubric_rank, lam, lam_rlm: lamRlm, lam_split: lamSplit, lam_peek: lamPeek, lam_map: lamMap,
    lam_filter: lamFilter, lam_reduce: lamReduce, lam_concat: lamConcat, lam_cross: lamCross, lam_M: lamM,
    LamTypeError, LamRangeError, LamBudgetError, LamTerminationError,
  });
})();
