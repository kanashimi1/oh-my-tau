// Agent/completion/judgment/workpool/budget prelude for the JS eval runner.
// Port of the transport part of omp's eval/js/shared/prelude.txt. Every host
// round-trip goes through `globalThis.__ompHostCall(op, args)` (waits for the
// reply) with the op names from agent/completion/judgment/judgment-batch/
// workpool/handle bridges: __agent__, __completion__, __judge__,
// __judge_batch__, __workpool__, __budget__, __wait__, __status__, __cancel__,
// plus `write` for agent:// messaging. I/O helpers (read/write/output),
// display, tool, log and phase live in prelude.js, which is loaded before this
// file; this module only consumes them.
//
// The whole file runs inside an IIFE so top-level bindings never collide with
// user cells, and it is idempotent: the guard flag skips re-definition on
// re-run.
(() => {
  if (globalThis.__omp_agents_prelude_loaded__) return;
  globalThis.__omp_agents_prelude_loaded__ = true;

  const isNil = value => value === undefined || value === null;
  const isPlainObject = value => value !== null && typeof value === "object" && !Array.isArray(value);

  // Accepts one trailing options object (preferred) or positional optional
  // args, mirroring the TS prelude so `wait(h, { timeout })` and
  // `wait(h, 10)` behave the same way.
  const positionalOptions = (name, args, keys, example) => {
    for (let index = keys.length; index < args.length; index++) {
      if (!isNil(args[index])) {
        throw new TypeError(
          `${name}() accepts at most ${keys.length} positional optional args; got ${args.length}. Pass ${name}(..., ${example}) for named options.`,
        );
      }
    }
    const options = {};
    for (let index = 0; index < keys.length && index < args.length; index++) {
      const value = args[index];
      if (!isNil(value)) options[keys[index]] = value;
    }
    return options;
  };
  const optionsArg = (name, value, rest, keys, example) => {
    if (isNil(value)) return positionalOptions(name, [value, ...rest], keys, example);
    if (isPlainObject(value)) {
      if (rest.some(arg => !isNil(arg))) {
        throw new TypeError(
          `${name}() takes either a single trailing options object like ${example} or positional optional args; do not mix both forms.`,
        );
      }
      return value;
    }
    if (typeof value === "object") {
      const kind = Array.isArray(value) ? "an array" : (value.constructor?.name ?? "object");
      throw new TypeError(
        `${name}() options must be a plain object like ${example}, null/undefined, or positional optional args, not ${kind}.`,
      );
    }
    return positionalOptions(name, [value, ...rest], keys, example);
  };

  const hostCall = (op, args) => globalThis.__ompHostCall(op, args);

  const hasOwn = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  const HANDLE_UNSET = Symbol("handle-unset");
  const timeoutError = message => {
    const error = new Error(message);
    error.name = "TimeoutError";
    return error;
  };

  class EvalHandle {
    constructor(kind, id, schema) {
      this.kind = kind;
      this.id = id;
      this._schema = schema;
      this._result = HANDLE_UNSET;
    }

    async status() {
      const snapshot = await hostCall("__status__", {
        item: { kind: this.kind, id: this.id },
      });
      return snapshot?.status ?? "failed";
    }

    async done() {
      return (await this.status()) !== "running";
    }

    async wait(opts, ...rest) {
      if (this._result !== HANDLE_UNSET) return this._result;
      const options = optionsArg("AgentHandle.wait", opts, rest, ["timeout"], "{ timeout }");
      return (await wait([this], options))[0];
    }

    async cancel() {
      const result = await hostCall("__cancel__", {
        item: { kind: this.kind, id: this.id },
      });
      return Boolean(result?.cancelled);
    }
  }

  class AgentHandle extends EvalHandle {
    constructor(id, agentName, schema) {
      super("agent", id, schema);
      this.agent = agentName;
      this.handle = `agent://${id}`;
    }

    toString() {
      return `<agent ${this.id} (${this.agent})>`;
    }

    async send(message) {
      return await hostCall("write", {
        path: this.handle,
        content: String(message),
        i: "Messaging agent",
      });
    }

    async output(opts = {}) {
      return await globalThis.output(this.id, opts);
    }
  }

  class CompletionHandle extends EvalHandle {
    constructor(id, schema) {
      super("completion", id, schema);
    }

    toString() {
      return `<completion ${this.id}>`;
    }
  }

  // Immediate-return wrapper for the async `completion()`/`agent()` factories.
  // JS cannot hand back a handle synchronously — the bridge allocates the id
  // over an async round-trip — so the factory returns this thenable instead:
  // `await` it to get the resolved handle (for dag wiring via `.id`/`.handle`),
  // or call the handle methods directly (`h.wait()`, `h.status()`, …) and they
  // await resolution internally.
  class PendingHandle {
    constructor(promise) {
      this._promise = promise;
    }

    then(onFulfilled, onRejected) {
      return this._promise.then(onFulfilled, onRejected);
    }

    catch(onRejected) {
      return this._promise.catch(onRejected);
    }

    finally(onFinally) {
      return this._promise.finally(onFinally);
    }

    async status() {
      return (await this._promise).status();
    }

    async done() {
      return (await this._promise).done();
    }

    async wait(...args) {
      return (await this._promise).wait(...args);
    }

    async cancel() {
      return (await this._promise).cancel();
    }

    async send(message) {
      return (await this._promise).send(message);
    }

    async output(options) {
      return (await this._promise).output(options);
    }
  }

  const resolveHandleSnapshot = (handle, snapshot) => {
    const status = snapshot?.status ?? "failed";
    if (status === "running") throw timeoutError(`${handle.kind} handle ${handle.id} is still running`);
    if (status === "failed" || status === "cancelled") {
      throw new Error(snapshot?.error || `${handle.kind} handle ${handle.id} failed`);
    }
    const value = hasOwn(snapshot ?? {}, "data")
      ? snapshot.data
      : handle._schema !== undefined
        ? JSON.parse(snapshot?.text ?? "")
        : (snapshot?.text ?? "");
    handle._result = value;
    return value;
  };

  const wait = async (handles, opts, ...rest) => {
    const { timeout, raiseErrors = true } = optionsArg("wait", opts, rest, ["timeout", "raiseErrors"], "{ timeout, raiseErrors }");
    const isHandleLike = value => value instanceof EvalHandle || value instanceof PendingHandle;
    const raw = isHandleLike(handles) ? [handles] : Array.from(handles ?? []);
    const items = await Promise.all(raw);
    for (const handle of items) {
      if (!(handle instanceof EvalHandle)) throw new TypeError("wait() expects agent or completion handles");
    }
    const results = new Array(items.length);
    const pending = [];
    const pendingIndexes = [];
    for (let index = 0; index < items.length; index++) {
      const handle = items[index];
      if (handle._result === HANDLE_UNSET) {
        pending.push({ kind: handle.kind, id: handle.id });
        pendingIndexes.push(index);
      } else {
        results[index] = handle._result;
      }
    }
    if (pending.length > 0) {
      const args = { items: pending };
      if (timeout !== undefined) args.timeoutMs = Math.max(0, Number(timeout) * 1000);
      const response = await hostCall("__wait__", args);
      const snapshots = Array.isArray(response?.items) ? response.items : [];
      if (snapshots.length !== pending.length) throw new Error("wait() returned an incomplete handle result");
      for (let offset = 0; offset < pendingIndexes.length; offset++) {
        const index = pendingIndexes[offset];
        try {
          results[index] = resolveHandleSnapshot(items[index], snapshots[offset]);
        } catch (error) {
          if (error?.name === "TimeoutError") throw error;
          results[index] = error instanceof Error ? error : new Error(String(error));
        }
      }
    }
    if (raiseErrors) {
      const failure = results.find(value => value instanceof Error);
      if (failure) throw failure;
    }
    return results;
  };

  const completion = (prompt, opts, ...rest) => {
    const promise = (async () => {
      const options = optionsArg("completion", opts, rest, ["model", "system", "schema"], "{ model, system, schema }");
      const result = await hostCall("__completion__", { prompt, ...options });
      if (!result || typeof result.id !== "string") throw new Error("completion() did not return a handle");
      return new CompletionHandle(result.id, options.schema);
    })();
    return new PendingHandle(promise);
  };

  const checkQuestions = questions => {
    if (questions === null || typeof questions !== "object" || Array.isArray(questions)) {
      throw new TypeError("judge(state, questions) expects questions as an object keyed by id");
    }
  };

  const judge = async (state, questions) => {
    checkQuestions(questions);
    const result = await hostCall("__judge__", { state, questions });
    if (!result || !isPlainObject(result.answers)) throw new Error("judge() did not return answers");
    return result.answers;
  };

  // One settled judgeBatch item: `answers` on success, else `error`.
  class JudgmentItem {
    constructor(item) {
      this.key = item.key;
      this.answers = item.answers;
      this.error = item.error;
      this.model = item.model;
    }

    get ok() {
      return this.error === undefined;
    }

    toString() {
      return this.error === undefined
        ? `<judgment ${JSON.stringify(this.key)} ${JSON.stringify(this.answers)}>`
        : `<judgment ${JSON.stringify(this.key)} error=${JSON.stringify(this.error)}>`;
    }
  }

  // Host-owned bulk judgment run. Pull settled items with `drain()` across as
  // many cells as needed; the run outlives any single cell.
  class JudgmentBatch {
    constructor(id, total, intent) {
      this.id = id;
      this.total = total;
      this.intent = intent;
    }

    async _call(op, args = {}) {
      return await hostCall("__judge_batch__", { op, id: this.id, ...args });
    }

    async status() {
      return await this._call("status");
    }

    // Items settled since the last drain as `[[key, JudgmentItem]]`; waits up
    // to `timeout` seconds for at least one, else returns `[]`.
    async drain(opts, ...rest) {
      const { timeout } = optionsArg("JudgmentBatch.drain", opts, rest, ["timeout"], "{ timeout }");
      const args = timeout === undefined ? {} : { timeoutMs: Math.max(0, Number(timeout) * 1000) };
      const result = await this._call("drain", args);
      const items = Array.isArray(result?.items) ? result.items : [];
      return items.map(item => [item.key, new JudgmentItem(item)]);
    }

    // Yield `[key, JudgmentItem]` as items settle until `timeout` seconds
    // elapse or the run completes.
    async *drainIter(opts, ...rest) {
      const { timeout } = optionsArg("JudgmentBatch.drainIter", opts, rest, ["timeout"], "{ timeout }");
      const deadline = timeout === undefined ? undefined : Date.now() + Number(timeout) * 1000;
      while (true) {
        const remaining = deadline === undefined ? undefined : Math.max(0, (deadline - Date.now()) / 1000);
        const items = await this.drain({ timeout: remaining });
        if (items.length === 0) return;
        yield* items;
      }
    }

    async results() {
      return (await this._call("results"))?.results ?? {};
    }

    async failed() {
      return (await this._call("failed"))?.failed ?? {};
    }

    async cancel() {
      return Boolean((await this._call("cancel"))?.cancelled);
    }

    async close() {
      await this._call("close");
    }

    toString() {
      return `<judgeBatch ${this.id} total=${this.total}>`;
    }
  }

  const judgeBatchFrom = result => {
    if (!result || typeof result.id !== "string") throw new Error("judgeBatch() did not return a batch");
    return new JudgmentBatch(result.id, Number(result.total) || 0, result.intent ?? "Judging");
  };

  // Judge every state with the same questions on the host. `states` is
  // `{ key: state }` or an array (keys are indices). `intent` is an optional
  // nonempty progress/job label. Item failures land in `JudgmentItem.error`;
  // only a run that dies wholesale rejects `drain()`.
  const judgeBatch = async (states, questions, opts, ...rest) => {
    checkQuestions(questions);
    const options = optionsArg(
      "judgeBatch",
      opts,
      rest,
      ["concurrency", "retries", "minOk", "intent"],
      "{ intent, concurrency, retries, minOk }",
    );
    if (
      options.intent !== undefined &&
      (typeof options.intent !== "string" || options.intent.trim().length === 0)
    ) {
      throw new TypeError("judgeBatch() intent must be a non-empty string");
    }
    let items;
    if (Array.isArray(states)) items = states.map((state, key) => ({ key, state }));
    else if (isPlainObject(states)) items = Object.entries(states).map(([key, state]) => ({ key, state }));
    else throw new TypeError("judgeBatch(states, questions) expects states as an object or array");
    return judgeBatchFrom(
      await hostCall("__judge_batch__", { op: "create", items, questions, ...options }),
    );
  };

  // Re-create a batch ref by id (after a VM reset or from another runtime).
  judgeBatch.attach = async id =>
    judgeBatchFrom(await hostCall("__judge_batch__", { op: "attach", id: String(id) }));

  const agent = (prompt, opts, ...rest) => {
    const promise = (async () => {
      const options = optionsArg(
        "agent",
        opts,
        rest,
        ["agent", "label", "schema", "isolated", "apply", "merge", "schemaMode", "tools"],
        "{ agent, label, schema, schemaMode, isolated, apply, merge, tools }",
      );
      const result = await hostCall("__agent__", { prompt, ...options });
      if (!result || typeof result.id !== "string") throw new Error("agent() did not return a handle");
      return new AgentHandle(result.id, result.agent, options.schema);
    })();
    return new PendingHandle(promise);
  };

  class WorkPool {
    constructor(name, agentName, limit) {
      this.name = name;
      this.agent = agentName;
      this.limit = limit;
    }

    async push(...items) {
      if (!items.every(item => typeof item === "string")) {
        throw new TypeError("WorkPool.push() expects string items");
      }
      const result = await hostCall("__workpool__", {
        op: "push",
        name: this.name,
        items,
      });
      return result?.ids ?? [];
    }

    async status() {
      return await hostCall("__workpool__", {
        op: "status",
        name: this.name,
      });
    }

    async peek() {
      return await hostCall("__workpool__", {
        op: "peek",
        name: this.name,
      });
    }

    async close() {
      return await hostCall("__workpool__", {
        op: "close",
        name: this.name,
      });
    }

    toString() {
      return `<workpool ${this.name} (${this.agent}) ${this.limit} agents>`;
    }
  }

  const workpool = async (agentName, opts, ...rest) => {
    if (isPlainObject(agentName) && isNil(opts) && rest.every(isNil)) {
      opts = agentName;
      agentName = undefined;
    }
    const options = optionsArg(
      "workpool",
      opts,
      rest,
      ["name", "context", "tools"],
      "{ name, context, tools }",
    );
    const result = await hostCall("__workpool__", {
      op: "create",
      ...(agentName === undefined ? {} : { agent: agentName }),
      ...options,
    });
    if (!result || typeof result.name !== "string") throw new Error("workpool() did not return a pool");
    return new WorkPool(result.name, result.agent, result.limit);
  };

  const budgetSnap = async () => {
    const snapshot = await hostCall("__budget__", {});
    return snapshot && typeof snapshot === "object" ? snapshot : {};
  };

  const budget = {
    total: async () => {
      const snapshot = await budgetSnap();
      return snapshot.total ?? null;
    },
    spent: async () => Number((await budgetSnap()).spent ?? 0),
    remaining: async () => {
      const snapshot = await budgetSnap();
      return snapshot.total == null ? Infinity : Math.max(0, Number(snapshot.total) - Number(snapshot.spent ?? 0));
    },
    hard: async () => Boolean((await budgetSnap()).hard),
  };

  // RLM sub-call: `llm_query(prompt, { context, model, depth, frame, pointers, k, select, on_low,
  // low, escalate_model })` -> text. Default model is the session's smol role; memoized and
  // budgeted host-side (errors reject the promise).
  // Frame (TTIC): the sub-call sees only prompt + context + pointers, never the parent
  // transcript (`frame: "only"` is the default and the only accepted value).
  // SRLM sampling: `k > 1` or `select` ('consistency' | 'uncertainty') draws k memo-aware,
  // budgeted samples and resolves to a String object (use `String(x)` / `+ ""` for a primitive)
  // carrying `.agreement`, `.candidates`, `.k`, `.low`, `.expanded`, `.escalated`, `.raw`;
  // `on_low` ('return' | 'expand' | 'escalate') applies when agreement < `low` (default 0.5).
  const llm_query = async (prompt, opts, ...rest) => {
    const options = { ...optionsArg(
      "llm_query",
      opts,
      rest,
      ["context", "model", "depth", "frame", "pointers", "k", "select", "on_low", "low", "escalate_model"],
      "{ context, model, depth, frame, pointers, k, select, on_low, low, escalate_model }",
    ) };
    const sampled = (options.k !== undefined && Number(options.k) !== 1) || options.select != null;
    if (options.frame === "only") delete options.frame;
    const result = await hostCall("__llm_query__", { prompt, ...options });
    if (!result || typeof result.text !== "string") throw new Error("llm_query() did not return text");
    if (!sampled) return result.text;
    return Object.assign(new String(result.text), {
      agreement: result.agreement,
      candidates: result.candidates ?? [],
      k: result.k,
      low: Boolean(result.low),
      expanded: Boolean(result.expanded),
      escalated: result.escalated ?? null,
      raw: result,
    });
  };
  const llm_stats = async () => await hostCall("__rlm_stats__", {});
  const rlm_context = async () => await hostCall("__rlm_context__", {});

  Object.assign(globalThis, {
    llm_query,
    llm_stats,
    rlm_context,
    completion,
    agent,
    wait,
    AgentHandle,
    CompletionHandle,
    PendingHandle,
    judge,
    judgeBatch,
    JudgmentBatch,
    JudgmentItem,
    workpool,
    WorkPool,
    budget,
  });
})();
