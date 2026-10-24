"""Eval agent/completion/judgment/workpool/budget prelude helpers (Python).

Ported from vendor-src/oh-my-pi/packages/coding-agent/src/eval/py/prelude.py,
transport section only: every host round-trip goes through the runner-injected
``__omp_host_call(op, args)`` host-op protocol (op names and payload shapes are
verbatim from agent-bridge.ts, completion-bridge.ts, judgment-bridge.ts,
judgment-batch-bridge.ts, workpool-bridge.ts and handle-bridge.ts). I/O
helpers (read/write/output), ``display`` and the ``tool`` proxy live in
prelude.py, which is loaded before this file; this module only consumes them.
"""

if "__omp_agents_prelude_loaded__" not in globals():
    __omp_agents_prelude_loaded__ = True

    import asyncio
    import json
    import math
    import time

    _HANDLE_UNSET = object()

    class _Handle:
        """Shared process-local agent/completion handle behavior."""

        __slots__ = ("id", "_schema", "_result")

        kind = ""

        def __init__(self, id, schema=None):
            self.id = id
            self._schema = schema
            self._result = _HANDLE_UNSET

        @property
        def status(self):
            snapshot = _bridge_call(
                "__status__",
                {"item": {"kind": self.kind, "id": self.id}},
            )
            return snapshot.get("status") if isinstance(snapshot, dict) else "failed"

        def done(self):
            return self.status != "running"

        def wait(self, timeout=None):
            if self._result is not _HANDLE_UNSET:
                return self._result
            return wait([self], timeout=timeout)[0]

        def cancel(self):
            result = _bridge_call(
                "__cancel__",
                {"item": {"kind": self.kind, "id": self.id}},
            )
            return bool(result.get("cancelled")) if isinstance(result, dict) else False

        def __await__(self):
            # `to_thread` copies the cell's contextvars into the worker; a bare
            # `run_in_executor` drops them, so the bridge loses its run identity.
            return asyncio.to_thread(self.wait).__await__()

    class AgentHandle(_Handle):
        """Background subagent handle returned by ``agent()``."""

        __slots__ = ("agent", "handle")
        kind = "agent"

        def __init__(self, id, agent, schema=None):
            super().__init__(id, schema)
            self.agent = agent
            self.handle = f"agent://{id}"

        def __repr__(self):
            return f"<agent {self.id} ({self.agent})>"

        def send(self, message):
            return _bridge_call(
                "write",
                {
                    "path": self.handle,
                    "content": str(message),
                    "i": "Messaging agent",
                },
            )

        def output(self, **kwargs):
            read_output = globals().get("output")
            if read_output is None:
                raise RuntimeError("output() helper is not loaded (prelude.py missing)")
            return read_output(self.id, **kwargs)

    class CompletionHandle(_Handle):
        """Background one-shot completion handle returned by ``completion()``."""

        __slots__ = ()
        kind = "completion"

        def __repr__(self):
            return f"<completion {self.id}>"

    def _handle_value(handle, snapshot):
        status = snapshot.get("status") if isinstance(snapshot, dict) else "failed"
        if status == "running":
            raise TimeoutError(f"{handle.kind} handle {handle.id} is still running")
        if status in ("failed", "cancelled"):
            message = (
                snapshot.get("error")
                if isinstance(snapshot, dict)
                else f"{handle.kind} handle {handle.id} failed"
            )
            raise RuntimeError(message or f"{handle.kind} handle {handle.id} failed")
        if isinstance(snapshot, dict) and "data" in snapshot:
            value = snapshot["data"]
        else:
            text = snapshot.get("text", "") if isinstance(snapshot, dict) else ""
            value = json.loads(text) if handle._schema is not None else text
        handle._result = value
        return value

    def wait(handles, timeout=None, *, raise_errors=True):
        """Wait for agent/completion handles in input order."""
        items = [handles] if isinstance(handles, _Handle) else list(handles)
        for handle in items:
            if not isinstance(handle, _Handle):
                raise TypeError("wait() expects agent or completion handles")
        results = [None] * len(items)
        pending = []
        pending_indexes = []
        for index, handle in enumerate(items):
            if handle._result is _HANDLE_UNSET:
                pending.append({"kind": handle.kind, "id": handle.id})
                pending_indexes.append(index)
            else:
                results[index] = handle._result
        if pending:
            args = {"items": pending}
            if timeout is not None:
                args["timeoutMs"] = max(0, float(timeout) * 1000)
            response = _bridge_call("__wait__", args)
            snapshots = response.get("items", []) if isinstance(response, dict) else []
            for index, handle, snapshot in zip(
                pending_indexes,
                (items[index] for index in pending_indexes),
                snapshots,
            ):
                try:
                    results[index] = _handle_value(handle, snapshot)
                except RuntimeError as error:
                    results[index] = error
            if len(snapshots) != len(pending):
                raise RuntimeError("wait() returned an incomplete handle result")
        if raise_errors:
            for result in results:
                if isinstance(result, RuntimeError):
                    raise result
        return results

    def completion(prompt, *, model="default", system=None, schema=None):
        """Start a stateless completion and return its handle."""
        args = {"prompt": prompt, "model": model}
        if system is not None:
            args["system"] = system
        if schema is not None:
            args["schema"] = schema
        result = _bridge_call("__completion__", args)
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            raise RuntimeError("completion() did not return a handle")
        return CompletionHandle(result["id"], schema)

    def _check_questions(questions):
        if not isinstance(questions, dict):
            raise TypeError("judge(state, questions) expects questions as a dict keyed by id")

    async def judge(state, questions):
        """Answer typed questions over one ``state``; returns ``{id: answer}`` (choice/bool/score)."""
        _check_questions(questions)
        result = await asyncio.to_thread(
            _bridge_call, "__judge__", {"state": state, "questions": questions}
        )
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise RuntimeError("judge() did not return answers")
        return result["answers"]

    class JudgmentItem:
        """One settled ``judge_batch`` item: ``answers`` on success, else ``error``."""

        __slots__ = ("key", "answers", "error", "model")

        def __init__(self, item):
            self.key = item.get("key")
            self.answers = item.get("answers")
            self.error = item.get("error")
            self.model = item.get("model")

        @property
        def ok(self):
            return self.error is None

        def __repr__(self):
            if self.error is not None:
                return f"<judgment {self.key!r} error={self.error!r}>"
            return f"<judgment {self.key!r} {self.answers!r}>"

    class JudgmentBatch:
        """Host-owned bulk judgment run. Pull settled items with ``await drain()`` across as many cells as needed."""

        __slots__ = ("id", "total", "intent")

        def __init__(self, id, total, intent):
            self.id = id
            self.total = total
            self.intent = intent

        def _call(self, op, **args):
            return _bridge_call("__judge_batch__", {"op": op, "id": self.id, **args})

        def status(self):
            """Snapshot: ``intent``, ``done``, ``total``, ``failed``, ``running``, ``model``, ``elapsedS``."""
            return self._call("status")

        async def drain(self, timeout=None):
            """Items settled since the last drain as ``[(key, JudgmentItem)]``; waits up to ``timeout`` seconds for at least one, else ``[]``."""
            args = {}
            if timeout is not None:
                args["timeoutMs"] = max(0, float(timeout) * 1000)
            result = await asyncio.to_thread(self._call, "drain", **args)
            items = result.get("items", []) if isinstance(result, dict) else []
            return [(item.get("key"), JudgmentItem(item)) for item in items]

        async def drain_iter(self, timeout=None):
            """Yield ``(key, JudgmentItem)`` as items settle until ``timeout`` seconds elapse or the run completes."""
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            while True:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                items = await self.drain(remaining)
                if not items:
                    return
                for entry in items:
                    yield entry

        def results(self):
            """``{key: answers}`` for every item judged so far."""
            result = self._call("results")
            return result.get("results", {}) if isinstance(result, dict) else {}

        def failed(self):
            """``{key: error}`` for every item that failed so far."""
            result = self._call("failed")
            return result.get("failed", {}) if isinstance(result, dict) else {}

        def cancel(self):
            result = self._call("cancel")
            return bool(result.get("cancelled")) if isinstance(result, dict) else False

        def close(self):
            """Cancel if running and release the host-side run."""
            self._call("close")
            return None

        def __repr__(self):
            return f"<judge_batch {self.id} total={self.total}>"

    def _judge_batch_from(result):
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            raise RuntimeError("judge_batch() did not return a batch")
        return JudgmentBatch(
            result["id"],
            int(result.get("total") or 0),
            result.get("intent") or "Judging",
        )

    def judge_batch(
        states,
        questions,
        *,
        concurrency=None,
        retries=None,
        min_ok=None,
        intent=None,
    ):
        """Judge every state with the same ``questions`` on the host; returns a ``JudgmentBatch`` to drain across cells.

        ``states`` is ``{key: state}`` or a list (keys are indices). ``intent`` is an
        optional nonempty progress/job label. Item failures land in ``JudgmentItem.error``;
        only a run that dies wholesale raises from ``drain()``.
        """
        _check_questions(questions)
        if intent is not None and (
            not isinstance(intent, str) or not intent.strip()
        ):
            raise TypeError("judge_batch() intent must be a non-empty string")
        if isinstance(states, dict):
            items = [{"key": key, "state": state} for key, state in states.items()]
        elif isinstance(states, (list, tuple)):
            items = [{"key": index, "state": state} for index, state in enumerate(states)]
        else:
            raise TypeError("judge_batch(states, questions) expects states as a dict or list")
        args = {"op": "create", "items": items, "questions": questions}
        if concurrency is not None:
            args["concurrency"] = int(concurrency)
        if retries is not None:
            args["retries"] = int(retries)
        if min_ok is not None:
            args["minOk"] = int(min_ok)
        if intent is not None:
            args["intent"] = intent
        return _judge_batch_from(_bridge_call("__judge_batch__", args))

    def _attach_judge_batch(id):
        """Re-create a ``JudgmentBatch`` ref by id (after a kernel reset or from another runtime)."""
        return _judge_batch_from(_bridge_call("__judge_batch__", {"op": "attach", "id": str(id)}))

    judge_batch.attach = _attach_judge_batch

    def agent(
        prompt,
        *,
        agent=None,
        label=None,
        schema=None,
        schema_mode=None,
        isolated=None,
        apply=None,
        merge=None,
        tools=None,
    ):
        """Start a background subagent and return its handle."""
        args = {"prompt": prompt}
        if agent is not None:
            args["agent"] = agent
        if label is not None:
            args["label"] = label
        if schema is not None:
            args["schema"] = schema
        if schema_mode is not None:
            args["schemaMode"] = schema_mode
        if isolated is not None:
            args["isolated"] = bool(isolated)
        if apply is not None:
            args["apply"] = bool(apply)
        if merge is not None:
            args["merge"] = bool(merge)
        if tools is not None:
            args["tools"] = list(tools)
        result = _bridge_call("__agent__", args)
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            raise RuntimeError("agent() did not return a handle")
        return AgentHandle(result["id"], result.get("agent"), schema)

    class WorkPool:
        """Pool of keep-alive subagents fed through the host workpool bridge."""

        __slots__ = ("name", "agent", "limit")

        def __init__(self, name, agent, limit):
            self.name = name
            self.agent = agent
            self.limit = limit

        def push(self, *items):
            if not all(isinstance(item, str) for item in items):
                raise TypeError("WorkPool.push() expects string items")
            result = _bridge_call(
                "__workpool__",
                {"op": "push", "name": self.name, "items": list(items)},
            )
            return result.get("ids", []) if isinstance(result, dict) else []

        def status(self):
            return _bridge_call(
                "__workpool__",
                {"op": "status", "name": self.name},
            )

        def peek(self):
            return _bridge_call(
                "__workpool__",
                {"op": "peek", "name": self.name},
            )

        def close(self):
            return _bridge_call(
                "__workpool__",
                {"op": "close", "name": self.name},
            )

        def __repr__(self):
            return f"<workpool {self.name} ({self.agent}) {self.limit} agents>"

    def workpool(agent=None, *, name=None, context=None, tools=None):
        """Create a pool of keep-alive subagents."""
        args = {"op": "create"}
        if agent is not None:
            args["agent"] = agent
        if name is not None:
            args["name"] = name
        if context is not None:
            args["context"] = context
        if tools is not None:
            args["tools"] = list(tools)
        result = _bridge_call("__workpool__", args)
        if not isinstance(result, dict) or not isinstance(result.get("name"), str):
            raise RuntimeError("workpool() did not return a pool")
        return WorkPool(result["name"], result.get("agent"), result.get("limit"))

    class _Budget:
        """Live view of the host Goal Mode token budget via the host bridge."""

        @property
        def total(self):
            snap = _bridge_call("__budget__", {})
            return (snap or {}).get("total")

        @property
        def hard(self):
            snap = _bridge_call("__budget__", {})
            return bool((snap or {}).get("hard"))

        def spent(self):
            snap = _bridge_call("__budget__", {})
            return int((snap or {}).get("spent") or 0)

        def remaining(self):
            snap = _bridge_call("__budget__", {}) or {}
            total = snap.get("total")
            if total is None:
                return math.inf
            return max(0, total - int(snap.get("spent") or 0))

        def __repr__(self):
            try:
                snap = _bridge_call("__budget__", {}) or {}
                return f"<budget total={snap.get('total')} spent={snap.get('spent')}>"
            except Exception:
                return "<budget unavailable>"

    budget = _Budget()

    class _LLMAnswer(str):
        """`llm_query(..., k=K)` result: the selected answer (a `str`) plus the SRLM sampling metadata.

        `.agreement` (plurality share of the K samples), `.candidates` (per-sample dicts: text,
        confidence, score, cached, tokens), `.k` (samples actually drawn), `.low` (agreement below
        the `low=` threshold after `on_low`), `.expanded`, `.escalated` (None or dict), `.raw`."""

    def llm_query(prompt, context=None, model=None, *, depth=1, frame="only", pointers=None,
                  k=1, select=None, on_low=None, low=None, escalate_model=None):
        """RLM sub-call: ask a cheap sub-LLM `prompt` (optionally over `context`); returns text.

        Default model is the session's smol role. Memoized by (model, context, prompt)
        and charged against hard max-depth/calls/tokens budgets (error in the cell when
        exceeded). Accounting: `llm_stats()`.

        Frame (TTIC): the sub-call sees ONLY its active frame -- `prompt`, the explicit `context`
        and optional `pointers` (list of path/id strings listed in a <pointers> block) -- never
        the parent transcript. `frame="only"` is the default and the only accepted value.

        SRLM sampling: `k=K` (>1) or `select=` draws K samples (memo-aware, budget-counted),
        groups them by normalized answer and returns the plurality answer as a str subclass with
        `.agreement` / `.candidates`. `select='consistency'` (default) takes the earliest member
        of the plurality group; `select='uncertainty'` asks for a verbalized confidence and takes
        the member maximizing ln(confidence/100) * reply_tokens (<= 0, best closest to 0).
        If agreement < `low` (default 0.5): `on_low='return'` (default) flags `.low`,
        `'expand'` draws K more samples and reselects, `'escalate'` asks `escalate_model`
        (default the `default` tier) once and returns that answer."""
        args = {"prompt": prompt, "depth": depth}
        if context is not None:
            args["context"] = context
        if model is not None:
            args["model"] = model
        if frame != "only":
            args["frame"] = frame
        if pointers:
            args["pointers"] = [str(p) for p in pointers]
        sampled = k != 1 or select is not None
        if sampled:
            args["k"] = k
        for key, val in (("select", select), ("on_low", on_low), ("low", low),
                         ("escalate_model", escalate_model)):
            if val is not None:
                args[key] = val
        result = _bridge_call("__llm_query__", args)
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise RuntimeError("llm_query() did not return text")
        if not sampled:
            return result["text"]
        out = _LLMAnswer(result["text"])
        out.agreement = result.get("agreement")
        out.candidates = result.get("candidates") or []
        out.k = result.get("k")
        out.low = bool(result.get("low"))
        out.expanded = bool(result.get("expanded"))
        out.escalated = result.get("escalated")
        out.raw = result
        return out

    def llm_map(prompt, text, chunk_lines=150, model=None):
        """Engine-run map: split `text` into chunks of `chunk_lines` lines and run
        `llm_query(prompt, context=chunk)` on every chunk; returns the list of replies in order."""
        lines = str(text).splitlines()
        chunks = ["\n".join(lines[i:i + chunk_lines]) for i in range(0, len(lines), chunk_lines)]
        return [llm_query(prompt, context=c, model=model) for c in chunks]

    def llm_stats():
        """Sub-call accounting: calls, memo_hits, input/output/total tokens, limits, per-call records."""
        return _bridge_call("__rlm_stats__", {})

    def rlm_context():
        """The context loaded by `/rlm` (None when none)."""
        return _bridge_call("__rlm_context__", {})
