from __future__ import annotations

# OMP prelude helpers (loaded once into the runner namespace).
# Transport: host ops go through the runner-injected `__omp_host_call(op, args)`
# (raises OmpHostError) and `__omp_host_notify(op, args)`; display frames go
# through the runner-injected `__omp_display`.
if "__omp_prelude_loaded__" not in globals():
    __omp_prelude_loaded__ = True
    from pathlib import Path
    import asyncio, collections.abc, contextvars, inspect, os, json, math, re, time, types, typing
    from urllib.parse import unquote


    # __omp_display is injected by runner.py before the prelude executes; it
    # mirrors IPython's display() semantics with the same MIME bundle output.
    _omp_display = __omp_display  # type: ignore[name-defined]

    _PRESENTABLE_REPRS = (
        "_repr_mimebundle_",
        "_repr_html_",
        "_repr_json_",
        "_repr_markdown_",
        "_repr_png_",
        "_repr_jpeg_",
        "_repr_svg_",
        "_repr_latex_",
    )

    def display(value):
        """Render a value. Falls back to a JSON+text/plain bundle for plain dict/list/tuple."""
        if isinstance(value, dict) and value.get("type") == "image":
            _omp_display(value)
            return
        if any(hasattr(value, attr) for attr in _PRESENTABLE_REPRS):
            _omp_display(value)
            return
        if isinstance(value, (dict, list, tuple)):
            try:
                bundle = {"application/json": value, "text/plain": repr(value)}
                _omp_display(bundle, raw=True)
                return
            except Exception:
                pass
        _omp_display(value)

    def _emit_status(op: str, **data):
        """Emit structured status event for TUI rendering."""
        _omp_display({"application/x-omp-status": {"op": op, **data}}, raw=True)

    def env(key: str | None = None, value: str | None = None):
        """Get/set environment variables."""
        if key is None:
            items = dict(sorted(os.environ.items()))
            _emit_status("env", count=len(items), keys=list(items.keys())[:20])
            return items
        if value is not None:
            os.environ[key] = value
            _emit_status("env", key=key, value=value, action="set")
            return value
        val = os.environ.get(key)
        _emit_status("env", key=key, value=val, action="get")
        return val

    _OMP_INTERNAL_URL_RE = re.compile(r"^([a-z][a-z0-9+.-]*)://(.*)$", re.IGNORECASE)
    _OMP_ARTIFACT_ID_RE = re.compile(r"^\d+$")
    _OMP_BARE_ARTIFACT_RE = re.compile(r"^artifact://\d+$", re.IGNORECASE)

    def _omp_url_roots() -> dict:
        """On-disk roots for internal-URL schemes, keyed by scheme (PI_EVAL_LOCAL_ROOTS)."""
        try:
            roots = json.loads(os.environ.get("PI_EVAL_LOCAL_ROOTS") or "{}")
        except (ValueError, TypeError):
            return {}
        return roots if isinstance(roots, dict) else {}

    def _should_delegate_read(path: str | Path) -> bool:
        """Delegate `scheme://` reads to the read tool unless the scheme has an injected root.

        `artifact://<id>` reads its file directly; selector forms
        (`artifact://3:raw:5-9`) stay with the read tool, which parses them."""
        if not isinstance(path, str):
            return False
        match = _OMP_INTERNAL_URL_RE.match(path)
        if match is None:
            return False
        scheme = match.group(1).lower()
        if scheme not in _omp_url_roots():
            return True
        return scheme == "artifact" and not _OMP_ARTIFACT_ID_RE.match(match.group(2))

    def _read_line_selector(offset: int, limit: int | None) -> str | None:
        if offset <= 1 and limit is None:
            return None
        start = max(1, offset)
        if limit is None:
            return f"{start}-"
        return f"{start}-{start + limit - 1}"

    def _read_tool_text(path: str) -> str:
        result = _bridge_call("read", {"path": path})
        if isinstance(result, dict) and "text" in result:
            return result["text"]
        return result

    def _artifact_file(path: str) -> Path | None:
        """This session's file for `artifact://<id>` (`<id>.<tool>.log` in its
        artifacts dir), or None when the id is not there or no dir is known."""
        root = _omp_url_roots().get("artifact")
        if not root:
            return None
        artifact_id = path[11:]
        try:
            names = os.listdir(root)
        except FileNotFoundError:
            return None
        for name in names:
            if name.startswith(f"{artifact_id}."):
                return Path(os.path.join(os.path.abspath(root), name))
        return None

    def _read_through_tool(path: str, offset: int, limit: int | None) -> str:
        if limit is not None and limit <= 0:
            return ""
        selector = _read_line_selector(offset, limit)
        return _read_tool_text(path if selector is None else f"{path}:{selector}")

    def _resolve_omp_path(path: str | Path) -> Path:
        """Map a helper path to a real filesystem Path.

        A `scheme://…` whose scheme has an injected on-disk root (via
        PI_EVAL_LOCAL_ROOTS) is rewritten under that root so it lands where
        `read scheme://…` resolves — not a literal `scheme:/` directory under
        the cwd (which `Path("scheme://x")` collapses to). Plain paths pass
        through unchanged; any other `scheme://` is rejected, as is
        `artifact://`, whose root is a lookup dir for reads, not a path prefix."""
        if not isinstance(path, str):
            return Path(path)
        match = _OMP_INTERNAL_URL_RE.match(path)
        if not match:
            return Path(path)
        scheme = match.group(1).lower()
        root = _omp_url_roots().get(scheme)
        if not root or scheme == "artifact":
            raise ValueError(f"Protocol paths are not supported by this helper: {path}")
        relative = unquote(match.group(2).replace("\\", "/"))
        # Mirror the host `path.resolve`/`resolveLocalUrlToPath`: normalize and
        # make absolute WITHOUT realpath'ing symlinks (Path.resolve would turn
        # /tmp into /private/tmp and diverge from the read-side resolution).
        root_path = os.path.abspath(root)
        if relative == "":
            return Path(root_path)
        rel_path = Path(relative)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"Unsafe {scheme}:// path (absolute or traversal): {path}")
        resolved = os.path.abspath(os.path.join(root_path, relative))
        if resolved != root_path and not resolved.startswith(root_path + os.sep):
            raise ValueError(f"{scheme}:// path escapes its root: {path}")
        return Path(resolved)

    def read(path: str | Path, offset: int = 1, limit: int | None = None) -> str:
        """Read file or read-tool URI contents. offset/limit are 1-indexed lines."""
        if isinstance(path, str) and _OMP_BARE_ARTIFACT_RE.match(path):
            if limit is not None and limit <= 0:
                return ""
            p = _artifact_file(path)
            if p is None:
                # Another session's, or no artifacts dir was injected: the read tool searches every registered one.
                return _read_through_tool(f"{path}:raw", offset, limit)
        elif _should_delegate_read(path):
            return _read_through_tool(path, offset, limit)
        else:
            p = _resolve_omp_path(path)
        data = p.read_text(encoding="utf-8")
        lines = data.splitlines(keepends=True)
        if offset > 1 or limit is not None:
            start = max(0, offset - 1)
            end = start + limit if limit else len(lines)
            lines = lines[start:end]
            data = "".join(lines)
        preview = data[:500]
        _emit_status("read", path=str(p), chars=len(data), preview=preview)
        return data

    def write(path: str | Path, content: str) -> Path:
        """Write file contents (create parents)."""
        p = _resolve_omp_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        _emit_status("write", path=str(p), chars=len(content))
        return p

    def output(
        *ids: str,
        format: str = "raw",
        query: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str | dict | list[dict]:
        """Read task/agent output by ID. Returns text or JSON depending on format.

        Args:
            *ids: Output IDs to read (e.g., 'scout_0', 'reviewer_1')
            format: 'raw' (default), 'json' (dict with metadata), 'stripped' (no ANSI)
            query: jq-like query for JSON outputs (e.g., '.endpoints[0].file')
            offset: Line number to start reading from (1-indexed)
            limit: Maximum number of lines to read

        Returns:
            Single ID: str (format='raw'/'stripped') or dict (format='json')
            Multiple IDs: list of dict with 'id' and 'content'/'data' keys

        Examples:
            output('scout_0')  # Read as raw text
            output('reviewer_0', format='json')  # Read with metadata
            output('scout_0', query='.files[0]')  # Extract JSON field
            output('scout_0', offset=10, limit=20)  # Lines 10-29
            output('scout_0', 'reviewer_1')  # Read multiple outputs
        """
        # Prefer PI_ARTIFACTS_DIR so subagents resolve through the parent's
        # shared artifacts dir; fall back to deriving from PI_SESSION_FILE
        # for legacy callers / top-level sessions where the two coincide.
        artifacts_dir = os.environ.get("PI_ARTIFACTS_DIR")
        if not artifacts_dir:
            session_file = os.environ.get("PI_SESSION_FILE")
            if not session_file:
                _emit_status("output", error="No session file available")
                raise RuntimeError("No session - output artifacts unavailable")
            artifacts_dir = session_file.rsplit(".", 1)[0]  # Strip .jsonl extension
        if not Path(artifacts_dir).exists():
            _emit_status(
                "output", error="Artifacts directory not found", path=artifacts_dir
            )
            raise RuntimeError(f"No artifacts directory found: {artifacts_dir}")

        if not ids:
            raise ValueError("At least one output ID is required")
        if query and (offset is not None or limit is not None):
            raise ValueError("query cannot be combined with offset/limit")

        results: list[dict] = []
        not_found: list[str] = []

        for output_id in ids:
            output_path = Path(artifacts_dir) / f"{output_id}.md"
            if not output_path.exists():
                not_found.append(output_id)
                continue

            raw_content = output_path.read_text(encoding="utf-8")
            raw_lines = raw_content.splitlines()
            total_lines = len(raw_lines)

            selected_content = raw_content
            range_info: dict | None = None

            # Handle query
            if query:
                try:
                    json_value = json.loads(raw_content)
                except json.JSONDecodeError as e:
                    _emit_status("output", id=output_id, error=f"Not valid JSON: {e}")
                    raise ValueError(f"Output {output_id} is not valid JSON: {e}")

                # Apply jq-like query
                result_value = _apply_query(json_value, query)
                try:
                    selected_content = (
                        json.dumps(result_value, indent=2)
                        if result_value is not None
                        else "null"
                    )
                except (TypeError, ValueError):
                    selected_content = str(result_value)

            # Handle offset/limit
            elif offset is not None or limit is not None:
                start_line = max(1, offset or 1)
                if start_line > total_lines:
                    _emit_status(
                        "output",
                        id=output_id,
                        error=f"Offset {start_line} beyond end ({total_lines} lines)",
                    )
                    raise ValueError(
                        f"Offset {start_line} is beyond end of output ({total_lines} lines) for {output_id}"
                    )

                effective_limit = (
                    limit if limit is not None else total_lines - start_line + 1
                )
                end_line = min(total_lines, start_line + effective_limit - 1)
                selected_lines = raw_lines[start_line - 1 : end_line]
                selected_content = "\n".join(selected_lines)
                range_info = {
                    "start_line": start_line,
                    "end_line": end_line,
                    "total_lines": total_lines,
                }

            # Strip ANSI codes if requested
            if format == "stripped":
                selected_content = re.sub(r"\x1b\[[0-9;]*m", "", selected_content)

            # Build result
            if format == "json":
                result_data = {
                    "id": output_id,
                    "path": str(output_path),
                    "line_count": total_lines
                    if not query
                    else len(selected_content.splitlines()),
                    "char_count": len(raw_content)
                    if not query
                    else len(selected_content),
                    "content": selected_content,
                }
                if range_info:
                    result_data["range"] = range_info
                if query:
                    result_data["query"] = query
                results.append(result_data)
            else:
                results.append({"id": output_id, "content": selected_content})

        # Handle not found
        if not_found:
            available = sorted([f.stem for f in Path(artifacts_dir).glob("*.md")])
            error_msg = f"Output not found: {', '.join(not_found)}"
            if available:
                error_msg += f"\n\nAvailable outputs: {', '.join(available[:20])}"
                if len(available) > 20:
                    error_msg += f" (and {len(available) - 20} more)"
            _emit_status("output", not_found=not_found, available_count=len(available))
            raise FileNotFoundError(error_msg)

        # Return format
        if len(ids) == 1:
            if format == "json":
                _emit_status("output", id=ids[0], chars=results[0]["char_count"])
                return results[0]
            _emit_status("output", id=ids[0], chars=len(results[0]["content"]))
            return results[0]["content"]

        # Multiple IDs
        if format == "json":
            total_chars = sum(r["char_count"] for r in results)
            _emit_status("output", count=len(results), total_chars=total_chars)
            return results

        combined_output: list[dict] = []
        for r in results:
            combined_output.append({"id": r["id"], "content": r["content"]})
        total_chars = sum(len(r["content"]) for r in combined_output)
        _emit_status("output", count=len(combined_output), total_chars=total_chars)
        return combined_output

    def _apply_query(data: any, query: str) -> any:
        """Apply jq-like query to data. Supports .key, [index], and chaining."""
        if not query:
            return data

        query = query.strip()
        if query.startswith("."):
            query = query[1:]
        if not query:
            return data

        # Parse query into tokens
        tokens = []
        current_token = ""
        i = 0
        while i < len(query):
            ch = query[i]
            if ch == ".":
                if current_token:
                    tokens.append(("key", current_token))
                    current_token = ""
            elif ch == "[":
                if current_token:
                    tokens.append(("key", current_token))
                    current_token = ""
                # Find matching ]
                j = i + 1
                while j < len(query) and query[j] != "]":
                    j += 1
                bracket_content = query[i + 1 : j]
                if bracket_content.startswith('"') and bracket_content.endswith('"'):
                    tokens.append(("key", bracket_content[1:-1]))
                else:
                    tokens.append(("index", int(bracket_content)))
                i = j
            else:
                current_token += ch
            i += 1
        if current_token:
            tokens.append(("key", current_token))

        # Apply tokens
        current = data
        for token_type, value in tokens:
            if token_type == "index":
                if not isinstance(current, list) or value >= len(current):
                    return None
                current = current[value]
            elif token_type == "key":
                if not isinstance(current, dict) or value not in current:
                    return None
                current = current[value]

        return current

    def _bridge_call(name: str, args: dict):
        """Invoke one host op through the runner transport and return its result.

        Errors surface as OmpHostError (a RuntimeError) carrying the host message."""
        if "__omp_host_call" not in globals():
            raise RuntimeError("tool bridge is unavailable in this kernel")
        return __omp_host_call(name, args)  # type: ignore[name-defined]

    def _surface_bridged_tool_images(value):
        """Surface bridge metadata/images without leaking opaque payloads to cell code."""
        if not isinstance(value, dict):
            return value
        images = value.get("images")
        if not isinstance(images, list) or not images:
            return value
        displayed = 0
        for image in images:
            if not isinstance(image, dict):
                continue
            data = image.get("data")
            mime_type = image.get("mimeType")
            if not isinstance(data, str) or not isinstance(mime_type, str):
                continue
            _omp_display({"application/x-omp-image": image}, raw=True)
            displayed += 1
        if displayed == 0:
            return value
        surfaced = {key: item for key, item in value.items() if key != "images"}
        suffix = "" if displayed == 1 else "s"
        surfaced["images"] = f"({displayed} image{suffix} displayed)"
        return surfaced

    async def _omp_prelude(name: str, parameters):
        """Invoke one enabled eval prelude capability through the host bridge."""
        value = await asyncio.to_thread(
            _bridge_call,
            "__prelude__",
            {"name": name, "parameters": parameters},
        )
        return _surface_bridged_tool_images(value)

    _OMP_CALL_OCCURRENCES = contextvars.ContextVar("omp_call_occurrences", default=None)

    def __omp_reset_call_occurrences__():
        _OMP_CALL_OCCURRENCES.set(None)

    async def __omp_with_call_site__(site_id: str, action, args):
        occurrences = dict(_OMP_CALL_OCCURRENCES.get() or {})
        occurrence = occurrences.get(site_id, 0)
        occurrences[site_id] = occurrence + 1
        _OMP_CALL_OCCURRENCES.set(occurrences)
        # `action` is the async `_ToolCallable.__call__`: calling it only builds
        # the coroutine, so awaiting here keeps the call site's context alive.
        result = action(args)
        if inspect.isawaitable(result):
            return await result
        return result

    class _ToolCallable:
        """Invokes one host-side tool via the runner's host-call transport."""

        __slots__ = ("_name",)

        def __init__(self, name: str):
            self._name = name

        def __repr__(self) -> str:
            return f"<tool.{self._name}>"

        async def __call__(self, args=None, /, **kwargs):
            if args is None:
                merged: dict = {}
            elif isinstance(args, dict):
                merged = dict(args)
            else:
                raise TypeError(
                    f"tool.{self._name}(...) expects a dict of arguments (got {type(args).__name__})"
                )
            merged.update(kwargs)
            value = await asyncio.to_thread(_bridge_call, self._name, merged)
            return _surface_bridged_tool_images(value)

    def _annotation_schema(annotation) -> dict:
        """Map supported Python annotations to JSON Schema."""
        if annotation is inspect.Parameter.empty or annotation is typing.Any:
            return {}

        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if origin is typing.Annotated:
            schema = _annotation_schema(args[0])
            description = next((item for item in args[1:] if isinstance(item, str)), None)
            if description is not None:
                schema = {**schema, "description": description}
            return schema
        if origin is typing.Literal:
            return {"enum": list(args)}
        if origin in (typing.Union, types.UnionType):
            non_null = [item for item in args if item is not type(None)]
            if len(non_null) == 1 and len(non_null) != len(args):
                return {
                    "anyOf": [
                        _annotation_schema(non_null[0]),
                        {"type": "null"},
                    ]
                }
            return {}

        if annotation is str:
            return {"type": "string"}
        if annotation is int:
            return {"type": "integer"}
        if annotation is float:
            return {"type": "number"}
        if annotation is bool:
            return {"type": "boolean"}

        array_origins = {
            list,
            tuple,
            set,
            collections.abc.Sequence,
        }
        if annotation in array_origins or origin in array_origins:
            schema = {"type": "array"}
            if args:
                schema["items"] = _annotation_schema(args[0])
            return schema

        object_origins = {
            dict,
            collections.abc.Mapping,
        }
        if annotation in object_origins or origin in object_origins:
            schema = {"type": "object"}
            if len(args) >= 2:
                schema["additionalProperties"] = _annotation_schema(args[1])
            return schema
        return {}

    def _tool_schema(fn) -> dict:
        """Infer one eval-defined tool's object schema from its signature."""
        signature = inspect.signature(fn)
        try:
            hints = typing.get_type_hints(fn, include_extras=True)
        except Exception:
            hints = getattr(fn, "__annotations__", {})
        properties = {}
        required = []
        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                raise TypeError("tool parameters must be keyword-capable")
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            schema = _annotation_schema(hints.get(parameter.name, parameter.annotation))
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter.name)
            else:
                try:
                    json.dumps(parameter.default)
                except (TypeError, ValueError):
                    pass
                else:
                    schema = {**schema, "default": parameter.default}
            properties[parameter.name] = schema
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    class _EvalTool:
        """Kernel-owned function and its model-facing tool metadata."""

        __slots__ = ("name", "fn", "description", "parameters")

        def __init__(self, name, fn, description, parameters):
            self.name = name
            self.fn = fn
            self.description = description
            self.parameters = parameters

        def describe(self) -> dict:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }

    __omp_tools__: dict[str, _EvalTool] = {}
    globals()["__omp_tools__"] = __omp_tools__
    _TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

    class _ToolProxy:
        """Define kernel tools or invoke host-side tools by attribute."""

        # Marker identifying the genuine prelude bridge to the Python shadow
        # planner (runner.py): a retained user `tool` binding that is not
        # JSON-safe is omitted from the shadow snapshot exactly like this
        # proxy. (`__slots__` is empty, so this lives on the class, not the
        # instance; accidental user collision is out of threat model.)
        __omp_tool_bridge__ = True

        __slots__ = ()

        def __call__(self, fn=None, /, *, name=None, description=None):
            if fn is None:
                return lambda decorated: self(
                    decorated,
                    name=name,
                    description=description,
                )
            if not callable(fn):
                raise TypeError("@tool expects a function")
            resolved_name = name or getattr(fn, "__name__", "")
            if not isinstance(resolved_name, str) or _TOOL_NAME_RE.fullmatch(resolved_name) is None:
                raise ValueError(f"invalid tool name {resolved_name!r}")
            schema = _tool_schema(fn)
            resolved_description = (
                description
                if isinstance(description, str) and description
                else inspect.getdoc(fn) or f"Python tool {resolved_name}"
            )
            __omp_tools__[resolved_name] = _EvalTool(
                resolved_name,
                fn,
                resolved_description,
                schema,
            )
            _emit_status(
                "tool_define",
                name=resolved_name,
                params=list(schema["properties"]),
            )
            return fn

        def defined(self) -> list[str]:
            return list(__omp_tools__)

        def undefine(self, name) -> bool:
            return __omp_tools__.pop(name, None) is not None

        def __getattr__(self, name: str) -> _ToolCallable:
            if name.startswith("_"):
                raise AttributeError(name)
            return _ToolCallable(name)

        def __getitem__(self, name: str) -> _ToolCallable:
            return _ToolCallable(name)

        def __repr__(self) -> str:
            return (
                "<tool proxy>"
                if "__omp_host_call" in globals()
                else "<tool proxy unavailable>"
            )

    tool = _ToolProxy()

    def log(message):
        """Emit a status ``log`` event for TUI rendering."""
        _emit_status("log", message=str(message))

    def phase(title):
        """Record the current readable phase and emit a status ``phase`` event."""
        globals()["__omp_current_phase__"] = str(title)
        _emit_status("phase", title=str(title))
