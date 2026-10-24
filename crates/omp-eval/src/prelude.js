// I/O and tool prelude for the JS eval runner. Port of the I/O and tool part of
// omp's eval/js/shared/prelude.txt. Every helper talks to the host only through
// `globalThis.__ompHostCall(op, args)` (waits for the reply) and
// `globalThis.__ompHostNotify(op, args)` (fire-and-forget status). Agent,
// completion, judge, workpool and budget helpers live in prelude_agents.js.
//
// The whole file runs inside a function so top-level bindings never collide
// with user cells, and it is safe to re-run: globals are simply reassigned.
(() => {
  const isNil = value => value === undefined || value === null;
  const isPlainObject = value => value !== null && typeof value === "object" && !Array.isArray(value);

  // Accepts either one trailing options object or positional optional args,
  // mirroring the TS prelude so `read(path, { offset })` and `read(path, 1, 10)`
  // behave the same way.
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
  const hostNotify = (op, args) => globalThis.__ompHostNotify(op, args);

  // Structured-clone-free projection for display(): values that JSON cannot
  // carry are rendered as strings so the notify frame always serializes.

  // TS split (prelude.txt): plain paths are read/written in the worker itself;
  // only URL paths (`scheme://…`: artifact://, agent://, omp://, local://…) go
  // through the host `read` tool, with the line window encoded as a selector.
  const hasScheme = path => typeof path === "string" && /^[A-Za-z][A-Za-z0-9+.-]*:\/\//.test(path);
  const withReadLineSelector = (path, options) => {
    const offset = typeof options.offset === "number" ? options.offset : 1;
    const limit = typeof options.limit === "number" ? options.limit : undefined;
    if (offset <= 1 && limit === undefined) return path;
    if (limit !== undefined && limit <= 0) return null;
    const start = Math.max(1, offset);
    if (limit === undefined) return `${path}:${start}-`;
    return `${path}:${start}-${start + limit - 1}`;
  };
  const resolveCellPath = rawPath => {
    const p = String(rawPath);
    return p.startsWith("/") ? p : `${process.cwd()}/${p}`;
  };
  const readToolText = async path => {
    const res = await hostCall("read", { path });
    return res && typeof res === "object" && "text" in res ? res.text : res;
  };

  const read = async (path, opts, ...rest) => {
    const options = optionsArg("read", opts, rest, ["offset", "limit"], "{ offset, limit }");
    if (hasScheme(path)) {
      const toolPath = withReadLineSelector(path, options);
      return toolPath === null ? "" : readToolText(toolPath);
    }
    let text = await Bun.file(resolveCellPath(path)).text();
    const offset = typeof options.offset === "number" ? options.offset : 1;
    const limit = typeof options.limit === "number" ? options.limit : undefined;
    if (limit !== undefined && limit <= 0) return "";
    if (offset > 1 || limit !== undefined) {
      const lines = text.split(/\r?\n/);
      const start = Math.max(0, offset - 1);
      const end = limit !== undefined ? start + limit : lines.length;
      text = lines.slice(start, end).join("\n");
    }
    return text;
  };

  const write = async (path, data) => {
    const isBlob = typeof Blob !== "undefined" && data instanceof Blob;
    if (typeof data !== "string" && !isBlob && !(data instanceof ArrayBuffer) && !ArrayBuffer.isView(data)) {
      throw new TypeError("write() expects string, Blob, ArrayBuffer, or TypedArray data");
    }
    const filePath = resolveCellPath(path);
    await Bun.write(filePath, data);
    return filePath;
  };

  // env(), env(key), env(key, value): the worker's own process environment.
  const env = (key, value) => {
    if (isNil(key)) {
      return Object.fromEntries(Object.entries(process.env).sort(([a], [b]) => a.localeCompare(b)));
    }
    if (value !== undefined) {
      process.env[String(key)] = String(value);
      return value;
    }
    return process.env[String(key)];
  };

  // tool(fn, { name, description, parameters }) registers a JS-side tool;
  // tool.<name>(args) calls a host tool through the tool bridge.
  const toolRegistry = (globalThis.__omp_tools__ ??= new Map());
  const isValidToolName = name => typeof name === "string" && /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(name);
  const registerTool = (fn, opts = {}) => {
    if (typeof fn !== "function") throw new TypeError("tool() expects a function");
    if (!isPlainObject(opts)) throw new TypeError("tool() options must be a plain object");
    const name = opts.name ?? fn.name;
    if (!isValidToolName(name)) {
      throw new TypeError(`invalid tool name ${JSON.stringify(name)}`);
    }
    const description = opts.description ?? `JS tool ${name}`;
    const parameters = opts.parameters ?? { type: "object", properties: {}, additionalProperties: true };
    toolRegistry.set(name, { name, fn, description, parameters });
    hostNotify("tool_define", { name, params: Object.keys(parameters.properties ?? {}) });
    return fn;
  };
  const tool = new Proxy(registerTool, {
    apply(_target, _thisArg, args) {
      return registerTool(args[0], args[1]);
    },
    get(_target, prop) {
      if (prop === "defined") return () => [...toolRegistry.keys()];
      if (prop === "undefine") return name => toolRegistry.delete(name);
      if (typeof prop !== "string") return undefined;
      return async args => hostCall(`tool.${prop}`, args ?? {});
    },
  });

  // output(id..., { offset, limit }): read agent:// results. One id yields the
  // value itself, several ids yield an array in input order.
  const output = async (...args) => {
    let opts = {};
    let ids = args;
    if (args.length > 0) {
      const last = args.at(-1);
      if (last && typeof last === "object" && !Array.isArray(last)) {
        opts = last;
        ids = args.slice(0, -1);
      }
    }
    const values = await Promise.all(ids.map(id => tool.read({ path: `agent://${id}`, ...opts })));
    return values.length === 1 ? values[0] : values;
  };

  const log = message => hostNotify("log", { message: String(message) });

  const phase = title => {
    globalThis.__omp_phase__ = String(title);
    hostNotify("phase", { title: String(title) });
  };

  const display = (...values) => globalThis.__ompDisplay(...values);

  Object.assign(globalThis, { display, tool, output, log, phase, read, write, env });
})();
