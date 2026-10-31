"use strict";
// Static no-undef check for the omp web UI assets (plain browser scripts and
// ES modules), mirroring the browser's global script semantics:
//
//   * index.html loads the classic scripts in order into one shared global
//     scope, so every top-level declaration of every classic script is a
//     potential global for every other file.
//   * Each file hangs its exports off `window.OMP` (see util.js), so `OMP`
//     is a known global; property accesses on globals (`OMP.foo`,
//     `window.bar`) are never flagged.
//   * Files with top-level `import`/`export` (e.g. viewer3d.js) are parsed
//     as ES modules: their imports declare their own names and their
//     top-level declarations do NOT leak into other files' globals.
//
// Unknown identifiers are reported with file:line:col. Exit 0 only when
// clean, so this can gate commits/CI. Run: node test/check-no-undef.js
//
// Requires `npm install` (acorn) in this directory.

const fs = require("fs");
const path = require("path");
const acorn = require("acorn");

const ASSETS = path.resolve(__dirname, "..");

// Browser/WHATWG globals the plain scripts may reference by bare name.
// Intentionally narrow: an identifier not declared in-file, not a top-level
// declaration of another classic script, and not listed here is an error.
const BROWSER_GLOBALS = new Set([
  "window", "document", "navigator", "location", "history", "localStorage",
  "sessionStorage", "screen", "alert", "confirm", "prompt", "fetch",
  "WebSocket", "XMLHttpRequest", "URL", "URLSearchParams", "FormData",
  "AbortController", "AbortSignal", "Blob", "File", "FileReader",
  "CustomEvent", "Event", "EventTarget", "MessageChannel", "MessagePort",
  "requestAnimationFrame", "cancelAnimationFrame",
  "requestIdleCallback", "cancelIdleCallback",
  "setTimeout", "clearTimeout", "setInterval", "clearInterval",
  "queueMicrotask", "structuredClone", "atob", "btoa",
  "getComputedStyle", "matchMedia", "ResizeObserver", "MutationObserver",
  "IntersectionObserver", "PerformanceObserver", "performance",
  "console", "JSON", "Math", "Date", "Promise", "Array", "Object",
  "String", "Number", "Boolean", "Symbol", "BigInt", "Function",
  "RegExp", "Error", "TypeError", "RangeError", "SyntaxError",
  "Map", "Set", "WeakMap", "WeakSet", "WeakRef", "Proxy", "Reflect",
  "parseInt", "parseFloat", "isNaN", "isFinite", "encodeURIComponent",
  "decodeURIComponent", "encodeURI", "decodeURI",
  "Intl", "undefined", "NaN", "Infinity", "globalThis",
  "crypto", "TextEncoder", "TextDecoder", "Worker", "SharedWorker",
  "Notification", "visualViewport", "speechSynthesis", "caches",
  "Audio", "Image", "Option", "HTMLElement", "Element", "Node", "MediaRecorder",
  "Uint8Array", "Uint8ClampedArray", "Int8Array", "Uint16Array", "Int16Array",
  "Uint32Array", "Int32Array", "Float32Array", "Float64Array",
  "BigInt64Array", "BigUint64Array", "ArrayBuffer", "SharedArrayBuffer",
  "DataView", "Atomics", "TextDecoderStream",
]);

// Node globals, used only by the JS under test/.
const NODE_GLOBALS = new Set([
  "require", "module", "exports", "__dirname", "__filename", "process",
  "Buffer", "global", "setImmediate", "clearImmediate",
]);

// Shared namespace every script attaches to (window.OMP).
BROWSER_GLOBALS.add("OMP");

// Files excluded from checking (vendored minified bundles).
const SKIP = new Set(["vendor", "node_modules"]);

const MODULE_HINT = /(?:^|\n)\s*(?:import\s[\s{"'*]|import\.meta|export\s+(?:const|let|var|function|class|default|\{|\*))/;

function listFiles(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (!SKIP.has(entry.name)) listFiles(p, out);
    } else if (entry.name.endsWith(".js")) {
      out.push(p);
    }
  }
  return out;
}

function parse(source, file) {
  const opts = { ecmaVersion: "latest", locations: true, allowReturnOutsideFunction: true, allowAwaitOutsideFunction: true };
  const sourceType = MODULE_HINT.test(source) ? "module" : "script";
  try {
    return { ast: acorn.parse(source, { ...opts, sourceType }), sourceType };
  } catch (first) {
    try {
      return { ast: acorn.parse(source, { ...opts, sourceType: sourceType === "module" ? "script" : "module" }), sourceType: sourceType === "module" ? "script" : "module" };
    } catch (second) {
      throw new Error(`${file}: parse error (${sourceType}): ${first.message}`);
    }
  }
}

// ---- scope-aware identifier walk -------------------------------------------

const CHILD_KEYS_SKIP = new Set(["type", "start", "end", "loc", "range"]);

function childNodes(node) {
  const out = [];
  for (const key of Object.keys(node)) {
    if (CHILD_KEYS_SKIP.has(key)) continue;
    const v = node[key];
    if (Array.isArray(v)) {
      for (const c of v) if (c && typeof c.type === "string") out.push([key, c]);
    } else if (v && typeof v.type === "string") {
      out.push([key, v]);
    }
  }
  return out;
}

function patternNames(pattern, out) {
  if (!pattern) return;
  switch (pattern.type) {
    case "Identifier": out.push(pattern.name); break;
    case "ObjectPattern":
      for (const p of pattern.properties) {
        if (p.type === "RestElement") patternNames(p.argument, out);
        else patternNames(p.value, out);
      }
      break;
    case "ArrayPattern":
      for (const el of pattern.elements) if (el) patternNames(el, out);
      break;
    case "AssignmentPattern": patternNames(pattern.left, out); break;
    case "RestElement": patternNames(pattern.argument, out); break;
    default: break;
  }
}

class Scope {
  constructor(kind, parent) { this.kind = kind; this.parent = parent; this.names = new Set(); }
  has(name) {
    for (let s = this; s; s = s.parent) if (s.names.has(name)) return true;
    return false;
  }
}

// Pre-declare names in `scope` (and, for `var`, in the nearest function
// scope) so use-before-declaration is not flagged.
function declare(node, scope, hoistVars = false) {
  switch (node.type) {
    case "VariableDeclaration": {
      const target = node.kind === "var" && hoistVars ? functionScope(scope) : scope;
      for (const d of node.declarations) {
        const names = [];
        patternNames(d.id, names);
        for (const n of names) target.names.add(n);
      }
      break;
    }
    case "FunctionDeclaration":
    case "ClassDeclaration":
      if (node.id) scope.names.add(node.id.name);
      break;
    default: break;
  }
}

function functionScope(scope) {
  let s = scope;
  while (s.kind !== "function" && s.parent) s = s.parent;
  return s;
}

// Walk a statement list of a block, entering nested blocks (but not nested
// functions) to collect var-hoisted names.
function hoistVarsDeep(node, fnScope) {
  for (const [, child] of childNodes(node)) {
    if (child.type === "FunctionDeclaration" || child.type === "FunctionExpression" ||
        child.type === "ArrowFunctionExpression") continue;
    declare(child, fnScope, true);
    if (child.type === "BlockStatement" || child.type === "SwitchStatement" ||
        child.type === "ForStatement" || child.type === "ForInStatement" ||
        child.type === "ForOfStatement" || child.type === "TryStatement") {
      hoistVarsDeep(child, fnScope);
    }
  }
}

// An Identifier node is a *reference* unless it occupies a binding/key slot.
function isReference(node, parent, key) {
  if (!parent) return false;
  switch (parent.type) {
    case "MemberExpression": return parent.computed && key === "property";
    case "Property": case "PropertyDefinition": case "MethodDefinition":
      return key === "value" || parent.computed;
    case "VariableDeclarator": return key === "init";
    case "FunctionDeclaration": case "FunctionExpression":
    case "ClassDeclaration": case "ClassExpression":
      return key === "body" || key === "superClass";
    case "LabeledStatement": case "BreakStatement": case "ContinueStatement":
      return false;
    case "ImportDefaultSpecifier": case "ImportNamespaceSpecifier":
    case "ImportSpecifier": case "ExportSpecifier":
      return false;
    default: return true;
  }
}

function checkFile(file, extraGlobals) {
  const source = fs.readFileSync(file, "utf8");
  const isTestFile = path.relative(ASSETS, file).startsWith("test");
  const { ast, sourceType } = parse(source, file);
  const errors = [];
  const moduleScope = new Scope("module", null);
  if (sourceType === "module") {
    for (const stmt of ast.body) {
      if (stmt.type === "ImportDeclaration") {
        for (const spec of stmt.specifiers) moduleScope.names.add(spec.local.name);
      } else {
        declare(stmt, moduleScope, true);
      }
    }
  } else {
    for (const stmt of ast.body) declare(stmt, moduleScope, true);
    hoistVarsDeep({ type: "Program", body: ast.body }, functionScope(moduleScope));
  }

  (function walk(node, parent, key, scope) {
    if (node.type === "Identifier") {
      if (isReference(node, parent, key) &&
          !scope.has(node.name) && !BROWSER_GLOBALS.has(node.name) &&
          !extraGlobals.has(node.name) && node.name !== "arguments" &&
          !(isTestFile && NODE_GLOBALS.has(node.name))) {
        const line = node.loc.start.line;
        const text = source.split("\n")[line - 1].trim();
        errors.push(`${file}:${line}:${node.loc.start.column}: '${node.name}' is not defined (near: ${text.slice(0, 90)})`);
      }
      return;
    }
    let inner = scope;
    if (node.type === "FunctionDeclaration" || node.type === "FunctionExpression" ||
        node.type === "ArrowFunctionExpression") {
      inner = new Scope("function", scope);
      const names = [];
      patternNames(node.id, names);
      for (const p of node.params) patternNames(p, names);
      if (node.typeParams) {} // no TS
      for (const n of names) inner.names.add(n);
      if (node.body.type === "BlockStatement") {
        declare(node.body, inner, true);
        hoistVarsDeep(node.body, functionScope(inner));
      }
    } else if (node.type === "BlockStatement" || node.type === "StaticBlock") {
      inner = new Scope("block", scope);
      if (node.type === "BlockStatement") for (const stmt of node.body) declare(stmt, inner, true);
    } else if (node.type === "ForStatement" || node.type === "ForInStatement" || node.type === "ForOfStatement") {
      inner = new Scope("block", scope);
      if (node.init && node.init.type === "VariableDeclaration" && node.init.kind !== "var") {
        declare(node.init, inner, true);
      }
      if (node.left && node.left.type === "VariableDeclaration" && node.left.kind !== "var") {
        declare(node.left, inner, true);
      }
    } else if (node.type === "CatchClause") {
      inner = new Scope("block", scope);
      if (node.param) {
        const names = [];
        patternNames(node.param, names);
        for (const n of names) inner.names.add(n);
      }
    } else if (node.type === "SwitchStatement") {
      inner = new Scope("block", scope);
      for (const c of node.cases) for (const stmt of c.consequent) declare(stmt, inner, true);
    }
    for (const [ck, child] of childNodes(node)) walk(child, node, ck, inner);
  })(ast, null, null, moduleScope);

  return { errors, sourceType };
}

function main() {
  const files = listFiles(ASSETS).filter((f) => !f.includes("node_modules")).sort();
  // Classic scripts share one global scope: collect every top-level name
  // (and window.X assignment targets) across all script files first.
  const extraGlobals = new Set();
  for (const file of files) {
    const source = fs.readFileSync(file, "utf8");
    if (MODULE_HINT.test(source)) continue; // module tops don't leak
    try {
      const { ast } = parse(source, file);
      for (const stmt of ast.body) {
        const names = [];
        if (stmt.type === "VariableDeclaration") {
          for (const d of stmt.declarations) {
            patternNames(d.id, names);
            // const X = window.Y || {} style global namespace aliases
            if (d.init && d.init.type === "MemberExpression" && d.init.object.type === "Identifier") {
              names.push(d.id.name);
            }
          }
        } else if (stmt.type === "FunctionDeclaration" && stmt.id) {
          names.push(stmt.id.name);
        } else if (stmt.type === "ClassDeclaration" && stmt.id) {
          names.push(stmt.id.name);
        } else if (stmt.type === "ExpressionStatement" && stmt.expression.type === "AssignmentExpression") {
          const t = stmt.expression.left;
          if (t.type === "MemberExpression" && !t.computed && t.object.type === "Identifier" &&
              (t.object.name === "window" || t.object.name === "globalThis")) {
            names.push(t.property.name);
          }
        }
        for (const n of names) extraGlobals.add(n);
      }
    } catch { /* parse errors reported below */ }
  }

  const allErrors = [];
  for (const file of files) {
    try {
      const { errors, sourceType } = checkFile(file, extraGlobals);
      process.stderr.write(`ok   ${path.relative(ASSETS, file)} (${sourceType})\n`);
      allErrors.push(...errors);
    } catch (e) {
      allErrors.push(e.message);
    }
  }
  if (allErrors.length) {
    process.stderr.write(`\n${allErrors.length} problem(s):\n`);
    for (const e of allErrors) process.stderr.write(`  ${e}\n`);
    process.exitCode = 1;
  } else {
    process.stderr.write(`\nno-undef check passed (${files.length} files)\n`);
  }
}

main();
