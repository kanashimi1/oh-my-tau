"use strict";
// Tool-call cards: one per toolCall id, fed by the assistant message (args),
// tool_execution_* events (status, live output) and the toolResult message.
(function (OMP) {
  const { el, esc } = OMP;
  const MAX_PREVIEW_LINES = 400;

  // Tool families -> inline SVG icons (re-implemented, stroke-based, theme-coloured).
  const FAMILIES = [
    [/^(read|view|open)/, "read"],
    [/^(grep|glob|find|search|ls|list|ripgrep)/, "search"],
    [/^(edit|patch|apply|str_replace|multiedit)/, "edit"],
    [/^(write|create|save)/, "write"],
    [/^(bash|shell|exec|command|eval|python|node|bun)/, "bash"],
    [/^(web|fetch|http|browser|curl)/, "web"],
    [/^(task|agent|spawn|todo|ask)/, "task"],
  ];
  const ICON_PATHS = {
    read: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h4"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
    edit: '<path d="M17 3a2.83 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5z"/>',
    write: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M12 12v6M9 15h6"/>',
    bash: '<path d="M4 17l6-5-6-5"/><path d="M12 19h8"/>',
    web: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3c3 3.6 3 14.4 0 18M12 3c-3 3.6-3 14.4 0 18"/>',
    task: '<circle cx="6" cy="5" r="2.2"/><circle cx="6" cy="19" r="2.2"/><circle cx="18" cy="12" r="2.2"/><path d="M8 5.5h4a2 2 0 0 1 2 2v2M8 18.5h4a2 2 0 0 0 2-2v-2"/>',
    other: '<circle cx="12" cy="12" r="3"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/>',
  };
  function familyOf(name) {
    for (const [re, fam] of FAMILIES) if (re.test(name)) return fam;
    return "other";
  }
  function icon(name) {
    return el("span.tool-ic", { html: `<svg viewBox="0 0 24 24" aria-hidden="true">${ICON_PATHS[familyOf(name)] || ICON_PATHS.other}</svg>` });
  }

  const LANGS = { ts: "typescript", tsx: "typescript", js: "javascript", mjs: "javascript", cjs: "javascript", py: "python", rs: "rust", json: "json", md: "markdown", html: "html", css: "css", scss: "scss", sh: "bash", bash: "bash", zsh: "bash", toml: "toml", yaml: "yaml", yml: "yaml", go: "go", c: "c", h: "c", cpp: "cpp", hpp: "cpp", java: "java", rb: "ruby", sql: "sql", lua: "lua", txt: "text" };
  function extLang(path) {
    const m = /\.([a-z0-9]+)$/i.exec(String(path || ""));
    return m ? (LANGS[m[1].toLowerCase()] || "") : "";
  }

  /** 'Open' action -> artifacts panel (guarded; no-op while artifacts slice is absent). */
  function openBtn(kind, title, content, lang) {
    return el("button.open-btn", {
      text: "Open",
      onclick: () => OMP.artifacts?.open({ kind, title, content, lang }),
    });
  }

  function firstLine(s, max = 160) {
    s = String(s ?? "").trim().split("\n")[0];
    return s.length > max ? s.slice(0, max) + "…" : s;
  }

  /** One-line summary of the call's arguments. */
  function summary(name, args) {
    args = args || {};
    const pick = (...keys) => keys.map((k) => args[k]).find((v) => typeof v === "string" && v.trim());
    switch (name) {
      case "bash": return "$ " + firstLine(args.command);
      case "read": {
        const p = pick("path", "file_path", "sel") || "";
        const range = args.offset || args.limit ? ` :${args.offset || 1}${args.limit ? "+" + args.limit : ""}` : "";
        return p + range;
      }
      case "edit": case "write": {
        const direct = pick("path", "file_path");
        if (direct) return direct;
        // hashline edits: `[PATH#TAG]` headers inside `input`
        const files = [...String(args.input || "").matchAll(/^\[([^\]#]+)#[0-9A-Fa-f]+\]/gm)].map((m) => m[1]);
        return files.length > 2 ? `${files[0]} +${files.length - 1} files` : files.join(", ");
      }
      case "grep": return `${firstLine(args.pattern, 80)}${args.path ? "  in " + args.path : ""}`;
      case "glob": case "find": return pick("pattern", "path", "paths") || JSON.stringify(args.patterns || "");
      case "task": {
        const tasks = Array.isArray(args.tasks) ? args.tasks : [];
        const agent = args.agent ? args.agent + ": " : "";
        if (tasks.length > 1) return `${agent}${tasks.length} tasks`;
        return agent + firstLine((tasks[0] && (tasks[0].description || tasks[0].task)) || args.description || args.task || "");
      }
      case "eval": return `${args.language || ""} ${firstLine(args.code || args.title || "")}`.trim();
      case "web_search": return pick("query") || "";
      case "todo": return Array.isArray(args.ops) ? args.ops.map((o) => o.op).join(", ") : pick("op") || "";
      case "ask": return firstLine((args.questions && args.questions[0] && args.questions[0].question) || "");
      default: {
        const s = pick("command", "path", "file_path", "pattern", "query", "url", "description", "title", "name", "code", "text");
        return s ? firstLine(s) : firstLine(JSON.stringify(args), 140);
      }
    }
  }

  function textOf(result) {
    if (!result) return "";
    if (typeof result === "string") return result;
    if (Array.isArray(result.content)) return OMP.contentText(result.content);
    if (typeof result.content === "string") return result.content;
    return "";
  }

  function pre(text, cls) {
    const lines = String(text).split("\n");
    const node = el("pre" + (cls ? "." + cls : ""));
    const render = (all) => {
      const shown = all ? lines : lines.slice(0, MAX_PREVIEW_LINES);
      if (cls === "diff") node.innerHTML = OMP.highlightDiff(shown.join("\n"));
      else node.textContent = shown.join("\n");
    };
    render(false);
    if (lines.length <= MAX_PREVIEW_LINES) return node;
    const more = el("button.more-btn", {
      text: `show all ${lines.length} lines`,
      onclick: () => { render(true); more.remove(); },
    });
    return el("div", null, node, more);
  }

  /** Numbered omp edit diffs (`+12|text`, `-12|text`) to a unified-ish view. */
  function editDiff(diff) {
    return String(diff).split("\n").map((l) => l.replace(/^([+-]?)\s*\d+\s*\|/, (_, s) => (s || " "))).join("\n");
  }

  /** The artifact a finished call can open in the side panel, or null. */
  function artifactFor(card) {
    const r = card.result, args = card.args || {};
    if (!r || card.isError) return null;
    const details = r.details || {};
    const diffs = [];
    if (details.diff) diffs.push([details.path, details.diff]);
    if (Array.isArray(details.perFileResults)) for (const f of details.perFileResults) if (f.diff) diffs.push([f.path, f.diff]);
    if (diffs.length) return { kind: "diff", title: details.path || diffs[0][0] || "diff", content: diffs.map(([, d]) => editDiff(d)).join("\n---\n") };
    if (card.name === "write" && typeof args.content === "string") return { kind: "code", title: args.path || "file", content: args.content, lang: extLang(args.path) };
    const text = textOf(r);
    if (card.name === "read" && text) {
      const path = args.path || args.file_path || "result";
      return { kind: "code", title: path, content: text, lang: extLang(path) };
    }
    return null;
  }

  function argsView(name, args) {
    args = args || {};
    const parts = [];
    if (name === "bash" && args.command) {
      parts.push(el("div.tool-label", { text: "command" }), pre(args.command));
    } else if (name === "write" && typeof args.content === "string") {
      parts.push(el("div.tool-label", { text: args.path || "content" }), pre(args.content));
      parts.push(openBtn("code", args.path || "file", args.content, extLang(args.path)));
    } else if (name === "edit" && (args.old_string !== undefined || args.new_string !== undefined)) {
      const d = String(args.old_string || "").split("\n").map((l) => "-" + l).concat(String(args.new_string || "").split("\n").map((l) => "+" + l));
      parts.push(el("div.tool-label", { text: args.path || "edit" }), pre(d.join("\n"), "diff"));
      parts.push(openBtn("diff", args.path || "edit", d.join("\n"), ""));
    } else if (name === "eval" && args.code) {
      parts.push(el("div.tool-label", { text: args.language || "code" }));
      const box = el("pre");
      box.innerHTML = OMP.highlight(args.code, args.language === "js" ? "js" : "py");
      parts.push(box);
    } else if (Object.keys(args).length) {
      parts.push(el("div.tool-label", { text: "arguments" }), pre(JSON.stringify(args, null, 2)));
    }
    return parts;
  }

  function resultView(name, card) {
    const parts = [];
    const r = card.result;
    if (!r) return parts;
    const details = r.details || {};
    const diffs = [];
    if (details.diff) diffs.push([details.path, details.diff]);
    if (Array.isArray(details.perFileResults)) for (const f of details.perFileResults) if (f.diff) diffs.push([f.path, f.diff]);
    for (const [path, diff] of diffs) parts.push(el("div.tool-label", { text: "diff " + (path || "") }), pre(editDiff(diff), "diff"));
    if (diffs.length) parts.push(openBtn("diff", details.path || diffs[0][0] || "diff", diffs.map(([p, d]) => editDiff(d)).join("\n---\n"), ""));
    const text = textOf(r);
    if (text && !(diffs.length && !card.isError && text.length < 400)) {
      parts.push(el("div.tool-label", { text: card.isError ? "error" : "output" }), pre(text));
    }
    if (name === "read" && text && !card.isError) {
      const p = (card.args && (card.args.path || card.args.file_path)) || "result";
      parts.push(openBtn("code", p, text, extLang(p)));
    }
    for (const img of OMP.contentImages(r.content)) {
      parts.push(el("img", { src: `data:${img.mimeType};base64,${img.data}` }));
    }
    return parts;
  }

  class ToolCard {
    constructor(id, name) {
      this.id = id;
      this.name = name;
      this.args = {};
      this.status = "pending"; // pending | running | ok | error
      this.result = null;
      this.partial = "";
      this.isError = false;
      this.startedAt = null;
      this.endedAt = null;
      this.open = false;
      this.node = el("div.tool.pending");
      this.head = el("div.tool-head", { onclick: () => this.toggle() });
      this.inline = el("div.tool-inline-preview-slot", { hidden: true });
      this.inlineKey = "";
      this.body = el("div.tool-body", { hidden: true });
      this.tail = el("pre.tool-tail", { hidden: true });
      this.node.append(this.head, this.inline, this.tail, this.body);
      this.renderHead();
    }
    setArgs(args) {
      if (args && typeof args === "object") this.args = args;
      this.renderHead();
      if (this.open) this.renderBody();
    }
    start(args) {
      if (args) this.args = args;
      this.status = "running";
      this.startedAt = this.startedAt || Date.now();
      this.renderHead();
    }
    update(partial) {
      const text = textOf(partial);
      if (!text) return;
      this.partial = text;
      if (this.status === "pending") this.start();
      if (!this.open) {
        this.tail.hidden = false;
        this.tail.textContent = text.split("\n").slice(-8).join("\n");
      } else this.renderBody();
    }
    finish(result, isError) {
      this.result = result || this.result;
      this.isError = !!isError;
      this.status = isError ? "error" : "ok";
      this.endedAt = this.endedAt || Date.now();
      this.tail.hidden = true;
      this.renderHead();
      if (isError && !this.open) this.toggle(true);
      else if (this.open) this.renderBody();
    }
    toggle(force) {
      this.open = force ?? !this.open;
      this.body.hidden = !this.open;
      if (this.open) { this.tail.hidden = true; this.renderBody(); }
      this.renderHead();
    }
    renderHead() {
      const mark = { pending: "○", running: "", ok: "✓", error: "✗" }[this.status];
      this.node.className = "tool " + this.status;
      const wall = this.result && this.result.details && this.result.details.wallTimeMs;
      const span = this.startedAt && this.endedAt ? this.endedAt - this.startedAt : 0;
      // span < 5ms means both ends were stamped by one burst of frames (replay, batched delivery): unknown, not 0ms
      const dur = typeof wall === "number" ? OMP.fmtDuration(wall) : span >= 5 ? OMP.fmtDuration(span) : "";
      const sum = summary(this.name, this.args);
      this.head.replaceChildren(
        icon(this.name),
        el("span.tool-sum", { text: sum, title: `${this.name} — ${sum}` }),
        el("span.tool-dur", { text: dur }),
        el("span.status", { text: mark }),
      );
      const file = ["write", "edit", "read"].includes(this.name) && this.status === "ok" && (this.args.path || this.args.file_path);
      this.renderInline(file);
      if (OMP.preview) OMP.preview.noteArgs(this.args);
      if (file && OMP.preview && OMP.preview.kind(file) && OMP.preview.kind(file) !== "text") {
        this.head.insertBefore(el("button.tool-open.tool-preview", {
          text: "Preview",
          title: `Preview ${file}`,
          onclick: (e) => { e.stopPropagation(); OMP.preview.open(file); },
        }), this.head.querySelector(".tool-dur"));
      }
      const art = OMP.artifacts && artifactFor(this);
      if (art) {
        this.head.insertBefore(el("button.tool-open", {
          text: "Open",
          title: "Open in side panel",
          onclick: (e) => { e.stopPropagation(); OMP.artifacts.open(art); },
        }), this.head.querySelector(".tool-dur"));
      }
    }
    renderInline(file) {
      const images = this.result ? OMP.contentImages(this.result.content) : [];
      const key = (file || "") + (images.length ? `#images:${images.length}` : "");
      if (key === this.inlineKey) return;
      this.inlineKey = key;
      const preview = file && OMP.preview && OMP.preview.inline ? OMP.preview.inline(file) : null;
      const nodes = [];
      if (preview) nodes.push(preview);
      else for (const [i, img] of images.entries()) {
        nodes.push(el("div.chat-file-preview", { "data-kind": "image" },
          el("img", { src: `data:${img.mimeType};base64,${img.data}`, alt: `Tool image ${i + 1}` })));
      }
      this.inline.hidden = nodes.length === 0;
      this.inline.replaceChildren(...nodes);
    }
    renderBody() {
      const parts = argsView(this.name, this.args);
      if (this.result) parts.push(...resultView(this.name, this));
      else if (this.partial) parts.push(el("div.tool-label", { text: "output (live)" }), pre(this.partial));
      this.body.replaceChildren(...parts);
    }
  }

  OMP.ToolCard = ToolCard;
  OMP.toolSummary = summary;
})(window.OMP);
