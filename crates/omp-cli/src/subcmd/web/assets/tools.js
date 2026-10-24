"use strict";
// Tool-call cards: one per toolCall id, fed by the assistant message (args),
// tool_execution_* events (status, live output) and the toolResult message.
(function (OMP) {
  const { el, esc } = OMP;
  const MAX_PREVIEW_LINES = 400;

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
      case "edit": case "write": return pick("path", "file_path") || "";
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

  function argsView(name, args) {
    args = args || {};
    const parts = [];
    if (name === "bash" && args.command) {
      parts.push(el("div.tool-label", { text: "command" }), pre(args.command));
    } else if (name === "write" && typeof args.content === "string") {
      parts.push(el("div.tool-label", { text: args.path || "content" }), pre(args.content));
    } else if (name === "edit" && (args.old_string !== undefined || args.new_string !== undefined)) {
      const d = String(args.old_string || "").split("\n").map((l) => "-" + l).concat(String(args.new_string || "").split("\n").map((l) => "+" + l));
      parts.push(el("div.tool-label", { text: args.path || "edit" }), pre(d.join("\n"), "diff"));
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
    const text = textOf(r);
    if (text && !(diffs.length && !card.isError && text.length < 400)) {
      parts.push(el("div.tool-label", { text: card.isError ? "error" : "output" }), pre(text));
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
      this.body = el("div.tool-body", { hidden: true });
      this.tail = el("pre.tool-tail", { hidden: true });
      this.node.append(this.head, this.tail, this.body);
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
      const icon = { pending: "○", running: "●", ok: "✓", error: "✗" }[this.status];
      this.node.className = "tool " + this.status;
      const dur = this.startedAt && this.endedAt ? OMP.fmtDuration(this.endedAt - this.startedAt) : "";
      this.head.replaceChildren(
        el("span.status", { text: icon }),
        el("span.tool-name", { text: this.name }),
        el("span.tool-sum", { text: summary(this.name, this.args), title: summary(this.name, this.args) }),
        el("span.tool-dur", { text: dur }),
        el("span.faint", { text: this.open ? "▾" : "▸" }),
      );
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
