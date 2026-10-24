"use strict";
// The conversation view: renders stored messages and live rpc events.
(function (OMP) {
  const { el } = OMP;

  class Transcript {
    constructor(root, scroller) {
      this.root = root;
      this.scroller = scroller;
      this.reset();
      this.scheduleScroll = OMP.rafThrottle(() => this.scrollToBottom());
      this.scheduleFlush = OMP.rafThrottle(() => this.flushUpdates());
    }

    flushUpdates() {
      const updates = [...this.pendingUpdates];
      this.pendingUpdates.clear();
      for (const [id, msg] of updates) this.updateAssistant(this.assistantEntry(id), msg, true);
    }

    reset() {
      this.root.replaceChildren();
      this.live = new Map(); // messageId -> {node, parts, msg}
      this.cards = new Map(); // toolCallId -> ToolCard
      this.pendingUpdates = new Map(); // messageId -> newest streaming message
      this.stick = true;
    }

    isNearBottom() {
      const s = this.scroller;
      return s.scrollHeight - s.scrollTop - s.clientHeight < 120;
    }
    scrollToBottom(force) {
      if (force || this.stick) this.scroller.scrollTop = this.scroller.scrollHeight;
    }
    append(node) {
      this.stick = this.isNearBottom() || this.stick;
      this.root.append(node);
      this.scheduleScroll();
      return node;
    }

    loadMessages(messages) {
      this.reset();
      for (const m of messages || []) this.renderMessage(m);
      this.stick = true;
      this.scrollToBottom(true);
    }

    addSystem(text, cls = "") {
      const long = String(text).split("\n").length > 18;
      return this.append(el("div.system" + (cls ? "." + cls : "") + (long ? ".collapsible" : ""), { text }));
    }
    addDivider(text) {
      return this.append(el("div.divider", { text }));
    }
    addError(text) {
      return this.append(el("div.error-box", { text }));
    }

    card(id, name) {
      let c = this.cards.get(id);
      if (!c) {
        c = new OMP.ToolCard(id, name || "tool");
        this.cards.set(id, c);
      }
      return c;
    }

    /** Render one complete message (history or message_end without a live node). */
    renderMessage(msg, messageId) {
      if (!msg || typeof msg !== "object") return;
      switch (msg.role) {
        case "user": return this.renderUser(msg);
        case "assistant": {
          const entry = this.assistantEntry(messageId);
          this.updateAssistant(entry, msg, false);
          return entry.node;
        }
        case "toolResult": return this.toolResult(msg);
        case "bashExecution": {
          const c = this.card("bash-" + (msg.timestamp || Math.random()), "bash");
          c.setArgs({ command: msg.command });
          const out = (msg.output || "") + (msg.exitCode ? `\n[exit ${msg.exitCode}]` : "") + (msg.cancelled ? "\n[cancelled]" : "");
          c.finish({ content: [{ type: "text", text: out }] }, !!msg.exitCode);
          return this.append(c.node);
        }
        case "compactionSummary": case "branchSummary": {
          const label = msg.role === "compactionSummary"
            ? `Context compacted${msg.tokensBefore ? ` (${OMP.fmtTokens(msg.tokensBefore)} tokens before)` : ""}`
            : "Branch summary";
          const box = el("details.summary-box", null, el("summary", { text: label }), el("div.md", { html: OMP.md(msg.summary || "") }));
          return this.append(box);
        }
        case "custom": case "hookMessage": {
          if (msg.display === false) return;
          const text = typeof msg.content === "string" ? msg.content : OMP.contentText(msg.content);
          if (!text) return;
          return this.append(el("div.system", { text: (msg.customType ? `[${msg.customType}] ` : "") + text }));
        }
        default: return;
      }
    }

    renderUser(msg) {
      const text = OMP.contentText(msg.content);
      const node = el("div.msg.user");
      if (text) node.append(el("div", { text }));
      const imgs = OMP.contentImages(msg.content);
      if (imgs.length) node.append(el("div.imgs", null, imgs.map((i) => el("img", { src: `data:${i.mimeType};base64,${i.data}` }))));
      if (!text && !imgs.length) return;
      return this.append(node);
    }

    toolResult(msg) {
      const c = this.cards.get(msg.toolCallId);
      if (c) {
        c.finish(msg, msg.isError);
        return;
      }
      const card = this.card(msg.toolCallId || "r" + Math.random(), msg.toolName);
      card.finish(msg, msg.isError);
      this.append(card.node);
    }

    assistantEntry(messageId) {
      if (messageId && this.live.has(messageId)) return this.live.get(messageId);
      const entry = { node: el("div.msg.assistant"), parts: [], tail: null };
      if (messageId) this.live.set(messageId, entry);
      this.append(entry.node);
      return entry;
    }

    updateAssistant(entry, msg, streaming) {
      const stick = this.isNearBottom();
      const content = Array.isArray(msg.content) ? msg.content : [{ type: "text", text: String(msg.content || "") }];
      content.forEach((block, i) => {
        let part = entry.parts[i];
        const kind = block.type;
        if (!part || part.kind !== kind) {
          if (part) part.node.remove();
          part = { kind, node: null, last: null };
          if (kind === "text") part.node = el("div.md");
          else if (kind === "thinking" || kind === "redactedThinking") {
            part.body = el("div.md");
            part.summary = el("summary", { text: "Thinking" });
            part.node = el("details.thinking", null, part.summary, part.body);
          } else if (kind === "toolCall") {
            part.card = this.card(block.id, block.name);
            part.node = part.card.node;
          } else part.node = el("div");
          const before = entry.parts.slice(i + 1).find((p) => p && p.node.isConnected);
          entry.node.insertBefore(part.node, before ? before.node : entry.tail);
          entry.parts[i] = part;
        }
        if (kind === "text") {
          if (part.last !== block.text) { part.node.innerHTML = OMP.md(block.text); part.last = block.text; }
        } else if (kind === "thinking") {
          const t = block.thinking || "";
          if (part.last !== t) {
            part.body.innerHTML = OMP.md(t);
            part.last = t;
            part.words = t.trim() ? t.trim().split(/\s+/).length : 0;
          }
          part.summary.textContent = streaming && i === content.length - 1 ? `Thinking… (${part.words} words)` : `Thought (${part.words} words)`;
        } else if (kind === "redactedThinking") {
          part.summary.textContent = "Thinking (redacted)";
        } else if (kind === "toolCall") {
          part.card.name = block.name || part.card.name;
          part.card.setArgs(block.arguments);
        }
      });
      for (let i = content.length; i < entry.parts.length; i++) if (entry.parts[i]) entry.parts[i].node.remove();
      entry.parts.length = content.length;
      if (!streaming) {
        if (entry.tail) entry.tail.remove();
        entry.tail = null;
        if (msg.stopReason === "error" && msg.errorMessage) entry.tail = el("div.error-box", { text: msg.errorMessage });
        else if (msg.stopReason === "aborted") entry.tail = el("div.divider", { text: "interrupted" });
        else if (msg.stopReason === "length") entry.tail = el("div.system.warn", { text: "Output truncated: max tokens reached." });
        if (entry.tail) entry.node.append(entry.tail);
        const u = msg.usage;
        if (u && (u.output || u.input) && msg.stopReason !== "toolUse") {
          const meta = el("div.meta", null,
            el("span", { text: msg.model || "" }),
            el("span", { text: `↑${OMP.fmtTokens((u.input || 0) + (u.cacheRead || 0) + (u.cacheWrite || 0))} ↓${OMP.fmtTokens(u.output)}` }),
            u.cost && u.cost.total ? el("span", { text: OMP.fmtCost(u.cost.total) }) : null);
          if (entry.meta) entry.meta.replaceWith(meta); else entry.node.append(meta);
          entry.meta = meta;
        }
      }
      if (!entry.node.childNodes.length) entry.node.hidden = true; else entry.node.hidden = false;
      if (stick) this.scheduleScroll();
    }

    /** Apply one rpc event frame. Returns true when handled. */
    onEvent(f) {
      switch (f.type) {
        case "message_start":
          if (f.message && f.message.role === "assistant") this.updateAssistant(this.assistantEntry(f.messageId), f.message, true);
          else this.renderMessage(f.message, f.messageId);
          return true;
        case "message_update":
          if (f.message && f.message.role === "assistant") {
            // Coalesce deltas: each update carries the full message, so only
            // the newest one per frame needs rendering.
            this.pendingUpdates.set(f.messageId, f.message);
            this.scheduleFlush();
          }
          return true;
        case "message_end":
          this.pendingUpdates.delete(f.messageId);
          if (f.message && f.message.role === "assistant") {
            this.updateAssistant(this.assistantEntry(f.messageId), f.message, false);
            if (f.messageId) this.live.delete(f.messageId);
          } else if (f.message && f.message.role === "toolResult") this.toolResult(f.message);
          return true;
        case "tool_execution_start":
          this.card(f.toolCallId, f.toolName).start(f.args);
          return true;
        case "tool_execution_update":
          this.card(f.toolCallId, f.toolName).update(f.partialResult);
          return true;
        case "tool_execution_end":
          this.card(f.toolCallId, f.toolName).finish(f.result, f.isError);
          return true;
        case "command_output":
          this.addSystem(f.text || "");
          return true;
        case "notice":
          this.addSystem(f.message || f.text || "", f.level === "error" ? "error" : f.level === "warning" ? "warn" : "");
          return true;
        case "auto_compaction_start": this.addDivider("compacting context…"); return true;
        case "auto_compaction_end":
          if (f.errorMessage) this.addSystem("Compaction failed: " + f.errorMessage, "error");
          else this.addDivider(f.aborted ? "compaction aborted" : "context compacted");
          return true;
        case "auto_retry_start":
          this.addSystem(`Retrying (attempt ${f.attempt || "?"}${f.maxAttempts ? "/" + f.maxAttempts : ""})${f.errorMessage ? ": " + f.errorMessage : ""}`, "warn");
          return true;
        case "error":
          this.addError(f.message || f.error || JSON.stringify(f));
          return true;
        default:
          return false;
      }
    }
  }

  OMP.Transcript = Transcript;
})(window.OMP);
