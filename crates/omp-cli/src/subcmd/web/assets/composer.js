"use strict";
// Prompt editor: autosize, slash-command autocomplete, history, image
// attachments (paste / drop / picker), steer & follow-up while streaming.
(function (OMP) {
  const { el } = OMP;
  const HISTORY_KEY = "omp-web-history";
  const HISTORY_MAX = 200;

  class Composer {
    constructor({ onSubmit, onAbort }) {
      this.ta = OMP.$("#composer");
      this.sendBtn = OMP.$("#send-btn");
      this.stopBtn = OMP.$("#stop-btn");
      this.acBox = OMP.$("#autocomplete");
      this.attBox = OMP.$("#attachments");
      this.fileInput = OMP.$("#file-input");
      this.onSubmit = onSubmit;
      this.onAbort = onAbort;
      this.commands = [];
      this.images = [];
      this.streaming = false;
      this.ac = { open: false, items: [], sel: 0 };
      try { this.history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { this.history = []; }
      this.histPos = -1;
      this.draft = "";
      this.bind();
    }

    bind() {
      const ta = this.ta;
      ta.addEventListener("input", () => { this.autosize(); this.updateAc(); this.updateMode(); });
      ta.addEventListener("keydown", (e) => this.onKey(e));
      ta.addEventListener("blur", () => setTimeout(() => this.closeAc(), 150));
      ta.addEventListener("paste", (e) => {
        const files = [...(e.clipboardData?.items || [])].filter((i) => i.type.startsWith("image/")).map((i) => i.getAsFile());
        if (files.length) { e.preventDefault(); files.forEach((f) => this.addImage(f)); }
      });
      const wrap = OMP.$("#composer-wrap");
      wrap.addEventListener("dragover", (e) => { e.preventDefault(); });
      wrap.addEventListener("drop", (e) => {
        const files = [...(e.dataTransfer?.files || [])].filter((f) => f.type.startsWith("image/"));
        if (files.length) { e.preventDefault(); files.forEach((f) => this.addImage(f)); }
      });
      this.sendBtn.addEventListener("click", () => this.submit(this.streaming ? "steer" : "normal"));
      this.stopBtn.addEventListener("click", () => this.onAbort());
      OMP.$("#attach-btn").addEventListener("click", () => this.fileInput.click());
      this.fileInput.addEventListener("change", () => {
        [...this.fileInput.files].forEach((f) => this.addImage(f));
        this.fileInput.value = "";
      });
    }

    setEnabled(on) {
      this.ta.disabled = !on;
      this.sendBtn.disabled = !on;
      if (on) this.ta.focus();
    }
    setStreaming(on) {
      this.streaming = on;
      this.stopBtn.hidden = !on;
      this.sendBtn.textContent = on ? "Steer" : "Send";
      this.sendBtn.title = on ? "Enter: steer the running turn · Alt+Enter: queue a follow-up" : "Send (Enter)";
    }
    setCommands(list) { this.commands = Array.isArray(list) ? list : []; }
    setText(text) {
      this.ta.value = text || "";
      this.autosize();
      this.updateMode();
      this.ta.focus();
      this.ta.setSelectionRange(this.ta.value.length, this.ta.value.length);
    }
    focus() { if (!this.ta.disabled) this.ta.focus(); }

    autosize() {
      this.ta.style.height = "auto";
      this.ta.style.height = Math.min(this.ta.scrollHeight + 2, window.innerHeight * 0.4) + "px";
    }
    updateMode() { this.ta.classList.toggle("bash", this.ta.value.startsWith("!")); }

    addImage(file) {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const url = String(reader.result);
        const data = url.slice(url.indexOf(",") + 1);
        this.images.push({ type: "image", data, mimeType: file.type || "image/png" });
        this.renderAttachments();
      };
      reader.readAsDataURL(file);
    }
    renderAttachments() {
      this.attBox.replaceChildren(...this.images.map((img, i) => el("div.attachment", null,
        el("img", { src: `data:${img.mimeType};base64,${img.data}` }),
        el("button", { text: "×", title: "Remove", onclick: () => { this.images.splice(i, 1); this.renderAttachments(); } }))));
    }

    submit(mode) {
      const text = this.ta.value;
      if (!text.trim() && !this.images.length) return;
      const images = this.images;
      if (this.onSubmit(text, images, mode) === false) return;
      if (text.trim()) {
        this.history = this.history.filter((h) => h !== text);
        this.history.push(text);
        if (this.history.length > HISTORY_MAX) this.history.splice(0, this.history.length - HISTORY_MAX);
        try { localStorage.setItem(HISTORY_KEY, JSON.stringify(this.history)); } catch { /* quota */ }
      }
      this.histPos = -1;
      this.images = [];
      this.renderAttachments();
      this.setText("");
      this.closeAc();
    }

    onKey(e) {
      if (e.isComposing) return;
      if (this.ac.open) {
        if (e.key === "ArrowDown") { e.preventDefault(); this.moveAc(1); return; }
        if (e.key === "ArrowUp") { e.preventDefault(); this.moveAc(-1); return; }
        if (e.key === "Tab") { e.preventDefault(); this.acceptAc(); return; }
        if (e.key === "Escape") { e.preventDefault(); this.closeAc(); return; }
        if (e.key === "Enter" && !e.shiftKey) {
          const item = this.ac.items[this.ac.sel];
          const typed = this.ta.value.trim().slice(1);
          if (item && typed !== item.name && !(item.aliases || []).includes(typed)) { e.preventDefault(); this.acceptAc(); return; }
        }
      }
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (e.altKey) this.submit("followUp");
        else this.submit(this.streaming ? "steer" : "normal");
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        this.onAbort();
        return;
      }
      if (e.key === "ArrowUp" && !e.shiftKey && this.caretOnFirstLine()) {
        if (!this.history.length) return;
        e.preventDefault();
        if (this.histPos === -1) { this.draft = this.ta.value; this.histPos = this.history.length; }
        this.histPos = Math.max(0, this.histPos - 1);
        this.setText(this.history[this.histPos]);
        return;
      }
      if (e.key === "ArrowDown" && !e.shiftKey && this.histPos !== -1 && this.caretOnLastLine()) {
        e.preventDefault();
        this.histPos++;
        if (this.histPos >= this.history.length) { this.histPos = -1; this.setText(this.draft); }
        else this.setText(this.history[this.histPos]);
      }
    }
    caretOnFirstLine() { return !this.ta.value.slice(0, this.ta.selectionStart).includes("\n"); }
    caretOnLastLine() { return !this.ta.value.slice(this.ta.selectionEnd).includes("\n"); }

    // ── slash autocomplete ──
    updateAc() {
      const v = this.ta.value;
      const m = v.match(/^\/([^\s]*)$/);
      if (!m || this.ta.selectionStart !== v.length) { this.closeAc(); return; }
      const q = m[1].toLowerCase();
      const scored = [];
      for (const c of this.commands) {
        const names = [c.name, ...(c.aliases || [])];
        let best = -1;
        for (const n of names) {
          const ln = n.toLowerCase();
          if (ln.startsWith(q)) best = Math.max(best, n === c.name ? 3 : 2);
          else if (q && ln.includes(q)) best = Math.max(best, 1);
        }
        if (best < 0 && q && (c.description || "").toLowerCase().includes(q)) best = 0;
        if (best >= 0) scored.push([best, c]);
      }
      scored.sort((a, b) => b[0] - a[0] || a[1].name.localeCompare(b[1].name));
      this.ac.items = scored.slice(0, 60).map((s) => s[1]);
      this.ac.sel = 0;
      this.ac.open = this.ac.items.length > 0;
      this.renderAc();
    }
    renderAc() {
      this.acBox.hidden = !this.ac.open;
      if (!this.ac.open) return;
      this.acBox.replaceChildren(...this.ac.items.map((c, i) => el("div.ac-item" + (i === this.ac.sel ? ".sel" : ""), {
        onmousedown: (e) => { e.preventDefault(); this.ac.sel = i; this.acceptAc(); },
      }, el("span.ac-name", { text: "/" + c.name }),
      c.input && c.input.hint ? el("span.ac-hint", { text: c.input.hint }) : null,
      el("span.ac-desc", { text: c.description || "" }))));
      const sel = this.acBox.children[this.ac.sel];
      if (sel) sel.scrollIntoView({ block: "nearest" });
    }
    moveAc(d) {
      const n = this.ac.items.length;
      this.ac.sel = (this.ac.sel + d + n) % n;
      this.renderAc();
    }
    acceptAc() {
      const c = this.ac.items[this.ac.sel];
      if (!c) return;
      this.setText("/" + c.name + " ");
      this.closeAc();
    }
    closeAc() {
      this.ac.open = false;
      this.acBox.hidden = true;
    }
  }

  OMP.Composer = Composer;
})(window.OMP);
