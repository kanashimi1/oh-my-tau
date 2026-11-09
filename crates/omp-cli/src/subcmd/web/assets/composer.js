"use strict";
// Prompt editor: autosize, slash-command autocomplete, history, image
// attachments (paste / drop / picker), steer & follow-up while streaming.
(function (OMP) {
  const { el } = OMP;
  const HISTORY_KEY = "omp-web-history";
  const HISTORY_MAX = 200;

  // Magic keywords (omp-tui `magic::KEYWORDS`): a standalone, case-sensitive
  // prose word — not inside code spans/fences, not glued into an identifier,
  // path or call. Painted with the TUI's HSL hue sweep.
  const KEYWORDS = [
    ["ultrathink", 0, 330], ["orchestrate", 150, 280], ["workflowz", 30, 150],
    ["jevify", 300, 420], ["ultracode", 410, 270],
  ];
  const KW_RE = new RegExp(`(?<![\\w./:\\-])(${KEYWORDS.map((k) => k[0]).join("|")})(?![\\w(/:\\-]|\\.\\w)`, "g");
  const kwGradient = (word) => {
    const [, from, to] = KEYWORDS.find((k) => k[0] === word);
    const stops = [0, 0.25, 0.5, 0.75, 1].map((t) => `hsl(${(from + (to - from) * t) % 360} 90% 62%)`);
    return `linear-gradient(90deg, ${stops.join(", ")})`;
  };
  /** Character ranges of fenced blocks and inline code spans (not prose). */
  function codeRanges(text) {
    const out = [];
    const fence = /^(```|~~~)[^\n]*\n[\s\S]*?(?:^\1[^\n]*$|(?![\s\S]))/gm;
    let m;
    while ((m = fence.exec(text))) out.push([m.index, m.index + m[0].length]);
    const inline = /(`+)[^`]*?\1/g;
    while ((m = inline.exec(text))) out.push([m.index, m.index + m[0].length]);
    return out;
  }
  /** HTML for the highlight backdrop: escaped text with keyword spans. */
  function highlightHtml(text) {
    const code = codeRanges(text);
    let html = "";
    let last = 0;
    for (const m of text.matchAll(KW_RE)) {
      if (code.some(([a, b]) => m.index >= a && m.index < b)) continue;
      html += OMP.esc(text.slice(last, m.index)) + `<span class="kw" style="background-image:${kwGradient(m[1])}">${m[1]}</span>`;
      last = m.index + m[1].length;
    }
    // Trailing newline needs a glyph so the backdrop keeps the same height.
    return html + OMP.esc(text.slice(last)) + "\u200b";
  }

  const fmtSize = (n) => (n < 1024 ? `${n} B` : n < 1 << 20 ? `${(n / 1024).toFixed(0)} KB` : `${(n / (1 << 20)).toFixed(1)} MB`);

  // Polished upward arrow for the send button (swap keeps the stop state on the
  // separate #stop-btn untouched).
  const SEND_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5"/><path d="m5 12 7-7 7 7"/></svg>';

  class Composer {
    constructor({ onSubmit, onAbort }) {
      this.ta = OMP.$("#composer");
      this.sendBtn = OMP.$("#send-btn");
      this.stopBtn = OMP.$("#stop-btn");
      this.acBox = OMP.$("#autocomplete");
      this.attBox = OMP.$("#attachments");
      this.fileInput = OMP.$("#file-input");
      this.micBtn = OMP.$("#mic-btn");
      this.files = []; // non-image attachments: {name, size, path?, error?, upload}
      this.rec = null; // live dictation: {recorder, stream, chunks, started, timer, cancelled}
      this.hl = OMP.$("#composer-hl");
      this.onSubmit = onSubmit;
      this.onAbort = onAbort;
      this.commands = [];
      this.images = [];
      this.streaming = false;
      this.ac = { open: false, items: [], sel: 0 };
      this.tc = { tail: "" };       // inline text-completion ghost
      this.sessionPrompts = [];     // user prompts of the open session
      try { this.history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { this.history = []; }
      this.histPos = -1;
      this.draft = "";
      this.sendBtn.innerHTML = SEND_ICON;
      this.sendBtn.setAttribute("aria-label", "Send");
      // Keep the microphone immediately next to send, whatever the markup order.
      this.sendBtn.before(this.micBtn);
      this.micBtn.setAttribute("type", "button");
      this.micBtn.setAttribute("aria-label", "Dictate");
      this.micBtn.setAttribute("aria-pressed", "false");
      this.bind();
    }

    bind() {
      const ta = this.ta;
      ta.addEventListener("input", () => { this.updateAc(); this.updateTc(); this.autosize(); this.updateMode(); });
      ta.addEventListener("keydown", (e) => this.onKey(e));
      ta.addEventListener("scroll", () => { this.hl.scrollTop = ta.scrollTop; });
      ta.addEventListener("blur", () => setTimeout(() => { this.closeAc(); this.clearTc(); }, 150));
      ta.addEventListener("paste", (e) => {
        const files = [...(e.clipboardData?.items || [])].filter((i) => i.kind === "file").map((i) => i.getAsFile());
        if (files.length) { e.preventDefault(); files.forEach((f) => this.addFile(f)); }
      });
      const wrap = OMP.$("#composer-wrap");
      wrap.addEventListener("dragover", (e) => { e.preventDefault(); });
      wrap.addEventListener("drop", (e) => {
        const files = [...(e.dataTransfer?.files || [])];
        if (files.length) { e.preventDefault(); files.forEach((f) => this.addFile(f)); }
      });
      this.sendBtn.addEventListener("click", () => this.submit(this.streaming ? "steer" : "normal"));
      this.stopBtn.addEventListener("click", () => this.onAbort());
      OMP.$("#attach-btn").addEventListener("click", () => this.fileInput.click());
      this.fileInput.addEventListener("change", () => {
        [...this.fileInput.files].forEach((f) => this.addFile(f));
        this.fileInput.value = "";
      });
      this.micBtn.addEventListener("click", () => (this.rec ? this.stopDictation(false) : this.startDictation()));
    }

    // ── attachments ────────────────────────────────────────────────────────
    /** Images go to the model as images; any other file is uploaded to the
     *  agent host and its path is added to the prompt. */
    addFile(file) {
      if (!file) return;
      if (file.type.startsWith("image/") && file.type !== "image/svg+xml") return this.addImage(file);
      const att = { name: file.name || "file", size: file.size };
      att.upload = fetch("/api/uploads?name=" + encodeURIComponent(att.name), { method: "POST", body: file })
        .then(async (r) => {
          const body = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error(body.error || r.statusText);
          att.path = body.path;
        })
        .catch((e) => { att.error = e.message; OMP.toast(`${att.name}: ${e.message}`, "error"); })
        .finally(() => this.renderAttachments());
      this.files.push(att);
      this.renderAttachments();
    }

    // ── dictation ──────────────────────────────────────────────────────────
    async startDictation() {
      if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
        return OMP.toast("Microphone recording is not available here", "error");
      }
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      } catch (e) {
        const denied = e && (e.name === "NotAllowedError" || e.name === "SecurityError");
        return OMP.toast(denied
          ? "Microphone access denied. Allow it in the prompt, and in Windows Settings → Privacy → Microphone (desktop apps)."
          : `Microphone: ${e.message || e}`, "error", 9000);
      }
      const type = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/webm"].find((t) => MediaRecorder.isTypeSupported(t));
      const recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
      const rec = { recorder, stream, chunks: [], started: Date.now(), cancelled: false };
      recorder.ondataavailable = (e) => { if (e.data && e.data.size) rec.chunks.push(e.data); };
      recorder.onstop = () => this.finishDictation(rec);
      recorder.start(250);
      this.rec = rec;
      const time = this.micBtn.querySelector(".mic-time");
      const tick = () => {
        const s = Math.floor((Date.now() - rec.started) / 1000);
        time.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
        if (s >= 600) this.stopDictation(false); // 10 min cap
      };
      tick();
      time.hidden = false;
      rec.timer = setInterval(tick, 500);
      this.micBtn.classList.add("recording");
      this.micBtn.setAttribute("aria-pressed", "true");
      this.micBtn.title = "Recording… click to insert the text · Esc cancels";
    }

    stopDictation(cancel) {
      const rec = this.rec;
      if (!rec) return;
      rec.cancelled = cancel;
      clearInterval(rec.timer);
      this.rec = null;
      this.micBtn.classList.remove("recording");
      this.micBtn.setAttribute("aria-pressed", "false");
      this.micBtn.querySelector(".mic-time").hidden = true;
      this.micBtn.title = "Dictate (click to start, click again to insert the text · Esc cancels)";
      if (rec.recorder.state !== "inactive") rec.recorder.stop();
      rec.stream.getTracks().forEach((t) => t.stop());
    }

    async finishDictation(rec) {
      if (rec.cancelled || !rec.chunks.length) return;
      const blob = new Blob(rec.chunks, { type: rec.recorder.mimeType || "audio/webm" });
      const ext = /ogg/.test(blob.type) ? "ogg" : "webm";
      this.micBtn.classList.add("busy");
      this.micBtn.disabled = true;
      try {
        const r = await fetch(`/api/transcribe?ext=${ext}`, { method: "POST", body: blob });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.error || r.statusText);
        const text = String(body.text || "").trim();
        if (!text) OMP.toast("Didn't catch any speech", "info");
        else this.insertText(text);
      } catch (e) {
        OMP.toast(`Dictation: ${e.message}`, "error", 9000);
      } finally {
        this.micBtn.classList.remove("busy");
        this.micBtn.disabled = false;
      }
    }

    /** Insert `text` at the caret, spaced from its neighbours. */
    insertText(text) {
      const ta = this.ta;
      const start = ta.selectionStart ?? ta.value.length, end = ta.selectionEnd ?? start;
      const before = ta.value.slice(0, start), after = ta.value.slice(end);
      const lead = before && !/\s$/.test(before) ? " " : "";
      const trail = after && !/^\s/.test(after) ? " " : "";
      ta.value = before + lead + text + trail + after;
      const caret = (before + lead + text).length;
      this.autosize();
      this.updateMode();
      ta.focus();
      ta.setSelectionRange(caret, caret);
    }

    setEnabled(on) {
      this.ta.disabled = !on;
      this.sendBtn.disabled = !on;
      if (on) this.ta.focus();
    }
    setStreaming(on) {
      this.streaming = on;
      this.stopBtn.hidden = !on;
      this.sendBtn.classList.toggle("steer", on); // icon button: keep its SVG
      this.sendBtn.title = on ? "Enter: steer the running turn · Alt+Enter: queue a follow-up" : "Send (Enter)";
    }
    setCommands(list) { this.commands = Array.isArray(list) ? list : []; }
    setText(text) {
      this.ta.value = text || "";
      this.updateTc();
      this.autosize();
      this.updateMode();
      this.ta.focus();
      this.ta.setSelectionRange(this.ta.value.length, this.ta.value.length);
    }
    focus() { if (!this.ta.disabled) this.ta.focus(); }

    autosize() {
      this.ta.style.height = "auto";
      this.ta.style.height = Math.min(this.ta.scrollHeight + 2, window.innerHeight * 0.4) + "px";
      this.highlight();
    }
    highlight() {
      const text = this.ta.value;
      const on = KW_RE.test(text);
      KW_RE.lastIndex = 0;
      this.ta.parentElement.classList.toggle("magic", on);
      // Ghost completion shows only while the caret sits at the end of the
      // draft; any caret move or selection hides it.
      const atEnd = this.ta.selectionStart === text.length && this.ta.selectionEnd === this.ta.selectionStart;
      const ghost = atEnd ? this.tc.tail : "";
      this.ta.parentElement.classList.toggle("tc", !!ghost);
      if (!on && !ghost) { this.hl.innerHTML = ""; this.hl.scrollTop = this.ta.scrollTop; return; }
      let html = on ? highlightHtml(text) : OMP.esc(text);
      if (ghost) {
        if (html.endsWith("\u200b")) html = html.slice(0, -1);
        html += `<span class="tc-ghost">${OMP.esc(ghost)}</span>\u200b`;
      }
      this.hl.innerHTML = html;
      this.hl.scrollTop = this.ta.scrollTop;
    }

    // -- inline text completion (prompt history + this session's prompts) --
    /** Session user prompts (app.js feeds them after the history loads). */
    setSessionPrompts(list) { this.sessionPrompts = Array.isArray(list) ? list : []; }
    /** Newest-first completion sources: this session first, then global history. */
    completionSources() {
      return (this.sessionPrompts || []).slice(-200).reverse()
        .concat(this.history.slice(-200).reverse());
    }
    updateTc() {
      const v = this.ta.value;
      this.tc.tail = (!this.ac.open && !v.startsWith("/") && v.trim().length >= 2)
        ? OMP.textComplete.match(v, this.completionSources())
        : "";
    }
    acceptTc() {
      const full = this.ta.value + this.tc.tail;
      this.tc.tail = "";
      this.setText(full);
      this.updateTc();
      this.highlight();
    }
    clearTc() {
      if (!this.tc.tail) return;
      this.tc.tail = "";
      this.highlight();
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
    /** External API (contract): attach an image from raw base64 data;
     *  `annotate` opens the pin editor on it right away. */
    addImageData({ data, mimeType, annotate, caption }) {
      if (!data) return;
      const att = { type: "image", data, mimeType: mimeType || "image/png" };
      if (caption) att.caption = caption; // what the image shows (preview pins), sent with the pin notes
      this.images.push(att);
      this.renderAttachments();
      // One-gesture capture flow: the pin editor opens right away unless the
      // user opted out (localStorage toggle; the chip click always opens it).
      let autoAnnotate = true;
      try { autoAnnotate = localStorage.getItem("omp-annotate-on-capture") !== "false"; } catch { /* storage unavailable */ }
      if (annotate && autoAnnotate) this.annotate(att);
    }
    async annotate(att) {
      if (await OMP.annotate(att)) {
        this.renderAttachments();
        const count = att.pins?.length || 0;
        OMP.toast(count ? `Saved ${count} image pin${count === 1 ? "" : "s"}` : "Image pins cleared", "success", 2200);
      }
      this.focus();
    }
    renderAttachments() {
      const images = this.images.map((img, i) => el("div.attachment", null,
        el("img", {
          src: `data:${img.mimeType};base64,${img.data}`,
          title: "Click to add pins with comments",
          onclick: () => this.annotate(img),
        }),
        img.pins?.length ? el("span.pin-badge", { text: String(img.pins.length), title: "Pins" }) : null,
        el("button.annotate-btn", { text: "✎", title: "Annotate: add pins with comments", onclick: () => this.annotate(img) }),
        el("button", { text: "×", title: "Remove", onclick: () => { this.images.splice(i, 1); this.renderAttachments(); } })));
      const files = this.files.map((f, i) => el("div.attachment.file-chip" + (f.error ? ".failed" : f.path ? "" : ".uploading"), { title: f.error || f.path || "Uploading…" },
        el("span.file-icon", { text: f.error ? "!" : f.path ? "📄" : "⏳" }),
        el("span.file-name", { text: f.name }),
        el("span.file-size", { text: fmtSize(f.size) }),
        el("button", { text: "×", title: "Remove", onclick: () => { this.files.splice(i, 1); this.renderAttachments(); } })));
      this.attBox.replaceChildren(...images, ...files);
    }

    async submit(mode) {
      if (this.files.some((f) => !f.path && !f.error)) {
        OMP.toast("Uploading attachments…", "info", 2000);
        await Promise.all(this.files.map((f) => f.upload));
      }
      const raw = this.ta.value;
      const ready = this.files.filter((f) => f.path);
      if (!raw.trim() && !this.images.length && !ready.length) return;
      const notes = [
        OMP.pinNotes(this.images),
        ready.length ? "Attached files:\n" + ready.map((f) => `- ${f.path} (${f.name}, ${fmtSize(f.size)})`).join("\n") : "",
      ].filter(Boolean).join("\n\n");
      const text = notes ? (raw.trim() ? `${raw}\n\n${notes}` : notes) : raw;
      const images = this.images.map(({ type, data, mimeType }) => ({ type, data, mimeType }));
      if (this.onSubmit(text, images, mode) === false) return;
      if (raw.trim()) {
        this.history = this.history.filter((h) => h !== raw);
        this.history.push(raw);
        if (this.history.length > HISTORY_MAX) this.history.splice(0, this.history.length - HISTORY_MAX);
        try { localStorage.setItem(HISTORY_KEY, JSON.stringify(this.history)); } catch { /* quota */ }
      }
      this.histPos = -1;
      this.images = [];
      this.files = [];
      this.renderAttachments();
      this.setText("");
      this.closeAc();
    }

    onKey(e) {
      if (e.isComposing) return;
      // Inline ghost: Tab or Right at the end accepts, Escape dismisses,
      // caret moves off the end cancel it.
      if (this.tc.tail && this.ta.selectionStart === this.ta.value.length
        && this.ta.selectionEnd === this.ta.selectionStart) {
        if (e.key === "Tab" || e.key === "ArrowRight") { e.preventDefault(); this.acceptTc(); return; }
        if (e.key === "Escape") { e.preventDefault(); this.clearTc(); return; }
        if (e.key === "ArrowLeft" || e.key === "Home" || e.key === "PageUp") this.clearTc();
      }
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
        if (this.rec) this.stopDictation(true);
        else this.onAbort();
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

  // Inline text completion (pure, unit-tested): the completion TAIL for the
  // first candidate (sources are newest-first) that begins with the whole
  // draft, case-insensitively. Slash drafts are the command autocomplete's
  // domain; tails stop at the first newline (no surprise multi-line inserts).
  OMP.textComplete = {
    match(draft, candidates) {
      const v = String(draft ?? "");
      if (v.length < 2 || v.startsWith("/")) return "";
      const vl = v.toLowerCase();
      const seen = new Set();
      for (const raw of candidates || []) {
        if (typeof raw !== "string" || raw.length <= v.length) continue;
        if (!raw.toLowerCase().startsWith(vl)) continue;
        let tail = raw.slice(v.length);
        const nl = tail.indexOf("\n");
        if (nl === 0) continue;
        if (nl > 0) tail = tail.slice(0, nl);
        if (!tail.trim()) continue;
        const key = (v + tail).toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        return tail;
      }
      return "";
    },
  };
})(window.OMP);
