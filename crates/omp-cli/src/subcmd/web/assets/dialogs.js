"use strict";
// Modals: generic picker / text prompt / folder browser, and the renderer for
// the agent's `extension_ui_request` dialogs (select, confirm, input, editor, ask).
(function (OMP) {
  const { el } = OMP;
  const root = () => OMP.$("#modal-root");

  /** Open a modal; resolves through `done(value)`; Esc/backdrop -> done(null). */
  function modal({ title, body, foot, wide, onKey, cls }) {
    let finished = false;
    let resolve;
    const promise = new Promise((r) => (resolve = r));
    const box = el("div.modal" + (wide ? ".wide" : "") + (cls ? "." + cls : ""));
    const backdrop = el("div.modal-backdrop", null, box);
    const done = (value) => {
      if (finished) return;
      finished = true;
      backdrop.remove();
      document.removeEventListener("keydown", keyHandler, true);
      resolve(value);
      OMP.dialogs.afterClose();
    };
    const keyHandler = (e) => {
      if (!backdrop.isConnected) return;
      if (root().lastElementChild !== backdrop) return;
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(null); return; }
      if (onKey) onKey(e, done);
    };
    box.append(el("div.modal-head", null, el("span", { text: title || "" })));
    box.append(el("div.modal-body", null, body || []));
    if (foot) box.append(el("div.modal-foot", null, foot(done)));
    backdrop.addEventListener("mousedown", (e) => { if (e.target === backdrop) done(null); });
    document.addEventListener("keydown", keyHandler, true);
    root().append(backdrop);
    const focus = box.querySelector("input, textarea, .option.sel, button.primary");
    if (focus) setTimeout(() => focus.focus(), 0);
    return { promise, done, box };
  }

  /**
   * Pick one of `items` ({label, description?, tag?}). Resolves the index or null.
   * opts: {searchable, initial, multi (returns index array), hint}
   */
  function pick(title, items, opts = {}) {
    let query = "";
    let cursor = Math.min(opts.initial || 0, Math.max(0, items.length - 1));
    const checked = new Set(opts.checked || []);
    const list = el("div.options");
    const search = opts.searchable || items.length > 12
      ? el("input", { type: "text", placeholder: "Type to filter…", autocomplete: "off" })
      : null;
    let visible = [];
    let m;
    const render = () => {
      const q = query.toLowerCase();
      visible = items.map((it, i) => [it, i]).filter(([it]) => !q ||
        (it.label + " " + (it.description || "") + " " + (it.tag || "")).toLowerCase().includes(q));
      cursor = Math.max(0, Math.min(cursor, visible.length - 1));
      list.replaceChildren(...visible.map(([it, i], k) => {
        const node = el("button.option" + (k === cursor ? ".sel" : "") + (it.cls ? "." + it.cls : "") + (opts.multi ? (checked.has(i) ? ".checked" : ".uncheck") : ""), {
          onclick: () => { cursor = k; choose(); },
          onmousemove: () => { if (cursor !== k) { cursor = k; render(); } },
        }, el("span", null, el("span.label" + (it.mode ? ".ultra-text" : ""), { text: it.label, dataset: it.mode ? { mode: it.mode } : null }), it.tag ? el("span.tag", { text: it.tag }) : null),
        it.description ? el("span.desc", { text: it.description }) : null);
        return node;
      }));
      const sel = list.children[cursor];
      if (sel) sel.scrollIntoView({ block: "nearest" });
    };
    const choose = () => {
      const row = visible[cursor];
      if (!row) return;
      if (opts.multi) {
        if (checked.has(row[1])) checked.delete(row[1]); else checked.add(row[1]);
        render();
        return;
      }
      m.done(row[1]);
    };
    if (search) search.addEventListener("input", () => { query = search.value; cursor = 0; render(); });
    m = modal({
      title,
      wide: opts.wide,
      body: [search, list, opts.extra].filter(Boolean),
      foot: (done) => [
        el("span.hint", { text: opts.hint || (opts.multi ? "↑↓ move · Enter toggle · Esc cancel" : "↑↓ move · Enter select · Esc cancel") }),
        opts.multi ? el("button.primary", { text: "Done", onclick: () => done([...checked]) }) : null,
        el("button.secondary", { text: "Cancel", onclick: () => done(null) }),
      ],
      onKey: (e, done) => {
        if (e.key === "ArrowDown") { e.preventDefault(); cursor = Math.min(visible.length - 1, cursor + 1); render(); }
        else if (e.key === "ArrowUp") { e.preventDefault(); cursor = Math.max(0, cursor - 1); render(); }
        else if (e.key === "PageDown") { e.preventDefault(); cursor = Math.min(visible.length - 1, cursor + 10); render(); }
        else if (e.key === "PageUp") { e.preventDefault(); cursor = Math.max(0, cursor - 10); render(); }
        else if (e.key === "Enter" && !e.isComposing) {
          e.preventDefault();
          if (opts.multi && (e.ctrlKey || e.metaKey)) done([...checked]); else choose();
        }
      },
    });
    render();
    if (search) setTimeout(() => search.focus(), 0);
    return m;
  }

  function promptText(title, { value = "", placeholder = "", multiline = false, message = "" } = {}) {
    const input = multiline
      ? el("textarea", { value, placeholder })
      : el("input", { type: "text", value, placeholder, autocomplete: "off" });
    return modal({
      title,
      wide: multiline,
      body: [message ? el("p.muted", { text: message }) : null, input].filter(Boolean),
      foot: (done) => [
        el("span.hint", { text: multiline ? "Ctrl+Enter save · Esc cancel" : "Enter submit · Esc cancel" }),
        el("button.secondary", { text: "Cancel", onclick: () => done(null) }),
        el("button.primary", { text: multiline ? "Save" : "OK", onclick: () => done(input.value) }),
      ],
      onKey: (e, done) => {
        if (e.key === "Enter" && !e.isComposing && (!multiline || e.ctrlKey || e.metaKey)) { e.preventDefault(); done(input.value); }
      },
    });
  }

  function confirmBox(title, message) {
    return modal({
      title,
      body: [el("p", { text: message || "" })],
      foot: (done) => [
        el("button.secondary", { text: "No", onclick: () => done(false) }),
        el("button.primary", { text: "Yes", onclick: () => done(true) }),
      ],
      onKey: (e, done) => {
        if (e.key === "Enter" || e.key === "y") { e.preventDefault(); done(true); }
        else if (e.key === "n") { e.preventDefault(); done(false); }
      },
    });
  }

  /** Folder browser backed by /api/fs. Resolves the chosen absolute path or null. */
  function pickFolder(start, title = "Choose working directory") {
    let current = null;
    const crumbs = el("div.crumbs");
    const pathInput = el("input", { type: "text", placeholder: "/path/to/project", autocomplete: "off" });
    const list = el("div.options");
    const load = async (path) => {
      try {
        const d = await OMP.api("/api/fs?path=" + encodeURIComponent(path || ""));
        current = d.path;
        pathInput.value = d.path;
        crumbs.textContent = d.isGit ? "git repository" : "";
        const rows = [];
        if (d.parent) rows.push(el("button.option", { onclick: () => load(d.parent) }, el("span.label", { text: ".." })));
        for (const name of d.dirs) {
          rows.push(el("button.option", { onclick: () => load(d.path.replace(/\/$/, "") + "/" + name) }, el("span.label", { text: name + "/" })));
        }
        list.replaceChildren(...rows);
      } catch (e) {
        OMP.toast(String(e.message || e), "error");
      }
    };
    pathInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); load(pathInput.value); } });
    load(start);
    return modal({
      title,
      wide: true,
      body: [pathInput, crumbs, list],
      foot: (done) => [
        el("span.hint", { text: "Click a folder to open it · Enter in the path box jumps" }),
        el("button.secondary", { text: "Cancel", onclick: () => done(null) }),
        el("button.primary", { text: "Use this folder", onclick: () => done(current) }),
      ],
    });
  }

  // ── agent dialogs ────────────────────────────────────────────────────────
  const queue = [];
  let active = null; // {id, modal}

  function optionList(frame) {
    const details = Array.isArray(frame.optionDetails) ? frame.optionDetails : [];
    return (frame.options || []).map((o, i) => {
      if (typeof o === "string") return { label: o, description: details[i] && details[i].description };
      return { label: o.label || String(o.value ?? ""), description: o.description, value: o.value };
    });
  }

  async function runAsk(frame) {
    const answers = [];
    for (const q of frame.questions || []) {
      const opts = (q.options || []).map((o, i) => ({
        label: typeof o === "string" ? o : o.label,
        description: typeof o === "string" ? undefined : o.description,
        tag: q.recommended === i ? "recommended" : undefined,
      }));
      opts.push({ label: "Other…", description: "Type a custom answer" });
      const title = (q.header ? q.header + " — " : "") + (q.question || "");
      const res = await pick(title, opts, { multi: !!q.multi, initial: q.recommended || 0, wide: true }).promise;
      if (res == null) return null;
      const picked = Array.isArray(res) ? res : [res];
      const other = picked.includes(opts.length - 1);
      const answer = { id: q.id, selectedOptions: picked.filter((i) => i !== opts.length - 1).map((i) => opts[i].label) };
      if (other) {
        const custom = await promptText(q.question || "Your answer").promise;
        if (custom == null) return null;
        answer.customInput = custom;
      }
      answers.push(answer);
    }
    return { answers };
  }

  function show(frame) {
    const method = frame.method;
    let m = null;
    let result;
    if (method === "select") {
      const items = optionList(frame);
      m = pick(frame.title || "Select", items, { searchable: frame.searchable || frame.fullscreen, initial: frame.initialIndex || 0, wide: frame.fullscreen });
      result = m.promise.then((i) => (i == null ? { cancelled: true } : { value: items[i].value ?? items[i].label }));
    } else if (method === "confirm") {
      m = confirmBox(frame.title || "Confirm", frame.message);
      result = m.promise.then((v) => (v == null ? { cancelled: true } : { confirmed: v, value: v }));
    } else if (method === "editor") {
      m = promptText(frame.title || "Edit", { value: frame.prefill || "", multiline: true });
      result = m.promise.then((v) => (v == null ? { cancelled: true } : { value: v }));
    } else if (method === "ask") {
      result = runAsk(frame).then((v) => v || { cancelled: true });
    } else {
      m = promptText(frame.title || "Input", { placeholder: frame.placeholder || "", message: frame.message });
      result = m.promise.then((v) => (v == null ? { cancelled: true } : { value: v }));
    }
    const entry = { id: frame.id, modal: m, withdrawn: false };
    active = entry;
    let timer = null;
    if (frame.timeout) timer = setTimeout(() => { if (entry.modal) entry.modal.done(null); }, frame.timeout);
    result.then((value) => {
      clearTimeout(timer);
      if (active === entry) active = null;
      if (!entry.withdrawn && answerFn) answerFn(frame.id, value);
      pump();
    });
  }

  let answerFn = null;
  function withdraw() {
    const entry = active;
    active = null;
    entry.withdrawn = true;
    if (entry.modal) entry.modal.done(null);
    else root().replaceChildren(); // multi-step ask: drop whatever step is open
    pump();
  }
  function pump() {
    if (active || !queue.length) return;
    show(queue.shift());
  }

  OMP.dialogs = {
    modal, pick, promptText, confirmBox, pickFolder,
    afterClose() {},
    /** Queue an agent dialog; `answer(id, payload)` sends the response. */
    request(frame, answer) {
      answerFn = answer;
      if ((active && active.id === frame.id) || queue.some((q) => q.id === frame.id)) return;
      queue.push(frame);
      pump();
    },
    /** Withdraw a dialog (cancelDialog / answered in another tab). */
    cancel(id) {
      const qi = queue.findIndex((q) => q.id === id);
      if (qi >= 0) queue.splice(qi, 1);
      if (active && active.id === id) withdraw();
    },
    clear() {
      queue.length = 0;
      if (active) withdraw();
      root().replaceChildren();
    },
  };
})(window.OMP);
